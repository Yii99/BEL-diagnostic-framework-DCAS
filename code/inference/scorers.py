"""Scoring backends for the RelConBEL diagnostic.

Every backend answers one question: given a mention (optionally with its
document context) and a set of candidate concepts, how strongly does the model
prefer each candidate? The framework in ``score_pairs.py`` then compares the
gold concept's score against each hard negative's.

A candidate is a *concept*, not a string, so each candidate carries a group of
surface forms (aliases) and its score is the max over that group — the same
"best-matching alias wins" rule a deployed linker uses. Backends that represent
an entity as a whole (ArboEL encodes ``name [SEP] definition``) ignore the alias
group and score the entity directly.

    --model bm25[:k1=1.5,b=0.75]        BM25Scorer            sparse, CPU-only
    --model sapbert                     DenseEncoderScorer    SapBERT alias
    --model <hf_id_or_path>             DenseEncoderScorer    any AutoModel
    --model biosyn[:<hf_id>,alpha=0.5]  BioSynScorer          dense + char TF-IDF
    --model biosyn-auto                 BioSynScorer          per-dataset checkpoint
    --model crossencoder:<hf_id>        CrossEncoderScorer    pairwise reranker
    --model arboel:<ckpt_dir_or_bin>    ArboELBiencoderScorer BLINK-style biencoder
    --model angel[:<hf_id_or_path>]     Seq2SeqLikelihoodScorer  ANGEL / GenBioEL
    --model genbioel:<path>             Seq2SeqLikelihoodScorer  GenBioEL
    --model genre[:<hf_id>]             Seq2SeqLikelihoodScorer  GENRE

Generative systems (GENRE / GenBioEL / ANGEL) normally decode against a prefix
trie built from a benchmark-specific target dictionary, so they cannot emit a
negative that is absent from that trie — which is exactly the blind spot this
project is probing. They are therefore scored by *forced likelihood* of each
candidate name instead of by constrained decoding:

    score(m, e) = 1/|a| * sum_t log P(a_t | source(m), a_<t)   maximised over aliases a of e

This is the same quantity beam search maximises (``normalize_scores=True``,
``length_penalty=1``), just evaluated on candidates we choose rather than on the
trie's reachable set.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MentionQuery:
    """One mention, with everything any backend might need."""

    mention_id: str
    text: str
    text_idx: int
    context_left: str = ""
    context_right: str = ""


@dataclass
class CandidateBatch:
    """Candidate concepts for one mention, in CSR form over the string table.

    ``alias_idx[group_ptr[i]:group_ptr[i + 1]]`` are the string-table indices of
    candidate ``i``'s surface forms. Every group holds at least one alias.
    """

    entity_ids: list[str]
    alias_idx: np.ndarray
    group_ptr: np.ndarray

    def __len__(self) -> int:
        return len(self.entity_ids)


def groupwise_max(flat: np.ndarray, group_ptr: np.ndarray) -> np.ndarray:
    """Max-reduce ``flat`` over CSR groups. Assumes no group is empty."""
    if len(group_ptr) <= 1:
        return np.empty(0, dtype=np.float32)
    return np.maximum.reduceat(flat.astype(np.float32), group_ptr[:-1])


def groupwise_argmax(flat: np.ndarray, group_ptr: np.ndarray) -> np.ndarray:
    """Index into ``flat`` of each group's max, for reporting the winning alias."""
    out = np.empty(len(group_ptr) - 1, dtype=np.int64)
    for i in range(len(group_ptr) - 1):
        lo, hi = group_ptr[i], group_ptr[i + 1]
        out[i] = lo + int(np.argmax(flat[lo:hi]))
    return out


class Scorer(ABC):
    """Common interface for every backend.

    ``fit`` is called once with every string that can appear as a mention or an
    alias in this run. Backends that need the entity dictionary set
    ``needs_entities`` and get ``fit_entities`` with every entity ID that will be
    scored, so entity representations are built once instead of per mention.
    """

    name: str
    needs_entities: bool = False
    needs_context: bool = False

    def fit(self, strings: list[str]) -> None:
        self._strings = strings

    def fit_entities(self, entity_ids: list[str], concepts: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def score(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray:
        """One score per candidate concept, in ``cands`` order."""

    def score_flat(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray | None:
        """Per-alias scores, or None when the backend scores whole entities.

        07 uses this only to report which alias won; ranking never depends on it.
        """
        return None


class StringPairScorer(Scorer):
    """Backend that scores (mention string, alias string) pairs.

    Subclasses implement ``score_strings``; alias-group maxing is handled here.
    """

    @abstractmethod
    def score_strings(self, query_idx: int, doc_idx: np.ndarray) -> np.ndarray:
        """Score the query string against each string-table index in ``doc_idx``."""

    def score(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray:
        if len(cands) == 0:
            return np.empty(0, dtype=np.float32)
        flat = self.score_strings(query.text_idx, cands.alias_idx)
        self._last_flat = flat
        return groupwise_max(flat, cands.group_ptr)

    def score_flat(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray | None:
        return getattr(self, "_last_flat", None)


class BM25Scorer(StringPairScorer):
    """Robertson-Walker BM25 over the run's string table.

        BM25(q, d) = sum_t idf(t) * tf(t,d) * (k1 + 1)
                     / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    Tokenisation is lowercase alphanumeric with no stemming or stopword removal:
    biomedical surface forms are short and general-purpose stopword lists cut
    into them ("vitamin a", "factor v").

    IDF is computed over the evaluation string table (mention texts + gold
    aliases + negative names), not over all of UMLS, so BM25 sees exactly the
    strings the dense encoders see.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.name = f"bm25(k1={k1},b={b})"
        self.k1 = float(k1)
        self.b = float(b)

    @staticmethod
    def _tokenize(s: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", s.lower())

    def fit(self, strings: list[str]) -> None:
        t0 = time.time()
        print(f"  BM25: tokenising {len(strings):,} strings ...", flush=True)
        self.tokens = [self._tokenize(s) for s in strings]
        self.doc_tf = [Counter(t) for t in self.tokens]
        self.doc_lens = np.array([len(t) for t in self.tokens], dtype=np.float32)
        n_docs = len(strings)
        self.avgdl = float(self.doc_lens.mean()) if n_docs else 0.0
        df: Counter = Counter()
        for tf in self.doc_tf:
            df.update(tf.keys())
        self.idf = {
            t: float(np.log((n_docs - dfv + 0.5) / (dfv + 0.5) + 1.0))
            for t, dfv in df.items()
        }
        print(
            f"  BM25 fit: {n_docs:,} docs, {len(df):,} terms, avgdl={self.avgdl:.2f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    def score_strings(self, query_idx: int, doc_idx: np.ndarray) -> np.ndarray:
        q_terms = set(self.tokens[query_idx])
        scores = np.zeros(len(doc_idx), dtype=np.float32)
        if not q_terms or self.avgdl <= 0:
            return scores
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for i, d in enumerate(doc_idx):
            tf_d = self.doc_tf[d]
            denom_norm = k1 * (1.0 - b + b * float(self.doc_lens[d]) / avgdl)
            s = 0.0
            for t in q_terms:
                tf = tf_d.get(t, 0)
                if tf:
                    s += self.idf.get(t, 0.0) * tf * (k1 + 1.0) / (tf + denom_norm)
            scores[i] = s
        return scores


class DenseEncoderScorer(StringPairScorer):
    """Cosine similarity between L2-normalised pooled embeddings.

    SapBERT's released recipe pools the ``[CLS]`` token, which is the default
    here; ``--pooling mean`` switches to attention-masked mean pooling for
    checkpoints trained that way.
    """

    def __init__(self, model_name: str, *, device: str = "cpu", batch: int = 256,
                 max_len: int = 32, pooling: str = "cls", dtype: str = "float32",
                 normalize: bool = True):
        self.name = f"dense:{model_name.rsplit('/', 1)[-1]}({pooling})"
        self.model_name = model_name
        self.device = device
        self.batch = batch
        self.max_len = max_len
        self.pooling = pooling
        self.dtype = dtype
        self.normalize = normalize
        self.embs: np.ndarray | None = None

    def fit(self, strings: list[str]) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"  loading {self.model_name} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(self.device)
        model.eval()

        dim = model.config.hidden_size
        embs = np.empty((len(strings), dim), dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(strings), self.batch):
                chunk = strings[i : i + self.batch]
                enc = tok(chunk, padding=True, truncation=True,
                          max_length=self.max_len, return_tensors="pt").to(self.device)
                out = model(**enc).last_hidden_state.float()
                if self.pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                else:
                    vec = out[:, 0, :]
                if self.normalize:
                    vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                embs[i : i + len(chunk)] = vec.cpu().numpy()
                if (i // self.batch) % 200 == 0:
                    print(f"    encoded {i + len(chunk):,}/{len(strings):,} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        print(f"  encoded {len(strings):,} strings in {time.time() - t0:.0f}s", flush=True)
        self.embs = embs
        del model

    def score_strings(self, query_idx: int, doc_idx: np.ndarray) -> np.ndarray:
        return self.embs[doc_idx] @ self.embs[query_idx]


class KrissBertScorer(Scorer):
    """KRISSBERT (Zhang et al. 2021): knowledge-rich self-supervised contextual EL.

    Unlike SapBERT and BioSyn, KRISSBERT encodes the mention *in its context*: the
    released convention is ``[CLS] <ENT> mention </ENT> <left ctx> <right ctx> [SEP]``
    pooled at ``[CLS]`` with no L2 normalisation (``usage/utils.py:generate_vectors``,
    ``ContextualMention.to_tensor``). That is why it is a ``Scorer`` and not a
    ``StringPairScorer`` -- the latter scores the mention *string*, which is exactly
    the context-free behaviour KRISSBERT's authors criticise.

    **Deviation, and it is not a small one.** In the released system an entity is
    represented by *prototype* embeddings: the same encoder applied to
    self-supervised mention examples of that entity, averaged per CUI. Those
    examples are not released ("we are currently unable to release the
    self-supervised mention examples, because they require the UMLS and PubMed
    licenses"), and they cannot be reconstructed from what we have. This backend
    therefore falls back to the degenerate prototype: the entity's own name encoded
    as a mention with empty context, ``<ENT> name </ENT>``.

    Consequences to keep in mind when reading its row:

    * this is *not* the configuration that reported 58.3% top-1 on MedMentions;
    * the mention side keeps its context advantage, the entity side loses the
      prototype advantage, so the comparison to SapBERT isolates contextual mention
      encoding rather than the full KRISSBERT system.

    Both facts are recorded in ``run_meta.json`` under ``provenance.deviation``.
    """

    needs_context = True

    def __init__(self, model_name: str = "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL",
                 *, device: str = "cpu", batch: int = 256, max_len: int = 64,
                 normalize: bool = False, context_window: int = 0):
        self.name = f"krissbert:{model_name.rsplit('/', 1)[-1]}"
        self.model_name = model_name
        self.device = device
        self.batch = batch
        self.max_len = max_len
        self.normalize = normalize
        self.context_window = context_window
        self.embs: np.ndarray | None = None

    def _encode(self, items: list[tuple[str, str, str]]) -> np.ndarray:
        """Encode (mention, left context, right context) triples, CLS-pooled."""
        torch = self._torch
        tok, model = self._tok, self._model
        out = np.empty((len(items), model.config.hidden_size), dtype=np.float32)
        ent_open = tok.convert_tokens_to_ids(["<ENT>"])
        ent_close = tok.convert_tokens_to_ids(["</ENT>"])
        for start in range(0, len(items), self.batch):
            chunk = items[start:start + self.batch]
            rows = []
            for mention, ctx_l, ctx_r in chunk:
                m_ids = tok.encode(mention, add_special_tokens=False,
                                   max_length=self.max_len, truncation=True)
                core = ent_open + m_ids + ent_close
                budget = self.max_len - len(core) - 2
                l_ids = r_ids = []
                if budget > 0:
                    l_ids = tok.encode(ctx_l, add_special_tokens=False)[-(budget // 2):] if ctx_l else []
                    r_ids = tok.encode(ctx_r, add_special_tokens=False)[: budget - len(l_ids)] if ctx_r else []
                ids = [tok.cls_token_id] + l_ids + core + r_ids + [tok.sep_token_id]
                ids = ids[: self.max_len]
                rows.append(ids)
            width = max(len(r) for r in rows)
            pad = tok.pad_token_id
            batch_ids = torch.tensor([r + [pad] * (width - len(r)) for r in rows],
                                     device=self.device)
            attn = (batch_ids != pad).long()
            with torch.no_grad():
                hid = model(input_ids=batch_ids,
                            token_type_ids=torch.zeros_like(batch_ids),
                            attention_mask=attn).last_hidden_state
            vec = hid[:, 0, :].float()
            if self.normalize:
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out[start:start + len(chunk)] = vec.cpu().numpy()
        return out

    def fit(self, strings: list[str]) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._strings = strings
        self._torch = torch
        print(f"  loading {self.model_name} ...", flush=True)
        self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        for marker in ("<ENT>", "</ENT>"):
            if self._tok.convert_tokens_to_ids(marker) == self._tok.unk_token_id:
                raise SystemExit(
                    f"ERROR: {marker} missing from the KRISSBERT vocabulary; the "
                    "mention markers would become [UNK] and every score would be noise"
                )
        t0 = time.time()
        self.embs = self._encode([(s, "", "") for s in strings])
        print(f"  KRISSBERT: encoded {len(strings):,} entity names as context-free "
              f"prototypes in {time.time() - t0:.0f}s", flush=True)

    def score(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray:
        if len(cands) == 0:
            return np.empty(0, dtype=np.float32)
        left, right = query.context_left, query.context_right
        if self.context_window:
            left, right = left[-self.context_window:], right[: self.context_window]
        q = self._encode([(query.text, left, right)])[0]
        flat = (self.embs[cands.alias_idx] @ q).astype(np.float32)
        self._last_flat = flat
        return groupwise_max(flat, cands.group_ptr)

    def score_flat(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray | None:
        return getattr(self, "_last_flat", None)


def _revive_sklearn_tfidf(vec):
    """Make a TfidfVectorizer pickled by sklearn 0.24 usable under sklearn 1.x.

    BioSyn's ``sparse_encoder.pk`` was pickled in 2020. The object unpickles but
    is missing attributes the modern code reads, and its fitted state lives in
    ``_tfidf._idf_diag`` where sklearn>=1.0 expects ``idf_``/``n_features_in_``.
    Reviving it keeps the *exact* vocabulary and IDF the model was trained with
    (verified: identical to a fresh ``char (1,2)`` fit on the benchmark's
    train_dictionary -- 1122 features, IDF correlation 1.0000).
    """
    import numpy as _np

    for key, val in dict(sublinear_tf=False, norm="l2", use_idf=True, smooth_idf=True,
                         dtype=_np.float64).items():
        if not hasattr(vec, key):
            setattr(vec, key, val)
    tt = getattr(vec, "_tfidf", None)
    if tt is not None:
        for key, val in dict(sublinear_tf=False, norm="l2", use_idf=True,
                             smooth_idf=True).items():
            if not hasattr(tt, key):
                setattr(tt, key, val)
        if not hasattr(tt, "idf_") and hasattr(tt, "_idf_diag"):
            idf = _np.ravel(tt._idf_diag.diagonal())
            tt.idf_ = idf
            tt.n_features_in_ = len(idf)
    return vec


class BioSynScorer(StringPairScorer):
    """BioSyn's native hybrid score (Sung et al., ACL 2020).

        score(q, d) = <CLS(q), CLS(d)> + sparse_weight * cos(tfidf(q), tfidf(d))

    Reproduced from the released code, because every deviation matters here:

    * The dense half is a **raw dot product of un-normalised [CLS] vectors**
      (``rerankNet.forward``: ``torch.bmm(query_embed, candidate_embeds.T)``;
      ``biosyn.get_score_matrix``: ``np.matmul``). L2-normalising it changes the
      relative weight of the two halves completely.
    * ``sparse_weight`` is **learned** and shipped in the checkpoint as
      ``sparse_weight.pt``: 0.410 (ncbi-disease), 0.098 (bc5cdr-disease),
      0.075 (bc5cdr-chemical). A 50/50 convex mix is rank-equivalent to
      ``sparse_weight = 1.0``, i.e. 2.4x-13x too much sparse signal, which
      inflates exactly the lexical negatives.
    * The sparse half is ``TfidfVectorizer(analyzer='char', ngram_range=(1, 2))``
      **fit on the benchmark's train dictionary** and shipped as
      ``sparse_encoder.pk``; it is loaded and only *transformed* with, never
      refit on the evaluation strings.
    * Inputs go through BioSyn's ``TextPreprocess`` (lowercase + punctuation
      stripped) first.

    ``alpha=`` remains available as an explicit non-native ablation
    (``alpha*dense_cos + (1-alpha)*sparse``); ``run_meta.json`` records which
    scoring mode was used.
    """

    DATASET_CHECKPOINTS = {
        "ncbi_disease": "dmis-lab/biosyn-sapbert-ncbi-disease",
        "bc5cdr_disease": "dmis-lab/biosyn-sapbert-bc5cdr-disease",
        "bc5cdr_chemical": "dmis-lab/biosyn-sapbert-bc5cdr-chemical",
        "nlmchem": "dmis-lab/biosyn-sapbert-bc5cdr-chemical",
        "medmentions": "dmis-lab/biosyn-sapbert-bc5cdr-disease",
    }
    IN_DOMAIN = frozenset({"ncbi_disease", "bc5cdr_disease", "bc5cdr_chemical"})
    DEFAULT_MODEL = "dmis-lab/biosyn-sapbert-bc5cdr-disease"

    _PUNCT_RE = re.compile(r"[\s{}]+".format(re.escape(__import__("string").punctuation)))

    @classmethod
    def preprocess(cls, text: str) -> str:
        return " ".join(cls._PUNCT_RE.split(text.lower())).strip()

    def __init__(self, model_name: str | None = None, *, device: str = "cpu",
                 batch: int = 256, max_len: int = 32, alpha: float | None = None,
                 sparse_weight: float | None = None, pooling: str = "cls"):
        hf_id = model_name or self.DEFAULT_MODEL
        self.hf_id = hf_id
        self.alpha = None if alpha is None else float(alpha)
        self.sparse_weight = None if sparse_weight is None else float(sparse_weight)
        mode = f"alpha={self.alpha}" if self.alpha is not None else "native"
        self.name = f"biosyn:{hf_id.rsplit('/', 1)[-1]}({mode})"
        self.dense = DenseEncoderScorer(hf_id, device=device, batch=batch,
                                        max_len=max_len, pooling=pooling,
                                        normalize=self.alpha is not None)
        self._sparse = None

    def _load_sparse_assets(self):
        """Return (fitted TfidfVectorizer, learned sparse_weight)."""
        import pickle

        import torch

        local = Path(self.hf_id)
        if local.is_dir():
            enc_path, w_path = local / "sparse_encoder.pk", local / "sparse_weight.pt"
        else:
            from huggingface_hub import hf_hub_download

            enc_path = Path(hf_hub_download(self.hf_id, "sparse_encoder.pk"))
            w_path = Path(hf_hub_download(self.hf_id, "sparse_weight.pt"))
        with enc_path.open("rb") as f:
            vec = _revive_sklearn_tfidf(pickle.load(f))
        raw = torch.load(w_path, map_location="cpu", weights_only=False)
        weight = float(raw.detach() if hasattr(raw, "detach") else raw)
        return vec, weight

    def fit(self, strings: list[str]) -> None:
        processed = [self.preprocess(s) for s in strings]
        self.dense.fit(processed)

        t0 = time.time()
        vec, learned_w = self._load_sparse_assets()
        if self.sparse_weight is None:
            self.sparse_weight = learned_w
        X = vec.transform(processed).astype(np.float32)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        from scipy.sparse import diags

        self._sparse = diags(1.0 / norms) @ X
        print(f"  BioSyn sparse: char(1,2) TF-IDF from the checkpoint, "
              f"{len(vec.vocabulary_):,} features, learned sparse_weight={learned_w:.4f}"
              f"{'' if self.alpha is None else f' (overridden by alpha={self.alpha})'} "
              f"({time.time() - t0:.1f}s)", flush=True)

    def score_strings(self, query_idx: int, doc_idx: np.ndarray) -> np.ndarray:
        dense_s = self.dense.score_strings(query_idx, doc_idx)
        q = self._sparse[query_idx]
        sparse_s = (self._sparse[doc_idx] @ q.T).toarray().reshape(-1).astype(np.float32)
        if self.alpha is not None:
            return (self.alpha * dense_s + (1.0 - self.alpha) * sparse_s).astype(np.float32)
        return (dense_s + self.sparse_weight * sparse_s).astype(np.float32)


class CrossEncoderScorer(Scorer):
    """``[CLS] mention [SEP] alias [SEP]`` -> one logit per pair.

    No reusable per-string embedding exists, so cost is one forward pass per
    (mention, alias) pair. Use it on a sampled run, not on all 5.6M MedMentions
    pairs.
    """

    def __init__(self, model_name: str, *, device: str = "cpu", batch: int = 64,
                 max_len: int = 128):
        self.name = f"crossencoder:{model_name.rsplit('/', 1)[-1]}"
        self.model_name = model_name
        self.device = device
        self.batch = batch
        self.max_len = max_len

    def fit(self, strings: list[str]) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"  loading cross-encoder {self.model_name} ...", flush=True)
        self._strings = strings
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self._model.eval()

    def _score_pairs(self, query_text: str, doc_texts: list[str]) -> np.ndarray:
        import torch

        out = np.empty(len(doc_texts), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(doc_texts), self.batch):
                chunk = doc_texts[i : i + self.batch]
                enc = self._tok([query_text] * len(chunk), chunk, padding=True,
                                truncation=True, max_length=self.max_len,
                                return_tensors="pt").to(self.device)
                logits = self._model(**enc).logits
                if logits.ndim == 2 and logits.shape[1] == 2:
                    s = logits[:, 1] - logits[:, 0]
                else:
                    s = logits.reshape(-1)
                out[i : i + len(chunk)] = s.float().cpu().numpy()
        return out

    def score(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray:
        if len(cands) == 0:
            return np.empty(0, dtype=np.float32)
        texts = [self._strings[i] for i in cands.alias_idx]
        flat = self._score_pairs(query.text, texts)
        self._last_flat = flat
        return groupwise_max(flat, cands.group_ptr)

    def score_flat(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray | None:
        return getattr(self, "_last_flat", None)


class ArboELBiencoderScorer(Scorer):
    """Dot product between a context-aware mention encoder and an entity encoder.

    ArboEL is a BLINK derivative: two independent BERT towers, no projection
    head (``add_linear=False`` in the released MedMentions config), scored as a
    raw dot product of ``[CLS]`` vectors.

      mention side: ``[CLS] left [unused1] mention [unused2] right [SEP]``
      entity side:  ``[CLS] name [unused3] description [SEP]``

    The tags are arboEL's, from ``blink/common/params.py``
    (``ENT_START_TAG="[unused1]"``, ``ENT_END_TAG="[unused2]"``,
    ``ENT_TITLE_TAG="[unused3]"``) -- *not* upstream BLINK's
    ``[unused0]/[unused1]/[unused2]``. BioBERT's vocab has no ``[unused0]`` at
    all, so the upstream constants silently tokenise to ``[UNK]``; the
    constructor asserts every tag resolves to a real id.

    ``description`` is not the UMLS definition (empty for most concepts) but
    arboEL's own alias-expanded string, e.g.

        Apoptosis ( Biologic Function , morphologic abnormality : apoptotic cell
        death ; Apoptotic Process ; Programmed Cell Death ; ... )

    Pass ``entity_dict=<arboEL entity_dictionary.pickle>`` to reuse the exact
    pre-tokenised entity representations the model was trained and evaluated
    with; concepts missing from it (our negatives come from UMLS 2020AA, that
    dictionary from 2017AA) are rebuilt with the same template from
    concepts.pkl. The split is reported so the mismatch stays visible.

    The published weights are a ``BiEncoderRanker`` state dict
    (``context_encoder.*`` / ``cand_encoder.*``); this loads the two towers
    directly rather than importing the arboEL package, whose BLINK fork pins an
    old transformers.

    Entity representations depend only on the concept, so every entity that
    appears in the run is encoded once and cached. Alias groups are ignored:
    ArboEL scores the concept, not its surface forms.
    """

    ENT_START, ENT_END, ENT_TITLE = "[unused1]", "[unused2]", "[unused3]"
    needs_entities = True
    needs_context = True

    def __init__(self, ckpt: str, *, bert_model: str | None = None, device: str = "cpu",
                 batch: int = 128, max_context_len: int = 128, max_cand_len: int = 128,
                 lowercase: bool = False, entity_dict: str | None = None):
        self.name = f"arboel-biencoder:{Path(ckpt).name}"
        self.ckpt = ckpt
        self.bert_model = bert_model
        self.device = device
        self.batch = batch
        self.max_context_len = max_context_len
        self.max_cand_len = max_cand_len
        self.lowercase = lowercase
        self.entity_dict = entity_dict
        self._ent_ids: dict[str, list[int]] = {}
        self._ent_row: dict[str, int] = {}
        self._ent_embs: np.ndarray | None = None

    def _load_towers(self):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        ckpt = Path(self.ckpt)
        bin_path = ckpt if ckpt.is_file() else ckpt / "pytorch_model.bin"
        if not bin_path.exists():
            raise SystemExit(f"ERROR: ArboEL checkpoint not found: {bin_path}")
        ckpt_dir = bin_path.parent
        vocab = ckpt_dir / "vocab.txt"
        config_json = ckpt_dir / "config.json"
        if self.bert_model:
            print(f"  ArboEL: tokenizer/config from {self.bert_model}", flush=True)
            self._tok = AutoTokenizer.from_pretrained(self.bert_model,
                                                      do_lower_case=self.lowercase)
            config = AutoConfig.from_pretrained(self.bert_model)
        elif vocab.exists() and config_json.exists():
            from transformers import BertConfig, BertTokenizerFast

            print(f"  ArboEL: tokenizer/config from {ckpt_dir}", flush=True)
            self._tok = BertTokenizerFast.from_pretrained(
                str(ckpt_dir), do_lower_case=self.lowercase
            )
            config = BertConfig.from_json_file(str(config_json))
        else:
            raise SystemExit(
                f"ERROR: {ckpt_dir} has no vocab.txt/config.json; pass bert=<path to the "
                "BERT it was initialised from> (e.g. arboEL/models/biobert-base-cased-v1.1)."
            )

        print(f"  ArboEL: loading state dict {bin_path} ...", flush=True)
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)

        towers = {}
        for prefix, key in (("context_encoder", "ctx"), ("cand_encoder", "cand")):
            sub = {}
            for k, v in state.items():
                if not k.startswith(prefix + "."):
                    continue
                rest = k[len(prefix) + 1 :]
                if rest.startswith("bert_model."):
                    rest = rest[len("bert_model.") :]
                sub[rest] = v
            if not sub:
                raise SystemExit(
                    f"ERROR: no '{prefix}.*' weights in {bin_path}. Keys look like: "
                    f"{list(state)[:5]}"
                )
            model = AutoModel.from_config(config)
            missing, unexpected = model.load_state_dict(sub, strict=False)
            drop = [k for k in unexpected if not k.endswith("position_ids")]
            heavy_missing = [k for k in missing if "pooler" not in k]
            if heavy_missing or drop:
                print(f"  WARNING: {prefix}: missing={heavy_missing[:5]} unexpected={drop[:5]}")
            model.to(self.device).eval()
            towers[key] = model
        self._ctx_model, self._cand_model = towers["ctx"], towers["cand"]

        vocab = self._tok.get_vocab()
        if len(vocab) != config.vocab_size:
            raise SystemExit(
                f"ERROR: tokenizer vocab has {len(vocab)} entries but the model expects "
                f"{config.vocab_size}. The tokenizer did not load the checkpoint vocab."
            )
        probe = self._tok.tokenize("Apoptosis of epithelial cells")
        if not probe or all(t == self._tok.unk_token for t in probe):
            raise SystemExit(f"ERROR: tokenizer produced only unknown tokens: {probe}")
        for tag in (self.ENT_START, self.ENT_END, self.ENT_TITLE):
            if tag not in vocab:
                raise SystemExit(
                    f"ERROR: ArboEL tag {tag} is not in the checkpoint vocab, so the "
                    f"mention/entity markers would be meaningless."
                )
        print(f"  ArboEL: vocab={len(vocab)}, tags "
              f"{self.ENT_START}={vocab[self.ENT_START]} "
              f"{self.ENT_END}={vocab[self.ENT_END]} "
              f"{self.ENT_TITLE}={vocab[self.ENT_TITLE]}, probe={probe[:4]}", flush=True)

    def _context_ids(self, query: MentionQuery) -> list[int]:
        """arboEL/blink/biencoder/data_process.py:get_context_representation."""
        tok = self._tok
        mention_tokens = [self.ENT_START] + tok.tokenize(query.text) + [self.ENT_END]
        left = tok.tokenize(query.context_left) if query.context_left else []
        right = tok.tokenize(query.context_right) if query.context_right else []
        left_quota = (self.max_context_len - len(mention_tokens)) // 2 - 1
        right_quota = self.max_context_len - len(mention_tokens) - left_quota - 2
        left_quota, right_quota = max(left_quota, 0), max(right_quota, 0)
        if len(left) <= left_quota:
            if len(right) > right_quota:
                right_quota += left_quota - len(left)
        elif len(right) <= right_quota:
            left_quota += right_quota - len(right)
        tokens = (left[-left_quota:] if left_quota else []) + mention_tokens \
            + (right[:right_quota] if right_quota else [])
        tokens = [tok.cls_token] + tokens[: self.max_context_len - 2] + [tok.sep_token]
        return tok.convert_tokens_to_ids(tokens)

    def _load_entity_dict(self, entity_ids: list[str]) -> None:
        """Reuse arboEL's own pre-tokenised entity representations where present."""
        import gc
        import pickle

        path = Path(self.entity_dict)
        if not path.exists():
            raise SystemExit(f"ERROR: entity_dict not found: {path}")
        print(f"  ArboEL: loading entity dictionary {path} ...", flush=True)
        t0 = time.time()
        with path.open("rb") as f:
            records = pickle.load(f)
        wanted = set(entity_ids)
        tok = self._tok
        for rec in records:
            cui = rec.get("cui")
            if cui not in wanted:
                continue
            if "ids" in rec:
                ids = [int(x) for x in rec["ids"]]
                while ids and ids[-1] == 0:
                    ids.pop()
                self._ent_ids[cui] = ids
                continue
            title, desc = rec.get("title", ""), rec.get("description", "")
            tokens = tok.tokenize(title) + [self.ENT_TITLE] + tok.tokenize(desc)
            tokens = [tok.cls_token] + tokens[: self.max_cand_len - 2] + [tok.sep_token]
            self._ent_ids[cui] = tok.convert_tokens_to_ids(tokens)
        del records
        gc.collect()
        print(f"  ArboEL: reused {len(self._ent_ids):,}/{len(entity_ids):,} entity "
              f"representations from arboEL's dictionary ({time.time() - t0:.0f}s)",
              flush=True)

    def _entity_ids(self, entity_id: str, rec: dict) -> list[int]:
        cached = self._ent_ids.get(entity_id)
        if cached:
            return cached
        tok = self._tok
        if "title" in rec and "description" in rec:
            name, desc = rec["title"], rec["description"]
        else:
            name = rec.get("preferred_name") or next(iter(rec.get("aliases") or []), "")
            types = " , ".join(rec.get("semantic_types") or [])
            aliases = " ; ".join(a for a in (rec.get("aliases") or []) if a and a != name)
            if aliases:
                desc = f"{name} ( {types} : {aliases} )"
            elif types:
                desc = f"{name} ( {types} )"
            else:
                desc = name
        tokens = tok.tokenize(name) + [self.ENT_TITLE] + tok.tokenize(desc)
        tokens = [tok.cls_token] + tokens[: self.max_cand_len - 2] + [tok.sep_token]
        return tok.convert_tokens_to_ids(tokens)

    def _encode(self, model, id_lists: list[list[int]]) -> np.ndarray:
        import torch

        pad = self._tok.pad_token_id
        width = max(len(x) for x in id_lists)
        ids = np.full((len(id_lists), width), pad, dtype=np.int64)
        mask = np.zeros((len(id_lists), width), dtype=np.int64)
        for i, x in enumerate(id_lists):
            ids[i, : len(x)] = x
            mask[i, : len(x)] = 1
        with torch.no_grad():
            out = model(
                input_ids=torch.from_numpy(ids).to(self.device),
                attention_mask=torch.from_numpy(mask).to(self.device),
            ).last_hidden_state[:, 0, :]
        return out.float().cpu().numpy()

    def fit(self, strings: list[str]) -> None:
        self._strings = strings
        self._load_towers()

    def fit_entities(self, entity_ids: list[str], concepts: dict) -> None:
        if self.entity_dict:
            self._load_entity_dict(entity_ids)
        else:
            print("  WARNING: no entity_dict= given; entity descriptions are rebuilt "
                  "from concepts.pkl rather than reused from arboEL's own dictionary.",
                  flush=True)
        t0 = time.time()
        print(f"  ArboEL: encoding {len(entity_ids):,} entities ...", flush=True)
        dim = self._cand_model.config.hidden_size
        embs = np.empty((len(entity_ids), dim), dtype=np.float32)
        for i in range(0, len(entity_ids), self.batch):
            chunk = entity_ids[i : i + self.batch]
            embs[i : i + len(chunk)] = self._encode(
                self._cand_model,
                [self._entity_ids(e, concepts.get(e, {})) for e in chunk],
            )
            if (i // self.batch) % 200 == 0:
                print(f"    {i + len(chunk):,}/{len(entity_ids):,} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        self._ent_embs = embs
        self._ent_row = {e: i for i, e in enumerate(entity_ids)}
        print(f"  ArboEL: entity cache ready ({time.time() - t0:.0f}s)", flush=True)

    def score(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray:
        if len(cands) == 0:
            return np.empty(0, dtype=np.float32)
        ctx = self._encode(self._ctx_model, [self._context_ids(query)])[0]
        rows = np.fromiter((self._ent_row[e] for e in cands.entity_ids),
                           dtype=np.int64, count=len(cands))
        return (self._ent_embs[rows] @ ctx).astype(np.float32)


_ANGEL_REPLACED_PUNCT = "!\"$%'()*+,-./;<=>?[]_`{|}"
_ANGEL_PUNCT_TABLE = {ord(c): " " for c in _ANGEL_REPLACED_PUNCT}


def angel_normalise(name: str) -> str:
    """Entity-name surface form used by ANGEL / GenBioEL target_kb and trie."""
    return " ".join(name.lower().translate(_ANGEL_PUNCT_TABLE).split())


class Seq2SeqLikelihoodScorer(Scorer):
    """Length-normalised log-likelihood of each candidate name.

    Two input templates, matching the checkpoints' training format.

    ``angel`` (GenBioEL and ANGEL, which reuses GenBioEL's data pipeline)::

        encoder : "<left>START <mention> END<right>"
        decoder : "</s> <mention> is <candidate name>"
        scored  : the candidate-name tokens plus EOS

    The ``"<mention> is"`` decoder prefix is GenBioEL's ``prefix_mention_is``
    setting, used by ANGEL's released inference script; the prefix tokens are
    force-fed and excluded from the score. Both repos lowercase their benchmark
    text, hence ``lowercase=True`` by default for this template.

    ``genre``::

        encoder : "<left>[START_ENT] <mention> [END_ENT]<right>"
        decoder : "</s><s> <candidate name>"

    The encoder states are computed once per mention and reused across
    candidates, so the cost is one decoder pass per (mention, alias) pair.
    """

    needs_context = True

    HF_DEFAULTS = {"genre": "facebook/genre-linking-blink"}
    ANGEL_CHECKPOINTS = {
        "ncbi_disease": "dmis-lab/ANGEL_ncbi",
        "bc5cdr_disease": "dmis-lab/ANGEL_bc5cdr",
        "bc5cdr_chemical": "dmis-lab/ANGEL_bc5cdr",
        "medmentions": "dmis-lab/ANGEL_mm",
        "nlmchem": "dmis-lab/ANGEL_bc5cdr",
    }
    IN_DOMAIN = frozenset({"ncbi_disease", "bc5cdr_disease", "bc5cdr_chemical", "medmentions"})

    def __init__(self, model_name: str, *, template: str = "angel", label: str | None = None,
                 tokenizer_name: str | None = None, device: str = "cpu", batch: int = 32,
                 max_source_len: int = 256, max_target_len: int = 48,
                 lowercase: bool | None = None, entity_norm: bool | None = None,
                 length_norm: str = "mean", context_window: int = 0):
        self.name = f"{label or template}:{model_name.rsplit('/', 1)[-1]}"
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or (
            "facebook/bart-large" if template == "angel" else model_name
        )
        self.template = template
        self.device = device
        self.batch = batch
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.lowercase = (template == "angel") if lowercase is None else lowercase
        self.entity_norm = (template == "angel") if entity_norm is None else entity_norm
        self.length_norm = length_norm
        self.context_window = context_window
        self._src_cache: tuple[str, object] | None = None

    def fit(self, strings: list[str]) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._strings = strings
        print(f"  loading seq2seq {self.model_name} (tokenizer {self.tokenizer_name}) ...",
              flush=True)
        self._tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        self._torch = torch
        cfg = self._model.config
        self.eos_id = cfg.eos_token_id
        self.bos_id = cfg.bos_token_id
        self.pad_id = cfg.pad_token_id
        self.is_id = self._tok(" is", add_special_tokens=False)["input_ids"][-1]

    def _source_text(self, query: MentionQuery) -> str:
        left, right = query.context_left, query.context_right
        if self.context_window:
            left = left[-self.context_window :]
            right = right[: self.context_window]
        if self.template == "genre":
            text = f"{left}[START_ENT] {query.text} [END_ENT]{right}"
        else:
            text = f"{left}START {query.text} END{right}"
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower() if self.lowercase else text

    def _body_ids(self, text: str) -> list[int]:
        """Token ids of ``' ' + text`` without BOS/EOS, as GenBioEL encodes."""
        return self._tok(" " + text, add_special_tokens=False)["input_ids"]

    def entity_surface(self, name: str) -> str:
        """The exact string this checkpoint was fine-tuned to generate for a concept.

        Candidate names must be fed in the form the model's decoding trie holds,
        otherwise forced likelihood measures the wrong thing: 12.99% of our
        candidate names (104273/802760 on medmentions) differ from their ANGEL
        surface form, so their log-likelihood is depressed by punctuation tokens
        the checkpoint was never trained to emit -- nothing to do with whether the
        concept is right. The penalty is concentrated in punctuation-heavy names
        (chemicals, strains, 'Copper 2+'), i.e. unevenly across our negative
        buckets, so it biases exactly the per-bucket comparison we report.

        Deliberately *not* applied to the encoder source or the ``<mention> is``
        decoder prefix: ANGEL's own test.source keeps commas and hyphens, and
        11.2% of its test.target mention prefixes contain punctuation. Those two
        are lowercased only, which ``_source_text`` and ``_prefix_ids`` already do.

        This maps candidate names onto the same surface form our own alias-selection
        protocol chose; it never consults ANGEL's target_kb, so no candidate is
        dropped or given a different treatment depending on whether the authors
        happened to include its concept (their mm target_kb covers 99.4% of our
        gold CUIs but only 75.5% of our negative CUIs -- looking names up there
        would hand gold and negatives two different code paths).
        """
        if self.entity_norm:
            return angel_normalise(name)
        return name.lower() if self.lowercase else name

    def _encoder_state(self, query: MentionQuery):
        torch = self._torch
        src = self._source_text(query)
        if self._src_cache and self._src_cache[0] == src:
            return self._src_cache[1]
        ids = [self.bos_id] + self._body_ids(src)[: self.max_source_len - 2] + [self.eos_id]
        input_ids = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            enc = self._model.get_encoder()(input_ids=input_ids,
                                            attention_mask=torch.ones_like(input_ids))
        self._src_cache = (src, (enc.last_hidden_state, input_ids.shape[1]))
        return self._src_cache[1]

    def _prefix_ids(self, query: MentionQuery) -> list[int]:
        if self.template == "genre":
            return [self.bos_id]
        mention = query.text.lower() if self.lowercase else query.text
        return self._body_ids(mention) + [self.is_id]

    def score(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray:
        if len(cands) == 0:
            return np.empty(0, dtype=np.float32)
        flat = self._score_aliases(query, [self._strings[i] for i in cands.alias_idx])
        self._last_flat = flat
        return groupwise_max(flat, cands.group_ptr)

    def score_flat(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray | None:
        return getattr(self, "_last_flat", None)

    def _score_aliases(self, query: MentionQuery, aliases: list[str]) -> np.ndarray:
        from transformers.modeling_outputs import BaseModelOutput

        torch = self._torch
        enc_hidden, src_len = self._encoder_state(query)
        prefix = self._prefix_ids(query)
        targets = []
        for alias in aliases:
            name = self.entity_surface(alias)
            body = self._body_ids(name)[: self.max_target_len - 1]
            targets.append(body + [self.eos_id])

        out = np.empty(len(aliases), dtype=np.float32)
        for i in range(0, len(targets), self.batch):
            chunk = targets[i : i + self.batch]
            width = len(prefix) + max(len(t) for t in chunk)
            n = len(chunk)
            dec = np.full((n, width + 1), self.pad_id, dtype=np.int64)
            lab = np.full((n, width + 1), -100, dtype=np.int64)
            dec_mask = np.zeros((n, width + 1), dtype=np.int64)
            for r, tgt in enumerate(chunk):
                seq = [self.eos_id] + prefix + tgt
                dec[r, : len(seq)] = seq
                dec_mask[r, : len(seq)] = 1
                lab[r, len(prefix) : len(prefix) + len(tgt)] = tgt
            dec_t = torch.from_numpy(dec).to(self.device)
            enc_batch = BaseModelOutput(
                last_hidden_state=enc_hidden.expand(n, -1, -1).contiguous()
            )
            with torch.no_grad():
                logits = self._model(
                    encoder_outputs=enc_batch,
                    attention_mask=torch.ones((n, src_len), dtype=torch.long,
                                              device=self.device),
                    decoder_input_ids=dec_t,
                    decoder_attention_mask=torch.from_numpy(dec_mask).to(self.device),
                ).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            lab_t = torch.from_numpy(lab).to(self.device)
            picked = logprobs.gather(2, lab_t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            keep = (lab_t != -100).float()
            total = (picked * keep).sum(1)
            if self.length_norm == "mean":
                total = total / keep.sum(1).clamp(min=1.0)
            out[i : i + n] = total.cpu().numpy()
        return out


class RestrictiveLLMScorer(Scorer):
    """LLM4BioEL's restrictive-decoding half, applied to an off-the-shelf LLM.

    *Lin et al., Findings of EMNLP 2025* ("Guiding LLMs for Biomedical Entity
    Linking via Restrictive and Contrastive Decoding"). Tuning-free: no weights are
    adapted, only the decoding is constrained.

    Their prompt, verbatim in shape (``run_bioEL.py:organize_prompts``)::

        Question: [<lowercased document, mention wrapped in START ... END>
        Answer: <gold entity name>
        ==
        ... n in-context examples ...
        Question: <the input>
        Answer:

    and the candidate is scored by the forced log-likelihood of its name after
    ``"Answer: "``, length-normalised. Restrictive decoding constrains generation
    to the candidate set and maximises exactly this quantity, so taking the argmax
    over candidates *is* their decoding, without beam-search error -- the same
    equivalence constrained_decode.py verifies for the other generative
    backends.

    **The contrastive half is deliberately not implemented.** It blends the LLM's
    distribution with one derived from a retriever's scores over the candidates,
    and in their pipeline that retriever (SapBERT) also *produces* the candidate
    set by taking top-k out of the whole KB. Here the candidate set is given -- it
    is our gold plus typed negatives -- so the retriever's primary job does not
    exist, and folding SapBERT scores in would (a) duplicate the SapBERT row we
    already report and (b) make any change unattributable between the two halves.
    Their own ablation puts the cost at 0.3-1.4 Hits@1 (NCBI 93.2 -> 91.8,
    BC5CDR 92.2 -> 91.9, COMETA 84.4 -> 83.6), while dropping few-shot costs
    1.4-9.4 -- so the part we keep is the part that matters. This corresponds to
    ``--use_dynamics 0`` in their code, i.e. a configuration they themselves run.

    Name the row accordingly (``llm4bioel-restrictive:<backbone>``), never plain
    "LLM4BioEL".

    ICL examples come from ``mentions_train.jsonl`` (built by
    ``code/preprocess/parse_train_splits.sh``), drawn per mention with a seeded RNG.
    Their retrieval-based selection needs mention-mention similarity; random
    selection is their ``w/ random prompts`` variant and costs 1.1-3.0 Hits@1, so
    it is a documented configuration rather than an invention. ``n_shots`` defaults
    to 10, their default and the size of the pool they ship.
    """

    needs_context = True

    def __init__(self, model_name: str, *, device: str = "cpu", batch: int = 8,
                 n_shots: int = 10, seed: int = 17, context_window: int = 400,
                 max_source_len: int = 2048, max_target_len: int = 48,
                 length_norm: str = "mean", dtype: str = "bfloat16",
                 train_mentions: Path | str | None = None,
                 concepts_slim: Path | str | None = None):
        self.name = f"llm4bioel-restrictive:{model_name.rsplit('/', 1)[-1]}"
        self.model_name = model_name
        self.device = device
        self.batch = batch
        self.n_shots = n_shots
        self.seed = seed
        self.context_window = context_window
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.length_norm = length_norm
        self.dtype = dtype
        self.train_mentions = Path(train_mentions) if train_mentions else None
        self.concepts_slim = Path(concepts_slim) if concepts_slim else None
        self._shots: list[tuple[str, str]] = []

    @staticmethod
    def _question(mention: str, left: str, right: str) -> str:
        """Their prompt body: a lowercased document with the mention marked.

        The leading '[' is theirs (see data/ncbi_test_*.json) and is kept so the
        surface form the model conditions on matches the released files.
        """
        text = f"[{left}START {mention} END{right}"
        return re.sub(r"\s+", " ", text).strip().lower()

    def _load_shots(self) -> None:
        """Build the ICL pool: (question, gold name) pairs from the train split."""
        if not self.train_mentions or not self.train_mentions.is_file():
            print("  LLM4BioEL: no train mentions given -> zero-shot "
                  "(their w/o few-shot ablation)", flush=True)
            return
        import json
        import pickle

        from inference import pipeline as _io

        names: dict[str, str] = {}
        if self.concepts_slim and self.concepts_slim.is_file():
            slim = pickle.loads(self.concepts_slim.read_bytes())
            names = {eid: (rec.get("preferred_name") or "")
                     for eid, rec in slim.items()}
        pool = []
        with self.train_mentions.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                gold = [names.get(e, "") for e in (rec.get("gold_entity_ids") or [])]
                gold = [g for g in gold if g]
                if not gold:
                    continue
                left, right, _ = _io.mention_context(rec)
                if self.context_window:
                    left = left[-self.context_window:]
                    right = right[: self.context_window]
                pool.append((self._question(rec.get("mention") or "", left, right),
                             gold[0].lower()))
        self._shots = pool
        print(f"  LLM4BioEL: {len(pool):,} ICL candidates from "
              f"{self.train_mentions.name}, n_shots={self.n_shots}", flush=True)

    def _prefix_for(self, query: MentionQuery) -> str:
        import random

        rng = random.Random(f"{self.seed}:{query.mention_id}")
        block = ""
        if self._shots and self.n_shots > 0:
            picks = rng.sample(self._shots, min(self.n_shots, len(self._shots)))
            for question, answer in picks:
                block += f"Question: {question}\nAnswer: {answer}\n==\n"
        left, right = query.context_left, query.context_right
        if self.context_window:
            left, right = left[-self.context_window:], right[: self.context_window]
        block += f"Question: {self._question(query.text, left, right)}\nAnswer:"
        return block

    def fit(self, strings: list[str]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._strings = strings
        self._torch = torch
        print(f"  loading {self.model_name} ({self.dtype}) ...", flush=True)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[self.dtype]
        kwargs = {"dtype": dt, "low_cpu_mem_usage": True}
        if self.device.startswith("cuda") and torch.cuda.device_count() > 1:
            kwargs["device_map"] = "auto"
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **kwargs).to(self.device)
        self._model.eval()
        self._load_shots()

    def score(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray:
        if len(cands) == 0:
            return np.empty(0, dtype=np.float32)
        flat = self._score_names(query, [self._strings[i] for i in cands.alias_idx])
        self._last_flat = flat
        return groupwise_max(flat, cands.group_ptr)

    def score_flat(self, query: MentionQuery, cands: CandidateBatch) -> np.ndarray | None:
        return getattr(self, "_last_flat", None)

    def _score_names(self, query: MentionQuery, names: list[str]) -> np.ndarray:
        torch = self._torch
        tok = self._tok
        prefix_ids = tok(self._prefix_for(query), add_special_tokens=True)["input_ids"]
        prefix_ids = prefix_ids[-self.max_source_len:]
        targets = [
            tok(" " + n.lower(), add_special_tokens=False)["input_ids"][: self.max_target_len - 1]
            + [tok.eos_token_id]
            for n in names
        ]
        out = np.empty(len(names), dtype=np.float32)
        dev = next(self._model.parameters()).device
        for i in range(0, len(targets), self.batch):
            chunk = targets[i : i + self.batch]
            width = len(prefix_ids) + max(len(t) for t in chunk)
            pad = tok.pad_token_id
            ids = np.full((len(chunk), width), pad, dtype=np.int64)
            lab = np.full((len(chunk), width), -100, dtype=np.int64)
            attn = np.zeros((len(chunk), width), dtype=np.int64)
            for r, tgt in enumerate(chunk):
                seq = prefix_ids + tgt
                ids[r, : len(seq)] = seq
                attn[r, : len(seq)] = 1
                lab[r, len(prefix_ids) : len(seq)] = tgt
            with torch.no_grad():
                logits = self._model(
                    input_ids=torch.from_numpy(ids).to(dev),
                    attention_mask=torch.from_numpy(attn).to(dev),
                ).logits.float()
            lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            tgt_t = torch.from_numpy(lab[:, 1:]).to(dev)
            keep = (tgt_t != -100)
            picked = lp.gather(2, tgt_t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            total = (picked * keep).sum(1)
            if self.length_norm == "mean":
                total = total / keep.sum(1).clamp(min=1)
            out[i : i + len(chunk)] = total.cpu().numpy()
        return out


def _kv(spec: str) -> tuple[str | None, dict[str, str]]:
    """Split ``"name:bare,key=value,..."`` into (bare token, {key: value})."""
    bare: str | None = None
    kv: dict[str, str] = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip()] = v.strip()
        else:
            bare = tok
    return bare, kv


def make_scorer(args) -> tuple[Scorer, dict]:
    """Build a Scorer from ``args.model``. Returns ``(scorer, provenance)``.

    ``provenance`` records what was actually loaded (resolved checkpoint,
    whether it is in-domain for ``args.dataset``) so 07 can store it in
    run_meta.json instead of leaving it implicit in the model slug.
    """
    spec = args.model.strip()
    head, _, rest = spec.partition(":")
    head_lc = head.lower()
    bare, kv = _kv(rest)
    prov: dict = {"model_spec": spec}

    if head_lc == "bm25":
        s = BM25Scorer(k1=float(kv.get("k1", 1.5)), b=float(kv.get("b", 0.75)))
        return s, prov

    if head_lc == "crossencoder":
        if not bare:
            raise SystemExit("ERROR: --model crossencoder:<hf_id> requires a checkpoint")
        return CrossEncoderScorer(bare, device=args.device,
                                  batch=max(1, args.batch_size // 4),
                                  max_len=max(args.max_length, 128)), prov

    if head_lc in {"biosyn", "biosyn-auto"}:
        hf_id = bare or kv.get("model")
        if hf_id is None:
            hf_id = BioSynScorer.DATASET_CHECKPOINTS.get(args.dataset, BioSynScorer.DEFAULT_MODEL)
            prov["checkpoint_selected_for_dataset"] = args.dataset
        prov["checkpoint"] = hf_id
        prov["in_domain"] = args.dataset in BioSynScorer.IN_DOMAIN
        if not prov["in_domain"]:
            print(f"  WARNING: no released BioSyn checkpoint for '{args.dataset}'; "
                  f"transferring {hf_id}. Report it as out-of-domain.", flush=True)
        prov["scoring"] = "native" if "alpha" not in kv else f"alpha={kv['alpha']}"
        return BioSynScorer(hf_id, device=args.device, batch=args.batch_size,
                            max_len=args.max_length,
                            alpha=float(kv["alpha"]) if "alpha" in kv else None,
                            sparse_weight=(float(kv["sparse_weight"])
                                           if "sparse_weight" in kv else None),
                            pooling=args.pooling), prov

    if head_lc == "arboel":
        ckpt = bare or kv.get("ckpt")
        if not ckpt:
            raise SystemExit(
                "ERROR: --model arboel:<checkpoint dir or pytorch_model.bin>[,bert=<path>]\n"
                "       ArboEL weights are not on the Hub; train with arboEL/ or use the "
                "authors' released checkpoint."
            )
        prov["entity_dict"] = kv.get("entity_dict")
        return ArboELBiencoderScorer(
            ckpt, bert_model=kv.get("bert"), device=args.device,
            batch=args.batch_size,
            max_context_len=int(kv.get("max_context_len", 128)),
            max_cand_len=int(kv.get("max_cand_len", 128)),
            lowercase=kv.get("lowercase", "false").lower() == "true",
            entity_dict=kv.get("entity_dict"),
        ), prov

    if head_lc in {"angel", "genbioel", "genre"}:
        template = "genre" if head_lc == "genre" else "angel"
        ckpt = bare or kv.get("model")
        if ckpt is None:
            if head_lc == "angel":
                ckpt = Seq2SeqLikelihoodScorer.ANGEL_CHECKPOINTS.get(args.dataset)
                prov["checkpoint_selected_for_dataset"] = args.dataset
                prov["in_domain"] = args.dataset in Seq2SeqLikelihoodScorer.IN_DOMAIN
                if ckpt is None:
                    raise SystemExit(f"ERROR: no default ANGEL checkpoint for '{args.dataset}'")
                if not prov["in_domain"]:
                    print(f"  WARNING: no released ANGEL checkpoint for '{args.dataset}'; "
                          f"transferring {ckpt}.", flush=True)
            elif head_lc == "genre":
                ckpt = Seq2SeqLikelihoodScorer.HF_DEFAULTS["genre"]
            else:
                raise SystemExit(
                    "ERROR: --model genbioel:<path to checkpoint>\n"
                    "       GenBioEL weights are distributed from the authors' repo "
                    "(github.com/Yuanhy1997/GenBioEL), not the Hub."
                )
        prov["checkpoint"] = ckpt
        prov["template"] = template
        lowercase = kv.get("lowercase")
        entity_norm = kv.get("entity_norm")
        prov["entity_norm"] = entity_norm if entity_norm is not None else "default"
        return Seq2SeqLikelihoodScorer(
            ckpt, template=template, label=head_lc,
            tokenizer_name=kv.get("tokenizer"), device=args.device,
            batch=max(1, args.gen_batch_size), max_source_len=args.max_source_length,
            max_target_len=args.max_target_length,
            lowercase=None if lowercase is None else lowercase.lower() == "true",
            entity_norm=None if entity_norm is None else entity_norm.lower() == "true",
            length_norm=args.length_norm, context_window=args.context_window,
        ), prov

    if head_lc in {"llm4bioel", "restrictive"}:
        BACKBONES = {
            "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
            "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
            "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
            "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        }
        backbone = kv.get("backbone", "llama3-70b")
        ckpt = BACKBONES.get(backbone, backbone)
        prov["checkpoint"] = ckpt
        prov["in_domain"] = False
        prov["variant"] = "restrictive decoding only (their --use_dynamics 0)"
        prov["contrastive"] = (
            "not implemented: the candidate set is given by our protocol, so the "
            "retriever's role (top-k candidate generation) does not apply; their "
            "own ablation costs 0.3-1.4 Hits@1"
        )
        ds_dir = Path(getattr(args, "dataset_dir", ".") or ".")
        train = kv.get("train_mentions") or (ds_dir / "mentions_train.jsonl")
        slim = kv.get("concepts_slim") or (ds_dir / "concepts_slim.pkl")
        prov["n_shots"] = int(kv.get("shots", 10))
        return RestrictiveLLMScorer(
            ckpt, device=args.device, batch=max(1, args.gen_batch_size),
            n_shots=int(kv.get("shots", 10)), seed=args.seed,
            context_window=args.context_window or 400,
            max_source_len=max(512, args.max_source_length),
            max_target_len=args.max_target_length,
            length_norm=args.length_norm,
            dtype=kv.get("dtype", "bfloat16"),
            train_mentions=train, concepts_slim=slim,
        ), prov

    if head_lc == "krissbert":
        ckpt = kv.get("checkpoint", "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL")
        prov["checkpoint"] = ckpt
        prov["deviation"] = (
            "entity prototypes are not released (UMLS/PubMed licence); entities are "
            "represented by their own name encoded with empty context, so this is not "
            "the configuration behind the paper's reported numbers"
        )
        prov["in_domain"] = True
        return KrissBertScorer(
            ckpt, device=args.device, batch=args.batch_size,
            max_len=max(32, args.max_length),
            normalize=kv.get("normalize", "false").lower() == "true",
            context_window=args.context_window,
        ), prov

    if head_lc == "sapbert":
        hf_id = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    elif head_lc == "pubmedbert":
        hf_id = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    else:
        hf_id = spec
    prov["checkpoint"] = hf_id
    return DenseEncoderScorer(hf_id, device=args.device, batch=args.batch_size,
                              max_len=args.max_length, pooling=args.pooling), prov
