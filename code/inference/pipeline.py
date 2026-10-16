from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(CODE_DIR))

from common.utils import iter_jsonl, mention_context, write_jsonl

ONT_BUCKETS = (
    "parents",
    "children",
    "broader",
    "narrower",
    "siblings",
    "siblings_via_parent",
)

KIND_TO_TYPE = {
    "lexical_tfidf": "lexical_tfidf",
    "lexical_dice": "lexical_dice",
    "ontological": "ontology_related",
    "semtype": "same_semantic_type",
    "random": "random",
    "embedding_biobert": "embedding_biobert",
    "embedding_mention": "embedding_mention",
    "retrieved_sapbert": "retrieved_sapbert",
    "retrieved_arboel": "retrieved_arboel",
    "retrieved_arboel2017": "retrieved_arboel2017",
    "embedding_ctx": "embedding_ctx",
    "lexonto_both": "lex_and_onto",
    "lexonto_lexno": "lex_no_onto",
    "lexonto_ontono": "onto_no_lex",
    "retrieved_bioelx": "retrieved_bioelx",
    "retrieved_gebert": "retrieved_gebert",
    "retrieved_bergamot": "retrieved_bergamot",
    "retrieved_biosyn": "retrieved_biosyn",
    "retrieved_krissbert": "retrieved_krissbert",
    "retrieved_angel": "retrieved_angel",
    "retrieved_genre": "retrieved_genre",
}

BUCKETED_KINDS = frozenset({"ontological", "semtype"})

NEG_TYPE_ORDER = (
    "random",
    "lexical_tfidf",
    "lexical_dice",
    "embedding_biobert",
    "same_semantic_type",
    "ontology_related",
    "retrieved_sapbert",
    "retrieved_arboel",
    "retrieved_arboel2017",
"retrieved_bioelx", "retrieved_gebert", "retrieved_bergamot", "retrieved_biosyn", "retrieved_krissbert", "retrieved_angel", "retrieved_genre", "lex_no_onto", "onto_no_lex", "lex_and_onto", "embedding_ctx", )

DEFAULT_KINDS = (
    "random",
    "lexical_tfidf",
    "lexical_dice",
    "embedding_biobert",
    "semtype",
    "ontological",
)

_MENTION_ID_RE = re.compile(rb'"mention_id"\s*:\s*"([^"]+)"')


def negatives_path(dataset_dir: Path, kind: str) -> Path:
    return dataset_dir / f"negatives_{kind}.jsonl"


def discover_kinds(dataset_dir: Path, requested: list[str] | None = None) -> list[str]:
    """Return the negative kinds present in ``dataset_dir``, in table order.

    ``requested`` names kinds explicitly; a requested kind whose file is missing
    is an error (silently dropping it would make two runs incomparable while
    both look complete).
    """
    if requested:
        missing = [k for k in requested if not negatives_path(dataset_dir, k).exists()]
        if missing:
            raise SystemExit(
                f"ERROR: requested negative kinds not found in {dataset_dir}: {missing}"
            )
        unknown = [k for k in requested if k not in KIND_TO_TYPE]
        if unknown:
            raise SystemExit(f"ERROR: unknown negative kinds {unknown}; known: {sorted(KIND_TO_TYPE)}")
        return list(requested)
    return [k for k in DEFAULT_KINDS if negatives_path(dataset_dir, k).exists()]


def derive_model_slug(model_spec: str) -> str:
    """Filesystem-safe slug for a ``--model`` spec.

    ``biosyn:dmis-lab/biosyn-sapbert-ncbi-disease,alpha=0.5`` becomes
    ``biosyn-biosyn-sapbert-ncbi-disease-alpha0.5``. Long specs are truncated
    with a short hash so directory names stay readable but distinct.
    """
    import hashlib

    spec = model_spec.strip()
    slug = spec.replace("/", "-").replace(":", "-").replace(",", "-").replace("=", "")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if len(slug) > 80:
        digest = hashlib.sha1(spec.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[:70]}-{digest}"
    return slug or "model"


def run_dir(out_dir: Path, model_slug: str, dataset: str, run_tag: str) -> Path:
    return out_dir / model_slug / dataset / run_tag


CONTEXT_SOURCES = ("title_abstract", "abstract", "title", "sentence", "mention_only")


@dataclass
class NegativeStore:
    """Byte-offset index over ``negatives_<kind>.jsonl`` files."""

    dataset_dir: Path
    kinds: list[str]
    max_per_bucket: int = 0
    n_equivalent_filtered: int = field(default=0, init=False)
    _offsets: dict[str, dict[str, int]] = field(default_factory=dict, init=False)
    _handles: dict[str, object] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for kind in self.kinds:
            path = negatives_path(self.dataset_dir, kind)
            offsets: dict[str, int] = {}
            with path.open("rb") as f:
                pos = f.tell()
                for line in f:
                    m = _MENTION_ID_RE.search(line)
                    if m:
                        offsets[m.group(1).decode("utf-8")] = pos
                    pos += len(line)
            self._offsets[kind] = offsets
            self._handles[kind] = path.open("rb")

    def coverage(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._offsets.items()}

    def close(self) -> None:
        for f in self._handles.values():
            f.close()

    def record(self, kind: str, mention_id: str) -> dict | None:
        pos = self._offsets[kind].get(mention_id)
        if pos is None:
            return None
        f = self._handles[kind]
        f.seek(pos)
        rec = json.loads(f.readline().decode("utf-8"))
        if rec.get("mention_id") != mention_id:
            raise RuntimeError(
                f"negative index desync in {kind}: expected {mention_id}, got {rec.get('mention_id')}"
            )
        return rec

    def iter_negatives(self, mention_id: str, exclude: set[str]):
        """Yield ``(entity_id, name, negative_type, negative_subtype)``.

        ``exclude`` holds the gold-equivalent entity IDs. The negative builders
        already exclude them, but a stray equivalent would silently look like a
        model error, so it is filtered (and counted) here too.
        """
        for kind in self.kinds:
            rec = self.record(kind, mention_id)
            if rec is None:
                continue
            neg_type = KIND_TO_TYPE[kind]
            payload = rec.get("negatives") or {}
            if kind in BUCKETED_KINDS:
                blocks = payload.items()
            else:
                blocks = ((neg_type, payload),)
            for bucket, block in blocks:
                ids = block.get("entity_id_list") or block.get("cui_list") or []
                names = block.get("name_list") or []
                kept = 0
                for eid, name in zip(ids, names):
                    if eid in exclude:
                        self.n_equivalent_filtered += 1
                        continue
                    if not name:
                        continue
                    yield str(eid), name, neg_type, str(bucket)
                    kept += 1
                    if self.max_per_bucket and kept >= self.max_per_bucket:
                        break


def anchor_entity_ids(rec: dict) -> list[str]:
    """Gold anchor entity IDs: the concepts the mention is scored against."""
    for key in ("gold_entity_ids", "gold_cuis"):
        vals = rec.get(key)
        if vals:
            return [str(v) for v in vals]
    if rec.get("gold_cui"):
        return [str(rec["gold_cui"])]
    return []


def equivalent_entity_ids(rec: dict) -> set[str]:
    """Everything that must never count as a negative for this mention."""
    out = set(anchor_entity_ids(rec))
    for key in ("gold_equivalent_entity_ids", "gold_equivalent_cuis"):
        for v in rec.get(key) or []:
            out.add(str(v))
    return out
