"""Embedding negatives, retrieved by the mention surface form.

Both sides are encoded with the same BioBERT and L2 normalised, so the inner
product is the cosine:

  query side   the mention surface form, the same string the lexical builders
               query with. Unique surface forms are encoded once
  entity side  the entity name: its preferred name, and under
               --text-mode preferred_aliases up to --max-aliases aliases
               joined by the tokenizer's SEP token

Querying by the mention rather than by the gold concept matters twice over.
It keeps this bucket comparable with the lexical ones, which also ask "what
looks like what the author wrote", and it lets two mentions of the same gold
concept get different negatives -- a gold-queried bucket gives `WD` and
`Wilson disease` the identical list.

The entity index is built here on first use and cached under --index-dir, so
this is the only script that needs BioBERT.

    python code/preprocess/embedding_negatives.py \\
        --concepts outputs/kb/umls_2020AA/concepts.pkl \\
        --mentions outputs/ncbi_disease/mentions_test.jsonl \\
        --output outputs/ncbi_disease/negatives_embedding_mention.jsonl
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.params import parse_embedding_negatives_args
from common.utils import (
    entity_space,
    gold_entity_ids,
    ids_payload,
    load_jsonl,
    preferred_name,
    write_jsonl,
)


def entity_text(rec: dict, tokenizer, mode: str, max_aliases: int) -> str:
    preferred = preferred_name(rec).strip()
    if mode == "preferred" or max_aliases <= 0:
        return preferred
    aliases = []
    seen = {preferred.casefold()} if preferred else set()
    for alias in sorted(set(rec.get("aliases") or []), key=lambda x: (len(x), x.casefold())):
        alias = str(alias).strip()
        if alias and alias.casefold() not in seen:
            seen.add(alias.casefold())
            aliases.append(alias)
            if len(aliases) >= max_aliases:
                break
    return f" {tokenizer.sep_token} ".join([preferred, *aliases])


def encode(texts, tokenizer, model, torch, device, max_length: int, pooling: str) -> np.ndarray:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc).last_hidden_state.float()
    if pooling == "mean":
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    else:
        vec = out[:, 0, :]
    vec = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec.cpu().numpy().astype(np.float32)


def index_cache_key(args) -> dict:
    """Index invalidation key. Field names match the metadata.json written by the
    earlier 03_embedding_negatives.py, so an index it built is still reused."""
    stat = args.concepts.stat()
    return {"model": args.model, "revision": args.revision, "pooling": args.pooling,
            "text_mode": args.text_mode, "max_aliases": args.max_aliases,
            "max_length": args.index_max_length, "storage_dtype": args.storage_dtype,
            "concepts_size": stat.st_size, "concepts_mtime_ns": stat.st_mtime_ns,
            "normalized": True}


def build_or_load_index(args, tokenizer, model, torch, device):
    index_dir = args.index_dir or args.concepts.with_name("embedding_index_biobert")
    meta_path = index_dir / "metadata.json"
    ids_path = index_dir / "entity_ids.txt"
    emb_path = index_dir / "embeddings.npy"
    wanted = index_cache_key(args)

    if not args.rebuild_index and meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("settings") == wanted and ids_path.exists() and emb_path.exists():
            entity_ids = ids_path.read_text(encoding="utf-8").splitlines()
            E = np.load(emb_path, mmap_mode="r")
            if E.shape[0] == len(entity_ids):
                print(f"[index] reuse {index_dir} ({len(entity_ids):,} entities "
                      f"x {E.shape[1]})", flush=True)
                return entity_ids, E

    index_dir.mkdir(parents=True, exist_ok=True)
    print(f"[index] build from {args.concepts}", flush=True)
    with args.concepts.open("rb") as f:
        concepts = pickle.load(f)
    records = [(str(eid), rec) for eid, rec in concepts.items() if preferred_name(rec).strip()]
    del concepts
    entity_ids = [eid for eid, _ in records]
    hidden = int(model.config.hidden_size)
    dtype = np.float16 if args.storage_dtype == "float16" else np.float32
    tmp_path = index_dir / "embeddings.incomplete.npy"
    E = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=dtype,
                                  shape=(len(records), hidden))
    t0 = time.time()
    for start in range(0, len(records), args.encode_batch_size):
        chunk = records[start:start + args.encode_batch_size]
        texts = [entity_text(rec, tokenizer, args.text_mode, args.max_aliases)
                 for _, rec in chunk]
        vecs = encode(texts, tokenizer, model, torch, device,
                      args.index_max_length, args.pooling)
        E[start:start + len(vecs)] = vecs.astype(dtype, copy=False)
        if start == 0 or (start // args.encode_batch_size) % 100 == 0:
            print(f"[index] {start + len(vecs):,}/{len(records):,} entities "
                  f"({time.time() - t0:.0f}s)", flush=True)
    E.flush()
    del E, records
    tmp_path.replace(emb_path)
    ids_path.write_text("\n".join(entity_ids) + "\n", encoding="utf-8")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"settings": wanted, "n_entities": len(entity_ids),
                   "hidden_size": hidden}, f, indent=2)
    print(f"[index] wrote {index_dir} ({len(entity_ids):,} entities)", flush=True)
    return entity_ids, np.load(emb_path, mmap_mode="r")


def search(Q, E, torch, device, retrieve_k: int, chunk: int):
    """Exact inner-product top-k over the whole index, chunked over entities.

    The matmul is float32 even when the index is stored as float16, so the
    scores match an exact float32 search.
    """
    n_q = Q.shape[0]
    best_s = torch.full((n_q, retrieve_k), -2.0, device=device, dtype=torch.float32)
    best_i = torch.full((n_q, retrieve_k), -1, device=device, dtype=torch.int64)
    Qt = torch.from_numpy(Q).to(device)
    t0 = time.time()
    for c0 in range(0, E.shape[0], chunk):
        C = torch.from_numpy(np.ascontiguousarray(E[c0:c0 + chunk])).to(device).float()
        for q0 in range(0, n_q, 4096):
            s = Qt[q0:q0 + 4096] @ C.T
            k = min(retrieve_k, s.shape[1])
            v, ix = s.topk(k, dim=1)
            cs = torch.cat([best_s[q0:q0 + 4096], v], 1)
            ci = torch.cat([best_i[q0:q0 + 4096], ix + c0], 1)
            v2, ix2 = cs.topk(retrieve_k, dim=1)
            best_s[q0:q0 + 4096] = v2
            best_i[q0:q0 + 4096] = torch.gather(ci, 1, ix2)
        del C
        print(f"[search] {min(c0 + chunk, E.shape[0]):,}/{E.shape[0]:,} entities "
              f"({time.time() - t0:.0f}s)", flush=True)
    return best_s.cpu().numpy(), best_i.cpu().numpy()


def main(argv=None) -> int:
    args = parse_embedding_negatives_args(argv, __doc__)
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device)
    print(f"[model] {args.model}@{args.revision} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(args.model, revision=args.revision).to(device).eval()

    entity_ids, E = build_or_load_index(args, tokenizer, model, torch, device)

    mentions = load_jsonl(args.mentions)
    uniq = sorted({(m.get("mention") or "").strip() for m in mentions
                   if (m.get("mention") or "").strip()})
    print(f"[queries] {len(mentions):,} mentions -> {len(uniq):,} unique surface forms",
          flush=True)
    t0 = time.time()
    Q = np.empty((len(uniq), E.shape[1]), dtype=np.float32)
    for i in range(0, len(uniq), args.query_batch_size):
        chunk = uniq[i:i + args.query_batch_size]
        Q[i:i + len(chunk)] = encode(chunk, tokenizer, model, torch, device,
                                     args.max_length, args.pooling)
    print(f"[queries] encoded in {time.time() - t0:.0f}s", flush=True)
    del model

    retrieve_k = min(max(args.retrieve_k, args.top_k + 10), len(entity_ids))
    best_s, best_i = search(Q, E, torch, device, retrieve_k, args.search_chunk)
    hits = {surface: [(entity_ids[int(r)], float(s))
                      for r, s in zip(best_i[row], best_s[row]) if r >= 0]
            for row, surface in enumerate(uniq)}

    with args.concepts.open("rb") as f:
        concepts = pickle.load(f)

    results = []
    n_short = 0
    for m in mentions:
        surface = (m.get("mention") or "").strip()
        exclude = set(gold_entity_ids(m)) | set(m.get("gold_equivalent_entity_ids") or [])
        ids, names, scores = [], [], []
        for eid, score in hits.get(surface, []):
            if eid in exclude or eid in ids:
                continue
            rec = concepts.get(eid)
            if rec is None:
                continue
            name = preferred_name(rec)
            if not name:
                continue
            ids.append(eid)
            names.append(name)
            scores.append(score)
            if len(ids) >= args.top_k:
                break
        if len(ids) < args.top_k:
            n_short += 1
        space = entity_space(m)
        neg = ids_payload(ids, space)
        neg.update({"name_list": names, "score_list": scores})
        results.append({
            "entity_space": space,
            "doc_id": m["doc_id"], "mention_id": m["mention_id"], "mention": m["mention"],
            "gold_id": m.get("gold_id"),
            "gold_entity_ids": m.get("gold_entity_ids", []),
            "negatives": neg,
        })

    write_jsonl(args.output, results)
    print(f"wrote {args.output} ({len(results):,} mentions)")
    print(f"  queried by mention surface form, {len(uniq):,} unique queries")
    if n_short:
        print(f"  {n_short:,} mentions got fewer than {args.top_k} negatives "
              f"(retrieve_k={retrieve_k} exhausted after filtering)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
