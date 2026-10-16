from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.params import parse_lexical_negatives_args
from common.utils import entity_space, gold_entity_ids, ids_payload, load_jsonl, write_jsonl

INDEX_NAME = {"tfidf": "lexical_index.pkl", "dice": "lexical_dice_index.pkl"}


def build_alias_index(concepts: dict[str, dict], max_aliases: int):
    aliases: list[str] = []
    entity_ids: list[str] = []
    for eid, rec in concepts.items():
        vals = sorted(set(rec.get("aliases") or []), key=lambda s: (len(s), s.casefold()))[:max_aliases]
        for alias in vals:
            aliases.append(alias)
            entity_ids.append(str(eid))
    return aliases, np.asarray(entity_ids, dtype=object)


def settings(metric: str, max_aliases: int, concepts_path: Path) -> dict:
    """Cache-invalidation key. Must stay byte-stable or existing indexes rebuild."""
    stat = concepts_path.stat()
    common = {"max_aliases": max_aliases,
              "concepts_size": stat.st_size,
              "concepts_mtime_ns": stat.st_mtime_ns}
    if metric == "tfidf":
        return {"metric": "tfidf_cosine", "analyzer": "char_wb", "ngram_range": (3, 5),
                "min_df": 2, "max_df": 0.9, "sublinear_tf": True, **common}
    return {"metric": "dice", "ngram": (2, 2), **common}


def fit_vectorizer(metric: str, aliases: list[str]):
    """Returns (vec, X, row_len). row_len is None for tfidf."""
    if metric == "tfidf":
        kw = dict(analyzer="char_wb", ngram_range=(3, 5), lowercase=True,
                  norm="l2", sublinear_tf=True, dtype=np.float32)
        vec = TfidfVectorizer(min_df=2, max_df=0.9, **kw)
        try:
            X = vec.fit_transform(aliases).tocsr()
        except ValueError:
            vec = TfidfVectorizer(min_df=1, max_df=1.0, **kw)
            X = vec.fit_transform(aliases).tocsr()
        return vec, X, None

    vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 2), lowercase=True,
                          binary=True, dtype=np.float32)
    X = vec.fit_transform(aliases).tocsr()
    X.data[:] = 1.0
    row_len = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    return vec, X, row_len


def load_or_build(args):
    index_path = args.index or args.concepts.with_name(INDEX_NAME[args.metric])
    wanted = settings(args.metric, args.max_aliases_per_entity, args.concepts)
    if index_path.exists() and not args.rebuild_index:
        with index_path.open("rb") as f:
            cache = pickle.load(f)
        if cache.get("settings") == wanted:
            return (cache["aliases"], cache["entity_ids"], cache["vec"], cache["X"],
                    cache.get("row_len"))
    with args.concepts.open("rb") as f:
        concepts = pickle.load(f)
    aliases, entity_ids = build_alias_index(concepts, args.max_aliases_per_entity)
    del concepts
    vec, X, row_len = fit_vectorizer(args.metric, aliases)
    payload = {"settings": wanted, "aliases": aliases, "entity_ids": entity_ids,
               "vec": vec, "X": X}
    if row_len is not None:
        payload["row_len"] = row_len
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return aliases, entity_ids, vec, X, row_len


def build_queries(metric: str, vec, mentions: list[dict]):
    """Returns (Q, q_len). q_len is None for tfidf."""
    Q = vec.transform([m["mention"] for m in mentions])
    if metric == "tfidf":
        return Q, None
    Q = Q.tocsr()
    Q.data[:] = 1.0
    return Q, np.asarray(Q.sum(axis=1)).ravel().astype(np.float32)


def column_scores(metric: str, col, row_len, q_len, row: int):
    """Similarity of every candidate alias in this column, plus their row indices."""
    idxs = col.indices
    if metric == "tfidf":
        return idxs, col.data
    if not len(idxs):
        return idxs, np.asarray([], dtype=np.float32)
    inter = col.data.astype(np.float32, copy=False)
    denom = row_len[idxs] + q_len[row]
    vals = np.zeros(len(idxs), dtype=np.float32)
    nz = denom > 0
    vals[nz] = 2.0 * inter[nz] / denom[nz]
    return idxs, vals


def top_local(vals, retrieve_k: int):
    if not len(vals):
        return []
    k = min(retrieve_k, len(vals))
    if 0 < k < len(vals):
        local = np.argpartition(-vals, k - 1)[:k]
        return local[np.argsort(-vals[local])]
    return np.argsort(-vals)


def main(argv=None) -> int:
    args = parse_lexical_negatives_args(argv)
    aliases, entity_ids, vec, X, row_len = load_or_build(args)
    mentions = load_jsonl(args.mentions)
    Q, q_len = build_queries(args.metric, vec, mentions)

    results = []
    t0 = time.time()
    for i in range(0, len(mentions), args.batch_size):
        batch = Q[i:i + args.batch_size]
        sims = (X @ batch.T).tocsc()
        for j in range(sims.shape[1]):
            m = mentions[i + j]
            idxs, vals = column_scores(args.metric, sims[:, j], row_len, q_len, i + j)
            exclude = gold_entity_ids(m)
            ids, names, scores = [], [], []
            seen: set[str] = set()
            for loc in top_local(vals, args.retrieve_k):
                idx = int(idxs[loc])
                eid = str(entity_ids[idx])
                if eid in exclude or eid in seen:
                    continue
                seen.add(eid)
                ids.append(eid)
                names.append(aliases[idx])
                scores.append(float(vals[loc]))
                if len(ids) >= args.top_k:
                    break
            space = entity_space(m)
            neg = ids_payload(ids, space)
            neg.update({"name_list": names, "score_list": scores})
            results.append({
                "entity_space": space,
                "doc_id": m["doc_id"], "mention_id": m["mention_id"], "mention": m["mention"],
                "gold_id": m.get("gold_id"), "gold_entity_ids": m.get("gold_entity_ids", []),
                "negatives": neg,
            })
        print(f"{min(i + args.batch_size, len(mentions))}/{len(mentions)} ({time.time()-t0:.0f}s)", flush=True)
    write_jsonl(args.output, results)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
