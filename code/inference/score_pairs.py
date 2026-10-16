"""Score every (mention, gold, hard-negative) triple with one baseline.

This is the foundation of the inference stage: every table and figure downstream
aggregates the parquet written here.

Per mention, with gold concept g and each negative concept n::

    gold_score     = max over gold aliases of scorer(mention, alias)    (--gold-name-mode)
    negative_score = scorer(mention, name(n))                           (--neg-name-mode)
    margin         = gold_score - negative_score
    success        = gold_score >  negative_score
    confused       = negative_score >= gold_score

Ties count as confusion: a linker that cannot separate the gold concept from a
parent concept has not distinguished them, and reporting ties as successes would
flatter every backend that emits coarse scores.

Inputs
    <dataset-dir>/mentions_test.jsonl              from 02
    <dataset-dir>/negatives_<kind>.jsonl           from 03-06
    <kb>/concepts.pkl                              from 01

Outputs
    <out-dir>/<model_slug>/<dataset>/<run-tag>/pair_scores.parquet
    <out-dir>/<model_slug>/<dataset>/<run-tag>/run_meta.json
    <out-dir>/<model_slug>/<dataset>/<run-tag>/excluded_mentions.jsonl

Example
    python code/inference/score_pairs.py \
        --dataset ncbi_disease \
        --dataset-dir outputs/ncbi_disease \
        --concepts outputs/kb/umls_2020AA/concepts.pkl \
        --out-dir outputs/scores \
        --model sapbert --device cuda
"""

from __future__ import annotations

import gc
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.params import parse_score_pairs_args
from common.utils import alias_list
from inference import pipeline as _io
from inference.scorers import (
    CandidateBatch,
    MentionQuery,
    groupwise_argmax,
    make_scorer,
)

PARQUET_SCHEMA_FIELDS = (
    "mention_id", "mention_text", "gold_entity_id", "n_gold_anchors",
    "negative_entity_id", "negative_type", "negative_subtype", "negative_name",
    "gold_score", "negative_score", "margin", "success", "confused", "gold_best_alias",
)


def build_slim_cache(dataset_dir: Path, mentions_path: Path, concepts_path: Path,
                     cache_path: Path) -> dict:
    kinds = _io.discover_kinds(dataset_dir)
    print(f"[slim] collecting referenced entities from {mentions_path.name} "
          f"and {len(kinds)} negative files ...", flush=True)
    needed: set[str] = set()
    for rec in _io.iter_jsonl(mentions_path):
        needed |= _io.equivalent_entity_ids(rec)
    store = _io.NegativeStore(dataset_dir, kinds)
    try:
        for rec in _io.iter_jsonl(mentions_path):
            mid = rec["mention_id"]
            for kind in kinds:
                neg = store.record(kind, mid)
                if neg is None:
                    continue
                payload = neg.get("negatives") or {}
                blocks = payload.values() if kind in _io.BUCKETED_KINDS else (payload,)
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    ids = block.get("entity_id_list") or block.get("cui_list") or []
                    needed.update(str(x) for x in ids)
    finally:
        store.close()
    print(f"[slim] {len(needed):,} distinct entities referenced; loading {concepts_path} ...",
          flush=True)
    t0 = time.time()
    with concepts_path.open("rb") as f:
        concepts = pickle.load(f)
    print(f"[slim] loaded {len(concepts):,} concepts in {time.time() - t0:.0f}s", flush=True)
    slim = {}
    for eid in needed:
        rec = concepts.get(eid)
        if rec is None:
            continue
        slim[eid] = {
            "preferred_name": rec.get("preferred_name") or "",
            "aliases": list(rec.get("aliases") or []),
            "definition": rec.get("definition") or "",
        }
    del concepts
    gc.collect()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(slim, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_path)
    print(f"[slim] wrote {cache_path} ({len(slim):,} entities, "
          f"{cache_path.stat().st_size / 1e6:.0f} MB)", flush=True)
    return slim


def load_concepts(args) -> dict:
    cache_path = args.dataset_dir / "concepts_slim.pkl"
    if cache_path.exists():
        t0 = time.time()
        with cache_path.open("rb") as f:
            slim = pickle.load(f)
        print(f"[slim] reused {cache_path} ({len(slim):,} entities, "
              f"{time.time() - t0:.0f}s)", flush=True)
        return slim
    return build_slim_cache(args.dataset_dir, args.mentions, args.concepts, cache_path)


def build_plan(args, concepts: dict, needs_context: bool):
    """Return (per_mention, strings, entity_ids, excluded, stats)."""
    strings: list[str] = []
    str_idx: dict[str, int] = {}

    def add(s: str) -> int:
        i = str_idx.get(s)
        if i is None:
            i = len(strings)
            str_idx[s] = i
            strings.append(s)
        return i

    entity_ids: list[str] = []
    entity_pos: dict[str, int] = {}

    def add_entity(e: str) -> int:
        i = entity_pos.get(e)
        if i is None:
            i = len(entity_ids)
            entity_pos[e] = i
            entity_ids.append(e)
        return i

    type_ids: dict[str, int] = {}
    subtype_ids: dict[str, int] = {}

    mentions = list(_io.iter_jsonl(args.mentions))
    if args.mention_subset:
        wanted = {ln.strip() for ln in args.mention_subset.read_text().splitlines() if ln.strip()}
        mentions = [m for m in mentions if m["mention_id"] in wanted]
        if not mentions:
            raise SystemExit(f"ERROR: no mention in {args.mentions} matched {args.mention_subset}")
        print(f"mention subset : {args.mention_subset} -> {len(mentions):,} of "
              f"{len(wanted):,} listed ids", flush=True)
    elif args.sample_mentions and args.sample_mentions < len(mentions):
        random.Random(args.seed).shuffle(mentions)
        mentions = mentions[: args.sample_mentions]
        mentions.sort(key=lambda r: r["mention_id"])
    if args.limit:
        mentions = mentions[: args.limit]

    kinds = _io.discover_kinds(args.dataset_dir, args.neg_kinds)
    print(f"negative kinds : {kinds}", flush=True)
    store = _io.NegativeStore(args.dataset_dir, kinds, max_per_bucket=args.max_per_bucket)

    per_mention: list[dict] = []
    excluded: list[dict] = []
    stats = {
        "context_sources": {},
        "n_equivalent_negatives_filtered": 0,
        "n_negatives_missing_concept": 0,
        "n_mentions_without_negatives": 0,
    }
    t0 = time.time()
    try:
        for k, rec in enumerate(mentions):
            mid = rec["mention_id"]
            anchors = _io.anchor_entity_ids(rec)
            exclude = _io.equivalent_entity_ids(rec)

            gold_ids: list[str] = []
            gold_alias_idx: list[int] = []
            gold_ptr: list[int] = [0]
            for eid in anchors:
                crec = concepts.get(eid)
                if crec is None:
                    continue
                names = (alias_list(crec, args.max_gold_aliases)
                         if args.gold_name_mode == "max_alias"
                         else alias_list(crec, 1))
                if not names:
                    continue
                gold_ids.append(eid)
                gold_alias_idx.extend(add(n) for n in names)
                gold_ptr.append(len(gold_alias_idx))
                add_entity(eid)
            if not gold_ids:
                excluded.append({
                    "mention_id": mid, "mention_text": rec.get("mention"),
                    "gold_entity_ids": anchors,
                    "reason": ("gold_not_in_concepts"
                               if not any(a in concepts for a in anchors)
                               else "gold_has_no_aliases"),
                })
                continue

            neg_e: list[int] = []
            neg_alias_idx: list[int] = []
            neg_ptr: list[int] = [0]
            neg_type: list[int] = []
            neg_subtype: list[int] = []
            for eid, name, ntype, subtype in store.iter_negatives(mid, exclude):
                if args.neg_name_mode == "max_alias":
                    crec = concepts.get(eid)
                    if crec is None:
                        stats["n_negatives_missing_concept"] += 1
                        names = [name]
                    else:
                        names = alias_list(crec, args.max_neg_aliases) or [name]
                else:
                    names = [name]
                neg_e.append(add_entity(eid))
                neg_alias_idx.extend(add(n) for n in names)
                neg_ptr.append(len(neg_alias_idx))
                neg_type.append(type_ids.setdefault(ntype, len(type_ids)))
                neg_subtype.append(subtype_ids.setdefault(subtype, len(subtype_ids)))
            if not neg_e:
                stats["n_mentions_without_negatives"] += 1
                continue

            entry = {
                "mention_id": mid,
                "mention_text": rec.get("mention") or "",
                "mention_idx": add(rec.get("mention") or ""),
                "gold_ids": gold_ids,
                "gold_alias_idx": np.asarray(gold_alias_idx, dtype=np.int64),
                "gold_ptr": np.asarray(gold_ptr, dtype=np.int64),
                "neg_e": np.asarray(neg_e, dtype=np.int64),
                "neg_alias_idx": np.asarray(neg_alias_idx, dtype=np.int64),
                "neg_ptr": np.asarray(neg_ptr, dtype=np.int64),
                "neg_type": np.asarray(neg_type, dtype=np.int16),
                "neg_subtype": np.asarray(neg_subtype, dtype=np.int16),
            }
            if needs_context:
                left, right, source = _io.mention_context(rec)
                if args.context_window:
                    left = left[-args.context_window :]
                    right = right[: args.context_window]
                entry["ctx"] = (left, right)
                stats["context_sources"][source] = stats["context_sources"].get(source, 0) + 1
            per_mention.append(entry)
            if (k + 1) % 5000 == 0:
                print(f"  planned {k + 1:,}/{len(mentions):,} mentions "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        stats["n_equivalent_negatives_filtered"] = store.n_equivalent_filtered
        store.close()

    stats["kinds"] = kinds
    stats["n_input_mentions"] = len(mentions)
    stats["type_names"] = {v: k for k, v in type_ids.items()}
    stats["subtype_names"] = {v: k for k, v in subtype_ids.items()}
    return per_mention, strings, entity_ids, excluded, stats


def main(argv=None) -> int:
    args = parse_score_pairs_args(argv)
    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.slim_cache_only:
        load_concepts(args)
        return 0

    model_slug = args.model_slug or _io.derive_model_slug(args.model)
    out_dir = _io.run_dir(args.out_dir, model_slug, args.dataset, args.run_tag)
    pair_path = out_dir / "pair_scores.parquet"
    if pair_path.exists() and not args.overwrite:
        print(f"[SKIP] {pair_path} exists (use --overwrite to rescore)")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset     : {args.dataset}  ({args.dataset_dir})")
    print(f"model       : {args.model}  -> slug {model_slug}")
    print(f"device      : {args.device}")
    print(f"names       : gold={args.gold_name_mode} neg={args.neg_name_mode}")
    print(f"output      : {pair_path}")

    scorer, provenance = make_scorer(args)
    print(f"scorer      : {type(scorer).__name__} ({scorer.name})", flush=True)

    concepts = load_concepts(args)
    t_plan = time.time()
    per_mention, strings, entity_ids, excluded, stats = build_plan(
        args, concepts, needs_context=scorer.needs_context
    )
    n_pairs = int(sum(len(m["neg_e"]) for m in per_mention))
    print(f"planned     : {len(per_mention):,} mentions, {n_pairs:,} pairs, "
          f"{len(strings):,} unique strings, {len(entity_ids):,} entities "
          f"({time.time() - t_plan:.0f}s)", flush=True)
    if excluded:
        _io.write_jsonl(out_dir / "excluded_mentions.jsonl", excluded)
        print(f"excluded    : {len(excluded):,} mentions whose gold concept has no usable "
              f"name in the KB (logged to excluded_mentions.jsonl)")
        for e in excluded[:3]:
            print(f"   {e['mention_id']}  '{e['mention_text']}'  {e['gold_entity_ids']}  "
                  f"({e['reason']})")
    if not per_mention:
        raise SystemExit("ERROR: nothing to score")

    t_fit = time.time()
    scorer.fit(strings)
    if scorer.needs_entities:
        scorer.fit_entities(entity_ids, concepts)
    fit_seconds = time.time() - t_fit

    schema = pa.schema([
        ("mention_id", pa.string()),
        ("mention_text", pa.string()),
        ("gold_entity_id", pa.string()),
        ("n_gold_anchors", pa.int16()),
        ("negative_entity_id", pa.string()),
        ("negative_type", pa.string()),
        ("negative_subtype", pa.string()),
        ("negative_name", pa.string()),
        ("gold_score", pa.float32()),
        ("negative_score", pa.float32()),
        ("margin", pa.float32()),
        ("success", pa.bool_()),
        ("confused", pa.bool_()),
        ("gold_best_alias", pa.string()),
    ])
    type_names = stats["type_names"]
    subtype_names = stats["subtype_names"]

    writer = pq.ParquetWriter(pair_path, schema, compression="zstd")
    buf: dict[str, list] = {name: [] for name in PARQUET_SCHEMA_FIELDS}
    n_rows = 0
    n_confused = 0
    t0 = time.time()
    try:
        for i, m in enumerate(per_mention):
            left, right = m.get("ctx", ("", ""))
            query = MentionQuery(
                mention_id=m["mention_id"], text=m["mention_text"],
                text_idx=m["mention_idx"], context_left=left, context_right=right,
            )
            gold_batch = CandidateBatch(m["gold_ids"], m["gold_alias_idx"], m["gold_ptr"])
            gold_scores = scorer.score(query, gold_batch)
            gold_flat = scorer.score_flat(query, gold_batch)
            best = int(np.argmax(gold_scores))
            gold_score = float(gold_scores[best])
            if gold_flat is not None and len(gold_flat) == len(m["gold_alias_idx"]):
                alias_pick = groupwise_argmax(gold_flat, m["gold_ptr"])[best]
                gold_best_alias = strings[int(m["gold_alias_idx"][alias_pick])]
            else:
                gold_best_alias = strings[int(m["gold_alias_idx"][m["gold_ptr"][best]])]

            neg_entities = [entity_ids[j] for j in m["neg_e"]]
            neg_batch = CandidateBatch(neg_entities, m["neg_alias_idx"], m["neg_ptr"])
            neg_scores = scorer.score(query, neg_batch)
            neg_flat = scorer.score_flat(query, neg_batch)
            if neg_flat is not None and len(neg_flat) == len(m["neg_alias_idx"]):
                picks = groupwise_argmax(neg_flat, m["neg_ptr"])
                neg_names = [strings[int(m["neg_alias_idx"][p])] for p in picks]
            else:
                neg_names = [strings[int(m["neg_alias_idx"][p])] for p in m["neg_ptr"][:-1]]

            k = len(neg_entities)
            margins = gold_score - neg_scores
            success = margins > 0
            n_confused += int((~success).sum())
            buf["mention_id"].extend([m["mention_id"]] * k)
            buf["mention_text"].extend([m["mention_text"]] * k)
            buf["gold_entity_id"].extend([m["gold_ids"][best]] * k)
            buf["n_gold_anchors"].extend([len(m["gold_ids"])] * k)
            buf["negative_entity_id"].extend(neg_entities)
            buf["negative_type"].extend(type_names[int(t)] for t in m["neg_type"])
            buf["negative_subtype"].extend(subtype_names[int(t)] for t in m["neg_subtype"])
            buf["negative_name"].extend(neg_names)
            buf["gold_score"].extend([gold_score] * k)
            buf["negative_score"].extend(float(x) for x in neg_scores)
            buf["margin"].extend(float(x) for x in margins)
            buf["success"].extend(bool(x) for x in success)
            buf["confused"].extend(bool(not x) for x in success)
            buf["gold_best_alias"].extend([gold_best_alias] * k)
            n_rows += k

            if (i + 1) % args.flush_mentions == 0:
                writer.write_table(pa.table(buf, schema=schema))
                buf = {name: [] for name in PARQUET_SCHEMA_FIELDS}
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                eta = (len(per_mention) - i - 1) / max(rate, 1e-9)
                print(f"  scored {i + 1:,}/{len(per_mention):,} mentions, {n_rows:,} pairs "
                      f"({time.time() - t0:.0f}s, {rate:.1f} m/s, ETA {eta / 60:.1f} min)",
                      flush=True)
        if buf["mention_id"]:
            writer.write_table(pa.table(buf, schema=schema))
    finally:
        writer.close()

    score_seconds = time.time() - t0
    meta = {
        "dataset": args.dataset,
        "model_spec": args.model,
        "model_slug": model_slug,
        "scorer_class": type(scorer).__name__,
        "scorer_name": scorer.name,
        "provenance": provenance,
        "run_tag": args.run_tag,
        "device": args.device,
        "negative_kinds": stats["kinds"],
        "negative_types": sorted(type_names.values()),
        "n_input_mentions": stats["n_input_mentions"],
        "n_scored_mentions": len(per_mention),
        "n_excluded_mentions": len(excluded),
        "n_mentions_without_negatives": stats["n_mentions_without_negatives"],
        "n_equivalent_negatives_filtered": stats["n_equivalent_negatives_filtered"],
        "n_negatives_missing_concept": stats["n_negatives_missing_concept"],
        "n_pairs": n_rows,
        "n_confused_pairs": n_confused,
        "pairwise_success_rate": (n_rows - n_confused) / n_rows if n_rows else None,
        "unique_strings": len(strings),
        "unique_entities": len(entity_ids),
        "context_sources": stats["context_sources"],
        "fit_seconds": round(fit_seconds, 1),
        "score_seconds": round(score_seconds, 1),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {pair_path} ({n_rows:,} rows)")
    print(f"overall pairwise success rate: {meta['pairwise_success_rate']:.4f}"
          if n_rows else "no pairs scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
