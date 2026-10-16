from __future__ import annotations

import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.params import parse_random_negatives_args
from common.utils import collect_negative_ids, entity_space, gold_entity_ids, ids_payload, load_jsonl, preferred_name, write_jsonl


def index_negative_files(paths: list[Path]):
    out: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        if not path.exists():
            print(f"WARNING: exclusion file not found: {path}", file=sys.stderr)
            continue
        for rec in load_jsonl(path):
            out[rec["mention_id"]].update(collect_negative_ids(rec))
    return out


def main(argv=None) -> int:
    args = parse_random_negatives_args(argv)
    with args.concepts.open("rb") as f:
        concepts = pickle.load(f)
    mentions = load_jsonl(args.mentions)
    excluded_by_mid = index_negative_files(args.exclude_negatives)
    pool = [str(eid) for eid, rec in concepts.items() if rec.get("aliases")]
    tax_pool: dict[str, list[str]] = defaultdict(list)
    if args.match_taxon:
        for eid in pool:
            tax_id = str(concepts[eid].get("tax_id") or "")
            if tax_id:
                tax_pool[tax_id].append(eid)
    results = []
    for m in mentions:
        exclude = gold_entity_ids(m) | excluded_by_mid.get(m["mention_id"], set())
        candidate_pool = pool
        if args.match_taxon:
            tax_ids = [str(x) for x in m.get("gold_tax_ids", []) if x]
            if len(set(tax_ids)) == 1 and tax_ids[0] in tax_pool:
                candidate_pool = tax_pool[tax_ids[0]]
        rng = random.Random(f"{args.seed}|{m['mention_id']}")
        ids, seen = [], set()
        attempts = 0
        max_attempts = max(args.num * 200, 2000)
        while candidate_pool and len(ids) < args.num and attempts < max_attempts:
            eid = str(rng.choice(candidate_pool)); attempts += 1
            if eid in exclude or eid in seen:
                continue
            seen.add(eid); ids.append(eid)
        if len(ids) < args.num and len(candidate_pool) <= 100000:
            for eid in candidate_pool:
                eid = str(eid)
                if eid not in exclude and eid not in seen:
                    seen.add(eid); ids.append(eid)
                    if len(ids) >= args.num:
                        break
        space = entity_space(m, concepts)
        neg = ids_payload(ids, space)
        neg["name_list"] = [preferred_name(concepts[x]) for x in ids]
        results.append({"entity_space": space, "doc_id": m["doc_id"], "mention_id": m["mention_id"],
                        "mention": m["mention"], "gold_id": m.get("gold_id"),
                        "gold_entity_ids": m.get("gold_entity_ids", []), "negatives": neg})
    write_jsonl(args.output, results)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
