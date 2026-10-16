from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.params import parse_ontological_negatives_args
from common.utils import cap_ids, ids_payload, load_jsonl, preferred_name, write_jsonl

RELATIONS = ("parents", "children", "broader", "narrower", "siblings", "siblings_via_parent")


def main(argv=None) -> int:
    args = parse_ontological_negatives_args(argv)
    with args.concepts.open("rb") as f:
        concepts = pickle.load(f)
    mentions = load_jsonl(args.mentions)
    results = []
    for m in mentions:
        anchors = set(m.get("gold_cuis") or m.get("gold_entity_ids") or [])
        exclude = set(m.get("gold_equivalent_cuis") or m.get("gold_equivalent_entity_ids") or anchors)
        buckets = {}
        for rel in RELATIONS:
            provenance: dict[str, set[str]] = defaultdict(set)
            for anchor in anchors:
                rec = concepts.get(anchor)
                if not rec:
                    continue
                for candidate in rec.get(rel, []):
                    if candidate in concepts and candidate not in exclude:
                        provenance[candidate].add(anchor)
            ids = cap_ids(list(provenance), args.cap, f"{args.seed}|{m['mention_id']}|{rel}")
            payload = ids_payload(ids, "umls")
            payload["name_list"] = [preferred_name(concepts[x]) for x in ids]
            payload["via_anchor_list"] = [sorted(provenance[x]) for x in ids]
            buckets[rel] = payload
        results.append({
            "entity_space": "umls", "doc_id": m["doc_id"], "mention_id": m["mention_id"],
            "mention": m["mention"], "gold_id": m.get("gold_id"), "gold_cuis": sorted(anchors),
            "gold_equivalent_cuis": sorted(exclude), "negatives": buckets,
        })
    write_jsonl(args.output, results)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
