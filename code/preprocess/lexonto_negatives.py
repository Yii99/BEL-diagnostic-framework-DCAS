from __future__ import annotations
import json, pickle, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.params import parse_lexonto_negatives_args
from common.utils import load_jsonl

RELATIONS = ("parents", "children", "broader", "narrower", "siblings", "siblings_via_parent")


def trigrams(s: str) -> frozenset:
    out = set()
    for w in s.lower().split():
        w = f" {w} "
        for i in range(len(w) - 2):
            out.add(w[i:i + 3])
    return frozenset(out)

def main(argv=None) -> int:
    args = parse_lexonto_negatives_args(argv, __doc__); t0 = time.time()
    print(f"[load] {args.concepts}", flush=True)
    with args.concepts.open("rb") as f:
        concepts = pickle.load(f)
    print(f"[load] {len(concepts):,} concepts ({time.time()-t0:.0f}s)", flush=True)
    tri_cache: dict[str, frozenset] = {}

    def max_dice(m_tri, cid):
        rec = concepts.get(cid)
        if not rec:
            return -1.0, ""
        names = [rec.get("preferred_name") or ""] + list(rec.get("aliases") or [])
        best, best_n = -1.0, ""
        for name in [n for n in names if n][: args.max_aliases]:
            tri = tri_cache.get(name)
            if tri is None:
                tri = trigrams(name); tri_cache[name] = tri
            denom = len(m_tri) + len(tri)
            d = 2 * len(m_tri & tri) / denom if denom else 0.0
            if d > best:
                best, best_n = d, name
        return best, best_n

    for ds in args.datasets:
        ddir = args.outputs_root / ds
        mentions = list(load_jsonl(ddir / "mentions_test.jsonl"))
        lexpool: dict[str, set] = {}
        for kind in ("lexical_tfidf", "lexical_dice"):
            for r in load_jsonl(ddir / f"negatives_{kind}.jsonl"):
                ids = r["negatives"].get("cui_list") or r["negatives"].get("entity_id_list") or []
                lexpool.setdefault(r["mention_id"], set()).update(str(x) for x in ids)
        outs = {k: (ddir / f"negatives_lexonto_{k}.jsonl").open("w") for k in ("both", "lexno", "ontono")}
        kept = 0
        rng = random.Random(args.seed)
        for i, m in enumerate(mentions):
            anchors = set(m.get("gold_cuis") or m.get("gold_entity_ids") or [])
            exclude = set(m.get("gold_equivalent_cuis") or m.get("gold_equivalent_entity_ids") or anchors)
            onto = set()
            for a in anchors:
                rec = concepts.get(a)
                if not rec:
                    continue
                for rel in RELATIONS:
                    onto.update(c for c in rec.get(rel) or [] if c in concepts and c not in exclude)
            if not onto:
                continue
            m_tri = trigrams(m.get("mention") or "")
            if not m_tri:
                continue
            scored = [(cid, *max_dice(m_tri, cid)) for cid in onto]
            both = sorted([x for x in scored if x[1] >= args.hi], key=lambda x: -x[1])[: args.k]
            if not both:
                continue
            ontono_pool = [x for x in scored if 0.0 <= x[1] < args.lo]
            if not ontono_pool:
                continue
            pool = [c for c in lexpool.get(m["mention_id"], set()) if c not in onto and c not in exclude]
            lex_scored = sorted(((cid, *max_dice(m_tri, cid)) for cid in pool), key=lambda x: -x[1])
            lex_scored = [x for x in lex_scored if x[1] >= 0]
            lexno, used = [], set()
            for _, d_b, _n in both:
                cand = min((x for x in lex_scored if x[0] not in used),
                           key=lambda x: abs(x[1] - d_b), default=None)
                if cand is None:
                    break
                lexno.append(cand); used.add(cand[0])
            k_m = min(len(both), len(lexno), len(ontono_pool), args.k)
            if k_m == 0:
                continue
            both, lexno = both[:k_m], lexno[:k_m]
            ontono = rng.sample(ontono_pool, k_m)
            base = {"entity_space": "umls", "doc_id": m["doc_id"], "mention_id": m["mention_id"],
                    "mention": m.get("mention") or "", "gold_id": (sorted(anchors) or [""])[0],
                    "gold_entity_ids": sorted(anchors), "gold_equivalent_cuis": sorted(exclude)}
            for key, rows in (("both", both), ("lexno", lexno), ("ontono", ontono)):
                names = [(concepts[c].get("preferred_name") or n or "?") if key == "ontono" else n
                         for c, d, n in rows]
                payload = {"entity_id_list": [c for c, d, n in rows], "cui_list": [c for c, d, n in rows],
                           "name_list": names, "score_list": [round(d, 4) for c, d, n in rows]}
                outs[key].write(json.dumps({**base, "negatives": payload}, ensure_ascii=False) + "\n")
            kept += 1
            if (i + 1) % 2000 == 0:
                print(f"  {ds}: {i+1:,}/{len(mentions):,} mentions, kept {kept:,} ({time.time()-t0:.0f}s)", flush=True)
        for f in outs.values():
            f.close()
        print(f"[done] {ds}: kept {kept:,}/{len(mentions):,} mentions with all three buckets", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
