from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.params import parse_build_umls_concept_dict_args

MRCONSO = {"CUI": 0, "LAT": 1, "TS": 2, "STT": 4, "ISPREF": 6, "SAB": 11, "TTY": 12, "CODE": 13, "STR": 14}
MRSTY = {"CUI": 0, "TUI": 1, "STY": 3}
MRREL = {"CUI1": 0, "REL": 3, "CUI2": 4}
MRDEF = {"CUI": 0, "DEF": 5}
REL_BUCKETS = {"PAR": "parents", "CHD": "children", "RB": "broader", "RN": "narrower", "SIB": "siblings"}


def parse_mrconso(path: Path, source_sabs: set[str]):
    preferred: dict[str, str] = {}
    aliases: dict[str, list[str]] = defaultdict(list)
    seen_alias: dict[str, set[str]] = defaultdict(set)
    source_ids_by_cui: dict[str, set[str]] = defaultdict(set)
    source_to_cuis: dict[str, set[str]] = defaultdict(set)
    t0 = time.time()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            cols = line.rstrip("\n").split("|")
            if len(cols) <= MRCONSO["STR"]:
                continue
            cui = cols[MRCONSO["CUI"]]
            text = cols[MRCONSO["STR"]].strip()
            if text:
                k = text.casefold()
                if k not in seen_alias[cui]:
                    seen_alias[cui].add(k)
                    aliases[cui].append(text)
                if cui not in preferred and cols[MRCONSO["TS"]] == "P" and cols[MRCONSO["STT"]] == "PF" and cols[MRCONSO["ISPREF"]] == "Y":
                    preferred[cui] = text
            sab = cols[MRCONSO["SAB"]]
            code = cols[MRCONSO["CODE"]]
            if sab in source_sabs and code:
                key = f"{sab}:{code}"
                source_ids_by_cui[cui].add(key)
                source_to_cuis[key].add(cui)
            if n % 1_000_000 == 0:
                print(f"MRCONSO {n:,} lines ({time.time()-t0:.0f}s)", flush=True)
    for cui, vals in aliases.items():
        preferred.setdefault(cui, vals[0])
    return preferred, dict(aliases), source_ids_by_cui, source_to_cuis


def parse_mrsty(path: Path):
    tuis: dict[str, list[str]] = defaultdict(list)
    stys: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    tui_to_sty: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            cols = line.rstrip("\n").split("|")
            if len(cols) <= MRSTY["STY"]:
                continue
            cui, tui, sty = cols[0], cols[1], cols[3].strip()
            tui_to_sty.setdefault(tui, sty)
            if tui not in seen[cui]:
                seen[cui].add(tui)
                tuis[cui].append(tui)
                stys[cui].append(sty)
    return dict(tuis), dict(stys), tui_to_sty


def parse_mrrel(path: Path):
    bucket_names = sorted(set(REL_BUCKETS.values()))
    rel: dict[str, dict[str, set[str]]] = defaultdict(lambda: {b: set() for b in bucket_names})
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            cols = line.rstrip("\n").split("|")
            if len(cols) <= MRREL["CUI2"]:
                continue
            r = cols[MRREL["REL"]]
            if r not in REL_BUCKETS:
                continue
            c1, c2 = cols[MRREL["CUI1"]], cols[MRREL["CUI2"]]
            if c1 and c2 and c1 != c2:
                rel[c1][REL_BUCKETS[r]].add(c2)
            if n % 5_000_000 == 0:
                print(f"MRREL {n:,} lines", flush=True)
    return rel


def parse_mrdef(path: Path):
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            cols = line.rstrip("\n").split("|")
            if len(cols) <= MRDEF["DEF"]:
                continue
            cui, text = cols[MRDEF["CUI"]], cols[MRDEF["DEF"]].strip()
            if text and cui not in out:
                out[cui] = text
    return out


def derive_siblings_via_parent(rel, cap: int):
    for cui in list(rel.keys()):
        sibs: set[str] = set()
        for parent in rel[cui].get("parents", set()):
            for child in rel.get(parent, {}).get("children", set()):
                if child != cui:
                    sibs.add(child)
                    if cap > 0 and len(sibs) >= cap:
                        break
            if cap > 0 and len(sibs) >= cap:
                break
        rel[cui]["siblings_via_parent"] = sibs
    for cui in rel:
        rel[cui].setdefault("siblings_via_parent", set())


def main(argv=None) -> int:
    args = parse_build_umls_concept_dict_args(argv)
    meta = args.umls_meta
    required = {name: meta / f"{name}.RRF" for name in ["MRCONSO", "MRSTY", "MRREL"]}
    for p in required.values():
        if not p.exists():
            sys.exit(f"ERROR: missing {p}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    preferred, aliases, source_ids_by_cui, source_to_cuis = parse_mrconso(required["MRCONSO"], set(args.source_equivalence_sabs))
    tuis, stys, tui_to_sty = parse_mrsty(required["MRSTY"])
    rel = parse_mrrel(required["MRREL"])
    derive_siblings_via_parent(rel, args.siblings_via_parent_cap)
    defs = parse_mrdef(meta / "MRDEF.RRF")

    all_cuis = set(aliases) | set(tuis) | set(rel)
    concepts: dict[str, dict] = {}
    for cui in all_cuis:
        rr = rel.get(cui, {})
        concepts[cui] = {
            "entity_space": "umls",
            "entity_id": cui,
            "preferred_name": preferred.get(cui, ""),
            "aliases": aliases.get(cui, []),
            "tuis": tuis.get(cui, []),
            "semantic_types": stys.get(cui, []),
            "definition": defs.get(cui, ""),
            "source_ids": sorted(source_ids_by_cui.get(cui, set())),
            "parents": sorted(rr.get("parents", set())),
            "children": sorted(rr.get("children", set())),
            "broader": sorted(rr.get("broader", set())),
            "narrower": sorted(rr.get("narrower", set())),
            "siblings": sorted(rr.get("siblings", set())),
            "siblings_via_parent": sorted(rr.get("siblings_via_parent", set())),
        }

    tui_to_cuis: dict[str, list[str]] = defaultdict(list)
    for cui, rec in concepts.items():
        for tui in rec["tuis"]:
            tui_to_cuis[tui].append(cui)

    with (args.out_dir / "concepts.pkl").open("wb") as f:
        pickle.dump(concepts, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (args.out_dir / "source_to_cuis.pkl").open("wb") as f:
        pickle.dump({k: sorted(v) for k, v in source_to_cuis.items()}, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (args.out_dir / "tui_to_cuis.pkl").open("wb") as f:
        pickle.dump(dict(tui_to_cuis), f, protocol=pickle.HIGHEST_PROTOCOL)
    (args.out_dir / "tui_to_sty.json").write_text(json.dumps(tui_to_sty, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "concepts.meta.json").write_text(json.dumps({
        "entity_space": "umls",
        "umls_meta": str(meta),
        "n_entities": len(concepts),
        "source_equivalence_sabs": args.source_equivalence_sabs,
    }, indent=2), encoding="utf-8")
    sample = dict(list(concepts.items())[:20])
    (args.out_dir / "concepts_sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(concepts):,} UMLS concepts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
