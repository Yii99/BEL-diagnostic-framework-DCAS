from __future__ import annotations

import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.params import parse_benchmark_args
from common.utils import (
    MESH_UI_RE,
    MULTI_ID_SPLIT_RE,
    UMLS_CUI_RE,
    write_jsonl,
)


def load_pmids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"[.!?]\s+|\n", text):
        end = m.end()
        seg = text[start:end].strip()
        if seg:
            spans.append((start, end, seg))
        start = end
    if start < len(text):
        seg = text[start:].strip()
        if seg:
            spans.append((start, len(text), seg))
    return spans


def find_sentence(spans: list[tuple[int, int, str]], offset: int) -> str:
    for s, e, txt in spans:
        if s <= offset < e:
            return txt
    return ""


def split_source_ids(raw: str) -> list[str]:
    if not raw or raw.strip() in {"", "-1"}:
        return []
    return [x.strip() for x in MULTI_ID_SPLIT_RE.split(raw) if x.strip() and x.strip() != "-1"]


def source_key(raw: str, dataset: str) -> str:
    s = raw.strip()
    if ":" in s:
        p, code = s.split(":", 1)
        p = p.upper()
        if p in {"MESH", "MSH"}:
            return f"MSH:{code.upper()}"
        if p == "OMIM":
            return f"OMIM:{code}"
        if p in {"UMLS", "CUI"}:
            return f"UMLS:{code.upper()}"
        return f"{p}:{code}"
    u = s.upper()
    if MESH_UI_RE.fullmatch(u):
        return f"MSH:{u}"
    if dataset == "ncbi" and s.isdigit():
        return f"OMIM:{s}"
    if UMLS_CUI_RE.fullmatch(u):
        return f"UMLS:{u}"
    return f"UNKNOWN:{s}"


def parse_umls_field(raw: str) -> list[str]:
    s = raw.strip()
    if s.startswith("UMLS:"):
        s = s[5:]
    if not s or s == "-":
        return []
    vals = []
    for token in re.split(r"[;,|+]", s):
        token = token.strip().upper()
        if UMLS_CUI_RE.fullmatch(token):
            vals.append(token)
    return sorted(set(vals))


def parse_records(input_path: Path, dataset: str, concepts: dict, source_to_cuis: dict, keep_pmids: set[str] | None):
    title = ""
    abstract = ""
    current_pmid: str | None = None
    sent_spans: list[tuple[int, int, str]] = []
    ann_idx = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line:
                current_pmid = None
                title = ""
                abstract = ""
                sent_spans = []
                ann_idx = 0
                continue
            if "|t|" in line[:40]:
                current_pmid, _, title = line.split("|", 2)
                continue
            if "|a|" in line[:40]:
                _, _, abstract = line.split("|", 2)
                sent_spans = split_sentences(title + " " + abstract)
                continue
            if "\t" not in line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            pmid = parts[0]
            if keep_pmids is not None and pmid not in keep_pmids:
                continue
            if current_pmid is not None and pmid != current_pmid:
                pass
            try:
                start, end = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            mention = parts[3]
            annotation_type = parts[4]
            raw_gold = parts[5]
            source_ids = split_source_ids(raw_gold)
            source_keys = [source_key(x, dataset) for x in source_ids]

            if len(parts) >= 7 and parts[6].strip().startswith("UMLS:"):
                gold_cuis = parse_umls_field(parts[6])
            elif raw_gold.startswith("UMLS:"):
                gold_cuis = parse_umls_field(raw_gold)
            else:
                gold_cuis = []

            equivalent = set(gold_cuis)
            for sk in source_keys:
                if sk.startswith("UMLS:"):
                    equivalent.add(sk.split(":", 1)[1])
                equivalent.update(source_to_cuis.get(sk, []))

            tuis: set[str] = set()
            semantic_types: set[str] = set()
            for cui in gold_cuis:
                rec = concepts.get(cui, {})
                tuis.update(rec.get("tuis", []))
                semantic_types.update(rec.get("semantic_types", []))
            if dataset == "medmentions":
                for t in re.split(r"[,;|]", annotation_type):
                    t = t.strip()
                    if re.fullmatch(r"T\d{3}", t):
                        tuis.add(t)

            ann_idx += 1
            full_text = title + " " + abstract
            namespace = source_keys[0].split(":", 1)[0] if len(source_keys) == 1 else ("MULTI" if source_keys else "UNKNOWN")
            yield {
                "entity_space": "umls",
                "doc_id": pmid,
                "mention_id": f"{pmid}:{start}:{end}:{ann_idx}",
                "mention": mention,
                "start": start,
                "end": end,
                "annotation_type": annotation_type,
                "gold_id": raw_gold,
                "gold_ids": source_ids,
                "gold_namespace": namespace,
                "gold_source_keys": source_keys,
                "gold_cuis": gold_cuis,
                "gold_cui": gold_cuis[0] if len(gold_cuis) == 1 else None,
                "gold_entity_ids": gold_cuis,
                "gold_equivalent_cuis": sorted(equivalent),
                "gold_equivalent_entity_ids": sorted(equivalent),
                "mapping_status": "mapped" if gold_cuis else "unmapped",
                "tuis": sorted(tuis),
                "semantic_types": sorted(semantic_types),
                "sentence": find_sentence(sent_spans, start),
                "title": title,
                "abstract": abstract,
            }


def stratified_sample(records: list[dict], k: int, seed: int) -> list[dict]:
    if k <= 0 or k >= len(records):
        return list(records)
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = (r.get("tuis") or [r.get("annotation_type") or "UNK"])[0]
        buckets[key].append(r)
    keys = sorted(buckets)
    per = max(1, k // max(1, len(keys)))
    picked: list[dict] = []
    leftovers: list[dict] = []
    for key in keys:
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        picked.extend(bucket[:per])
        leftovers.extend(bucket[per:])
    rng.shuffle(picked)
    if len(picked) > k:
        return picked[:k]
    rng.shuffle(leftovers)
    return picked + leftovers[: k - len(picked)]


def main(argv=None) -> int:
    args = parse_benchmark_args(argv)
    with args.concepts.open("rb") as f:
        concepts = pickle.load(f)
    source_path = args.source_to_cuis or args.concepts.with_name("source_to_cuis.pkl")
    if source_path.exists():
        with source_path.open("rb") as f:
            source_to_cuis = pickle.load(f)
    else:
        source_to_cuis = {}
        print(f"WARNING: {source_path} missing; equivalence filtering uses mapped anchors only.", file=sys.stderr)

    keep_pmids = load_pmids(args.pmids_file)
    records = list(parse_records(args.input, args.dataset, concepts, source_to_cuis, keep_pmids))
    if not args.keep_unmapped:
        records = [r for r in records if r["gold_cuis"]]
    write_jsonl(args.output, records)
    print(f"wrote {args.output} ({len(records):,} mentions)")

    if args.pilot_output is not None and args.pilot_size > 0:
        pilot = stratified_sample(records, args.pilot_size, args.seed)
        write_jsonl(args.pilot_output, pilot)
        print(f"wrote {args.pilot_output} ({len(pilot):,} mentions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
