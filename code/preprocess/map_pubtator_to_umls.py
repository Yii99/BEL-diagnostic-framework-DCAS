from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.params import parse_pubtator_umls_mapping_args
from common.utils import (
    INTEGER_RE,
    MESH_UI_RE,
    MR_CODE,
    MR_CUI,
    MR_SAB,
    MR_TTY,
    MULTI_ID_SPLIT_RE,
    UMLS_CUI_RE,
    mesh_primary_tty,
    pct,
)


@dataclass(frozen=True)
class SourceId:
    namespace: str
    code: str

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.code}"


@dataclass
class MappingResult:
    source: SourceId
    cuis: set[str]
    method: str

    @property
    def status(self) -> str:
        if not self.cuis:
            return "unmapped"
        if len(self.cuis) == 1:
            return "mapped_unique"
        return "mapped_multiple"


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}.umls{input_path.suffix}")
    return input_path.with_name(f"{input_path.name}.umls.txt")


def is_annotation_line(line: str, gold_col: int) -> bool:
    """Return True for tab-separated PubTator annotation rows."""
    if not line or line.startswith("\n"):
        return False
    if "|t|" in line or "|a|" in line:
        return False
    cols = line.rstrip("\r\n").split("\t")
    return len(cols) > gold_col


def split_gold_ids(raw_gold: str) -> list[str]:
    """Split multi-ID annotations while preserving the original file untouched."""
    raw_gold = raw_gold.strip()
    if not raw_gold or raw_gold == "-1":
        return []
    parts = [p.strip() for p in MULTI_ID_SPLIT_RE.split(raw_gold)]
    return [p for p in parts if p and p != "-1"]


def normalize_source_id(raw_id: str, dataset: str) -> SourceId | None:
    """
    Convert one atomic benchmark ID to a source namespace + source code.

    Important ambiguity:
      - bare Cxxxxxxx is UMLS CUI (7 digits after C)
      - MeSH C UIs are C + 6 digits or C + 9 digits
    """
    s = raw_id.strip()
    if not s or s == "-1":
        return None

    if ":" in s:
        prefix, code = s.split(":", 1)
        p = prefix.strip().upper()
        code = code.strip()

        if p in {"MESH", "MSH"}:
            code = code.upper()
            if not MESH_UI_RE.fullmatch(code):
                return SourceId("MSH", code)
            return SourceId("MSH", code)

        if p == "OMIM":
            return SourceId("OMIM", code)

        if p in {"UMLS", "CUI"}:
            return SourceId("UMLS", code.upper())

        return SourceId(p, code)

    u = s.upper()

    if MESH_UI_RE.fullmatch(u):
        return SourceId("MSH", u)

    if dataset == "medmentions" and UMLS_CUI_RE.fullmatch(u):
        return SourceId("UMLS", u)

    if dataset == "ncbi" and INTEGER_RE.fullmatch(s):
        return SourceId("OMIM", s)

    if dataset == "generic" and UMLS_CUI_RE.fullmatch(u):
        return SourceId("UMLS", u)

    return SourceId("UNKNOWN", s)


def collect_requested_ids(
    input_path: Path,
    dataset: str,
    gold_col: int,
) -> tuple[dict[str, SourceId], dict[str, int]]:
    """Collect unique normalized source IDs and occurrence counts from input."""
    requested: dict[str, SourceId] = {}
    counts: dict[str, int] = defaultdict(int)

    with input_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not is_annotation_line(line, gold_col):
                continue
            cols = line.rstrip("\r\n").split("\t")
            raw_gold = cols[gold_col]
            for atomic in split_gold_ids(raw_gold):
                sid = normalize_source_id(atomic, dataset)
                if sid is None:
                    continue
                requested[sid.key] = sid
                counts[sid.key] += 1

    return requested, dict(counts)


def scan_mrconso(
    mrconso_path: Path,
    requested: dict[str, SourceId],
    *,
    mesh_code_fallback: bool,
) -> dict[str, MappingResult]:
    """
    Scan MRCONSO once and resolve only source IDs requested by the input file.

    MeSH:
      primary D... = SAB MSH + CODE + TTY MH
      primary C... = SAB MSH + CODE + TTY NM
      optional fallback = SAB MSH + CODE

    OMIM:
      SAB OMIM + CODE

    UMLS:
      direct CUI, but only if the CUI is observed in this MRCONSO release.

    No LAT / SUPPRESS filter is applied. SAB=MSH is English-only in UMLS (other
    languages use MSHGER, MSHFRE, ...) and the TTY in {MH, NM} gate already
    excludes every suppressed row: measured on 2020AA and 2017AA over all three
    corpora, 100% of accepted rows are (ENG, N), so filtering would be inert.
    """
    mesh_codes = {
        sid.code
        for sid in requested.values()
        if sid.namespace == "MSH"
    }
    omim_codes = {
        sid.code
        for sid in requested.values()
        if sid.namespace == "OMIM"
    }
    direct_cuis = {
        sid.code
        for sid in requested.values()
        if sid.namespace == "UMLS"
    }

    mesh_primary: dict[str, set[str]] = defaultdict(set)
    mesh_fallback: dict[str, set[str]] = defaultdict(set)
    omim_hits: dict[str, set[str]] = defaultdict(set)
    seen_direct: set[str] = set()

    max_needed = max(MR_CUI, MR_SAB, MR_TTY, MR_CODE)

    with mrconso_path.open("r", encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, start=1):
            cols = line.rstrip("\n").split("|")
            if len(cols) <= max_needed:
                continue

            cui = cols[MR_CUI]
            sab = cols[MR_SAB]
            tty = cols[MR_TTY]
            code = cols[MR_CODE]

            if cui in direct_cuis:
                seen_direct.add(cui)

            if sab == "MSH" and code in mesh_codes:
                if mesh_code_fallback:
                    mesh_fallback[code].add(cui)

                expected_tty = mesh_primary_tty(code)
                if expected_tty is not None and tty == expected_tty:
                    mesh_primary[code].add(cui)

            elif sab == "OMIM" and code in omim_codes:
                omim_hits[code].add(cui)

            if n % 5_000_000 == 0:
                print(f"  scanned MRCONSO: {n:,} lines", file=sys.stderr, flush=True)

    results: dict[str, MappingResult] = {}

    for key, sid in requested.items():
        if sid.namespace == "MSH":
            primary = set(mesh_primary.get(sid.code, ()))
            if primary:
                results[key] = MappingResult(sid, primary, "MSH_CODE_TTY")
            elif mesh_code_fallback:
                fallback = set(mesh_fallback.get(sid.code, ()))
                results[key] = MappingResult(sid, fallback, "MSH_CODE_FALLBACK")
            else:
                results[key] = MappingResult(sid, set(), "MSH_CODE_TTY")

        elif sid.namespace == "OMIM":
            results[key] = MappingResult(
                sid,
                set(omim_hits.get(sid.code, ())),
                "OMIM_CODE",
            )

        elif sid.namespace == "UMLS":
            cuis = {sid.code} if sid.code in seen_direct else set()
            results[key] = MappingResult(sid, cuis, "DIRECT_CUI")

        else:
            results[key] = MappingResult(sid, set(), "UNSUPPORTED_NAMESPACE")

    return results


def map_raw_gold(
    raw_gold: str,
    dataset: str,
    results: dict[str, MappingResult],
) -> tuple[list[str], list[str]]:
    """
    Return (union_cuis, atomic_statuses) for the original benchmark gold field.
    """
    cuis: set[str] = set()
    statuses: list[str] = []

    atoms = split_gold_ids(raw_gold)
    if not atoms:
        return [], ["no_normalization"]

    for atomic in atoms:
        sid = normalize_source_id(atomic, dataset)
        if sid is None:
            statuses.append(f"{atomic}=no_normalization")
            continue
        res = results.get(sid.key)
        if res is None:
            statuses.append(f"{atomic}=not_indexed")
            continue
        cuis.update(res.cuis)
        statuses.append(f"{atomic}={res.status}")

    return sorted(cuis), statuses


def write_augmented_pubtator(
    input_path: Path,
    output_path: Path,
    dataset: str,
    gold_col: int,
    append_mode: str,
    results: dict[str, MappingResult],
) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not is_annotation_line(line, gold_col):
                dst.write(line)
                continue

            newline = "\n" if line.endswith("\n") else ""
            cols = line.rstrip("\r\n").split("\t")
            raw_gold = cols[gold_col]
            cuis, _ = map_raw_gold(raw_gold, dataset, results)
            umls_value = "UMLS:" + (";".join(cuis) if cuis else "-")

            if cuis:
                stats["annotation_mapped"] += 1
                if len(cuis) == 1:
                    stats["annotation_unique_cui"] += 1
                else:
                    stats["annotation_multiple_cui"] += 1
            else:
                stats["annotation_unmapped"] += 1

            if append_mode == "column":
                if len(cols) > gold_col + 1 and cols[-1].startswith("UMLS:"):
                    cols[-1] = umls_value
                else:
                    cols.append(umls_value)
            else:
                base_gold = re.sub(r"\\s+UMLS:[^\\s]+$", "", cols[gold_col]).strip()
                cols[gold_col] = f"{base_gold} {umls_value}"

            dst.write("\t".join(cols) + newline)

    return dict(stats)


def write_mapping_tsv(
    path: Path,
    requested: dict[str, SourceId],
    counts: dict[str, int],
    results: dict[str, MappingResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "source_namespace",
                "source_id",
                "mention_count",
                "mapping_status",
                "mapping_method",
                "n_cui",
                "umls_cuis",
            ]
        )
        for key in sorted(requested):
            sid = requested[key]
            res = results[key]
            cuis = sorted(res.cuis)
            writer.writerow(
                [
                    sid.namespace,
                    sid.code,
                    counts.get(key, 0),
                    res.status,
                    res.method,
                    len(cuis),
                    ";".join(cuis),
                ]
            )


def print_summary(
    requested: dict[str, SourceId],
    counts: dict[str, int],
    results: dict[str, MappingResult],
    annotation_stats: dict[str, int],
) -> None:
    n_unique = len(requested)
    n_unique_mapped = sum(bool(r.cuis) for r in results.values())
    n_unique_multiple = sum(len(r.cuis) > 1 for r in results.values())
    n_unique_unmapped = n_unique - n_unique_mapped

    mention_total = sum(counts.values())
    mention_mapped = sum(
        counts.get(k, 0) for k, r in results.items() if r.cuis
    )

    print("\n== Mapping summary ==")
    print(f"unique source IDs       : {n_unique:,}")
    print(f"unique IDs mapped       : {n_unique_mapped:,}")
    print(f"unique IDs multiple CUI : {n_unique_multiple:,}")
    print(f"unique IDs unmapped     : {n_unique_unmapped:,}")
    print(f"unique anchor coverage  : {pct(n_unique_mapped, n_unique)}%")
    print(f"atomic ID occurrences   : {mention_total:,}")
    print(f"atomic occurrence cover.: {pct(mention_mapped, mention_total)}%")

    for k in sorted(annotation_stats):
        print(f"{k:24s}: {annotation_stats[k]:,}")


def main(argv: list[str] | None = None) -> int:
    args = parse_pubtator_umls_mapping_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    if not args.mrconso.exists():
        raise SystemExit(f"MRCONSO not found: {args.mrconso}")

    output = args.output or default_output_path(args.input)
    mapping_tsv = args.mapping_tsv or Path(str(output) + ".mapping.tsv")

    print(f"dataset  : {args.dataset}")
    print(f"input    : {args.input}")
    print(f"MRCONSO  : {args.mrconso}")
    print(f"output   : {output}")
    print(f"map TSV  : {mapping_tsv}")

    requested, counts = collect_requested_ids(
        args.input, args.dataset, args.gold_col
    )
    print(f"requested unique source IDs: {len(requested):,}")

    results = scan_mrconso(
        args.mrconso,
        requested,
        mesh_code_fallback=args.mesh_code_fallback,
    )

    write_mapping_tsv(mapping_tsv, requested, counts, results)
    annotation_stats = write_augmented_pubtator(
        args.input,
        output,
        args.dataset,
        args.gold_col,
        args.append_mode,
        results,
    )

    print_summary(requested, counts, results, annotation_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
