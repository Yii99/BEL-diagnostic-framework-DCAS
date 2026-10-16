from __future__ import annotations

import csv
import gzip
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

MR_CUI = 0
MR_LAT = 1
MR_TS = 2
MR_LUI = 3
MR_STT = 4
MR_SUI = 5
MR_ISPREF = 6
MR_AUI = 7
MR_SAUI = 8
MR_SCUI = 9
MR_SDUI = 10
MR_SAB = 11
MR_TTY = 12
MR_CODE = 13
MR_STR = 14
MR_SRL = 15
MR_SUPPRESS = 16
MR_CVF = 17

MESH_UI_RE = re.compile(r"^[DC](?:\d{6}|\d{9})$")
MESH_D_RE = re.compile(r"^D(?:\d{6}|\d{9})$")
MESH_C_RE = re.compile(r"^C(?:\d{6}|\d{9})$")
UMLS_CUI_RE = re.compile(r"^C\d{7}$")
INTEGER_RE = re.compile(r"^\d+$")

MULTI_ID_SPLIT_RE = re.compile(r"[|+,;]")

NO_GOLD_VALUES = {"", "-", "-1", "NIL", "NULL", "None", "none", "NA", "N/A"}

XML_ENCODING_UTF8_RE = re.compile(r'(<\?xml[^>]*encoding=["\'])utf8(["\'][^>]*\?>)', re.IGNORECASE)
INVALID_XML_CHAR_RE = re.compile(
    r"[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]"
)
BARE_AMP_RE = re.compile(r"&(?!amp;|lt;|gt;|quot;|apos;|#[0-9]+;|#x[0-9A-Fa-f]+;)")


@dataclass(frozen=True)
class GoldItem:
    """One mention-level gold annotation, normalized for coverage analysis."""

    dataset: str
    split: str
    entity_type: str
    raw_id: str
    normalized_id: str
    namespace: str
    doc_id: str = ""
    mention_text: str = ""
    source_file: str = ""


@dataclass
class CuiInfo:
    """Minimal MRCONSO-derived information for a mapped CUI."""

    total_rows: int = 0
    eng_unsuppressed_names: Set[str] = field(default_factory=set)
    source_rows: Counter = field(default_factory=Counter)


@dataclass
class MappingResult:
    """Resolved mapping result for one normalized benchmark gold ID."""

    status: str
    cuis: List[str]
    match_tier: str = ""
    match_fields: List[str] = field(default_factory=list)
    match_ttys: List[str] = field(default_factory=list)

def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def read_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line.rstrip("\n")


def split_multi_ids(raw_id: str) -> List[str]:
    """Split common multi-ID strings while keeping plain IDs unchanged."""
    raw_id = (raw_id or "").strip()
    if raw_id in NO_GOLD_VALUES:
        return [raw_id]
    parts = MULTI_ID_SPLIT_RE.split(raw_id)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [raw_id]


def infer_split_from_name(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    text = f"{parent}/{name}"
    if any(x in text for x in ["training", "train", "trng"]):
        return "train"
    if any(x in text for x in ["development", "develop", "devel", "dev"]):
        return "dev"
    if "test" in text:
        return "test"
    if "nlmiat" in text:
        return "nlmiat"
    return "unknown"


def normalize_gold_id(raw_id: str, dataset: str, entity_type: str = "") -> Tuple[str, str]:
    """Normalize benchmark IDs and infer a source namespace for UMLS lookup.

    Returns (namespace, normalized_id). The namespace is one of:
    MSH, OMIM, UMLS, NO_GOLD, UNKNOWN.
    """
    dataset = dataset.lower()
    raw = (raw_id or "").strip()
    raw = raw.replace("\ufeff", "")

    if raw in NO_GOLD_VALUES:
        return "NO_GOLD", raw

    upper = raw.upper()
    if upper.startswith("MESH:"):
        code = raw.split(":", 1)[1].strip()
        return "MSH", code
    if upper.startswith("MSH:"):
        code = raw.split(":", 1)[1].strip()
        return "MSH", code
    if upper.startswith("OMIM:"):
        code = raw.split(":", 1)[1].strip()
        return "OMIM", code
    if upper.startswith("UMLS:"):
        code = raw.split(":", 1)[1].strip()
        return "UMLS", code

    if UMLS_CUI_RE.match(raw):
        return "UMLS", raw

    if MESH_UI_RE.match(raw):
        return "MSH", raw

    if INTEGER_RE.match(raw):
        return "OMIM", raw

    return "UNKNOWN", raw


def parse_pubtator_file(
    path: Path,
    dataset: str,
    split: Optional[str] = None,
    allowed_entity_types: Optional[Set[str]] = None,
    pmid_to_split: Optional[Dict[str, str]] = None,
) -> List[GoldItem]:
    """Parse PubTator-like files.

    Expected annotation row:
        PMID <tab> start <tab> end <tab> mention <tab> type <tab> id
    Title/abstract rows such as PMID|t|... and PMID|a|... are ignored.
    """
    items: List[GoldItem] = []
    split = split or infer_split_from_name(path)

    for line in read_lines(path):
        if not line or "|t|" in line or "|a|" in line:
            continue
        fields = line.split("\t")
        if len(fields) < 6:
            continue

        doc_id = fields[0].strip()
        mention_text = fields[3].strip()
        entity_type = fields[4].strip()
        raw_id_field = fields[5].strip()

        if allowed_entity_types and entity_type not in allowed_entity_types:
            continue

        row_split = pmid_to_split.get(doc_id, split) if pmid_to_split else split
        for raw_id in split_multi_ids(raw_id_field):
            namespace, norm_id = normalize_gold_id(raw_id, dataset, entity_type)
            items.append(
                GoldItem(
                    dataset=dataset,
                    split=row_split,
                    entity_type=entity_type,
                    raw_id=raw_id,
                    normalized_id=norm_id,
                    namespace=namespace,
                    doc_id=doc_id,
                    mention_text=mention_text,
                    source_file=str(path),
                )
            )

    return items


def discover_bc5cdr_files(data_path: Path) -> List[Tuple[str, Path]]:
    if data_path.is_file():
        return [(infer_split_from_name(data_path), data_path)]

    candidates = [
        ("train", "CDR_TrainingSet.PubTator.txt"),
        ("dev", "CDR_DevelopmentSet.PubTator.txt"),
        ("test", "CDR_TestSet.PubTator.txt"),
    ]
    found: List[Tuple[str, Path]] = []
    for split, fname in candidates:
        matches = list(data_path.rglob(fname))
        if matches:
            found.append((split, matches[0]))
    return found


def load_bc5cdr(data_path: Path) -> List[GoldItem]:
    files = discover_bc5cdr_files(data_path)
    if not files:
        raise FileNotFoundError(f"Could not find BC5CDR PubTator files under {data_path}")
    items: List[GoldItem] = []
    for split, path in files:
        items.extend(
            parse_pubtator_file(
                path=path,
                dataset="bc5cdr",
                split=split,
                allowed_entity_types={"Chemical", "Disease"},
            )
        )
    return items


def discover_ncbi_files(data_path: Path) -> List[Tuple[str, Path]]:
    if data_path.is_file():
        return [(infer_split_from_name(data_path), data_path)]

    patterns = [
        ("train", "*train*corpus*.txt"),
        ("dev", "*develop*corpus*.txt"),
        ("dev", "*devel*corpus*.txt"),
        ("test", "*test*corpus*.txt"),
    ]
    found: List[Tuple[str, Path]] = []
    seen: Set[Path] = set()
    for split, pattern in patterns:
        for p in data_path.rglob(pattern):
            if p not in seen:
                found.append((split, p))
                seen.add(p)
    return found


def load_ncbi(data_path: Path) -> List[GoldItem]:
    files = discover_ncbi_files(data_path)
    if not files:
        raise FileNotFoundError(f"Could not find NCBI-Disease corpus files under {data_path}")
    items: List[GoldItem] = []
    for split, path in files:
        items.extend(
            parse_pubtator_file(
                path=path,
                dataset="ncbi",
                split=split,
                allowed_entity_types={"SpecificDisease", "DiseaseClass", "Modifier", "CompositeMention"},
            )
        )
    return items


def read_pmid_list(path: Path) -> Set[str]:
    pmids: Set[str] = set()
    for line in read_lines(path):
        line = line.strip()
        if line:
            pmids.add(line)
    return pmids


def load_medmentions(data_path: Path) -> List[GoldItem]:
    """Load MedMentions PubTator file and split using corpus_pubtator_pmids_* files."""
    if data_path.is_file():
        return parse_pubtator_file(data_path, dataset="medmentions", split="unknown")

    corpus_candidates = [
        data_path / "data" / "corpus_pubtator.txt",
        data_path / "corpus_pubtator.txt",
    ]
    corpus_path = next((p for p in corpus_candidates if p.exists()), None)
    if corpus_path is None:
        matches = list(data_path.rglob("corpus_pubtator.txt"))
        corpus_path = matches[0] if matches else None
    if corpus_path is None:
        raise FileNotFoundError(f"Could not find corpus_pubtator.txt under {data_path}")

    pmid_to_split: Dict[str, str] = {}
    split_patterns = [
        ("train", "corpus_pubtator_pmids_trng"),
        ("train", "corpus_pubtator_pmids_train"),
        ("dev", "corpus_pubtator_pmids_dev"),
        ("test", "corpus_pubtator_pmids_test"),
    ]
    for split, fname in split_patterns:
        matches = list(data_path.rglob(fname))
        if not matches:
            continue
        for pmid in read_pmid_list(matches[0]):
            pmid_to_split[pmid] = split

    if not pmid_to_split:
        warn(f"No MedMentions PMID split files found under {data_path}; using split=unknown")

    return parse_pubtator_file(
        path=corpus_path,
        dataset="medmentions",
        split="unknown",
        pmid_to_split=pmid_to_split or None,
    )


def read_xml_text_tolerant(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")

    text = XML_ENCODING_UTF8_RE.sub(r"\1UTF-8\2", text, count=1)
    text = INVALID_XML_CHAR_RE.sub("", text)
    text = BARE_AMP_RE.sub("&amp;", text)
    return text


def get_infon(annotation: ET.Element, key: str) -> str:
    for infon in annotation.findall("infon"):
        if infon.attrib.get("key") == key:
            return (infon.text or "").strip()
    return ""


def parse_nlmchem_xml_file(path: Path, split: Optional[str] = None) -> List[GoldItem]:
    split = split or infer_split_from_name(path)
    items: List[GoldItem] = []

    try:
        root = ET.fromstring(read_xml_text_tolerant(path))
    except ET.ParseError as exc:
        raise ET.ParseError(f"Failed to parse NLM-Chem XML file {path}: {exc}") from exc

    for doc in root.findall("document"):
        doc_id_elem = doc.find("id")
        doc_id = (doc_id_elem.text or "").strip() if doc_id_elem is not None else ""

        for ann in doc.findall(".//annotation"):
            entity_type = get_infon(ann, "type")
            if entity_type != "Chemical":
                continue
            raw_id_field = get_infon(ann, "identifier")
            if not raw_id_field:
                continue
            text_elem = ann.find("text")
            mention_text = (text_elem.text or "").strip() if text_elem is not None else ""

            for raw_id in split_multi_ids(raw_id_field):
                namespace, norm_id = normalize_gold_id(raw_id, "nlmchem", entity_type)
                items.append(
                    GoldItem(
                        dataset="nlmchem",
                        split=split,
                        entity_type=entity_type,
                        raw_id=raw_id,
                        normalized_id=norm_id,
                        namespace=namespace,
                        doc_id=doc_id,
                        mention_text=mention_text,
                        source_file=str(path),
                    )
                )

    return items

def discover_nlmchem_files(data_path: Path) -> List[Tuple[str, Path]]:
    if data_path.is_file():
        return [(infer_split_from_name(data_path), data_path)]
    patterns = ["*NLM*Chem*BioC*.xml", "*.BioC.xml", "*.xml"]
    found: List[Tuple[str, Path]] = []
    seen: Set[Path] = set()
    for pattern in patterns:
        for p in data_path.rglob(pattern):
            if p not in seen:
                found.append((infer_split_from_name(p), p))
                seen.add(p)
    return sorted(found, key=lambda x: (x[0], str(x[1])))


def load_nlmchem(data_path: Path) -> List[GoldItem]:
    files = discover_nlmchem_files(data_path)
    if not files:
        raise FileNotFoundError(f"Could not find NLM-Chem XML files under {data_path}")
    items: List[GoldItem] = []
    for split, path in files:
        items.extend(parse_nlmchem_xml_file(path, split=split))
    return items


def load_gold_items(dataset: str, data: str) -> List[GoldItem]:
    dataset = dataset.lower()
    if dataset == "nbci":
        dataset = "ncbi"
    data_path = Path(data)

    if dataset == "bc5cdr":
        return load_bc5cdr(data_path)
    if dataset == "ncbi":
        return load_ncbi(data_path)
    if dataset == "medmentions":
        return load_medmentions(data_path)
    if dataset == "nlmchem":
        return load_nlmchem(data_path)
    raise ValueError(f"Unsupported dataset: {dataset}")


def collect_unique_gold_keys(items: Sequence[GoldItem]) -> Dict[str, Set[str]]:
    ns_to_ids: DefaultDict[str, Set[str]] = defaultdict(set)
    for item in items:
        if item.namespace == "NO_GOLD":
            continue
        ns_to_ids[item.namespace].add(item.normalized_id)
    return dict(ns_to_ids)


def mesh_primary_tty(code: str) -> Optional[str]:
    """Return expected MeSH TTY for primary mapping.

    Dxxxxxx = MeSH Descriptor. Main heading terms use TTY=MH.
    Cxxxxxx = MeSH Supplementary Concept Record. Name terms usually use TTY=NM.
    Both forms also occur as 9-digit UIs in recent MeSH releases.
    """
    if MESH_D_RE.match(code):
        return "MH"
    if MESH_C_RE.match(code):
        return "NM"
    return None


def add_match(
    matches: DefaultDict[Tuple[str, str], DefaultDict[str, DefaultDict[str, Set[str]]]],
    namespace: str,
    code: str,
    tier: str,
    cui: str,
    field_name: str,
    tty: str,
) -> None:
    key = (namespace, code)
    label = f"{field_name}:TTY={tty or '<blank>'}"
    matches[key][tier][cui].add(label)


def scan_mrconso_for_mappings(
    mrconso_path: Path,
    ns_to_ids: Dict[str, Set[str]],
    lang: str = "ENG",
    include_suppressed: bool = False,
) -> Tuple[
    DefaultDict[Tuple[str, str], DefaultDict[str, DefaultDict[str, Set[str]]]],
    Dict[str, CuiInfo],
]:

    wanted_msh = ns_to_ids.get("MSH", set())
    wanted_omim = ns_to_ids.get("OMIM", set())
    wanted_umls = ns_to_ids.get("UMLS", set())

    wanted_all_cuis: Set[str] = set(wanted_umls)

    matches: DefaultDict[Tuple[str, str], DefaultDict[str, DefaultDict[str, Set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    cui_info: Dict[str, CuiInfo] = {}

    def update_cui_info(cui: str, lat: str, sab: str, string: str, suppress: str) -> None:
        info = cui_info.setdefault(cui, CuiInfo())
        info.total_rows += 1
        info.source_rows[sab] += 1
        if lat == lang and string and (include_suppressed or suppress == "N"):
            info.eng_unsuppressed_names.add(string)

    with mrconso_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            fields = line.rstrip("\n").split("|")
            if len(fields) <= MR_SUPPRESS:
                warn(f"Skipping malformed MRCONSO line {line_no}: expected >=17 fields")
                continue

            cui = fields[MR_CUI]
            lat = fields[MR_LAT]
            scui = fields[MR_SCUI]
            sdui = fields[MR_SDUI]
            sab = fields[MR_SAB]
            tty = fields[MR_TTY]
            code = fields[MR_CODE]
            string = fields[MR_STR]
            suppress = fields[MR_SUPPRESS]
            sab_upper = sab.upper()

            if cui in wanted_umls:
                add_match(matches, "UMLS", cui, "direct", cui, "CUI", tty)
                wanted_all_cuis.add(cui)
                update_cui_info(cui, lat, sab, string, suppress)
                continue

            if sab == "MSH" and wanted_msh:
                for field_name, value in (("CODE", code), ("SDUI", sdui), ("SCUI", scui)):
                    if not value or value not in wanted_msh:
                        continue
                    expected_tty = mesh_primary_tty(value)
                    if field_name == "CODE" and expected_tty and tty == expected_tty:
                        add_match(matches, "MSH", value, "primary", cui, field_name, tty)
                    else:
                        add_match(matches, "MSH", value, "fallback", cui, field_name, tty)
                    wanted_all_cuis.add(cui)
                    update_cui_info(cui, lat, sab, string, suppress)

            if sab == "OMIM" and wanted_omim:
                for field_name, value in (("CODE", code), ("SDUI", sdui), ("SCUI", scui)):
                    if not value or value not in wanted_omim:
                        continue
                    tier = "primary" if field_name == "CODE" else "fallback"
                    add_match(matches, "OMIM", value, tier, cui, field_name, tty)
                    wanted_all_cuis.add(cui)
                    update_cui_info(cui, lat, sab, string, suppress)

    return matches, cui_info


def resolve_mapping(
    namespace: str,
    normalized_id: str,
    matches: DefaultDict[Tuple[str, str], DefaultDict[str, DefaultDict[str, Set[str]]]],
) -> MappingResult:
    if namespace == "NO_GOLD":
        return MappingResult(status="excluded_no_gold_id", cuis=[])
    if namespace == "UNKNOWN":
        return MappingResult(status="unsupported_namespace", cuis=[])

    key = (namespace, normalized_id)
    tier_to_cuis = matches.get(key, {})

    for tier in ("direct", "primary", "fallback"):
        cui_to_labels = tier_to_cuis.get(tier, {})
        if not cui_to_labels:
            continue
        cuis = sorted(cui_to_labels.keys())
        labels = sorted({label for label_set in cui_to_labels.values() for label in label_set})
        fields = sorted({label.split(":", 1)[0] for label in labels})
        ttys = sorted({label.split("TTY=", 1)[1] for label in labels if "TTY=" in label})
        if tier == "direct":
            status = "mapped_direct_cui" if len(cuis) == 1 else "mapped_direct_multiple"
        elif tier == "primary":
            status = "mapped_primary_unique" if len(cuis) == 1 else "mapped_primary_multiple"
        else:
            status = "mapped_fallback_unique" if len(cuis) == 1 else "mapped_fallback_multiple"
        return MappingResult(status=status, cuis=cuis, match_tier=tier, match_fields=fields, match_ttys=ttys)

    return MappingResult(status="not_found", cuis=[])


def aggregate_gold_items(items: Sequence[GoldItem]) -> Dict[Tuple[str, str, str, str, str], Dict[str, object]]:
    """Aggregate mention-level items by dataset/split/type/namespace/id."""
    agg: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}
    for item in items:
        key = (item.dataset, item.split, item.entity_type, item.namespace, item.normalized_id)
        if key not in agg:
            agg[key] = {
                "dataset": item.dataset,
                "split": item.split,
                "entity_type": item.entity_type,
                "namespace": item.namespace,
                "gold_id": item.normalized_id,
                "raw_id_examples": set(),
                "mention_count": 0,
                "example_mention": item.mention_text,
                "source_files": set(),
            }
        agg[key]["raw_id_examples"].add(item.raw_id)
        agg[key]["mention_count"] += 1
        if item.source_file:
            agg[key]["source_files"].add(item.source_file)
    return agg


def is_mapped_status(status: str) -> bool:
    return status.startswith("mapped_")


def write_mapping_outputs(
    items: Sequence[GoldItem],
    matches: DefaultDict[Tuple[str, str], DefaultDict[str, DefaultDict[str, Set[str]]]],
    cui_info: Dict[str, CuiInfo],
    out_dir: Path,
    include_no_gold_id: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = aggregate_gold_items(items)
    detail_rows: List[Dict[str, object]] = []

    for key, row in sorted(agg.items(), key=lambda kv: kv[0]):
        namespace = str(row["namespace"])
        gold_id = str(row["gold_id"])
        result = resolve_mapping(namespace, gold_id, matches)

        if result.status == "excluded_no_gold_id" and not include_no_gold_id:
            pass
        else:
            total_rows = sum(cui_info.get(cui, CuiInfo()).total_rows for cui in result.cuis)
            total_names = sum(len(cui_info.get(cui, CuiInfo()).eng_unsuppressed_names) for cui in result.cuis)
            detail_rows.append(
                {
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "entity_type": row["entity_type"],
                    "namespace": namespace,
                    "gold_id": gold_id,
                    "raw_id_examples": ";".join(sorted(row["raw_id_examples"])),
                    "mention_count": row["mention_count"],
                    "status": result.status,
                    "match_tier": result.match_tier,
                    "match_fields": ";".join(result.match_fields),
                    "match_ttys": ";".join(result.match_ttys),
                    "mapped_cuis": ";".join(result.cuis),
                    "num_mapped_cuis": len(result.cuis),
                    "total_mrconso_rows_for_mapped_cuis": total_rows,
                    "total_eng_unsuppressed_names_for_mapped_cuis": total_names,
                    "example_mention": row["example_mention"],
                    "source_files": ";".join(sorted(row["source_files"])),
                }
            )

    detail_path = out_dir / "mapping_detail.tsv"
    detail_fields = [
        "dataset",
        "split",
        "entity_type",
        "namespace",
        "gold_id",
        "raw_id_examples",
        "mention_count",
        "status",
        "match_tier",
        "match_fields",
        "match_ttys",
        "mapped_cuis",
        "num_mapped_cuis",
        "total_mrconso_rows_for_mapped_cuis",
        "total_eng_unsuppressed_names_for_mapped_cuis",
        "example_mention",
        "source_files",
    ]
    write_tsv(detail_path, detail_rows, detail_fields)

    unmapped_rows = [
        row for row in detail_rows if row["status"] in {"not_found", "unsupported_namespace"}
    ]
    unmapped_path = out_dir / "unmapped_ids.tsv"
    write_tsv(unmapped_path, unmapped_rows, detail_fields)

    summary_rows = build_coverage_summary(agg, matches)
    summary_path = out_dir / "coverage_summary.tsv"
    summary_fields = [
        "dataset",
        "split",
        "entity_type",
        "mention_gold_ids",
        "mapped_mention_gold_ids",
        "mention_coverage_pct",
        "unique_gold_ids",
        "mapped_unique_gold_ids",
        "unique_coverage_pct",
        "primary_unique_gold_ids",
        "primary_unique_coverage_pct",
        "fallback_unique_gold_ids",
        "multiple_unique_gold_ids",
        "unmapped_unique_gold_ids",
        "unsupported_unique_gold_ids",
        "excluded_no_gold_mentions",
        "excluded_no_gold_unique_ids",
        "status_counts",
    ]
    write_tsv(summary_path, summary_rows, summary_fields)


def build_coverage_summary(
    agg: Dict[Tuple[str, str, str, str, str], Dict[str, object]],
    matches: DefaultDict[Tuple[str, str], DefaultDict[str, DefaultDict[str, Set[str]]]],
) -> List[Dict[str, object]]:
    groups: DefaultDict[Tuple[str, str, str], Dict[str, object]] = defaultdict(
        lambda: {
            "mention_gold_ids": 0,
            "mapped_mention_gold_ids": 0,
            "unique_gold_ids": 0,
            "mapped_unique_gold_ids": 0,
            "primary_unique_gold_ids": 0,
            "fallback_unique_gold_ids": 0,
            "multiple_unique_gold_ids": 0,
            "unmapped_unique_gold_ids": 0,
            "unsupported_unique_gold_ids": 0,
            "excluded_no_gold_mentions": 0,
            "excluded_no_gold_unique_ids": 0,
            "status_counts": Counter(),
        }
    )

    for _key, row in agg.items():
        group_key = (str(row["dataset"]), str(row["split"]), str(row["entity_type"]))
        g = groups[group_key]
        namespace = str(row["namespace"])
        gold_id = str(row["gold_id"])
        mention_count = int(row["mention_count"])
        result = resolve_mapping(namespace, gold_id, matches)
        status = result.status
        g["status_counts"][status] += 1

        if status == "excluded_no_gold_id":
            g["excluded_no_gold_mentions"] += mention_count
            g["excluded_no_gold_unique_ids"] += 1
            continue

        g["mention_gold_ids"] += mention_count
        g["unique_gold_ids"] += 1

        if is_mapped_status(status):
            g["mapped_mention_gold_ids"] += mention_count
            g["mapped_unique_gold_ids"] += 1

        if status == "mapped_primary_unique" or status == "mapped_direct_cui":
            g["primary_unique_gold_ids"] += 1
        elif status == "mapped_fallback_unique":
            g["fallback_unique_gold_ids"] += 1
        elif "multiple" in status:
            g["multiple_unique_gold_ids"] += 1
        elif status == "not_found":
            g["unmapped_unique_gold_ids"] += 1
        elif status == "unsupported_namespace":
            g["unsupported_unique_gold_ids"] += 1

    rows: List[Dict[str, object]] = []
    for (dataset, split, entity_type), g in sorted(groups.items()):
        mention_total = int(g["mention_gold_ids"])
        mention_mapped = int(g["mapped_mention_gold_ids"])
        unique_total = int(g["unique_gold_ids"])
        unique_mapped = int(g["mapped_unique_gold_ids"])
        primary_unique = int(g["primary_unique_gold_ids"])
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "entity_type": entity_type,
                "mention_gold_ids": mention_total,
                "mapped_mention_gold_ids": mention_mapped,
                "mention_coverage_pct": pct(mention_mapped, mention_total),
                "unique_gold_ids": unique_total,
                "mapped_unique_gold_ids": unique_mapped,
                "unique_coverage_pct": pct(unique_mapped, unique_total),
                "primary_unique_gold_ids": primary_unique,
                "primary_unique_coverage_pct": pct(primary_unique, unique_total),
                "fallback_unique_gold_ids": g["fallback_unique_gold_ids"],
                "multiple_unique_gold_ids": g["multiple_unique_gold_ids"],
                "unmapped_unique_gold_ids": g["unmapped_unique_gold_ids"],
                "unsupported_unique_gold_ids": g["unsupported_unique_gold_ids"],
                "excluded_no_gold_mentions": g["excluded_no_gold_mentions"],
                "excluded_no_gold_unique_ids": g["excluded_no_gold_unique_ids"],
                "status_counts": ";".join(
                    f"{k}:{v}" for k, v in sorted(g["status_counts"].items())
                ),
            }
        )
    return rows


def pct(num: int, den: int) -> str:
    if den == 0:
        return "0.00"
    return f"{100.0 * num / den:.2f}"


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def open_text(path: Path):
    """Open plain-text or gzip-compressed UTF-8 text for reading."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def load_jsonl(path: Path) -> List[dict]:
    """Load a UTF-8 JSONL file into memory."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream a JSONL file. Use this instead of ``load_jsonl`` when the file is
    large: MedMentions has 40k mentions and 5.6M negatives."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Write records as UTF-8 JSONL, preserving non-ASCII biomedical text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def unique_preserve(values: Iterable[str]) -> List[str]:
    """Case-insensitive de-duplication while preserving first-seen surface form."""
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        v = str(value).strip()
        if not v or v == "-":
            continue
        key = v.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def mention_context(rec: dict) -> Tuple[str, str, str]:
    """Return ``(context_left, context_right, source)`` for one mention record.

    ``source`` names which fallback produced the split, so callers can report
    how often the offsets were usable.
    """
    mention = rec.get("mention") or ""
    title = (rec.get("title") or "").strip()
    abstract = rec.get("abstract") or ""
    start, end = rec.get("start"), rec.get("end")

    candidates: List[Tuple[str, str]] = []
    if title and abstract:
        candidates.append(("title_abstract", f"{title} {abstract}"))
    if abstract:
        candidates.append(("abstract", abstract))
    if title:
        candidates.append(("title", title))

    if isinstance(start, int) and isinstance(end, int) and end > start:
        for source, doc in candidates:
            if end <= len(doc) and doc[start:end] == mention:
                return doc[:start], doc[end:], source

    sentence = rec.get("sentence") or ""
    for source, text in (("sentence", sentence), *candidates):
        if mention and text:
            idx = text.find(mention)
            if idx != -1:
                return text[:idx], text[idx + len(mention):], source

    return "", "", "mention_only"


def alias_list(rec: dict, cap: int) -> List[str]:
    """Preferred name first, then the remaining aliases shortest-first, capped.

    The shortest alias is usually the canonical form, and scoring takes the max
    over this list, so the order only matters once ``cap`` truncates it.
    """
    seen: dict[str, None] = {}
    name = rec.get("preferred_name") or ""
    if name:
        seen[name] = None
    for a in rec.get("aliases") or []:
        if a:
            seen.setdefault(a, None)
    out = list(seen)
    if len(out) <= 1:
        return out
    ordered = out[:1] + sorted(out[1:], key=lambda s: (len(s), s.casefold()))
    return ordered[:cap] if cap else ordered


def preferred_name(rec: dict) -> str:
    """Return preferred entity name, falling back to the first alias."""
    return rec.get("preferred_name") or next(iter(rec.get("aliases") or []), "")


def gold_entity_ids(mention: dict) -> Set[str]:
    """Return all entity IDs that must be treated as gold-equivalent."""
    ids = mention.get("gold_equivalent_entity_ids") or mention.get("gold_entity_ids")
    if ids:
        return {str(x) for x in ids}
    if mention.get("gold_cuis"):
        return {str(x) for x in mention["gold_cuis"]}
    if mention.get("gold_cui"):
        return {str(mention["gold_cui"])}
    return set()


def entity_space(mention: dict, concepts: Optional[dict] = None) -> str:
    """Resolve the entity ID space (e.g. umls) for generic samplers."""
    value = mention.get("entity_space")
    if value:
        return str(value)
    if concepts:
        for rec in concepts.values():
            if rec.get("entity_space"):
                return str(rec["entity_space"])
            break
    return "entity"


def ids_payload(ids: List[str], space: str) -> dict:
    """Build a backward-compatible negative-ID payload for an entity space."""
    payload = {"entity_id_list": ids}
    if space == "umls":
        payload["cui_list"] = ids
    return payload


def collect_negative_ids(record: dict) -> Set[str]:
    """Collect entity IDs only from the ``negatives`` subtree of a record."""
    found: Set[str] = set()

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in {"entity_id_list", "cui_list"} and isinstance(value, list):
                    found.update(str(x) for x in value)
                else:
                    walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(record.get("negatives", {}))
    return found


def cap_ids(ids: List[str], cap: int, seed_key: str) -> List[str]:
    """Deterministically cap candidate IDs using a per-record seed key."""
    unique_ids = sorted(set(ids))
    if cap <= 0 or len(unique_ids) <= cap:
        return unique_ids
    rng = random.Random(seed_key)
    return rng.sample(unique_ids, cap)
