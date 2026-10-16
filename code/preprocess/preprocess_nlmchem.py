from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.params import parse_nlmchem_preprocess_args


MESH_UI_RE = re.compile(r"^[DC](?:\d{6}|\d{9})$")
MULTI_ID_SPLIT_RE = re.compile(r"[|+,;]")
INVALID_XML_CONTROL_RE = re.compile(
    "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)
BARE_AMPERSAND_RE = re.compile(
    r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z_:][A-Za-z0-9_.:-]*;)"
)


@dataclass(frozen=True)
class NlmChemAnnotation:
    doc_id: str
    annotation_id: str
    annotation_type: str
    start: int
    end: int
    mention_text: str
    raw_identifier: str
    mesh_ids: tuple[str, ...]
    n_locations: int

    @property
    def is_multi_normalized(self) -> bool:
        return len(self.mesh_ids) > 1

    @property
    def is_single_gold(self) -> bool:
        return len(self.mesh_ids) == 1

    @property
    def normalized_gold(self) -> str:
        if not self.mesh_ids:
            return "-1"
        return ",".join(self.mesh_ids)


@dataclass(frozen=True)
class NlmChemDocument:
    doc_id: str
    title: str
    abstract: str
    annotations: tuple[NlmChemAnnotation, ...]


def _strip_namespace(tag: str) -> str:
    """Remove an optional XML namespace from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1]


def _children_by_tag(element: ET.Element, tag: str) -> list[ET.Element]:
    return [child for child in list(element) if _strip_namespace(child.tag) == tag]


def _first_child(element: ET.Element, tag: str) -> ET.Element | None:
    for child in list(element):
        if _strip_namespace(child.tag) == tag:
            return child
    return None


def _child_text(element: ET.Element, tag: str, default: str = "") -> str:
    child = _first_child(element, tag)
    if child is None or child.text is None:
        return default
    return child.text


def _infons(element: ET.Element) -> dict[str, str]:
    values: dict[str, str] = {}
    for child in list(element):
        if _strip_namespace(child.tag) != "infon":
            continue
        key = (child.attrib.get("key") or "").strip()
        if key:
            values[key] = child.text or ""
    return values


def _get_infon_case_insensitive(infons: dict[str, str], *keys: str) -> str:
    lowered = {k.lower(): v for k, v in infons.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None:
            return value.strip()
    return ""


def normalize_mesh_id(raw_id: str) -> str | None:
    """Normalize one atomic NLM-Chem identifier to a bare MeSH UI."""
    value = raw_id.strip()
    if not value or value == "-1":
        return None

    if ":" in value:
        prefix, code = value.split(":", 1)
        if prefix.strip().upper() in {"MESH", "MSH"}:
            value = code.strip()

    value = value.upper()
    if MESH_UI_RE.fullmatch(value):
        return value
    return None


def parse_mesh_ids(raw_identifier: str) -> tuple[str, ...]:
    """
    Parse a possibly multi-normalized identifier field.

    Examples
    --------
    MESH:D004390 -> (D004390,)
    D010710,MESH:D012965 -> (D010710, D012965)
    D012428,MESH:D011725,MESH:D056831 -> three IDs
    """
    if not raw_identifier.strip():
        return ()

    result: list[str] = []
    seen: set[str] = set()
    for part in MULTI_ID_SPLIT_RE.split(raw_identifier):
        mesh_id = normalize_mesh_id(part)
        if mesh_id is None or mesh_id in seen:
            continue
        seen.add(mesh_id)
        result.append(mesh_id)
    return tuple(result)


def _parse_xml_root(xml_path: Path) -> tuple[ET.Element, bool]:
    """
    Parse BioC XML, falling back to conservative sanitization for malformed XML.

    Returns
    -------
    (root, used_sanitized_reader)
    """
    try:
        return ET.parse(xml_path).getroot(), False
    except ET.ParseError:
        raw = xml_path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        text = INVALID_XML_CONTROL_RE.sub("", text)
        text = BARE_AMPERSAND_RE.sub("&amp;", text)
        try:
            return ET.fromstring(text), True
        except ET.ParseError as exc:
            raise RuntimeError(
                f"Failed to parse NLM-Chem XML even after sanitization: {xml_path}\n{exc}"
            ) from exc


def _passage_type(passage: ET.Element) -> str:
    infons = _infons(passage)
    return _get_infon_case_insensitive(infons, "type", "section_type").lower()


def _extract_identifier(annotation: ET.Element) -> str:
    infons = _infons(annotation)

    value = _get_infon_case_insensitive(
        infons,
        "identifier",
        "mesh",
        "mesh_id",
        "meshid",
    )
    return value.strip()


def _extract_annotation_type(annotation: ET.Element) -> str:
    infons = _infons(annotation)
    return _get_infon_case_insensitive(infons, "type")


def _extract_locations(annotation: ET.Element) -> list[tuple[int, int]]:
    locations: list[tuple[int, int]] = []
    for location in _children_by_tag(annotation, "location"):
        try:
            offset = int(location.attrib["offset"])
            length = int(location.attrib["length"])
        except (KeyError, ValueError):
            continue
        if length < 0:
            continue
        locations.append((offset, length))
    return locations


def _passage_annotations(
    passage: ET.Element,
    doc_id: str,
    chemical_type: str,
) -> list[NlmChemAnnotation]:
    annotations: list[NlmChemAnnotation] = []

    for annotation in _children_by_tag(passage, "annotation"):
        ann_type = _extract_annotation_type(annotation)
        if ann_type.lower() != chemical_type.lower():
            continue

        locations = _extract_locations(annotation)
        if not locations:
            continue

        start = min(offset for offset, _ in locations)
        end = max(offset + length for offset, length in locations)

        mention_text = _child_text(annotation, "text", default="").strip()
        raw_identifier = _extract_identifier(annotation)
        mesh_ids = parse_mesh_ids(raw_identifier)

        annotations.append(
            NlmChemAnnotation(
                doc_id=doc_id,
                annotation_id=annotation.attrib.get("id", ""),
                annotation_type=ann_type,
                start=start,
                end=end,
                mention_text=mention_text,
                raw_identifier=raw_identifier,
                mesh_ids=mesh_ids,
                n_locations=len(locations),
            )
        )

    return annotations


def iter_nlmchem_documents(
    xml_path: Path,
    *,
    chemical_type: str = "Chemical",
) -> tuple[list[NlmChemDocument], bool]:
    root, sanitized = _parse_xml_root(xml_path)
    documents: list[NlmChemDocument] = []

    for document in root.iter():
        if _strip_namespace(document.tag) != "document":
            continue

        doc_id = _child_text(document, "id", default="").strip()
        if not doc_id:
            raise ValueError("Encountered a BioC <document> without <id>.")

        title_parts: list[str] = []
        abstract_parts: list[str] = []
        annotations: list[NlmChemAnnotation] = []

        passages = _children_by_tag(document, "passage")
        for passage in passages:
            passage_text = _child_text(passage, "text", default="").strip()
            ptype = _passage_type(passage)

            if passage_text:
                if ptype == "title":
                    title_parts.append(passage_text)
                else:
                    abstract_parts.append(passage_text)

            annotations.extend(
                _passage_annotations(
                    passage,
                    doc_id=doc_id,
                    chemical_type=chemical_type,
                )
            )

        documents.append(
            NlmChemDocument(
                doc_id=doc_id,
                title=" ".join(title_parts),
                abstract=" ".join(abstract_parts),
                annotations=tuple(annotations),
            )
        )

    return documents, sanitized


def default_output_paths(input_path: Path) -> tuple[Path, Path, Path]:
    name = input_path.name
    if name.endswith(".BioC.xml"):
        base = name[: -len(".BioC.xml")]
    else:
        base = input_path.stem

    full = input_path.with_name(f"{base}.PubTator.txt")
    single = input_path.with_name(f"{base}.PubTator.single.txt")
    report = input_path.with_name(f"{base}.PubTator.preprocess.tsv")
    return full, single, report


def _write_pubtator_document(
    fh,
    document: NlmChemDocument,
    annotations: Iterable[NlmChemAnnotation],
) -> None:
    fh.write(f"{document.doc_id}|t|{document.title}\n")
    fh.write(f"{document.doc_id}|a|{document.abstract}\n")

    for ann in annotations:
        fh.write(
            "\t".join(
                [
                    ann.doc_id,
                    str(ann.start),
                    str(ann.end),
                    ann.mention_text,
                    "Chemical",
                    ann.normalized_gold,
                ]
            )
            + "\n"
        )
    fh.write("\n")


def write_outputs(
    documents: list[NlmChemDocument],
    *,
    output_path: Path,
    single_output_path: Path,
    report_path: Path,
) -> Counter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    single_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()

    with (
        output_path.open("w", encoding="utf-8", newline="") as full_fh,
        single_output_path.open("w", encoding="utf-8", newline="") as single_fh,
        report_path.open("w", encoding="utf-8", newline="") as report_fh,
    ):
        writer = csv.writer(report_fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "doc_id",
                "annotation_id",
                "annotation_type",
                "start",
                "end",
                "mention_text",
                "raw_identifier",
                "normalized_mesh_ids",
                "n_gold_ids",
                "is_multi_normalized",
                "n_locations",
                "included_in_single_eval",
            ]
        )

        for document in documents:
            stats["documents"] += 1
            full_annotations = list(document.annotations)
            single_annotations = [
                ann for ann in full_annotations if ann.is_single_gold
            ]

            _write_pubtator_document(full_fh, document, full_annotations)
            _write_pubtator_document(single_fh, document, single_annotations)

            for ann in full_annotations:
                stats["chemical_annotations"] += 1
                if ann.is_single_gold:
                    stats["single_gold_annotations"] += 1
                elif ann.is_multi_normalized:
                    stats["multi_normalized_annotations"] += 1
                else:
                    stats["missing_or_invalid_gold_annotations"] += 1

                if ann.n_locations > 1:
                    stats["multi_location_annotations"] += 1
                if any(len(mesh_id) == 10 for mesh_id in ann.mesh_ids):
                    stats["extended_mesh_ui_annotations"] += 1

                writer.writerow(
                    [
                        ann.doc_id,
                        ann.annotation_id,
                        ann.annotation_type,
                        ann.start,
                        ann.end,
                        ann.mention_text,
                        ann.raw_identifier,
                        ";".join(ann.mesh_ids),
                        len(ann.mesh_ids),
                        str(ann.is_multi_normalized).lower(),
                        ann.n_locations,
                        str(ann.is_single_gold).lower(),
                    ]
                )

    return stats


def main(argv: list[str] | None = None) -> None:
    args = parse_nlmchem_preprocess_args(argv)

    if not args.input.is_file():
        raise FileNotFoundError(f"Input BioC XML not found: {args.input}")

    default_full, default_single, default_report = default_output_paths(args.input)
    output_path = args.output or default_full
    single_output_path = args.single_output or default_single
    report_path = args.report or default_report

    documents, sanitized = iter_nlmchem_documents(
        args.input,
        chemical_type=args.chemical_type,
    )
    stats = write_outputs(
        documents,
        output_path=output_path,
        single_output_path=single_output_path,
        report_path=report_path,
    )

    print(f"Input: {args.input}")
    print(f"Full PubTator: {output_path}")
    print(f"Single-gold PubTator: {single_output_path}")
    print(f"Preprocess report: {report_path}")
    print(f"Sanitized XML fallback used: {sanitized}")
    for key in (
        "documents",
        "chemical_annotations",
        "single_gold_annotations",
        "multi_normalized_annotations",
        "missing_or_invalid_gold_annotations",
        "extended_mesh_ui_annotations",
        "multi_location_annotations",
    ):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
