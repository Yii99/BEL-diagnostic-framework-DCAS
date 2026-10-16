from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_umls_coverage_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for UMLS MRCONSO coverage analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute how many benchmark gold entity IDs can be mapped to UMLS CUIs "
            "through MRCONSO.RRF. Supports BC5CDR, NCBI-Disease, MedMentions, "
            "and NLM-Chem."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["bc5cdr", "ncbi", "medmentions", "nlmchem"],
        help=(
            "Dataset to parse."
        ),
    )

    parser.add_argument(
        "--data",
        required=True,
        help=(
            "Dataset path. This can be a dataset root directory or a single file. "
            "For MedMentions ST21pv, pass the st21pv root directory containing "
            "data/corpus_pubtator.txt and corpus_pubtator_pmids_* files."
        ),
    )

    parser.add_argument(
        "--kb",
        required=True,
        help="Path to UMLS META/MRCONSO.RRF.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for coverage_summary.tsv, mapping_detail.tsv, and unmapped_ids.tsv.",
    )

    parser.add_argument(
        "--lang",
        default="ENG",
        help="Language used when counting names / aliases from MRCONSO. Default: ENG.",
    )

    parser.add_argument(
        "--include-suppressed",
        action="store_true",
        help=(
            "Count suppressed MRCONSO strings when reporting alias/name statistics. "
            "Mapping itself is still based on source IDs; this flag mainly affects "
            "name counts."
        ),
    )

    parser.add_argument(
        "--include-no-gold-id",
        action="store_true",
        help=(
            "Include no-normalization annotations such as -1 in mapping_detail.tsv. "
            "They are always excluded from coverage denominators."
        ),
    )

    return parser

def build_pubtator_umls_mapping_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for PubTator -> UMLS anchor mapping."""
    parser = argparse.ArgumentParser(
        description="Append UMLS CUI anchor(s) to PubTator annotation lines."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("bc5cdr", "ncbi", "nlmchem", "medmentions", "generic"),
        help=(
            "Controls namespace inference for bare gold IDs. "
            "bc5cdr: D/C IDs -> MeSH; ncbi: D/C -> MeSH and numeric -> OMIM; "
            "nlmchem: ..."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input PubTator file.",
    )
    parser.add_argument(
        "--mrconso",
        type=Path,
        required=True,
        help="Path to UMLS META/MRCONSO.RRF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PubTator file. Default: insert '.umls' before input suffix.",
    )
    parser.add_argument(
        "--mapping-tsv",
        type=Path,
        default=None,
        help="Sidecar mapping table. Default: <output>.mapping.tsv",
    )
    parser.add_argument(
        "--gold-col",
        type=int,
        default=5,
        help="0-based PubTator annotation column containing gold ID (default: 5).",
    )
    parser.add_argument(
        "--mesh-code-fallback",
        action="store_true",
        help=(
            "If MeSH expected-TTY mapping is empty, use all SAB=MSH + CODE matches. "
            "Off by default because precision is preferred for relational anchors."
        ),
    )
    parser.add_argument(
        "--append-mode",
        choices=("column", "same-field"),
        default="column",
        help=(
            "'column': add a new tab-separated UMLS field (recommended). "
            "'same-field': append ' UMLS:...' to the original gold field."
        ),
    )
    return parser


def parse_pubtator_umls_mapping_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments for PubTator -> UMLS anchor mapping."""
    return build_pubtator_umls_mapping_parser().parse_args(argv)

def build_nlmchem_preprocess_parser() -> argparse.ArgumentParser:
    """Build CLI parser for NLM-Chem BioC XML -> PubTator preprocessing."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert NLM-Chem BioC XML to PubTator, normalize MeSH IDs, "
            "and create a single-gold evaluation file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input NLM-Chem BioC XML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Full PubTator output. Default: <input-base>.PubTator.txt",
    )
    parser.add_argument(
        "--single-output",
        type=Path,
        default=None,
        help=(
            "PubTator output containing only annotations with exactly one "
            "valid MeSH gold ID. Default: <input-base>.PubTator.single.txt"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Annotation-level preprocessing report TSV.",
    )
    parser.add_argument(
        "--chemical-type",
        default="Chemical",
        help="BioC annotation type kept as mention-level chemical (default: Chemical).",
    )
    return parser


def parse_nlmchem_preprocess_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments for NLM-Chem preprocessing."""
    return build_nlmchem_preprocess_parser().parse_args(argv)


def _add_common_negative_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--concepts", type=Path, required=True, help="Path to concepts.pkl.")
    p.add_argument("--mentions", type=Path, required=True, help="Per-mention JSONL from step 02.")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    p.add_argument("--seed", type=int, default=17)


def parse_build_umls_concept_dict_args(argv=None):
    p = argparse.ArgumentParser(description="Build RelConBEL UMLS concept dictionary.")
    p.add_argument("--umls-meta", type=Path, required=True, help="UMLS META directory containing MRCONSO/MRSTY/MRREL RRF files.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output KB directory (concepts.pkl is written here).")
    p.add_argument("--source-equivalence-sabs", nargs="+", default=["MSH", "OMIM"], help="SABs retained for source-level equivalence filtering.")
    p.add_argument("--siblings-via-parent-cap", type=int, default=200)
    return p.parse_args(argv)


def parse_benchmark_args(argv=None):
    p = argparse.ArgumentParser(description="Parse test benchmark PubTator into unified RelConBEL mention JSONL.")
    p.add_argument("--dataset", choices=["bc5cdr", "ncbi", "nlmchem", "medmentions", "generic"], required=True)
    p.add_argument("--input", type=Path, required=True, help="PubTator input. BC5CDR/NCBI/NLM-Chem should already contain appended UMLS field.")
    p.add_argument("--concepts", type=Path, required=True, help="UMLS concepts.pkl from step 01.")
    p.add_argument("--source-to-cuis", type=Path, default=None, help="Optional source_to_cuis.pkl from step 01. Defaults beside concepts.pkl.")
    p.add_argument("--output", type=Path, required=True, help="Full mention JSONL output.")
    p.add_argument("--pmids-file", type=Path, default=None, help="Optional PMID allow-list, mainly for MedMentions test split.")
    p.add_argument("--pilot-output", type=Path, default=None)
    p.add_argument("--pilot-size", type=int, default=0, help="0 disables pilot sampling.")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--keep-unmapped", action="store_true", help="Keep mentions with no UMLS anchors. Default: drop them from output.")
    return p.parse_args(argv)


def parse_lexical_negatives_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build lexical negatives for UMLS or NCBI Gene, TF-IDF cosine or Sorensen-Dice.")
    _add_common_negative_args(p)
    p.add_argument("--metric", choices=("tfidf", "dice"), default="tfidf",
                   help="tfidf: char_wb 3-5 grams, TF-IDF, L2 normalised. "
                        "dice: char_wb bigrams, binary, 2|AnB|/(|A|+|B|).")
    p.add_argument("--index", type=Path, default=None,
                   help="Index cache path. Defaults beside concepts.pkl, one file per metric.")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--retrieve-k", type=int, default=400)
    p.add_argument("--max-aliases-per-entity", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args(argv)


def parse_ontological_negatives_args(argv=None):
    p = argparse.ArgumentParser(description="Build UMLS relation-conditioned negatives from one or more gold anchor CUIs.")
    _add_common_negative_args(p)
    p.add_argument("--cap", type=int, default=20, help="Maximum candidates per relation bucket.")
    return p.parse_args(argv)


def parse_random_negatives_args(argv=None):
    p = argparse.ArgumentParser(description="Build generic random negatives for UMLS or NCBI Gene.")
    _add_common_negative_args(p)
    p.add_argument("--num", type=int, default=20)
    p.add_argument("--exclude-negatives", type=Path, action="append", default=[], help="Existing negative JSONL to exclude; may be repeated.")
    p.add_argument("--match-taxon", action="store_true", help="For Gene KB, sample from the gold gene's taxon when available.")
    return p.parse_args(argv)


def parse_lexonto_negatives_args(argv=None, description=None):
    """Controlled lexical/ontological negatives. Batch interface over --datasets."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    p.add_argument("--concepts", type=Path, default=ROOT / "outputs" / "kb" / "umls_2020AA" / "concepts.pkl")
    p.add_argument("--hi", type=float, default=0.6, help="Dice threshold for 'lexically similar'")
    p.add_argument("--lo", type=float, default=0.3, help="Dice ceiling for 'not lexically similar'")
    p.add_argument("--k", type=int, default=4, help="max negatives per bucket per mention")
    p.add_argument("--max-aliases", type=int, default=8, help="aliases per concept scanned for max-Dice")
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args(argv)


def parse_embedding_negatives_args(argv=None, description=None):
    """Embedding negatives retrieved by the mention surface form.

    Builds the BioBERT entity index on first use and caches it under
    --index-dir, then retrieves against it.
    """
    p = argparse.ArgumentParser(description=description)
    _add_common_negative_args(p)
    p.add_argument("--index-dir", type=Path, default=None,
                   help="Entity index cache. Default: <concepts-dir>/embedding_index_biobert.")

    p.add_argument("--model", default="dmis-lab/biobert-base-cased-v1.2")
    p.add_argument("--revision", default="main",
                   help="Model revision; pin a commit hash for final experiments. "
                        "Part of the index cache key.")
    p.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    p.add_argument("--device", default="cuda")

    p.add_argument("--text-mode", choices=("preferred", "preferred_aliases"),
                   default="preferred_aliases",
                   help="Entity text: the preferred name alone, or the preferred name "
                        "and up to --max-aliases aliases joined by the tokenizer's SEP.")
    p.add_argument("--max-aliases", type=int, default=3)
    p.add_argument("--index-max-length", type=int, default=64,
                   help="Token budget for the entity text.")
    p.add_argument("--encode-batch-size", type=int, default=256)
    p.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--rebuild-index", action="store_true")

    p.add_argument("--max-length", type=int, default=64,
                   help="Token budget for the mention surface form.")
    p.add_argument("--query-batch-size", type=int, default=256)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--retrieve-k", type=int, default=100,
                   help="Fetch this many before filtering gold and equivalents.")
    p.add_argument("--search-chunk", type=int, default=1_000_000,
                   help="Entities per search chunk, trades memory for passes.")
    return p.parse_args(argv)


def build_medmentions_test_split_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract the MedMentions test split from corpus_pubtator.txt "
            "using a file containing test PMIDs."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the full MedMentions corpus_pubtator.txt.",
    )
    parser.add_argument(
        "--pmids",
        required=True,
        help="Path to the test PMID list, one PMID per line.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PubTator file containing only test documents.",
    )
    return parser


def parse_medmentions_test_split_args(argv=None):
    return build_medmentions_test_split_parser().parse_args(argv)


def build_score_pairs_parser() -> argparse.ArgumentParser:
    """CLI for 07_score_pairs.py: score (mention, gold, negative) triples."""
    p = argparse.ArgumentParser(
        description=(
            "Score every (mention, gold, hard-negative) triple with one baseline "
            "and write a pair-level parquet."
        )
    )
    p.add_argument("--dataset", required=True,
                   help="Benchmark name, used for output paths and for picking "
                        "dataset-specific checkpoints (e.g. bc5cdr_disease).")
    p.add_argument("--dataset-dir", type=Path, required=True,
                   help="Directory holding mentions_test.jsonl and negatives_*.jsonl.")
    p.add_argument("--concepts", type=Path, required=True, help="KB concepts.pkl from step 01.")
    p.add_argument("--mentions", type=Path, default=None,
                   help="Mention JSONL. Default: <dataset-dir>/mentions_test.jsonl.")
    p.add_argument("--mention-subset", type=Path, default=None,
                   help="Optional file of mention_ids (one per line): score only these mentions, "
                        "so sampled baselines share the exact same subset "
                        "(see inference/make_mention_subsets.py).")
    p.add_argument("--neg-kinds", nargs="*", default=None,
                   help="Negative kinds to score (file suffixes of negatives_<kind>.jsonl). "
                        "Default: every kind present in --dataset-dir.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Score root; results land in <out-dir>/<model_slug>/<dataset>/<run-tag>/.")
    p.add_argument("--model", default=None,
                   help="Backend spec: bm25 | sapbert | biosyn[:hf,alpha=] | arboel:<ckpt> | "
                        "angel[:hf] | genbioel:<path> | genre[:hf] | crossencoder:<hf> | <hf_id>. "
                        "Required unless --slim-cache-only.")
    p.add_argument("--model-slug", default=None, help="Override the derived output directory name.")
    p.add_argument("--run-tag", default="full",
                   help="Subdirectory for this run, e.g. full or pilot2000. Default: full.")

    p.add_argument("--max-per-bucket", type=int, default=0,
                   help="Cap negatives kept per bucket per mention (0 = keep all).")
    p.add_argument("--sample-mentions", type=int, default=0,
                   help="Randomly sample this many mentions (0 = all).")
    p.add_argument("--limit", type=int, default=0, help="Debug: stop after N mentions.")
    p.add_argument("--seed", type=int, default=17)

    p.add_argument("--gold-name-mode", choices=("max_alias", "preferred"), default="max_alias",
                   help="max_alias: gold score is the best-scoring gold alias. "
                        "preferred: only the preferred name, symmetric with negatives.")
    p.add_argument("--max-gold-aliases", type=int, default=32,
                   help="Alias cap per gold anchor concept under max_alias.")
    p.add_argument("--neg-name-mode", choices=("given", "max_alias"), default="given",
                   help="given: score the name the negative builder emitted. "
                        "max_alias: best-scoring alias of the negative concept.")
    p.add_argument("--max-neg-aliases", type=int, default=8,
                   help="Alias cap per negative concept under max_alias.")

    p.add_argument("--device", default=None, help="cuda | cpu. Default: cuda when available.")
    p.add_argument("--batch-size", type=int, default=256, help="Encoder batch size.")
    p.add_argument("--max-length", type=int, default=32, help="Max tokens per name/alias.")
    p.add_argument("--pooling", choices=("cls", "mean"), default="cls")

    p.add_argument("--gen-batch-size", type=int, default=32,
                   help="Candidates scored per decoder forward pass.")
    p.add_argument("--max-source-length", type=int, default=256)
    p.add_argument("--max-target-length", type=int, default=48)
    p.add_argument("--length-norm", choices=("mean", "sum"), default="mean",
                   help="mean matches beam search with normalize_scores=True.")
    p.add_argument("--context-window", type=int, default=0,
                   help="Characters of context kept each side of the mention "
                        "(0 = whole document).")

    p.add_argument("--flush-mentions", type=int, default=2000,
                   help="Mentions per parquet row group.")
    p.add_argument("--overwrite", action="store_true", help="Rescore even if the parquet exists.")
    p.add_argument("--slim-cache-only", action="store_true",
                   help="Build <dataset-dir>/concepts_slim.pkl and exit. Lets a CPU job do the "
                        "one expensive concepts.pkl load that all later GPU runs reuse.")
    return p


def parse_score_pairs_args(argv=None):
    args = build_score_pairs_parser().parse_args(argv)
    if args.model is None and not args.slim_cache_only:
        raise SystemExit("ERROR: --model is required (or pass --slim-cache-only)")
    if args.mentions is None:
        args.mentions = args.dataset_dir / "mentions_test.jsonl"
    if args.device is None:
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    return args


def build_metrics_parser() -> argparse.ArgumentParser:
    """CLI for 08_metrics.py: aggregate pair scores into diagnostic tables."""
    p = argparse.ArgumentParser(
        description="Aggregate pair-level scores into per-negative-type diagnostic tables."
    )
    p.add_argument("--pair-scores", type=Path, nargs="+", required=True,
                   help="pair_scores.parquet files, or run directories containing one.")
    p.add_argument("--out-dir", type=Path, required=True, help="Where the CSV tables are written.")
    p.add_argument("--mentions-root", type=Path, default=None,
                   help="outputs/ root holding <dataset>/mentions_test.jsonl. When given, "
                        "metrics are also broken down by mention annotation_type.")
    p.add_argument("--tag", default="",
                   help="Optional filename prefix, e.g. pilot.")
    return p


def parse_metrics_args(argv=None):
    return build_metrics_parser().parse_args(argv)
