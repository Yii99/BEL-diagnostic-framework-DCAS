from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.utils import alias_list

PAIR_FIELDS = (
    "mention_id", "mention_text", "gold_entity_id", "n_gold_anchors",
    "negative_entity_id", "negative_type", "negative_subtype", "negative_name",
    "gold_score", "negative_score", "margin", "success", "confused", "gold_best_alias",
)


def base_schema(extra: dict | None = None):
    """The 07 parquet schema, optionally widened with backend-specific columns."""
    import pyarrow as pa

    fields = [
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
    ]
    if extra:
        fields += list(extra.items())
    return pa.schema(fields)


class PairWriter:
    """Buffered parquet writer: one row per (mention, negative) pair.

    Ties count as confusion, matching 07: `success = margin > 0`.
    """

    def __init__(self, path: Path, extra: dict | None = None, flush_mentions: int = 500):
        import pyarrow.parquet as pq

        self.path = path
        self.extra_names = list(extra) if extra else []
        self.schema = base_schema(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(path, self.schema, compression="zstd")
        self._flush_mentions = flush_mentions
        self._reset()
        self.n_rows = 0
        self.n_confused = 0
        self._since_flush = 0

    def _reset(self) -> None:
        self._buf = {name: [] for name in list(PAIR_FIELDS) + self.extra_names}

    def add(self, *, mention_id: str, mention_text: str, gold_entity_id: str,
            n_gold_anchors: int, gold_score: float, gold_best_alias: str,
            neg_ids: list[str], neg_names: list[str], neg_types: list[str],
            neg_subtypes: list[str], neg_scores: np.ndarray,
            extra: dict[str, list] | None = None) -> None:
        k = len(neg_ids)
        if k == 0:
            return
        scores = np.asarray(neg_scores, dtype=np.float32)
        margins = np.float32(gold_score) - scores
        b = self._buf
        b["mention_id"].extend([mention_id] * k)
        b["mention_text"].extend([mention_text] * k)
        b["gold_entity_id"].extend([gold_entity_id] * k)
        b["n_gold_anchors"].extend([n_gold_anchors] * k)
        b["negative_entity_id"].extend(neg_ids)
        b["negative_type"].extend(neg_types)
        b["negative_subtype"].extend(neg_subtypes)
        b["negative_name"].extend(neg_names)
        b["gold_score"].extend([float(gold_score)] * k)
        b["negative_score"].extend(float(x) for x in scores)
        b["margin"].extend(float(x) for x in margins)
        success = margins > 0
        b["success"].extend(bool(x) for x in success)
        b["confused"].extend(bool(not x) for x in success)
        b["gold_best_alias"].extend([gold_best_alias] * k)
        for name in self.extra_names:
            vals = (extra or {}).get(name)
            if vals is None:
                raise KeyError(f"extra column '{name}' missing for {mention_id}")
            if len(vals) != k:
                raise ValueError(f"extra column '{name}' has {len(vals)} values, expected {k}")
            b[name].extend(vals)
        self.n_rows += k
        self.n_confused += int((~success).sum())
        self._since_flush += 1
        if self._since_flush >= self._flush_mentions:
            self.flush()

    def flush(self) -> None:
        import pyarrow as pa

        if self._buf["mention_id"]:
            self._writer.write_table(pa.table(self._buf, schema=self.schema))
            self._reset()
        self._since_flush = 0

    def close(self) -> None:
        self.flush()
        self._writer.close()

    @property
    def pairwise_success_rate(self) -> float | None:
        return (self.n_rows - self.n_confused) / self.n_rows if self.n_rows else None


def load_concepts_slim(dataset_dir: Path) -> dict:
    """Load the per-benchmark concept cache written by 07 --slim-cache-only."""
    path = dataset_dir / "concepts_slim.pkl"
    if not path.exists():
        raise SystemExit(
            f"ERROR: {path} not found. Build it once with\n"
            f"  python code/inference/score_pairs.py --dataset <name> "
            f"--dataset-dir {dataset_dir} --concepts <kb>/concepts.pkl "
            f"--out-dir outputs/scores --slim-cache-only"
        )
    with path.open("rb") as f:
        return pickle.load(f)


def gold_alias_groups(rec: dict, concepts: dict, mode: str, cap: int
                      ) -> tuple[list[str], list[list[str]]]:
    """Return (anchor entity ids, alias group per anchor) for the gold concept."""
    from inference import pipeline as _io

    ids: list[str] = []
    groups: list[list[str]] = []
    for eid in _io.anchor_entity_ids(rec):
        crec = concepts.get(eid)
        if crec is None:
            continue
        names = alias_list(crec, cap if mode == "max_alias" else 1)
        if not names:
            continue
        ids.append(eid)
        groups.append(names)
    return ids, groups


def select_mentions(mentions: list[dict], *, sample: int = 0, sample_docs: int = 0,
                    limit: int = 0, seed: int = 17, subset: Path | None = None) -> list[dict]:
    """Pick the mentions to score.

    A `subset` file (one mention_id per line) wins over any sampling: two scripts
    with different RNG paths would otherwise draw different 2000-mention samples
    from the same seed, which quietly makes their numbers incomparable.
    """
    if subset is not None:
        wanted = {ln.strip() for ln in Path(subset).read_text().splitlines() if ln.strip()}
        mentions = [m for m in mentions if m["mention_id"] in wanted]
        if not mentions:
            raise SystemExit(f"ERROR: no mention matched the subset file {subset}")
        return mentions[:limit] if limit else mentions
    rng = random.Random(seed)
    if sample_docs:
        doc_ids = sorted({m["doc_id"] for m in mentions})
        if sample_docs < len(doc_ids):
            keep = set(rng.sample(doc_ids, sample_docs))
            mentions = [m for m in mentions if m["doc_id"] in keep]
    elif sample and sample < len(mentions):
        idx = sorted(rng.sample(range(len(mentions)), sample))
        mentions = [mentions[i] for i in idx]
    if limit:
        mentions = mentions[:limit]
    return mentions


def write_run_meta(out_dir: Path, meta: dict) -> None:
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                          encoding="utf-8")
