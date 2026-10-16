from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.params import parse_metrics_args
from inference import pipeline as _io

PAIR_COLUMNS = [
    "mention_id", "negative_type", "negative_subtype",
    "gold_score", "negative_score", "margin", "confused",
]


def resolve_runs(paths: list[Path]) -> list[Path]:
    """Accept parquet files or run directories; return parquet paths."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            f = p / "pair_scores.parquet"
            if not f.exists():
                print(f"WARNING: no pair_scores.parquet in {p}, skipping")
                continue
            out.append(f)
        elif p.exists():
            out.append(p)
        else:
            print(f"WARNING: {p} not found, skipping")
    return out


def run_identity(parquet: Path) -> dict:
    """Dataset / model / run_tag for one parquet, from run_meta.json or the path."""
    meta_path = parquet.parent / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "dataset": meta.get("dataset", ""),
            "model": meta.get("model_slug", ""),
            "model_spec": meta.get("model_spec", ""),
            "run_tag": meta.get("run_tag", ""),
            "n_scored_mentions": meta.get("n_scored_mentions"),
            "in_domain": meta.get("provenance", {}).get("in_domain"),
            "checkpoint": meta.get("provenance", {}).get("checkpoint", ""),
        }
    parts = parquet.parts
    return {
        "dataset": parts[-3] if len(parts) >= 3 else "",
        "model": parts[-4] if len(parts) >= 4 else "",
        "model_spec": "",
        "run_tag": parts[-2] if len(parts) >= 2 else "",
        "n_scored_mentions": None,
        "in_domain": None,
        "checkpoint": "",
    }


def pair_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    n_conf = int(df["confused"].sum())
    return {
        "n_pairs": n,
        "pairwise_success": (n - n_conf) / n if n else float("nan"),
        "confusion_rate": n_conf / n if n else float("nan"),
        "mean_margin": float(df["margin"].mean()) if n else float("nan"),
        "median_margin": float(df["margin"].median()) if n else float("nan"),
        "p10_margin": float(df["margin"].quantile(0.10)) if n else float("nan"),
    }


def _auc(score, label) -> float:
    """P(score of a survivor > score of a failure), by the rank identity.

    Equivalent to sklearn's roc_auc_score with ties handled by average ranks;
    written out so this module keeps its two-dependency footprint.
    """
    label = pd.Series(label).astype(bool).to_numpy()
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def separation_stats(df: pd.DataFrame, scale: float) -> dict:
    """How close the gold sits to its *nearest* competitor in this bucket.

    ``pair_stats`` averages over every pair, which is dominated by the easy ones:
    a mention can carry twenty negatives, nineteen beaten comfortably, and still
    be one tie away from a wrong prediction. The quantity that decides the
    prediction is the smallest margin over the bucket, per mention.

    Raw margins are not comparable across models -- they are cosines for
    SapBERT (~0.4), un-normalised dot products for BioSyn (~40), and
    length-normalised log-likelihoods for GENRE (~3.8) -- so each run's margins
    are divided by that run's own mean gold-vs-random margin. A normalised
    margin of 1.0 means "as clearly separated as an average random negative";
    0.0 means the decision sits exactly on the boundary. Runs with no random
    bucket, or a non-positive scale, report NaN rather than a meaningless ratio.
    """
    if not scale or scale <= 0 or df.empty:
        return {"median_norm_margin": float("nan"), "p10_norm_margin": float("nan"),
                "share_margin_below_10pct": float("nan"),
                "median_norm_margin_correct": float("nan"),
                "share_below_10pct_correct": float("nan"),
                "share_below_25pct_correct": float("nan")}
    nearest = df.groupby("mention_id", observed=True)["margin"].min() / scale
    won = nearest[nearest > 0]
    return {
        "median_norm_margin": float(nearest.median()),
        "p10_norm_margin": float(nearest.quantile(0.10)),
        "share_margin_below_10pct": float((nearest < 0.10).mean()),
        "median_norm_margin_correct": float(won.median()) if len(won) else float("nan"),
        "share_below_10pct_correct": float((won < 0.10).mean()) if len(won) else float("nan"),
        "share_below_25pct_correct": float((won < 0.25).mean()) if len(won) else float("nan"),
    }


def calibration_stats(df: pd.DataFrame) -> dict:
    """Does the magnitude of the gold's own score predict that it survives?

    ``gold_score_auc`` is the probability that a mention the model gets right
    scored higher than one it gets wrong. 0.5 means the absolute score carries
    no information about reliability; 1.0 means it fully determines it. The two
    quartile columns show that the information is usually one-sided: a low score
    predicts failure far better than a high score predicts success.
    """
    if df.empty:
        return {"gold_score_auc": float("nan"), "survival_q1": float("nan"),
                "survival_q4": float("nan")}
    g = df.groupby("mention_id", observed=True).agg(
        gold=("gold_score", "first"), n_conf=("confused", "sum"))
    g["survived"] = g["n_conf"] == 0
    out = {"gold_score_auc": _auc(g["gold"], g["survived"])}
    if len(g) < 8 or g["gold"].nunique() < 4:
        out.update({"survival_q1": float("nan"), "survival_q4": float("nan")})
        return out
    q = pd.qcut(g["gold"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    surv = g["survived"].groupby(q, observed=False).mean()
    out["survival_q1"] = float(surv.get("Q1", float("nan")))
    out["survival_q4"] = float(surv.get("Q4", float("nan")))
    return out


def mention_stats(df: pd.DataFrame) -> dict:
    """Controlled-candidate-set metrics over the mentions present in ``df``."""
    per_mention = df.groupby("mention_id", observed=True)["confused"].sum()
    if per_mention.empty:
        return {"n_mentions": 0, "top1_rate": float("nan"), "mrr": float("nan"),
                "mean_confusers": float("nan")}
    return {
        "n_mentions": int(len(per_mention)),
        "top1_rate": float((per_mention == 0).mean()),
        "mrr": float((1.0 / (1.0 + per_mention)).mean()),
        "mean_confusers": float(per_mention.mean()),
    }


def margin_scale(df: pd.DataFrame) -> float:
    """The run's own unit of separation: mean gold-vs-random margin.

    Random negatives are the one bucket every run shares and the one whose
    members are, by construction, not confusable. Their mean margin is therefore
    a per-model, per-dataset yardstick for "clearly separated".
    """
    rnd = df[df["negative_type"] == "random"]
    return float(rnd["margin"].mean()) if len(rnd) else float("nan")


def score_range(df: pd.DataFrame) -> tuple[float, float]:
    """Min-max range of every score the run produced, gold and negative pooled.

    Dividing margins by this range puts each backend on a common [0, 1] scale --
    cosines, un-normalised dot products and mean log-likelihoods are otherwise
    two orders of magnitude apart. It is a rescaling of the scores themselves,
    not a new quantity: a normalised margin of 0.01 means the gold and its
    competitor sit one hundredth of the model's whole score range apart.
    """
    lo = float(min(df["gold_score"].min(), df["negative_score"].min()))
    hi = float(max(df["gold_score"].max(), df["negative_score"].max()))
    return lo, hi


def by_score_quartile(df: pd.DataFrame, ident: dict, lo: float, hi: float) -> list[dict]:
    """Margin to the nearest competitor, split by how high the gold itself scored.

    This is the table behind Observation 2. For each negative family we ask
    whether a mention on which the model scores the gold highly is also a mention
    where the gold is safely ahead. Quartiles are cut on ranks because several
    backends emit heavily tied scores.
    """
    rng = hi - lo
    if not rng > 0:
        return []
    rows = []
    gold = (df.groupby("mention_id", observed=True)["gold_score"].first() - lo) / rng
    q = pd.qcut(gold.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    for ntype, sub in df.groupby("negative_type", observed=True):
        near = sub.groupby("mention_id", observed=True)["negative_score"].max()
        near = (near - lo) / rng
        j = pd.DataFrame({"gold": gold, "best_neg": near}).dropna()
        j["margin"] = j["gold"] - j["best_neg"]
        j["q"] = q.reindex(j.index)
        def emit(label: str, g: pd.DataFrame) -> None:
            if g.empty:
                return
            rows.append({**ident, "negative_type": ntype, "score_quartile": label,
                         "n_mentions": int(len(g)),
                         "mean_gold_norm": float(g["gold"].mean()),
                         "mean_best_negative_norm": float(g["best_neg"].mean()),
                         "mean_margin_norm": float(g["margin"].mean()),
                         "median_gold_norm": float(g["gold"].median()),
                         "median_best_negative_norm": float(g["best_neg"].median()),
                         "median_margin_norm": float(g["margin"].median()),
                         "tie_rate": float((g["margin"] == 0).mean()),
                         "top1_rate": float((g["margin"] > 0).mean())})

        for qi, g in j.groupby("q", observed=False):
            emit(str(qi), g)
        emit("ALL", j)
    return rows


def metrics_by(df: pd.DataFrame, key: str, ident: dict, scale: float) -> list[dict]:
    rows = []
    for value, sub in df.groupby(key, observed=True):
        if len(sub) == 0:
            continue
        rows.append({**ident, key: value, **pair_stats(sub), **mention_stats(sub),
                     **separation_stats(sub, scale), **calibration_stats(sub)})
    rows.append({**ident, key: "ALL", **pair_stats(df), **mention_stats(df),
                 **separation_stats(df, scale), **calibration_stats(df)})
    return rows


def main(argv=None) -> int:
    args = parse_metrics_args(argv)
    parquets = resolve_runs(args.pair_scores)
    if not parquets:
        raise SystemExit("ERROR: no pair_scores.parquet resolved from --pair-scores")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_type: list[dict] = []
    by_quartile: list[dict] = []
    by_subtype: list[dict] = []
    by_mention_type: list[dict] = []
    summary: list[dict] = []

    for parquet in parquets:
        ident = run_identity(parquet)
        try:
            df = pd.read_parquet(parquet, columns=PAIR_COLUMNS)
        except Exception as exc:
            print(f"WARNING: skipping {parquet}: {type(exc).__name__}: {exc}")
            continue
        print(f"[{ident['model']} / {ident['dataset']} / {ident['run_tag']}] "
              f"{len(df):,} pairs", flush=True)

        s_lo, s_hi = score_range(df)
        by_quartile.extend(by_score_quartile(df, ident, s_lo, s_hi))
        scale = margin_scale(df)
        if not (scale > 0):
            print(f"  WARNING: no usable random-negative margin scale "
                  f"({scale}); normalised-margin columns will be NaN")
        by_type.extend(metrics_by(df, "negative_type", ident, scale))
        by_subtype.extend(metrics_by(df, "negative_subtype", ident, scale))
        summary.append({**ident, **pair_stats(df), **mention_stats(df),
                        **separation_stats(df, scale), **calibration_stats(df),
                        "random_margin_scale": scale,
                        "score_min": s_lo, "score_max": s_hi,
                        "n_negative_types": int(df["negative_type"].nunique())})

        if args.mentions_root:
            mpath = args.mentions_root / ident["dataset"] / "mentions_test.jsonl"
            if mpath.exists():
                ann = {r["mention_id"]: (r.get("annotation_type") or "")
                       for r in _io.iter_jsonl(mpath)}
                df["mention_type"] = df["mention_id"].map(ann).fillna("")
                for (mt, nt), sub in df.groupby(["mention_type", "negative_type"],
                                                observed=True):
                    by_mention_type.append({
                        **ident, "mention_type": mt, "negative_type": nt,
                        **pair_stats(sub), **mention_stats(sub),
                        **separation_stats(sub, scale), **calibration_stats(sub),
                    })
                for mt, sub in df.groupby("mention_type", observed=True):
                    by_mention_type.append({
                        **ident, "mention_type": mt, "negative_type": "ALL",
                        **pair_stats(sub), **mention_stats(sub),
                        **separation_stats(sub, scale), **calibration_stats(sub),
                    })
                df.drop(columns=["mention_type"], inplace=True)
            else:
                print(f"  WARNING: {mpath} not found; skipping mention-type breakdown")
        del df

    prefix = f"{args.tag}_" if args.tag else ""
    type_order = {t: i for i, t in enumerate(_io.NEG_TYPE_ORDER)}

    def write(rows: list[dict], name: str, sort_cols: list[str]) -> None:
        if not rows:
            return
        out = pd.DataFrame(rows)
        if "negative_type" in out.columns:
            out["_order"] = out["negative_type"].map(
                lambda t: type_order.get(t, len(type_order) + (t == "ALL"))
            )
            sort_cols = sort_cols + ["_order"]
        out = out.sort_values(sort_cols).drop(columns=["_order"], errors="ignore")
        path = args.out_dir / f"{prefix}{name}.csv"
        out.to_csv(path, index=False, float_format="%.6f")
        print(f"wrote {path} ({len(out)} rows)")

    write(by_type, "by_negative_type", ["dataset", "model"])
    write(by_subtype, "by_negative_subtype", ["dataset", "model", "negative_subtype"])
    write(by_mention_type, "by_mention_type", ["dataset", "model", "mention_type"])
    write(by_quartile, "by_score_quartile",
          ["dataset", "model", "score_quartile"])
    write(summary, "summary", ["dataset", "model"])

    if by_type:
        t = pd.DataFrame(by_type)
        for dataset, sub in t.groupby("dataset"):
            pivot = sub.pivot_table(index=["model", "run_tag"], columns="negative_type",
                                    values="top1_rate", observed=True)
            cols = [c for c in _io.NEG_TYPE_ORDER if c in pivot.columns]
            cols += [c for c in pivot.columns if c not in cols]
            pivot = pivot[cols]
            path = args.out_dir / f"{prefix}top1_{dataset}.csv"
            pivot.to_csv(path, float_format="%.4f")
            print(f"\n=== {dataset}: mention-level top-1 vs each negative type ===")
            print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
