# Overview

Inference: score every (mention, gold, negative) triple with one linker, then
aggregate those scores into the diagnostic tables. Reads what
`code/preprocess` wrote, adds nothing to it.

The modules, none of which is run directly:

| | |
|---|---|
| `pipeline.py` | path conventions, the negative taxonomy, and `NegativeStore`, a byte-offset index over the `negatives_*.jsonl` files so nothing loads them whole |
| `scorers.py` | the scoring backends. Each answers one question: given a mention and a set of candidate concepts, how strongly does the model prefer each one |
| `pair_writer.py` | the parquet schema, shared by the backends that need their own driver |

# Scoring pairs
- Goal: for each mention, score the gold concept and every hard negative with
  the same linker, so the two can be compared directly
- Inputs:
    - `mentions_*.jsonl` and the `negatives_*.jsonl` of one benchmark
    - `concepts.pkl`
    - `--model`, see the table below
- Definition, per mention with gold `g` and each negative `n`
    - `gold_score     = max over g's aliases of scorer(mention, alias)`
    - `negative_score = scorer(mention, name(n))`
    - `margin         = gold_score - negative_score`
    - `success        = gold_score > negative_score`, so a tie counts as
      confusion. A linker that cannot separate the gold from a parent concept
      has not distinguished them
- A candidate is a concept, not a string, so its score is the max over its
  aliases, the same "best matching alias wins" rule a deployed linker uses
- Outputs, under `<out-dir>/<model_slug>/<dataset>/<run_tag>/`
    - `pair_scores.parquet`, one row per (mention, negative): `gold_score`,
      `negative_score`, `margin`, `success`, `confused`, `negative_type`,
      `negative_subtype`, `gold_best_alias`
    - `run_meta.json`, the checkpoint, `in_domain`, and the pair and exclusion
      counts
    - `excluded_mentions.jsonl`, mentions whose gold is absent from the KB or
      has no alias

- run the script
```bash
python code/inference/score_pairs.py \
--dataset "ncbi_disease" \
--dataset-dir "your_output_path/ncbi_disease" \
--concepts "your_kb_path/concepts.pkl" \
--out-dir "your_output_path/scores" \
--model "sapbert" \
--device "cuda"
```

- `--model` values
    - `bm25[:k1=1.5,b=0.75]` sparse, CPU only
    - `sapbert`, or any Hugging Face id or path, for a dense alias encoder
    - `biosyn[:<hf_id>,alpha=0.5]`, `biosyn-auto` for the per-dataset checkpoint
    - `crossencoder:<hf_id>`, `arboel:<ckpt>`
    - `angel[:<hf_id>]`, `genbioel:<path>`, `genre[:<hf_id>]`
- Generative linkers are scored by forced likelihood, not by constrained
  decoding: `score(m, e) = 1/|a| * sum_t log P(a_t | source(m), a_<t)`,
  maximised over `e`'s aliases. A trie built from a benchmark dictionary cannot
  emit a negative that is absent from it, which is the blind spot the paper
  probes, so the candidates have to be chosen by us rather than by the trie
- optional: `--run-tag` (default `full`), `--neg-kinds` to restrict the negative
  families, `--gold-name-mode` (`max_alias` or `preferred`), `--neg-name-mode`
  (`given` or `max_alias`), `--max-gold-aliases` (default 32),
  `--max-neg-aliases` (default 8), `--mention-subset`, `--batch-size`,
  `--pooling`, `--length-norm`, `--context-window`

# Aggregating metrics
- Goal: turn the pair scores into the diagnostic tables, at four levels
- Inputs: one or more `pair_scores.parquet`, optionally `--mentions-root` for
  the per-mention-type breakdown
- Outputs, in `--out-dir`
    - `by_negative_type.csv`, one row per (dataset, model, negative_type), plus
      a pooled `ALL` row that puts every negative family into one candidate set
    - `by_negative_subtype.csv`, the finer buckets: the six ontology relations,
      the per-TUI semantic types
    - `by_mention_type.csv`, only with `--mentions-root`
    - `summary.csv`, one row per run
- What is in them
    - pairwise: `n_pairs`, `pairwise_success`, `confusion_rate`,
      `mean` / `median` / `p10_margin`
    - mention level: `top1_rate`, the fraction of mentions where the gold beats
      every negative of that family, and `mrr`
    - separation: how close the decision sits to the boundary, in units of the
      run's own mean gold-vs-random margin. `median_norm_margin`,
      `share_margin_below_10pct`, and the `*_correct` variants, which condition
      on the model winning the mention and so measure how fragile its successes
      are rather than how many failures it has
    - calibration: `gold_score_auc`, `survival_q1`, `survival_q4`, whether the
      magnitude of the gold's own score predicts that it survives

- run the script
```bash
python code/inference/metrics.py \
--pair-scores "your_output_path/scores/*/ncbi_disease/full" \
--mentions-root "your_output_path" \
--out-dir "your_output_path/metrics"
```

- Raw margins are not comparable across models: cosines for the alias
  encoders, un-normalised dot products for BioSyn, length-normalised
  log-likelihoods for the generative ones. Compare margins within a model, and
  across models use `top1_rate`, `mrr`, or the normalised margin columns
- Check `in_domain` in `run_meta.json` before quoting a row. A checkpoint
  transferred from the nearest domain is flagged there
