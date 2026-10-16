# Overview

Preprocessing: turn raw source data into the files the rest of the pipeline reads.

# Building UMLS concept dictionary
- Goal: flatten UMLS into a `{cui: record}` dictionary plus reverse indexes
- Inputs:
    - UMLS raw files, `MRCONSO.RRF`, `MRSTY.RRF`, `MRREL.RRF`, `MRDEF.RRF`
    - files are stored under `data/KB/umls/`, we use 2020AA and 2017AA version
    - `MRDEF.RRF` is optional, it only fills `definition`
- Outputs
    - `concepts.pkl`, one record per CUI: preferred name, aliases, TUIs,
      semantic types, definition, source IDs, and the relation buckets
      parents / children / broader / narrower / siblings / siblings_via_parent
    - `source_to_cuis.pkl`, `"MSH:D003924" -> [cui, ...]`
    - `tui_to_cuis.pkl`, semantic type -> CUIs
    - `tui_to_sty.json`, TUI -> semantic type name
    - `concepts.meta.json`, entity count, source path, SABs
    - `concepts_sample.json`, first 20 records

- run the script
```bash
python code/preprocess/build_concept_dict.py \
--umls-meta "your_data_path/umls/${UMLS_VERSION}/META" \
--out-dir "your_output_path/"
```

# Preprocessing NLM-Chem
- Goal: NLM-Chem ships BioC XML only, the rest of the pipeline wants PubTator. We convert it to fit PubTator version.
- Inputs:
    - `BC7T2-NLMChem-corpus-{train,dev,test}.BioC.xml`
    - files are stored under `data/NLMChem/`
    - only annotations of type `Chemical` are kept, see `--chemical-type`
- Outputs, all written next to the input
    - `<base>.PubTator.txt`, every Chemical annotation. An annotation with
      several normalized IDs keeps them all, comma separated
    - `<base>.PubTator.single.txt`, only annotations with exactly one valid
      MeSH ID. This is the file the identity evaluation uses
    - `<base>.PubTator.preprocess.tsv`, audit table: raw ID, normalized IDs,
      multi-normalization flag, span count
    - counts printed at the end: documents, chemical / single-gold /
      multi-normalized / invalid-gold / extended-UI / multi-location annotations

- run the script
```bash
python code/preprocess/preprocess_nlmchem.py \
--input "your_data_path/NLMChem/BC7T2-NLMChem-corpus-test.BioC.xml"
```

- optional: `--output`, `--single-output`, `--report`, `--chemical-type`
  (default `Chemical`)
- then feed `.PubTator.single.txt` to the mapping step below. MeSH UIs are
  dataset independent there, so `--dataset bc5cdr` and `nlmchem` both work

# Mapping PubTator gold IDs to UMLS
- Goal: benchmarks label mentions with MeSH / OMIM IDs, not CUIs. Append the
  matching UMLS CUI to each annotation line so everything lives in one ID space
- Inputs:
    - a PubTator file, gold ID is read from column 5 by default
    - `MRCONSO.RRF` from the same UMLS version
    - `--dataset` controls how bare IDs are read: `bc5cdr` D/C -> MeSH,
      `ncbi` D/C -> MeSH and numeric -> OMIM, also `nlmchem` and `generic`
- Outputs
    - the same PubTator file with a UMLS field appended, default name inserts
      `.umls` before the suffix
    - `<output>.mapping.tsv`, one row per source ID: namespace, source_id,
      mention_count, mapping_status, mapping_method, n_cui, umls_cuis
    - status is `mapped_unique` / `mapped_multiple` / `unmapped`, printed as a
      summary at the end

- run the script
```bash
python code/preprocess/map_pubtator_to_umls.py \
--dataset "bc5cdr" \
--input "your_data_path/pubtator_file.txt" \
--mrconso "your_data_path/umls/${UMLS_VERSION}/META/MRCONSO.RRF"
```

- optional: `--output`, `--mapping-tsv`, `--gold-col` (default 5),
  `--append-mode` (`column` adds a new tab field, `same-field` appends to the
  gold field), `--mesh-code-fallback` (loosen MeSH matching, off for precision)

# Parsing benchmarks into mentions
- Goal: turn benchmarks into mention level. 
- Inputs:
    - the `.umls.txt` from the mapping step. MedMentions use its raw `corpus_pubtator.txt`
    - `concepts.pkl` for `tuis` / `semantic_types`
    - `source_to_cuis.pkl` to expand the gold into its equivalent CUIs. Because we want to exclude the gold CUIs in our diagnostic negative sets.
- Outputs
    - `mentions_*.jsonl`, one line record a mention with `mention_id` (`pmid:start:end:idx`),
      `mention`, `start`, `end`, `annotation_type`, `gold_cuis`,
      `gold_equivalent_cuis`, `tuis`, `semantic_types`, `mapping_status`, and
      the rebuilt context `sentence` / `title` / `abstract`

- run the script
```bash
python code/preprocess/parse_benchmark.py \
--dataset "bc5cdr" \
--input "your_data_path/pubtator_file.umls.txt" \
--concepts "your_kb_path/concepts.pkl" \
--source-to-cuis "your_kb_path/source_to_cuis.pkl" \
--output "your_output_path/mentions_test.jsonl"
```



# Constructing negatives
Each builder reads `mentions_*.jsonl` from the previous step and writes one
`negatives_*.jsonl` next to it, one record per mention with the same shape:
`mention_id`, `mention`, `doc_id`, `gold_entity_ids` and a `negatives` payload
holding `entity_id_list` / `cui_list` plus `name_list`. The gold and its
equivalent CUIs are always excluded.

## Lexical
- Goal: to select candidates that look like the mention but are a different concept.
- Inputs:
    - `mentions_*.jsonl`, `concepts.pkl`
    - `--metric tfidf` char_wb 3-5 grams, TF-IDF, L2 normalised, so the sparse
      product is the cosine. `--metric dice` char_wb bigrams, binary,
      `2|AnB|/(|A|+|B|)`
- Outputs
    - `negatives_lexical_tfidf.jsonl` or `negatives_lexical_dice.jsonl`, with a
      `score_list` in descending order
    - an alias index cached beside `concepts.pkl`, one file per metric. It is
      rebuilt only when `concepts.pkl` changes or with `--rebuild-index`

- run the script
```bash
python code/preprocess/lexical_negatives.py \
--metric "tfidf" \
--concepts "your_kb_path/concepts.pkl" \
--mentions "your_output_path/mentions_test.jsonl" \
--output "your_output_path/negatives_lexical_tfidf.jsonl"
```

- optional: `--top-k` (default 20), `--retrieve-k` (default 400),
  `--max-aliases-per-entity` (default 10), `--batch-size`, `--index`

## Ontological
- Goal: to select candidates that are related to the gold in the UMLS hierarchy but are
  not the same concept
- Inputs: `mentions_*.jsonl`, `concepts.pkl`
- Outputs
    - `negatives_ontological.jsonl`. `negatives` here is six buckets: `parents`, `children`, `broader`, `narrower`, `siblings`,
      `siblings_via_parent`
    - each bucket also carries `via_anchor_list`, which gold anchor it came from

- run the script
```bash
python code/preprocess/ontological_negatives.py \
--concepts "your_kb_path/concepts.pkl" \
--mentions "your_output_path/mentions_test.jsonl" \
--output "your_output_path/negatives_ontological.jsonl"
```

- optional: `--cap` (default 20) candidates per bucket, `--seed`

## Embedding
- Goal: to select candidates whose names embed close to what the author wrote.
  Both sides use the same BioBERT and are L2 normalised, so the inner product
  is the cosine. The query is the mention surface form, the same string the
  lexical builders query with. The entity side is the entity name: the
  preferred name, plus up to `--max-aliases` aliases joined by the tokenizer's
  SEP token under `--text-mode preferred_aliases`
- Inputs:
    - `mentions_*.jsonl`, `concepts.pkl`
- Outputs
    - `negatives_embedding_mention.jsonl`, with a `score_list` in descending
      order
    - the entity index, cached under `--index-dir`, by default
      `<concepts-dir>/embedding_index_biobert`: `entity_ids.txt`,
      `embeddings.npy`, `metadata.json`. Built on first use and reused until
      `concepts.pkl` or an encoder setting changes, or with `--rebuild-index`
- Querying by the mention and not by the gold concept keeps this bucket
  comparable with the lexical ones, and lets two mentions of the same gold
  concept get different negatives. Unique surface forms are encoded once
- This is the only script that needs BioBERT, so the index is built here rather
  than in a separate step. Needs a GPU, `torch` and `transformers`

- run the script
```bash
python code/preprocess/embedding_negatives.py \
--concepts "your_kb_path/concepts.pkl" \
--mentions "your_output_path/mentions_test.jsonl" \
--output "your_output_path/negatives_embedding_mention.jsonl"
```

- optional, entity index: `--text-mode`, `--max-aliases` (default 3),
  `--index-max-length` (default 64), `--pooling` (default mean),
  `--encode-batch-size`, `--storage-dtype`, `--rebuild-index`, `--index-dir`
- optional, query and retrieval: `--top-k` (default 20), `--retrieve-k`
  (default 100), `--max-length` (default 64), `--query-batch-size`,
  `--search-chunk`
- optional, encoder: `--model`, `--revision` (part of the index cache key, pin a
  commit hash for final experiments), `--device`

## Random
- Goal: the control bucket, unrelated concepts. By construction not confusable,
  so it is also the unit of separation the margin is normalised by
- Inputs:
    - `mentions_*.jsonl`, `concepts.pkl`
    - `--exclude-negatives`, repeat it for every negative file already built,
      so a random draw cannot collide with one of them
- Outputs
    - `negatives_random.jsonl`
- Run this last, it depends on the other builders' output

- run the script
```bash
python code/preprocess/random_negatives.py \
--concepts "your_kb_path/concepts.pkl" \
--mentions "your_output_path/mentions_test.jsonl" \
--output "your_output_path/negatives_random.jsonl" \
--exclude-negatives "your_output_path/negatives_lexical_tfidf.jsonl" \
--exclude-negatives "your_output_path/negatives_lexical_dice.jsonl" \
--exclude-negatives "your_output_path/negatives_ontological.jsonl"
```

- optional: `--num` (default 20), `--seed`

## Lexical-ontological disentanglement
- Goal: separate the two confusion sources. Three balanced buckets per mention,
  the mention is kept only if all three are non-empty
    - `both` ontologically related AND lexically similar (Dice >= `--hi`)
    - `lexno` lexically similar but NOT ontologically related, Dice-matched per
      mention to the `both` bucket
    - `ontono` ontologically related but NOT lexically similar (Dice < `--lo`)
- Inputs:
    - `concepts.pkl`, and per benchmark directory `mentions_test.jsonl` plus
      the already built `negatives_lexical_tfidf.jsonl` and
      `negatives_lexical_dice.jsonl`, whose union is the lexical pool
- Outputs
    - `negatives_lexonto_both.jsonl`, `negatives_lexonto_lexno.jsonl`,
      `negatives_lexonto_ontono.jsonl`
- Takes `--datasets` and loops over benchmark directories, unlike the builders
  above which take a single `--mentions` / `--output`

- run the script
```bash
python code/preprocess/lexonto_negatives.py \
--datasets "ncbi_disease" "bc5cdr_disease" \
--concepts "your_kb_path/concepts.pkl" \
--outputs-root "your_output_path"
```

- optional: `--hi` (default 0.6), `--lo` (default 0.3), `--k` (default 4)
  negatives per bucket, `--max-aliases` (default 8), `--seed`
