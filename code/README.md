# Overview

## Datasets

| Dataset | Entity Type | Download |
|---|---|---|
| BC5CDR | Chemical/Disease | [Download](https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/) |
| NCBI-Disease | Disease  | [Download](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) |
| NLM-Chem | Chemical  | [Download](https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-NLMChem-corpus_v2.BioC.xml.gz) |
| MedMentions | General biomedical | [Download](https://github.com/chanzuckerberg/MedMentions) |

## Knowledge Base

| Knowledge Base | Version used | Download |
|---|---|---|
| UMLS | 2020 | [UMLS Archives](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html) |

We use **UMLS 2020** in our experiments.

## Our UMLS-mapped benchmarks
We mapped the annotations of BC5CDR, NLM-Chem and NCBI-Disease to UMLS. The mapped data is in
[`data/`](../data), the code that generates it in [`preprocess/`](./preprocess).

The statistics are as follows:

| Dataset | train | dev | test |
|---|---|---|---|
| BC5CDR | 9,160 | 9,363 | 9,469 |
| NCBI-Disease | 5,086 | 785 | 960 |
| NLM-Chem | 19,385 | 4,927 | 11,169 |
