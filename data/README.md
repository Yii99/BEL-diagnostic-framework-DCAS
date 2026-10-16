# Benchmarks

Three benchmarks in their original PubTator form, plus the UMLS-mapped version we
work from. MedMentions is already annotated with CUIs and needs no mapping; it is
not included here, see its own repository.

## What we added

The benchmarks annotate mentions with their own identifiers -- MeSH descriptors and
supplementary records, OMIM entries -- not with UMLS CUIs. Our evaluation uses UMLS
as the knowledge base, so every source identifier is resolved to the CUIs that carry
it. `code/preprocess/map_pubtator_to_umls.py` does this against MRCONSO.

Two files per split:

| | |
|---|---|
| `*.umls.txt` | the original file with one column appended: `UMLS:C0023895` or `UMLS:C0267809;C1835713;...` when the identifier maps to several equivalent concepts |
| `*.umls.txt.mapping.tsv` | one row per source identifier: `source_namespace`, `source_id`, `mention_count`, `mapping_status`, `mapping_method`, `n_cui`, `umls_cuis` |

A mention keeps every CUI its identifier maps to. All of them count as correct, and
none of them is ever sampled as a negative.

`mapping_status` is `mapped_unique`, `mapped_multiple`, or `unmapped`. Unmapped
identifiers are retired or source-local codes with no UMLS counterpart; the mentions
carrying them are dropped, which is what the counts below already reflect.

## Mentions after mapping

| Dataset | train | dev | test |
|---|---|---|---|
| BC5CDR | 9,160 | 9,363 | 9,469 |
| NCBI-Disease | 5,086 | 785 | 960 |
| NLM-Chem | 19,385 | 4,927 | 11,169 |

## Layout

```
BC5CDR/CDR.Corpus.v010516/CDR_{Training,Development,Test}Set.PubTator[.umls].txt
NCBI_Disease/NCBI{train,develop,test}set_corpus[.umls].txt
NLMChem/BC7T2-NLMChem-corpus-{train,dev,test}.PubTator.single[.umls].txt
```

NLM-Chem is distributed as BioC XML with multiple identifiers per mention;
`code/preprocess/preprocess_nlmchem.py` converts it to PubTator and splits those
into single-identifier annotations, which is the `.single` in the name.

## Sources

| | |
|---|---|
| BC5CDR | https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/ |
| NCBI-Disease | https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/ |
| NLM-Chem | https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/ |
| MedMentions | https://github.com/chanzuckerberg/MedMentions |
| UMLS | https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html |

UMLS needs a licence; the benchmarks keep the licences of their own distributions.
