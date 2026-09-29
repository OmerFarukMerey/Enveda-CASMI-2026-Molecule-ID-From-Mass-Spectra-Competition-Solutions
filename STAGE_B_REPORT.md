# Stage B: first reference-retrieval baseline

**Date:** 29–30 September 2026. **Scope:** the simplest adduct- and precursor-gated spectral library search, evaluated on the fixed Stage A class-1 proxy under two reference-library rules. The report contains aggregate numbers only. No model was trained, no search parameter was tuned, and nothing was submitted to Kaggle. The visible `test.parquet` placeholder was not used.

## Method

`scripts/evaluate_retrieval.py` reads the Stage A manifest (`data/analysis/validation.json`, seed 2026). It checks that the manifest's file name, row count, row groups, sources, supplied keys and official canonical keys match the Parquet it is given. It then applies one of two class-1 reference rules before any candidate is scored:

| Rule | Which spectra of a class-1 structure are in the library |
| --- | --- |
| `designated` | Only the single reference row the manifest chose (`reference_filter`). |
| `library` | Every row from a source other than the query source `enveda-np-examples` (`class1_library_filter`). Same-structure rows whose peak arrays exactly copy the query are dropped and counted. |

Under both rules, all 5,200 structure-holdout groups (every alias, every source) and invalid-label groups stay out of the library. The `library` rule is closer to inference, where every public spectrum of a structure is available; it is for reference evaluation only, not supervised training.

Fixed search settings, chosen before the first run and identical for both rules:

| Step | Rule |
| --- | --- |
| Candidate gate | Identical observed `adduct` string; `|reference precursor − query precursor| ≤ 20 ppm` of the query precursor (inclusive) |
| Peak cleaning | Keep finite peaks with `0 < m/z ≤ precursor + 2 Da` and `0 < intensity ≤ 1`; keep the 100 most intense, then sort by m/z |
| Similarity | Unshifted cosine on √intensity; each peak matched at most once, ±0.02 Da inclusive, greedy by weight product |
| Candidate list | Score **every** gated reference; merge references by official tautomer-canonical InChIKey14 and keep the best; rank by cosine ↓, ppm error ↑, row index ↑, key ↑; at most 25 distinct structures |
| Scoring | `src/metric.py` (official-notebook semantics). A query with no gated reference submits no row and scores zero |

`src/spectral_search.py` holds the peak preparation and cosine. The evaluator streams metadata in Arrow batches and loads peak arrays one selected row at a time.

## Results on the 250 class-1 queries

| Metric | `designated` | `library` |
| --- | ---: | ---: |
| **MRR@25** | **0.296** | **0.890** |
| Candidate recall@25 | 0.404 (101/250) | 0.964 (241/250) |
| Top-1 | 0.248 (62/250) | 0.836 (209/250) |
| Reachable queries (a same-structure reference passed the gate) | 107 | 242 |
| Recall@25 / top-1 / MRR among reachable queries | 0.944 / 0.579 / 0.692 | 0.996 / 0.864 / 0.919 |
| Queries with any gated candidate | 230 | 246 |
| Eligible library rows | 2,388,656 | 2,444,495 |
| Gated reference rows / query–reference pairs | 50,277 / 53,685 | 73,084 / 79,543 |
| Exact query copies removed | 0 | 0 |
| Wall time, peak RSS (one Apple-silicon process) | 522 s, 516 MB | 834 s, 402 MB |

Both rules remove the same 93,929 structure-holdout rows. The `designated` run reproduced the first Stage B run exactly. Two full `library` runs gave identical metrics.

Under `library`, the correct hit came from eight different sources: GNPS 67, Pluskal 58, MoNA 36, RIKEN 33, MassBank 26, drug_plus 10, MS-DIAL 7, Spectraverse 4. The designated reference source is therefore no longer a meaningful slice. The per-designated-source breakdown stays in the output JSON only for comparison with the first run.

## Interpretation

**1. The single-reference proxy strongly understated class-1 retrieval.** With one hand-picked reference, only 107 queries were reachable. Of the 143 designated references that failed the gate, 94 had a different or missing adduct label and 49 were more than 20 ppm away. When every other-source spectrum is available, 242 queries are reachable and the same unchanged search reaches MRR@25 0.890.

**2. Ranking works once the answer is reachable.** Among reachable queries, the answer is in the top 25 in 99.6% of cases and at rank 1 in 86%.

**3. There is no sign of copied spectra, but independence is not proven.** No same-structure reference exactly copied a query spectrum. The best same-structure cosine per reachable query has quantiles 0.66 (10%), 0.80 (25%), **0.89 (median)**, 0.92 (75%) and 0.94 (90%); none reaches 0.99. Copies of one measurement would typically score near 1.0, so the hits look like separate measurements. Metadata cannot prove different acquisitions, however, and a reprocessed copy of the same raw data could score in this range. Treat 0.890 as optimistic for class 1.

**4. The remaining class-1 gap is small.** At most 8 queries (3.2%) are unreachable under the exact-adduct gate, and 4 have no gated candidate at all. A neutral-mass gate across compatible adducts can add at most those 8 queries. It is still worth a check, but its ceiling on this proxy is low.

## Scope limits

- **Class 1 only.** The 250 queries are common natural products, and 246 of their structures occur in RIKEN alone (Stage A). They are probably easier than the hidden test's class-1 molecules; the proxy is not a hidden-test estimate.
- **Single spectrum per query.** Train has no acquisition-group IDs, so the multi-spectrum aggregation the real test allows was not evaluated.
- **Classes 2 and 3 score zero by construction.** Their answers have no reference spectrum, so library search cannot return them. The 5,200-group structure holdout is excluded from the library and is **not** reported as a class-2 or class-3 estimate. No PubChem/COCONUT catalog has been built.
- Runtime was measured on a local laptop, not in a Kaggle Notebook.

## Reproduce

```bash
uv run python scripts/evaluate_retrieval.py --class1-references designated --output data/analysis/retrieval_designated.json
uv run python scripts/evaluate_retrieval.py --class1-references library --output data/analysis/retrieval_library.json
```

`--train` and `--manifest` default to `data/raw/train.parquet` and `data/analysis/validation.json`. Outputs stay under the git-ignored `data/` directory.

## Next decision

On this proxy, simple library search already handles class 1: MRR@25 0.890, with at most 3.2% of queries left to gain from a looser gate. The larger unexplored loss is classes 2 and 3, where this method scores zero. The next priority should therefore be **Stage C: a frozen, license-compatible structure catalog** (PubChem/COCONUT) and a measurement of its coverage on the structure holdout. The neutral-mass gate is a small class-1 follow-up. Cosine variants should wait until candidate coverage for classes 2 and 3 exists.
