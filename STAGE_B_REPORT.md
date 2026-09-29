# Stage B: first reference-retrieval baseline

**Date:** 29 September 2026. **Scope:** the simplest adduct- and precursor-gated spectral library search, evaluated on the fixed Stage A class-1 proxy. The report contains aggregate numbers only. No model was trained, no parameter was tuned, and nothing was submitted to Kaggle. The visible `test.parquet` placeholder was not used.

## Method

`scripts/evaluate_retrieval.py` reads the Stage A manifest (`data/analysis/validation.json`, seed 2026) and applies `reference_filter(manifest)` before any candidate is scored. Held-out structures, invalid-label groups and every non-reference spectrum of a class-1 group are therefore absent from the library. The script checks that the manifest's file name, row count, row groups, sources, supplied keys and official canonical keys match the Parquet it is given.

Fixed baseline settings, chosen before the run and not tuned:

| Step | Rule |
| --- | --- |
| Candidate gate | Identical observed `adduct` string; `|reference precursor − query precursor| ≤ 20 ppm` of the query precursor (inclusive) |
| Peak cleaning | Keep finite peaks with `0 < m/z ≤ precursor + 2 Da` and `0 < intensity ≤ 1`; keep the 100 most intense, then sort by m/z |
| Similarity | Unshifted cosine on √intensity; each peak matched at most once, ±0.02 Da inclusive, greedy by weight product |
| Candidate list | Score **every** gated reference; merge references by official tautomer-canonical InChIKey14 and keep the best; rank by cosine ↓, ppm error ↑, row index ↑, key ↑; at most 25 distinct structures |
| Scoring | `src/metric.py` (official-notebook semantics). A query with no gated reference submits no row and scores zero |

`src/spectral_search.py` holds the peak preparation and cosine. The evaluator streams metadata in Arrow batches and loads peak arrays one selected row at a time.

## Results on the 250 class-1 queries

| Metric | Value |
| --- | ---: |
| Candidate recall@25 | **0.404** (101/250) |
| Top-1 | **0.248** (62/250) |
| MRR@25 | **0.296** |
| Queries with ≥1 gated candidate | 230 (0.92) |
| Designated reference passes the gate | 107 (0.428) |
| Eligible library rows | 2,388,656 |
| Rows removed as structure-holdout aliases | 93,929 |
| Gated reference rows / query–reference pairs | 50,277 / 53,685 |
| Wall time, peak memory (one Apple-silicon process) | 538 s, 337 MB RSS |

By source of the designated reference: RIKEN n=246, recall 0.402, top-1 0.248, MRR 0.295; GNPS n=3, recall 0.333, MRR 0.167; MassBank n=1, hit at rank 1. The GNPS and MassBank slices are too small to interpret.

## What limits the score

**The gate, not the ranking, is the bottleneck on this proxy.** Stage A allows exactly one reference spectrum per class-1 structure. That reference passes the baseline gate for only **107/250** queries, which caps candidate recall at 0.428. Of the 143 failures, **94** have a different or missing adduct label and **49** have the same adduct but a precursor difference above 20 ppm (the median difference is 6.2 ppm when adducts agree).

Within the 107 queries whose reference passes the gate, the search places the answer in the top 25 for **101 (94%)**, at rank 1 for **62 (58%)**, and reaches a conditional MRR of about **0.69**. Spectral ranking is therefore already useful once the right reference is reachable.

Two points follow, and both are inferences rather than measured hidden-test facts:

1. **The single-reference proxy probably understates class-1 performance.** At inference, every public spectrum of a structure is available, not one hand-picked row. The Stage A design picked the reference by source priority, not by adduct or collision-energy compatibility. A manifest that keeps all non-query-source spectra of the structure as references (still removing exact duplicates, and still removing the query source) would measure the realistic library.
2. **Strict adduct string equality is costly.** Comparing neutral masses computed from each adduct would admit references recorded with a different but chemically compatible adduct. Heterogeneous adduct labelling across libraries should be audited before relying on string equality.

## Scope limits

- This is a single-spectrum class-1 **proxy**. Train has no acquisition-group IDs, so the multi-spectrum aggregation the real test allows was not evaluated.
- The 5,200-group structure holdout is excluded from the library, so this search cannot retrieve those answers. Their result is zero by construction and is **not** reported as a class-2 or class-3 estimate. No PubChem/COCONUT catalog has been built.
- The results come from one full run. Determinism and tie-breaking are covered by tests; a second full run was not repeated.
- Runtime was measured on a local laptop, not in a Kaggle Notebook.

## Reproduce

```bash
uv run python scripts/evaluate_retrieval.py --train data/raw/train.parquet --manifest data/analysis/validation.json --output data/analysis/retrieval_baseline.json
```

The output JSON stays under the git-ignored `data/` directory.

## Next decision

Before any peak-cleaning or similarity variant, fix what is being measured: (1) add a class-1 reference policy that keeps the whole public library for the query structure, minus the query source and exact duplicates, and report both the current single-reference and the full-library numbers; (2) test a neutral-mass gate across compatible adducts against the current exact-adduct gate on the same folds. Only after these should Stage B compare cosine variants, and only then should Stage C catalog work begin.
