# Stage F (early): first Kaggle submission

**Date:** 30 September 2026. **Scope:** package the Stage B library search and the Stage C catalog as one offline Kaggle Notebook, prove it runs within the rules, submit once, and compare the public score with our proxies. Stage F was deliberately moved ahead of Stage D, so every later ranking change lands on a working submission path.

## Result

| | Value |
| --- | --- |
| Submission | 30 September 2026, kernel `ilkeryasin8/casmi26-submission` version 1 |
| **Public leaderboard MRR@25** | **0.193** |
| Leaderboard on the same day (public API, 1,868 teams) | top 0.464; 90th percentile 0.373; median 0.312; 1,238 teams at or above 0.193 |

Kaggle's public leaderboard scores only part of the hidden test; the final ranking uses the rest, and the class mix is unknown. The score is far below the leaderboard median, so the pipeline is a working baseline, not a competitive entry.

## Pipeline

`src/pipeline.py`, run by `scripts/predict.py`, with fixed settings; nothing was tuned on the leaderboard or the placeholder:

1. **Library:** each test spectrum is searched in all labelled training rows with its own adduct (exact label) within ±20 ppm. The Stage B √intensity cosine (±0.02 Da, top 100 peaks) is used; a structure's score is its best cosine over the molecule's spectra.
2. **Catalog:** each spectrum's neutral mass (`src/adducts.py`) selects PubChem 2026-09-27 structures within ±10 ppm. Structures supported by more of the molecule's spectra rank first, then COCONUT membership, PubMed links and CID.
3. **Fusion:** library structures with cosine ≥ 0.7 first; the threshold follows the common GNPS library-search convention and is not tuned. Catalog structures follow, then weaker library structures, up to 25 distinct structures. Structures are identified by the first standard InChIKey block. A molecule with no candidate at all receives the filler `C`; none of the 400 placeholder molecules needed it.

**RDKit is not installed in the Kaggle image** (probe run, internet off), so inference does not use it. The shared code lives in `src/library_search.py` and `src/catalog.py`; Stage B/C evaluators use the same modules. After the refactor, the Stage B single-reference evaluation reproduced MRR@25 0.2963539682539683 and the identical eligible, gated and pair counts.

## Kaggle packaging

| Item | Detail |
| --- | --- |
| Environment (probe, internet off) | Python 3.12.13, numpy 2.0.2, pyarrow 24.0.0, 4 CPUs, 32 GB RAM, 20.9 GB working disk, no GPU requested, **no RDKit** |
| Private dataset `ilkeryasin8/casmi26-pubchem-catalog` | Stage C arrays (3.3 GB) + `CID-SMILES`; Kaggle decompressed the upload to 8.9 GB |
| Private dataset `ilkeryasin8/casmi26-code` | `src/` + `scripts/predict.py`, built by `kaggle/package_code.py`; Kaggle unpacked the zip into folders |
| Notebook | `kaggle/submission/run.py` + `kernel-metadata.json`: locates the inputs, runs `scripts/predict.py`, writes `/kaggle/working/submission.csv` |
| Placeholder run on Kaggle | Completed; pipeline 276 s, whole run about 11 minutes; output **byte-identical** to the local run |

The local check of that output with the official chemistry found 400 rows of 25 guesses and no unparseable SMILES. **10 of the 10,000 slots** (in 8 rows) are tautomer duplicates under the official key. These are valid but wasted guesses, a consequence of deduplicating by the standard InChIKey without RDKit.

## The same pipeline on our proxies

`scripts/evaluate_pipeline.py` runs `predict` unchanged on the Stage A proxies. Each proxy query counts as a one-spectrum molecule. The library follows the Stage B `library` rule, so held-out answers can only come from the catalog. Run time was 1,222 s.

| Proxy | n | Recall@25 | Top-1 | MRR@25 | Earlier single-source result |
| --- | ---: | ---: | ---: | ---: | --- |
| Class-1 cross-source | 250 | 0.996 | 0.856 | **0.911** | library only: 0.890 (Stage B) |
| Structure holdout | 5,200 | 0.350 | 0.091 | **0.145** | catalog only: 0.157 (Stage C) |
| …RIKEN | 1,540 | 0.561 | 0.160 | 0.255 | 0.280 |
| …GNPS | 3,660 | 0.261 | 0.062 | 0.099 | 0.105 |

The fusion helps class 1: the catalog rescues queries whose library reference fails the gate. It costs the holdout 0.012 MRR, because a wrong library structure with cosine ≥ 0.7 now occupies the top slots for some held-out molecules. 1,342 of the 5,450 proxy queries had a library hit ≥ 0.7. At most 250 of those are class-1 queries, so at least 1,092 held-out molecules had a wrong structure placed ahead of every catalog candidate.

## Interpreting the gap

The proxies predict roughly 0.9 for class 1 and 0.15 for unseen structures. A leaderboard score of 0.193 is therefore consistent with a hidden test dominated by molecules this pipeline cannot rank well. It does not determine the class mix [INFERENCE]. Plausible causes, none of them measured yet:

- **Our class-1 proxy is easier than the hidden class 1.** It uses common natural products with rich library coverage.
- **Library hits below 0.7 are effectively dropped.** For every molecule, 25 catalog structures fill the list first. Cross-instrument timsTOF-versus-library cosines below 0.7 therefore never appear.
- **Spectrum-free catalog ranking is weak** (Stage C), and it is the only mechanism for classes 2 and 3.
- **Other teams use a stronger common baseline.** The leaderboard median of about 0.31 suggests this [INFERENCE]. Public notebooks were not inspected.

## Reproduce

```bash
uv run python scripts/predict.py            # local placeholder run -> data/submission/submission.csv
uv run python scripts/evaluate_pipeline.py  # proxy scores of the same pipeline
uv run python kaggle/package_code.py        # code dataset folder under data/kaggle/code-dataset
uvx --from kaggle kaggle datasets version -p data/kaggle/code-dataset -m "<message>"
uvx --from kaggle kaggle kernels push -p kaggle/submission
uvx --from kaggle kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -k ilkeryasin8/casmi26-submission -v <version> -f submission.csv -m "<message>"
```

## Next decision

Before Stage D spectral reranking of catalog candidates, find out why the leaderboard is far below the proxies. Three cheap checks come first:

1. **Library behavior on test-like spectra.** Measure the library cosine distribution for test-like timsTOF queries and whether dropping hits below 0.7 loses answers.
2. **Candidate source.** Evaluate the 275k training structures as a mass-window candidate source next to PubChem. Test molecules may be closer to the training chemistry than to PubChem as a whole.
3. **The ~0.31 baseline.** Read the highest-voted public notebooks to identify the approach behind the cluster around 0.31. Its ideas count as external methods: check the license, and cite it before reusing anything.

Each change is scored on the same proxies before it spends a submission.
