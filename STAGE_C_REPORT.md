# Stage C: structure-catalog coverage

**Date:** 30 September 2026. **Scope:** can a frozen public structure catalog supply the answers for held-out structures that have no reference spectrum in our library, and how far does a spectrum-free ranking take us? The report contains aggregate numbers only. No model was trained and nothing was submitted to Kaggle.

## External data: rules and licenses

- The competition overview states: "Freely & publicly available external data is allowed, including pre-trained models" ([overview](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)). The [rules](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules) (§6) require external data to be publicly available at no cost to other participants. Winning code must be released under MIT, while input data under an incompatible license is exempt from that grant.
- **PubChem:** NCBI "places no restrictions on the use or distribution of the data" but notes that original submitters may claim rights ([NCBI policies](https://www.ncbi.nlm.nih.gov/home/about/policies/), [README-Extras](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/README-Extras)).
- **COCONUT:** released under CC0 ([download page](https://coconut.naturalproducts.net/download)).

## Frozen snapshots

Downloaded into the git-ignored `data/external/`. The PubChem MD5s match the `.md5` files NCBI publishes next to each file.

| File | Snapshot | Bytes | MD5 |
| --- | --- | ---: | --- |
| `CID-Mass.gz` | PubChem 2026-09-27 | 1,391,793,064 | `a0be37b3643ef14cfb116b3eb33ae5cd` |
| `CID-InChI-Key.gz` | PubChem 2026-09-27 | 7,370,358,430 | `d39fd5843f2c70181c54f399bd2b4e86` |
| `CID-SMILES.gz` | PubChem 2026-09-27 | 1,486,837,335 | `ebca7c072bd278eda3c8ef636049f1e2` |
| `CID-PMID.gz` | PubChem 2026-09-27 | 326,400,588 | `ea46bb397c50f3ee3e00fbe711222805` |
| `coconut_csv_lite-09-2026.zip` | COCONUT September 2026 | 199,089,109 | `03246224bbe41fe389d47ea95c4f8eec` |

## Catalog index

`scripts/build_catalog.py` joins the files by CID and writes mass-sorted numpy arrays (3.3 GB) to `data/catalog/`. Build time was 332 s with 3.1 GB peak RSS.

| Step | Rows |
| --- | ---: |
| PubChem compounds read | 124,717,116 |
| Multi-component records removed (salts, mixtures: `.` in SMILES) | 11,884,436 |
| Missing InChIKey | 3 |
| **Kept** | **112,832,677** |
| Distinct structures (first standard InChIKey block) | 94,740,933 |
| Kept rows whose structure is in COCONUT | 1,549,217 |

Candidates are identified by the first block of PubChem's **standard** InChIKey. Running the competition's tautomer canonicalization over 113 million structures is not feasible. The evaluation below needed about 40 minutes for roughly 390,000 candidate SMILES, which extrapolates to about a week of single-core time for the full catalog [INFERENCE]. The **official metric** is applied only to the final top-25 lists, using PubChem SMILES.

## Method

`scripts/evaluate_catalog.py` evaluates the 5,200 structure-holdout queries (3,660 GNPS, 1,540 RIKEN). All their spectra are already excluded from the reference library (Stage A).

1. **Neutral mass.** `src/adducts.py` parses the observed adduct in `[kM±groups]z±` notation, covering all 31 adduct labels in the holdout. It computes the neutral monoisotopic mass with exact element and electron masses. Tests recover caffeine's mass from 13 adduct types.
2. **Candidate window.** All catalog rows within **±10 ppm** of that mass, deduplicated by structure. The tolerance was fixed before the run; coverage is also reported at 5 and 20 ppm.
3. **Ranking without spectra.** Three priors, fixed before the run:
   - `cid`: lowest PubChem CID first (older records first);
   - `pmid`: most PubMed links first;
   - `coconut_pmid`: COCONUT natural products first, then as `pmid`.
4. **Scoring.** The top-25 SMILES are scored with `src/metric.py` (official semantics).

## Results

**Coverage funnel (5,200 queries):**

| Stage | Queries | Share |
| --- | ---: | ---: |
| Answer structure present in PubChem (any mass) | 4,236 | 81.5% |
| …and inside the ±5 / ±10 / ±20 ppm window | 3,673 / **3,835** / 3,940 | 70.6 / **73.8** / 75.8% |
| Answer absent from PubChem | 964 | 18.5% |
| Precursor label inconsistent (organizer's `precursor_error_ppm` > 20 or null) | 435 | 8.4% |

Candidate windows are large. The number of distinct structures in a ±10 ppm window has quantiles **290 (10%), 5,510 (median), 14,256 (90%) and 44,623 (max)**. Only 58 queries have 25 or fewer candidates, and 6 windows are empty.

**Ranking (official metric):**

| Prior | Subset | n | Recall@25 | Top-1 | MRR@25 |
| --- | --- | ---: | ---: | ---: | ---: |
| `cid` | all | 5,200 | 0.204 | 0.058 | 0.090 |
| `pmid` | all | 5,200 | 0.283 | 0.103 | 0.147 |
| **`coconut_pmid`** | **all** | **5,200** | **0.353** | **0.103** | **0.157** |
| `coconut_pmid` | answer in 10 ppm window | 3,835 | 0.477 | 0.139 | 0.212 |
| `coconut_pmid` | GNPS queries | 3,660 | 0.263 | 0.067 | 0.105 |
| `coconut_pmid` | RIKEN queries | 1,540 | 0.565 | 0.188 | 0.280 |

The run took 2,399 s, most of it canonicalizing about 390,000 candidate SMILES for official scoring, with 1.5 GB peak RSS.

## Interpretation

**1. Coverage is not the main problem; ranking is.** 74% of held-out answers fall inside the ±10 ppm window. A typical window, however, holds thousands of candidate structures, and without spectral evidence the best prior places the answer in the top 25 for only 48% of those reachable queries (MRR 0.212). Following the plan's diagnostic rule (high recall, low MRR → ranking work), **Stage D, spectrum-based reranking of catalog candidates, is next.**

**2. The literature prior is optimistic here.** Every held-out molecule has public reference spectra, so it is likely better studied than a genuine class-2 molecule, which has none. The `pmid` and `coconut_pmid` gains over `cid` should therefore be expected to shrink on the hidden class-2 set. The priors are useful as a tie-breaker, not as the ranker.

**3. A quarter of GNPS answers are absent from PubChem.** 956 of the 964 misses are GNPS structures (26% of GNPS queries); only 8 are RIKEN (0.5%). A targeted rescan of the full `CID-InChI-Key` file found **none** of these 964 structures in *any* PubChem record, including the excluded salts and mixtures, so the component filter did not cause the misses. These 964 queries form a natural **class-3-like subset relative to this snapshot**: catalog lookup cannot answer them. Whether the GNPS structures are genuinely novel or partly mis-annotated cannot be decided from the metadata.

**4. About 8% of queries have inconsistent precursor labels** (> 20 ppm error or missing). They fail any mass window regardless of method. They are kept in all headline numbers; the `label_consistent_precursor` subset is in the output JSON.

## Scope limits

- **This is not a certified class-2 estimate.** Membership was checked against one frozen PubChem snapshot. The held-out molecules have public spectra, unlike real class-2 molecules.
- **Answer matching uses the first standard InChIKey block.** Coverage counts use the standard InChIKey first block (with every Stage A alias), while ranking metrics use the official tautomer-canonical key on the top-25 SMILES. These can differ for a few tautomeric structures: 38 of the 964 absent answers have more than one alias.
- **Single spectrum per query; no spectral evidence in ranking.**
- **Runtime and memory are laptop measurements, not a Kaggle Notebook.** A Kaggle submission would need the 3.3 GB index plus candidate SMILES as offline inputs; Kaggle dataset size limits were not checked.

## Reproduce

```bash
# Download the five snapshot files above into data/external/pubchem-2026-09-27/ and data/external/coconut-2026-09/
uv run python scripts/build_catalog.py
uv run python scripts/evaluate_catalog.py
```

Defaults point at those directories and write `data/catalog/pubchem-2026-09-27/` and `data/analysis/catalog_baseline.json`.

## Next decision

**Stage D:** rerank the ±10 ppm PubChem candidates with spectral evidence, using the same folds and official metric, keeping `coconut_pmid` as a tie-breaker. First, audit openly licensed options for their availability, license and offline runtime:
- in-silico fragmentation (predicted spectra per candidate);
- spectrum-to-fingerprint models;
- fragment/neutral-loss matching against the candidate structure.

Then fuse with the Stage B library search, whose class-1 proxy MRR is 0.890. The 964 structures absent from PubChem are the natural test set for later Stage E generation work.
