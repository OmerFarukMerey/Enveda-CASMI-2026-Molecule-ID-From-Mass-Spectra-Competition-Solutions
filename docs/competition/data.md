<!-- index: competition files and columns, hidden-test facts, the 11 source libraries, novelty classes 1-3 -->
# Competition data

Source: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data (read 2026-09-30).
Licence: CC BY-NC 4.0. **Not in this repo.** Download into `data/input/competitions/<slug>/`, see [../../data/README.md](../../data/README.md).

## Files (3.04 GB)
| File | Rows | Notes |
|---|---|---|
| `train.parquet` | ~2.5 M spectra, ~275 k structures | labelled; **no `molecule_id` / `spectrum_id`** |
| `test.parquet` | visible: placeholder drawn from train | replaced by the hidden test at scoring (~1,500 spectra / ~400 molecules) |
| `sample_submission.csv` | | format reference |

## Hidden test
~400 molecules, 1–16 spectra each (median 3), **all Bruker timsTOF**, monoisotopic mass 157–1,159 Da (median 348).
10 adducts: `[M+H]+ [M+NH4]+ [M-H2O+H]+ [M-2H2O+H]+ [M+Na]+ [M+K]+ [M-H]- [M-H2O-H]- [M+CH2O2-H]- [M+Cl]-`.
Minimal cleaning: precursor/structure-inconsistent spectra dropped, base peak <1,000 counts dropped, peaks above
precursor+2 Da removed.

**Novelty classes (hidden mix):**
1. In public spectral libraries → reachable by spectral similarity to train.
2. Known structure (PubChem or COCONUT) without public spectra → database retrieval + ranking.
3. Novel (not in PubChem) → de novo or analog generation.

## Columns
Shared (test = 12 columns): `molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities (base=1.0),
base_peak_intensity, adduct, ionization_mode, instrument_type, precursor_mz, collision_energy_ev (list),
collision_energy_orig, collision_energy_orig_units`.
Train-only: `normalized_smiles` (label), `inchikey`, `inchikey14`, `molecular_formula`, `ingest_lib`, `adduct_orig`,
`precursor_error_ppm` (uncleaned, a label-quality signal), `num_peaks`. Use `collision_energy_ev` (NCE is converted
approximately).

## Source libraries (`ingest_lib`)
| Library | Spectra | Structures | Note |
|---|---|---|---|
| enveda-180 | 1,153,785 | 182,941 | same timsTOF as test, but synthetic drug-like chemistry |
| pluskal_ms2 (MSnLib) | 527,581 | 46,821 | Orbitrap, NCE, multi-energy |
| riken | 347,171 | 15,892 | plant specialised metabolites |
| gnps | 220,849 | 45,750 | largest NP collection, heterogeneous |
| massbank | 101,727 | 9,180 | curated consortium |
| mona | 92,416 | 11,681 | |
| spectraverse | 50,933 | 9,631 | harmonised aggregation |
| msdial | 40,765 | 9,127 | |
| drug_plus | 2,545 | 2,539 | no CE metadata |
| **enveda-np-examples** | 1,151 | 250 | **closest to test** (same instrument and pipeline, common NPs) |
| masaryk | 652 | 416 | |

Only enveda-180 and enveda-np-examples are uniformly timsTOF (about 46% of train rows).

## Suggested external resources (from the data page)
MIST-CF / SIRIUS (formula annotation), matchms, MassIVE / GeMS / DreaMS (unlabelled pretraining), FragHub. GNPS
propagated ("suspect") annotations are excluded from train because their labels are inferred.
