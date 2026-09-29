# Stage A: data audit and validation evidence

**Date:** 29 September 2026. **Scope:** full local Parquet audit, leak-aware validation proxies and local scoring parity. This report contains aggregated counts only; no competition rows, spectra, labels or manifests are committed. The visible test file is an organizer-supplied **placeholder drawn from training**, not the hidden scoring distribution. No model or leaderboard result is reported.

## Data and method

The audit streams `data/raw/train.parquet` and `data/raw/test.parquet` in 8,192-row Arrow batches; it computes exact finite-value percentiles from compact numeric arrays and counts per-source structure overlaps on the **provided** `inchikey14`. All 2,539,608 training and 1,213 placeholder rows were scanned. Reproduce the summary with:

```bash
uv sync --group dev
uv run python scripts/profile_data.py --train data/raw/train.parquet --test data/raw/test.parquet --output data/analysis/profile.json
```

`data/` is git-ignored; `profile.json` contains aggregated statistics only and is locally reproducible. Input files must be downloaded by a participant who accepted the [Kaggle rules](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules), as described in [RESEARCH_PLAN.md](RESEARCH_PLAN.md). The [organizer's data description](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data) supplies class definitions and acquisition context; counts below come from our local scan unless labeled otherwise.

## Training inventory and source shift

The training file has **2,539,608 spectra, 275,810 distinct provided `inchikey14` values, 18 columns and 21 Parquet row groups**. It contains no `molecule_id` or `spectrum_id`; these exist only in the test schema. Each source is counted independently, so source structure counts overlap.

| `ingest_lib` | Spectra | Distinct provided keys | Missing raw base peak | Missing converted energy | Median peaks | Rows with peak > precursor + 2 Da | Rows with \|precursor error\| > 20 ppm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| enveda-180 | 1,153,785 | 182,941 | 0 | 0 | 138 | 282,613 | 0 |
| pluskal_ms2 | 527,581 | 46,821 | 527,581 | 0 | 21 | 4,259 | 50,790 |
| riken | 347,171 | 15,892 | 347,171 | 31,680 | 9 | 107,684 | 40,060 |
| gnps | 220,849 | 45,750 | 0 | 220,849 | 46 | 68,061 | 16,147 |
| massbank | 101,727 | 9,180 | 101,727 | 1,912 | 11 | 4,391 | 6,990 |
| mona | 92,416 | 11,681 | 92,416 | 15,745 | 57 | 39,157 | 5,345 |
| spectraverse | 50,933 | 9,631 | 50,933 | 47,075 | 47 | 6,391 | 5 |
| msdial | 40,765 | 9,127 | 40,765 | 16,985 | 11 | 3,245 | 87 |
| drug_plus | 2,545 | 2,539 | 2,545 | 2,545 | 108 | 79 | 89 |
| enveda-np-examples | 1,184 | 250 | 0 | 0 | 141.5 | 0 | 1 |
| masaryk | 652 | 416 | 0 | 652 | 9 | 0 | 0 |

The organizer's page lists **1,151** `enveda-np-examples` spectra; the downloaded Parquet contains **1,184**. Use the local file for reproducible counts. The two Enveda sources are uniformly labeled timsTOF, but `enveda-180` is predominantly synthetic chemistry while `enveda-np-examples` contains common natural products. Treat source, instrument and chemistry as separate factors; instrument match alone is not a representative validation sample. [Dataset description](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)

Polarity and mass coverage vary sharply by source. `enveda-np-examples` has **795 positive / 389 negative** spectra (median precursor **387.65 m/z**) versus RIKEN **313,488 / 33,683** (median **399.14 m/z**) and Pluskal **319,627 / 207,954** (median **343.11 m/z**). Spectraverse is heavier (median **748.53 m/z**) and its most frequent adduct is `[M+NH4]+` (**16,810** rows); RIKEN has **121,677** `[M+Na]+` rows. GNPS is mostly Orbitrap by instrument label (**118,223** rows), and Pluskal is entirely Orbitrap (**527,581**). The ignored `profile.json` retains complete per-source categorical counts and finite-value quantiles; these examples are not a guessed hidden-test mix.

Across train, **1,163,138/2,539,608** base-peak intensities and **337,443/2,539,608** converted collision-energy lists are null; `precursor_error_ppm` is null for 4,286 rows. All aligned m/z/intensity arrays have equal lengths; no rows have an empty list or nonfinite/negative/>1 normalized intensity, and every finite row maximum is within 1±10⁻⁶. In contrast, **515,880 rows** contain at least one finite fragment m/z greater than finite precursor + 2 Da. The placeholder has zero such rows, consistent with the organizer's described cleaning. This motivates an explicit, fold-fitted peak policy before spectral matching; it does not establish the best cutoff.

The training median peak count is **42** (95th percentile 628; maximum 73,318). The precursor m/z median is **350.1088**, but the maximum is **2,430,333** in MoNA; do not silently assume every training precursor is in a test-like range. For non-null precursor error, median absolute error is **1.512 ppm**, 95th percentile **17.181 ppm**, and **119,514 rows exceed 20 ppm**. Outliers are source-dependent: RIKEN has 40,060, Pluskal 50,790 and GNPS 16,147 rows above that threshold. These are quality signals, not proof that any individual label is wrong; a global ppm filter would change source coverage unevenly. Energy conventions are also heterogeneous: `enveda-180`/`enveda-np-examples` report eV, Pluskal reports NCE, GNPS lacks converted eV throughout, while RIKEN mixes eV, NCE, V and unknown. Do not equate raw collision-energy numbers across these libraries.

## Structure overlap and label identity

Of the **250 provided keys** in `enveda-np-examples`, **246** occur in RIKEN, **233** in GNPS, **231** in MoNA, **224** in Pluskal, and **zero** in `enveda-180`. Counts overlap across libraries; they are not independent totals. A class-1 reference-search proxy therefore has genuine cross-source structure overlap, but shared structure does **not** prove independent measurement or exclude near-duplicate spectra. At the provided-key level, **1,612 keys** have multiple literal `normalized_smiles` strings and **118 keys** have multiple literal molecular formulas; string differences alone are not label errors.

**Crucial identity warning:** the official metric derives the scoring key by `Chem.MolFromSmiles` → RDKit 2026.03.3 `TautomerEnumerator().Canonicalize` → `Chem.MolToInchiKey(... )[:14]`. The supplied training `inchikey14` is **not always that key**. A reproducible 1-in-32 hash sample of **8,552** supplied keys (first encountered label per key) found **135** whose scoring key differs. This sample is not a full mismatch census. Grouping folds by only the provided key can leak scoring-equivalent molecules across validation and reference/training. The split manifest must conservatively join provided and scoring identities before partitioning. [Official metric](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank)

## Placeholder boundary

The downloaded `test.parquet` has **1,213 spectra / 400 molecule IDs**, 1–9 spectra per ID. All are marked timsTOF; 959 are `[M+H]+`, 193 `[M-H]-` and 61 other adducts. Median peak count is 230. The 400 IDs match `sample_submission.csv`, but they are examples from the training data and will be replaced for scoring. No thresholds, class proportions or accuracy claims in this report are derived from the placeholder distribution. [Dataset description](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)

## Exact local scoring contract

`src/metric.py` implements the [official evaluator](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank) chemistry using pinned RDKit 2026.03.3: parse each answer and guess as SMILES, canonicalize its tautomer, generate the InChIKey and compare the first 14 characters. Do **not** compare the provided training `inchikey14` directly, and do **not** use full molecular formula as a match gate. Each molecule contributes the reciprocal of its **first** correct nonempty candidate's one-based rank, or zero; average over answer molecules, not spectra. Multiple accepted answers in the official solution are comma-separated, but the competition says there is one correct structure per molecule.

The official scorer strips whitespace and removes empty semicolon fields *before* assigning ranks; invalid nonempty SMILES occupy a rank but cannot match. An absent answer ID scores zero if another row was submitted; an extra ID does not affect the mean but is still parsed and can make the submission invalid if its row is empty or exceeds the rank limit. The official scorer rejects an empty submission, null required columns, duplicate IDs, no guesses in an explicit row and more than 25 nonempty guesses. The competition's submission instructions additionally require each test ID exactly once. `validate_submission` enforces that stronger export contract, plus parseable and distinct normalized structures; those last checks are deliberate quality safeguards, **not** a claim that the official scorer rejects every duplicate or invalid guess.

The comparison against the downloaded official metric Notebook exercised stereochemistry/tautomers, missing and extra IDs, blank tokens, invalid guesses, unknown-ID rank overflow and blank unknown-ID rows. Numeric MRR values agreed (`0.75`, `0.5`, `0.5` for the three scoring scenarios); the two invalid extra-row scenarios raised in both implementations. Exception class names differ (`ParticipantVisibleError` versus local `ValueError`), not scoring behavior.

## Fixed validation proxies and leakage rules

`scripts/build_validation.py` reads every training label and unions each supplied `inchikey14` with its **official tautomer-canonical** InChIKey14. Transitive aliases are one identity group, not separate molecules. The full scan yielded **274,183** valid groups versus **275,810** distinct supplied keys; zero rows had unusable keys or labels. This conservative grouping prevents a supplied-key/scoring-key mismatch from crossing a validation boundary. It does not certify chemical equivalence beyond the official 14-character key.

RDKit emitted tautomer-enumeration limit and kekulization diagnostics during the scan; none caused a row to fail the final canonical-label conversion. The default enumerator limits match the pinned official scorer, but a 14-character key and capped enumeration remain operational definitions, not a proof of full chemical identity.

The committed code generates a local, git-ignored, format-v2 manifest with seed **2026** and original zero-based Parquet row indices. Its **class-1 cross-source proxy** uses one `enveda-np-examples` query spectrum per group and one same-identity, different-source reference: **250 pairs**, with **246 RIKEN, 3 GNPS and 1 MassBank** references under the explicit natural-product-first source priority. It excludes identical m/z/intensity arrays from paired query/reference candidates (zero such pairs were rejected in this run). It cannot exclude near-duplicates or establish independent acquisition from the available metadata. For these groups, only the chosen reference spectrum may be used for training/reference indexing; the query and all other same-group spectra are held out.

The **structure-disjoint proxy** considers groups with at least one RIKEN/GNPS query candidate, excluding class-1 groups and invalid identities. It deterministically hashes the seed and group to withhold **5,200 of 52,004 eligible groups** (5,454 supplied-key aliases). One query spectrum per withheld group is selected by source preference then fixed identity/row tie-breaks: **3,660 GNPS and 1,540 RIKEN**. Remove **all** rows bearing any withheld alias from the training corpus and reference index before fitting features or tuning. `reference_filter(manifest)` applies these exclusions, rejects unidentified/bad-label rows and permits only the explicit class-1 reference row for each class-1 group; external spectral/model-training corpora need the same identity exclusions separately.

This holdout measures unfamiliar structure/spectrum retrieval under the chosen source mix; **it is not a verified class-2 set**. That requires a frozen, legally usable PubChem/COCONUT candidate catalog and independently checked membership/absence for each withheld answer. It is **not a class-3 set** either: prove the answers are absent from *all* candidate catalogs, reference spectra and generative pretraining/supervision first. No catalog was downloaded or audited in Stage A, so neither coverage nor true novelty is claimed. The manifest checks group separation, row policies and pair identities; a repeat `--check` rebuild compares the exact generated JSON. It cannot detect aliasing in uninspected outside data or near-identical acquisitions.

## Stage B decision

Start a simple adduct-/precursor-mass-gated spectral search on these fixed proxies, evaluate connectivity-level candidate recall@25 and MRR@25 with the local scorer, and report results by source. Compare peak curation and scoring variants only against the same folds; the observed fragment-above-precursor and precursor-error outliers argue against an unexamined global filter. Require a frozen licensed catalog and membership audit before reporting class-2 coverage; defer class-3 performance claims until the complete novelty exclusions are auditable. No model was trained and no Kaggle submission was made in Stage A.
