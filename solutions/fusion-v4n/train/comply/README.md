<!-- index: training code and provenance of our engine-1 replacements for stack1 compliance step 2 (fragnet stage A / B refit, engine-row runner, 141-feature rows, LightGBM ranker replacing ranker_v4b); data, commands, what is and is not reproducible -->
# Compliance step 2: training code of our engine-1 parts

Context: `docs/competition/stack1-compliance-2026-10-17.md` (inventory C1 to C5, C9 to C11) and `scratch/comply/step2/PLAN.md`.
The host's rule (thread 748882): third-party weights must be reproducible with their original training code and permitted
data. These scripts produce every weight that step 2 puts into stack1's engine 1 in place of the third-party ones.

| Weight (dataset `huseyinemreaksoy/casmi26-ours-e1` unless noted) | Replaces | Script | Data |
|---|---|---|---|
| `fpnet_own_s0.pt` (in `casmi26-ours-pcnets`, step 1) = `base_s0` | `fpnet_full1.pt` (engine bank), `fe_A/B/D.pt` (model_views slots) | `clean/train_fpnet.py` (ours; loop derived from the Apache-2.0 fold-safe loop, `THIRD_PARTY_NOTICES.md`) | `train.parquet` only; review `reviews/ownfp-2026-10-10.md` |
| `fragnet_stageA.npz` | `fe_v4/res/fragnet/stageA.npz` | `fit_fragnet.py` | `countsA.npz` shipped in `ahmedberatozer/casmi26-v4b-models` (MIT; card: built only from `train.parquet` and COCONUT) |
| `fragnet_stageB.txt` | `fe_v4/res/fragnet/stageB.txt` | `fit_fragnet.py` | `featB.npz` shipped in the same dataset, part B |
| `e1_ranker_s0..3.txt`, `e1_features.json` | `ranker_v4b.pkl` | `e1_engine_rows.py` -> `e1_rows.py` -> `fit_ranker_e1.py` | engine rows of `train.parquet` molecules (answer hidden) against the pool of `casmi26-v2-pool` (`train.parquet` + COCONUT) |
| (none) | `fe_v4/res/derivation/prior_tables.pkl` | dropped: the `derivation` family (19 features) is removed from engine 1 | |

No NIST data and no third-party weights enter any of these fits: the engine bank and view slots hold our network, stage A / B
are ours, and the public ranker is never read (its stored hyper-parameter values are re-typed in `fit_ranker_e1.py`).
Code used as a library: `casmi` and `fe_v4` of `casmi26-v4b-models` (MIT since 2026-10-13: engine, families, `fit_chain`,
`stageB_features`); `prototypes/local_engine.py` (ours: query spectra, masks).

## Files
- `fit_fragnet.py`: stage A (smoothed LLR tables from the shipped count tables, chains A and H, alpha 80) and stage B
  (LightGBM lambdarank, 300 rounds, the parameters recorded in the shipped model). Stage A comes out byte-identical to the
  shipped table; stage B is equivalent, not identical (step 0: prediction correlation 0.998, within-group MRR 0.800 against 0.799).
- `e1_engine_rows.py`: runs the engine exactly as the step-2 notebook builds it on a molecule table (versions c2 / c1 / drop:
  which library spectra and pool row of the answer are hidden) and stores every candidate row.
- `e1_rows.py`: rows -> the 141-feature matrix: the seven tie-dependent rank features recomputed with average ties (the engine's
  stable ties encode pool row order), fragnet recomputed with our stage A / B, derivation dropped. `parity` mode: with the
  shipped stage A / B it reproduces the stored fragnet columns exactly (checked 2026-10-17 on 70 lists, max |diff| 0.0).
- `fit_ranker_e1.py`: `feat` (caches), `fit` (final ranker: the public ranker's hyper-parameters, 500 rounds, seeds 0 to 3, no
  folds), `cv` (out-of-fold scores, 5 scaffold folds, for the local replay only).

## Commands (CPU; paths of this machine)
```bash
P=.venv/Scripts/python; T=solutions/fusion-v4n/train/comply; DS=C:/Users/hemre/casmi-work/comply/step2/ds/casmi26-ours-e1
$P $T/fit_fragnet.py --code D:/casmi-data/ranker_study/ds/v4b/code --out $DS            # fragnet_stageA.npz / stageB.txt are copies of its output
# rows: one run per (set, version); the P4 runs of the ownfp study + the n0 runs of scratch/comply/step2/queue.py are equivalent
$P $T/e1_engine_rows.py --mols <set>/mols.parquet --queries <set>/queries.parquet --ver c2 --net <base_s0_final.pt> \
    --stage_a $DS/fragnet_stageA.npz --stage_b $DS/fragnet_stageB.txt --out <rows dir> --worker 0/4
$P $T/fit_ranker_e1.py feat --workers 2
$P $T/fit_ranker_e1.py fit --out <dir> --threads 4
```
The molecule sets (hold-out c2 / c1, training plan, mixed regime, Enveda panel) are subsets of `train.parquet`; their tables
(mid, the `train.parquet` rows of the query spectra, answer key, pool row, query library) ship in the dataset as `e1_sets.parquet`.

## Limits (stated, not hidden)
- `countsA.npz` / `featB.npz` are training data built by the dataset author from `train.parquet` and COCONUT; their builder is
  not shipped. Stage A / B are reproducible from them with our driver, not from raw `train.parquet`.
- The rows of the hold-out, training-plan and panel sets were written by our earlier scripts (`scratch/ranker/run_engine.py`,
  `scratch/pcjoin/run_join.py`) with the same engine, masks and network; `e1_engine_rows.py` is their no-join path without the
  public ranker and the derivation family. Agreement of the two on the ranker's features is checked in the step-2 notes.
- LightGBM results depend on its version and thread count; the fit used LightGBM 4.6.0.
