# vh2: clean hold-out harness kernel (never submitted)

A private Kaggle kernel that runs the full gnx11 pipeline (public 0.505 line, `submissions/gnx11-0.505/`) on a synthetic
test made of 300 train molecules held out of `train.parquet`, and scores every stage. Kernel:
`huseyinemreaksoy/casmi26-vh2-gnx11` (version 4 is the run behind the numbers below; about 4 h 50 min on one T4).

## What it does

1. **Builder cell** (`vh2_cell_builder.py`, added after the session cell): writes a synthetic `test.parquet`,
   `sample_submission.csv` and a reduced `train.parquet` without any row of the held-out structures (by InChIKey-14),
   then points the pipeline at them. The selection is embedded (`vh2_selection.json`), so the kernel needs no RDKit here.
2. **Pre-flight cell** (`vh2_cell_preflight.py`): aborts in the first minutes if an engine would skip a molecule
   (ids match, every adduct has a neutral-mass rule, no empty spectrum, target within 10 ppm of the answer, a pool
   structure inside engine 2's 30 ppm window, mass under the largest that passed before).
3. **Clean main engine** (`VH_CLEAN_MAIN`): the frozen engine loads `fpnet_fold0.pt`
   (dmitriigluzdov/casmi26-fold-safe-fpnet) in place of `fpnet_full1`, and the hold-out is drawn from fold 0, so the
   engine's FPNet never saw these molecules.
4. **Snapshots** of `submission.csv` after each stage: pre-network, network, extlib, P5, FR, APRIOR, UNIQ, NPX, E180X 1.1.
5. **Scoring cell** (`vh2_cell_score.py`): MRR@25 and top-1 per group for every snapshot (`vh/scores.json`).
6. **Cleanup cell** (`vh2_cell_cleanup.py`): deletes work folders so the output lists in one page.

Groups: `nphard` 150 (natural products in an NP table, no external reference spectrum, query spectra within 5 ppm),
`npref0` 35 (np-examples answers without an external reference), `e180` 115 (enveda-180 controls).

## Result (version 4, gnx11)

| group | final MRR | top-1 |
|---|---|---|
| nphard | 0.716 | 87 / 150 |
| npref0 | 0.862 | 27 / 35 |
| e180 | 0.859 | 88 / 115 |

NPX off: -0.005 on nphard (2 up / 6 down). The public 0.506 notebook on the same molecules: nphard -0.001, e180 -0.024.

## What it cannot do

Every hold-out answer is still a structure the static pool marks as a train structure (`pool_meta` src 1 / 3), and the
main engine ranks marked structures far higher (mean engine rank 4.6 against 32.7 on nphard). Rules that move rank 1 by
trusting the engine look good here and lose on the board: the gated duel was +0.010 / +0.043 / +0.004 here and scored
public 0.478 against 0.505 (`docs/experiments.md` rows 167-168). Use this harness for harm checks, stage ablations and
base-against-base comparisons, not to validate rank-1 rules. It also cannot measure candidate-coverage gains: the
answers are always in the pool.

Engine 2 skips a molecule with no pool structure within 30 ppm of its neutral-mass target (its pool stops at 1,300 Da);
the pipeline then stops at "Engine 2 did not cover all active molecules". The pre-flight exists because of that.

## Files

| file | purpose |
|---|---|
| `kernel/casmi26-vh2-gnx11.ipynb`, `kernel/kernel-metadata.json` | the exact kernel pushed as version 4 |
| `build_vh2.py` | builds `kernel/` from `submissions/gnx11-0.505/notebook.ipynb`; reproduces the pushed notebook byte for byte |
| `vh2_cell_builder.py`, `vh2_cell_preflight.py`, `vh2_cell_score.py`, `vh2_cell_cleanup.py` | the cells the builder inserts |
| `vh2_selection.json` | the 300 held-out structures (InChIKey-14, exact mass, metric key) and the selection limits |
| `vh2_select.py` | how the selection was drawn; needs the local caches it names (not in git) and `data/input` |
| `vh2_preflight_check.py` | local diagnosis of engine 2's mass window for a synthetic test |
| `vh2_score.py` | offline replay of rank-1 rules and ablations on a fetched run; needs cell sources kept in `scratch/` |
| `vh_fetch.py` | fetches only `vh/` from the kernel output (large pages, back-off on HTTP 429) |

Run: `python -I solutions/fusion-v4n/kaggle/vh2/build_vh2.py`, then `kaggle kernels push -p solutions/fusion-v4n/kaggle/vh2/kernel`,
then `python solutions/fusion-v4n/kaggle/vh2/vh_fetch.py huseyinemreaksoy/casmi26-vh2-gnx11 <out_dir>`.
