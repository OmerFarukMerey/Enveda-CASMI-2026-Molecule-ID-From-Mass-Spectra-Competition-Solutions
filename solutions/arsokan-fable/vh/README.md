# vh: validation harness for the n0 pipeline

Paths inside the scripts are those of the container they were written in (repo root `/home/user/Enveda-CASMI-2026-...-Solutions/`, working dir `scratch/vh/`, stage notebooks of vpd / w30c under `scratch/om-vpd`, `scratch/om-w30c`); adjust the `R=` line and run from one working directory.

Runs the n0 notebook (`submissions/n0-0.492/notebook.ipynb`) on held-out training molecules with a reduced train file,
and replays later re-ranking stages offline on the lists it saves. Built 2026-11-17. See NOTES.md for decisions.

**What it is good for.** A loss detector: judge a re-ranker by its loss side and by "correct rank 1 demoted" per
class; treat the gain side as an upper bound. It is NOT a net-score predictor (class-2 truths are always in the
candidate pool here, so gains from lifting a truth at rank 2-5 are inflated). On the four changes whose hidden-test
effect is known (N-1c LAM > 0.40, TA-1, vpd, w30c) the loss-side reading was right four times out of four.

## Files

| File | Purpose |
|---|---|
| `select_pool.py` | Eligible pool (cross-lab molecules; enveda-only molecules in FPNet hold and sanpier validation), their query spectra, the N-1c graph on the pool. Needs `train_meta.parquet` (columns of train + row index) and the two split tables in `ext/`. |
| `select_final.py` | Stratified, network-enriched sample: `selection.json` (300) and `selection_smoke.json` (16). |
| `cell_builder.py` | Notebook cell: streams train.parquet, writes reduced train + held-out test + sample submission to /tmp/vh_pool, sets `CASMI_*` env vars, asserts class-2 keys are gone and class-1 keys remain. |
| `cell_score.py` | Notebook cell: scores every `submission*.csv` snapshot per class with the metric key. |
| `build_nb.py` | Builds the harness notebook from n0: redirect in the session cell, three path patches in the embedded pipeline (hash re-pinned), N-1c and overlay read `COMP/test.parquet`, builder + scoring cells. |
| `replay.py` | Offline acceptance replay: N-1c LAM curve, TA-1, vpd, w30c on the kernel's saved lists. `python replay.py <out_dir> <reduced train.parquet>`. |
| `audit.py`, `task10.py`, `task10b.py` | Stage-by-stage audit of n0, error anatomy, rule tests. |

## Rebuild

1. `python select_pool.py && python select_final.py` (needs the competition train file under `data/input/...`).
2. `python build_nb.py selection_smoke.json nb/casmi-vh-smoke.ipynb`; same for `selection.json`.
3. Kernel metadata: copy `submissions/n0-0.492/kernel-metadata.json`, set `id` to `YOUR_KAGGLE_USERNAME/casmi-vh-<tag>`,
   `is_private` true, `enable_internet` false. Same datasets and kernel source as n0. Push privately; never submit.
4. Smoke (16 molecules, ~25 min on T4) then full (300 molecules, ~2 h 40 min).
5. Download `vh/*`, `submission*.csv`, `variants/*.csv`, `frozen_candidate_evidence.parquet`; rebuild the reduced
   train locally by running the builder cell with local paths; run `replay.py`.

## Classes (300 molecules, real timsTOF query spectra, visible-test adduct set)

- C1x (110): molecule also in another library; all its enveda-180 rows leave train, the Orbitrap rows stay.
- C1s (60): enveda-only; the 40 / 60 eV rows are the query and leave train, the 20 eV and stepped rows stay.
- C2 (130): every row of the molecule leaves train. The truth stays in the candidate pool (known limitation).
- Class 3 is not simulated.

## Known limits

Neural scorers, fpnet_full1 and the rankers were trained on the full train file; the candidate pool contains every
training structure. Level (MRR 0.856) is far above the hidden test (0.492). One fold, one seed, 300 molecules.
