<!-- index: arsokan-fable: Arsokan's line on the public 0.487 / 0.492 notebooks, TA-1 train-analog rerank, offline harnesses, findings -->
# arsokan-fable

Work from a separate Arsokan session (2026-11). Base: copies of two public notebooks
(ozertuu "From Peaks to Structures", 0.487; mathleteutd "0.492 Test-Set Network Rerank", 0.492), plus new stages
measured offline. The submitted notebooks are kept as files: n0 in `submissions/n0-0.492/`, t0 in `kaggle/t0/`.

Submission log and findings: [EXPERIMENTS.md](EXPERIMENTS.md). Validation harness and how to read it:
[VALIDATION-HARNESS.md](VALIDATION-HARNESS.md). Stage descriptions: [stages/README.md](stages/README.md).

## Layout

```
stages/train_analog_rerank.py       TA-1, new stage added to the pipeline (submission t0)
stages/pc_wide_rerank_NEGATIVE.py   PubChem-wide rerank: tried, no gain on the hidden test (w0-w3)
offline/net_bench.py, net_eval.py, net_grid.py   offline harness of the test-set network step
offline/an_prep.py, an_eval3.py, an_eval5.py     offline measurement of TA-1
offline/pc_ranker2.py, libval.py, sim_frigid.py  other measurements (PubChem ranker, 2nd spectrum library, FRIGID)
offline/dl_out.py                   helper to download a Kaggle kernel's output (uses the local Kaggle API config;
                                    no credentials in the repo)
lab/pcrank_heldout_lab.py, lab/frigid_lab.py     experiment scripts that ran as Kaggle notebooks (not submissions)
lab/pcwin_lab.py                    lab: are the truths our lists miss in the PubChem mass index at all?
lab/pcw_stage_standalone_test.py    standalone Kaggle test of the PubChem-wide stage (w0-w3)
offline/an_eval2.py, an_eval4.py, an_prep30.py, ta_local.py   more TA-1 measurement (variants, K 30, local run)
offline/net_bench_prep.py           builds the network-harness inputs from pccand.parquet (before net_bench.py)
offline/pc_ranker.py                first PubChem-window ranker (pc_ranker2.py is the later one)
offline/libscan.py                  best extlib2 match per test molecule
offline/sim_gen.py                  GEN (positional isomers of listed candidates) end-to-end simulation
offline/lab_mass.py                 romanrozen's timsTOF mass prior on a validation replay
kaggle/t0/                          exact t0 notebook (n0 + TA-1) and kernel-metadata (owner redacted)
```

## Deviations from the layout contract

This folder holds notebook cells and measurement scripts, not a buildable notebook: there is no `tools/paths.py` or
`tests/`, `kaggle/` holds the exact submitted t0 notebook (no builder), and `solution.json` has no build / push / test entry points or `kernel_slug`.

## Rebuilding the offline harnesses

The author's `/root/...` files are not in the repo. What is needed:

- Competition data (`train.parquet` ~2.5M rows, `test.parquet`). Never read `train.parquet` whole; use
  `pyarrow.dataset` with a filter and small batches (8 GB machines ran out of memory twice).
- Harness output: kernel `xiangqizhang/casmi26-fold0-harness` -> `submission.csv`, `harness/truth.parquet`,
  `harness/test.parquet`. Download with `offline/dl_out.py` (the CLI hangs on large outputs).
- TA-1: `offline/an_prep.py` -> `an_eval3.py` (variants: `an_eval2.py`, `an_eval4.py`, `an_prep30.py`, `an_eval5.py`;
  `ta_local.py` runs the stage locally). A ready feature table `harness_ta_features.parquet` exists in the handoff
  zip but is not committed (data).
- Network harness: `lab/pcrank_heldout_lab.py` on Kaggle (GPU, ~2 h) writes `pccand.parquet` -> `net_bench_prep.py` ->
  `net_bench.py` -> `net_eval.py` / `net_grid.py`.
- Leak-free measurement: the public models saw every molecule of casmi-val-v2/v3 (all are in `train.parquet`). Use
  the dmitriigluzdov fold-safe FPNet (`split.parquet` fold=='hold', `fpnet_fold0.pt`) and sanpier
  `best_validation_model.pt` (`molecules.parquet` split=='validation'); their intersection is 3,619 enveda molecules.

## Paths: adjust to your environment

The `offline/` scripts contain absolute paths of the author's working machine (`/root/pipe`, `/root/lab/...`,
`/root/data/train.parquet`, `/root/ext/...`). The code is kept exactly as it ran; **adjust these paths to your
environment** before running (e.g. point them at `data/` of this repo). They do not use this repo's `tools/paths.py`.
The `stages/` and `lab/` scripts use Kaggle paths (`/kaggle/input`, `/kaggle/working`).

The scripts have no module docstrings (kept unchanged); their INDEX.md lines come from the OVERRIDES table in
`tools/gen_index.py`.

## External artifacts and provenance

- `lab/frigid_lab.py` clones `coleygroup/FRIGID` at run time (internet on) and loads `DLM.ckpt` / `mist_msg.pt`
  from Kaggle inputs; method follows lszlst's public "de novo hedge" cell. Lab only, not in any submission.
- p1 / `offline/libval.py`: second spectrum library `nakhun/casmi26-extlib2` (Kaggle dataset). Not in the pipeline.
- The base notebooks attach the external spectrum library `takumuhata/casmi26-extlib`: **its source and licence are
  not documented** (see Open questions). Note: the overlay cell of n0/t0 calls it "CC0" and
  `submissions/ax13n-0.488/README.txt` says "CC0 (GNPS + MassBank + MoNA, no NIST per its card)"; the team still has
  to confirm this against the dataset card and the competition rule.

## Open questions

- **External data / licence:** the external spectrum library used (`takumuhata/casmi26-extlib`) has no source or
  licence information. The competition's external-data rule must be checked by the team before the final selection.
