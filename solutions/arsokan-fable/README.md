<!-- index: arsokan-fable: Arsokan's line on the public 0.487 / 0.492 notebooks, TA-1 train-analog rerank, offline harnesses, findings -->
# arsokan-fable

Work from a separate Arsokan session (2026-11). Base: copies of two public notebooks
(ozertuu "From Peaks to Structures", 0.487; mathleteutd "0.492 Test-Set Network Rerank", 0.492), plus new stages
measured offline. The public notebooks themselves are not in this folder; only our own stages and measurement scripts.

Submission log and findings: [EXPERIMENTS.md](EXPERIMENTS.md). Stage descriptions: [stages/README.md](stages/README.md).

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
```

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
  not documented** (see Open questions).

## Open questions

- **External data / licence:** the external spectrum library used (`takumuhata/casmi26-extlib`) has no source or
  licence information. The competition's external-data rule must be checked by the team before the final selection.
