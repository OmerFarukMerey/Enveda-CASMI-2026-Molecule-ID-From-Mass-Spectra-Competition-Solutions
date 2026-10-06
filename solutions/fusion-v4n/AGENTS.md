<!-- index: fusion-v4n for agents: where to change what, invariants, checks before handing back -->
# AGENTS.md: solutions/fusion-v4n

Repo-wide rules: [../../AGENTS.md](../../AGENTS.md). Machine-readable manifest: [solution.json](solution.json).

## Mental model
A Kaggle notebook assembled from `nb/*.py` (cells, run in filename order) plus `nb/embed/**` (modules written to
`/kaggle/working/ours/` at runtime). Heavy models are **not** in this repo: they come from 13 public Kaggle datasets
(`solution.json → kaggle.datasets`) mounted under `/kaggle/input`.

## Where to change what
| Goal | Edit |
|---|---|
| Tune a number or toggle a stage | `nb/00_config.py` (`CFG`), or `build_notebook.py --set KEY=VALUE` for a one-off variant |
| Change how final lists are merged or fused | `nb/embed/fusion_core.py`, then run `tests/test_fusion_equivalence.py` (it must still pass with the default CFG) |
| Add a re-scoring term | `nb/07_rescore.py` (produce `{mid: {smiles: score}}`) + `fusion_core.build_submission` (consume it) + a `CFG` flag |
| Engine-2 behaviour | `nb/embed/eng/eng_runner.py` (knobs arrive via `$CASMI_ENG_CFG`). Leave `pv.py`, `pv_fp.py`, `casmi_engine.py` verbatim unless necessary |
| PubChem channel | `nb/embed/pc/probe_core2.py`, `pc_runner.py` |
| PubChem join (candidates ranked by the main engine) | `nb/embed/pc_join.py`, called from `nb/06_base_lists.py`; flags `PC_JOIN_*` |
| The second ranker below the library gate | `nb/embed/fs_pass.py` (`RANKER='fs' / 'fs_avg'`, `RANKER_LIB_GATE`); with the flag off the code path must stay bit-identical (`tests/test_fs_pass.py`) |
| Replay the final stages locally with real forward-model scores | `reviews/fwd-replay-2026-10-05.md` (scripts under `scratch/fwd/`, not in git) |
| Local tooling | `tools/*.py`. Paths come from `tools/paths.py` |

## Invariants (do not break)
- The default `CFG` must reproduce the reference (LB 0.399). New behaviour goes behind a flag whose default keeps the
  old behaviour, and the equivalence test must pass.
- `reference/v4n/` is the verbatim upstream snapshot. Never edit it; add `reference/<new-version>/` for upstream updates.
- Anything that depends on test masses or formulas is computed inside the run, never precomputed from the visible test.
- Stage outputs stay JSON-serialisable (they are dumped for `tools/refuse.py`).
- 9 h budget on a T4: ICE 5400 s + GL 4000 s + engine-2 + PubChem + v1. Raising `TOPN`, `ICE_UNION_K` or `PC_N1` costs time.
- `formula_of`, `chem.score_key` and RDKit 2026.03.3 define the keys. Do not swap in another canonicaliser.

## Check before you hand back
```bash
python solutions/fusion-v4n/tests/test_fusion_equivalence.py
python solutions/fusion-v4n/tools/build_notebook.py        # must parse all cells
python tools/gen_index.py                                  # if files were added or removed
```
Log LB results in `EXPERIMENTS.md` and `../../docs/experiments.md`.
