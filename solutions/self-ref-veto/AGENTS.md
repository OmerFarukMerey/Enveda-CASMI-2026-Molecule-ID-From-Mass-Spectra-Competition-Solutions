<!-- index: self-ref-veto for agents: what the cells do, where to change what, invariants, checks before a submission -->
# AGENTS.md: solutions/self-ref-veto

Repo-wide rules: [../../AGENTS.md](../../AGENTS.md). Manifest: [solution.json](solution.json).

## Mental model
`tools/build.py` takes the public 0.492 notebook in `reference/` (verbatim, Apache-2.0), inserts the cells from
`cells/` and writes a private kernel folder to `kaggle/<slug>/`. Every variant is one function in `VARIANTS`; the base
notebook is never edited by hand.

## Where to change what
| Goal | Edit |
|---|---|
| Veto / confirmation behaviour | `cells/veto.py` (parameters `V1_TAU`, `V1_CAP`, `V2_ON`, `V2_CONF`, `V2_MARGIN`, `V1_LIB_TAU`) |
| The exact cell submitted as vpd | `cells/v1_veto_pm.py` (frozen: do not edit; it must match the submitted notebook) |
| Post-overlay deduplication | `cells/dedup_after_overlay.py` |
| Train-analog evidence with a rank-1 freeze | `cells/ta_param.py` (wraps `solutions/arsokan-fable/stages/train_analog_rerank.py`) |
| Leader-safe network rerank | `_n1c_leader_safe` in `tools/build.py` (a two-line patch of the N-1c cell) |
| A new variant | a new `@variant` in `tools/build.py` |
| Safety / power numbers | `analysis/np_veto_measures.py <competition_dir> [refs_per_adduct]` (writes a CSV to the current dir: run it from `scratch/`) |

## Invariants
- The veto never promotes a known structure by itself and never moves a candidate without own spectra down.
- Lists with `library_max >= 0.9` stay untouched; the external-library overlay runs after the veto.
- Every cell is wrapped in try / except and leaves `submission.csv` unchanged on failure.
- Do not tune thresholds on the visible `test.parquet` (drawn from train). Thresholds come from the leak-free NP panel.

## Check before a submission
```bash
python -I solutions/self-ref-veto/tools/build.py <variant>        # every code cell must parse
# local plumbing test (CPU): harness from the visible test, then run the cell against it
python -I solutions/self-ref-veto/tools/mkharness.py <competition_dir> scratch/harness
python -I solutions/self-ref-veto/tools/runcell.py scratch/harness solutions/self-ref-veto/cells/veto.py V1_TAU=0.40
```
Then push (smoke commit), read the log (`tools/getout.py`) for the `[v1]` and `[d1]` lines, and only then submit. Log the
result in `EXPERIMENTS.md` and `../../docs/experiments.md`.
