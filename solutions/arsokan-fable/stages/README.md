<!-- index: arsokan-fable pipeline stages: TA-1 train-analog evidence rerank (added) and the PubChem-wide rerank (negative) -->
# stages/

Cells that are pasted into the submission notebook (base: n0, the copy of mathleteutd's "0.492 Test-Set Network
Rerank"). Each one reads `/kaggle/working/submission.csv`, re-orders lists, and writes it back.

| File | Status |
|---|---|
| `train_analog_rerank.py` | **TA-1**, added to the pipeline (submission t0, public score pending) |
| `pc_wide_rerank_NEGATIVE.py` | tried (w0–w3), no gain on the hidden test; kept for reference, do not add |

## TA-1: train-analog evidence rerank

- For every candidate in a list, find the 8 most similar structures in the training set (Morgan Tanimoto 0.40–0.85).
  Training molecules at ≥ 0.85 are left out on purpose, so a candidate is not rewarded just for being a training
  molecule.
  Code detail: the 8 nearest training structures are taken first, those below 0.40 are dropped, and those at ≥ 0.85
  are skipped when evidence is summed, so near-identical molecules can use up some of the 8 slots.
- The more those analogs' measured spectra resemble the query spectrum (modified cosine), the more evidence the
  candidate gets. A right candidate's analogs fragment alike.
  Code detail: evidence = sum over analogs and shared adducts ([M+H]+, [M-H]-) of Tanimoto × modified cosine, using
  only pairs with at least 4 matched peaks.
- The list is re-ordered by `z(-log rank) + 0.75 * z(evidence)` (`TA_LAM = 0.75`).
- Gate: if the rank-1 candidate is a training molecule whose own spectrum matches the query at ≥ 0.6 (`TA_GATE`),
  the list is not touched. "Training molecule" here means a fingerprint neighbour with Tanimoto ≥ 0.999 (not an
  InChIKey match), with at least 4 matched peaks.
- Any failure leaves `submission.csv` unchanged: the new file is written to `submission.csv.tmp_ta` and swapped in
  with `os.replace` only at the end (the previous list is kept as `submission_pre_ta.csv`).
- **The file only defines `_ta_run()`; it does not call it.** In the notebook, call it inside a `try/except` so an
  error cannot stop the run (same pattern as the end of `pc_wide_rerank_NEGATIVE.py`):

  ```python
  try:
      _ta_run()
  except Exception as _e:
      import traceback; print('[ta] FAILED, submission untouched:', repr(_e)[:300]); traceback.print_exc()
  ```

  The exact call line used in the t0 notebook is not in this folder; confirm it against that notebook.
- Placement in the notebook: right after the network step (cell n1c), before the external-library overlay.

Offline measurement: `../offline/an_prep.py`, `an_eval3.py`, `an_eval5.py`; numbers in `../EXPERIMENTS.md`.
