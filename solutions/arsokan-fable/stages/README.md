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
- The more those analogs' measured spectra resemble the query spectrum (modified cosine), the more evidence the
  candidate gets. A right candidate's analogs fragment alike.
- The list is re-ordered by `z(-log rank) + 0.75 * z(evidence)` (`TA_LAM = 0.75`).
- Gate: if the rank-1 candidate is a training molecule whose own spectrum matches the query at ≥ 0.6 (`TA_GATE`),
  the list is not touched.
- Any failure leaves `submission.csv` unchanged.
- Placement in the notebook: right after the network step (cell n1c), before the external-library overlay.

Offline measurement: `../offline/an_prep.py`, `an_eval3.py`, `an_eval5.py`; numbers in `../EXPERIMENTS.md`.
