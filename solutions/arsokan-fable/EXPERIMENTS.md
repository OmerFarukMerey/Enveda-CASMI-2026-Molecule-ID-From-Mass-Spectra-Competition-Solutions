<!-- index: arsokan-fable submission log (p0..t0) and findings: FRIGID, second library, PubChem-wide rerank, network step, TA-1 -->
# arsokan-fable: experiments

## Submissions

Public scores as reported by the session (no rounding added). The ax13 / ax13t / ax13n submissions from another
session are logged in `docs/experiments.md` (rows 129–131) and not repeated here.

| Submission | Content | Public score |
|---|---|---|
| p0 | copy of ozertuu "From Peaks to Structures" (baseline) | 0.487 |
| p1 | p0 + second external spectrum library (nakhun/casmi26-extlib2) | 0.472 |
| w0 | p0 + full PubChem rerank (5 slots + promotion) | 0.485 |
| w1 | w0, no promotion, later slots | 0.485 |
| w2 | w0, high-confidence promotion only | 0.487 |
| w3 | w0, a single candidate to rank 2/3 only | 0.486 |
| n0 | copy of mathleteutd "0.492 Test-Set Network Rerank" (LAM 0.35) | 0.492 |
| t0 | n0 + TA-1 (train-analog evidence) | pending |

## Findings

- **FRIGID (de novo generation):** found 2 of the 81 validation molecules that were not in the list; the model
  recalls what it saw in training. Closed. (`lab/frigid_lab.py`, `offline/sim_frigid.py`)
- **Second spectrum library:** on 4,000 training molecules from the same instrument, match accuracy 63% at
  similarity ≥ 0.90 and 32% at 0.80–0.90; −0.015 on the public score (p1). (`offline/libval.py`)
- **Full PubChem scan:** on 1,300 molecules neither model had seen, the fingerprint model alone gives MRR 0.380 and
  the blend with the neural scorer 0.496; but no gain on the hidden test (w0–w3). The hidden test is harder than
  molecules held out from train. (`stages/pc_wide_rerank_NEGATIVE.py`, `lab/pcrank_heldout_lab.py`,
  `offline/pc_ranker2.py`)
- **Test-set network step:** in the offline harness 22 design variants are within noise; ceiling about +0.005, and
  the public 0.492 version already takes it. (`offline/net_bench.py`, `net_eval.py`, `net_grid.py`)
- **TA-1 (train-analog evidence):** on the lists of 400 "class 2" molecules the pipeline had not seen, MRR
  0.617 → 0.660 (gate on), 0.677 without the gate; 95% CI +0.041..+0.078; 75 improved, 28 worse. Public score not
  known yet (t0). (`stages/train_analog_rerank.py`, `offline/an_prep.py`, `an_eval3.py`, `an_eval5.py`)
