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

t0: Kaggle id 58419263, submitted 2026-11-16 09:21 Istanbul. Notebook `kaggle/t0/` (= n0 + one cell after N-1c).
n0 notebook: `submissions/n0-0.492/`. Older forks of earlier public notebooks, of no further value: g0 0.461, h0 0.483,
h1 0.478.

## Findings

Short list first (as agreed for the repo), then the detail from the handoff.


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

## Detail (handoff of 2026-11-16)

All measured; do not repeat.

1. **FRIGID** (coleygroup/FRIGID): putting its output at rank 2 hurts on all three sets (−0.013..−0.040). The owner of
   the reference implementation is at 0.480 too. CLOSED.
2. **GEN** (positional isomers of listed candidates, `offline/sim_gen.py`): 0 gain on an independent test. CLOSED.
3. **Second spectrum library (extlib2):** accuracy also 14% at cos 0.72–0.80. p2 (weak matches at rank 2) was not
   submitted. CLOSED.
4. **Full PubChem** (dmitriigluzdov/casmi26-pubchem-popularity-prior, 106M structures): the right structure is in the
   index for 97% of the 1,300 molecules; fingerprint model `fpnet_fold0` MRR 0.380; blend with sanpier's validation
   model (zf + 2*zs) 0.496, top-1 38%. No hidden-test gain (w0–w3): the pipeline already uses the PubChem channel with
   5 slots and already puts the same candidate first when it is sure. CLOSED.
   Side finding: the pipeline's PubChem promotion rule requires pop ≥ 5; only 0.5% of enveda-type truths meet it.
5. **Network step N-1c:** offline LAM curve has the same shape as the public scores (peak 0.35–0.5).
6. **TA-1:** Morgan r2 / 2048 bits. Structural similarity alone (no spectra) hurts, so the gain comes from spectral
   agreement. Copies of the truth at ≥ 0.95 were removed. Variants (K 8/15/30, lower bound 0.25–0.5, sum vs best,
   falling back to other adducts) all give 0.65–0.67: tuning does not grow it. Runtime ~9 min.
   Caveats: the harness base is leaky (its models saw those molecules), and the file holds only class 2; about 45% of
   the hidden test is said to be class 2. Expectation +0.01..+0.03, but in item 4 an offline gain did not carry over.
   Different from the repo's "ae" (analog edit) work: ae generates new structures, TA-1 re-orders the existing list.
7. **Other:** in the harness the truth is in 96.5% of lists but at rank 1 in only 45%. Weight / threshold tweaks in the
   public 0.487 cluster (teams with 70–150 submissions) move nothing. The visible test.parquet has 400 molecules
   locally, 120 inside Kaggle.

## After t0

- t0 > 0.492: TA-1 works on the hidden test. Next candidates (offline first): ungated (0.677), LAM 0.5 / 1.0,
  fallback to other adducts (+0.005), TA-1 + N-1c through one learned combiner. One submission a day, best single
  candidate.
- t0 ≤ 0.492: most likely it breaks library-found (class 1) lists, or the hidden test has few similar training
  molecules. Checks: re-order only ranks 2–25 (offline +0.008) or tighten the gate.

