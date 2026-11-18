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
| t0 | n0 + TA-1 (train-analog evidence) | 0.461 |
| lmg1 | n0 + leader-move guard LMG-1 (`vh/cell_lmg.py`; Kaggle id 58494853, 2026-11-18) | pending |

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
  0.617 → 0.660 (gate on), 0.677 without the gate; 95% CI +0.041..+0.078; 75 improved, 28 worse. Public score
  0.461 (t0), −0.031 against n0: see "t0 result" below. (`stages/train_analog_rerank.py`, `offline/an_prep.py`, `an_eval3.py`, `an_eval5.py`)

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

## After t0 (plan written before the score; superseded by "t0 result" below)

- t0 > 0.492: TA-1 works on the hidden test. Next candidates (offline first): ungated (0.677), LAM 0.5 / 1.0,
  fallback to other adducts (+0.005), TA-1 + N-1c through one learned combiner. One submission a day, best single
  candidate.
- t0 ≤ 0.492: most likely it breaks library-found (class 1) lists, or the hidden test has few similar training
  molecules. Checks: re-order only ranks 2–25 (offline +0.008) or tighten the gate.


## TA-1 re-measured (2026-11-16, second session)

Stage code as committed, harness lists (400 class-2 molecules), RDKit 2026.03.3, metric key from
`solutions/fusion-v4n/tools/metric.py`, paired bootstrap (10,000 resamples). Working files not in git.

**The headline reproduces: 0.6168 → 0.6602 (gate on) on the full train file.** But that run is not the hidden-test
situation: all 400 truths are in `train.parquet`, the gate fires on exactly the 180 lists whose rank 1 is the truth,
so a correct rank 1 can never be demoted there. With the truth removed from the analog pool (per molecule: truth key
plus training structures at Tanimoto ≥ 0.95 to it), the numbers are:

| Variant (leak-free pool) | MRR | Δ | 95% CI | win / loss | correct rank 1 demoted |
|---|---|---|---|---|---|
| base | 0.6168 | | | | |
| gate 0.6, LAM 0.75 (= t0) | 0.6636 | +0.0468 | +0.0297..+0.0643 | 54 / 17 | 6 |
| ungated | 0.6825 | +0.0657 | +0.0470..+0.0850 | 79 / 23 | 6 |
| ranks 2–25 only, gate on | 0.6250 | +0.0083 | +0.0026..+0.0142 | 25 / 14 | 0 |
| ranks 2–25 only, ungated | 0.6273 | +0.0105 | +0.0039..+0.0171 | 37 / 20 | 0 |
| gate 0.4 / 0.5 / 0.7 | 0.6545 / 0.6565 / 0.6685 | | | | 6 each |
| LAM 0.5, gate on | 0.6596 | +0.0429 | +0.0290..+0.0574 | 48 / 10 | 1 |
| LAM 1.0, gate on | 0.6645 | +0.0477 | +0.0273..+0.0687 | 55 / 26 | 11 |
| LAM 0.5, ungated | 0.6720 | +0.0553 | +0.0403..+0.0708 | 65 / 13 | 1 |
| LAM 1.0, ungated | 0.6853 | +0.0686 | +0.0461..+0.0915 | 81 / 34 | 11 |

By base rank 1 (leak-free, t0 settings): rank 1 = truth (180 lists) −0.0167, 0 wins / 6 losses; rank 1 = another
training structure (149) +0.0478; rank 1 not a training structure (71) +0.2057.

What it means:

- Most of the gain is rank-1 swaps (fixing rank 1 keeps +0.008..+0.010 of +0.047..+0.066), and the largest gain is
  where rank 1 is not a training structure.
- TA-1 does demote a correct rank 1 on class-2 lists (6 of 180), and the gate cannot stop it: a class-2 truth has no
  training twin to trigger it. On this file the gate only blocks gains (ungated is better at every setting).
- This file cannot show what the gate is for (class-1 lists, rank 1 a right library hit). The only hint: on the full
  train file, ungating demotes 14 truths that are in train (0.6505 vs 0.6602 gated). So the gate stays for class 1.
- LAM 0.5 is the safest shape: nearly the same total, best win / loss ratio, 1 correct rank 1 demoted.
- 135 of the 400 lists are never touched: no [M+H]+ or [M-H]- spectrum (base MRR 0.500 there vs 0.676 on the 265).
- Note on the short list above: the "95% CI +0.041..+0.078; 75 improved, 28 worse" figures match the **ungated**
  variant in the truth-removed setup (76 / 27, +0.0415..+0.0796), not the gated one.
- Caveats unchanged: base lists are leaky (the models saw these molecules), class 2 only, one fold, 400 molecules.

## t0 result (2026-11-16): 0.461, −0.031 against n0

TA-1 as written is harmful on the hidden test. n0 (0.492) stays the team best; t0 is not for final selection.
Hypothesis (not proven) is in `stages/README.md`. Diagnostic: with the truths left in train the harness still gives
0.660, so that setup cannot see the damage. **A class-2-only harness measurement is not a reason to submit.**

### How offline measurements carried over to the hidden test (this line)

| Measurement | Offline | Public | Carried over |
|---|---|---|---|
| Network step LAM curve | same shape | same shape | yes |
| extlib2 match accuracy | low accuracy | −0.015 | yes (a small plus had been expected) |
| Full PubChem rerank | +0.12 | 0 / −0.002 | no |
| TA-1 | +0.04 | −0.031 | no |

Common cause of the two failures: both sets held only molecules split off from train whose truth is not in the pool;
they did not represent the hidden test's library-found (class 1) and nowhere-found (class 3) molecules. First question
for any new idea: "can it break a rank 1 that is already right?" If yes, do not submit without a class-1 measurement.

Also: among the 400 harness molecules there are only 17 network links (109 in the visible test), so the interaction
of N-1c and TA-1 cannot be measured there.

### To rescue the TA idea (all offline before any submission)

a. A measurement set with class 1: query = part of a molecule's spectra, pool = the rest; different spectra, not copies
   (other collision energy or spectrum_id). Measure on mixed class-1 + class-2 lists.
b. Design candidates: no exclusion (every neighbour gives evidence, itself included; 0.617 → 0.648 on harness class 2,
   `an_eval2` "max95/sum95"); structural gate (rank 1 in train by tautomer-canonical InChIKey14 → do not touch);
   re-order ranks 2–25 only (+0.008 on the harness, safest).
c. One candidate only if (a) shows no class-1 loss and a class-2 gain; then one submission, announced first.


## Class-1 regime on the harness (2026-11-16): the harness cannot see the hidden loss

Evidence side made class-1-like without the copy leak: full train, minus the truth's rows that are copies of the query
(row-to-row modified cosine ≥ 0.95; harness queries are perturbed copies of train rows, median cosine 0.995). 282
molecules keep a truth spectrum in train ("C1"); only 126 keep one the stage can use. Class 2 = truth and its ≥ 0.95
neighbours removed ("P2", 400). Base lists are still the leaky harness lists. `dem` = correct rank 1 demoted.

| Design | C1 (282): Δ, win / loss, dem | P2 (400): Δ, win / loss, dem |
|---|---|---|
| t0 as written | +0.0358, 29 / 7, 0 | +0.0468, 54 / 17, 6 |
| no exclusion, ungated | +0.0584, 47 / 3, 0 | +0.0384, 59 / 26, 7 |
| no exclusion, gate 0.6 | +0.0515, 36 / 1, 0 | +0.0341, 46 / 20, 7 |
| structural gate (rank-1 key in train), exclusion | +0.0264, 21 / 6, 0 | +0.0286, 35 / 15, 6 |
| structural gate, no exclusion | +0.0432, 28 / 1, 0 | +0.0251, 33 / 16, 7 |
| ranks 2–25 only, exclusion | +0.0066, 21 / 13, 0 | +0.0105, 37 / 20, 0 |
| ranks 2–25 only, no exclusion | +0.0170, 27 / 4, 0 | +0.0079, 35 / 22, 0 |
| top-3 structural gate, no exclusion | +0.0033, 2 / 0, 0 | +0.0264, 25 / 5, 3 |

- **t0 as written still gains in the class-1 regime (+0.036, no correct rank 1 demoted), yet lost 0.031 on the hidden
  test.** Every design is positive in both regimes, so this harness cannot rank TA designs by hidden-test effect, not
  even by sign. The "truth's spectra feed wrong candidates / gate fails" hypothesis is not supported offline (the
  cosine gate does miss 26 touchable correct-rank-1 lists in C1, but none is demoted).
- Likely reason the harness is blind (inference, not measured): its queries are perturbed copies of same-lab train
  spectra, so train analog spectra resemble the query by construction; a separate-lab class 1 cannot be built from it
  (dropping the copies' ingest_lib leaves 10 molecules); and the base lists are not n0's lists.
- A 0.031 loss is large (about 6% of all lists going from rank 1 to rank 2 with no gains elsewhere): it points to
  damage on lists n0 gets right.
- **Decision: no TA candidate for submission.** TA work stops until there is a test with n0's own lists and query
  spectra from a different lab than the evidence spectra. If a diagnostic slot is ever spent, "ranks 2–25 only" is the
  informative one (it cannot demote rank 1); expected score change about zero.

## Validation harness (2026-11-17)

n0's own lists on 300 held-out molecules (cross-lab class 1, same-lab class 1, class 2), replaying TA-1, vpd, w30c
and the LAM curve: see [VALIDATION-HARNESS.md](VALIDATION-HARNESS.md). It reproduces the hidden-test loss mechanisms
(LAM > 0.35, w30c's demoted cross-lab leader, vpd = 0) but not TA-1's net sign; use it as a loss detector only.
Rule from here on: a re-ranker is judged by "correct rank 1 demoted" per class and its loss side; gains are an upper
bound.

