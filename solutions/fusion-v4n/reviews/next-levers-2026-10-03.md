<!-- index: fusion-v4n 2026-10-03/04 next levers on the candidate side after the PubChem join: what the local regime shows about the N=200 leaderboard drop, list depth and the first-pass shortlist, engine-2 candidates through the join or an early slot, the gate, the loss budget of truth-removed molecules, the structure generator, a natural-product prior in the selection step, and five flagged prototypes -->
# Next levers on the candidate side (2026-10-03 / 04)

Offline only. Nothing was submitted, pushed or committed, no Kaggle notebook was run, no existing file was edited, no
model was trained, the GPU was not used. Scripts: `scratch/next_levers/` (git-ignored, log in `LOG.md`). Data and
outputs: `D:\casmi-data\next_levers\` (0.5 GB). Prototypes with tests (no competition data needed):
`prototypes/eng_join.py`, `pc_window.py`, `join_depth.py`, `gen_join.py`, `pc_select.py` and their `test_*.py`
(23 tests, all pass). MRR = MRR@25 on the metric key; "±" is one standard error; differences are paired per molecule;
(wins / losses) in brackets. Reproduce: section 9.

**Leaderboard while this study ran**: v4b 0.409 (no join), v12a 0.420 (N 50), v13a **0.431** (N 100), v14a **0.406**
(N 200); all with the gate `lib_max` < 0.7 and the public ranker.

**How far to trust the numbers below.** They come from the mixed regime (380 hold-out training structures; types
`c1` = class-1-like, `c2` = truth a pool row without spectra, `dA` = truth removed from the v1 pool but a candidate of
engine-2, `dB` = removed from both; `dA` / `dB` are reported for the 325 molecules with a PubChem record unless stated).
**That regime predicted +0.010 to +0.024 on `dB` and −0.002 on `c2` / `c1` for N 100 → 200; the leaderboard lost
0.025.** Section 1 shows what the regime does and does not contain of that. Every row of the tables is marked:
**[R]** rests on the regime's end-to-end MRR (the kind of number that just mispredicted); **[C]** is a count that does
not pass through the ranker or the fusion (is the truth in the list, in the shortlist, in the window); **[T]** is a
statistic of `train.parquet`. Two replays: zero-forward (380 molecules, ICEBERG / GLACIER terms at zero) and ICEBERG
prefix (169 molecules; injected rows beyond PubChem rank 25 have no ICEBERG score). The local FPNet is the fold-safe
single model, weaker than the notebook's; engine-2's models and ICEBERG have seen the molecules.

## 0. Bottom line

1. **The N 200 drop: the regime shows the mechanism, at about a fifth of the size.** Deeper injection takes the first
   place from correct pool answers: on the v1 side 11 / 17 / 24 of 188 `c2` lists that were right at N 0 lose their
   top-1 at N 50 / 100 / 200 (three quarters of them to an injected row), `c2` MRR on the v1 side −0.001 / −0.013 /
   −0.023. Locally the engine-2 fusion repairs most of it (2 / 4 / 12 of 190 after the fusion; engine-2's models have
   seen these molecules), and the truths gained from fingerprint ranks 101 to 200 are placed badly (reciprocal rank
   0.087, none first; the training study's 0.10 is reproduced). The local net for 100 → 200 stays positive; a
   leaderboard loss of 0.025 needs a pool-side cost four to five times the local one. At N 500 the regime does show
   the loss (`c2` −0.037 ± 0.020 after the fusion, −0.058 on the v1 side). **So: a sign, yes; the size, no.** Three
   submissions of this series differ by 1.4 to 3.3 public molecules; the two diagnostic Kaggle runs decide whether
   the drop is real.
2. **Depth is not the next step. The first-pass shortlist is the one measured lever that brings more answers in
   without adding rows.** `PC_N1` 5,000 cuts the truth out of 7.7% of the PubChem-record molecules (17.9% of the
   enveda-180 stratum, the test's instrument) [C]. With 20,000 all 25 come back, 13 of them within the first 100 [C];
   end to end at N 100: +0.252 ± 0.080 (11 / 0) on those 25, −0.005 ± 0.005 on 40 controls, `c2` +0.004 ± 0.025 on 65
   lists [R]: about +0.017 on `dB`. A config change (`PC_N1=20000`), about +35 minutes.
3. **Engine-2's candidates through the join work on `dA`** (+0.170 ± 0.018, `c2` −0.004 ± 0.004, `c1` −0.003 ± 0.003
   [R]) and an early slot in the fusion gives half of that for less risk (+0.092, `c2` −0.002). But the population
   is small: engine-2's surplus over the pool is 2 to 3 candidates per window against several thousand PubChem-only
   structures. Worth a submission only as the low-risk choice.
4. **The gate and N by rule: nothing clear** (section 4). **A natural-product prior in the selection step: not
   supported end to end** (+0.003 ± 0.006 on `dB`, −0.014 ± 0.014 on `c2`), although it raises "truth among the first
   100" by 3.6 points (section 7).
5. **The generator adds to answers without a PubChem record** (+0.071 ± 0.038, 7 / 3; ICEBERG prefix +0.135 ± 0.066)
   and nothing to those with one; it costs `c2` 0.014 ± 0.007 on the v1 side and adds 13 rows per list, which is
   the mechanism of point 1. Not before the depth question is settled.
6. **Where the remaining loss sits** (`dB`, N 100, section 5): injected but ranked low 0.21 (the three order-coded
   rank features alone cost an injected truth 1.9 score units against a pool row, more than the whole gap); gated
   0.17; no PubChem record 0.11; listed beyond N 0.11; cut by the shortlist 0.06. The first and the fourth need a
   ranker fitted with injected rows; that is the training side's job and the thing that would make depth usable.

Ranked list, proposal for the next submission: section 8.

## 1. The N 200 drop: what the regime shows

`q7_depth_cost.py`, existing runs, molecules below the gate, against N 0 on the same molecules, zero-forward [R].

| `c2` (truth is a pool row), 271 lists | N 50 | N 100 | N 200 | N 500 (150 lists) | N 1000 (75) |
|---|---|---|---|---|---|
| injected rows among the first 25 of the joint list | 13.2 | 14.3 | 14.8 | 15.9 | 16.1 |
| injected rows among the final 25 / the final 5 | 11.4 / 1.8 | 11.9 / 1.9 | 12.2 / 2.0 | 12.8 / 2.3 | 13.2 / 2.3 |
| an injected row tops the v1-side list | 11.1% | 12.9% | 15.9% | 19.3% | 16.0% |
| an injected row tops the final list | 5.2% | 5.9% | 8.9% | 10.7% | 12.0% |
| lists right at N 0 that lose the first place, v1 side (to an injected row) | 11 of 188 (9) | 17 (13) | 24 (18) | 21 of 111 (16) | 7 of 58 (5) |
| the same after the engine-2 fusion | 2 of 190 (0) | 4 (1) | 12 (3) | 14 of 113 (6) | 5 of 56 (3) |
| MRR change, v1 side | −0.001 | −0.013 | −0.023 | −0.084 | −0.044 |
| MRR change, final | +0.019 | +0.018 | +0.015 | −0.035 | −0.013 |

On the 150 molecules that have every depth: `c2` final +0.009 / +0.002 / −0.006 / −0.035 and v1 side −0.016 / −0.026 /
−0.035 / −0.084 for N 50 / 100 / 200 / 500. `c1` below the gate (24 lists): v1 side −0.010 / −0.040 / −0.055, final
+0.013 / −0.008 / −0.020.

| `c2`, v1 side, MRR change against N 0 | N 50 | N 100 | N 200 |
|---|---|---|---|
| PubChem window under 3,000 rows (92 lists) | +0.012 | −0.015 | **−0.040** |
| 3,000 to 10,000 rows (87) | +0.008 | −0.012 | −0.030 |
| above 10,000 rows (92) | −0.023 | −0.012 | −0.001 |
| `lib_max` below 0.3 (123) | −0.003 | −0.025 | −0.029 |
| 0.3 to 0.5 (80) | +0.012 | 0.000 | −0.011 |
| 0.5 to 0.7 (68) | −0.014 | −0.006 | −0.028 |

| Injected truth in the joint list (`dB`, public ranker), by fingerprint rank | truths | reciprocal rank | first | top 5 | median rank |
|---|---|---|---|---|---|
| N 200 run: ranks 1 to 25 | 133 | 0.660 | 53% | 83% | 1 |
| 26 to 50 | 15 | 0.357 | 20% | 67% | 3 |
| 51 to 100 | 12 | 0.415 | 25% | 75% | 3.5 |
| **101 to 200** | 13 | **0.087** | 0% | 15% | 27 |
| N 500 run: 201 to 500 | 8 | 0.081 | 0% | 13% | 34.5 |

Reading.
- **The gain side of depth is thin.** 100 → 200 adds 4.3% of the record molecules to the list [C], and the ranker
  places them at 0.087: +0.004 per record molecule before any cost. The measured `dB` gain (+0.010 ± 0.005) is mostly
  earlier truths moving up as list-relative features shift, not new truths. My interim line "the engine's placement
  hardly depends on the fingerprint rank" held for ranks 26 to 100 (0.36 to 0.42) and is **wrong beyond 100**.
- **The cost side grows with N and is hidden by engine-2**, exactly as the join review found for class 1. On the v1
  side each doubling costs `c2` about 0.01; after the fusion the local cost is 0.003 per doubling up to 200 and then
  breaks (N 500).
- **Where it sits**: small windows (N 200 is then most of the non-pool window, so structurally close isomers of the
  answer enter) and lists without library evidence.
- **Size check.** Take the local per-type changes for 100 → 200 (`dB` +0.010, `c2` −0.003 final or −0.010 on the v1
  side, `c1` −0.002) and any mixture of the mixed-regime review: the result is between −0.004 and +0.003. To reach
  −0.025 the pool types would have to lose about 0.05 per molecule, the local `c2` cost at N 500 on the v1 side. That
  is possible if on hidden molecules (a) engine-2 does not repair (its models have not seen them), (b) the notebook's
  stronger FPNet pair hands over harder decoys than the fold-safe model, (c) more lists fall below the gate than
  locally (`lib_max` is lower on timsTOF spectra). None of the three can be tested here.
- **What would test it on Kaggle without a label**: log per molecule how many injected rows are in the final 25 and
  whether the final top-1 is an injected row (`join_depth.depth_readout`); locally an injected row tops 5.9% of the
  truth-in-pool lists at N 100 and 8.9% at N 200. If the hidden run shows several times that, the drop is real and
  N 100 is already paying part of it.

## 2. List depth and the first-pass shortlist (question 2)

`q2_decompose.py`, 325 truth-removed molecules with a PubChem record, the channel as the notebook runs it (±10 ppm,
`PC_N1` 5000) with the fold-safe FPNet [C]:

| Where the truth is | share | cheap fix |
|---|---|---|
| listed within 25 / 50 / 100 / 200 | 59.1 / 65.2 / 70.5 / 74.8% | |
| in the 5,000 shortlist but beyond 200 (rank among shortlist rows: 330 / 1,055 / 1,973 at the 10 / 50 / 90% point) | 8.9% | none: needs a better scorer than f.z, not depth |
| **cut by pass 1** (partial ECFP4 score, 5,000 rows) | **7.7%** | **yes: `PC_N1=20000`** |
| no row in the tier at all (record exists, structure absent from the 105.9 M-row tier) | 5.2% | no (another tier build) |
| outside the ±10 ppm window | 3.4% | not needed on the test's instrument (below) |
| removed as a pool key or tautomer twin | 0 | by construction: the walk keeps the first structure of a key |

By stratum, missing at N 50: natural-product catalogue 38.3% (beyond N 21.5%, pass 1 6.2%, no tier row 6.2%, window
4.3%); enveda-180 20.5% (**pass 1 17.9%**, beyond N 2.6%); other 23.3%. By window size: windows above 10,000 rows
(126 molecules) hold the truth for 96.8% but the 5,000 shortlist keeps it for 78.6%; 3,000 to 10,000 rows: 98.0% and
95.9%. By fame, natural-product stratum: popularity below 3 → 80.5% within 100; 3 to 6 → 70.8%; above 6 → 61.3%, so
the local sets do not show "obscure answers sit deeper" (the obscure ones here have small windows).

**The shortlist.** `pc_deep2.py --n1 20000` on the 25 molecules whose truth pass 1 cut: all 25 are in the shortlist;
position 1, 1, 1, 2, 4, 5, 6, 6, 9, 19, 26, 31, 96, 126 and eleven beyond 200 [C]. On 40 control molecules with large
windows whose truth was listed: 30 keep their position, 10 move down (within 25 / 50 / 100: 34 / 36 / 39 → 33 / 36 /
38); 14 of the first 100 structures are new [C]. End to end at N 100 (`drop_n1_20k_n100`, `c2_n1_20k_n100`) against
N 100 with the 5,000 shortlist [R]:

| Molecules | `dB` | `dB`, v1 side | `c2` | `c2`, v1 side |
|---|---|---|---|---|
| 25 cut by pass 1 | **+0.252 ± 0.080 (11 / 0)** | | | |
| 40 controls | −0.005 ± 0.005 (0 / 2) | −0.010 ± 0.008 | +0.011 ± 0.032 (2 / 2) | +0.022 ± 0.036 |
| all 65 | +0.093 ± 0.034 (11 / 2) | +0.093 ± 0.035 | +0.004 ± 0.025 (3 / 4) | +0.014 ± 0.029 |

Scaled to all record molecules: 0.077 × 0.252 − 0.449 × 0.005 = **+0.017 on `dB`**, with no more injected rows than
N 100 has today. `c1` was not run (25 lists below the gate). Small samples; the `c2` side is "no sign of a cost", not
a proof.

**The window.** `q2_mass_error.py` on `train.parquet` [T]: error of the target mass per molecule on the Enveda timsTOF
libraries: median 1.7 ppm (enveda-180) and 1.5 ppm (enveda-np-examples), 99% point 4.1 / 6.0 ppm, outside 10 ppm
**0 of 182,941 and 0 of 250**, outside 5 ppm 0.04% / 2.4%. [M+H]+ carries a constant +0.0007 Da (+2.2 ppm, standard
deviation 0.9 ppm), [M−H]− +0.0003 Da; with the per-adduct median removed the error is above 5 ppm for 0.3 to 0.4%.
So a wider window or a second adduct hypothesis buys nothing on the test's instrument (the eleven local window losses
are library spectra from ion traps and mislabelled adducts), and half of the window is empty of answers. Narrowing to
±5 ppm, emulated on the deep lists (`q2_narrow_emul.py`) for the 302 molecules whose target is within 5 ppm [C]: 19% of
the listed structures fall away; truth first 18.9 → 19.9%, within 25 62.3 → 64.2%, within 50 68.9 → 71.9%, list MRR
0.290 → 0.304. ±3 ppm without the offset correction loses answers (enveda-180 within 25: 79.5 → 64.1%). The real ±5 ppm
channel run stopped at 229 of 380 molecules and no engine run used it: the end-to-end effect is **not measured**. It
halves pass 2, so it pays for the larger shortlist.

**Deeper N.** Section 1 and:

| Against N 100, gate 0.7 [R] | `dB` record | natural-product stratum | `c2` | `c2`, v1 side | `c1` |
|---|---|---|---|---|---|
| N 200, zero-forward (380 molecules) | +0.010 ± 0.005 (32 / 22) | +0.017 ± 0.007 | −0.002 ± 0.007 (20 / 19) | −0.008 ± 0.008 (23 / 30) | −0.002 ± 0.002 (1 / 3) |
| N 200, ICEBERG prefix | +0.024 ± 0.008 (18 / 3) | +0.032 ± 0.010 | −0.003 ± 0.008 | | −0.003 ± 0.004 |
| N 500, 150 molecules below the gate | +0.033 ± 0.015 (25 / 8) | +0.051 ± 0.019 | **−0.037 ± 0.020 (12 / 21)** | **−0.058 ± 0.023** | not run |
| N 500, ICEBERG prefix (118) | +0.042 ± 0.018 | +0.054 ± 0.024 | −0.040 ± 0.022 | | |
| N 1000, 75 molecules | +0.055 ± 0.024 (15 / 4) | | −0.016 ± 0.029 (8 / 7) | −0.054 ± 0.036 | |

Leaderboard: this is the table that mispredicted (first row). **A first-stage shortlist by fingerprint score
followed by the engine on the best few hundred is the natural design and it is what the join already is; it fails
at the second stage**, because the public ranker was never fitted with injected rows: it ranks a deep truth at 0.09
and lets deep decoys pass pool answers. A cheap pre-filter between FPNet and the engine does not change that (the
one tested, section 7, did not help end to end). Runtime is not the obstacle: measured 3.1 / 4.9 / 5.4 / 7.5 s per
molecule at N 0 / 200 / 500 / 1000 on one loaded desktop thread, i.e. about +12 / +24 / +60 / +125 minutes on Kaggle
for N 100 / 200 / 500 / 1000 (`join_depth.budget`; v13a ran 3 h 40 min of the 9 h); memory needs the caches cleared
every 24 / 12 / 4 / 2 molecules (0.66 MB per injected row; `join_depth.clear_every`; the notebook clears every 50,
which at N 200 lets them grow to 6.6 GB: **check this in the diagnostic runs**).

## 3. Engine-2's candidates through the same door (question 1)

Engine-2's top 40 hold 2 to 3 candidates outside the v1 candidate window per molecule (median 1 to 2; none for 35%
of the truth-in-pool lists). In `dA` the truth is among them for 90% of the lists. `run_inject.py --e2`: all of them
enter the v1 engine before the PubChem candidates (the K hardly matters). Against v12a (N 50; not re-run against
N 100), zero-forward / ICEBERG prefix [R]:

| Variant | `dA` | `dB` | `c2` | `c1` |
|---|---|---|---|---|
| join, always | **+0.170 ± 0.018 (120 / 20)** / +0.173 ± 0.026 | −0.001 ± 0.003 / −0.001 | −0.004 ± 0.004 (7 / 10) / 0.000 | −0.003 ± 0.003 (1 / 3) / 0.000 |
| join, v1 side alone | +0.224 ± 0.021 | −0.004 ± 0.004 | −0.002 ± 0.003 (4 / 8) | −0.001 (0 / 4) |
| join only below the gate | +0.116 ± 0.015 / +0.129 | −0.001 / −0.001 | −0.003 / 0.000 | −0.003 / 0.000 |
| fusion side: engine-2's best outside-pool candidate (of its first three) to slot 2 | +0.092 ± 0.008 (103 / 9) / +0.084 ± 0.014 | −0.003 ± 0.001 / −0.006 | −0.002 ± 0.001 (0 / 8) / −0.007 ± 0.004 | −0.001 (0 / 1) / −0.004 |
| ... to slot 1 | +0.287 / +0.240 | −0.018 / −0.010 | −0.055 / −0.038 | −0.070 / −0.077 |
| ... to slot 3 | +0.040 / +0.043 | −0.001 / 0.000 | −0.001 / −0.001 | 0.000 / −0.004 |
| "absent from the v1 list of 25" instead of "outside the pool", slot 2 | +0.044 / +0.042 | −0.004 / −0.002 | +0.002 / 0.000 | −0.002 / −0.001 |
| join + slot 2 | +0.205 / +0.203 | −0.004 / −0.007 | −0.007 / −0.008 | −0.005 / −0.004 |
| `FUSE_ALPHA` 0.8 / 1.0 (reference) | +0.027 / +0.053 | −0.026 / −0.049 | −0.008 / −0.010 | +0.001 / +0.005 |

The injected truth is first in the joint list for 49% and in the top three for 71%; an injected engine-2 row tops a
truth-in-pool list in 0.3% (`c2`) and 0% (`c1`). The slot rule changes 2.1% of the `c2` lists and 0.7% of the `c1`
lists. Natural-product stratum of `dA`: +0.147 (join), +0.091 (slot).

**What share of hidden molecules is `dA`-like: probably under 1%.** Engine-2's surplus is ChEBI / LIPID MAPS minus
COCONUT minus training structures (62.7k). Per window that is 2 to 3 structures against several thousand PubChem-only
structures; the catalogue bits of the popularity dataset show the same from the other side (0.11% of the listed
non-pool structures are in COCONUT, LOTUS or NPAtlas: the pool already holds the catalogues). If PubChem-only answers
are 7 to 14% of the hidden molecules (the pubchem review's reading of the 0.037 probe) and a curated structure is
even a hundred times as likely an answer as a random isomer, `dA` is 0.4 to 0.7%. Leaderboard: +0.09 to +0.17 times
that share is +0.0004 to +0.0012, against a cost of about 0.001 to 0.002. The leaderboard value of the engine-2
fusion itself (+0.015 in the public lineage) is explained by `c2` and `c1` alone and does not identify a `dA` share.
The step adds 2 to 3 rows, so it does not feed the mechanism of section 1.

## 4. The gate and N by rule (question 3)

Existing runs, against v12a, zero-forward (`s0_baseline.py`, `q3_rules.py`) [R]:

| Rule | `dB` | `c2` | `c1` | `c2`, v1 side | `c1`, v1 side |
|---|---|---|---|---|---|
| N 50, gate 0.8 | +0.012 ± 0.005 (12 / 6) | −0.004 ± 0.003 | 0.000 (2 / 0) | −0.008 ± 0.004 (2 / 5) | −0.006 ± 0.005 (1 / 3) |
| N 50, gate 0.9 / none | +0.023 / +0.031 | −0.001 / 0.000 | −0.006 / −0.008 | −0.005 / −0.007 | −0.028 ± 0.010 / −0.028 |
| N 50, gate 0.6 / 0.5 | −0.027 / −0.050 | −0.002 / −0.006 | 0.000 | +0.001 / +0.002 | 0.000 |
| above the gate: inject, but an injected row may not pass a candidate with its own library match | +0.004 ± 0.003 | 0.000 | −0.008 ± 0.005 | 0.000 | −0.004 |
| above the gate: inject, but never first | +0.017 ± 0.005 (37 / 17) | 0.000 | −0.008 ± 0.005 (3 / 5) | −0.001 | −0.005 (1 / 5) |
| N 100 for windows above the median, else 50 | +0.006 ± 0.004 | +0.001 | 0.000 | +0.002 | 0.000 |
| N 25 / 50 / 100 by window tercile, and the reverse | +0.003 / +0.003 | −0.001 / −0.006 | 0.000 / −0.002 | 0.000 / −0.008 | 0.000 / −0.001 |

Nothing clear: every step buys 0.01 to 0.02 on `dB` for 0.005 to 0.008 on the pool types on the v1 side. Section 1
adds one hint in the other direction: the cost of depth sits in windows under 3,000 rows, so "N 100 only where the
non-pool window is large" (N by window size the other way round than first tried) is the per-molecule rule the data
suggests; with N 50 ↔ 100 it is worth +0.003 here, far below what a submission resolves.

## 5. Where the remaining loss on `dB` sits (question 4)

`q4_budget.py`, all 380 truth-removed molecules, loss = Σ(1 − reciprocal rank) / 380 [R; the shares are C]:

| Category (first that applies) | molecules, N 100 | MRR | loss, zero-forward | loss, ICEBERG prefix | at N 50 |
|---|---|---|---|---|---|
| truth injected | 157 (41%) | 0.481 (first 33%, listed 90%) | **0.214** | 0.186 | 0.195 |
| ... of it: the ranker's joint list / v1 stages / engine-2 fusion | | 0.580 → 0.575 → 0.481 | 0.173 / 0.002 / 0.039 | 0.154 / 0.007 / 0.025 | |
| truth in the PubChem top N but the molecule is gated | 72 (19%) | 0.091 | **0.172** | 0.179 | 0.161 |
| no PubChem record, not generated | 43 (11%) | 0.054 | 0.107 | 0.095 | 0.107 |
| listed beyond N (or beyond 200) | 40 (11%) | 0 | 0.105 | 0.136 | 0.139 |
| cut by pass 1 | 24 (6%) | 0 | 0.063 | 0.095 | 0.063 |
| record without a tier row | 15 (4%) | 0 | 0.039 | 0.041 | 0.039 |
| truth generated by the v1 generator | 19 (5%) | 0.454 | 0.027 | 0.023 | 0.036 |
| outside the window | 10 (3%) | 0 | 0.026 | 0.006 | 0.026 |

Total MRR of the type 0.245 (N 100) and 0.234 (N 50).

**Which features push an injected truth down** (N 50; 81 injected truths that are not first; LightGBM contributions
of the truth minus those of the row that tops the list). The winner is a pool row in 51 lists, an injected decoy in
28. Against a pool row: `mass_ppm_rank` −0.88, `el_rank` −0.71, `sf_rank` −0.36 (the three order-coded ranks:
injected rows are last among ties of their formula), together −1.9 against a total gap of −1.5; library features
−0.27; analog, `analog_struct` and `model_views` favour the truth (+0.7). Against an injected decoy the gap is small
and spread (`mass_ppm_rank` −0.29, `model_views` −0.21, `fragnet` −0.15). So the placement of injected truths is
governed by an accidental "pool first" prior; removing it for everyone costs the pool types (join review), and what
is needed is a ranker that has seen injected rows and decides per list.

Budget, as what to build: (1) a ranker fitted on joined lists: up to 0.17 here, and the precondition for depth;
(2) the gated molecules, 0.17: no rule found (section 4), also a ranker question; (3) answers in no database, 0.11;
(4) beyond N 0.11 and the shortlist 0.06: the shortlist is the cheap part.

## 6. The structure generator next to PubChem candidates (question 5)

`run_inject.py --ad_m 30`: `analog_derive.propose` on the engine's 20 best spectral analogs, the first 30 products by
support enter through the join next to the PubChem top 50, molecules below the gate only (13 products per list on
average). Against v12a [R]:

| Molecules | zero-forward | ICEBERG prefix | v1 side |
|---|---|---|---|
| `dB`, no PubChem record (55; 40 below the gate) | **+0.071 ± 0.038 (7 / 3)**, 0.178 → 0.250 | +0.135 ± 0.066 (4 / 0; 24 molecules) | +0.055 ± 0.037 |
| `dB`, PubChem record (325) | −0.004 ± 0.007 (24 / 29) | −0.007 ± 0.010 | −0.008 ± 0.006 |
| `c2` (380) | −0.001 ± 0.007 (17 / 21) | +0.003 ± 0.009 | **−0.014 ± 0.007 (12 / 28)** |
| `c1` (150) | +0.003 ± 0.007 (1 / 2) | 0.000 | +0.003 ± 0.007 |

Without a record the truth is among the products for 45% of the 40 molecules [C], enters as a generator row for 35%
(the v1 generator supplies it for 2.5% of them once the injection order is 'first'). With a record it is among the
products for 19.7% but PubChem already supplies it for 62%, and the extra rows cost as much as they bring. So it
still adds, only where the answer is in no database, at a v1-side `c2` cost of the same size as N 50 → 100. Hide-and-
regenerate overstates about three times (pitfall 6): +0.02 to +0.05 per class-3 molecule is the honest reading, on
55 molecules. 2.6 s per molecule on one thread (about +35 minutes on Kaggle).

## 7. A structure prior in the selection step (question 6)

`q6_prior.py` on the 195 record molecules that have a 1,000-deep list. Prior = RDKit Contrib `NP_Score` (the model
file ships with the wheel; 0.3 s per 1,000 structures). The pool table's `np_likeness` column is not computed
anywhere in the casmi package (no reference in the code; it is NaN for the rows inspected). NP-likeness of the truth:
median +0.45 against −0.73 for the listed structures. Truth among the first N [C]:

| Selection among the first 1,000 by f.z | all 195: N 50 / 100 / 200 | natural-product catalogue (156) | enveda-180 (22) | other (17) |
|---|---|---|---|---|
| plain f.z | 64.6 / 69.2 / 73.8% | 62.8 / 67.9 / 73.1 | 68.2 / 68.2 / 72.7 | 76.5 / 82.4 / 82.4 |
| z(f.z) + 0.25 z(NP) | 68.2 / 71.8 / 75.4 | 67.3 / 71.2 / 75.0 | 68.2 / 68.2 / 72.7 | 76.5 / 82.4 / 82.4 |
| **z(f.z) + 0.5 z(NP)** | 67.7 / 72.8 / 77.9 | 67.3 / 72.4 / 78.8 | 68.2 / 68.2 / 68.2 | 70.6 / 82.4 / 82.4 |
| z(f.z) + 1 z(NP) | 62.6 / 69.2 / 77.9 | 64.1 / 71.2 / 79.5 | 68.2 / 68.2 / 68.2 | **41.2 / 52.9 / 76.5** |
| z(f.z) + 2 z(NP) | 48.2 / 59.0 / 72.8 | 55.1 / 65.4 / 77.6 | 27.3 / 45.5 / 68.2 | 11.8 / 17.6 / 35.3 |
| a quarter of the N reserved for the highest NP | 66.7 / 73.8 / 76.9 | 65.4 / 73.7 / 76.9 | 68.2 / 68.2 / 72.7 | 76.5 / 82.4 / 82.4 |
| NP > −1 first / no halogen first | 51.3 / 54.9 / 60.5 and 57.9 / 65.1 / 72.8 | 58.3 / 62.8 / 67.3 and 59.0 / 66.7 / 74.4 | 22.7 / 22.7 / 36.4 and 50.0 / 54.5 / 68.2 | 23.5 / 23.5 / 29.4 and 58.8 / 64.7 / 64.7 |
| sp3 fraction (0.5) / proxy (sp3, O − N, halogen, S / P) | 66.2 / 69.2 / 72.3 and 65.1 / 71.3 / 74.4 | | | |
| catalogue bits first | 81.0 / 81.5 / 82.1 | 83.3 | 68.2 | 76.5 |

End to end, N 100 selected by z(f.z) + 0.5 z(NP), handed over in f.z order, against plain N 100, 150 molecules below
the gate [R]: `dB` record **+0.003 ± 0.006 (10 / 2)**, natural-product stratum +0.009 ± 0.005; **`c2` −0.014 ± 0.014
(8 / 11), v1 side −0.021 ± 0.015**; ICEBERG prefix `dB` +0.020 ± 0.015, `c2` +0.010 ± 0.016. N 200 with the prior and
λ = 1 were not run.

- **Not supported.** The prior brings 3.6 points more truths into the first 100, but the ones it brings sit at f.z
  ranks beyond 100, where the ranker places them at 0.09 (section 1), and the decoys it selects look like natural
  products, which is what the pool answers look like too.
- **Flattered locally**: the local natural-product truths are catalogue compounds. The cost shows in the synthetic
  strata from λ = 1 on (−30 points at N 100 in "other").
- **The catalogue bits are an artefact here**: 78% of the local truths carry one against 0.11% of the listed decoys,
  because these truths are COCONUT compounds taken out of the pool by hand. On the hidden test a catalogue compound
  is a pool row. It is membership, not fame, but it has no population to act on.
- This is not the popularity prior (documentation counts) that lost on the leaderboard; no documentation count was
  used anywhere in this study.

## 8. Ranked list, drop-ins, and the next submission

| # | Change | Flag, prototype | Local effect per type | Basis | Cost, runtime | Verdict |
|---|---|---|---|---|---|---|
| 1 | larger first-pass shortlist at N 100 | `PC_N1=20000` (existing flag) | `dB` +0.017 (25 pass-1 molecules +0.252 ± 0.080, 11 / 0; controls −0.005 ± 0.005); `c2` +0.004 ± 0.025; `c1` not run | truth in the shortlist 83.7 → 91.4%: [C]; MRR: [R], 65 molecules | pass 2 on up to 20,000 rows: 29 s against 15 s per large-window molecule on one thread; about +35 min on Kaggle with 4 workers; no memory | **submit first** |
| 2 | engine-2's best outside-pool candidate to slot 2 | `FUSE_ENG_SLOT=2`, `eng_join.slot_fuse` | `dA` +0.092 ± 0.008 / +0.084; `dB` −0.003; `c2` −0.002 / −0.007; `c1` −0.001 / −0.004 | [R] | none | low risk, expected ≈ 0 (the `dA` share is probably under 1%) |
| 3 | engine-2 candidates through the join | `ENG_JOIN_K=40`, `eng_join.join_records` | `dA` +0.170 ± 0.018; `dB` −0.001; `c2` −0.004 ± 0.004; `c1` −0.003 ± 0.003 | [R] | +2 to 3 rows per list, under 1 min | as 2, twice the size on `dA` |
| 4 | ±5 ppm window on the offset-corrected target | `PC_PPM=5`, `PC_MASS_SHIFT='enveda'`, `pc_window.py` | truth within 50: 68.9 → 71.9% (emulated); loses 0.3 to 0.4% of timsTOF answers | [T] + [C]; **no end-to-end run** | halves pass 2 (pays for 1) | bundle with 1 after 1 is read |
| 5 | generator products through the join | `GEN_JOIN_M=30`, `gen_join.py` | no record +0.071 ± 0.038 / +0.135 ± 0.066; record −0.004; `c2` v1 side −0.014 ± 0.007 | [R], 55 molecules | +13 rows per list, about +35 min | wait: it adds rows |
| – | N 200 and deeper | `PC_JOIN_N` | local +0.010 to +0.055 on `dB`; leaderboard −0.025 at 200 | [R], mispredicted | | **no**, until the ranker is fitted with injected rows |
| – | NP-likeness in the selection | `PC_SEL_LAM=0.5`, `pc_select.py` | `dB` +0.003 ± 0.006; `c2` −0.014 ± 0.014 | [R] | | not supported |
| – | gate 0.8, pin rules, N by window | | section 4 | [R] | | nothing clear |

**The safest next submission, given that N 100 is the best known depth: v13a + `PC_N1=20000`, one change.** It keeps
N, the gate and the ranker; it does not add rows; its gain side is a count (the answer is in the shortlist or it is
not), strongest on the stratum measured on the test's instrument; the one regime-dependent part is that the decoys
which newly enter do not hurt pool answers (no sign on 65 lists, not proven). Leaderboard range: the join's 0 → 100
step was +0.176 on `dB` locally and +0.022 on the leaderboard, a ratio of one eighth; +0.017 then reads **+0.002**,
with 0 to +0.005 as the range and a public resolution of ±0.01. It will not show as a jump; it is the change least
likely to lose. If the diagnostic runs say the N 200 drop is real, the stronger move is N 100 with the depth read-out
logged, and no step that adds injected rows (5, deeper N) before a ranker fitted on joined lists exists.

### Drop-ins (not applied)

1. `PC_N1`: `nb/00_config.py` `PC_N1=20000`. Nothing else (`nb/04` passes it as `CASMI_PC_N1`).
2. `FUSE_ENG_SLOT`: copy `prototypes/eng_join.py` to `nb/embed/`; `nb/00_config.py` add
   `FUSE_ENG_SLOT=None, FUSE_ENG_SLOT_TOP=3,`; `nb/08_submit.py` before `fusion_core.build_submission(`:
   ```python
   import eng_join
   if CFG['FUSE_ENG_SLOT']:
       _pk = set(pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'), columns=['key']).key)
       fusion_core.fuse_rrf = eng_join.slot_fuse(fusion_core.fuse_rrf, _pk, CFG['FUSE_ENG_SLOT'], CFG['FUSE_ENG_SLOT_TOP'])
   ```
   The same three lines in `tools/refuse.py` if local re-fusion should follow.
3. `ENG_JOIN_K`: `nb/00_config.py` add `ENG_JOIN_K=0, ENG_JOIN_MAX_LIB=None,`; `nb/05_v1_engine.py`: the condition
   of the `pc_join.install` block becomes `CFG['PC_JOIN_N'] > 0 or CFG['ENG_JOIN_K'] > 0`; `nb/06_base_lists.py`
   (engine-2 runs in `nb/03`, so `ENG` exists): the outer `if CFG['PC_JOIN_N'] > 0:` becomes the same condition and
   before `pc_join.set_injection(E, recs)`:
   ```python
   if CFG['ENG_JOIN_K'] > 0:
       _w = P.window(target, E.cfg.ppm_win)
       if len(_w) == 0: _w = P.window(target, E.cfg.ppm_fallback)
       recs = eng_join.join_records(ENG.get(mid) or ENG.get(str(mid)), set(P.key[c] for c in _w), CFG['ENG_JOIN_K'],
                                    lambda s, k: pc_join.build_records(s, k, chem, fragmod, P.bits), pc_records=recs, target=target,
                                    lib_max=pc_join.library_max(E, spectra, target) if CFG['ENG_JOIN_MAX_LIB'] is not None else None,
                                    max_lib=CFG['ENG_JOIN_MAX_LIB'])
   ```
4. `PC_PPM`, `PC_MASS_SHIFT`: `nb/embed/pc/probe_core2.py` `PPM = 10.0` → `PPM = float(os.environ.get('CASMI_PC_PPM', 10.0))`;
   `nb/04_pubchem.py` `_pc_env` gets `CASMI_PC_PPM=str(CFG['PC_PPM']), CASMI_PC_SHIFT=str(CFG['PC_MASS_SHIFT'] or '')`;
   `nb/embed/pc/pc_runner.py`: copy `pc_window.py` next to it and replace the task's target (not the one used for
   `best_pool_fz`) by `pc_window.shifted_target(sub.nm.values, sub.adduct.values, pc_window.ENVEDA_SHIFT if os.environ.get('CASMI_PC_SHIFT') == 'enveda' else None)`.
5. `GEN_JOIN_M`: copy `gen_join.py` and `analog_derive.py` to `nb/embed/`; config `GEN_JOIN_M=0, GEN_JOIN_PARENTS=20,
   GEN_JOIN_MAX_LIB=0.7`; in `nb/06` after the PubChem records, under the gate:
   ```python
   _an = E.analogs(spectra[:E.cfg.max_query_spectra], target, None, -1, -1)
   recs += gen_join.records(gen_join.parents(_an, L.struct_smiles, E.cfg.gen_min_sim, CFG['GEN_JOIN_PARENTS']), target,
                            _wk | {r['key'] for r in recs}, CFG['GEN_JOIN_M'], lambda s: pc_join.candidate_record(s, None, chem, fragmod, P.bits))
   ```
   (`_wk` = the window keys as in 3.) `analog_derive.py` is own code; no third-party notice needed.
6. Any N above 50: in `nb/06` replace `(gi + 1) % 50 == 0` by `(gi + 1) % join_depth.clear_every(CFG['PC_JOIN_N']) == 0`
   and log `join_depth.depth_readout(_p['pc_keys'], keys, None)` summed over molecules.
7. After any of these: `python tests/test_config_keys.py`, the prototype's test, `python tools/gen_index.py`
   (`INDEX.md` was not regenerated here: no existing file may be edited).

## 9. What failed, what is missing, reproduce

- **The regime mispredicted N 200** (first row of the depth table). Everything marked [R] shares that weakness: the
  pool-side cost is read after an engine-2 that has seen the molecules, and with a weaker FPNet than the notebook's.
  The v1-side columns are the more honest cost estimate and should be read first from now on.
- My interim statement that depth pays as long as the truth enters was wrong beyond rank 100.
- Engine-2 results are paired against N 50, the generator against N 50; neither was re-run against N 100.
- Not finished: the real ±5 ppm channel run (229 of 380), deep lists for 130 of the 325 record molecules (section 7
  rests on 195), N 200 with the prior, `c1` for the shortlist and for N 500. N 1000 ran on 75 molecules.
- The first analyses ran with default BLAS threads and slowed my own engine workers fourfold for ten minutes. Three
  waits hit the tool's time limits; the session ended once with three jobs running (all resumable; two were finished
  after the restart, one was left).
- GLACIER is absent from both replays; ICEBERG scores exist for PubChem ranks 1 to 25 and engine-2's top 40 only.

```bash
export CASMI_DATA='D:\casmi-data' PYTHONHASHSEED=0 OMP_NUM_THREADS=1; P=.venv/Scripts/python; S=scratch/next_levers; D=D:/casmi-data/next_levers
for t in eng_join pc_window join_depth gen_join pc_select; do $P solutions/fusion-v4n/prototypes/test_$t.py; done
$P $S/s0_baseline.py [--first 169] ; $P $S/q3_rules.py                      # baseline, gate / N grid, rules (section 4)
$P $S/q2_decompose.py ; $P $S/q2_mass_error.py ; $P $S/q2_narrow_emul.py    # section 2
bash $S/queue.sh 0 2 ; bash $S/queue.sh 1 2                                 # engine-2 injection, generator (resumable)
$P $S/q1_fusion.py [--first 169] ; $P $S/q1_inject.py [--first 169]         # section 3
$P $S/q4_budget.py --n 100 [--first 169]                                    # section 5
bash $S/queue2.sh A ; bash $S/queue2.sh B                                   # N 200 under the gate
$P $S/pc_deep2.py --ppm 10 --k 1000 --out pc_k1000 --only $D/deep_ids.txt   # deep lists; then queue3.sh 1 / 2 (N 500, 1000)
$P $S/pc_deep2.py --ppm 10 --k 200 --n1 20000 --out pc_n1_20k --only $D/pass1.txt   # and n1_ctrl.txt
$P $S/run_inject.py --ver drop --only $D/n1_all.txt --pc_file $D/pc_n1_20k/rows.jsonl --pc_n 100 --tag drop_n1_20k_n100   # and --ver c2
$P $S/q5_eval.py --tags drop=drop_n200,c2=c2_n200,c1=c1_n200 --refn 100 --name n200 [--first 169]    # same for n500, n1000, ad30, sel05, n1_20k
$P $S/q6_prior.py --write "sel_l05_n100:fz+0.5*nps:100" ; bash $S/queue4.sh 1 ; bash $S/queue4.sh 2   # section 7
$P $S/q7_depth_cost.py                                                      # section 1
```
