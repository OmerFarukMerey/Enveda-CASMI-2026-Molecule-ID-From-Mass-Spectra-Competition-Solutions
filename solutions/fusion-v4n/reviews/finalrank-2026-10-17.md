<!-- index: 2026-10-17 learned final re-ranker on stack1 (0.451): LightGBM lambdarank over ~90 candidates per list (ranks of B*, F0, PREFUSION, base, engine 1 raw, engine 2, PubChem channel; engine-1 score, pc_fz, ICEBERG / GLACIER / MetFrag-lite raw and within-formula-group; formula agreement, Tanimoto to the first rows, adducts, mass; popularity group switchable), 5 scaffold folds, five variants (free, top-5, PubChem rows frozen, top-5 frozen, formula slots); the three primary variants meet the pre-registered criterion in both arms (+0.06 to +0.09) but post-hoc decomposition shows the gain is the constructed list types (engine-2-only and generated rows are answers by construction in dA / dB), pool-answer lists are flat or negative, and a model trained on pool lists only reverses the sign; verdict: do not build; Kaggle availability and run time of every feature -->
# Learned final re-ranker on stack1 (2026-10-17)

Marks: **[V]** computed here (file named); **[C]** cited; **[I]** inference. Δ = paired per-molecule difference of MRR@25 on the
notebook's metric identity vs B* (stack1-like replay with stack1's frag-gap branch, `reviews/stack1err-2026-10-17.md`); composite =
per-molecule mean over its lists, then mean over molecules. Arms: **m** = answer's popularity masked (primary), **u** = real. ± = one
SE, CI = 95% percentile bootstrap over molecules (10,000). One answer at rank 1 on the public part (~124 molecules) = 0.008. **No
leaderboard estimate.** CPU only (at most 6 workers, below-normal priority), no GPU, nothing pushed / submitted / downloaded /
committed; the visible `test.parquet` was not read.

## 0. Bottom line

- **Pre-registered result:** the three primary variants (PubChem-channel rows frozen, as the coordinator asked after v73a) **meet the
  pre-registered criterion in both arms**: V3 +0.089 / +0.062, V4 +0.062 / +0.045, V5 +0.077 / +0.056 (all CI > 0, both sets > 0,
  c1 / c2 within 2 SE).
- **That pass is not valid evidence, and no patch should be built.** Post-hoc decomposition (labelled, section 4) shows the whole gain
  is the *construction* of the local list types. In dA / dB the answer was removed from the pool by design. So an engine-2-only row is
  the answer 7.4% of the time in dA, against 0.08% in pool-answer lists; a generated row 3.0% in dA / dB, against 0.02%. The model
  learns "rows outside engine 1's pool are good" from the 50 / 50 training mix.
- On the **natural pool-answer lists** (c1 / c2) the same models are flat or negative: V3 c1 −0.001, c2 −0.008. In pool lists V3
  put an engine-2-only row first 59 times, was right 0 times and lost 13 right first rows.
- **A model trained on pool-answer lists only reverses the sign:** V3 −0.023 / −0.031, dA −0.12 / −0.15. So the result is set by
  the training class mix, and the hidden mix is unknown. The leaderboard has priced the engine-2 class bet at about zero twice
  (E2R2 +0.001, v72b 0.000 [C experiments rows 65, 68]).
- **Isomer order for pool answers** (the 2-5 loss of stack1err) is barely reachable. With ICEBERG / GLACIER (very probably leaked)
  the pool-trained model gets c2 +0.025 ± 0.006; without them, c2 +0.010 ± 0.006 (not significant).

## 1. Set-up [V]

- Data: arb's capture of the stack1-like e006u replay plus B* (`casmi-work/stack1err/bstar.pkl`), mixed 380 + panel 226 molecules,
  3,174 of 3,186 lists. Twelve lists have no parsable candidate; they cannot change, which is why the base is 0.5692 / 0.6440 against
  0.5664 / 0.6408 in stack1err. Candidates are the union, deduplicated by metric identity, of B*, F0, PREFUSION, re-ranked base,
  engine 1's raw 60, engine 2's 40 and the PubChem channel's 25: 251,101 rows (about 79 per list).
  The answer is a candidate in 2,824 lists.
- Features (`scratch/finalrank/fr_features.py`): 86 without popularity, as listed in `scratch/finalrank/PREREG.md`. The
  popularity group (pool popularity, PubChem tier lsid + lpmid with the arm-m mask, top_pop) is switchable. No list type, set or arm.
- Models (`fr_model.py`): LightGBM lambdarank, fixed parameters (15 leaves, 300 rounds, lr 0.05), 3 seeds averaged.
  - 5 folds by the answer's Murcko scaffold (acyclic: InChIKey14), shared across sets, types and arms.
  - Trained on arm-m lists of the other folds.
  - M_full: groups = all candidates. M_top5: groups = B*'s first 5 rows.
  - A first run leaked earlier models' OOF scores into later models' feature lists (stacking). Only the importance table showed it.
    It was fixed (feature lists frozen before any score exists) and everything was re-run; all numbers here are from the fixed run.
- Pre-registration `scratch/finalrank/PREREG.md`, written before any feature table or model existed. It was amended before any
  outcome by the coordinator's v73a fact: primary variants keep PubChem-channel rows fixed.

## 2. Results vs B* (pre-registered; `casmi-work/finalrank/eval.md`) [V]

PC row = identity in the PubChem channel's 25 and in neither engine's rows. Worst = touched share × 124 × local loss rate × mean RR
lost, in answers (every local win assumed to vanish).

| Variant / model | arm | Δ ± SE [CI] | touched / first | w / l | c1 | c2 | dA | dB | worst | pre-reg |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 free (reported) | m | +0.119 ± 0.006 [+0.107, +0.130] | 2064 / 770 | 693 / 146 | −0.007 | **−0.022 ± 0.008** | +0.314 | +0.193 | 2.5 | fails (c2) |
| | u | +0.073 ± 0.006 | 2064 / 665 | 578 / 236 | | | +0.181 | +0.134 | 4.5 | |
| V2 top-5 (reported) | m | +0.081 ± 0.005 | 1871 / 674 | 408 / 69 | −0.005 | −0.009 | +0.220 | +0.115 | 1.6 | passes |
| | u | +0.058 ± 0.005 | 1857 / 573 | 382 / 112 | | | +0.134 | +0.108 | 2.7 | |
| **V3 PC frozen** | m | +0.089 ± 0.005 [+0.079, +0.099] | 2012 / 596 | 478 / 103 | −0.001 | −0.008 ± 0.007 | +0.286 | +0.081 | 1.8 | passes |
| | u | +0.062 ± 0.005 [+0.053, +0.071] | 2007 / 503 | 371 / 113 | | | +0.179 | +0.076 | 1.9 | |
| **V4 top-5, PC frozen** | m | +0.062 ± 0.004 [+0.054, +0.070] | 1540 / 521 | 314 / 56 | −0.004 | −0.003 | +0.188 | +0.068 | 1.3 | passes |
| | u | +0.045 ± 0.004 [+0.037, +0.053] | 1504 / 441 | 264 / 63 | | | +0.124 | +0.063 | 1.5 | |
| **V5 formula slots, PC frozen** | m | +0.077 ± 0.005 [+0.068, +0.087] | 1979 / 543 | 422 / 99 | −0.000 | −0.008 | +0.244 | +0.080 | 1.7 | passes |
| | u | +0.056 ± 0.005 [+0.047, +0.065] | 1975 / 465 | 341 / 109 | | | +0.160 | +0.075 | 1.9 | |
| V3, no ICEBERG / GLACIER | m / u | +0.076 / +0.051 | | | −0.005 | **−0.024 ± 0.007** | +0.263 / +0.166 | +0.079 / +0.074 | 2.0 / 2.1 | fails (c2) |
| V1, popularity on (diag.) | m / u | **+0.201 / +0.017** | | 775/80, 421/420 | +0.011 | +0.021 | +0.372 / +0.056 | +0.400 / **−0.018** | 1.5 / 7.7 | |
| V3, popularity on (diag.) | m / u | +0.111 / +0.037 | | | +0.010 | +0.021 | +0.322 / +0.058 | +0.092 / +0.055 | 1.4 / 4.0 | |

- In V3 / V4 / V5 the answer's being a PC row contributes exactly 0: the coordinator's constraint holds.
- **Popularity**: switched on, pc_pop and pool_pop become the 2nd and 4th features (12% / 5% of gain; top-5 model 14% / 13%). The gain
  turns into the familiar fame shape: arm m +0.20, arm u +0.02, dB u negative.

## 3. Feature importance (share of gain, M_full without popularity; `casmi-work/finalrank/importance_mean.csv`) [V]

| Feature | M_full | M_top5 |
|---|---|---|
| r_B / rr_B (stack1's own rank) | 0.384 / 0.127 | 0.101 / 0.043 |
| r_F0 (rank before E2R2) | 0.050 | 0.030 |
| tan_b0 / tan_e0 (Tanimoto to B*'s / engine 2's first row) | 0.034 / 0.020 | 0.027 / 0.026 |
| g_gl_gap / g_ice_gap (GLACIER / ICEBERG gap to the formula group's best) | 0.029 / 0.021 | 0.065 / 0.056 |
| ice_lz / gl_lz | 0.023 / 0.019 | 0.016 / 0.012 |
| g_rr_eng_z (engine-2 rank within the formula group) | 0.023 | 0.053 |
| e1gap / e1z (engine-1 raw score) | 0.023 / 0.006 | 0.034 / 0.027 |
| r_base | 0.022 | 0.122 |
| r_eng | 0.015 | 0.030 |
| MetFrag-lite (frag_lz, g_frag_z) | 0.008 / 0.007 | 0.009 / 0.006 |

The model mostly keeps stack1's order. It moves rows on source membership (ranks present or absent), forward-model gaps within
the formula group, and similarity to the two engines' first rows. MetFrag-lite carries little weight.

## 4. Post hoc (NOT pre-registered; diagnostics that decide the reading) [V `fr_posthoc.py`, `fr_posthoc2.py` → `posthoc.md`, `posthoc2.md`]

- **D5 / gen, the construction artefact.** P(candidate is the answer | its source), by list type:

  | Type | engine-2-only row | generated row (pid < 0) | pool row, not generated | PubChem / other row |
  |---|---|---|---|---|
  | c1 | 0.0000 | 0.0002 | 0.028 | 0.0000 |
  | c2 | 0.0008 | 0.0002 | 0.025 | 0.0000 |
  | dA | **0.074** | **0.030** | 0 | 0.001 |
  | dB | 0 | **0.030** | 0 | 0.022 |

  In dA / dB the answer was removed from the pool, so the true structure can only appear as an engine-2 row or a generated analog.
  Half of the training lists are of these types.
- **D1, gain by the answer's source** (V3, arm m composite share):
  - dA, engine-2-only answers: +0.221.
  - dA / dB, answers among engine 1's 60 rows: +0.039 / +0.074. All of these are generated rows (258 of 258).
  - c2, pool answers: −0.008. c1: −0.001.
  - V5 and V4: same pattern.
- **D4, first rows V3 changes on pool-answer lists:**
  - New first row from engine 2 only: 59 cases, 0 right (13 right first rows lost).
  - New first row from the pool: 76 cases, 26 right (21 right before).
- **D2, a model trained on pool-answer lists only** (c1 / c2; same folds), evaluated on c1 / c2:

  | Variant | c1 | c2 | c2 without ICEBERG / GLACIER |
  |---|---|---|---|
  | V3 | +0.007 ± 0.005 | +0.025 ± 0.006 (80 / 31) | +0.010 ± 0.006 |
  | V5 | +0.007 ± 0.005 | +0.024 ± 0.006 | +0.011 ± 0.006 |

  Applied to every type, it loses: V3 −0.023 / −0.031 (dA −0.12 / −0.15), V4 −0.015 / −0.023, V5 −0.018 / −0.027. Without
  ICEBERG / GLACIER the losses are about twice as large.
- **D3, V5-pool** (within-formula slots; only engine 1's rows move, engine-2-only and PubChem rows fixed): +0.033 / +0.030 (CI > 0).
  - c1 +0.004, c2 +0.008 (not significant).
  - dA +0.043 / +0.034, dB +0.083 / +0.078. This part is again generated rows.
  - Without ICEBERG / GLACIER: +0.028 / +0.024.

Reading [I]:
- The re-ranker learned the **class prior of the training mix**, not an isomer-order signal. Every variant's sign follows the
  share of constructed types it was trained on.
- The hidden share of "engine-2-library-only" answers is unknown and was priced near zero by the leaderboard (E2R2 +0.001, v72b 0.000).
- Generated-row promotion (dB, D3) is the one part that could transfer, because hidden PubChem-only answers are also outside the
  pool. But the local rate is inflated: 31% of dB answers are reconstructed by the generator here, against 4-7% for record-less
  timsTOF molecules [C budget section 3.5, genjoin]. Scaled to 4-7% reach and a class-C share of 0.2-0.4, the dB gain becomes about
  0.08 × (0.05 / 0.31) × 0.3 ≈ +0.004, **below one answer** [I, arithmetic].
- The one leaderboard test of a validated isomer re-rank went the other way: iso005 scored validation +0.034, leaderboard −0.025
  [C experiments row 70].

## 5. Kaggle availability and run time [V timings here; I for Kaggle]

A patch would be one cell after stack1's cell 36 (plus captures). Each input and whether it exists at that point:

| Feature input | Available at cell 36? |
|---|---|
| B* (submission after E2R2) | yes |
| F0 | needs a one-line snapshot of the submission before cell 36 |
| PREFUSION_LISTS, ENG ranks, ICE_SCORES, GL_SCORES | yes (globals, as arb captured them) |
| PC entries (pc, pc_fz, pc_keys, S, fz_top, best_pool_fz) | yes; pc_fz order after the ICE re-rank |
| BASE_TAIL_CANDIDATES (engine 1's 60) | yes |
| test.parquet adducts / precursors | yes |
| **Engine 1's raw ranker score and pid / generated flag** | **no**: the forward cell overwrites `BASE[m]` with `(smis[:25], keys[:25], lib_max)`, and the scores it held were already popularity-adjusted (`z + 0.15·POOL_POP`). Needs a capture cell after the base-list cell (the generated flag is in the `'|gen'` form suffix); the model would have to be retrained on the popularity-adjusted score |
| Engine-2 probabilities | no (`eng_lists.json` holds smiles / keys only); not used |
| MetFrag-lite for every candidate | stack1 computes it only for frag-gap molecules. The patch would call `frag_rescore.score_molecule` for all of them: 606 molecules in about 15 s with 6 workers here |
| Metric identity / formula / Morgan-2 for about 80-150 candidates per molecule | 60k smiles in 160 s with 6 workers, about 15 ms per smiles single-threaded (tautomer canonicalisation dominates). At a few hundred to 1,000 hidden molecules: about 2-10 CPU minutes on Kaggle's 4 cores |
| LightGBM inference | under a second |

Whole-table feature build: 154 s for 3,174 lists. Training (5 models × 5 folds × 3 seeds): about 150 s.

Provenance: the training data are the replay lists built from train.parquet molecules, PubChem and the chain's own outputs. The model is
trained by our scripts (`fr_features.py`, `fr_model.py`), which would ship with it. No NIST, no third-party weights beyond what stack1
already loads (and the compliance plan replaces).

## 6. Recommendation

**Do not build a patch.** The primary variants pass the letter of the pre-registration, but the post-hoc decomposition shows the
pass measures the local list construction (answers removed from the pool by design), not a transferable signal:
- pool-answer lists lose;
- a pool-trained model reverses the sign;
- the engine-2 class bet behind most of the gain has been worth about zero on the leaderboard twice.

The pre-registration's flaw, recorded for the next one: a learned model over source features must be evaluated with a class mix that
is fixed in advance from the hidden-mix estimate, not the 50 / 50 constructed mix, or trained and tested per class with the hidden
weights stated up front.

If the team wants one follow-up, the only lead is **generated-row placement within formula groups (D3)**. It needs a fresh
pre-registration with:
- the generator's reach measured on record-less molecules (not the panel);
- a Kaggle-scale cost check on the 400 diag67 pool answers.

By the arithmetic in section 4 it is under one public answer.

## 7. Files

`scratch/finalrank/` (git-ignored): `PREREG.md`, `STATE.md`, `fr_prep.py`, `fr_features.py`, `fr_model.py`, `fr_posthoc.py`,
`fr_posthoc2.py`. `C:\Users\hemre\casmi-work\finalrank\`: `prep_mx.pkl`, `prep_p2.pkl`, `feat.parquet`, `oof.parquet`, `eval.md`,
`eval.json`, `importance.csv`, `importance_mean.csv`, `posthoc.md`, `posthoc.json`, `posthoc2.md`, `posthoc2.json`, logs. Inputs read:
`casmi-work/arb/Tm150/lists`, `casmi-work/stack1err/bstar.pkl`, `D:/casmi-data/pubdiff/dump/*_U_v18b_*`, `casmi-work/e006/pc_pop_*`,
the regime queries, the pool popularity prior and the PubChem tier (popularity group only). Nothing written to D:.
