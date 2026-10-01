<!-- index: fusion-v4n detailed experiment log -->
# fusion-v4n experiment log

One row per change. Local = `tools/refuse.py` on the visible placeholder test (plumbing check, optimistic), LB = public leaderboard.
LB rows are also appended to `../../docs/experiments.md`.

| # | Date | Version | Change | Local | LB | Note |
|---|---|---|---|---|---|---|
| 0 | 2026-09-30 | ours-v1-baseline | Reproduction of the reference v4n (default CFG) | – | – (ref 0.399) | Kaggle commit v1 (notebook095c91ef58): COMPLETE in 1089 s on T4, 12-molecule smoke run. All stages ran: engine-2 12/12, PubChem 12/12, base lists 0 errors, derivation self-test 0 mismatches, ICE and GL 11/11 covered molecules scored (755/773 candidates). submission.csv valid (400 rows, 23–25 parseable guesses per smoke molecule). Not submitted. |
| 0b | 2026-09-30 | ours-v1-baseline | + cell ids, drop ice_site at the end (output hygiene only) | 0.958 (12 visible molecules; all exact library hits, plumbing check only) | **0.399** (submitted as Kaggle v3 = v1 code from the editor) | reproduces the reference exactly |

Queue of next experiments: [../../docs/plan/roadmap.md](../../docs/plan/roadmap.md) (P1–P4).
| 1 | 2026-09-30 | v2-pop015-fraggap05 | POP_MU=0.15, FRAG_LAM=0.5, FRAG_MODE=gap (Kaggle v4). Commit run: all stages OK in 1023 s, popularity applied 12/12, frag re-rank used 2×. Chosen before the held-out validation finished (user asked for a pre-reset submission); frag gains measured only on a same-formula isomer panel (+0.056 ± 0.015, optimistic) | 0.958 (12 visible) | **0.404** (+0.005 vs 0.399) | scored 2026-10-01 ~02:15 |

## Validation run val1 (2026-10-01): leaked, do not tune on it

`VALIDATION=True` held out the 250 enveda-np-examples structures and purged their spectra from the library (Kaggle
`casmi26-ours-val`, 2 h 42 min, default fusion MRR 0.626). Per-channel truth recall shows that the v1 engine still
**knows** these molecules: its truth is top-1 in 93.6% of cases and always in the list. Its FPNet, ranker and v2 pool were
all built from the full train.parquet, and its spectra were in training. Engine-2 builds its pool at runtime from the
purged train, so it has the truth in only 7.6% of lists. Consequences:
- every knob that trusts v1 more "wins": FUSE_ALPHA 0.3 / KRR 1 → 0.876 (+0.25), LambdaRank stacking → 0.985 (top feature
  b_rank). This contradicts the public LB sweep, where α=0.6 was best. **Channel-balance knobs cannot be tuned here.**
- re-scoring knobs are confounded the same way (a lower ICE/GL weight keeps v1's memorised top-1).
  ICE 1.0 / GL 0.5 +0.012 ± 0.006, POST_GL 0.5 +0.007 ± 0.006, FUSE_ENG_K 25 +0.003 ± 0.002, TOPN 80 +0.001: treat as not evidence.
- FILL_25 changes nothing here (lists are full).
Lesson: public artefacts trained on all of train.parquet leak into any train-derived holdout. Only untrained components
(frag_rescore, isomer-panel test) and the LB give trustworthy signal for this pipeline.

## v3 candidates (2026-10-01), each = v2 + one change. Commit runs all clean (16–19 min).

Validation (val1/val2, 250 held-out molecules, leaked: see above) paired vs v2, `tools/compare_configs.py`:

| Config | val MRR | Δ vs v2 ± SE | W/L | top-1 changed | top-25 changed | smoke (12 visible) |
|---|---|---|---|---|---|---|
| v1 (LB 0.399) | 0.6258 | −0.0001 ± 0.0096 | 21/27 | 35 | 242 | 0.958 |
| v2 (LB 0.404) | 0.6259 | — | — | — | — | 0.958 |
| v3a-fpens (S1) | 0.6239 | −0.0020 ± 0.0031 | 5/7 | 5 | 208 | 0.958 |
| v3b-fragall025 (S2) | 0.6272 | +0.0012 ± 0.0065 | 13/19 | 15 | 246 | 0.958 |
| v3c-icegl15 (S3) | 0.6255 | −0.0005 ± 0.0094 | 18/36 | 26 | 214 | **0.917** (one library hit demoted) |
| v3d-pop025 (S4) | 0.6237 | −0.0022 ± 0.0043 | 5/11 | 6 | 187 | 0.958 |

The validation cannot separate them: v1 vs v2 is −0.0001 here but +0.005 on the LB. Use the change counts as effect size: S1/S4
are tiny (likely below LB noise), S2/S3 are the informative LB probes. Presets: configs/v3*.json.

## v3b result and the de-leaked proxy (2026-10-01)

**v3b-fragall025 scored 0.390 on the public LB (v2: 0.404, −0.014 ≈ 5–6 molecules).** Predicted 0.406; wrong. The frag term
must not be applied to molecules ICE/GLACIER already score. The combo (v3e, frag all + FPNet ensemble) was never submitted.

`tools/sim_isomer_eval.py` shuffles same-formula candidates inside the base lists of val1 to erase the v1 ranker's
memorised isomer order (6 seeds). Calibration against the three LB points:

| Config | LB | sim MRR | Δ vs v2 ± SE |
|---|---|---|---|
| v1 | 0.399 | 0.408 | −0.044 ± 0.008 |
| v2 | 0.404 | 0.452 | — |
| v3b | 0.390 | 0.453 | +0.001 ± 0.004 |

It gets v2 > v1 right (gain overstated: enveda-np-examples are deliberately common compounds, so popularity is over-rewarded)
and misses v3b's drop. val1 has no library hits (class-2-like only); the drop may come from class-1 molecules. A class-1
validation (`VAL_PURGE='lib'`: only the held-out timsTOF rows are purged, all 250 structures keep 55.6k spectra from
other libraries) is running as `casmi26-ours-valc1`.

## v4 candidates: library gate (2026-10-01)

On 400 exact library hits (vis) v2 loses 18 answers to the ICE/GLACIER re-rank (MRR 0.975; re-rank off: 1.000).
Gate = ICE_LAM_LIB / GL_LAM_LIB for molecules with lib_max ≥ 0.9. Paired vs v2:

| Config | Library regime (400) | Class-2 sim (250, 6 seeds) | Smoke (12) | LB |
|---|---|---|---|---|
| v4a hard gate 0/0 | +0.0255 ± 0.0060, 18/0 | −0.0031 ± 0.0014, 1/7 (analog hits) | 1.000 | **0.404** (= v2: the gate at 0.9 is neutral on the LB) |
| v4c soft gate 0.5/0.5 | +0.0196 ± 0.0050, 16/0 | +0.0001 ± 0.0004, 3/4 | 0.958 | not submitted |
| gate 0.25/0.25 | +0.0238 ± 0.0056, 18/0 | −0.0004 ± 0.0004, 2/5 | – | – |
| v4b hard gate + POP_MU 0.25 | +0.0255 | +0.0193 ± 0.0037 (popularity-biased) | 1.000 | **0.409** (+0.005 vs v4a: popularity 0.25 > 0.15) |

Predictions made before scoring: v4a 0.404–0.410, v4b 0.400–0.414. v4a vs v2 isolates the gate; v4b vs v4a isolates
popularity 0.25. Soft vs hard is too small for the LB to resolve; decide it on the class-1 validation (valc1).

## Class-1 validation valc1 and the gate threshold (2026-10-01)

`VAL_PURGE='lib'` (Kaggle `casmi26-ours-valc1`): same 250 molecules, only their timsTOF rows purged, 55.6k spectra from
other libraries kept. Default fusion MRR 0.876; lib_max ≥ 0.9 in only 26% of molecules (≥ 0.8: 78%, median 0.86).
Paired vs v2 (class-2 sim = val1 isomer shuffle, 6 seeds; exact hits = vis):

| Config | Class-2 sim | Class-1 (valc1) | Exact hits (vis) |
|---|---|---|---|
| v1 | −0.044 | −0.026 ± 0.009 (8/22) | −0.004 |
| v3b frag all (LB 0.390) | +0.001 | +0.008 ± 0.005 (10/6) | +0.001 |
| S3 ICE/GL 1.5 | +0.011 | **−0.028 ± 0.008 (1/23)** | **−0.018** |
| S4 POP_MU 0.25 | +0.024 (biased) | +0.011 ± 0.004 (9/2) | +0.001 |
| v4a hard gate, τ 0.9 (submitted) | −0.003 | +0.012 ± 0.006 (6/0) | +0.026 (18/0) |
| v4b hard gate + POP 0.25 (submitted) | +0.019 (biased) | +0.023 ± 0.007 (14/1) | +0.026 |
| hard gate τ 0.8 (via LIB_TAU) | −0.017 ± 0.006 | +0.028 (17/5) | – |
| v4c soft gate 0.5, τ 0.9 | +0.000 | +0.006 (5/0) | +0.020 (16/0) |
| **v4d soft gate 0.5, GATE_TAU 0.8** | **+0.002 ± 0.004 (15/17)** | **+0.024 ± 0.007 (14/0)** | **+0.020 (16/0)** |
| soft gate, GATE_TAU 0.7 | −0.004 ± 0.005 (23/40) | +0.041 (22/0) | +0.020 |
| v4d + POP 0.25 | +0.026 (biased) | +0.029 ± 0.008 (19/1) | +0.020 |

GATE_TAU separates the re-score gate from LIB_TAU (which also steers the PubChem merge). None of the three regimes shows
v3b's LB drop, so these are "does no harm / helps here" checks, not LB predictions. All three still carry the v1 ranker's
memorisation, which favours trusting the ranker; the LB results of v4a/v4b test whether the gate transfers.

