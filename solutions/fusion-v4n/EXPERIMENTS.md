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

## v5: PubChem-channel popularity + GLACIER [M+H]+ only (2026-10-01)

From wangpenghua's public v29 notebook (reported LB 0.415). Commit runs clean; the popularity variant of the probe was
active (fz_top present for 12/12 molecules, arrays row-aligned with the tier) and GLACIER scored 11 of 12 molecules.
Predictions before scoring: v5a 0.408–0.418, v5c 0.410–0.425.

| Preset | Content | Smoke (12) | LB |
|---|---|---|---|
| v5a-v29replica | POP_MU 0.15, PC_POP_LAM 0.25, PC_POP_UNION 200, GL_MH_ONLY | 0.958 | **0.393** (predicted 0.408–0.418: wrong) |
| v5b-v29-fraggap-softgate08 | v5a + frag gap + soft gate at 0.8 | 0.958 | not submitted (superseded by v5c) |
| v5c-v4b-pcpop-glmh | v4b (POP_MU 0.25, frag gap, hard gate) + PC popularity + GL_MH_ONLY | 1.000 | **0.405** (v4b: 0.409; predicted 0.410–0.425: wrong) |

**Post-mortem (2026-10-01).** v5a was not a replica: wangpenghua has no public datasets, so the two blank inputs of v29
are his private popularity arrays. We substituted dmitriigluzdov/casmi26-pubchem-popularity-prior. With it, "PubChem-channel
popularity + GLACIER [M+H]+" costs −0.004 on top of v4b and about −0.01 on top of reference + pool popularity (bob 0.401,
v2 0.404). The research notes had already flagged that thivnti's notebook (v4n + PubChem-channel popularity + pool μ 0.2,
same public dataset) sits below v4n. Working hypothesis: the PubChem-channel prior with this dataset is the harmful part;
GLACIER [M+H]+ alone is untested by us. Lesson: a code replica without the same data is not a replica; say so before submitting.
Best remains v4b at 0.409.

## Test harness v2: leak-reduced validation (2026-10-01, evening)

Why the old harness misled us: the engine's FP bank `fpnet_full1` was trained on every training structure, including
enveda-np-examples (89% top-1 there), whereas `FPNet A` (fpnet_0.pt, also in casmi26-v4b-models) excludes that library
(42% top-1; figures from dmitriigluzdov/casmi26-natural-product-panel-benchmark). The same author's fold-safe FPNet
reports that a popularity prior helps famous natural products and **hurts ordinary molecules** (top-1 71% -> 58% at
0.25; even NP-catalogue answers peak at 0.15-0.25 and fall at 0.6). val1 held only famous NPs (median popularity 13.0),
so it over-rewarded popularity.

New validation runs (`VAL_EXTRAS` also dumps the PubChem channel with popularity and GLACIER on [M+H]+ only, so v5-style
configs can be replayed locally):

| Run | Structures | FP bank (never saw them) | Popularity median |
|---|---|---|---|
| valA (`casmi26-ours-vala`) | 250 enveda-np-examples, all spectra purged | FPNet A | 13.0 (famous) |
| valF (`casmi26-ours-valf`) | 244 NP-catalogue structures of the fold-0 hold-out, all spectra purged, 6 spectra each from any library | fold-safe `fpnet_fold0` | 5.4 (ordinary) |

Acceptance test for the harness: it must reproduce the order of our seven LB results
(v4b 0.409 > v5c 0.405 > v2 = v4a 0.404 > v1 0.399 > v5a 0.393 > v3b 0.390) before any config chosen with it is submitted.

## Reviews of 2026-10-02 (reports in reviews/) and what changed

**Audit** (reviews/audit-2026-10-02.md): metric identical to the official one on 37 submission/label pairs; six LB configs
match what ran; 12 Kaggle runs re-fuse exactly; the PubChem-popularity port matches v29 line for line; pool popularity arrays
aligned. Harness bugs found and fixed: (1) the validation purge used the stored inchikey14, leaving tautomer twins of 14/250
answers (27/244 in valF) in the library -> purge and sample by the metric key from train_structs.parquet; (2) valF relabelled
Orbitrap/QTOF spectra as timsTOF -> real labels kept for non-np-examples sets; (3) regime_eval's result cache ignored
PC_POP_LAM / GL_MH_ONLY; (4) the isomer shuffle ran on the 80-deep cache before the TOPN cut and lost the truth in ~15% of
molecules -> lists are cut first (earlier sim numbers are affected); (5) TOPN 60 replay of an 80-deep cache is inexact in
stage C for 38-57 of 250 lists -> future validation runs dump at TOPN 60. Structural limits: the public LB is ~132
molecules, so nearly all gaps between our seven scores are under 2 SE; validation truths are never PubChem-only, so
PubChem-channel changes cannot be measured locally; popularity alone scores 0.726 on val1.

**Research** (reviews/research-2026-10-02.md): the 0.415 notebook's popularity arrays are private ("Dataset private");
its code with public arrays scores 0.393-0.401 for three teams. With public data the PubChem-channel prior costs
0.007-0.018 and GLACIER on [M+H]+ only gains 0.007-0.010 (analyticaobscura: bob v10 + that filter = 0.408-0.411).

**Directions** (reviews/directions-2026-10-02.md): an estimated 38-46% of hidden molecules score ~0 (truth never listed);
isomer order is the main loss when the truth is listed; engine-2 has the truth in 7.6% of its lists without library
analogs and its fusion costs class-2-like molecules -> FUSE_MIN_LIB (no cost on valc1: +0.002, 2-3 wins / 0-1 losses).

valA / valF (running since 23:05 on 2026-10-01) were built before fixes (1), (2) and (5): drop the twin molecules when
scoring valA; treat valF as low-trust.

## valA / valF results (2026-10-02): the harness fails its acceptance test

Both runs finished with VAL_EXTRAS; molecules whose metric key has a twin left in the library were dropped (valA 250 -> 236,
valF 244 -> 217). Per-channel truth recall on the twin-free sets:

| Run | FP bank | v1 base top-1 / listed | engine-2 top-1 / listed |
|---|---|---|---|
| valA (famous NPs) | FPNet A (never saw them) | 94.9% / 100% | 0.4% / 2.5% |
| valF (ordinary NPs) | fpnet_fold0 (never saw them) | 76.5% / 90.8% | 0.0% / 0.5% |

Swapping the FP bank did not remove the leak: the v1 ranker and its feature resources still recognise training structures.
`tools/calibrate.py` over our seven LB points: valA Spearman -0.13 (2/10 large gaps ordered correctly), valF **-0.95** (0/10).
valF penalises popularity (ordinary molecules), the LB rewards it. **No local set orders our submissions like the LB;
local numbers are "does it harm" checks only.**

Paired vs v4b (0.409):

| Config | valA (236) | valF (217) |
|---|---|---|
| v6a: GL_MH_ONLY | +0.007 ± 0.005 (9/4) | -0.002 ± 0.005 (7/7) |
| FUSE_MIN_LIB 0.5 | **+0.125 ± 0.016 (52/1)** | **+0.139 ± 0.018 (53/0)** |
| FUSE_MIN_LIB 0.7 | +0.187 ± 0.019 (79/1) | +0.183 ± 0.019 (73/1) |
| POP_MU 0.15 | -0.000 ± 0.004 | +0.002 ± 0.005 |
| POP_MU 0.4 | +0.014 ± 0.006 (16/9) | -0.011 ± 0.006 (10/22) |
| FRAG_LAM 0 | +0.005 ± 0.005 | -0.000 ± 0.003 |
| PC_POP_LAM 0.25 | -0.001 ± 0.001 | +0.001 ± 0.004 |

The size of the FUSE_MIN_LIB gain is inflated (v1 is memorised here, so skipping the fusion and its second ICE/GL
re-rank keeps a correct top-1). The premise is leak-independent: without library analogs engine-2 lists the truth for
0.5-2.5% of molecules. Presets v7a (v4b + FUSE_MIN_LIB 0.5) and v7b (v7a + GL_MH_ONLY) have commit runs; nothing submitted.
A teammate's submission "np1" (v4b + NP-tuned FPNet in engine-2) was made on 2026-10-02 06:09 UTC and is not from this tree.


## Adduct surrogate for ICEBERG (2026-10-02): flag added, not submitted

Source: `reviews/coverage-2026-10-02.md`. The packaged ICEBERG checkpoint has trained weights for [M+H]+ and [M+Na]+
only (the other 12 adduct columns are still at their initial values), GLACIER for [M+H]+ only. `ICE_SURROGATE`
(`nb/embed/surrogate.py`, default off) rewrites [M+NH4]+ spectra (precursor minus NH3, peaks above the [M+H]+ m/z
dropped) and water-loss spectra (relabelled) so the runner scores them against the [M+H]+ prediction. `'gap'` adds
them only for molecules without a native [M+H]+ / [M+Na]+ spectrum, `'all'` for every molecule. `GL_SURROGATE` hands
the same spectra to GLACIER (untested, default off). [M+K]+ and negative mode are not touched: both were at chance.

| Measurement (ICEBERG alone, truth's formula group, val1) | n | Result |
|---|---|---|
| [M+NH4]+ surrogate against native [M+H]+, same molecules | 18 | in-group MRR 0.340 against 0.331 |
| Water-loss surrogate against native [M+H]+, same molecules | 20 | 0.422 against 0.551 (worse than native) |
| Molecules with no covered spectrum, surrogate only | 16 | top-1 0.188, MRR 0.322 ± 0.092 (chance 0.035 / 0.123) |
| Covered molecules, native → native + surrogates | 44 | 0.487 → 0.545 (+0.059 ± 0.036) |

Coverage: np-examples hold-out 86.4% → 92.8% of molecules with `'gap'` (16 of 250); `'all'` adds spectra to 54
molecules (649 → 779 spectra, about +20% ICE jobs for those). Expected LB effect +0.001 to +0.006, below what one
submission resolves, so it should go in together with a larger change. No pipeline-level replay yet (needs an ICE
run with the flag on a validation set). Preset: `configs/v8a-v7b-icesurr-gap.json`, to be built on whichever of
v4b / v7b is best once the v7b score is known.

## PubChem merge: earlier slots and a forward-model tier (2026-10-02): flags added, not submitted

Source: `reviews/pubchem-2026-10-02.md`. Why popularity inside the PubChem channel lost on the LB (v5a 0.393, v5c
0.405): the channel lists only structures outside the pool, documented structures are already in the pool (81–95% of
those with pop ≥ 8), and a structure outside the pool has a median popularity of 1.1; at that level the prior removes
17–37% of what the channel returns. Keep `PC_POP_LAM = 0`. The public panel's 0.718 promotion rule is a fame artefact
(0.394 with the popularity masked, below the plain list at 0.424).

New knobs (default off; `tests/test_pc_fwd.py` checks stage B against `prototypes/pubchem_merge.py`, rule C):
- `SLOTS_GENTLE=[3, 6, 9, 12, 15]` (was 4, 8, 12, 16, 20): proposals one place earlier.
- `PC_FWD_T1=0.0`: the early slots (2, 4, 6, 8, 10) also when the best new proposal beats every same-formula base
  candidate of the top 25 in each forward model that scored both.

Gain side (the agent's PubChem-only simulation on valA, answer moved from the pool into the PubChem list, v7b):
0.188 → 0.242 (+0.054 ± 0.006, 114 wins, 0 losses). Cost side, replayed here with `tools/compare_configs.py`, paired
against v7b on molecules whose answer is in the pool:

| Config | valA (236) | valF (217) | valc1 (250) |
|---|---|---|---|
| slots 3, 6, 9, 12, 15 | +0.009 ± 0.006 (17/6) | +0.002 ± 0.002 (7/11) | +0.003 ± 0.003 (3/2) |
| `PC_FWD_T1` 0.0 | −0.001 ± 0.001 (0/1) | −0.001 ± 0.001 (0/3) | −0.000 (0/1) |
| both (rule C) | +0.011 ± 0.005 (17/6) | +0.001 ± 0.002 (7/13) | +0.003 ± 0.003 (3/2) |

The positive in-pool numbers are an artefact of the leaked sets (a PubChem decoy below a correct rank 1 protects the
answer in the later fusion); count them as zero cost. Expected LB effect 0 / +0.004 / +0.009: the PubChem list alone
scored 0.037 on the LB (reference author's probe), so even a perfect merge is worth about +0.03. Below what one
submission resolves; ship with another change. Preset: `configs/v8b-v7b-icesurr-pcfwd.json` (v7b + `ICE_SURROGATE`
gap + rule C).

## Library gate for the forward models, honest-prior sweep (2026-10-02)

Trigger: `docs/landscape/update-2026-10-02.md` 2.2. A copy of the v29 code that ran on CPU, where ICEBERG reached only
the 90 molecules with the lowest `lib_max`, appears to score 0.411; the same code on a GPU scores 0.393 (our v5a and
two others). Question: should ICEBERG / GLACIER be switched off above a much lower `lib_max` than 0.9?

`scratch/gate_sweep.py`, paired against v7a (v4b + `FUSE_MIN_LIB` 0.5), fp-honest regime of the isomer study (inside
each formula group of the base top-60 the order of an FPNet that never saw the molecule). "+ GL" = `POST_GL_LAM` 1.0,
a GLACIER-only re-rank after the engine fusion, as in the CPU run.

| Variant | valA, class-2-like (236) | valF, class-2-like (217) | valc1, class-1-like (250) |
|---|---|---|---|
| v4b (no engine-2 gate) | −0.088 ± 0.014 (5/48) | −0.072 ± 0.015 (10/44) | −0.001 ± 0.002 |
| hard gate τ 0.8 | −0.005 ± 0.005 | −0.013 ± 0.006 | −0.001 ± 0.012 |
| hard gate τ 0.8 + GL | +0.008 ± 0.006 (10/6) | −0.005 ± 0.006 (20/12) | +0.034 ± 0.011 (23/9) |
| hard gate τ 0.7 + GL | +0.004 ± 0.008 | −0.006 ± 0.006 | +0.050 ± 0.013 (33/9) |
| hard gate τ 0.5 + GL | −0.015 ± 0.012 | −0.012 ± 0.010 | +0.053 ± 0.013 (35/9) |
| hard gate τ 0.4 + GL (about the CPU run's quarter) | −0.024 ± 0.013 | −0.020 ± 0.011 | +0.053 ± 0.013 |
| hard gate τ 0.5, no GL | −0.083 ± 0.016 | −0.038 ± 0.013 | +0.007 ± 0.015 |
| **soft gate 0.5, τ 0.8** | **−0.000 ± 0.005 (8/8)** | **−0.001 ± 0.008 (17/14)** | **+0.034 ± 0.010 (21/6)** |
| soft gate 0.5, τ 0.6 | −0.018 ± 0.009 | +0.001 ± 0.010 | +0.047 ± 0.011 (29/6) |
| no stage-C re-rank (`POST_*` 0) | −0.088 ± 0.015 | −0.026 ± 0.010 | +0.024 ± 0.010 |

lib_max quartiles: valA 0.40 / 0.59 / 0.76, valF 0.25 / 0.58 / 0.85, valc1 0.81 / 0.86 / 0.90. So `lib_max` does not
separate the classes below about 0.8 (purged molecules reach 0.6 to 0.8 through analogs).

Reading:
- The engine-2 gate is the large effect: +0.07 to +0.09 on both class-2-like sets with an honest prior, neutral on
  class 1. This is what v7b tests on the LB.
- For class-1-like molecules the forward models hurt, and a gate at 0.7 to 0.8 gains 0.03 to 0.05. For class-2-like
  molecules a gate at 0.8 costs nothing and one at 0.5 or below costs 0.01 to 0.02. The CPU run's cut (lowest quarter,
  about τ 0.4) is therefore not what the caches recommend; the soft gate at 0.8 takes the class-1 gain without the
  class-2 cost. Expected LB effect with 16 to 33% class 1 and the valc1 gain halved for optimism: +0.003 to +0.008.
- Forward-model scores on these sets are optimistic (the models have probably seen the molecules), which if anything
  understates the gate.

Presets with clean commit runs pending: `v9a-v7b-softgate08` (v7b + soft gate 0.5 at τ 0.8) and
`v9b-v9a-icesurr-pcfwd` (v9a + `ICE_SURROGATE` gap + PubChem rule C). Nothing submitted.

## v7b result: 0.405 (2026-10-02). The engine-2 gate does not transfer

v7b = v4b + `GL_MH_ONLY` + `FUSE_MIN_LIB` 0.5 scored **0.405** against 0.409 for v4b. Predicted 0.421 (range 0.400 to
0.435): inside the range, 0.016 below the central value, the fourth optimistic miss out of six predictions.

- `GL_MH_ONLY` alone is worth about +0.004 for other teams (0.401 → 0.405, 17 teams). So the gate is about −0.008,
  one public-LB molecule, which a single submission cannot separate from zero. What is certain: it is not the +0.02 the
  roadmap carried, and nowhere near the local +0.13 (plain) or +0.07 to +0.09 (honest prior).
- Why the local sets are blind here (hypothesis): engine-2 searches its own candidate set (COCONUT, ChEBI, LIPID MAPS,
  training structures) with its own fingerprint models. In every local set the answer is a structure of the main
  pool, where the v1 engine already lists it, so engine-2 can only disturb the order. On the hidden test it can supply
  or support answers the v1 engine ranks badly or not at all. The same blindness applies to the PubChem channel.
- Consequence: `FUSE_MIN_LIB` stays off. Presets v7a, v7b, v8a, v8b, v9a, v9b are not to be submitted as they are
  (all carry the gate). v6a (v4b + `GL_MH_ONLY`) remains valid, expected 0.409 to 0.413.
- Consequence for validation: the honest-prior regime of `scratch/isomer/e9_pipeline.py` joins the plain caches as
  "harm check only". `lb_points.json` now has eight points.

## Ranker study (2026-10-02): the tie-order leak, our ranker, truth removed from the pool

Source: `reviews/ranker-2026-10-02.md`. The v1 engine now runs on the local machine (`prototypes/local_engine.py`,
4.2 s per molecule on one CPU thread; same top-1 as the Kaggle run for 250 of 250 molecules).

**Tie-order leak.** `casmi/engine.py::_rank_norm` breaks ties by candidate order = pool row order: training structures
first, COCONUT-only structures after them. Inside one formula `mass_ppm_rank`, `sf_rank` and `el_rank` are therefore
row-order codes, and the public ranker learned "earlier row = more likely". In-group top-1 on 3,100 hold-out molecules
with every fingerprint model fold-safe: 0.870 as run, 0.748 with order-free ties, 0.683 when the truth sits where a
COCONUT-only row sits, 0.469 for FPNet alone. For a hidden class-2 molecule (structure known, no spectrum) the truth
is a COCONUT-only row, so this prior points the wrong way; for class 1 (truth is a training structure) it is correct.
This is the first defect found that exists only on the hidden test and that no train-derived validation set can show.

**Flags (default off).** `RANKER` = `public` | `public_free` (public ranker, rank features recomputed with average
ties) | `ours` (our LightGBM ranker, private Kaggle dataset `huseyinemreaksoy/casmi26-ours-ranker`, 250 features,
trained on 3,788 hold-out molecules + the 250 enveda-np-examples molecules) | `blend` (z(ours) + z(public_free));
`RANKER_LIB_GATE` = lib_max at or above which the public ranker is kept as it is. `nb/embed/ranker_infer.py`,
`nb/05_v1_engine.py`, `nb/06_base_lists.py`. On commit runs the ungated branch is exercised on every molecule and any
error fails the run; on the scoring rerun an error falls back to the public ranker.

Pre-fusion whole-list MRR, paired against the public ranker in the same situation (class 2 = truth placed as a
COCONUT-only row):

| Variant, gated at lib_max 0.7 | class 2, enveda-np-examples (236) | class 2, valF (217) | class 2, hold-out (3,100) | class 1, enveda-np-examples (250) |
|---|---|---|---|---|
| `public_free` | +0.039 ± 0.013 | +0.068 ± 0.015 | +0.034 ± 0.003 | −0.005 ± 0.005 |
| `blend` | +0.046 ± 0.014 | | | −0.008 ± 0.007 |

After the whole v4b fusion on the three stage caches every variant is within ±0.02 and none is distinguishable from
zero (gated `public_free`: +0.009 ± 0.008, +0.013 ± 0.009, −0.002 ± 0.002; gated `blend`: +0.005, +0.007, −0.002): the
forward-model terms and the engine fusion absorb most of the base-list change. These caches have the truth in the
pool, like the ones that mispredicted v7b, so they do not predict the LB.

**Truth removed from the pool** (800 hold-out molecules, `drop_pid`; engine-2 and the PubChem channel not run): MRR
0.083 against 0.724 with the truth present. The generator proposes the truth for 15.5 ± 1.3% (natural-product
catalogue 20%, synthetic 2%); 49% of the molecules get no generated candidate. A classifier on list-shape features
tells truth-absent from truth-present lists with AUC 0.89 (0.76 for natural products).

Presets with commit runs: `v10a-v4b-tiefree-gate07` (v4b + `public_free`, gate 0.7; no new dataset) and
`v10b-v4b-blend-gate07` (v4b + `blend`, gate 0.7). Expected LB effect of either: −0.005 / +0.002 / +0.010. Nothing
submitted.

Commit runs of v10a and v10b (2026-10-02): clean. The ranker files load on Kaggle, the ungated branch ran on all 12
smoke molecules without error, 0 fallbacks. v10b's smoke `submission.csv` is byte-identical to v4b's; v10a's differs
in one molecule (m_068354, lib_max 1.0, ranks 12 and below). That is not the new code: the v1 engine itself has two
states for this molecule (base scores differ by up to 0.15 between runs with identical settings, e.g. v9a against
v9b), probably a time-budgeted feature. Known quirk: smoke outputs are comparable byte for byte only up to this molecule.

## Mixed regime (2026-10-02): why v7b failed, and a harness that sees answers outside the pool

Source: `reviews/mixed-regime-2026-10-02.md` (`prototypes/mixed_regime.py`, `scratch/mixed/`). Engine-2 and the PubChem
channel now run locally as well.

**Root cause of the v7b miss: a harness artefact.** Engine-2's COCONUT and ChEBI / LIPID MAPS files contain no
training structure (40 keys of 275,810). Training structures reach engine-2 only through the `train.parquet` it is
given, and every class-2 validation run (val1, valA, valF) deleted the hold-out structures from that file. Engine-2
therefore never had the answer (truth in its top 40: 3.0%, 0.9%, 8.0%; 99.6% on valc1) and could only add wrong
candidates, so gating it off looked like +0.07 to +0.14. On the hidden test a class-2 answer is a COCONUT row of
engine-2 like any other. **Every earlier local statement about engine-2 on class-2 molecules is void**, including
"engine-2 lists the truth for 0.5 to 7.6% of molecules without library references".

The regime: 380 hold-out molecules, each with the truth in the v1 pool and with it removed, 150 also class-1-like;
types `c1`, `c2` (truth in both engines' candidates), `dA` (truth only among engine-2's), `dB` (in neither), `c2x`
(the old harness). v4b level per type (no forward models / with ICEBERG on a 169-molecule prefix): c1 0.83 / 0.85,
c2 0.74 / 0.74, dA 0.28 / 0.40, dB 0.085 / 0.13. v7b − v4b: c1 −0.002, c2 −0.010, dA −0.088, dB +0.018, c2x +0.149;
−0.003 to −0.011 under mixtures at the LB level, against −0.004 on the LB. It does not reproduce v3b below v2 (frag
for all), so it is not a calibrated predictor; it is the first harness that gets the engine-2 question right.

What it supports (paired against v4b, three assumed mixtures): `RANKER='blend'` + gate 0.7 +0.005 to +0.011;
`public_free` + gate +0.003 to +0.006; `SLOTS_GENTLE=[3,6,9,12,15]` +0.002 to +0.004; `PC_FWD_T1=0.0` +0.003 (ICEBERG
prefix only). Rejected: `FUSE_MIN_LIB` at any value, `FUSE_ALPHA` 0.8 and 1.0. Not supported (weak evidence): the soft
library gate. With LB 0.41 the type levels imply that about half of the hidden molecules are `dB`-like: the answer is
in neither candidate set. That is where the score is lost.

Engine-2 finds an answer the v1 list lacks for 83 to 85% of such molecules (median rank 1 to 2), but the
reciprocal-rank fusion gives an engine-2-only candidate the weight of v1 rank 3.7, so it is almost never first.

Preset `v11-v4b-bundle`: v4b + everything the regime or the LB supports (`RANKER='blend'` gate 0.7, slots 3-6-9-12-15,
`PC_FWD_T1` 0.0, `GL_MH_ONLY`, `ICE_SURROGATE` gap). Each part is below what one submission resolves; together the
regime puts them at about +0.01 to +0.02. Commit run pushed, nothing submitted.

## Simulated spectra for FPNet (2026-10-02): pilot

Source: `reviews/simspec-2026-10-02.md`, `train/README.md` question 2. ICEBERG [M+H]+ spectra at 20 / 40 / 60 eV for
11,523 structures (37 predictions/s). Fine-tune `ft_sim25` against `ft_ctrl`, top-1 on real spectra: structures whose
simulated spectrum was trained on 0.407 → 0.520 (+0.113 ± 0.008; outside the MassSpecGym sources +0.131 ± 0.018),
structures not simulated +0.014 ± 0.007, everything else unchanged. Most of the gain is a prior for "fingerprints the
model was trained on": against a competitor that was simulated too the gain is +0.010 ± 0.006 per pair, and a truth
that was not simulated loses 0.027 against a simulated competitor. The same prior exists today for training structures
against COCONUT-only structures, which is the fingerprint-model twin of the ranker's tie-order code. Second
experiment running (`scratch/simspec/run_chain5_y11.sh`): all 21,014 window competitors of 431 targets simulated too.

## v10a result: 0.408 (2026-10-02)

v10a = v4b + `RANKER='public_free'`, `RANKER_LIB_GATE=0.7` scored **0.408** against 0.409 for v4b. Predicted 0.411
(0.404 to 0.419); the mixed regime gave +0.003 to +0.006. One public-LB molecule at rank 1 is 0.0076, so the result is
"no measurable effect": removing the tie-order code neither helps nor hurts at this resolution. It does not test our
own ranker (`blend`), which the mixed regime rates about twice as high (+0.005 to +0.011) and which is in v10b / v11.
Nine submissions of this pipeline now lie between 0.390 and 0.409. Changes to how pool candidates are ordered do not
move the public score; consistent with the estimate that about half of the hidden answers are in no candidate set.

## Simulated spectra, second experiment (2026-10-02): competitors simulated too

`scratch/simspec/run_chain5_y11.sh`, fine-tune `ft_simw`: the 21,014 window competitors of 431 targets were simulated
as well, so "the model was trained on this fingerprint" no longer singles out the truth (the situation of a full run).
Real spectra, paired against `ft_ctrl`:

| Queries | n | control top-1 / MRR | pilot (`ft_sim25`) | `ft_simw` | `ft_simw` − control, top-1 | MRR |
|---|---|---|---|---|---|---|
| T outside the MassSpecGym sources | 404 | 0.334 / 0.489 | 0.465 / 0.603 | 0.399 / 0.541 | +0.064 ± 0.017 | +0.051 ± 0.011 |
| np panel half Tp (timsTOF) | 246 | 0.325 / 0.501 | 0.451 / 0.598 | 0.378 / 0.526 | +0.053 ± 0.024 | +0.025 ± 0.017 |
| ... [M+H]+ queries | 129 | 0.395 / 0.582 | 0.643 / 0.755 | 0.512 / 0.649 | +0.116 ± 0.039 | +0.067 ± 0.026 |
| ... negative queries | 115 | 0.243 / 0.421 | 0.296 / 0.470 | 0.252 / 0.409 | +0.009 ± 0.023 | −0.012 ± 0.017 |
| C (not simulated) | 2,185 | 0.399 / 0.558 | 0.413 / 0.569 | 0.403 / 0.563 | +0.004 ± 0.007 | +0.005 ± 0.004 |
| np panel half Cp (not simulated) | 253 | 0.320 / 0.501 | 0.316 / 0.488 | 0.304 / 0.484 | −0.016 ± 0.022 | −0.017 ± 0.014 |
| hold all / hold not NP-catalogue | 3,000 | 0.669 / 0.702 | | 0.678 / 0.704 | +0.009 / +0.002 | |

About half of the pilot's gain survives: +0.05 to +0.06 top-1 inside the 10 ppm window for a natural product whose
simulated spectrum was trained on, and only for [M+H]+ queries (the simulation is [M+H]+ only; negative queries gain
nothing). Pairwise, the truth gains nothing against simulated competitors (−0.001 ± 0.006 on Tp) and +0.01 to +0.03
against competitors with real training spectra: the effect is the removal of the model's preference for structures it
has real spectra of. A structure that is not simulated loses 0.017 ± 0.005 per pair against simulated competitors,
which would apply to PubChem-only and novel answers after a full run.

Decision: the full run (436,515 structures, about 13 h of simulation and 5 to 10 h of training, then fine-tuning every
fingerprint model of the pipeline and re-fitting the ranker on the new features) is not started. The gain is a
fingerprint-level +0.05 top-1 for pool answers measured in [M+H]+; the ranker and the fusion sit on top of it, and
v10a showed that a change of similar size in how pool candidates are ordered does not move the public score. It stays
on the list behind the work on answers that are in no candidate set.

## PubChem join (2026-10-02/03): PubChem candidates ranked by the v1 engine

Source: `reviews/pcjoin-2026-10-02.md`, `nb/embed/pc_join.py`. Flags (default off): `PC_JOIN_N` (how many of the best
PubChem-only candidates enter the v1 engine as extra candidates and are placed by the ranker), `PC_JOIN_MAX_LIB` (no
injection at or above this `lib_max`), `PC_JOIN_ORDER`, `PC_JOIN_MIN_REL`. The PubChem channel lists
`max(25, PC_JOIN_N)` structures; the fusion still sees the first 25 for the slot merge. On commit runs every smoke
molecule is joined regardless of the gate so that a bug fails the commit; on the scoring rerun any failure falls back
to the run without injection.

Local measurement in the mixed regime (380 hold-out molecules; N 50, public ranker, against today's slot merge):

| Molecule type | no forward models | ICEBERG prefix (169) |
|---|---|---|
| truth only in PubChem (has a record) | 0.080 → 0.274, +0.194 ± 0.018 (153 / 27) | +0.116 ± 0.022 |
| ... natural-product catalogue structures only | +0.107 | +0.067 |
| truth in the pool, no spectra | +0.013 ± 0.006 | −0.005 ± 0.009 (gate 0.7) |
| class 1 | −0.006 ± 0.006; −0.029 ± 0.010 on the v1 side alone | −0.006 ± 0.004 (gate 0.7) |
| with the gate `lib_max` < 0.7 | truth only in PubChem +0.163 ± 0.018 (116 / 11), class 1 on the v1 side −0.002 | +0.098 ± 0.020 |
| gate 0.7 + `RANKER='blend'` | +0.204 ± 0.019 (128 / 9) | +0.127 ± 0.021 (47 / 3) |

The class-1 cost sits in lists with `lib_max` 0.7 to 0.9 and is hidden by engine-2 after the fusion; the gate at 0.7
removes it and keeps 84% of the gain. Runtime: about +1 s per molecule on Kaggle. Conditional LB range for the gated
join with the public ranker: +0.002 / +0.010 / +0.03 (scaled by the one public probe of the PubChem list alone, 0.037);
the mixtures of the mixed regime give +0.02 to +0.04. Caveats: the truths are masked training structures, engine-2 and
ICEBERG have seen them, natural-product structures gain a third of what synthetic ones gain.

Presets: `v12a-v4b-pcjoin50-lib07` (v4b + join, one change) and `v12b-v4b-pcjoin50-blend` (v12a + `RANKER='blend'`,
gate 0.7). Commit runs pushed 2026-10-03.

## v11 result: 0.405 (2026-10-03)

v11 = v4b + `RANKER='blend'` (gate 0.7) + slots 3-6-9-12-15 + `PC_FWD_T1` 0.0 + `GL_MH_ONLY` + `ICE_SURROGATE` gap
scored **0.405** against 0.409 for v4b. Predicted 0.415 (0.403 to 0.428); the mixed regime gave about +0.01 to +0.02.
Ten submissions of this pipeline now lie between 0.390 and 0.409. Our ranker, the order-free ties (v10a 0.408) and the
small fusion changes do not move the public score; the mixed regime does not predict effects of this size (it got the
sign of v7b right, not the size of v11). v12b (join + blend) is therefore not submitted; v12a (join, public ranker,
one change against v4b) goes in at the daily reset.

## v12a result: 0.420, new best (2026-10-03)

v12a = v4b + PubChem join (`PC_JOIN_N=50`, `PC_JOIN_MAX_LIB=0.7`, public ranker) scored **0.420** against 0.409 for
v4b: the first of eleven submissions of this pipeline outside the 0.390 to 0.409 band, and the first change that acts
on answers outside the main pool. Predicted 0.415 (0.403 to 0.44); the agent's probe-scaled range was +0.002 / +0.010
/ +0.03, its central value matches. The scoring run took about 3 h 50 min (others about 3 h 10 min).

## v13a result: 0.431, new best (2026-10-03)

v13a = v12a with `PC_JOIN_N=100` (was 50) scored **0.431** against 0.420: a second step of +0.011 from the same
mechanism, so the join is now supported by two submissions (0.409 → 0.420 → 0.431). Predicted 0.423 (0.412 to 0.432).
Local paired difference N 100 − N 50 (public ranker, no gate): truth only in PubChem +0.016 ± 0.007 without forward
models and +0.033 ± 0.014 on the ICEBERG prefix; truth in the pool without spectra +0.005 ± 0.008; class 1 −0.001 to
−0.005. The LB step from 50 to 100 is as large as the step from 0 to 50, which the local sets did not predict (they
show diminishing returns): hidden answers sit deeper in the PubChem list than the local, well-documented ones.
Next: deeper lists (N 200), the gate at 0.8, and `GL_MH_ONLY` on top; commit runs pushed, nothing submitted.

## v14a result: 0.406, and the diagnosis (2026-10-03/04)

v14a = v13a with `PC_JOIN_N=200` scored **0.406** against 0.431 (N 100), 0.420 (N 50) and 0.409 (no join). Predicted
0.436; the mixed regime said +0.010 to +0.024 on PubChem-only answers and about −0.002 elsewhere.

Not a silent failure. Two full-size commit runs on the 400 visible molecules with the join forced on every molecule
(`casmi26-diag-n100`, `casmi26-diag-n200`): 0 errors and 0 join fallbacks in both, resident memory at most 10.2 GB
(N 100) and 11.8 GB (N 200), the same run time (2.8 h), ICEBERG and GLACIER complete for all covered molecules, and
MRR 1.000 on the visible molecules in both.

So the drop is an effect of the extra candidates. `reviews/next-levers-2026-10-03.md` finds the mechanism locally at
about a fifth of the size: on the v1 side 11 / 17 / 24 of 188 correct top-1s of answers in the pool are lost to injected
rows at N 50 / 100 / 200, while a truth from fingerprint rank 101 to 200 is placed at reciprocal rank 0.09 by the public
ranker (none of 13 first). Locally engine-2 repairs most of the loss after the fusion; on the hidden test it evidently
does not. **N 100 is the best known depth; the regime cannot judge how many outside candidates should enter.**

Same day, from the two finished studies (`reviews/training-plan-2026-10-03.md`, `reviews/next-levers-2026-10-03.md`):
no model is worth GPU time now (a ranker fitted on joined lists is not distinguishable from the existing blend after
the fusion; a retrieval model for the PubChem window does not beat the fingerprint score; the simulated-spectra
fine-tune would cost 15 to 20 GPU hours for a feature block worth 0.015 to 0.05 MRR before the fusion). Supported small
changes at N 100: `PC_N1=20000` (the first pass cuts the truth for 7.7% of PubChem-record molecules), an early slot or
injection for engine-2's candidates, a 5 ppm window. Not supported: deeper N, a natural-product prior in the
selection, the gate at 0.8.

## Fusion credit for joined rows (2026-10-04): tested locally, no net gain

Idea: in the engine fusion a joined PubChem row can only collect the v1 share, because engine-2 cannot list it, so a
pool decoy that both engines list overtakes a joined truth that the v1 engine ranked first. `FUSE_INJ_CREDIT` (default
0; joined rows carry pool id −2 in the base list) gives a joined row that share of the engine-2 weight at its own rank.
`scratch/inj_credit_eval.py`, public ranker, N 100, gate 0.7, no forward models, paired against credit 0:

| Credit | truth only in PubChem (325) | truth in the pool, no spectra (380) | class 1 (150) |
|---|---|---|---|
| 0.5 | +0.027 ± 0.005 (57 / 0) | −0.013 ± 0.004 (0 / 20) | +0.002 |
| 1.0 | +0.049 ± 0.008 (71 / 0) | −0.037 ± 0.007 (0 / 37) | −0.003 |
| 2.0 | +0.075 ± 0.011 (77 / 1) | −0.171 ± 0.013 (1 / 139) | −0.009 |

The engine-2 half of the fusion is what keeps joined decoys below correct pool answers; what the credit gives to
PubChem-only answers it takes from pool answers at about the same rate. Under the assumed mixtures the net is zero,
and v14a showed the hidden test punishes extra decoys harder than the regime does. Not submitted; the flag stays off.

## v14b 0.416, v14c 0.431, v15b 0.431, v15a 0.417 (2026-10-04): what moves the join and what does not

All four are single changes on v13a (0.431, N 100, join gate `lib_max` < 0.7, public ranker).

| Submission | Change | Public LB | Local regime said |
|---|---|---|---|
| v14c | GLACIER on [M+H]+ only | 0.431 | not replayable |
| v15b | `PC_N1` 5,000 → 20,000 | 0.431 | +0.017 on PubChem-only answers |
| v14b | join gate 0.7 → 0.8 | 0.416 | about +0.01 on PubChem-only answers, 0 elsewhere |
| v15a | `RANKER='blend'` (gate 0.7) | 0.417 | +0.048 / +0.030 on PubChem-only answers, +0.017 / +0.003 on pool answers |
| v14a | N 100 → 200 | 0.406 | +0.010 to +0.024 on PubChem-only answers |

The three losses share one property: more joined rows end above pool candidates (deeper list, wider gate, a ranker
that scores joined rows higher: a joined decoy tops 18% of pool-answer lists under the blend against 8% under the
public ranker). The two neutral changes do not alter how many joined rows outrank pool rows. The public ranker's
accidental penalty on joined rows (last among ties, list-relative features) is therefore not a defect to remove but
the thing that makes N 100 work. Ranker work is closed: our ranker lost with the join (0.417) and did nothing without
it (0.405 to 0.408 against 0.409).

Same day, local studies (reports in `reviews/`): engine-2 scoring the joined rows (`eng2-join`), MIST as a second
retrieval model (`mist`) and a calibration of the regime to the leaderboard (`lbcal`) are negative; a 5 ppm PubChem
window with the +2 ppm [M+H]+ offset corrected (`masscal`) is a small coverage gain; a confidence gate that joins only
when the unjoined list looks wrong (`confgate`, flag `PC_JOIN_CONF`) keeps 88 to 94% of the local gain while touching
16 to 29% of the pool-answer lists instead of 72%. v17b (`PC_JOIN_CONF=0.7`) is submitted. `STRICT` (fail the run if
any stage fell back) exists for a diagnostic resubmission of N 200.

2026-10-05: v20a (v13a + `ICE_FT`, `STRICT`) scored **0.401** on the public leaderboard, 0.030 below v13a. The run
completed, so with `STRICT` no stage fell back on the hidden test: the loss comes from the fine-tuned ICEBERG intensity
model itself (ahmedberatozer/casmi26-ice-ft), not from a failure. It also shows that v13a's stages run clean on the
hidden test. `ICE_FT` stays off; a model fine-tuned on the training spectra without a held-out check is not to be
submitted again.

v19a (`PC_JOIN_N=200`, `PC_JOIN_CONF=0.5`) scored **0.409**: the score of the pipeline without any join. N 200
without the gate was 0.406 (v14a), N 100 with the gate at 0.7 was 0.430 (v17b). Restricting the deep join to lists
that look wrong does not rescue it, so the loss at N 200 is in the lists that need the join, not in the pool-answer
lists the gate protects: candidates 101 to 200 push the answer down among the first 100. N stays at 100; depth is
closed as a lever unless the selection of the candidates improves.

v22a (v13a with the 100 joined candidates selected by the hard-negative fine-tune, `PC_MODELS='hn'`) scored **0.423**,
0.008 below v13a; the scoring run took 4 h 50 min (second channel pass). Locally the selection holds the answer in the
first 100 for 77.8% instead of 70.5% of unseen molecules and the v1 side gains +0.028 on answer-removed molecules
(`reviews/hardneg-2026-10-05.md`); the full replay with real ICEBERG / GLACIER scores (`reviews/fwd-replay-2026-10-05.md`,
finished after the submission) showed dA +0.015, dB +0.030, c2 −0.005 and a joined row on top of 8.2% instead of 6.6%
of the pool-answer lists. The leaderboard again follows the exposure number, not the answer-removed gain: every change
that raised the share of pool-answer lists topped by a joined row has lost (N 200, gate 0.8, ranker blend, now the
fine-tuned selection, whose candidates look more like measured structures and are scored higher by the ranker and the
forward models). `PC_MODELS` stays off. Rule from here on: a join variant is not submitted unless the full replay with
real forward-model scores shows the exposure on c2 / c1 at or below v13a's.

## NP-tuned FPNet in engine 2 (2026-10-05): flag added, not submitted

Source: `reviews/np-eng2-2026-10-05.md`. The teammate's np1 (0.413 on v4b 0.409) used `goooman/casmi26-fp-npx4-soup`
(`fp_single_npx4soup.pt`, Apache-2.0): it joins the per-spectrum group of engine 2's fingerprint models. Flag
`ENG_EXTRA_FPNET` (default `None`), preset `v25a-v13a-np-eng2` = v13a + the flag. Full replay with real forward scores
on the 380 mixed-regime molecules, with minus without: `c1` +0.0044 ± 0.0035, `c2` +0.0020 ± 0.0023, `dA` +0.0028 ±
0.0018, `dB` -0.0026 ± 0.0023 (none beyond two standard errors); pool-answer lists topped by a joined row 25 -> 24
(`c2`) and 6 -> 6 (`c1`); engine 2's own list +0.008 ± 0.003. Limits: engine 2's fingerprint features are NaN for 58%
of the local molecules (a spectrum without collision energy makes the logits NaN; 0% of Enveda spectra in train), and
the regime holds no timsTOF natural product. The training data of the soup's base model is undocumented: not for a
final selection until the author answers. No commit run was built.

2026-10-06: v24a (v13a with `ICE_LAM` = `GL_LAM` = 0.75) scored **0.437**, +0.006 on v13a and the new best; the scoring
run took a little over 4 h. It is the first submission since the join that gains, and the first test of the exposure
rule in the gaining direction: in the full replay it lowers the pool-answer lists topped by a joined row from 25 to 22
(answer lost first 8 to 5) with c1 +0.005 ± 0.004, c2 +0.011 ± 0.004 (22 / 5), dA +0.001, dB +0.001
(`reviews/combo-2026-10-05.md`, context rows), and on the Enveda panel c1 / c2 gain in all four arms with dB −0.004 to
−0.009 (`reviews/panel2-2026-10-05.md`). The parameter search had found the same direction as its only consistent
signal (`reviews/param-search-2026-10-05.md`). Same-day studies that did not lead to a submission: calibration of the
replay to the leaderboard (none; veto only), parameter search, joined-row verifier, their combination, the pair-tail
ranker of engine 2, an NP-tuned engine-2 FPNet of our own, mass-window pruning, the Enveda panel regime (reviews of
2026-10-05). Next on the same axis: weights 0.5 (replay: c2 +0.026, c1 +0.009, dA −0.004, dB −0.007, 19 exposed lists).

Line search on the forward-model weights (`ICE_LAM` = `GL_LAM`, everything else v13a): 1.0 → 0.431 (v13a), 0.75 →
**0.437** (v24a), 0.6 → 0.430 (v24c), 0.5 → 0.433 (v24b). The curve is not monotone on either side of 0.75, so the four
points mostly measure the noise of a single submission, about ±0.004 here; the +0.006 of v24a is at the edge of it.
0.75 stays as the best measured setting, and no further submissions go to this axis. Lesson for the next candidates:
a single-knob change needs an expected effect well above 0.005 to be read off one submission.

2026-10-06, evening: v33b scored **0.442**, +0.005 on v24a and the new best. v33b = v24a with, below `lib_max` 0.7, the
base list from the rank average of the public ranker and a LightGBM ranker fitted to the fold-safe FPNet's features on
molecules that bank has not seen (`RANKER=fs_avg`, `nb/embed/fs_pass.py`, dataset casmi26-ours-ranker-fs;
`reviews/rankbank-2026-10-06.md`, `reviews/fsranker-2026-10-06.md`). The same day closed, all without a submission:
adduct surrogates, fingerprint-bank ensembles, the DreaMS NP heads, the Enveda-180 Zenodo release as a library, a fix
of the forward stage for molecules with extra adduct spectra, the generator join (reviews of 2026-10-06), and the
teammate's fine-tuned GLACIER (0.414). What the day showed: better fingerprint models do not reach the list through
the public ranker (it is fitted to the public models' features on molecules they memorised), but a ranker fitted in the
unseen-molecule regime does. Found on the way: the public engine breaks merged-adduct ties by set order, so base lists
differ between sessions for about a quarter of the molecules (a source of the ±0.004 leaderboard noise; not fixed yet).

2026-10-07: v34a (v33b with `FWD_A_SCALE=[0,0]`: no ICEBERG / GLACIER term in the first re-rank of the base list) scored
**0.431**, 0.011 below v33b: about three public molecules, outside the noise of one submission. Locally it was the cleanest
candidate we ever had (all eight type cells positive on two sets, exposure 23 → 20, family-wise p 0.002;
`reviews/fsranker2-2026-10-07.md`). The local sets therefore understate what the forward models contribute on the hidden
test, in spite of the argument that their leak should make them look better locally; the same was already visible in
the line search (0.6 → 0.430, 0.5 → 0.433) and in both fine-tuned forward models (0.401, 0.414): the stage is far more
sensitive on the leaderboard than in any replay. Closed: weakening or removing forward-model terms. `FWD_A_SCALE` stays
off. A teammate's run with `FUSE_ALPHA` 0.3 on v24a scored 0.406 (−0.031): the engine fusion at 0.6 also stays. The
same day's studies: no ranker refit beats the production recipe and more training lists help slowly
(`fsranker2`); engine 2's ranker rows contain the Enveda panel, so panel replays are in-sample for engine 2, and
`FUSE_KRR` 1 is the one candidate left from the audit (`regime-audit`, preset v35a, not submitted); run-to-run
variation inside the notebook is 0.0003 on 400 molecules, and the public leaderboard has about 130 molecules, so one
answer moving between place 1 and 2 is 0.0038 (`determ`).

2026-10-07, afternoon and evening. Two more submissions on v33b, neither above it.
- v33a (`RANKER='fs'`: our fitted ranker alone below the gate) scored **0.434** (−0.008). With v24a (weight 0, 0.437) and
  v33b (rank average, 0.442) the axis is closed: the two rankers complement each other and the average is the point to keep.
- v40a (v33b + `FUSE_SKIP_LIB=0.97`, `POP_LIB_OFF`, `FRAG_LIB_OFF`) scored **0.442**, the same as v33b. It was submitted on
  outside evidence (two forks of our public notebook moved this way, one at 0.426 against 0.421); the local replay, run
  afterwards, showed composite −0.002 ± 0.001 (`reviews/libprotect-2026-10-07.md`). Neutral on the public leaderboard.
  From here on every candidate is replayed locally BEFORE it is submitted.
- Not submitted: v41a (v40a + `FP_BANK='ens'`). Local replay with the true `fpnet_full1` bank: v33b, v40a and v41a are
  indistinguishable; `ens` alone is about +0.001, the protection about −0.002 (`reviews/fpens-2026-10-07.md`).
Studies of the day that closed a direction: more training lists for our ranker (flat curve, `fsranker3`), a coherent
leak-free chain for engine 2 from megayak's FPNets and rows (worse; his rows carry an answer-position leak, `eng2fs`),
a local regime with nothing seen before the forward models (does not track the leaderboard: 5 of 11 signs, `uregime`),
molecular networking among the test molecules (too sparse, oracle ceiling +0.004, `testnet`), DreaMS embedding
similarity for analog search (not better than the v1 kernel, `dreamsim`). Open: FIORA (open weights, MIT, MSnLib) as a
forward model for negative-mode-only molecules (`docs/landscape/forward-models-2026-10-07.md`), not downloaded yet.

2026-10-08: more popularity loses. v44a (`POP_MU` 0.35) scored **0.432** (−0.010 on v33b) and v46c (v44a + in-group
prior below the library gate) **0.426** (−0.016). Both were replayed locally first: v44a +0.0035 ± 0.0016 on the Enveda
panel and +0.0008 on the mixed natural products; v46c +0.016 on the panel, +0.002 mixed, −0.003 on the less famous half,
pooled null p 0.0008 (`reviews/popmu-2026-10-07.md`, `popgroup-2026-10-08.md`). The leaderboard is monotone the other way
(0.25: 0.442, 0.35: 0.432, plus in-group: 0.426), each step well outside one answer (0.0038): the hidden answers are far
less famous than any local set, as the circularity check had warned (in the bottom count quartile 45 to 78% of the
prior's signal comes from the library deposit itself, `reviews/popmodel-2026-10-08.md`). Closed: strengthening the
popularity prior in any form. Open question raised by the size of the loss: whether 0.25, chosen on an earlier chain
without the join and without our ranker (0.15 → 0.25 gave +0.005 then), is now too high; to be replayed locally before
anything is proposed. Studies of the day before: fitted popularity model (its gain is one membership flag; loses in the
replay), availability / vendor signal (adds nothing over the substance count), FIORA for negative mode (at chance on
natural products), a third ranker in the average (no diversity beyond reweighting), inference of the hidden fame
profile from past scores (not identifiable).

2026-10-08, after the two popularity losses: six studies, no candidate. None was submitted.
- `poplow`: a weaker prior (`POP_MU` 0.15, preset v47a) loses locally (−0.004 ± 0.002 on both sets); it gains only when
  the answer's own fame is masked. `obscure`: restricting or reweighting the local sets to little-documented answers does
  not make them track the leaderboard better (8 and 5 of 14 signs on the full sets; no subset interval excludes zero), so
  fame is not the one variable, and v47a stays unsupported (−0.004 on every real stratum).
- `nplike`: structure-only natural-product-likeness (RDKit NP_Score, own COCONUT-against-PubChem model, leak closed) is at
  chance on pool rows and weak on joined rows; 16 replay cells, none passes. Flags `NP_MU` / `NP_JOIN_MU` stay, default off.
- `newinfo`: the data has no CCS, retention time, MS1 or batch id; row order, isotope peaks, base-peak intensity and a
  source-organism prior carry nothing; the adduct set is real chemistry but too weak (`ADD_JOIN_MU`, preset v50a, not
  recommended).
- `metafit`: a final combiner fitted in the unseen regime (606 molecules) gains locally only by widening the join
  (exposure 24 → 42 lists) or by picking constants that already lost on the leaderboard (forward weights 0.5, `POP_MU`
  0.35); without those levers +0.002 ± 0.005. The hand-set constants are not measurably wrong.
- `rankobs`: ranker variants that lean less on fame (inverse-fame weights, obscure-only training, dropped or residualised
  feature families) do not beat a reseed of the production recipe on the obscure tercile; inverse-fame weights gain only
  on answers without a PubChem record (+0.0085 ± 0.0039, four seeds) and are neutral in the full replay (preset v48a,
  dataset not uploaded).

2026-10-08, evening: three single-change probes on v33b, all below it. v35a (`FUSE_KRR` 1) **0.437**, v47a (`POP_MU` 0.15)
**0.426**, v48a (inverse-fame ranker) **0.436**. Caveat that governs the reading: Kaggle changed the scoring on 2026-10-07
(stereo stripped before the tautomer key at 21:32 UTC, one-electron fix of the hidden test's peak masses at 23:49 UTC;
`docs/landscape/update-2026-10-08.md`) and announced a rerun of all submissions that had not landed by 2026-10-09 18:00
UTC. v33b's 0.442 is from before the change; all five submissions since (0.432, 0.426, 0.437, 0.426, 0.436) are from
after it, and every one is lower, including v48a, which is neutral locally. The popularity axis is non-monotone across
the change (0.15: 0.426, 0.25: 0.442 old regime, 0.35: 0.432), which a regime offset of about −0.005 to −0.010 on v33b
explains more simply than a sharp optimum at 0.25. Until v33b has a score in the new regime, none of the five deltas is
interpretable; the next submission is v33b itself, unchanged. Other studies of the day (no candidate): the one-electron
shift moves v33b by at most one or two answers locally and needs no correction (`emass`); stereo-collapsed rows 0 of 400
(`KEY_NOSTEREO`, off); engine 2 with 4 seeds and `N_ANALOG` 250 is flat; a library-split popularity count is inert
(`popsplit`, `POP_SPLIT`, off); `FUSE_KRR` is monotone locally and moves the same lists as v34a and `FUSE_ALPHA` 0.3
(`combo2`).

2026-10-09: the new scoring regime, measured. v33b resubmitted unchanged scores **0.433** (0.442 before Kaggle's change of
2026-10-07; the announced rerun of old submissions never landed). Against that baseline the probes of 10-08 read: v35a
+0.004, v48a +0.003, v44a −0.001, v47a −0.007, v46c −0.007; one answer between place 1 and 2 is 0.004 on the public
part (about 124 molecules), so nothing there is distinguishable from v33b, and the reading "hidden answers are
obscure" drawn from v44a / v46c is withdrawn. v59a (licence fix: CC BY ChEBI / LIPID MAPS set) **0.434**, the new
reference; v54a (`FUSE_KRR` 1 + `FRAG_LAM` 0.25) **0.431**: the locally additive +0.011 does not show.
Two diagnostic submissions of v59a: first row only **0.377**, first two rows **0.413**. So on the public part 37.7% of
the answers are at rank 1, 7.2% at rank 2, ranks 3–25 carry 0.021 in total (about 10% of the molecules at the local
mean reciprocal rank of such answers), and roughly 45% of the molecules have NO answer among our 25 rows (bounds 3% to
49%). Where we list the answer at all we are right at rank 1 in about two cases of three; the loss is coverage, not
ranking. Class fit with local profiles (`reviews/budget-2026-10-10.md`, `scratch/budget/b6_mix.py`): about 44% of the
hidden answers behave like "in none of our candidate sources or never listed". Consequences: constants around the
reference are closed for good; the open work is candidate coverage for answers outside the pool and the joint
spectrum–structure model as selector (`train/joint/`, phase 1 running). Other studies of the day without a candidate:
open MS/MS spectra dataset (93% already in train, `extlib`), metric-key votes and tail fill (answer affected in 0 lists,
`keyvote`), more engine-2 representatives (flat), public 0.436 / 0.433 notebooks (nothing we lack, `pubdiff`), fewer
spectra per molecule ("low-observation" means a small sample; no second regime mismatch, `lowobs`).

2026-10-10: coverage, first leaderboard test. v62a (generated one-step analogs, five rows from position 6) **0.433** and
v62b (from position 11) **0.434**, against v59a 0.434: no answer gained on the public part, none lost. The hidden
answers that we do not list are not one biotransformation away from our best-ranked parents (the F-like end of
`reviews/gen3-2026-10-10.md`), or the five slots are too few. Same day, without a candidate: our PubChem source is
complete inside its filter and other catalogues add nothing (`coverage2`); a joint spectrum–structure model adds
+0.013 ± 0.004 to our ranker on DEV but nothing or less on Enveda spectra, gate failed (`joint1`); the public 0.440
notebook (canqiang e006 = flexonafft v80 + ICEBERG-only re-rank where [M+H]+ is missing) has nothing we lack, and
showed a hole of ours: GLACIER's scores are dropped when ICEBERG returns nothing (`FWD_GL_ALONE`, off; `canqiang-e006`).
Licences: the host cleared datasets that are non-commercial only by inheritance from train.parquet, PubChem and COCONUT
(thread 745072, 2026-10-09); open: code without a licence, training-code delivery, prvsiyan's CC BY-NC-SA sets
(`docs/competition/clean-chain-plan-2026-10-10.md`). Running: own fold-safe fingerprint nets (`clean/`), an ensemble
with the e006 pipeline.

2026-10-10, evening: three probes built on the public 0.440 notebook (canqiang e006), none above v59a (0.434).
v65c (our list fused with e006's at weight 0.3) **0.430**; v66a (our pipeline + its strong-branch promotion) **0.434**:
the promotion changed nothing, as with popularity masked locally; v66c (v66a + popularity 0.10 on joined rows)
**0.428**: the joined-row popularity term costs. All differences are below one answer at rank 1 (0.008). A fusion
scoring below both parents (0.434 and 0.440) says the two lists are not complementary on the public part, or that the
public notebook's 0.440 is itself within noise of 0.434. e006r (the public notebook unchanged, CC BY ChEBI set) was
submitted on 10-11 to measure that. Prepared, not submitted: v67a / b / c (e006's chain fed with our base list,
`reviews/e006imp-2026-10-10.md`). Own fold-safe fingerprint net and refitted ranker (`clean/`, preset v63a): level with
the third-party net on all 11 hold-out subsets and on DEV / FINAL; the natural-product weakness is not the instrument
(same structures on timsTOF and elsewhere: −0.019 ± 0.022) but isomer density among catalogue structures
(`reviews/ownfp-2026-10-10.md`).

2026-10-11: e006r, the public e006 notebook rerun unchanged with the CC BY ChEBI set, scores **0.443** (its page: 0.440;
v59a 0.434). It is the baseline from here on (`submissions/e006r-0.443/`). Submitted on top of it the same day: v67a
(its chain fed with our base list), v67b (plus the final promotion), and the two halves of v67a, v67d (join only) and
v67e (rank average only), if their commit runs pass.

2026-10-12: the v67 line failed. v67a **0.411**, v67b **0.410**, v67d (join only) **0.398**, v67e (rank average only)
**0.411**, against e006r 0.443: four to six answers lost each. v67e differs from e006r only below the library gate and
by a rank average that gained in our own chain, so a real effect of this size is implausible; the scores match what
our base list scores WITHOUT forward models (the stand-in table the stage writes before e006's chain runs). Working
hypothesis: on the hidden set the e006 stage fails or runs out of time and the notebook submits the stand-in; the
commit runs could not show it because all 12 visible molecules are library hits. Not verified: hidden-run logs are
not available. Lesson: a path that the commit run cannot exercise must be run at full size (validation mode on
Kaggle) before it is submitted; four submissions went on one untested path.

2026-10-13: five single-switch patches of e006r (0.443), all in its own chain on its own lists: window 80 **0.443**,
library shield **0.443**, promoted row kept first **0.443**, joint ICEBERG + GLACIER **0.440**, joint + shield **0.440**.
Three change nothing on the public part; the joint re-rank costs 0.003 (less than one swap between rank 1 and 2), the
sign of the public pair (0.438 against 0.440) and the opposite of the local replay (+0.005 to +0.007). The promotion
being neutral here as in v66a says the hidden PubChem answers it would promote are already first. Re-ranking inside
the public chain is closed. Same days: the v67 loss is real, not a silent fallback (full-size validation on Kaggle,
`reviews/diag67-2026-10-12.md`); the metric has no tail effect (`tailfx`); GNPS-NIST14-MATCHES is allowed by the host
by name but 1,498 of its 1,508 structures are already in train (`docs/competition/gnps-nist14-2026-10-12.md`); a
learned fingerprint-conditioned generator fails its gate even with true fingerprints (top-25 10.6% / 0% / 4.0% against
the rule-based 53.1% / 4.5% / 43.0%, `reviews/fpgen-2026-10-12.md`).
