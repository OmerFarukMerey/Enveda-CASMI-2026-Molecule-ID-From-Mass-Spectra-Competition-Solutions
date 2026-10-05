<!-- index: fusion-v4n 2026-10-05 joint parameter search around v13a on the full local replay (real ICEBERG / GLACIER scores): one-at-a-time scan of 29 knobs, two random joint searches (1,600 configurations), exposure as a hard constraint, repeated 5-fold selection with held-out differences and a sign-flip null; nothing passes the acceptance rule, the only consistent direction is forward weights near 0.5 (a c2 / c1 gain paid by dB) -->
# Joint parameter search around v13a (2026-10-05)

Offline only. Nothing was submitted, pushed or committed, no `kaggle` command was run, the GPU was not used, no model
was trained, the visible `test.parquet` was not read, no NIST data. At most three worker processes at below-normal
priority, one BLAS thread each. Scripts: `scratch/search/` (git-ignored). Outputs: `D:\casmi-data\search\` (all new;
every other cache was read, not written). No file of the solution was changed and **no preset was written**: no
configuration passed the acceptance rule (section 5).

**Every number below was computed in this study** unless a file is named. MRR = MRR@25 on the metric key; "±" is one
standard error; differences are paired per molecule against v13a in the same replay; (wins / losses) in brackets;
"ne" = within two standard errors. Setting: the full replay of `reviews/fwd-replay-2026-10-05.md`
(`fusion_core.build_submission` on the 380 mixed-regime molecules, real ICEBERG / GLACIER scores masked to the
candidates `nb/07` would score for the configuration being replayed), types `c1` (150), `c2` (380), `dA` / `dB` (the
325 molecules with a PubChem record). **No leaderboard estimate is given for anything.**

## 0. Bottom line

1. **Nothing passes.** 1,764 configurations were evaluated (164 one-at-a-time, 1,000 + 600 joint); 1,448 are admissible
   on exposure. Under every objective the configuration selected on four folds is positive on the fifth
   (LB-like mix +0.0103, equal mix +0.0096), but in every case **`dB` loses more than one standard error held out**
   (−0.009 to −0.036) and under the LB-like mix `dA` loses 0.062. The search finds trades, not improvements.
2. **How much the search finds by chance:** with the labels "candidate" and "v13a" swapped at random per molecule, the
   best admissible configuration still shows +0.0089 on the LB-like mix in sample (95th percentile +0.0144) and
   +0.0116 on the equal mix (+0.0184). The observed in-sample bests, +0.0129 and +0.0150, are at p = 0.10 and 0.18.
3. **The standard error across the 15 folds is too small by a factor of about two** (the three repeats re-use the
   same molecules): under the null the criterion "held-out mean above two fold standard errors" fires in 17 to 21% of
   the draws. With one unit per molecule it fires in 5 to 7%. Both are reported; the verdict is the same under both
   for the rule as it was set.
4. **The one direction every objective and every fold agrees on is lower forward weights**: all 60 fold selections
   (four objectives) lower both, to `ICE_LAM` 0.25 to 0.75 and `GL_LAM` 0.25 to 0.6 (v13a: 1 and 1). Dose-response of
   the pair on `c2`: −0.025 at 1.5, 0 at 1, +0.011 at 0.75, +0.026 at 0.5, +0.018 at 0 (± 0.004 to 0.012), with the `c2`
   lists topped by a joined row going 33 / 25 / 22 / 19 / 16. The cost is on `dB`: +0.001 at 0.75, −0.007 ± 0.008 at
   0.5, −0.053 at 0. It is the trade the fwd-replay review described, located more precisely.
5. **Worth a submission: no**, by the rule set for this search. The nearest miss is in section 6, with what it would
   and would not test.

## 1. What was replayed, what could be varied, what could not

`scratch/search/s_lib.py` wraps the replay of `scratch/fwd/` (imported, not edited). One configuration on all four
types takes 0.8 s. v13a reproduces the fwd-replay numbers exactly (0.8223 / 0.7339 / 0.5016 / 0.3096, joined row first
on 25 `c2` and 6 `c1` lists) when run with the same caches. Checked at the end against the current `fusion_core.py`
(another study added a default-off flag during this one): 13 of 13 re-evaluated configurations identical.

**One correction to the reference.** The regime's `frag.json` holds a MetFrag score for only 45% of the joined rows in
the 71 molecules ICEBERG cannot score, while `nb/07` scores every base row. `s1_frag.py` computed the 7,930 missing
scores on the CPU (6 s; written to `D:\casmi-data\search\frag.json`, the regime's file untouched). With them v13a is
**`c1` 0.8223, `c2` 0.7338, `dA` 0.4978, `dB` 0.3093**; against the old cache `dA` −0.0039 ± 0.0023 (1 / 8), the others
within 0.0003; exposure unchanged (25 / 6). This is the reference everywhere below.

**Varied exactly** (cached stage outputs suffice): `ICE_LAM`, `GL_LAM`, `POST_ICE_LAM`, `POST_GL_LAM`, `ICE_LAM_LIB`,
`GL_LAM_LIB`, `GATE_TAU`, `LIB_TAU` (downwards), `REL_TH`, `SLOTS_AGG`, `SLOTS_GENTLE`, `FUSE_ALPHA`, `FUSE_KRR`,
`FUSE_N`, `FUSE_ENG_K` (≤ 40), `FUSE_SKIP_LIB`, `FUSE_MIN_LIB`, `FUSE_INJ_CREDIT`, `POP_MU`, `POP_LIB_OFF`, `FRAG_LAM`,
`FRAG_MODE`, `FRAG_LIB_OFF`, `FILL_25`, `PC_FWD_T1`, `PC_FWD_N`, `GL_MH_ONLY` (`gl_mh.json`), `TOPN` (≤ 60),
`ICE_UNION_K` (≤ 40), and `PC_JOIN_MAX_LIB` at any value ≤ 0.7 (the gate is applied when the lists are assembled; all
N 100 lists have forward scores: 1 candidate of 24,507 missing).

**Varied with a stated gap:** `PC_JOIN_N` 25 and 50 (`D:\casmi-data\pcjoin\rows`). Their lists contain candidates that
were never scored: 0.5% (95 of 20,664 on `c2`) at N 25 and **4.2%** (979 of 23,131) at N 50. They are in the scan with
that caveat and were kept out of the joint search.

**Could not be varied** (no approximation was made):

| Knob | Why |
|---|---|
| `PC_JOIN_N` 150, 200 | engine rows exist (`hnvar\join_cur`, `next_levers`), but rows 101 and beyond have no forward scores in `D:\casmi-data\fwd`; scoring them needs the GPU |
| `PC_JOIN_MAX_LIB` above 0.7 | the lists between 0.7 and 0.8 have unscored joined rows in this cache |
| `LIB_TAU` above 0.9 | PubChem-list candidates of molecules at 0.9 to 0.95 were never scored |
| `TOPN` above 60, `FUSE_ENG_K` / `ICE_UNION_K` above 40 | cached base lists hold 60 rows, engine-2 lists 40 |
| `PC_N1`, `PC_PPM`, `MASS_OFFSET_BY_ADDUCT`, `PC_POP_LAM`, `PC_MODELS`, `PC_JOIN_ORDER`, `PC_JOIN_MIN_REL` | need a new PubChem-channel or engine pass and new forward scores |
| `PC_JOIN_CONF`, `PC_JOIN_CONF_DEEP` | replayable from `confgate` predictions, not done: one more list source, and v17b (0.430) is already on the leaderboard |
| `RANKER`, `RANKER_LIB_GATE`, `FP_BANK` | change the base lists; the new top 60 has unscored rows |
| `ICE_FT`, `ICE_SURROGATE`, `GL_SURROGATE`, `ICE_BUDGET`, `GL_BUDGET` | other forward scores / Kaggle time budget |
| `ENG_*` | engine-2 would have to be re-run |

## 2. Objectives and the exposure constraint (fixed before the search)

- **LB-like mix**: `c1` 0.15, `c2` 0.36, `dA` 0, `dB` 0.22 (the remaining 0.27 is unreachable and identical in every
  configuration). This is mixture A of `reviews/mixed-regime-2026-10-02.md` section 5.2, the first reading of the
  budget in `docs/plan/roadmap.md` (class 1 about 16%, pool answer without spectra 36%, "truth never ranked" 38 to
  46%) scaled so that v4b lands at its leaderboard score. `docs/competition/data.md` names the three novelty classes
  and gives no shares; the mix is an assumption.
- **Equal mix**: 0.25 each.
- **Third objective.** `reviews/calib-2026-10-05.md` appeared while this search ran and concludes that there is **no
  calibrated score** (no composite orders the eleven join submissions out of sample). Two stand-ins are reported, both
  weak by their own source: **C12** = 0.5 `c1` + 0.5 `c2`, the only composite of that review above chance in sample
  (42 of 55 pairs; it fails leave-one-out there), and **CAL** = the least-squares type weights of
  `reviews/lbcal-2026-10-04.md` (`c1` 0, `c2` 0.124, `dA` 0.041, `dB` 0.013; "not identified" there).
- A mix difference is computed per molecule (each type's term divided by the type's size), so one molecule is one unit
  in every standard error, fold and permutation.
- **Admissible** = joined row first on no more `c2` lists than v13a (25 of 380) **and** on no more `c1` lists (6 of
  150). The count needs no answer, so it is taken on all lists, not per fold. Also reported, not used as a
  constraint: joined rows in the first 25 (v13a: 9.24 on `c2`, 1.80 on `c1`) and the share of lists with a joined row
  above the answer (7.9% / 2.7%).

## 3. One-at-a-time scan (164 settings)

`s2_scan.py`, `s3_scan_report.py`; the full table with every value is `D:\casmi-data\search\scan.md`. One row per knob,
ordered by the largest per-type difference any value produced. Ranges are the lowest / highest paired difference over
the scanned values.

| Knob | values | largest type difference | `c1` | `c2` | `dA` | `dB` | admissible values | best admissible, LB-like mix | best admissible, equal mix |
|---|---|---|---|---|---|---|---|---|---|
| `FUSE_MIN_LIB` (0.2 to 0.7) | 4 | 0.128 | −0.003 / 0 | −0.013 / +0.001 | −0.128 / 0 | 0 / +0.019 | 0 | none | none |
| `PC_JOIN_N` (0, 25, 50) | 3 | 0.119 | 0 / +0.004 | −0.012 / 0 | −0.071 / 0 | −0.119 / 0 | 1 (N 0) | none above 0 | none above 0 |
| `FUSE_ALPHA` (0 to 1.5) | 8 | 0.098 | −0.021 / +0.008 | −0.050 / 0 | −0.098 / +0.075 | −0.084 / +0.039 | 4 (≥ 0.7) | none above 0 | +0.0029 ± 0.0027 at 0.8 |
| `FRAG_MODE` all (`FRAG_LAM` 0.1 to 1) | 4 | 0.078 | −0.078 / +0.008 | −0.037 / 0 | −0.014 / +0.007 | −0.002 / +0.004 | 1 | +0.0005 ± 0.0022 at 0.1 | +0.0033 ± 0.0026 at 0.1 |
| `GL_LAM_LIB` (0.25 to 2) | 5 | 0.075 | −0.075 / 0 | −0.009 / 0 | 0 / +0.010 | 0 / +0.003 | 5 | none above 0 | +0.0004 ± 0.0004 at 0.25 |
| `ICE_LAM_LIB` (0.25 to 2) | 5 | 0.073 | −0.073 / 0 | −0.009 / 0 | 0 / +0.008 | 0 / +0.003 | 5 | none above 0 | none above 0 |
| `FRAG_LAM`, gap mode (0 to 4) | 6 | 0.070 | −0.070 / +0.011 | −0.036 / +0.002 | −0.019 / +0.007 | −0.011 / +0.001 | 2 (0, 0.25) | +0.0021 ± 0.0014 at 0.25 | +0.0043 ± 0.0018 at 0.25 |
| `PC_JOIN_MAX_LIB` (0.3 to 0.65) | 5 | 0.066 | 0 / +0.001 | −0.005 / +0.002 | −0.026 / 0 | −0.066 / 0 | 5 | none above 0 | none above 0 |
| `FUSE_SKIP_LIB` (0.5 to 0.95) | 5 | 0.065 | 0 / +0.006 | −0.002 / 0 | −0.065 / 0 | 0 / +0.008 | 4 | +0.0013 ± 0.0011 at 0.9 | +0.0001 at 0.95 |
| `FUSE_KRR` (0.5 to 60) | 7 | 0.063 | −0.042 / +0.012 | −0.056 / 0 | −0.004 / +0.006 | −0.063 / +0.009 | 4 (≥ 5) | none above 0 | none above 0 |
| `POST_GL_LAM` (0 to 2) | 7 | 0.059 | −0.059 / +0.004 | −0.029 / +0.011 | −0.012 / +0.014 | −0.017 / +0.011 | 5 | +0.0007 ± 0.0030 at 0 | +0.0023 ± 0.0022 at 0.75 |
| `ICE_LAM` = `GL_LAM` (0, 0.5, 0.75, 1.5) | 4 | 0.053 | −0.012 / +0.009 | −0.025 / +0.025 | −0.052 / +0.003 | −0.053 / +0.001 | 3 | **+0.0091 ± 0.0036 at 0.5** | +0.0060 ± 0.0045 at 0.5 |
| `POST_ICE_LAM` (0 to 2) | 7 | 0.053 | −0.053 / +0.002 | −0.033 / +0.008 | −0.010 / +0.020 | −0.012 / +0.005 | 5 | +0.0004 ± 0.0028 at 0 | +0.0016 ± 0.0016 at 0.75 |
| `POST_ICE_LAM` = `POST_GL_LAM` (0 to 0.75) | 4 | 0.052 | −0.009 / +0.007 | 0 / +0.026 | −0.052 / +0.008 | −0.044 / 0 | 4 | +0.0020 ± 0.0033 at 0.25 | none above 0 |
| `ICE_UNION_K` (0 to 30) | 4 | 0.051 | 0 / +0.001 | −0.003 / 0 | −0.051 / 0 | 0 / +0.002 | 4 | none above 0 | none above 0 |
| `FUSE_INJ_CREDIT` (0.25 to 1) | 3 | 0.038 | −0.003 / 0 | −0.038 / 0 | 0 / +0.008 | 0 / +0.029 | 0 | none | none |
| `LIB_TAU` (0.5 to 0.85) | 5 | 0.038 | 0 / +0.008 | −0.005 / 0 | −0.038 / 0 | −0.034 / 0 | 5 | none above 0 | none above 0 |
| `POP_MU` (0 to 1) | 7 | 0.035 | −0.017 / +0.001 | −0.035 / 0 | −0.001 / +0.012 | 0 / +0.010 | 6 | none above 0 | +0.0004 ± 0.0021 at 0.5 |
| `GL_LAM` (0 to 2) | 6 | 0.030 | −0.010 / +0.005 | −0.030 / +0.010 | −0.014 / +0.001 | −0.020 / +0.004 | 4 (≤ 0.75) | +0.0016 ± 0.0016 at 0.75 | +0.0025 ± 0.0023 at 0.75 |
| `ICE_LAM` (0 to 2) | 6 | 0.026 | −0.010 / +0.005 | −0.026 / +0.007 | −0.011 / +0.013 | −0.009 / 0 | 4 (≤ 0.75) | +0.0013 ± 0.0028 at 0.5 | +0.0018 ± 0.0017 at 0.75 |
| `LIB_TAU` with `GATE_TAU` 0.9 | 5 | 0.026 | 0 / +0.003 | −0.001 / 0 | −0.011 / 0 | −0.026 / 0 | 5 | none above 0 | none above 0 |
| `GATE_TAU` (0.5 to 0.95) | 6 | 0.025 | 0 / +0.008 | −0.004 / +0.001 | −0.018 / +0.003 | −0.025 / +0.001 | 6 | +0.0005 ± 0.0003 at 0.95 | +0.0013 ± 0.0005 at 0.95 |
| `SLOTS_GENTLE` (7 slot sets) | 7 | 0.023 | 0 / +0.007 | −0.010 / 0 | −0.009 / +0.007 | −0.023 / +0.002 | 6 | none above 0 | +0.0019 ± 0.0015 at [3, 6, 9, 12, 15] |
| `FUSE_ENG_K` (5 to 30) | 4 | 0.013 | 0 / +0.004 | −0.013 / 0 | 0 / +0.003 | 0 / +0.008 | 2 | +0.0003 at 30 | +0.0013 ± 0.0017 at 20 |
| `REL_TH` (0 to never) | 7 | 0.012 | 0 / +0.004 | −0.012 / 0 | 0 / +0.003 | 0 | 6 | +0.0001 | +0.0001 |
| `TOPN` (25 to 50) | 4 | 0.009 | −0.004 / +0.008 | −0.007 / 0 | 0 / +0.009 | 0 / +0.008 | 3 | +0.0015 ± 0.0019 at 30 | +0.0043 ± 0.0022 at 30 |
| `PC_FWD_T1` (−0.2 to 0.2) | 7 | 0.006 | 0 / +0.004 | −0.003 / 0 | 0 / +0.006 | 0 / +0.004 | 5 | +0.0011 ± 0.0007 at −0.05 | +0.0031 ± 0.0014 at −0.05 |
| `FUSE_N` (25 to 100) | 5 | 0.006 | 0 / +0.006 | −0.004 / 0 | 0 / +0.006 | −0.003 / +0.003 | 4 | none above 0 | +0.0022 ± 0.0019 at 25 |
| `GL_MH_ONLY` | 1 | 0.005 | +0.001 | +0.001 | +0.005 | +0.004 | 0 (26 `c2` lists) | none | none |
| `POP_LIB_OFF`, `FRAG_LIB_OFF`, `PC_FWD_N`, `SLOTS_AGG`, `FILL_25` | 13 | ≤ 0.004 | | | | | all | ≤ +0.0003 | ≤ +0.0011 |

Readings.

- **The knobs with the largest effects have no admissible value that helps.** `FUSE_MIN_LIB`, `FUSE_INJ_CREDIT` and
  `FUSE_ALPHA` below 0.6 raise the exposure at every value; `PC_JOIN_N`, `PC_JOIN_MAX_LIB`, `LIB_TAU`, `ICE_UNION_K`,
  `FUSE_SKIP_LIB` lose `dA` or `dB` monotonically as they move away from v13a; `ICE_LAM_LIB` / `GL_LAM_LIB` above 0
  cost `c1` up to 0.075 (v13a's 0 is the edge of the range and the best value); `FRAG_LAM` above 0.5 and `FRAG_MODE`
  all above 0.25 lose on every type; `POP_MU` is flat between 0.15 and 0.5 and loses outside. These agree in sign with
  the leaderboard facts for the same knobs (popularity 0.15 → 0.25, FRAG all-mode, engine-2 gate).
- **v13a sits on a ridge for the fusion weights**: `FUSE_KRR` 2 gains `c1` / `dA` / `dB` 0.005 to 0.006 and is
  inadmissible by one `c2` list (26); `FUSE_KRR` 5 loses `c2` 0.014; `FUSE_ALPHA` moves `dA` by about 0.015 per 0.1
  and `dB` by about 0.005 the other way.
- **The forward weights, moved together**, are the only single change with a mix gain above two standard errors:

| `ICE_LAM` = `GL_LAM` | `c1` | `c2` | `dA` | `dB` | LB-like mix | equal mix | `c2`: joined first / in first 25 / above the answer |
|---|---|---|---|---|---|---|---|
| 0 | +0.0088 ± 0.0069 (5 / 3) ne | +0.0178 ± 0.0117 (35 / 31) ne | −0.0520 ± 0.0144 (32 / 84) | −0.0534 ± 0.0133 (24 / 73) | −0.0040 ± 0.0059 | −0.0197 ± 0.0079 | 16 / 8.49 / 3.9% |
| 0.5 | +0.0094 ± 0.0059 (6 / 2) ne | **+0.0255 ± 0.0073 (30 / 10)** | −0.0038 ± 0.0080 (31 / 35) ne | −0.0069 ± 0.0083 (27 / 33) ne | +0.0091 ± 0.0036 | +0.0060 ± 0.0045 | 19 / 9.06 / 5.3% |
| 0.75 | +0.0048 ± 0.0036 (5 / 1) ne | **+0.0109 ± 0.0044 (22 / 5)** | +0.0014 ± 0.0045 (22 / 16) ne | +0.0011 ± 0.0063 (18 / 20) ne | +0.0049 ± 0.0022 | +0.0045 ± 0.0027 | 22 / 9.16 / 5.8% |
| 1 (v13a) | | | | | | | 25 / 9.24 / 7.9% |
| 1.5 | −0.0119 ± 0.0083 (1 / 8) ne | **−0.0245 ± 0.0059 (4 / 35)** | +0.0031 ± 0.0062 ne | −0.0007 ± 0.0047 ne | −0.0107 ± 0.0033 | −0.0085 ± 0.0042 | 33 / 9.35 / 10.8% |

  Each weight alone at 0.5 shows +0.007 (`ICE_LAM`) and +0.002 (`GL_LAM`) on `c2`: the two models have to come down
  together, which is why the one-at-a-time study of the fwd-replay review saw "no evidence".
- **Single knobs under the mixes**: 17 of the 29 knobs have no admissible value that gains more than one standard error
  under either mix; three have one above two (`FRAG_LAM` 0.25, `GATE_TAU` 0.95, `PC_FWD_T1` −0.05, all on the equal
  mix and all ≤ +0.0043), which is what 164 settings and two mixes produce without any effect.

## 4. Joint search (1,600 configurations)

`s4_joint.py`. **Random search over fixed grids with a fixed seed, not TPE**: an adaptive search would use all
molecules to propose points and contaminate the fold selection of section 5. Two searches instead of one, because the
replay is fast; both sets and the scan go into one candidate pool, and the null of section 5 is computed on that pool.

- **J1 (primary, 1,000 draws)**: `ICE_LAM`, `GL_LAM` in {0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.25}; `POST_ICE_LAM`,
  `POST_GL_LAM` in {None ×3, 0, 0.25, 0.5, 0.75, 1}; `FUSE_ALPHA` in {0.4 … 0.8}; `FUSE_KRR` in {1 … 5}. These are the
  six most sensitive knobs that have admissible values on both sides of v13a.
- **J2 (secondary, 600 draws)**: `ICE_LAM`, `GL_LAM` (same grid), `FRAG_LAM` {0, 0.25, 0.5, 0.75}, `POP_MU` {0.15, 0.25,
  0.35, 0.5}, `PC_JOIN_MAX_LIB` {0.5, 0.6, 0.65, 0.7}, `TOPN` {30, 40, 60}.
- **Deviation from "the six most sensitive"**: by raw effect size the top of the scan is `FUSE_MIN_LIB`, `PC_JOIN_N`,
  `FUSE_ALPHA`, `FRAG_MODE`, the `*_LAM_LIB` pair, `FRAG_LAM`. Left out of both searches: `FUSE_MIN_LIB` and
  `FUSE_INJ_CREDIT` (inadmissible at every value), `PC_JOIN_N` (incomplete forward scores below 100, none above),
  `ICE_LAM_LIB` / `GL_LAM_LIB` (monotone `c1` loss; leaderboard 0.000 for the switch), `FRAG_MODE` all (leaderboard
  0.390), `FUSE_SKIP_LIB` and `ICE_UNION_K` (monotone `dA` loss).
- Note on `POST_*_LAM`: a number switches the post-fusion re-rank on for library hits too (with `None` it follows
  `ICE_LAM_LIB` = 0), so every explicit value carries a small `c1` cost; this is the code's behaviour, not a replay
  artefact.

Admissible: 121 of 164 (scan), 764 of 1,000 (J1), 567 of 600 (J2).

## 5. Fold selection, held-out differences, null

`s5_cv.py`. 5 folds over the 380 molecules stratified by (class-1 molecule, PubChem record), 3 seeds, 15 folds. In each:
the admissible configuration with the largest mean difference to v13a on four folds (v13a itself is a candidate; ties
go to fewer changed knobs) is scored on the fifth. "fold SE" = standard deviation of the 15 held-out values / √15, the
error the rule names. "molecule SE" = each molecule's held-out value averaged over the three seeds, one unit per
molecule; this is the honest one, because the three repeats hold out the same 380 molecules.

**Null** (the equivalent permutation check; shuffling type labels is not defined here because a molecule carries all
its types): for each of 1,000 draws one random sign per molecule is applied to that molecule's differences in every
configuration and type, i.e. the labels "candidate" and "v13a" are swapped for the molecule. Correlations between
configurations stay; every true difference becomes zero. The whole procedure is repeated on each draw.

| Objective | Selected most often (of 15 folds; distinct selections) | In sample, best admissible | Held-out mean ± fold SE / molecule SE | `c1` held out | `c2` | `dA` | `dB` | Null: best in sample, mean (95%); p of observed | Null: held-out sd; p of observed | Null: "above 2 SE" fires, fold SE / molecule SE |
|---|---|---|---|---|---|---|---|---|---|---|
| LB-like | `ICE_LAM` 0.4, `GL_LAM` 0.5, `POST_ICE_LAM` 0.25, `POST_GL_LAM` 0, `FUSE_ALPHA` 0.4 (12; 3) | +0.0129 ± 0.0047 | **+0.0103 ± 0.0025 / 0.0045** | +0.0097 ± 0.0036 / 0.0071 | +0.0300 ± 0.0062 / 0.0095 | **−0.0616 ± 0.0062 / 0.0117** | **−0.0090 ± 0.0039** / 0.0100 | +0.0089 (+0.0144); 0.10 | 0.0051; 0.04 | 18.5% / 6.9% |
| equal | `ICE_LAM` 0.6, `GL_LAM` 0.25, `POST_GL_LAM` 0.75, `FUSE_ALPHA` 0.8, `FUSE_KRR` 2 (10; 4) | +0.0150 ± 0.0047 | **+0.0096 ± 0.0028 / 0.0045** | +0.0159 ± 0.0039 / 0.0080 | +0.0069 ± 0.0032 / 0.0079 | +0.0288 ± 0.0067 / 0.0076 | **−0.0133 ± 0.0045 / 0.0076** | +0.0116 (+0.0184); 0.18 | 0.0068; 0.10 | 21.0% / 7.3% |
| C12 | `ICE_LAM` 0.25, `GL_LAM` 0.5, `FRAG_LAM` 0, `POP_MU` 0.5, `PC_JOIN_MAX_LIB` 0.65 (8; 5) | +0.0301 ± 0.0082 | +0.0231 ± 0.0037 / 0.0073 | +0.0229 ± 0.0036 / 0.0093 | +0.0234 ± 0.0057 / 0.0101 | **−0.0186 ± 0.0091 / 0.0100** | **−0.0359 ± 0.0094 / 0.0113** | +0.0153 (+0.0242); 0.008 | 0.0091; 0.03 | 19.7% / 6.7% |
| CAL | `ICE_LAM` 0.4, `GL_LAM` 0.5, `POST_ICE_LAM` 0.5, `POST_GL_LAM` 0.5, `FUSE_KRR` 2 (8; 7) | +0.0039 ± 0.0012 | +0.0023 ± 0.0008 / 0.0012 | +0.0189 ± 0.0043 / 0.0081 | +0.0206 ± 0.0050 / 0.0081 | −0.0010 ± 0.0069 / 0.0079 | **−0.0171 ± 0.0035 / 0.0093** | +0.0025 (+0.0041); 0.08 | 0.0014; 0.07 | 17.4% / 4.9% |

Bold in the type columns = the type loses more than one standard error (the rule's third condition fails).

- **Rule check** (held-out objective above two standard errors AND admissible AND no type loses more than one): the
  objective clears two fold standard errors under all four mixes (2.9 to 6.2) and two molecule standard errors under
  three (CAL 1.9); every selection is admissible by construction; **the type condition fails in all four**, with either
  standard error. No configuration counts as an improvement.
- **Optimism**: the selected configuration scores +0.0130 on its training folds and +0.0103 on the held-out fold
  (LB-like); +0.0152 and +0.0096 (equal).
- **Stability**: the exact configuration recurs in 8 to 12 of 15 folds. Knob by knob: `GL_LAM` 0.5 in 14 of 15
  (LB-like), `ICE_LAM` 0.4 in 12; `FUSE_ALPHA` goes to 0.4 under the LB-like mix (which ignores `dA`) and to 0.8 under
  the equal mix: the fusion weight is chosen by the mix, not by the data. `FRAG_LAM`, `POP_MU`, `PC_JOIN_MAX_LIB`,
  `TOPN` stay at v13a's value in 10 to 15 folds except under C12, which has no answer-removed term and takes
  everything that moves MRR to `c1` / `c2`.
- **Size of a chance finding**: +0.009 (LB-like) to +0.012 (equal) in sample for the best of 1,448 admissible
  configurations when nothing is better than v13a; the search's observed bests are 1.3 to 1.5 times that. Only C12
  stands clear of its null in sample (p 0.008), and C12 is the objective that pays for it with `dB` −0.036.

**Deviation, reported separately: selection restricted to non-trades.** `--pareto`: in each fold a configuration is
eligible only if no type is more than one paired standard error below v13a on the four training folds (the null is
restricted the same way).

| Objective | Selected most often | Held-out mean ± fold SE / molecule SE | `c1` | `c2` | `dA` | `dB` | p of observed, in sample / held out |
|---|---|---|---|---|---|---|---|
| LB-like | `ICE_LAM` 0.4, `GL_LAM` 0.5, `POST_ICE_LAM` 0.5, `POST_GL_LAM` 0.5, `FUSE_KRR` 2 (9 of 15; 6) | +0.0084 ± 0.0022 / 0.0038 | +0.0146 ± 0.0039 / 0.0077 | +0.0238 ± 0.0043 / 0.0077 | +0.0018 ± 0.0053 / 0.0079 | **−0.0108 ± 0.0040 / 0.0085** | 0.18 / 0.07 |
| equal | `ICE_LAM` 0.6, `GL_LAM` 0.25, `POST_GL_LAM` 0.75, `FUSE_ALPHA` 0.8, `FUSE_KRR` 2 (5; 6) | +0.0082 ± 0.0023 / 0.0043 | +0.0124 ± 0.0034 / 0.0083 | +0.0087 ± 0.0037 / 0.0074 | +0.0268 ± 0.0071 / 0.0070 | **−0.0152 ± 0.0040 / 0.0077** | 0.29 / 0.12 |
| C12 | `ICE_LAM` 0.4, `GL_LAM` 0.5, `FRAG_LAM` 0.25, `POP_MU` 0.5 (10; 6) | +0.0200 ± 0.0032 / 0.0065 | +0.0175 ± 0.0038 / 0.0088 | +0.0225 ± 0.0037 / 0.0086 | +0.0005 ± 0.0063 / 0.0085 | **−0.0106 ± 0.0041 / 0.0090** | 0.05 / 0.03 |
| CAL | `ICE_LAM` 0.4, `GL_LAM` 0.5, `POST_ICE_LAM` 0.5, `POST_GL_LAM` 0.5, `FUSE_KRR` 2 (13; 3) | +0.0035 ± 0.0006 / 0.0011 | +0.0133 ± 0.0042 / 0.0078 | +0.0269 ± 0.0041 / 0.0078 | +0.0069 ± 0.0044 / 0.0079 | **−0.0066 ± 0.0034** / 0.0088 | 0.08 / 0.02 |

Still no pass under the rule as set: `dB` is more than one fold standard error down in all four. One of the sixteen
combinations of objective, selection mode and standard error does pass (CAL, non-trade selection, molecule SE:
+0.0035 ± 0.0011, `dB` −0.0066 ± 0.0088). It uses two choices made after seeing the first result and an objective
whose source calls its weights not identified; it is not counted.

## 6. Top admissible configurations

`s6_top.py`; full-sample paired differences. "Test folds" = the mean of the configuration's own 15 test-fold values
(no selection, so it equals the full-sample mean; its fold SE is given to show how much smaller than the paired SE it
is) and in how many folds the selection picked it.

**By the LB-like mix:**

| # | Changes to v13a | `c1` | `c2` | `dA` | `dB` | LB-like | equal | CAL | C12 | `c2`: joined first / in first 25 / above the answer | `c1`: joined first / in first 25 | test folds; selected in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `ICE_LAM` 0.4, `GL_LAM` 0.5, `POST_ICE_LAM` 0.25, `POST_GL_LAM` 0, `FUSE_ALPHA` 0.4 | +0.0101 ± 0.0073 (6 / 7) ne | +0.0367 ± 0.0101 (51 / 29) | **−0.0680 ± 0.0127 (40 / 114)** | −0.0084 ± 0.0103 (58 / 44) ne | +0.0129 ± 0.0047 | −0.0074 ± 0.0064 | +0.0017 ± 0.0015 | +0.0234 ± 0.0064 | 20 / 9.79 / 5.3% | 6 / 1.91 | +0.0129 ± 0.0021; 12 |
| 2 | `ICE_LAM` 0.25, `GL_LAM` 0.5, `FUSE_ALPHA` 0.5, `FUSE_KRR` 2 | +0.0146 ± 0.0080 (7 / 6) ne | +0.0296 ± 0.0087 (39 / 24) | **−0.0274 ± 0.0095 (36 / 79)** | −0.0060 ± 0.0094 (46 / 38) ne | +0.0115 ± 0.0043 | +0.0027 ± 0.0055 | +0.0025 ± 0.0013 | +0.0221 ± 0.0061 | 24 / 9.28 / 5.8% | 6 / 1.83 | +0.0115 ± 0.0024; 2 |
| 3 | `ICE_LAM` 0.4, `GL_LAM` 0.5, `POST_GL_LAM` 0.25, `FUSE_ALPHA` 0.4, `FUSE_KRR` 4 | +0.0097 ± 0.0068 (7 / 5) ne | +0.0299 ± 0.0088 (44 / 27) | **−0.0427 ± 0.0112 (37 / 97)** | −0.0032 ± 0.0095 (57 / 37) ne | +0.0115 ± 0.0043 | −0.0016 ± 0.0056 | +0.0019 ± 0.0013 | +0.0198 ± 0.0057 | 21 / 9.97 / 5.8% | 6 / 1.95 | +0.0115 ± 0.0018; 0 |
| 4 | `ICE_LAM` 0.4, `GL_LAM` 0.5, `POST_ICE_LAM` 0.5, `POST_GL_LAM` 0.5, `FUSE_KRR` 2 | +0.0133 ± 0.0074 (8 / 4) ne | +0.0292 ± 0.0082 (45 / 17) | +0.0077 ± 0.0082 (70 / 34) ne | −0.0053 ± 0.0089 (38 / 36) ne | +0.0114 ± 0.0040 | +0.0113 ± 0.0049 | +0.0039 ± 0.0012 | +0.0213 ± 0.0056 | 20 / 9.03 / 5.0% | 6 / 1.79 | +0.0114 ± 0.0019; 0 (9 with non-trade selection) |
| 5 | `ICE_LAM` 0.4, `GL_LAM` 0.5, `FRAG_LAM` 0.25, `POP_MU` 0.5 | **+0.0234 ± 0.0098 (13 / 3)** | +0.0239 ± 0.0088 (47 / 22) | +0.0020 ± 0.0089 (61 / 50) ne | −0.0065 ± 0.0093 (42 / 45) ne | +0.0107 ± 0.0044 | +0.0107 ± 0.0054 | +0.0030 ± 0.0012 | +0.0236 ± 0.0071 | 18 / 9.06 / 4.5% | 6 / 1.77 | +0.0107 ± 0.0018; 0 |

Under every mix the best of all 1,764 configurations is itself admissible: the exposure constraint does not bind at
the top, because every top configuration lowers the forward weights, and that lowers the exposure.
Rows 1 and 3 put more joined rows into the first 25 than v13a (9.79, 9.97 against 9.24; `FUSE_ALPHA` 0.4) although
fewer are first; under a stricter exposure reading they would be out. Top of the other mixes: equal,
`ICE_LAM` 0.6, `GL_LAM` 0.25, `POST_GL_LAM` 0.75, `FUSE_ALPHA` 0.8, `FUSE_KRR` 2: +0.0150 ± 0.0047 (`c1` +0.019, `c2`
+0.010 ne, `dA` +0.040, `dB` −0.010 ne; 20 / 5 lists); CAL, row 4 above; C12, `ICE_LAM` 0.25, `GL_LAM` 0.5, `FRAG_LAM`
0, `POP_MU` 0.5, `PC_JOIN_MAX_LIB` 0.65: +0.0301 ± 0.0082 (`dB` −0.023 ± 0.012; 16 / 6 lists). All tables:
`D:\casmi-data\search\top_all.md`.

**The nearest miss, for the record.** The simplest member of the family is the scan point `ICE_LAM` = `GL_LAM` = 0.5
(two knobs, section 3): `c2` +0.0255 ± 0.0073 (30 / 10), `c1` +0.0094 ± 0.0059, `dA` −0.0038 ± 0.0080, `dB` −0.0069 ±
0.0083, joined row first on 19 `c2` lists instead of 25, joined row above the answer 5.3% instead of 7.9%. It is the
selection of 15 of 15 folds when only the 164 scan settings are candidates (LB-like, held out +0.0091 ± 0.0036 per
molecule; null p 0.18 in sample, 0.06 held out) and it fails the rule on `dB` like the others (−0.0069 ± 0.0032 fold
SE). What a submission of it would test: whether the hidden test pays for `c2` / `c1` protection the way the exposure
history suggests. What it would not be: a configuration this search supports as better than v13a. It would also pass
the weaker test of `reviews/calib-2026-10-05.md` (exposure not above v13a, `dA` and `dB` not more than one paired
standard error below), with a `c2` gain six times that of the `JOIN_FWD_A` 0.25 preset there; that test, as that review
says, has never been checked against a submission that won.

## 7. Verdict

- **No preset.** `configs/v23a-v13a-search.json` was not written (the name `v23a` is in use by the calib study's
  `v23a-v13a-joinfwd-a025.json`). Defaults in `00_config.py` untouched.
- **Is anything worth a submission? No.** The search finds no configuration that is better than v13a beyond chance
  without taking the gain from `dB`. v13a is not a local optimum of any mix; it is on the boundary of the set where no
  type loses.
- **What the search does establish**: (1) 17 of 29 knobs have no admissible value that helps under either mix beyond
  one standard error; (2) the forward weights are the only place with a consistent signal, and its sign is "lower";
  (3) a local gain of about 0.01 on a mix is what 1,448 admissible configurations produce by chance on 380 molecules.

## 8. Honesty and limits

- **The forward models have very likely seen these molecules** (`reviews/mixed-regime-2026-10-02.md`). That inflates
  what the forward terms give `dA` / `dB` and understates what they cost `c2` / `c1`, so the leak pushes against the
  direction found here, not for it. It does not make the direction a leaderboard prediction.
- **The mixes are assumptions.** The LB-like mix has `dA` weight 0 and hides a 0.07 `dA` loss in its top row; the
  lbcal fit puts the leaderboard weight of `dB` at a fifteenth of its local size, under which rows 2, 4 and 5 of
  section 6 gain two to three standard errors. The search cannot choose between these readings; the type rule is what keeps it honest.
- **Exposure is measured as "joined row first"** on 380 / 150 lists; one list is 0.26% / 0.67%. `GL_MH_ONLY` is
  inadmissible here by one list (26) and scored 0.431 on the leaderboard: the constraint has no slack and no error bar.
- **The fold standard error named in the rule is anticonservative** (false-pass rate 17 to 21% under the null). The
  verdict does not depend on it, but any later use of "two standard errors across repeated folds" should use the
  per-molecule error.
- **The null keeps the observed spread of each configuration**, including configurations that truly differ from v13a
  (trades with large per-molecule variance); it answers "how large a best-of-1,448 arises from configurations this
  variable if none is better", which is the question asked.
- **Knob choice for the joint search used the scan on all molecules.** The values were then drawn at random, and the
  null covers the pooled candidate set, but the choice of dimensions is not inside the folds.
- **Local stand-ins unchanged from the fwd-replay**: fold-safe FPNet in the v1 engine and the PubChem list; local
  forward scores agree with a Kaggle dump at r 0.997 to 0.998, not bit-exact; 74 molecules without forward coverage.
- **Not done**: `PC_JOIN_CONF`; any configuration needing new forward scores (section 1); TPE (by design).
- **Failed or stopped**: one shell command that chained the verification and both selection runs exceeded the 10-minute
  foreground limit and was moved to the background; it completed. The first background job reported exit code 1 from
  its last `tail` although all six workers had finished and written their results. A first version of two scripts
  failed to parse in the shell heredoc and was rewritten as files. Nothing else failed.

## 9. Reproduce

```
export CASMI_DATA='D:\casmi-data'; P=.venv/Scripts/python; S=scratch/search
$P $S/s0_probe.py                          # load, timing, v13a check, forward and FRAG coverage per depth and gate
$P $S/s1_frag.py                           # 7,930 missing MetFrag scores -> D:/casmi-data/search/frag.json (6 s)
$P $S/s2_scan.py && $P $S/s3_scan_report.py   # 164 settings (4 min) -> scan.md, scan_rank.json
for w in 0 1 2; do $P $S/s4_joint.py J1 $w 3 & done; wait   # 1,000 draws, 6 min; resumable (stored keys are skipped)
for w in 0 1 2; do $P $S/s4_joint.py J2 $w 3 & done; wait   # 600 draws, 4 min
$P $S/s5_cv.py --B 1000                    # folds + null -> cv_all.json (5 min)
$P $S/s5_cv.py --B 1000 --pareto --tag pareto   # non-trade selection (15 min)
$P $S/s6_top.py                            # top_all.md
$P $S/s7_check.py                          # 13 stored configurations against the current fusion_core.py
```

Results are appended per evaluation to `D:\casmi-data\search\evals\<store>_w<i>.pkl` (per-molecule reciprocal ranks
and exposure flags for every configuration). The seven `tests/test_*.py` of the solution pass with the venv python
(run although no preset was added). This file is new: run `python tools/gen_index.py` when it is committed.
