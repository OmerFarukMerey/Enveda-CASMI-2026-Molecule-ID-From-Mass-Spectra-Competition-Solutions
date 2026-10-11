<!-- index: fusion-v4n 2026-10-07 audit of every learned or hand-calibrated stage other than the v1 ranker for the seen / unseen regime mismatch behind v33b: inventory with evidence (engine 2's two rankers, its FPNets, blend and analog weights, PubChem selection and REL_TH, join and library gates, stage-A weights, popularity, RRF weights, confidence gate), the finding that both public row files of engine 2 contain the 250 Enveda panel molecules (panel replays are in-sample for engine 2: +0.06 on its list, +0.03 on c2 after the fusion), engine 2's main ranker re-fitted on megayak's leak-free rows (worse with the public FPNets: not wired), the fusion weights re-measured with cross-fitted engine-2 rankers (FUSE_KRR 1: preset v35a, test), REL_TH never fires in any local replay -->
# Regime audit of the other stages: who was fitted on what, and two stages rebuilt (2026-10-07)

Offline only. Nothing was submitted, pushed, committed or uploaded; no `kaggle` command was run; nothing was downloaded; no
network was trained; the visible `test.parquet` was not read; no NIST data. GPU: inference only (GLACIER / ICEBERG runner, one at
a time, cap 2.8 GB, started when the other processes held 1.3 GB); while a game held the GPU 74 of 154 mixed-regime molecules were
scored by GLACIER with the same runner on the CPU. CPU: one to three processes of mine at below-normal priority. The machine was
shut down once during the study; every job was resumed from its cache, none was run twice. Scripts: `scratch/regime/`
(git-ignored). Outputs: `D:\casmi-data\regime\` (new; every other cache was read, not written). Changed files of the solution:
new `configs/v35a-v33b-krr1.json`, new `tests/test_fuse_krr.py`, this review. No code of the notebook was changed. `INDEX.md`
and `EXPERIMENTS.md` not edited (other agents are editing the tree).

**Every number below was computed in this study** unless a file is named. MRR = MRR@25 on the metric key; "±" = one standard
error of a paired per-molecule difference; (wins / losses; p) with p from a sign-flip null (20,000 draws, two-sided). **No
leaderboard estimate is given.** "Seen" = the molecule's structure was in the model's training data; "in-sample" = the
molecule's own rows were in a ranker's training rows.

## 0. Bottom line

1. **After v33b no other stage has the defect the public v1 ranker had.** Engine 2's main ranker, the one stage that reads
   fingerprint margins the way the v1 ranker does, was fitted on rows whose fingerprint features look like those of unseen
   molecules (answer's z-score of `f.z` 1.64, answer first by `f.z` 42%; on molecules its FPNets have seen 2.53 and 77%, on
   the Enveda panel, which they have not seen, 1.54 and 32%). Everything else is hand-set, chosen on the leaderboard, or
   fitted without fingerprint inputs (section 1).
2. **New, and it changes how the panel must be read: both row files of engine 2's rankers contain the 250 panel molecules.**
   226 of 245 panel answers are found in `prvsiyan/casmi26-ranker-features/rank_train.npz` and 244 of 245 in megayak's rows
   (0 to 2 by chance for the other query sets; the 380 mixed-regime molecules: at chance). The rankers the notebook fits are
   therefore in-sample on the panel. Cross-fitted (each molecule scored by rankers fitted without its rows) engine 2's own
   `c2` list drops from 0.699 to 0.641 (−0.059 ± 0.010, 7 / 70), `c1` from 0.911 to 0.858 (−0.053 ± 0.010); after the fusion
   v33b on the panel loses `c2` −0.032 ± 0.008 (1 / 24), `c1` −0.015 ± 0.005, `dA` −0.015 ± 0.005. **No local set tests
   engine 2 in the hidden test's regime**: on the panel its rankers are in-sample, on the mixed regime its FPNets have seen
   the molecules (42%) or its fingerprint features are NaN (58%). The cross-fitted panel built here is the first that does.
3. **Stage rebuilt 1, engine 2's main ranker on megayak's leak-free rows: worse, not wired.** With the public FPNets, which
   is what the notebook runs: cross-fitted panel `c2` list −0.051 ± 0.014 (22 / 72) against the cross-fitted public ranker,
   mixed regime −0.046 ± 0.009 (28 / 97). On megayak's own rows it wins (+0.058 ± 0.005): each ranker is better on the
   features of the FPNets its rows were made with. megayak's rows fit his two leak-free FPNets, not the public pair.
   Reading: **worse locally, do not ship.** The rank average of the two rankers is level everywhere (not better: not wired).
4. **Stage rebuilt 2, the weights of the rank fusion with engine 2 (`FUSE_ALPHA` 0.6, `FUSE_KRR` 3, hand-set by the public
   notebook's author before the PubChem join existed), re-measured with the cross-fitted engine 2.** The leak hides the
   direction on pool answers: `FUSE_ALPHA` 0.4 is `c2` −0.003 ± 0.005 with the in-sample engine 2 and +0.016 ± 0.007
   (p 0.02) with the cross-fitted one. The variant without a cost on any type in any set is **`FUSE_KRR` 1**: cross-fitted
   panel `c1` +0.006 ± 0.005, `c2` +0.013 ± 0.007, `dA` −0.006 ± 0.007, **`dB` +0.055 ± 0.010 (49 / 11)**; mixed regime
   +0.004, +0.001, +0.007, **+0.018 ± 0.007 (29 / 17)**; exposure 23 against 21 per 380. Reading: **better than v33b
   locally on answers that only the PubChem join can reach (ordinary evidence, three sets), equal on pool answers.**
   Preset `configs/v35a-v33b-krr1.json`, test `tests/test_fuse_krr.py`; no code change, no new dataset.
5. **`REL_TH` 600 has never fired in a local replay.** It is in the public models' `f.z` units; the fold-safe stand-in's scale
   is about twenty times smaller (largest tenth of `rel` above 27 to 48, 0% above 600). With the public A+B selection on the
   panel the aggressive slots go to 1.6% of the lists when the answer is a pool row the models know and to 9.8% when it is
   removed. Its role is bounded (section 2.3); nothing was built.
6. **Side findings.** (a) For molecules without collision energy engine 2's blend is worse with the NaN fingerprint
   features (0.570) than with those features zeroed (0.636; 204 mixed-regime `c2` lists, the stored `nofp` variant); the hidden test presumably has no
   such spectra. (b) `ENG_W_PV` is flat between 0.5 and 1.0 on leak-free rows (0.646 to 0.639).

## 1. Inventory

Status: **seen** = fitted on outputs of models that had memorised the fitting molecules; **unseen** = fitted on outputs for
molecules those models had not seen; **LB** = chosen by leaderboard submissions (the hidden test's own regime); **hand** = a
constant somebody set, data unknown; **none** = no fitted quantity reads a fingerprint model.

| Stage | Fitted on what, with which upstream models | Status | Evidence | Shift measured |
|---|---|---|---|---|
| v1 public ranker (`ranker_v4b.pkl`), at `lib_max` ≥ 0.7 and at half weight below | simulations of train molecules with the public models' features | **seen** | `reviews/fsranker-2026-10-06.md`, `rankbank-2026-10-06.md` | cited: answer's `fz_softmax` z 6.6 seen, 3.3 unseen. Still acts on the class-2 lists above the gate (106 of 380 regime, 86 of 250 panel lists) |
| v1 ranker of v33b (`casmi26-ours-ranker-fs`) | fold-safe FPNet features of 6,687 hold-out lists | unseen | `reviews/fsranker-2026-10-06.md` section 4 | its `j__pcrank` input was the fold-safe selection's rank; the notebook feeds the public A+B selection's (not measured) |
| Engine 2, main ranker (weight 0.88; `rank_train.npz`, 142,762 rows × 31, 819 molecules × 2 simulations; fitted in the notebook) | prvsiyan's rows; dataset card empty (`D:\casmi-data\ranker_study\ds\small\prv\dataset-metadata.json`); 12 of 31 inputs are fingerprint margins | **unseen by measurement; provenance unknown** | `scratch/regime/rg_rows0.py`, `rg_recs0.py` | answer's z(`f.z`) / answer first by `f.z`: its training rows 1.64 / 42%; megayak's leak-free rows 1.50 to 1.84 / 32 to 52%; panel records 1.54 / 32%; regime records with finite logits 2.53 / 77% |
| … the same ranker, as a memory of molecules | 226 of 245 panel answers are among its training molecules | in-sample on the panel | `rg_cf.py`, `rg_pvcv.py` | on its own rows in-sample 0.839, grouped CV 0.631 (+0.209 ± 0.020, 96 / 0; other molecules +0.207 ± 0.012) |
| Engine 2, second ranker (weight 0.12; megayak `sim_rank_rows_nofp.npz`, 2,250 molecules × 2) | no fingerprint features | none; in-sample on the panel (244 of 245) | megayak's README (`ranker_study\ds\megayak\README.md`), `rg_cf.py` | cross-fitted together with the main ranker (section 2.2) |
| Engine 2 FPNets (`fp_single_s2`, `fp_merged_m1`) | card empty; claimed to hold out `enveda-np-examples` | not a fitted stage; **have not seen the panel, both** | `f.z` alone on the panel 0.487 (megayak's leak-free pair 0.478 by his README), regime 0.845; `np-ours` review measured `s2` alone | the comment in `scratch/panel2/p2_eng2.py` ("saw the panel") is wrong for the FPNets and right for the rankers |
| `ENG_W_PV` 0.88, `ENG_W_A` 0.35 / 0.55, seeds | constants of the public notebook | hand | `nb/embed/eng/eng_runner.py` | leak-free rows, grouped CV, class-2: 0.646 / 0.644 / 0.643 / 0.639 at 0.5 / 0.7 / 0.88 / 1.0 (`cv_report.json`): flat |
| Engine 2 analog propagation (power 3, 200 analogs) | constants in `nb/embed/eng/pv.py` | hand; no fingerprint input | code | none (library and structure only) |
| PubChem selection (public A+B `f.z` over the mass window, `PC_N1` 5000) | no fitted quantity | none | `nb/embed/pc/` | — |
| `REL_TH` 600, slots 2.. / 4.. | constants of the reference notebook (`reference/v4n/notebook_code.py`: `LIB_TAU, REL_TH = 0.9, 600.0`), in the public models' `f.z` units | hand, tuning data unknown | `rg_pc.py` | section 2.3: never fires locally; 1.6% → 9.8% between a known and a removed pool answer |
| Join gate `PC_JOIN_MAX_LIB` 0.7, `PC_JOIN_N` 100 | leaderboard (v12a, v13a, v14a, v14b, v19a) | LB | `docs/experiments.md` rows 16 to 25 | `lib_max` is a spectral similarity, no model in it |
| `LIB_TAU` 0.9 | reference notebook | hand; no model in it | code | not measured |
| `RANKER_LIB_GATE` 0.7 | set equal to the join gate; one submission | hand / one LB point | `fsranker` review | not re-opened here (it belongs to the v1 ranker) |
| Stage A weights `ICE_LAM` = `GL_LAM` 0.75, `FRAG_LAM` 0.5 `gap`, `*_LAM_LIB` 0 | leaderboard line search and v2 / v3b / v4a | LB | `docs/experiments.md` rows 5 to 8, 27 to 29 | the forward models themselves have very probably seen the local molecules (`panel2` review) |
| `POP_MU` 0.25 | leaderboard (v4b) | LB | row 8 | — |
| `FUSE_ALPHA` 0.6, `FUSE_KRR` 3 | reference notebook; never varied on the leaderboard by us (only `FUSE_MIN_LIB`, v7b 0.405); every local test had a leaked engine 2 | **hand, tuning data unknown** | `EXPERIMENTS.md` lines 22 and 378 | section 3 |
| Confidence gate (`conf_gate.py`, off in v33b) | hold-out lists with fold-safe features | unseen | `reviews/confgate-2026-10-04.md` | not measured (off) |
| megayak's pair-tail rows (`sim_rank_rows_fppair.npz`; not in our pipeline) | his two leak-free FPNets, 2,250 molecules | unseen, **but another model's features** | section 2.2 | a ranker fitted on them loses 0.05 when fed the public FPNets' features |

## 2. Engine 2

### 2.1 The row files contain the panel

A row file has no identifiers. The MetFrag column of the base block (column 21) depends on the query spectra and the
structure only, so the answer row's value identifies a molecule (`rg_rows0.py`, exact match after rounding to 6 decimals):

| Answer rows of | n (value > 0) | value found among `rank_train.npz` answer rows | (value, top analog similarity) pair found |
|---|---|---|---|
| megayak query set `np` (the 250 Enveda standards) | 245 | **227** | **206** |
| megayak `gnps` / `riken` / `mona` / `msdial` / `massbank` / `pluskal_ms2` | 244 to 375 | 12 to 33 | 0 to 2 |
| panel records of our pipeline (`panel2\eng2\c2\recs.pkl`) | 245 | **226** (megayak rows: **244**) | |
| mixed-regime records | 323 | 27 (megayak rows: 30; 25 to 27 per query set, i.e. chance) | |

`rank_train.npz` holds 819 molecules; what the other 593 are is unknown. Consequence: on the panel the main ranker scores
lists it was fitted on. `rg_pvcv.py`, the public recipe on its own rows, class-2 simulation: in-sample 0.839 (tagged NP
standards, n 219) against 0.631 in 5-fold CV grouped by molecule; class-1 simulation 0.948 against 0.805.

### 2.2 The main ranker re-fitted on megayak's leak-free rows (stage rebuilt 1)

Variant `u`: `casmi_engine.fit_pv_rankers`' model class and hyper-parameters (HistGradientBoosting, priors 0.30 / 0.60 on the
class-1 simulation rows) on the base block of `sim_rank_rows_fppair.npz` instead of `rank_train.npz`; `ua`: the two rankers'
within-list ranks averaged. The second ranker and the 0.88 / 0.12 blend unchanged.

**(i) Stage level, three comparisons.** Answer's MRR@25; "list" = the key-deduplicated list engine 2 hands to the fusion.

| Set (who has seen what) | n | public | `u` | `u` − public | `ua` − public |
|---|---|---|---|---|---|
| megayak's rows, grouped 5-fold CV, class-2 simulation, six library sets (features of megayak's leak-free FPNets; `u` out of fold; the public ranker has not these molecules) | 2,000 | 0.633 | 0.695 | **+0.062 ± 0.005 (560 / 227)** | +0.019 ± 0.002 |
| … set `np` (the public ranker is in-sample) | 250 | 0.691 | 0.712 | +0.021 ± 0.016 (54 / 43) | +0.014 ± 0.009 |
| … class-1 simulation, six sets / `np` | 2,000 / 250 | 0.869 / 0.889 | 0.888 / 0.830 | +0.019 ± 0.003 (195 / 91) / **−0.059 ± 0.013 (11 / 47)** | +0.008 ± 0.001 / −0.004 ± 0.008 |
| **panel records, `c2`, both rankers cross-fitted** (public FPNets: unseen; rankers: out of fold; `rg_cf.py`) | 226 | 0.641 | 0.590 | **−0.051 ± 0.014 (22 / 72)**; nitrogen-free −0.072 ± 0.021 (n 113) | −0.009 ± 0.008 (16 / 37) |
| panel records, `c1`, cross-fitted | 226 | 0.858 | 0.853 | −0.006 ± 0.012 (21 / 23) | +0.013 ± 0.006 (15 / 2) |
| mixed-regime records, `c2`, `u` fitted without the `np` rows (rankers out of sample; FPNets have **seen** the molecules) finite logits | 153 | 0.785 | 0.764 | −0.021 ± 0.014 (10 / 27) | |
| … NaN logits (no collision energy) | 204 | 0.571 | 0.505 | −0.066 ± 0.013 (18 / 70) | |
| … all | 357 | 0.662 | 0.616 | **−0.046 ± 0.009 (28 / 97)** | |

The sign follows the FPNets that made the features, not the regime: `u` wins on megayak's FPNets' features and loses on the
public pair's, in both of our sets, by the same 0.05. The row-level feature quantiles of megayak's `np` rows and of our panel
records are close in all 31 columns (first 120 panel records; e.g. `f.z` minus the list maximum, median −608 against −622), so
the difference is in the joint structure, not in a scale.

**(ii) Full replay against v33b** (`rg_replay.py`: `fusion_core.build_submission`, real ICEBERG / GLACIER scores for every
candidate `nb/07` would score, no coverage gap; v33b reproduced to four decimals: panel 0.9439 / 0.8658 / 0.6722 / 0.3971,
regime 0.8463 / 0.7634 / 0.5538 / 0.3607). Exposure = pool-answer lists with a joined PubChem row first / answer lost first.

| Set, reference | Variant | `c1` | `c2` | `dA` | `dB` | exposure `c2`, `c1` |
|---|---|---|---|---|---|---|
| mixed regime (150 / 380 / 325 / 325), v33b: 21 / 4, 5 / 0 | `u` | +0.005 ± 0.004 (4 / 2) | **−0.012 ± 0.005 (13 / 32; p 0.03)** | −0.002 ± 0.006 | +0.003 ± 0.005 | 22 / 4, 5 / 0 |
| | `ua` | +0.006 ± 0.006 | −0.004 ± 0.002 (9 / 16) | +0.001 ± 0.005 | −0.004 ± 0.002 | 22 / 4, 5 / 0 |
| panel, 226 cross-fitted molecules, v33b with the cross-fitted public engine 2 (0.9297 / 0.8332 / 0.6540 / 0.3990): 3 / 1, 2 / 2 | `u` cross-fitted | −0.002 ± 0.006 | +0.002 ± 0.008 (8 / 19) | −0.015 ± 0.008 (10 / 35; p 0.05) | −0.003 ± 0.006 | 2 / 0, 2 / 2 |
| | `ua` cross-fitted | −0.001 ± 0.003 | +0.001 ± 0.006 | +0.002 ± 0.004 | +0.001 ± 0.002 | 2 / 0, 2 / 2 |
| panel, all 250, v33b as replayed so far (engine 2 in-sample) | cross-fitted public engine 2 | **−0.011 ± 0.005 (1 / 9)** | **−0.030 ± 0.007 (1 / 26)** | **−0.012 ± 0.004 (7 / 26)** | +0.001 ± 0.003 | 4 / 2, 3 / 3 (unchanged) |

Reading: **`u` is worse locally** (stage level beyond three standard errors in the two sets that use the notebook's FPNets;
after the fusion `c2` −0.012 on the regime); `ua` is equal after the fusion and not better at stage level. Neither meets the
condition for wiring. The last row is the size of the panel's in-sample leak after the fusion; it applies to every panel
number of the earlier reviews (`panel2`, `fsranker`), to both sides of their differences.

What would be the coherent version: megayak's two leak-free FPNets (`ours_fpnet_single_16k.pt`, `ours_fpnet_merged_26k.pt`,
144 MB each, in a dataset the notebook already attaches, **not on this machine**) together with a ranker fitted on his rows,
i.e. the engine-2 analogue of v33b's fold-safe chain. Indication only, not the same rows: on the 250 standards that chain's
ranker reaches 0.712 out of fold on megayak's rows, the public chain 0.641 cross-fitted on ours. Not built: the download
(288 MB) is above the limit of this run.

### 2.3 `REL_TH` and the slot rule

`rg_pc.py`, panel, `rel` = top PubChem `f.z` − best pool `f.z` (10 / 25 / 50 / 75 / 90% points), share above 600:

| Selection model | answer a pool row (`c2`) | answer's pool row removed (`dB`) |
|---|---|---|
| fold-safe (the stand-in of every local replay; unseen) | −9 / −3 / 5 / 16 / 27; **0%** | −6 / 0 / 11 / 25 / 48; **0%** |
| public A+B (the notebook's; has seen the structures) | −319 / −160 / −25 / 108 / 266; 1.6% | −170 / −36 / 126 / 370 / 583; **9.8%** |

- The threshold is in the public models' units, so no local replay (mixed regime, panel `ctrl` arm, the v33b replays) has
  ever put a list on the aggressive slots; the earlier "no measurable local effect" (`reviews/audit-2026-10-02.md`) and the
  parameter search's `REL_TH` row were measured on the stand-in's scale.
- With the public models the pool answer's own memorised `f.z` holds `rel` down (median −25 against +126). On the hidden
  test a class-2 answer is not memorised, so the aggressive share lies between 1.6% and 9.8%, nearer the second.
- Role: at most a tenth of the lists below `LIB_TAU`, a PubChem row at place 2 instead of 4. Nothing was built: an
  evaluation needs the public-model PubChem lists for the regime, which do not exist locally, and the bound is small.

## 3. The fusion weights with an engine 2 in the right regime (stage rebuilt 2)

`fuse_rrf`: score = 1 / (`FUSE_KRR` + rank in the v1 list) + `FUSE_ALPHA` / (`FUSE_KRR` + rank in engine 2's list). Engine 2
cannot list a joined PubChem row, so every joined row competes with one term against two. The constants are the reference
notebook's, set before the join existed.

**The unseen-regime version of the comparison** is the panel with both engine-2 rankers cross-fitted (`eng:pubcf`: FPNets
unseen, rankers out of fold, v1 side below the gate fold-safe as in v33b). "Public version" = 0.6 / 3 on the same lists.

| Variant | set | `c1` | `c2` | `dA` | `dB` | exposure `c2`, `c1` (reference) |
|---|---|---|---|---|---|---|
| **`FUSE_KRR` 1** | panel, cross-fitted (226) | +0.006 ± 0.005 (4 / 2) | +0.013 ± 0.007 (13 / 13; p 0.08) | −0.006 ± 0.007 (18 / 16) | **+0.055 ± 0.010 (49 / 11; p 0.000)** | 4 / 2, 3 / 3 (3 / 1, 2 / 2) |
| | panel, engine 2 in-sample (250) | +0.004 ± 0.004 | +0.003 ± 0.005 (8 / 12) | −0.000 ± 0.006 | **+0.061 ± 0.010 (58 / 11)** | 4 / 2, 4 / 4 (4 / 2, 3 / 3) |
| | mixed regime | +0.004 ± 0.005 (3 / 4) | +0.001 ± 0.003 (9 / 22) | +0.007 ± 0.007 (62 / 26) | **+0.018 ± 0.007 (29 / 17; p 0.007)** | 23 / 5, 5 / 0 (21 / 4, 5 / 0) |
| `FUSE_KRR` 2 | cross-fitted / in-sample / regime | +0.002 / +0.003 / +0.001 | +0.011 ± 0.006 / −0.000 / −0.001 | +0.002 / −0.001 / +0.008 ± 0.004 | +0.030 ± 0.007 / +0.032 ± 0.007 / +0.006 ± 0.004 | 4 / 2, 2 / 2; 5 / 3, 3 / 3; 22 / 5, 5 / 0 |
| `FUSE_ALPHA` 0.4 | panel, cross-fitted | +0.003 ± 0.005 | **+0.016 ± 0.007 (16 / 11; p 0.02)** | **−0.020 ± 0.008 (7 / 47)** | +0.055 ± 0.010 (59 / 2) | 4 / 3, 2 / 2 |
| | panel, engine 2 in-sample | −0.005 ± 0.003 (0 / 5) | −0.003 ± 0.005 | −0.027 ± 0.007 | +0.055 ± 0.009 | 5 / 4, 3 / 2 |
| | mixed regime | −0.000 ± 0.001 | −0.000 ± 0.003 | **−0.021 ± 0.006 (10 / 73)** | +0.020 ± 0.005 (49 / 1) | **25 / 7**, 5 / 0 |
| `FUSE_ALPHA` 0.5 | cross-fitted / in-sample / regime | +0.001 / −0.001 / −0.001 | +0.010 ± 0.006 / −0.002 / −0.001 | −0.013 ± 0.005 / −0.014 / −0.014 ± 0.005 | +0.028 ± 0.007 / +0.029 / +0.011 ± 0.003 | 4 / 2; 5 / 4; 22 / 6 |
| `FUSE_KRR` 1 + `FUSE_ALPHA` 0.5 | cross-fitted / in-sample / regime | +0.013 ± 0.006 / +0.001 / +0.004 | +0.020 ± 0.008 / +0.005 / +0.004 | −0.014 ± 0.009 / −0.014 / −0.010 | +0.078 ± 0.012 / +0.083 / +0.032 ± 0.008 | 5 / 2, 4 / 3; 6 / 4, 5 / 4; 24 / 7, 5 / 0 |
| `FUSE_ALPHA` 0.8 | cross-fitted / in-sample / regime | −0.001 / +0.001 / −0.001 | −0.012 ± 0.006 / −0.013 ± 0.005 / −0.015 ± 0.006 | +0.018 / +0.019 / +0.027 | −0.045 ± 0.008 / −0.039 / −0.024 | 2 / 1; 3 / 2; 21 / 4 |
| `FUSE_KRR` 6 / 10, `FUSE_ALPHA` 0 / 0.2 / 0.3 / 1.0 | all sets | every one loses a type beyond two standard errors in at least one set (`report_*.json`) | | | | |

"Agreement rule" = `FUSE_ALPHA` chosen per molecule by whether engine 2's first candidate is among the first three of the v1
base list (nine pairs of weights, emulated by switching between two replays). The best of them (0.6 when the engines
disagree, 0.3 when they agree) is `c2` +0.017 ± 0.007 on the cross-fitted panel and `c1` −0.010 ± 0.006, `dA`
−0.011 ± 0.004 on the regime. Eight of the nine lose a type beyond two standard errors in at least one set; the ninth (0.8 when
they disagree, 0.3 when they agree) neither loses nor gains beyond two in any set. No rule is consistent over the sets. Not
implemented.

Readings.
- **The regime shift is visible in the fusion optimum.** Every variant that trusts engine 2 less gains on `c2` with the
  cross-fitted engine 2 and is level with the in-sample one (`FUSE_ALPHA` 0.4: +0.016 against −0.003; `FUSE_KRR` 1: +0.013
  against +0.003; the combination +0.020 against +0.005). The earlier knob tests (`fwd-replay` section 6, the parameter
  search, the panel review) were all made with an engine 2 that was either in-sample or fed seen fingerprints.
- **`FUSE_KRR` 1 is the only variant with no type below zero by more than one standard error in any of the three sets**,
  and its `dB` gain is beyond two standard errors in all three. `dB` = the answer is neither a v1 pool row nor an engine-2
  candidate and enters through the join only: the segment the join was built for (+0.011 and +0.011 on the leaderboard,
  v12a and v13a). `FUSE_ALPHA` 0.4 buys the same `dB` with a `dA` loss beyond two standard errors in all sets and four
  more exposed lists on the regime.
- **Exposure rises by 1 to 2 lists** (regime `c2` 23 against 21 per 380, answer lost first 5 against 4; cross-fitted panel
  4 against 3, `c1` 3 against 2). That is below the limit of 5 per 380; it is not "not rising". The mechanism of the gain
  is the mechanism of the exposure: a joined row the v1 side ranks first keeps the place more often.
- **Which reading applies to v35a: "better than v33b locally" on `dB` (ordinary evidence), "equal locally" on `c1`, `c2`,
  `dA`.** The `c2` difference in the unseen regime (+0.013 ± 0.007) is below two standard errors and is not claimed.
  The choice among ten constant variants was made after seeing them; the neighbourhood (`FUSE_KRR` 2, the combination) is
  smooth and has the same signs, which is why the table shows it.
- Segment size on the hidden test is what decides, and it is not measurable here: `dB`-like answers (in PubChem's first
  100 for the mass, outside COCONUT and train) gain, pool answers below the gate whose list is topped by a joined decoy
  lose. On the public leaderboard one answer moving between place 1 and 2 is 0.0038 (`reviews/determ-2026-10-07.md`).

**Wired**: nothing new in the code; `FUSE_KRR` is an existing key (default 3.0, unchanged). Preset
`configs/v35a-v33b-krr1.json` = v33b's config + `FUSE_KRR` 1.0. `tests/test_fuse_krr.py`: defaults unchanged, the preset
differs from v33b by one key, `fuse_rrf` reproduces the reference scores at 3.0, a first row engine 2 cannot list keeps its
place at 1.0 and loses it at 3.0, lists engine 2 does not touch never change, `build_submission` passes the key, the build
tool writes it and attaches no new dataset. All 13 `tests/test_*.py` pass when run directly. Build check:
`tools/build_notebook.py --config configs/v35a-v33b-krr1.json --out scratch/sub_v35a` writes 10 cells that all parse, 16
datasets (v33b's). No dataset folder (no new weights). **Not run on Kaggle, not uploaded, not submitted.**

## 4. Honesty and limits

- **Who has seen what in section 3.** Cross-fitted panel: engine 2's FPNets unseen, its rankers out of fold (4 and 2 models
  per fold instead of the notebook's 8 and 4), v1 side below the gate unseen, above the gate public (structures seen);
  ICEBERG / GLACIER have very probably seen the molecules in every set. Mixed regime: engine 2's rankers out of sample, its
  FPNets have seen the molecules or give NaN features. Panel "in-sample": as every earlier panel replay.
- **24 panel molecules could not be matched** in `rank_train.npz` and are left out of the cross-fitted tables; a molecule
  is matched by one float, so a fold removes some extra rows that share a value (36 to 75 groups of `rank_train.npz`, 76 to
  324 of megayak's per fold), which costs the cross-fitted rankers a little training data.
- **`dA` / `dB` on the panel are not clean "answer outside" lists**: the analog generator re-creates 40% of the removed
  answers (`reviews/panel2-2026-10-05.md`). The joined candidates and the fusion's PubChem list are the fold-safe
  selection's in every variant, as in the v33b replay; `REL_TH` therefore never fires (section 2.3).
- **The fused list's MetFrag term** uses the scores cached for base-list candidates only, as in the v33b replay.
- **Forward scores**: 547 (panel) and 1,452 (regime) new candidates scored; GLACIER all, ICEBERG all but 2 and 3 (the known
  failures). 74 regime molecules' new GLACIER scores come from the CPU run of the same runner.
- **megayak's README numbers and the two FPNets' leak-free status are his claims**; his FPNets were not run.
- **Unknown stays unknown**: what `rank_train.npz` was built with (FPNets, pool, the other 593 molecules), what the
  reference notebook's author tuned `FUSE_ALPHA`, `FUSE_KRR`, `LIB_TAU` and `REL_TH` on, the `derivation` tables' roster.
- **Not done**: megayak's coherent chain (download above the limit); `REL_TH` in a scale-free form; a ranker gate other
  than 0.7; the `j__pcrank` input under the public selection; a Kaggle commit run of v35a; `EXPERIMENTS.md` / `INDEX.md`.
- **Failed or corrected on the way**: two list passes and one CV fold died when the machine's commit limit was reached
  (a game and other agents' jobs; rerun from the caches); a shell here-document truncated a script (rewritten with the file
  tool); one edit commented out the argument parser (fixed before any run used it); the first design compared the
  re-fitted ranker with the in-sample public ranker on the panel (−0.096), which is unfair to it and is what led to the
  matching of section 2.1.

## 5. Reproduce

```bash
export CASMI_DATA='D:\casmi-data' PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1; P=.venv/Scripts/python; S=scratch/regime
$P $S/rg_rows0.py; $P $S/rg_recs0.py; $P $S/rg_pc.py; $P $S/rg_pvcv.py          # sections 1, 2.1, 2.3
(cd $S && $P rg_cv.py) && $P $S/rg_cv_report.py                                  # megayak's rows, grouped CV
$P $S/rg_cf.py                                                                   # cross-fitted rankers for the panel
$P $S/rg_queue.py --cv 0; $P $S/rg_queue.py --jobs p2:c2:pubcf,p2:c2:ucf,p2:c2:uacf,p2:c1:pubcf,p2:c1:ucf,p2:c1:uacf --cv 0
$P $S/rg_stage.py; $P $S/rg_stage_cf.py                                          # stage level
for r in p2 mx; do $P $S/rg_replay.py --reg $r --phase cands --arms $([ $r = p2 ] && echo u,ua,pubcf,ucf,uacf || echo u,ua); done
$P $S/rg_fwdq.py                                                                 # GPU, shared mode, a few minutes
$P $S/rg_replay.py --reg p2 --arms u,ua,pubcf,ucf,uacf; $P $S/rg_replay.py --reg mx --arms u,ua
$P $S/rg_report.py --reg p2 --ref eng:pubcf --prefix cf: --subset cf_ok; $P $S/rg_report.py --reg mx --ref v33b --prefix ""
$P solutions/fusion-v4n/tests/test_fuse_krr.py
```

Outputs in `D:\casmi-data\regime\`: `cv_*`, `pvcv*`, `cf_*`, `eng2\<reg>_<mode>_<arm>\`, `fwd_<reg>\`, `stage*.json`,
`pc_rel_p2.json`, `replay_<reg>.parquet`, `report_*.json`, logs. This file, the preset and the test are new: run
`python tools/gen_index.py` when they are committed.
