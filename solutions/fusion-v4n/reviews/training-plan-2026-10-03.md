<!-- index: fusion-v4n 2026-10-03 training plan: does training our own models make sense now; a LightGBM ranker fitted on joined lists (PubChem rows injected) measured in the mixed regime at N 50 / 100 / 200, retrieval depth and second-stage re-scorers for the PubChem window, what the fingerprint model is worth to the ranker, and verdicts with GPU cost for the simulated-spectra fine-tune, a negative-mode forward model and a de novo generator -->
# Training plan (2026-10-03): does it make sense to train models of our own now?

Offline only. Nothing was submitted, pushed or committed, no Kaggle notebook was run, no existing file was edited, the
GPU was not used. Scripts: `scratch/trainplan/` (git-ignored, log in `scratch/trainplan/LOG.md`). Data, models and raw
outputs: `D:\casmi-data\trainplan\` (306 MB). MRR = MRR@25 on the metric key; "±" is one standard error; differences
are paired per molecule. Reproduce: section 8.

## 0. The answer

**Mostly no. One small model is worth having, and it costs no GPU time.**

| Question | Answer |
|---|---|
| Does it make sense to train models now? | Not the large ones. Nothing measured here or earlier says that a fingerprint-model fine-tune, a forward model or a generator would move the leaderboard, and the two leaderboard steps we have (0.409 → 0.420 → 0.431) came from a change without any training. |
| Which one, if any? | A LightGBM ranker fitted on joined lists (`OJi`, section 2). It is trained (22 minutes on two CPU threads) and sits in `D:\casmi-data\trainplan\models\OJi`. Its purpose is narrower than "rank better": keep injected PubChem decoys below pool answers, which is the only local explanation we have for the v14a loss. |
| GPU hours on the shared machine | **0 for the recommended model** (CPU only: 2 h of engine runs on three below-normal workers plus 22 min of fitting). For reference, the candidates I advise against: simulated-spectra fine-tune 15 to 20 GPU-hours (13 h simulation + 5 to 10 h training at duty 0.8; about 40 to 60 h of wall time if a game runs most of the time), own negative-mode forward model 1.5 to 3 GPU-days, own de novo decoder several GPU-days. |
| Expected leaderboard effect of the recommended model | Unknown sign. Locally +0.011 to +0.016 under the regime's mixtures against v13a, but the same regime gave +0.003 for N 200 and the leaderboard gave −0.025 (section 6). Treat one submission as a test, not as a gain. |

Ranked list (details in sections 2 to 5):

| # | Candidate | Verdict | Why in one line |
|---|---|---|---|
| a | Ranker fitted on joined lists | **trained; test once at N 100** | no measurable gain over the blend we already have after the fusion, but half the decoy-on-top rate before it; the join's cost side is what v14a exposed |
| b | Retrieval model / second stage for the PubChem window | **never as a fingerprint-only model; the engine is the second stage** | snapshot ensemble +0.010 ± 0.009 in the first 50, trained fingerprint re-scorer −0.009 ± 0.010; raising N failed on the leaderboard at 200 |
| c | Full simulated-spectra fine-tune | **later, probably never** | the whole fingerprint model is worth 0.015 to 0.05 MRR to the ranker before the fusion; the fine-tune handicaps exactly the PubChem-only answers the join brings in |
| d | Negative-mode forward model | **FIORA as is: later (offline test first); own training: never** | 7% of molecules have negative spectra only; forward models are worth 0.004 to 0.01 on the leaderboard for the 86% they cover |
| e | De novo / analog generator model | **never in this competition** | a generator conditioned on the same fingerprint cannot beat retrieval from a list that contains the truth: 18% first, 16% for natural products |

**Single next action:** one paired submission, v13a (join N 100, both gates 0.7) with the base-list score replaced by
`z(OJi) + z(public ranker, order-free, injected rows last among ties)` below `lib_max` 0.7. It needs the model
directory as a private Kaggle dataset and the five injection features in `pc_join.choose_score` (section 2.5). If it
does not beat 0.431 by more than 0.005, stop ranker work; no further model should be trained before a validation set
exists that reproduces the v14a drop.

## 1. What the earlier trained models did, and the failure they share

| Trained component | Local evidence | Leaderboard | Why it did not transfer |
|---|---|---|---|
| Our ranker, order-free ties (v10a, v11) | +0.04 to +0.08 MRR before the fusion on class-2 lists | 0.408 and 0.405 against 0.409 | it re-orders candidates that are already listed; the fusion (forward-model z-scores, engine-2) absorbs the base-list change (+0.009 ± 0.008 after it); and the local lists have the truth in the pool |
| FPNet fine-tunes (more steps, natural-product weighting) | ±0.01 top-1 | not submitted | the model is saturated on the data it has |
| FPNet with simulated spectra (pilot, window arm) | +0.05 to +0.06 top-1 at the fingerprint level for [M+H]+ queries of simulated structures | not submitted | most of the pilot's gain is a preference for structures the model was trained on; section 4 |
| Teammate: natural-product-tuned FPNet in engine-2; isomer-ranker channel | | 0.413 and 0.400 against 0.409 | same class: ordering of listed candidates |
| PubChem join (no training) | +0.16 per PubChem-only molecule | **0.420, then 0.431 at N 100, then 0.406 at N 200** | it brings answers into the list; and its cost side is not visible locally |

The shared failure mode is not "the models are bad". It is that every local list is built from training structures
whose answer is in the pool or is hidden by masking, with engine-2 and ICEBERG having seen the molecule. Gains on the
order of listed candidates are then measured against an optimistic pipeline and vanish; costs of adding candidates are
hidden by the same optimism. Each candidate below is checked against this.

## 2. Candidate (a): a ranker fitted on joined lists (run)

### 2.1 What was built
- **Molecules.** The first 600 hold-out molecules of the ranker study's sampling order that are not among the 380 regime
  molecules and share no metric key with them (`select_train.py`; 900 selected, deep PubChem lists finished for 601).
  Deep lists as in the pcjoin study (`pc_deep_train.py`: fold-safe FPNet, 200 structures outside the pool).
- **Lists** (`run_join_train.py`, fold-safe FPNet, order `first`, 0 errors): truth-removed and class-2 lists at N 100
  for 600 molecules (and at N 50 for the first 276 / 197), class-1 lists at N 50 for 50 molecules. In the
  truth-removed lists the truth is an injected PubChem row for 57% and in the list at all for 62%. Class-1 lists at
  N 100 (170 molecules) were finished after the fit and are not in the models.
- **Models** (`a_train.py`; LightGBM lambdarank, 400 rounds, two seeds, order-free rank features everywhere):
  `J` = the packaged ranker's 250 features on joined lists only; `Ji` = + 5 injection features (injected flag,
  PubChem rank, f.z minus the best pool f.z, share of injected rows above the best pool f.z, number of pool rows);
  `OJ` / `OJi` = the same plus the 3,356 lists without injection of the hold-out molecules outside the regime (the
  recipe of the existing ranker).

### 2.2 Choice of the variant, made before the regime was scored
Out of fold on the training molecules (5 folds grouped by scaffold), whole-list MRR before the fusion, paired against
the public ranker as the notebook runs it (`a_cv.csv`):

| Ranker | truth-removed, truth injected, N 100 (340) | class 2, N 100 (598) |
|---|---|---|
| public | 0.584 | 0.715 |
| public, order-free, injected last among ties | +0.021 ± 0.012 | +0.073 ± 0.009 |
| `J` | +0.122 ± 0.016 | −0.032 ± 0.011 |
| `Ji` | +0.094 ± 0.015 | +0.053 ± 0.011 |
| `OJ` | +0.133 ± 0.017 | −0.022 ± 0.011 |
| **`OJi`** | **+0.121 ± 0.016** | **+0.066 ± 0.010** |
| blend of `OJi` with the order-free public ranker | +0.087 ± 0.014 | +0.085 ± 0.010 |

Without the injection features a ranker fitted on joined lists gains on injected truths and **loses on pool truths**:
it cannot tell an injected decoy from a pool row. `OJi` is the only variant that gains on both sides and was fixed
as the primary arm. The injection features carry 12% of its gain (f.z relative to the best pool f.z: 10%).
Class-1 lists below `lib_max` 0.7 are 4 here: nothing can be said about them.

### 2.3 Mixed regime, against the submitted configurations
`a_eval.py`. Every arm is a whole configuration: v4b, join with `PC_JOIN_MAX_LIB` 0.7 and `RANKER_LIB_GATE` 0.7, so
molecules with `lib_max` ≥ 0.7 are identical to v4b in every arm. `public@100` = v13a (the reference), `public@50` =
v12a. `blend2` = the ranker of the ranker study (fitted without injected rows on 3,356 molecules, 600 of which are the
training molecules here) blended with the order-free public ranker; `b:OJi` = the same blend with `OJi`. `dB` / `dA`:
the 325 truth-removed molecules with a PubChem record; `dB_np` its 256 natural-product catalogue structures.

Before the fusion:

| Arm | truth injected (of 325) | injected truth first / in the top 5 | MRR over the 325 | class-2 lists below 0.7: injected decoy on top | their MRR |
|---|---|---|---|---|---|
| public@50 (v12a) | 65.2% | 36% / 74% | 0.333 | 5.9% | 0.758 |
| public@100 (v13a) | 70.5% | 35% / 72% | 0.356 | 8.3% | 0.746 |
| public@200 (v14a) | 74.8% | 37% / 70% | 0.380 | **10.6%** | 0.740 |
| blend2@100 / @200 | | 50% / 86%; 48% / 84% | 0.465 / 0.474 | 17.7% / 16.5% | 0.787 / 0.785 |
| `OJi`@100 / @200 | | 53% / 88%; 51% / 86% | 0.479 / 0.491 | 8.7% / 9.4% | 0.823 / 0.823 |
| `b:OJi`@100 / @200 | | 49% / 86%; 47% / 83% | 0.454 / 0.463 | **4.3% / 3.9%** | 0.849 / 0.849 |

After the whole fusion, paired against v13a (level of v13a zero-forward: `dB` 0.256, `dA` 0.450, `c2` 0.752, `c1`
0.831; on the ICEBERG prefix 0.237, 0.456, 0.740, 0.850):

| Arm, zero-forward (380) | `dB` | `dB_np` | `dA` | `c2` | `c1` | mixtures A / B / C |
|---|---|---|---|---|---|---|
| v4b | −0.176 ± 0.018 | −0.101 ± 0.015 | −0.182 | −0.013 ± 0.006 | +0.001 | −0.039 / −0.045 / −0.043 |
| public@50 | −0.013 ± 0.006 | −0.012 | −0.022 | +0.001 | +0.003 | −0.002 / −0.003 / −0.002 |
| public@200 | +0.010 ± 0.005 | +0.015 | +0.021 | +0.003 ± 0.005 | 0 | +0.003 / +0.004 / +0.003 |
| blend2@100 | +0.048 ± 0.011 | +0.065 | +0.048 | +0.017 ± 0.008 | +0.001 | +0.017 / +0.018 / +0.015 (± 0.004) |
| `OJi`@100 | +0.051 ± 0.013 | +0.074 ± 0.014 | +0.053 | +0.012 ± 0.008 | −0.001 ± 0.005 | +0.015 / +0.016 / +0.014 (± 0.004) |
| `b:OJi`@100 | +0.052 ± 0.011 | +0.064 | +0.038 | +0.023 ± 0.007 | 0.000 | +0.019 / +0.020 / +0.016 (± 0.003) |
| `OJi`@200 | +0.074 ± 0.013 | +0.094 | +0.066 | +0.018 ± 0.008 | −0.001 | +0.021 / +0.022 / +0.019 |

| Arm, ICEBERG prefix (169) | `dB` (145) | `dA` | `c2` | `c1` (64) | mixtures A / B / C |
|---|---|---|---|---|---|
| public@50 | −0.025 ± 0.013 | −0.034 | −0.003 | −0.003 | −0.006 / −0.007 / −0.007 |
| public@200 | +0.025 ± 0.008 | +0.030 | −0.005 ± 0.006 | 0 | +0.003 / +0.004 / +0.005 |
| blend2@100 | +0.030 ± 0.013 | +0.026 | +0.003 ± 0.015 | +0.007 | +0.009 / +0.010 / +0.010 (± 0.006) |
| `OJi`@100 | +0.029 ± 0.016 | +0.036 | +0.012 ± 0.014 | +0.007 | +0.011 / +0.012 / +0.011 (± 0.007) |
| `b:OJi`@100 | +0.024 ± 0.015 | +0.015 | +0.014 ± 0.012 | +0.007 | +0.011 / +0.012 / +0.010 (± 0.006) |

**The comparison that matters: `OJi` against the blend we already had** (same lists, same gates): N 100 `dB`
+0.003 ± 0.009, `c2` −0.006 ± 0.006 (zero-forward); −0.001 ± 0.012 and +0.009 ± 0.010 on the ICEBERG prefix; N 200
`dB` +0.013 ± 0.009, `c2` +0.006 ± 0.006. **Training on joined lists is not distinguishable from the existing ranker
after the fusion.** What it changes is before the fusion: half the decoy-on-top rate of blend2 at equal or better
placement of injected truths.

### 2.4 What can inflate these numbers
- The usual: truths are masked training structures with library relatives; engine-2 and ICEBERG have seen them;
  natural-product structures are the more honest stratum (shown as `dB_np`; here it gains more, not less).
- 380 molecules; the `c1` column rests on the 17% of class-1 lists below 0.7 (about 25 lists; 16 with the truth).
- Arms were not selected on the regime (`OJi` was fixed in 2.2), but four variants and three depths were scored; read
  single cells with one extra standard error.
- The public ranker and blend2 were fitted on simulations that include some regime or training molecules; `OJi`'s
  no-injection lists are from molecules outside the regime only.

### 2.5 What a submission needs (not applied)
`pc_join.choose_score` must build `OJi`'s five features from the injected mask and `injected_info` (PubChem rank) and
call the model on order-free rank features; `a_lib.join_features` and `a_lib.matrix` are the reference
implementation (they take the stored run arrays; the notebook has the same arrays per molecule). Model files:
`D:\casmi-data\trainplan\models\OJi\` (`features.json`, `ranker_s0.txt`, `ranker_s1.txt`; provenance: rows derived from
`train.parquet` with the public engine and the fold-safe public FPNet; no NIST data or NIST-trained model). The
notebook's lists carry three public fingerprint models in `model_views`, the training rows one fold-safe model: the
mismatch the ranker study checked and found harmless, not re-checked here. About one working day including the
parity check and the commit run.

## 3. Candidate (b): retrieval for the PubChem window

### 3.1 Where the truth sits (`b_depth.py`)
Rank of the truth inside the 5,000 shortlist (pool members counted, so slightly pessimistic), truth-removed molecules
with a PubChem record:

| Molecules | n | in the tier | in the shortlist | top 25 | 50 | 100 | 200 | 500 | 1,000 |
|---|---|---|---|---|---|---|---|---|---|
| regime, all | 325 | 91.4% | 83.7% | 50.8% | 60.0% | 65.5% | 71.1% | 76.9% | 79.1% |
| regime, natural-product catalogue | 256 | 89.5% | 83.2% | 45.3% | 55.1% | 61.7% | 68.4% | 75.4% | 78.1% |
| training molecules, all | 341 | 93.0% | 80.1% | 54.8% | 65.7% | 71.0% | 75.1% | 77.7% | 79.2% |
| training, natural-product catalogue | 170 | 86.5% | 79.4% | 46.5% | 58.8% | 66.5% | 71.8% | 75.9% | 77.6% |

Locally each doubling of N adds 4 to 6 points. On the leaderboard 50 → 100 added as much as 0 → 50, so hidden
answers sit deeper than these; and 100 → 200 lost 0.025, so depth is not free (section 6).

### 3.2 Can a better ordering replace depth? (both run)
- **Ensemble of fingerprint models** (`b_ensemble.py`, 311 regime molecules with a record and a list; the first 200
  re-scored): fold-safe model 68.2% in the first 50, two further fold-safe fine-tunes 68.2% and 68.8%, the average
  of the three **69.1% (+0.010 ± 0.009)**, MRR@25 +0.007 ± 0.007. They share an initialisation, so this is a lower
  bound for independent models; the notebook's channel already averages two (A + B). Public model A, which was
  trained on these molecules' real spectra, reaches 77.8% of a possible 78.1% and 48.9% first (fold-safe 19.0%):
  what lifts retrieval is having trained on the structure, which no model can have for a hidden PubChem-only answer.
- **Trained fingerprint-only second stage** (`b_rescore.py`: LightGBM lambdarank on 31 features of the predicted
  logits and the candidate's bits, fitted on 347 training lists): first 50 65.2% → 64.3% (**−0.009 ± 0.010**), first
  100 70.5% → 69.8%. Likelihood ranks exactly as f.z; cosine and soft Tanimoto are far worse (23% and 22% in the
  first 50). There is no information left in the fingerprint prediction that f.z does not use.
- **The engine is the second stage** (`b_engine_stage.py`, N 200, before the fusion). MRR of an injected truth in the
  joint list by its fingerprint rank:

  | Fingerprint rank of the truth | molecules (of 325) | public | blend2 | `OJi` |
  |---|---|---|---|---|
  | 1 to 25 | 192 | 0.57 | 0.68 | 0.70 |
  | 26 to 50 | 20 | 0.32 | 0.52 | 0.59 |
  | 51 to 100 | 17 | 0.35 | 0.53 | 0.60 |
  | 101 to 200 | 14 | **0.10** | 0.28 | 0.25 |

  Small counts in the deep rows. The public ranker gets almost nothing from a truth beyond rank 100 while every
  list pays for 100 more decoys.

Verdict for (b): no retrieval model to train. A first stage trained with PubChem-window negatives (the loss uses 63
pool negatives today) is the only untested idea; it needs fingerprints of PubChem negatives for a large part of the
training structures (about 11 CPU-hours for 64 negatives each) and a from-scratch run of unknown length, and the
ensemble and re-scorer results give no reason to expect a gain. Later, if at all.

## 4. Candidate (c): the full simulated-spectra fine-tune

- Measured earlier: +0.05 to +0.06 top-1 at the fingerprint level for [M+H]+ queries of simulated natural products,
  nothing for negative queries, and a non-simulated truth loses 0.017 ± 0.005 per pair against simulated competitors.
- Measured here (`c_ablate.py`, 3,100 class-2-like hold-out lists, 5-fold CV of our ranker recipe, before the
  fusion): removing **every** FPNet-derived feature (72 of 250) costs 0.015 ± 0.003 MRR (0.812 → 0.797); natural-product
  catalogue 0.026 ± 0.005; best analog below 0.5 0.044 ± 0.010; natural products with a weak analog 0.051 ± 0.013
  (315 molecules). So the whole fingerprint model is worth 0.015 to 0.05 to the ranker; a fine-tune that improves it
  by a tenth moves a fraction of that, before a fusion that already absorbed a base-list gain four times larger
  (our ranker, v10a / v11).
- It acts against the join. After a full run every pool competitor is a training target and a PubChem-only answer is
  not: the measured −0.017 per pair applies to exactly the answers that moved the leaderboard, and the list-relative
  f.z features that cost an injected row 0.2 to 0.3 score units today get worse.
- Cost: 13 h simulation + 5 to 10 h training on the GPU at duty 0.8 (about 40 h of wall time for the simulation
  alone if a game runs throughout), then fine-tunes of every pipeline checkpoint and a refit of the ranker; a
  working week. The commands are in `reviews/simspec-2026-10-02.md` section 7.2.

Verdict: later, probably never. It shares the failure mode (order of listed pool candidates) and adds a new one.

## 5. Candidates (d) and (e)

**(d) Negative-mode forward model.** On enveda-np-examples 7.2% of molecules have negative spectra only and 59% have
some. A forward model only orders listed candidates. Its leaderboard value where it does apply (86% of molecules):
GLACIER as a third term +0.011 in the public lineage, GLACIER off −0.005, "[M+H]+ only" +0.004; in the regime
ICEBERG is worth nothing where the truth is a pool row. A negative model of the same quality would add about
0.07 / 0.86 of that for the negative-only molecules, +0.001 to +0.002, and at most a few thousandths more as extra
evidence. FIORA (MIT, MSnLib-trained, Orbitrap HCD) costs 2 to 3 days of integration and can be tested offline first
(in-group MRR on negative-only molecules against chance 0.16; only 25 leak-free molecules): later. Own ICEBERG
training: 1.5 to 3 GPU-days on a shared 16 GB card where upstream asks for two 24 GB cards: never.

**(e) De novo or analog generator model.** Public options and their data are in `coverage-2026-10-02.md` section 5
(FRIGID, MIST + MolForge, DiffMS: no NIST; MSNovelist, Spec2Mol: NIST, excluded); audited exact-match rates are near
10%. A bound from our own data: a decoder conditioned on the predicted fingerprint searches a superset of what the
PubChem channel searches, and that channel puts the truth first for 18% (natural products 16%) when the truth is in
the list. For structures in no database that is the ceiling before any ranking loss; with a class-3 share of 10 to
30% and rank 3 to 5 in the final list it is +0.002 to +0.005 at best. Our fingerprint definition also differs from
the one the public decoders take, so either side would have to be retrained (days of GPU). Never in this
competition; the rule-based generator is the other agent's topic.

## 6. v14a (N 200: 0.406 against 0.431) and what it means for a ranker of ours

- **The regime missed it.** public@200 against public@100 locally: `dB` +0.010 ± 0.005 (ICEBERG prefix +0.025 ±
  0.008), `c2` +0.003 / −0.005, mixtures +0.003 to +0.005. The leaderboard says −0.025. The regime is therefore not a
  judge of anything that changes how many outside candidates enter the list, in either direction; it got 50 → 100
  right in sign and size (+0.013 on `dB`; leaderboard +0.011 overall, which is far more per affected molecule).
- **The only local hint** is the pair of columns in 2.3 and the table in 3.2: under the public ranker an injected
  decoy tops 5.9 / 8.3 / 10.6% of the class-2 lists below the gate at N 50 / 100 / 200, while a truth at fingerprint
  rank 101 to 200 is worth 0.10. Locally engine-2 repairs the displaced pool answers (it has seen these molecules:
  first for 51 to 79%); the pcjoin study measured the class-1 cost at −0.029 on the v1 side against −0.006 after the
  fusion. On the hidden test that repair is weaker, so the cost side is larger than any local number. This is a
  hypothesis, not a measurement. Other explanations that cannot be excluded from here: runtime or budget effects of
  200 candidates in the scoring run (forward-model budgets, fallbacks), and public-leaderboard noise (about ±0.01
  per submission; a 0.025 drop is beyond it).
- **For candidate (a):** its local gain after the fusion (+0.003 ± 0.009 over the existing blend) is no argument.
  Its argument is the cost side: `OJi` holds decoy-on-top at 9% and the blend with the public ranker at 4% for N 100
  and 200 alike, against 8 to 11% for the public ranker and 17% for blend2, and it keeps a deep truth at 0.25
  instead of 0.10. If the v14a loss is decoys displacing pool answers, this is the component that addresses it; if
  it is something else, the ranker does nothing. The regime cannot say which.
- **Should a ranker of ours accompany the join at N 100?** Not as an expected gain. The record is: order-free ties
  alone 0.408, our blend without the join 0.405, both against 0.409, i.e. no effect or slightly negative, and local
  predictions of +0.005 to +0.015 for them. With the join the ranker has a job it did not have then (injected rows),
  so one paired submission against v13a is justified as a test, with `b:OJi` (lowest decoy rate) rather than
  blend2 (highest). Do not combine it with a larger N in the same submission, and do not go back to N 200 on the
  strength of a ranker until that submission has been read.

## 7. Cost and validation per candidate

| Candidate | Changes in the pipeline | Molecule types | Local effect | Validation before a submission | GPU hours / working days | Risk |
|---|---|---|---|---|---|---|
| a `OJi` | base-list score below `lib_max` 0.7 on joined lists | PubChem-only answers; pool answers without spectra | +0.05 `dB`, +0.012 to +0.023 `c2` against v13a; nil against blend2 | parity of the inference features on stored rows; commit run; then the leaderboard | 0 / 1 | sign unknown on the leaderboard |
| b retrieval model | order of the PubChem list before injection | PubChem-only answers | ensemble +0.010 ± 0.009, re-scorer −0.009 ± 0.010 (first 50) | `b_rescore.py` / `b_ensemble.py` on the regime | 0 run; PubChem-negative first stage: unknown, days | no signal |
| c simulated fine-tune | every FPNet of the pipeline, then the ranker | pool answers without spectra, [M+H]+ only | ≤ 0.015 to 0.05 × a fraction before the fusion | fold-safe twin in the regime, with the join | 15 to 20 (40 to 60 h wall with games) / 5 | hurts injected answers |
| d FIORA / own negative model | forward-model term for negative spectra | 7% negative-only, listed truths only | not measured | in-group test on negative-only molecules | 0 / 2 to 3; own: 36 to 72 / 5 to 8 | domain (Orbitrap model, timsTOF test) |
| e generator model | new candidates | answers in no database | bound 16 to 18% first before ranking | hide-and-regenerate (3× optimistic) | days / 5 to 8 | high |

## 8. Reproduce

```bash
export CASMI_DATA='D:\casmi-data' PYTHONHASHSEED=0; P=.venv/Scripts/python; S=scratch/trainplan
$P $S/select_train.py --n 900 ; $P $S/pc_deep_train.py --workers 2 --k 200      # deep lists (8 s per molecule; 601 were run)
for i in 0 1 2; do bash $S/workers.sh $i 3 train:drop:100:600 train:c2:100:600 regime:drop:200 regime:c2:200 train:c1:100:170 & done
bash $S/follow.sh 50                                                             # the N 50 lists of the first molecules (stopped early)
$P $S/a_train.py --ns 100,50                                                     # CV table and models (22 min, 2 threads)
$P $S/a_eval.py --models OJi,Ji,OJ --ns 50,100 --ref public@100 --tag _n100
$P $S/a_eval.py --models OJi --ns 100,200 --ref public@100 --tag _n200
$P $S/b_depth.py ; $P $S/b_ensemble.py ; $P $S/b_rescore.py ; $P $S/b_engine_stage.py --n 200 ; $P $S/c_ablate.py
```

Outputs: `D:\casmi-data\trainplan\` (`a_cv.csv`, `a_eval_n100.json`, `a_eval_n200.json`, `a_eval_rr_*.parquet`,
`b_ensemble.parquet`, `b_rescore.csv`, `b_engine_stage_n200.parquet`, `c_ablate.csv`, `models\`, `rows\`,
`regime\rows\`, `logs\`). What failed or is weaker than it looks: the training set is 600 molecules, not 900 (the
deep lists were stopped when the plan changed to N 100); class-1 lists at N 100 are not in the models; the first
deep-list job was restarted once with fewer workers; the snapshot ensemble is not an ensemble of independent models;
`INDEX.md` was not regenerated (no existing file may be edited): run `python tools/gen_index.py` when this review is
committed.
