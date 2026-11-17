<!-- index: arsokan-fable validation harness (2026-11-17): n0's own lists on 300 held-out molecules with cross-lab class 1; acceptance test against known hidden-test effects; how to read it (loss detector) -->
# Validation harness "vh" (2026-11-17)

Why: three re-ranking ideas looked positive offline and were zero or negative on the hidden test (TA-1 −0.031, vpd
0.000, w30c −0.008). The earlier offline sets used another pipeline's lists, query spectra that are perturbed copies
of same-lab train spectra, and class 2 only.

**Verdict: usable as a loss detector, not as a net-score predictor.** Judge a re-ranker by its loss side and by
"correct rank 1 demoted" per class; treat its gain side as an upper bound.

## Design

The n0 notebook (`submissions/n0-0.492/`) run unchanged on a synthetic test built from `train.parquet`, with the
competition path redirected to a reduced train + synthetic test (private kernels `casmi-vh-smoke`, `casmi-vh-full`;
code not yet in git, see "Code" below). 300 molecules, real enveda-180 (timsTOF) spectra as queries, no perturbation:

| Group | n | What stays in the train file the pipeline sees |
|---|---|---|
| C1x, class 1 cross-lab | 110 | all enveda rows of the molecule removed; its rows in other libraries (pluskal_ms2, gnps; Orbitrap) stay |
| C1s, class 1 same-lab | 60 | query = 40 / 60 eV rows (removed); 20 eV, stepped-energy rows and other adducts stay |
| C2, class 2 | 130 | every row of the molecule removed; truth still in the candidate pools / PubChem |

Class 3 is not simulated. Network density matches the visible test (78 edges / 300 vs 109 / 400).

Two flaws of the public `xiangqizhang/casmi26-fold0-harness` that this one does not have (they affect our 11-16
measurements on that harness): its overlay (and N-1c in n0) globbed the VISIBLE test instead of the harness queries;
and it switched off the popularity sum in the PubChem channel (a pipeline change, not a redirect).

Residual leak: library search, spectrum cache, engine-2 features, N-1c and `library_max` come from the reduced train
(clean). The neural scorers (fpnet_full1, FPNet A+B, sanpier final model, popwenman v80, the LightGBM rankers) were
trained on full train (leak), and the candidate pool holds every training structure, so a class-2 truth is in the
list for 130 / 130 (far more than on the hidden test). Levels are therefore optimistic; only signs are read.

## Run (casmi-vh-full v1, 2 h 38 min on a T4, ~3.0 GPU h in total with the smoke run)

| | MRR | top-1 | pipeline `library_max` (median; share ≥ 0.9) |
|---|---|---|---|
| all 300 | 0.8563 | 0.750 | |
| C1x | 0.9439 | 0.900 | 0.83; 8% |
| C1s | 0.8944 | 0.817 | 0.84; 27% |
| C2 | 0.7646 | 0.592 | 0.58; 11% |

Unlike the visible test (every `library_max` = 1.0, so veto / TA stages never act there), cross-lab class-1 lists sit
almost entirely below the 0.9 gate.

## Acceptance test: changes whose hidden-test effect is known

Replayed offline on the kernel's saved files (offline N-1c at LAM 0.35 reproduces the kernel on 300 / 300 lists).
Δ against n0 (LAM 0.35); `dem` = correct rank 1 demoted.

| Change | Harness Δ | win / loss, dem | Hidden / public |
|---|---|---|---|
| N-1c off | +0.0003 | 6 / 4, 1 | −0.005 |
| LAM 0.25 | +0.0006 | 5 / 3 | −0.004 |
| LAM 0.30 | −0.0003 | 3 / 1 | −0.004 |
| LAM 0.38 | −0.0001 | 2 / 1, 0 | −0.002 |
| LAM 0.40 | −0.0018 | 2 / 2, 1 | −0.002 |
| LAM 0.42 | −0.0032 | 2 / 3, 2 | −0.002 |
| LAM 0.45 | −0.0055 | 2 / 6, 3 | −0.006 |
| LAM 0.50 | −0.0072 | 3 / 8, 5 | −0.008 |
| TA-1 (t0) | +0.0028 (CI −0.010..+0.016) | 13 / 9, 6 | −0.031 |
| vpd | +0.0008 | 3 / 0, 0 | 0.000 |
| w30c | +0.0014 | 6 / 1, 1 | −0.008 |

Reproduced: the decline above LAM 0.35 (sign and size; mechanism = correct leaders demoted); vpd ≈ 0; w30c's loss
(exactly one loss: a correct cross-lab class-1 leader sent to the bottom, −0.958; its `library_max` is 0.29 because
its Orbitrap reference does not resemble the timsTOF query, so "own spectra contradict the query" fires on the
truth; one such molecule on a ~130-molecule public split is −0.0074).

Not reproduced: TA-1's net sign (+0.003 here, +0.043 on the old harness, −0.031 hidden) and the rise from LAM 0.25 to
0.35 (flat here).

Why TA-1's net sign fails, measured: gain side +0.0146, loss side −0.0118. All 6 demoted correct leaders are class 2
(6 of 77 correct class-2 leaders); class 1 is protected by the gate. The gains come from lifting a class-2 truth that
already sits at rank 2–5, which is 40% of class-2 lists here because of the leaky pool; on the hidden test the truth
is in the list far less often, so the gain side mostly disappears while the loss side stays. So: **TA-1's losses are
real and fall on correct class-2 leaders; its offline gains were an artefact of the pool.**

By the loss-detector rule the harness gets all four known cases right: TA-1 (6 demotions) bad; w30c (1 cross-lab
demotion at −0.96) bad; vpd (0 demotions) harmless; LAM > 0.40 (1–5 demotions) bad.

## Team submissions of 2026-11-16 on this line (for reference)

n0 0.492; t0 (n0 + TA-1) 0.461; vpd (n0 + self-reference veto, TAU 0.30) 0.492; w30c (veto TAU 0.40, 30 references,
confirmation) 0.484. vpd and w30c are Ömer's; their veto path never executed in the commit run ("active lists 0" on
the visible test), so a dry run on a copy with the library gate off is needed before any further variant.

## Code

Builder cell, redirect patches, molecule selection and the replay script live in the helper session's working files
(`scratch/vh/`), to be committed separately. To make the harness a net predictor, class-2 truths would have to be
removed from the candidate-pool datasets for part of the set (private pool copies, one more ~2.6 GPU h run).

## Stage audit of n0 and error anatomy (same run, offline)

Consecutive pipeline snapshots (Δ MRR; win / loss; correct rank 1 demoted):

| Stage | Δ | win / loss | dem | Where |
|---|---|---|---|---|
| pipeline reference → e006 control | +0.0094 | 5 / 0 | 0 | class 2 |
| e006 control → frozen 0.465 final | −0.0035 | 0 / 8 | 0 | class 2 |
| V0 → V1 neural 0.25 | +0.0044 | 11 / 0 | 0 | class 2 |
| V1 → V2 (neural 0.5 on ranks 2–25) | +0.0027 | 10 / 5 | 0 | |
| V2 → CUSTOM (neural 0.75 on ranks 2–25) | −0.0017 | 4 / 7 | 0 | class 1 |
| CUSTOM → merge (fragment check + priors + mass prior) | −0.0040 | 1 / 13 | 0 | class 2 |
| merge → mass rank-1 rule | +0.0050 | 3 / 0 | 0 | class 2 |
| N-1c LAM 0.35 | −0.0003 | 4 / 6 | 1 | losses in class 1, gains in class 2 |
| extlib overlay | 0 | | | 0 matches |

- No stage of n0 demotes a correct leader except N-1c (one, same-lab class 1). After the base engine the leader
  changes on 6 of 300 lists. **No stage deserves removal**: the two negative ones act on ranks 2–25 only, and the
  public board contradicts removing them (w_mid 0.75 was +0.004; POP_W / SPEC_W and V8 MASS → BASE were +0.000); their
  harness losses are the leaky-pool artefact (harness truths are unpopular library compounds).
- Errors of n0 (truth not rank 1): C1x 11, C1s 11, C2 53. In 73 of 75 the leader is a same-formula isomer, usually
  a training structure, not a near-twin (median Tanimoto 0.5–0.6), and it is already rank 1 in the base engine. No
  later stage created these errors.
- `library_max` in `frozen_candidate_evidence.parquet` is one value per molecule, not per candidate, so n0's outputs
  cannot say which candidate the library hit belongs to.
- Re-weighting class 2 to a plausible hidden level (a model, not a measurement) flips TA-1 to −0.004 in one variant
  but does not make the harness a net predictor.

Candidates tested by the loss-detector rule (none worth a submission slot on its own):

| Change | Δ | win / loss | dem | Note |
|---|---|---|---|---|
| N-1c only where `library_max` < 0.9 | +0.0007 | 2 / 0 | 0 | principled gate; 2 lists |
| N-1c only where `library_max` < 0.7 | +0.0029 | 4 / 0 | 0 | threshold chosen after looking |
| confirmation only (0.85 / +0.20, gate 0.9; veto off) | +0.0033 | 2 / 0 | 0 | fires only in same-lab class 1 |
| confirmation 0.75 / +0.10 | +0.0011 | 2 / 2 | 1 | starts demoting cross-lab leaders |
| veto only 0.30 | +0.0008 | 3 / 0 | 0 | = vpd |
| veto only 0.40 | −0.0003 | 5 / 1 | 1 | w30c's loss is the veto threshold, not the confirmation |

## Can the class-1 isomer errors be separated? No (offline, same run)

For the top 10 of every list, each candidate's own-spectrum agreement with the query was computed from the reduced
train (best single pair and merged cosine on shared adducts, matched peaks).

- **Cross-lab class 1 (11 errors): in all 8 comparable cases the WRONG leader's own spectrum matches the query better
  than the truth's.** The truth's references are Orbitrap (0.86–0.96 on 5–36 peaks); the wrong leader is a
  same-formula isomer with a same-instrument enveda-180 reference (0.87–0.995 on 10–76 peaks). The molecule-level
  `library_max` there is the wrong isomer's match, so "promote the library-matched structure" would confirm the error.
- Same-lab class 1 (11 errors): truth better in 4 of 10 comparable cases.
- Control, the 148 class-1 lists n0 gets right: truth 0.95 / 0.90 (same-lab / cross-lab median) vs best same-formula
  rival 0.69 / 0.58. The signal is good on average, which is why the base engine is already right on about 90%.
- Swap rule, pre-registered (challenger ≥ 0.70, ≥ 6 peaks, ≥ 0.15 above the leader, leader has own spectra): fixes 3
  (all same-lab), demotes 2 correct leaders. Over a 40-cell grid no cell has ≥ 5 fixes with 0 demotions; no cell has
  a cross-lab fix without a cross-lab demotion. Without "leader has own spectra" the rule demotes 9–15 correct
  class-2 leaders.
- In n0, `library_max` is only a per-molecule gate after the base engine (LIB_TAU 0.9, LIB_HIT_GATE 0.78,
  MASS_LIB_TAU 0.9); there is no library promotion threshold to lower.

Conclusion: the remaining class-1 errors are isomer pairs whose same-instrument spectra are nearly identical while the
truth's only reference comes from another instrument. Spectral agreement cannot break that tie in the right
direction; re-ordering n0's output lists is exhausted as a source of gain. Only another kind of evidence could help.

## Second run on natural products: `casmi-vh-np` (2026-11-17)

Hidden-like set: the 250 `enveda-np-examples` molecules (timsTOF natural products, "closest to test"), queries = their
own spectra, all their np-examples rows removed from the reduced train; plus 50 fresh enveda-only same-lab class-1
molecules. All 250 keep outside-library references (median 168 rows); none is in enveda-180, none is left without a
reference, so there is one NP group ("NPx", cross-lab class 1) and no NP class 2. Run: 3 h 40 min on a T4.

Limits: these are common natural products with references everywhere; most of n0's models were trained on them
(only the fold-safe FPNet excluded all 250). The set prices "known NP with a cross-lab reference", not molecules
without references.

n0 on this set: NPx MRR 0.9326, top-1 0.892, truth in list 0.992, `library_max` p10 / median / p90 0.70 / 0.86 / 0.93;
C1s MRR 0.9467. About twice as easy as the hidden test (0.492): the hidden set must be dominated by molecules
without a usable reference.

### Acceptance replay: the signs match the board here

Δ against n0 on the 300; `dem` = correct leaders demoted.

| Change | Δ | NPx win / loss | dem | Hidden / public |
|---|---|---|---|---|
| LAM 0.25 | −0.0031 | 1 / 2 | 2 | −0.004 |
| LAM 0.30 | −0.0014 | 1 / 1 | 1 | −0.004 |
| LAM 0.38 | −0.0002 | 0 / 1 | 0 | −0.002 |
| LAM 0.40 | −0.0007 | 0 / 2 | 0 | −0.002 |
| LAM 0.45 | −0.0036 | 1 / 4 | 2 | −0.006 |
| LAM 0.50 | −0.0052 | 1 / 6 | 2 | −0.008 |
| TA-1 (t0) | −0.0104 (CI −0.0211..−0.0003) | 3 / 10 | 6 | −0.031 |
| vpd | −0.0068 | 0 / 3 | 2 | 0.000 |
| w30c | −0.0068 | 0 / 3 | 2 | −0.008 |

The LAM curve peaks exactly at 0.35 with both sides lower, TA-1 is clearly negative, w30c is negative at the right
size. Only vpd differs (−0.007 here, 0.000 hidden; 2 demotions, which can be zero events on ~130 public molecules).
**For re-rankers that act on molecules with references, this run is a net-sign predictor**, which the first harness
was not. It says nothing about molecules without references.

### Instrument-aware swap rule: failed out of sample

Rule (pre-registered, calibrated on molecules outside both harness sets): convert each candidate's own-spectrum cosine
to a log likelihood ratio per reference-instrument family (timsTOF same-molecule median 0.94 but the nearest isomer
reaches 0.93 in 10% of cases; Orbitrap family same-molecule median 0.86, isomers rarely above 0.35), swap the leader
for a same-formula challenger in ranks 2–5 when the difference is ≥ log 3.

- First harness: C1x 7 fixes / 0 demotions, C1s 3 / 2 (criterion failed on C1s).
- NP run (frozen rule): **4 fixes, 4 correct leaders demoted**; never fires on the 50 C1s. Every fire is Orbitrap
  against Orbitrap. The premise "an enveda-180 isomer leads while the truth has only outside references" occurred
  0 times in 250 NP molecules (43 had an enveda isomer available): the 7 / 0 came from how the first harness was
  built. Line closed.

### Stage audit on natural products: n0's late stages lose 0.016 here

NPx MRR 0.9485 at the public-0.466 stage → 0.9326 final.

| Step | NPx Δ | win / loss | correct leaders demoted |
|---|---|---|---|
| neural 0.5 / 0.75 on ranks 2–25 | −0.0045 | 2 / 12 | 0 |
| merge step (fragment check, reason `fragment_neural_top1`) | −0.0102 | 9 / 9 | 7 |
| mass rank-1 rule | −0.0020 | 0 / 1 | 1 |
| N-1c LAM 0.35 | +0.0008 | 4 / 6 | 3 |
| extlib overlay | 0.0000 | 4 / 5 | 4 |

The fragment check moved 8 leaders, 7 of them correct; all 7 had `library_max` 0.70–0.87 and the truth first in both
engines. These stages gained on the public board (0.466 → 0.492), so they pay on molecules this set does not
contain; the audit says they cost on known NPs with a cross-lab reference. A leader-move guard is being evaluated
(hypothesis from this run, so the first harness is its out-of-sample set).

### Leader-move guard LMG-1 (candidate; `vh/cell_lmg.py`, not submitted yet)

Anatomy of the merge step's leader moves: NP run 8 moves, 0 good (7 demote a correct leader, 1 wrong → wrong); in 7
of 8 the new leader is not a training structure while the old one is. First harness 2 moves, both class 2: one good
(truth promoted; `library_max` 0.39, engines disagree, fragment margin 0.51, the largest of all ten), one neutral.

Guard G1: undo the move when the replaced leader was rank 1 in both engines and `library_max` ≥ 0.7. Downstream
stages (mass rule, N-1c, overlay) replayed, not assumed.

| Guard | Run | Blocked | bad / good / neutral | Final Δ | win / loss |
|---|---|---|---|---|---|
| G1 | NP (in-sample) | 6 of 8 | 6 / 0 / 0 | +0.0033 | 2 / 0 |
| G2 (both engines only) | NP | 7 of 8 | 6 / 0 / 1 | +0.0036 | 3 / 0 |
| G3 (`library_max` ≥ 0.7 only) | NP | 6 of 8 | 6 / 0 / 0 | +0.0033 | 2 / 0 |
| G1 / G2 / G3 | first harness (out-of-sample) | 0 of 2 | – | 0 | – |
| block every move | first harness | 2 of 2 | 0 / 1 / 1 | −0.0017 | 0 / 1 |

Honest size: 6 blocked demotions give 2 final wins, because n0 already repairs 4 of them later (2 by N-1c, 2 by the
overlay). The guard was suggested by the NP run, so that evidence is in-sample; the out-of-sample run only shows it
blocks nothing there and leaves the one good move alone. Public history agrees it is small: tightening the fragment
margin scored exactly the base (0.477). Side effect: N-1c reads all lists, so 11 other NP lists end in a different
order (no reciprocal rank changed in this run; not guaranteed on the hidden set).

Overlay: its 4 NP demotions are other molecules; in three the replaced leader was rank 1 in both engines and the
external hit beat `library_max` by 0.008–0.018. A guard on engine agreement would block 3 bad / 0 good on NP, but it
was found after looking and the overlay is worth about +0.006 on the board; not a candidate without a fresh set.

#### LMG-1 in-notebook checks (2026-11-17, both pass; not submitted)

- `casmi-vh-lmg16` (harness + n0 + LMG-1, 16 molecules incl. the 8 merge-moved NP molecules, same reduced train as the
  NP run): `LMG-1 DRY RUN | status applied | lists 16 | merge leader moves 8 (merge report: 8) | guarded 6 |
  restored 6 | unguarded lists identical True`. V2 lists, unguarded merge output and `library_max` identical to the
  300-run on 16 / 16. Final lists differ on 5, all restored ones; MRR on these 16 (chosen because the guard acts on
  them, so a function check, not an estimate) 0.7706 → 0.8644. No N-1c neighbours inside the 16, so guard + N-1c
  together is covered only by the offline replay on 300.
- `casmi-vh-lmg-sub` (n0 notebook + the LMG-1 cell after cell 18, nothing else changed, COMMIT_MODE 'smoke'): every
  n0 stage logged as in n0; `merge leader moves 0 | guarded 0 | restored 0 | unguarded lists identical True` on the
  visible test; `submission.csv` byte-identical to n0's smoke output.
- Expected public effect: on ~130 public molecules the guard likely acts on 0–2 lists, so 0.492 unchanged is a
  plausible outcome.
