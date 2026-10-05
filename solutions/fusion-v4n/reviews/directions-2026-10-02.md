<!-- index: fusion-v4n 2026-10-02 direction study: where v4b loses reciprocal rank on the labelled stage caches (loss decomposition, oracles, per-signal isomer quality, a leak-reduced "fp-honest" regime), and a ranked evaluation of six next directions -->
# Directions after v4b (2026-10-02): where reciprocal rank is lost and what to build next

Offline only: cached stage outputs were re-fused, a few public checkpoints were scored on the local GPU, nothing was
submitted or pushed, no tracked file was changed. Scripts: `scratch/directions/` (git-ignored); their outputs:
`scratch/directions/out/`; large inputs: `D:\casmi-data\directions\` (3.6 GB). Reproduce: section 6.
Config everywhere: **v4b** (`POP_MU 0.25, FRAG_LAM 0.5, FRAG_MODE gap, ICE_LAM_LIB 0, GL_LAM_LIB 0, TOPN 60`, LB 0.409).

## 0. Bottom line

1. **In every cache the truth is already in the lists.** It is in the v1 base top-60 for 100% of the molecules of
   val1, valc1 and vis, and the top formula group is the right one 97 to 100% of the time. The caches therefore measure
   one thing well, the order of same-formula isomers, and cannot see the molecules whose truth never enters a list.
   From the LB (0.409) and the regime scores below, about **38 to 46% of hidden molecules score close to zero**
   (inference, section 2.7). That bucket is larger than everything measured here.
2. **Where the truth is present, the loss is isomer order.** With the ranker's memorised isomer order replaced by a
   fingerprint model that never saw the molecules (new "fp-honest" regime), v4b scores 0.575 and loses 0.425:
   0.374 to same-formula isomers above the truth, 0.040 to truths pushed out of the top 25, 0.011 to wrong formulas.
3. **Engine-2 is a class-1 engine and its fusion costs class-2 molecules 0.18 MRR.** Alone it scores 0.915 on valc1 and
   0.058 on val1 (truth in 7.6% of its lists although its pool holds 499k COCONUT structures). Fusing it only when
   `lib_max >= 0.6` recovers +0.093 ± 0.015 on the class-2 side and touches 3.2% of class-1-like molecules.
4. **ICEBERG / GLACIER cover 86% of molecules; on the other 14% nothing works.** MRR 0.33 against 0.61 (0.26 after
   stratifying by size). The frag "gap" term is at chance there (top-1 for 1 of 34 molecules, random 4.4%).
5. **Training our own FPNet runs locally but does not beat the public one.** The public 5-hour fold-safe recipe picks
   the right isomer in 27.6% of val1 groups, the public FPNet A in 41.6%; both never saw these molecules.

Recommended to start now: (1) regime-aware gating of the channels, beginning with the engine-2 library gate and growing
into the learned re-ranker of direction (b); (2) a local run of the v1 engine with a fold-safe FP bank and `drop_pid`,
which makes the "truth not in the lists" bucket measurable and is the shared prerequisite of (b) and (d).

## 1. Regimes and what each can be trusted for

| Regime | Source | What differs | Known bias |
|---|---|---|---|
| val1 plain | `runs/val1`, 250 natural products, all spectra purged | nothing | v1 ranker memorised the isomer order (in-group top-1 93%) |
| val1 sim60 | same, base list cut to 60, same-formula candidates shuffled, 6 seeds | ranker carries no isomer information | pessimistic for the ranker, over-rewards every other term |
| **val1 fp-honest** (new) | same; inside each formula group the order is that of FPNet A (`fpnet_0.pt`, trained without enveda-np-examples) | isomer order comes from a model that never saw the molecule | order between formula groups and top-60 recall still come from the leaked ranker; ICE/GL very likely saw the molecules; popularity over-rewarded |
| valc1 | `runs/valc1`, same molecules, only timsTOF rows purged | class-1-like | v1 ranker saw the purged timsTOF spectra |
| vis | `runs/vis`, 400 exact library hits (enveda-180) | library regime | ranker right 400 of 400; chemistry is synthetic, not natural products |

Two corrections to the existing harness. (a) `regime_eval.py` shuffles the 80-deep dump and then cuts to `TOPN 60`, so
10.7% of the truths leave the list by the shuffle alone (v4b 0.494 with my six seeds; `SEARCH-2026-10-01.md` has 0.476
with twelve). `sim60` cuts first (0.514). (b) fp-honest
reproduces the LB order v1 < v2 ≈ v4a < v4b; like every regime so far it does not see v3b's drop:

| Regime | v1 (LB 0.399) | v2 (0.404) | v3b (0.390) | v4a (0.404) | v4b (0.409) |
|---|---|---|---|---|---|
| plain | 0.6258 | 0.6259 | 0.6272 | 0.6265 | 0.6245 |
| sim60 | 0.4341 (−0.060 ± 0.008) | 0.4940 | 0.4918 | 0.4898 | 0.5140 (+0.020 ± 0.005) |
| fp-honest (A) | 0.5307 (−0.031 ± 0.010) | 0.5616 | 0.5679 (+0.006 ± 0.005) | 0.5603 (−0.001 ± 0.001) | 0.5751 (+0.014 ± 0.006) |

Exposure of the 250 validation molecules: 245 have GNPS, MoNA or MassBank spectra in train, the sources of
MassSpecGym, so ICEBERG and GLACIER have very likely been trained on them. Their popularity is far above ordinary
natural products (median 13 against 5.4 for the NP-catalogue structures of the fold-0 hold-out).

## 2. Part 1: error analysis of v4b

### 2.1 Where the truth is and where reciprocal rank goes

"Cut" = truth present in a list but not in the final 25. "Wrong formula" = gain if the wrong-formula candidates above
the truth were removed. "Isomers" = the rest, same-formula candidates above the truth.

| Regime | n | MRR | top-1 | truth in final 25 | lost: absent everywhere | lost: cut | lost: wrong formula | lost: isomers |
|---|---|---|---|---|---|---|---|---|
| val1 plain | 250 | 0.624 | 0.400 | 0.980 | 0 | 0.020 | 0.014 | 0.341 |
| val1 sim60 | 250 | 0.514 | 0.336 | 0.936 | 0 | 0.064 | 0.014 | 0.408 |
| val1 fp-honest (A) | 250 | 0.575 | 0.380 | 0.960 | 0 | 0.040 | 0.011 | 0.374 |
| val1 fp-honest (fold-0 FPNet) | 250 | 0.543 | 0.356 | 0.960 | 0 | 0.040 | 0.011 | 0.407 |
| valc1 | 250 | 0.925 | 0.884 | 0.980 | 0 | 0.020 | 0.000 | 0.055 |
| vis | 400 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 0 |

Truth recall per list: v1 base top-60 100% in all three runs; engine-2 top-40 7.6% (val1), 99.6% (valc1), 100% (vis);
PubChem list 0% everywhere, by construction (it only lists keys that are not in the pool, and every truth is in the pool).

### 2.2 Oracles

| Regime | v4b | wrong formulas removed | best same-formula isomer picked, truth in final 25 | same, truth anywhere in the lists | truth promoted to rank 1 if in final 25 | promoted if anywhere |
|---|---|---|---|---|---|---|
| val1 plain | 0.624 | 0.639 | 0.963 | 0.983 | 0.980 | 1.000 |
| val1 sim60 | 0.514 | 0.528 | 0.921 | 0.985 | 0.936 | 1.000 |
| val1 fp-honest (A) | 0.575 | 0.586 | 0.947 | 0.987 | 0.960 | 1.000 |
| valc1 | 0.925 | 0.925 | 0.980 | 1.000 | 0.980 | 1.000 |
| vis | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

A perfect formula filter is worth +0.011 to +0.015; a perfect isomer pick +0.37 to +0.41. Isotope patterns and other
formula checks have nothing to gain on molecules whose truth is in the pool.

### 2.3 Strata (val1 fp-honest, FPNet A)

| Stratum | n | MRR | top-1 | lost: cut | lost: isomers |
|---|---|---|---|---|---|
| has an [M+H]+ or [M+Na]+ spectrum (ICE/GL can score) | 216 | 0.614 | 0.426 | 0.042 | 0.332 |
| none (ICE/GL cannot score) | 34 | 0.328 | 0.088 | 0.029 | 0.641 |
| positive spectra only | 102 | 0.670 | 0.510 | 0.039 | 0.277 |
| both polarities | 130 | 0.531 | 0.315 | 0.046 | 0.414 |
| negative only | 18 | 0.361 | 0.111 | 0.000 | 0.637 |
| mass < 250 | 46 | 0.680 | 0.522 | 0.022 | 0.292 |
| 250 to 350 | 76 | 0.590 | 0.382 | 0.039 | 0.366 |
| 350 to 450 | 67 | 0.629 | 0.433 | 0.015 | 0.335 |
| 450 to 600 | 33 | 0.418 | 0.212 | 0.121 | 0.451 |
| > 600 | 28 | 0.420 | 0.214 | 0.036 | 0.534 |
| 1 spectrum | 30 | 0.679 | 0.533 | 0.033 | 0.287 |
| 2 to 3 | 75 | 0.599 | 0.440 | 0.067 | 0.326 |
| 4 to 6 | 83 | 0.566 | 0.373 | 0.024 | 0.385 |
| 7 or more | 62 | 0.507 | 0.242 | 0.032 | 0.460 |
| same-formula candidates in base top-60: 1 | 7 | 0.357 | 0.143 | 0 | 0.286 (plus 0.357 wrong formula) |
| 2 to 5 | 11 | 0.788 | 0.636 | 0 | 0.212 |
| 6 to 15 | 25 | 0.710 | 0.560 | 0 | 0.287 |
| 16 to 40 | 51 | 0.653 | 0.490 | 0.039 | 0.307 |
| 41 to 60 | 156 | 0.523 | 0.308 | 0.051 | 0.426 |

- **Adduct coverage.** 34 of 250 molecules (13.6%) have no [M+H]+ / [M+Na]+ spectrum: 13 [M-H]- only, 7 [M+NH4]+ only,
  14 mixtures of [M-H]-, formate and [M+NH4]+ (16 of the 34 have an [M+NH4]+ spectrum, 18 are negative only). Of all
  enveda-np-examples spectra, 54.9% are [M+H]+ or [M+Na]+; the forward models never look at the other 45%. Uncovered
  molecules are larger (median 437 Da against 337 Da), but within each size class the gap stays: 0.363 against 0.653
  at or below 450 Da, 0.289 against 0.465 above. Size-standardised gap: 0.26.
- **Size.** Above 450 Da (24% of molecules) MRR is 0.42 against 0.63. Both isomer signals fade: the ICE + GL pick is
  right for 68% / 63% / 35% / 29% of groups at ≤ 300 / 300 to 450 / 450 to 600 / > 600 Da, FPNet A for 46% / 44% / 38% / 22%.
- **Number of spectra.** More spectra go with lower MRR here. This is confounded: molecules with 7 or more spectra
  are heavier (median 399 Da against 288 Da) and sit in larger formula groups. On valc1 the direction is reversed
  (0.89 for 1 spectrum, 0.965 for 7 or more). No conclusion.
- **Same-formula candidates.** 62% of molecules have 41 or more same-formula candidates in the base top-60 (the list is
  mostly isomers); they score 0.52 against 0.66 for the rest.

### 2.4 How good each signal is inside the truth's formula group

Group = the candidates of the base top-60 with the truth's formula (key-deduplicated). "top-1" = the signal's best
member is the truth. Random is 1 / group size.

| Signal | val1 top-1 / MRR (243 groups, mean size 43, random 0.050) | valc1 (248, size 48, random 0.033) | vis (396, size 39, random 0.054) | Trust |
|---|---|---|---|---|
| v1 ranker score | 0.934 / 0.962 | 0.798 / 0.880 | 1.000 / 1.000 | leaked in all three |
| FPNet A, never saw the molecules | **0.416 / 0.567** | n/a | n/a | honest; the best stand-in for the ranker on hidden class 2 |
| fold-safe FPNet (public 5 h checkpoint) | 0.276 / 0.452 | n/a | n/a | honest |
| ICEBERG | 0.502 / 0.626 | 0.421 / 0.556 | 0.643 / 0.758 | val1/valc1 optimistic (seen in MassSpecGym); vis honest but drug-like |
| GLACIER | 0.517 / 0.644 | 0.425 / 0.553 | 0.684 / 0.794 | same |
| ICE + GL (z-sum) | 0.589 / 0.694 | 0.519 / 0.634 | 0.689 / 0.795 | same |
| ICE + GL, generated candidates removed from the group | 0.628 / 0.733 | 0.623 / 0.732 | 0.810 / 0.883 | same |
| frag, ICE-covered molecules | 0.177 / 0.310 | 0.079 / 0.207 | 0.351 / 0.529 | honest (untrained) |
| frag, ICE-uncovered molecules (the "gap" term) | **0.029 / 0.162** (n 34, random 0.044) | 0.000 / 0.103 (random 0.022) | 0.138 / 0.391 (n 29, random 0.067) | honest |
| popularity (pool members) | 0.733 / 0.840 | 0.733 / 0.842 | **0.031 / 0.156** (random 0.080) | biased: famous natural products; below random on synthetic library compounds |
| FPNet A + ICE + GL | 0.646 / 0.746 | | | mixed |
| FPNet A + 0.5 (ICE + GL) + 0.25 pop | 0.847 / 0.900 | | | popularity-driven |

Further facts from the same tables (`signals_deep.py`, `combiner_variants.py`):
- **Adding frag to ICE + GL lowers the pick rate** on natural products (val1 0.589 → 0.517, valc1 0.519 → 0.453 at
  weight 1) and is neutral on vis (0.689 → 0.684). This is consistent with v3b (frag for all molecules) losing on the LB.
- **Agreement is a strong confidence signal.** When ICEBERG and GLACIER name the same best member (53% of val1 groups,
  40% valc1, 62% vis) it is the truth in 80% / 78% / 90% of cases; when they disagree the z-sum pick is right in
  35% / 34% / 35%. By quartile of the z-margin between the best and second member: 32% / 48% / 60% / 96% (val1) and
  32% / 59% / 88% / 98% (vis). The second set is honest (vis molecules are not in MassSpecGym).
- **A learned combiner over ICE, GL and frag does not beat the z-sum** at this sample size. LambdaRank with raw scores
  and context (mass, group size, spectra count), generated members removed: vis 0.830 against 0.810 (5-fold CV),
  val1 0.604 against 0.628, valc1 0.609 against 0.623; trained on vis and applied to val1 0.517 against 0.628.
  With the "generated" flag as a feature it jumps to 0.84 on vis: a regime leak (the truth is never a generated
  structure in these runs), not a finding.

### 2.5 What each channel adds or costs (paired against v4b)

| Change | val1 plain | val1 sim60 | val1 fp-honest (A) | valc1 |
|---|---|---|---|---|
| no engine-2 fusion | +0.322 ± 0.020 | +0.100 ± 0.015 | **+0.179 ± 0.019** (127 wins / 15 losses) | +0.017 ± 0.008 |
| no PubChem channel | +0.005 | +0.013 ± 0.006 | +0.000 ± 0.007 | −0.001 |
| no ICE / GL | −0.081 | −0.237 | −0.145 ± 0.019 | **+0.037 ± 0.011** |
| no popularity | −0.010 | −0.079 | −0.041 ± 0.011 | −0.050 ± 0.011 |
| no frag gap | +0.008 | −0.002 | −0.000 ± 0.004 | +0.005 |
| no generated candidates | −0.021 ± 0.008 | +0.049 ± 0.008 | +0.003 ± 0.011 | −0.002 |

- **Engine-2.** Its lists hold the truth for 7.6% of val1 molecules, although its runtime pool contains 499,133 COCONUT
  structures and 249 of the 250 answers are in COCONUT: it cannot rank a structure without reference spectra. The RRF
  then adds 0.6 / (3 + r) to its favourites, so a truth it does not list is overtaken whenever engine-2's first
  choices are also in the v1 top few. 10 truths leave the top 25 this way. On the LB engine-2 is worth +0.015
  (v4n 0.384 → 0.399), so its gain on library molecules outweighs this cost. A gate keeps both (section 3 f).
- **ICE / GL on class-1-like molecules still costs 0.037** on valc1 because only 26% of them pass the 0.9 gate
  (median lib_max 0.86). Already known (v4d, `GATE_TAU 0.8`); listed for completeness.
- **Generated candidates** take 14% (val1), 28% (valc1) and 27% (vis) of the final 25 slots, and 7% / 16% / 29% of the
  top 5, on molecules whose truth is a pool structure. The sign of removing them depends on the regime, so the caches
  cannot price them. What they return on real class-3 molecules is not measured anywhere (section 3 d).
- **Frag gap** is neutral in every regime (±0.005), as the chance-level in-group numbers predict.

### 2.6 What is trustworthy

| Finding | Status |
|---|---|
| Truth present but mis-ordered: isomers are 88 to 91% of the measured loss, formulas 2 to 4% | holds in every regime; the absolute level is inflated (recall and formula order come from the leaked ranker) |
| FPNet in-group top-1 0.42 (A) and 0.28 (fold-safe) on natural products | honest: neither model was trained on these molecules |
| Ranker in-group top-1 0.93, valc1 0.925, every absolute MRR of val1 plain | leaked, do not use |
| ICE / GL in-group 0.59 on val1 | optimistic (245 of 250 molecules are in GNPS / MoNA / MassBank); 0.69 on vis is honest but for drug-like chemistry; no leak-free natural-product number exists yet |
| Frag at chance on ICE-uncovered molecules; frag lowers ICE + GL | honest (untrained scorer), small n (34) for the first |
| Adduct coverage: 13.6% of molecules, 45% of spectra unused | fact of the data; the 0.26 MRR gap is measured with leaked recall |
| Engine-2 recall 7.6% without reference spectra, 99.6% with | honest in direction (its own models may have seen the molecules, which could only raise the 7.6%) |
| Cost of engine-2 fusion on class-2 molecules (+0.18 when removed) | direction trustworthy, size inflated: on the hidden test the truth is not always in the v1 list |
| Popularity: 0.73 alone on val1, below random on vis | both real; val1 is the favourable extreme, so every popularity gain here is an upper bound |
| Agreement / margin precision of ICE and GL | vis honest, val1 optimistic, same shape in both |
| Anything about PubChem-only answers and class 3 | not measurable: no cache contains such a molecule |

### 2.7 The part the caches cannot see: a budget for the hidden test

Regime scores: class-1-like 0.92 (valc1, optimistic), class-2 with the truth in the pool 0.54 to 0.58 (fp-honest,
optimistic). LB: 0.409. Two readings of the class mix:

| Assumption | Class 1 | Class 2, truth in the pool and ranked | Molecules scoring about 0 |
|---|---|---|---|
| class 1 = 16% (forum figure quoted in `docs/landscape/ideas-2026-09-30.md`) | 16% × 0.92 = 0.147 | 0.262 / 0.575 = 46% | **38%** |
| class 1 = 33% (engine-2 alone: 0.915 on valc1, 0.058 on val1; its earlier solo version is the forum's 0.339 solution) | 33% × 0.92 = 0.304 | 0.105 / 0.5 = 21% | **46%** |

The second reading fits the sizes of two LB effects better than the first. Popularity 0 → 0.25 gave +0.010 on the LB
(v1 → v4b; the gate and the frag gap were neutral): 21% × 0.041 = +0.009 against 46% × 0.041 = +0.019. ICE + GL
together are worth roughly +0.01 to +0.02 in the public lineage (v3 0.358 → v4l 0.380 with other changes mixed in):
21% × 0.145 = +0.030 against 46% × 0.145 = +0.067. It also implies that engine-2's net +0.015 is a gain of about 0.16
MRR per class-1 molecule minus a loss of about 0.18 per class-2 molecule, which is what the gate in section 3 (f)
exploits. Either way, **the largest single loss is molecules whose truth is not ranked at all** (novel structures,
PubChem-only answers that can at best enter at slot 2 or 4, answers outside the base top-60), and no local data
measures it. This is an inference from a handful of numbers, not a measurement.

## 3. Part 2: directions

Effort is working time on this machine. Gains are public-LB MRR and rough; one LB molecule is 0.0025 to 0.008 depending
on which third is public, and a paired LB difference has a standard error near 0.01.

### (a) Train our own fingerprint model

What was done (`train_fpnet_win.py`, a port of the public `train_fpnet.py`): the public loop needs Linux `fork`, the
public `casmi` package (`traindata.py`, `build.py`, `spectra.py`), `pool_fp.npy`, `train_structs.parquet` and
`split.parquet`. With threads instead of `fork` it runs natively on Windows.
- Spectrum cache from `train.parquet`: 60 s (2,539,608 spectra, 133 M peaks, 1.1 GB).
- Public `fpnet_fold0.pt` re-scored locally (15 s): hold-out top-1 67%, natural-product panel 35%, as its README says.
- Training speed: batch 64 → 13.4 it/s = 860 spectra/s, 3.8 GB peak; batch 128 → 4.6 it/s = 590 spectra/s with the GPU
  shared (5.9 GB peak). Batch 256 (the public setting) did not finish 50 steps in two attempts while other applications
  held 3 to 8 GB of the card; it needs about 10 GB, so an idle GPU. The data threads are not the limit (1 to 2% wait).
  The public run made 1,180 spectra/s on an RTX 6000, so its 5-hour schedule (21 M samples) is about 6 to 7 h here.
- Drop-in: yes. `fpnet.ModelBank` takes a list of paths and `save_ckpt` writes what `load_model` reads. Needed: the
  checkpoint as a Kaggle dataset and one more `FP_BANK` branch in `nb/05_v1_engine.py` (as for `fold0`). The PubChem
  channel has its own bank (FPNet A + B in `pc_runner.py`) and would need the same.

| Fold-safe checkpoint, hold-out queries | top-1 | MRR@25 |
|---|---|---|
| all | 0.672 | 0.785 |
| timsTOF spectra | 0.705 | 0.814 |
| other instruments | 0.619 | 0.738 |
| NP-catalogue structures | 0.393 | 0.547 |
| NP-catalogue, positive / negative | 0.411 / 0.363 | 0.561 / 0.529 |
| not in an NP catalogue | 0.699 | 0.810 |
| enveda-np-examples (timsTOF), positive / negative | 0.375 / 0.304 | 0.555 / 0.482 |

Assessment.
- The weakness is natural-product chemistry (39% against 70%), not the instrument: timsTOF queries are the easier
  ones. Only 300 NP-catalogue structures have timsTOF spectra in train (250 of them are enveda-np-examples), so a
  "timsTOF-focused" model has nothing to focus on. `--tims_p` in the port is there to test it; I expect no gain.
- The public recipe is clearly weaker than the checkpoints already in the pipeline: 27.6% against 41.6% in-group top-1
  on the same unseen molecules. Averaging a weaker member into the bank is unlikely to help. LB history agrees that the
  bank is not the lever: A → full1 moved 0.380 → 0.378.
- Effort: 1 day to reproduce, 2 to 4 GPU-days to approach A (longer schedule, two seeds). Expected gain as a bank member:
  0 to +0.005. Risk: low. **Its real use is as a tool**: a fold-safe checkpoint of A's quality is what direction (b)
  needs to produce leak-free rows.
- Validation before any submission: the 11-subset hold-out table above, then `honest_fp_regime.py --ckpt <ours>`
  (in-group top-1 on val1; target above 0.42).

### (b) A learned re-ranker over the channel signals, trained on leak-free data

What is leak-free. Every learned component has seen all of train except: FPNet A / B (not enveda-np-examples), the
fold-safe FPNet (not fold 0 and not the 265 NP structures), ICEBERG / GLACIER (not what is outside MassSpecGym), frag
(untrained), popularity (no spectra). The v1 LightGBM ranker, the fe_v4 models, engine-2's fingerprint models and both
engine-2 rankers have unknown rosters; treat them as leaked. So rows are leak-free only if the isomer evidence comes
from a fold-safe FP bank and the leaked ranker score is not a feature.

How much data.

| Set | Structures | Notes |
|---|---|---|
| fold-0 hold-out, in the pool | 55,102 | 36,583 with timsTOF spectra (almost all enveda-180, synthetic) |
| of these, NP-catalogue | 5,433 | median popularity 5.35 ("ordinary"); only 9 have timsTOF spectra |
| NP-catalogue and not in GNPS / MoNA / MassBank | 1,309 | also unseen by ICE / GL |
| no PubChem record, natural-product libraries | 2,034 (10,496 in all of train) | class-3-like |
| enveda-np-examples | 250 | the only test-like instrument and chemistry; unseen by A and fold-0 |

How to build it. (1) Run the v1 engine locally with `FP_BANK` = fold-safe on the hold-out structures, library purged
(class 2), and with `drop_pid` (truth removed from the pool: PubChem-only and class-3 simulation). The engine cells use
no `fork`; the FPNet, `casmi` and `derive` code ran natively here, so a native Windows run is plausible but not yet
verified. Kaggle timing was 5 s per molecule (1,242 s for 250). (2) Add ICE / GL scores locally (their runners are pure
PyTorch with a DGL shim; Kaggle throughput was 40 and 83 predictions/s). (3) Class-1 rows can only come from the 250
enveda-np-examples with bank A and `VAL_PURGE='lib'`. (4) Check the residual leak first: on the hold-out the ranker's
in-group top-1 should fall from 0.93 to the FPNet level (0.3 to 0.4); if it does not, the fe_v4 models leak and their
features must be dropped. Engine-2 (12.5 s per molecule) and the PubChem channel (9 s) are only needed for the gate.

Evidence so far. For: confidence features are strongly predictive (agreement precision 0.80 to 0.90 against 0.35;
margin quartiles 0.32 to 0.96), and the channel that should be gated is known (engine-2 by library evidence, ICE / GL
by `lib_max`, popularity by chemistry: 0.73 on natural products, below random on synthetic compounds). Against: at
n = 200 to 370 a learned combiner of ICE, GL and frag does not beat the z-sum and does not transfer between vis and
val1. A re-ranker needs thousands of rows and features that describe the regime, not more capacity.

- Effort: 3 to 5 days (1 to 2 for the local runs). Expected gain +0.01 to +0.03, most of it from gating.
- Risk: medium to high. Training molecules are library compounds measured mostly on other instruments; popularity and
  library-evidence features differ from the hidden test; residual leak through the ranker and fe models.
- Validation before a submission: grouped CV on the hold-out rows, then two frozen tests that the model never trained
  on, val1 fp-honest and valc1, plus the acceptance rule already in `EXPERIMENTS.md` (reproduce the order of the LB points).

### (c) Better isomer separation

- **`frag_rescore.py`.** In-group top-1 0.18 (natural products, positive mode) and 0.35 (vis), at chance in the gap
  molecules, and it lowers ICE + GL when added. Tuning it further is not supported by anything here. The gap term can
  stay or go (±0.000).
- **More adducts for the forward models.** 16 of the 34 uncovered molecules have an [M+NH4]+ spectrum. Ammonium adducts
  mostly lose NH3 and then fragment like [M+H]+, and [M-H2O+H]+ is an [M+H]+ fragment, so scoring these spectra against
  the [M+H]+ prediction (precursor region masked) is a plausible surrogate. The checkpoints are MassSpecGym-only, so
  other adduct embeddings are untrained. Test first on the enveda-np-examples molecules that have both an [M+H]+ and an
  [M+NH4]+ spectrum: in-group top-1 of the surrogate against the native score. Effort 1 to 2 days once ICE runs locally.
- **Negative mode.** 18 of 250 molecules are negative only, and FPNet itself is weaker there (0.36 against 0.41 top-1).
  No MassSpecGym forward model covers it. FIORA (disabled branch in the e5c notebook) is the candidate; its weights must
  be confirmed to be trained on open data only before any use. Effort 2 to 3 days, risk medium (new dependency, eligibility).
- **Isotope pattern, formula checks.** Bounded by +0.011 to +0.015 where the truth is in the pool; skip.
- **Agreement-weighted re-scoring** (use the ICE / GL term at full weight only when both agree or the margin is
  large) belongs to (b) as a feature and can be tried as a rule in a day.
- Expected gain: coverage is 13.6% of molecules × 0.26 MRR × the class-2 share (21 to 46%) = at most +0.007 to +0.016
  if fully closed; realistic +0.003 to +0.01. Risk low for the surrogate, medium for negative mode.

### (d) Class-3 molecules (not in PubChem)

Today: only the v1 engine proposes structures outside its pool. `casmi/derive.py` applies 1 or 2 of about 50
biosynthetic transformations (hydroxylation, methylation, acylation, glycosylation and their reversals) to the 6
strongest spectral analogs, keeps products within the mass tolerance, at most 60 per parent and 150 in total, and flags
them for the ranker. Engine-2 and the PubChem channel cannot produce a novel structure. Generated candidates hold 14 to
28% of the final slots on molecules that are not class 3.

Measured here (`class3_reach.py`): with an **oracle** analog search (parents = the training structures most similar to
the truth by Morgan Tanimoto, the truth itself excluded), the transformation set reaches

| Targets (n = 300 each) | reach, engine limits (6 parents, 150 products) | reach, 20 parents, no cap | products per molecule | median Tanimoto of the nearest parent |
|---|---|---|---|---|
| training structures without a PubChem record, natural-product libraries (10,496 exist) | 33% | 49% | 28 / 109 | 0.82 |
| PubChem-known NP-catalogue structures (the usual deletion simulation) | 40% | 46% | 36 / 165 | 0.75 |

Reach by similarity of the nearest parent (class-3-like set): 8% below 0.6, 39% at 0.6 to 0.8, 61% above 0.8. So the
rule set is not the bottleneck where close relatives exist; the open questions are whether spectra find the right
parent and whether the truth is ranked among 30 to 100 positional isomers, exactly the comparison the forward models
are weakest at. Hidden novel compounds will have fewer close relatives than these (the forum's 3× overstatement).

Feasible next: (1) measure the real thing with `drop_pid` on the 2,034 hold-out structures without a PubChem record
(reach@25 and MRR of the generated channel, fold-safe bank), which also prices the slots the generator takes;
(2) if reach@25 is above about 10%, rank generated candidates with the same in-group machinery and give them slots by
confidence instead of by ranker score. Generative spectrum-to-structure models are not worth their integration cost
now (published top-10 accuracy is a few percent with the formula known).
- Effort: 1 to 2 days for the measurement (shared with (b)), 3 to 5 for changes. Expected gain 0 to +0.015. Risk high.

### (e) The `research/competition-analysis` branch

A from-scratch pipeline (library cosine search, PubChem/COCONUT catalog with a literature prior, MetFrag-lite and
analog propagation in Stage D), best LB 0.260. Nothing in its scoring path is ahead of ours. Worth taking:
- its **structure hold-out with a PubChem-absent subset** (5,200 queries, 964 without any PubChem record) and the
  250-query cross-source class-1 proxy, as fixed validation manifests for (b) and (d);
- its **PubChem 2026-09-27 catalog build** (CID, mass, InChIKey, COCONUT flag, PubMed counts) as an independent
  source for popularity arrays, since the v5 post-mortem traces our loss to a substituted popularity dataset;
- its measurement that 74% of held-out answers sit inside a ±10 ppm window that holds a median of 5,510 structures:
  the reason the PubChem channel needs a gate, not more depth.
Not worth merging: the library search, the fusion rule (cosine ≥ 0.7 first), the adduct parser. Effort: half a day.

### (f) Other findings from the caches

1. **Engine-2 library gate** (new knob, for example `FUSE_MIN_LIB`: fuse engine-2 only if `lib_max >= tau`).

| tau | class-2 side: molecules gated | Δ MRR, val1 fp-honest | class-1 side: valc1 molecules gated | engine-2 alone on those |
|---|---|---|---|---|
| 0.4 | 26% | +0.067 ± 0.012 | 1.6% | 0.56 |
| 0.5 | 38% | +0.083 ± 0.013 | 2.4% | 0.71 |
| 0.6 | 51% | +0.093 ± 0.015 | 3.2% | 0.78 |
| 0.7 | 67% | +0.113 ± 0.016 | 9.6% | 0.72 |
| 0.8 | 83% | +0.133 ± 0.018 | 22.4% | 0.78 |
| off entirely | 100% | +0.179 ± 0.019 | 100% | 0.92 |

   At 0.5 to 0.6 the gate removes engine-2 from half of the class-2-like molecules and from 2 to 3% of the class-1-like
   ones. Scaled by a class-2 share of 21 to 46% and halved for the inflated regime: **+0.01 to +0.02**. The unknown is
   whether engine-2 finds answers on the hidden test without library evidence; the caches say it does not (0.039 alone
   below `lib_max` 0.5 on val1). Offline it is validated on the cost side only, so it needs one submission, and that
   submission also tells which reading of section 2.7 is right. A sharper gate needs engine-2's own library score in
   `eng.json` (it is not dumped today).
2. **fp-honest as a harness regime.** `honest_fp_regime.reorder_by` is a drop-in sibling of `shuffle_isomers`; the
   FPNet scores for val1 are cached (`out/fz_A_val1.json`). It ranks the LB points better than plain and gives smaller,
   more plausible effect sizes than the shuffle.
3. **The sim's cut artefact** (section 1): cut to `TOPN` before shuffling.
4. **Popularity is a chemistry-dependent prior**: 0.73 top-1 alone on famous natural products, 0.03 (below random) on
   enveda-180 compounds. A weight that depends on the candidate group (for example the share of NP-catalogue members)
   is a (b) feature; raising `POP_MU` globally is not supported by anything honest here.

## 4. Ranking

| # | Direction | Effort | Expected LB gain | Risk | Validated offline by |
|---|---|---|---|---|---|
| 1 | (f1) + (b): regime-aware gating, first the engine-2 library gate, then the learned re-ranker on leak-reduced rows | 2 h for the gate, 3 to 5 days for (b) | +0.01 to +0.02, then up to +0.03 | medium | fp-honest and valc1 for the gate (cost side); grouped CV + two frozen regimes for (b) |
| 2 | (d) with the `drop_pid` measurement: what happens when the truth is not in the pool | 1 to 2 days to measure, 3 to 5 to change | 0 to +0.015, on the largest bucket | high | reach@25 and MRR on 2,034 PubChem-absent hold-out structures |
| 3 | (c) forward-model coverage: [M+NH4]+ / water-loss surrogate, then negative mode | 2 to 4 days | +0.003 to +0.01 | low to medium | in-group top-1 on molecules with both adduct types |
| 4 | (a) own FPNet | 2 to 4 GPU-days | 0 to +0.005 directly | low | hold-out table, in-group top-1 on val1 |
| 5 | (e) merge validation manifests and the catalog build | 0.5 day | 0 (enabler) | low | n/a |
| – | frag tuning, isotope checks, more popularity, generative models | | about 0 | | |

**Start now:**
1. **Gating, growing into (b).** Add the engine-2 library gate and the fp-honest regime to the harness, dump engine-2's
   own library score, and submit the gate at 0.5 or 0.6 when a submission is free. In parallel build the leak-reduced
   rows. This is the only direction where the caches show a large, mechanistically clear effect.
2. **The local v1-engine run with a fold-safe bank and `drop_pid`.** It is the prerequisite of both (b) and (d), needs
   no Kaggle GPU, and turns the largest loss bucket from an inference into a number. (a) enters here as a tool: a
   fold-safe checkpoint at FPNet A's quality.

## 5. Open points and caveats

- No leak-free estimate of ICE / GL on natural products exists. The 1,309 hold-out NP-catalogue structures outside
  GNPS / MoNA / MassBank would give one once ICE / GL run locally.
- The class mix of the hidden test is unknown; section 2.7 is an inference. The 16% figure is a forum claim.
- FPNet A's training roster is not published. Its 42% against the ranker's 93% on these molecules agrees with the
  dataset card (trained without enveda-np-examples), but it is not proof.
- `class3_reach.py` matches products on the standard InChIKey first block, not the metric key (the tautomer
  canonicalisation costs up to 0.3 s per large product), so a few hits are missed.
- The native Windows run of the v1 engine is untested. Only FPNet training and scoring, `derive` and the spectrum
  cache were run.
- `INDEX.md` was not regenerated (this study does not modify tracked files): run `python tools/gen_index.py`.

## 6. Reproduce

```bash
export CASMI_DATA='D:\casmi-data'; P=.venv/Scripts/python; S=scratch/directions
$P $S/error_analysis.py          # sections 2.1 to 2.4 for plain runs (first run canonicalises keys: about 15 min)
$P $S/sim60_analysis.py          # sim60 decomposition and strata
$P $S/fusion_ablate.py           # section 2.5 (plain, sim60, valc1, vis)
$P $S/signals_deep.py && $P $S/combiner_variants.py    # agreement, margins, learned combiner
$P $S/gen_ablate.py              # generated candidates
$P $S/honest_fp_regime.py --ckpt 'D:\casmi-data\directions\dl\fpnet_0.pt' --tag A     # fp-honest regime (also --tag fold0)
$P $S/calib_fphonest.py && $P $S/eng2_gate.py          # LB calibration, engine-2 gate table
$P $S/class3_reach.py --n 300    # oracle reach of the generator
# FPNet (run dir D:\casmi-data\directions\run, casmi package in D:\casmi-data\directions\code)
$P $S/train_fpnet_win.py --code 'D:\casmi-data\directions\code' --run 'D:\casmi-data\directions\run' --eval_only <ckpt.pt>
$P $S/train_fpnet_win.py --code ... --run ... --bs 64 --max_steps 150 --eval_groups 100    # speed check
```
Downloaded with the Kaggle CLI into `D:\casmi-data\directions\dl`: `pool_fp.npy`, `fp_bits.npy`,
`train_structs.parquet` (ahmedberatozer/casmi26-v2-pool), `fpnet_fold0.pt` (dmitriigluzdov/casmi26-fold-safe-fpnet),
`fpnet_0.pt` (ahmedberatozer/casmi26-v3-models); code files `traindata.py`, `build.py`, `spectra.py`, `derive.py`,
`engine.py` from ahmedberatozer/casmi26-v4b-models and `train_fpnet.py` from the fold-safe dataset (Apache-2.0).
Checkpoints were loaded with `weights_only=True`.
