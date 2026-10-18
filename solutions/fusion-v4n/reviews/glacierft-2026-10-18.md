<!-- index: 2026-10-18 GLACIER fine-tuned with the official ms-pred model / loss on Enveda-180 timsTOF + general [M+H]+ spectra (intensity head only, scaffold-disjoint split, early stopping on validation isomer MRR): held-out timsTOF isomer MRR +0.008 (CI spans 0), MassSpecGym in / out split, no memorisation tilt, stack1-like replay +0.002 (fails); fails its pre-registration; no Kaggle candidate -->
# GLACIER fine-tuned on timsTOF with the official ms-pred code (2026-10-18)

Marks: **[V]** computed here (file named); **[C]** cited; **[I]** inference. Isomer RR = 1 / (1 + decoys above the truth + 0.5 ties)
inside the truth's formula group, score = stack1's runner (`gl_runner.py`, unchanged, `--ckpt`) mean entropy similarity over the
molecule's [M+H]+ timsTOF spectra. Pre-registration: `scratch/glacierft/PREREG.md` (+ addendum 1), written before any training step
and any outcome. Nothing pushed, submitted, downloaded or committed; the visible `test.parquet` was not read; no NIST data.
GPU: 0.92 GPU-h training + 0.27 GPU-h scoring, all through `guard_run.py` with `GPU_LOCK` (lock released between 60-min chunks).

## 0. Bottom line
- **Fails the pre-registration.** Held-out timsTOF isomer MRR (H, 443 molecules not in MassSpecGym and never trained on): base 0.8915
  → fine-tuned 0.8998, **+0.0084 ± 0.0078, 95% CI [−0.0070, +0.0238]** (32 better / 25 worse). PASS-1 needed CI > 0.
- **Replay (secondary) fails too:** stack1-like lists of the 266 timsTOF molecules, composite **+0.0016 [−0.0031, +0.0063] (m),
  +0.0020 [−0.0036, +0.0079] (u)**; panel pool lists (c1 + c2) +0.0006 ± 0.0035 (12 / 14); 4 answers gained / 3 lost at rank 1 on
  452 panel pool lists. Size criterion (+0.0133 composite = one public answer) missed by ~7x.
- The fine-tune does learn timsTOF intensities (truth similarity 0.536 → 0.596 on H) but decoys rise with it: the gain is a level
  shift, not better isomer separation.
- **No memorisation tilt** (the v20a / v24a-fmgl worry): decoys that are training structures do not gain on unseen answers (tilt
  +0.0005 [−0.0040, +0.0049]).
- No Kaggle candidate. Integration would have been trivial (same 60.5 MB `{hyper_parameters, state_dict}` file, `gl_runner.py --ckpt`,
  same architecture, T4 runtime unchanged).

## 1. Set-up [V `train/glacierft/`, `casmi-work/glacierft/data/split.json`]
- **Code**: ms-pred @ `708148c2a8eb` (MIT) imported unchanged: `JointModel`, `_common_step(batch, 'train')` (cosine intensity loss +
  Sinkhorn rank-1 contrastive loss vs decoys), `configure_optimizers()`, `TreeProcessor`, `IntenContrDataset.collate_fn`, `mutate()`,
  `process_spec_file` / `max_inten_spec` target processing. No installs: pygmtools stand-in (ours, tested against a numpy reference),
  the casmi26-glacier dgl / torch_scatter / LinSATNet stand-ins, pytorch_lightning from the iceberg wheel site.
- **Recipe** (fixed in advance): official MassSpecGym checkpoint; **intensity head only** (6.73 M of 15.1 M parameters; Graphormer and
  fragment generator frozen, fragment loss off, so no MAGMa labels); lr 5e-5, warm-up 500, batch 16, 5 of 20 decoys (half `mutate`,
  half same-formula pool), contr_threshold 0.4, sk_tau 0.05; samples 67% Enveda-180 timsTOF / 33% other libraries' Orbitrap / Q-TOF;
  conditioning exactly as the runner (5-eV CE bucket, timsTOF → QTOF token). Early stopping on V isomer MRR, patience 3.
- **Split** (generic Murcko scaffolds, keys = tautomer-canonical IK14): R = all 606 replay molecules; **H1** = 43 replay timsTOF molecules
  not in MassSpecGym (stack1's own candidate groups, median 39); **H2** = 400 Enveda-180 timsTOF molecules not in MassSpecGym, groups =
  truth + 40 nearest same-formula pool structures (engine-like hard decoys); **V** = 200 more (scaffold-disjoint from H / R). Training
  excludes every evaluation key and scaffold (98k structures): 377,650 timsTOF spectra / 107,932 structures + 286,138 general spectra /
  32,544 structures. Decoys of H groups stay in training unless they share a key or scaffold (6,080 of 12,393 H decoys are training
  structures), as on Kaggle.
- **Training** [V `run_ft1/summary.json`]: V 0.9193 (base) → 0.9316 (1k steps) → 0.9349 (4k, selected) → 0.926-0.931 (5k-7k); stopped
  by patience at 7,000 steps (112k samples), 0.92 GPU-h. `best.pt` sha256 `02b326df…`.

## 2. Primary [V `casmi-work/glacierft/primary.md`, `primary.json`]
| Set | n | base | fine-tuned | delta ± SE [95% CI] | better / worse | truth sim base → ft |
|---|---|---|---|---|---|---|
| **H = H1 + H2** | 443 | 0.8915 | 0.8998 | **+0.0084 ± 0.0078 [−0.0070, +0.0238]** | 32 / 25 | 0.536 → 0.596 |
| H1 (stack1 groups) | 43 | 0.6678 | 0.6889 | +0.0211 ± 0.0309 [−0.0388, +0.0804] | 10 / 6 | 0.551 → 0.600 |
| H2 (Enveda-180) | 400 | 0.9155 | 0.9225 | +0.0070 ± 0.0079 [−0.0080, +0.0225] | 22 / 19 | 0.535 → 0.596 |
| V (early stopping, selection-biased) | 200 | 0.9193 | 0.9349 | +0.0156 | | |

Verdict: KILL-1 not triggered (V +0.0156 ≥ 0.005); PASS-1 (a) **fails** (CI spans 0), (b) holds, (c) no tilt. **FAIL.**

**MassSpecGym in / out** (requested by the main session; same runner scores, stack1's groups of the 228 replay timsTOF molecules):

| Molecules | n | base | fine-tuned | delta ± SE |
|---|---|---|---|---|
| panel, **in** MassSpecGym | 185 | 0.5655 | 0.5970 | +0.0315 ± 0.0184 |
| panel, **not** in MassSpecGym | 8 | 0.2895 | 0.3341 | +0.0446 ± 0.0687 |
| Enveda-180 replay, not in MassSpecGym | 35 | 0.7543 | 0.7700 | +0.0157 ± 0.0349 |
| H2 Enveda-180, not in MassSpecGym | 400 | 0.9155 | 0.9225 | +0.0070 ± 0.0079 |

Readings [I]: (1) the only natural products base GLACIER has not seen (8 panel molecules) sit at 0.29 against 0.57 for the 185 it has
seen in MassSpecGym; with n = 8 this is indicative only, but it is in line with the isoxe finding that GLACIER's panel lead is largely
training overlap. (2) The Enveda-180 not-in-MSG molecules are drug-like and easy for GLACIER (0.75 stack1 groups, 0.92 engine-like
groups), so they are not a good proxy for hidden natural products. (3) The fine-tune's larger in-group gain on panel-in-MSG (+0.03) does not
reach stack1's lists (replay below): the group metric scores all candidates with the truth's formula, the replay only re-orders the 40-row
window, where E2R2 and the other channels decide most first rows.

## 3. Secondary: stack1-like replay [V `casmi-work/glacierft/replay.md`]
Cell 34's GLACIER branch rebuilt for 1,340 / 1,340 lists (all reproduce F); base = same-job rerun of the base checkpoint (== msfwd's B' in
1,542 / 1,552 lists). FT vs base, timsTOF molecules (266):

| arm | touched (w / l) | composite [95% CI] | c1 / c2 / dA / dB | panel pool c1 + c2 | worst case | criteria failed |
|---|---|---|---|---|---|---|
| m | 876 (47 / 45) | +0.0016 [−0.0031, +0.0063] | +0.0022 / +0.0011 / −0.0011 / +0.0043 | +0.0006 ± 0.0035 (12 / 14) | 0.68 RR | c2, c3, c5, c7 |
| u | 876 (42 / 41) | +0.0020 [−0.0036, +0.0079] | +0.0022 / +0.0011 / +0.0014 / +0.0035 | same | 0.80 RR | c2, c3, c5, c7 |

Answers at rank 1 on panel pool lists: +4 / −3 (of 452); Enveda-180 pool lists +1 / −0 (of 40). Worst case about 0.7-0.8 public RR lost.

## 4. Kaggle integration cost (not pursued)
Same architecture and file layout: replace `ckpt/glacier.pt` (60.5 MB → 60.5 MB) or pass `--ckpt`; T4 runtime unchanged (identical
forward pass). Compliance would be clean (official code, permitted data, our training code delivered here). Nothing to integrate.

## 5. Why it does not help, and next step [I]
- Head-only fine-tuning moves every candidate's predicted spectrum towards timsTOF-like intensities; that lifts the truth's similarity
  (+0.06) and the decoys' about equally, so the order changes little. Isomer order depends on *which fragments* form, which the frozen
  fragment generator (and the MassSpecGym chemistry it learned) decides.
- The honest proxy for hidden natural products (panel molecules not in MassSpecGym) has n = 8; no local set can show a reliable effect.
- Not recommended: a full (encoder + fragment) fine-tune with MAGMa labels (4-15 CPU-h of labels, unverified LinSAT stand-in in the loss)
  for an expected effect that this head-only run bounds at under one public answer on the replay. If anything, a larger pool of unseen
  timsTOF natural products (not in MassSpecGym) is the missing evaluation resource, not another forward-model variant.

## 6. Files
Code (repo): `solutions/fusion-v4n/train/glacierft/` (env, shim, build_data, data, train_glacierft, README),
`solutions/fusion-v4n/tests/test_glacierft.py`. Glue (git-ignored): `scratch/glacierft/` (PREREG, STATE, r_meta, run_gpu, eval_runner,
primary, replay_eval). Outputs: `C:\Users\hemre\casmi-work\glacierft\` (data/, run_ft1/best.pt + train.log + summary.json, eval/,
primary.md/json, replay.md/json). ms-pred checkout `C:\Users\hemre\casmi-work\ext\ms-pred` (only `src/` imported).
