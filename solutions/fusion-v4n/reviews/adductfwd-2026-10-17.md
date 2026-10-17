<!-- index: 2026-10-17 forward-model evidence for stack1's molecules without an [M+H]+ spectrum: exact replay of stack1's final stage (3,186 / 3,186 lists), size of the opportunity (segments, MRR, isomer-order loss, what GLACIER buys on [M+H]+ molecules, ceiling 0.5-0.9 public answers before the unseen-molecule discount), three options compared (FIORA gap term: already downloaded and killed 10-07; retraining GLACIER / ICEBERG with ms-pred on train.parquet: cost, compliance, why the two earlier fine-tunes lost; adduct-shift surrogate), the surrogate pre-registered and measured with GLACIER on the GPU (V_sur / V_neg / V_both: all fail, effect ~0 although the in-group signal is above chance); no candidate -->
# Forward evidence for molecules without [M+H]+ (2026-10-17)

Marks: **[V]** computed here (file named); **[C]** cited from our reviews; **[I]** inference or estimate. MRR = MRR@25 on the
notebook's metric identity; composite and arms m (answer's popularity masked) / u (real) as in `stack1err-2026-10-17.md`. "±" = one
SE; CI = 95% percentile bootstrap over molecules. One answer moved to rank 1 on the public part (~124 molecules) = 0.008. **No
leaderboard estimate.** No downloads, no training, nothing submitted, pushed or committed; the visible `test.parquet` was not read.
GPU: two GLACIER inference jobs (13k predictions, 3.5 min, 0.4 GB) through `guard_run.py` under the `GPU_LOCK` protocol.

## 0. Bottom line

- **The opportunity is small by arithmetic.** Molecules without an [M+H]+ spectrum are 19% of the local set (116 / 606) and 14.2% of
  the timsTOF panel (32 / 226, the closest proxy of the hidden test; the hidden adduct mix is not documented). Their MRR is 0.427 /
  0.468 against 0.600 / 0.682 for [M+H]+ molecules, but **most of that gap is not forward-model evidence**: with every forward term
  removed from the final window the two groups still differ by 0.13-0.15 (0.423 / 0.465 against 0.557 / 0.623). GLACIER adds
  +0.043 / +0.058 per [M+H]+ molecule, i.e. it recovers only 16-22% of the gain a perfect isomer order would give.
- **Ceiling** (a gap term as good as GLACIER is on [M+H]+ molecules, which have very probably been seen by GLACIER): +0.043 to +0.086
  per gap molecule, composite +0.006 / +0.012 on the panel, **0.5-0.9 public answers** after the 0.6 database scaling, about half
  of that after the unseen-molecule discount [I]. No gap-only forward term can reach one answer unless it orders isomers better
  than GLACIER does on [M+H]+.
- **Measured (pre-registered, `scratch/adductfwd/PREREG.md`)**: GLACIER on adduct-shifted surrogate spectra ([M+NH4]+ / water loss
  → [M+H]+; [M-H]- and formate / chloride → [M+H]+ by +2 protons) in stack1's GLACIER branch for gap molecules. **All three
  variants fail**: composite −0.0010 to +0.0013, every CI spans 0, expected public effect ≤ 0.09 answers. The surrogate score is
  above chance inside formula groups (+0.16 to +0.32 RR; GLACIER on native [M+H]+ +0.41 to +0.44), so the signal exists but, at
  stack1's weight, it moves as many answers down as up.
- **Options ranked**: (c) measured here, dead; (a) FIORA OS was already downloaded (with approval) and tested on 2026-10-07: at
  chance on the panel's gap molecules, every gap-term form failed the replay; nothing to download now; (b) retraining GLACIER on
  negative / other adducts is compliant but costs days plus 5-80 GPU-hours [I], is capped by the same ceiling, and the two earlier
  fine-tuned forward models lost on the leaderboard (−0.030, −0.023). **No candidate; nothing needs approval to proceed, and I do
  not recommend spending the approval on (b).**

## 1. Harness [V `scratch/adductfwd/af_lib.py`]

stack1's final stage rebuilt from the arb capture (`casmi-work/arb/Tm150/lists`): metric fusion of PREFUSION_LISTS and engine 2 (alpha
0.6, offset 3, window 40), the branch of cell 34's `rerank_row` (GLACIER z-sum where GLACIER scores exist and the ICEBERG dict is
non-empty; frag-gap MetFrag-lite where neither has a finite score; ICEBERG fallback where ICEBERG exists without GLACIER), then
`safe_extend` and cell 36's E2R2. **It reproduces stack1err's B\* on 3,186 / 3,186 lists** (glacier 2,534, frag-gap 532, ICEBERG
fallback 94, no engine-2 list 26). Two facts from the cells that matter here [V `scratch/comply/stack1_src`]: the final list's only
isomer-sensitive term is **GLACIER on [M+H]+ spectra** (`CELL19_ICE = False`, `BASE_ICE = False`; ICEBERG only re-orders the PubChem
channel and is the fallback for [M+Na]+-only molecules); molecules without [M+H]+ get MetFrag-lite (λ 0.5) or the ICEBERG fallback.

## 2. Size of the opportunity [V `af_quantify.py` → `casmi-work/adductfwd/quantify.md`]

Segments (query adducts, label-free): **H** has [M+H]+; without it **G_sur** has [M+NH4]+ / water loss, **G_na** [M+Na]+, **G_neg**
negative only, **G_oth** the rest.

| set | H | G_sur | G_na | G_neg | G_oth | without [M+H]+ |
|---|---|---|---|---|---|---|
| mixed (380) | 296 | 25 | 8 | 49 | 2 | 84 (22.1%) |
| panel, timsTOF (226) | 194 | 12 | 2 | 18 | 0 | 32 (14.2%) |

Lists per arm: B\* (stack1), U (same chain, no forward re-rank in the final window), O (perfect isomer order in the final window: the
answer first among its formula group's slots, the rest of the chain unchanged). Both sets, per molecule, arm m / u:

| segment | molecules | B\* | U | O | B\*−U (forward term now) | O−B\* (isomer-order loss) | capture (B\*−U)/(O−U) |
|---|---|---|---|---|---|---|---|
| H | 490 | 0.600 / 0.682 | 0.557 / 0.623 | 0.824 / 0.894 | +0.043 ± 0.004 / +0.058 ± 0.005 | 0.225 / 0.212 | 0.16 / 0.22 |
| G_sur | 37 | 0.416 / 0.443 | 0.421 / 0.450 | 0.732 / 0.813 | −0.005 / −0.007 | 0.316 / 0.370 | ≈ 0 |
| G_na | 10 | 0.447 / 0.507 | 0.433 / 0.483 | 0.625 / 0.700 | +0.014 / +0.024 (ICEBERG) | 0.179 / 0.193 | 0.07 / 0.11 |
| G_neg | 67 | 0.431 / 0.479 | 0.424 / 0.472 | 0.741 / 0.825 | +0.006 / +0.007 (MetFrag-lite) | 0.311 / 0.347 | ≈ 0.02 |
| **all without [M+H]+** | 116 | **0.427 / 0.468** | 0.423 / 0.465 | 0.728 / 0.811 | +0.003 / +0.004 | **0.302 / 0.343** | ≈ 0.01 |
| panel, without [M+H]+ | 32 | 0.501 / 0.561 | 0.502 / 0.570 | 0.815 / 0.965 | −0.001 / −0.009 | 0.314 / 0.403 | ≈ 0 |

Readings:
- Isomer order costs the gap molecules 0.30-0.40 RR each (H: 0.21-0.25), i.e. 0.045-0.066 of the composite. That is the prize for
  a perfect forward model, not for a realistic one.
- Today's gap terms do nothing measurable (MetFrag-lite +0.006 on G_neg, −0.005 on G_sur; ICEBERG on [M+Na]+ +0.014 / +0.024, n 10).
- **Ceiling**, a gap term with GLACIER's capture on H: per gap molecule +0.045 / +0.071 (both sets), +0.043 / +0.086 (panel);
  composite +0.0087 / +0.0136 (both sets), **+0.0061 / +0.0122 (panel share)** = 0.77 / 1.52 public answers, **0.46 / 0.91 after the
  0.6 database scaling** (29-52% of hidden answers are in no database [C coverage2]). GLACIER's capture on H is optimistic: it has
  very probably seen most local molecules, and its in-group MRR on 13 panel molecules outside its sources is about half
  [C `fiora-2026-10-07.md` D]. A realistic ceiling is therefore 0.25-0.45 public answers [I].

## 3. Option (c): adduct-shift surrogate, pre-registered and measured [V `af_fwd.py`, `af_eval.py` → `casmi-work/adductfwd/eval.md`]

Written before any score (`scratch/adductfwd/PREREG.md`, including the size pre-check of section 2). Surrogate spectra:
[M+NH4]+ → [M+H]+ (precursor −17.0265, peaks above it dropped; `nb/embed/surrogate.py`), water losses relabelled; [M-H]- → +2.014552
on precursor and peaks; formate / chloride first cut to [M-H]-; [M+Na]+ / [M+K]+ not used. GLACIER (packaged MassSpecGym
checkpoint) scored every row of the gap molecules' final windows: 103 molecules, 4,367 candidates, 0 failures. Variant: those scores
enter stack1's GLACIER branch at GL_LAM 1.0 for the gap molecules; everything else is B\*. Pass (both arms): CI lower bound > 0 and
E_pub = Δ × (panel share / local share) × 0.6 ≥ 0.008; vetoes on type cells and per-set wins / losses.

| variant | arm | touched (w / l) | first rows | composite Δ [95% CI] | E_pub (answers) | c1 / c2 / dA / dB | verdict |
|---|---|---|---|---|---|---|---|
| V_sur (37 molecules) | m | 128 (27 / 23) | 46 | +0.0004 [−0.0019, +0.0028] | 0.03 | +0.002 / −0.001 / +0.002 / +0.001 | fail |
| | u | 128 (34 / 26) | 45 | +0.0013 [−0.0016, +0.0044] | 0.09 | +0.002 / −0.001 / +0.004 / +0.003 | fail |
| V_neg (66) | m | 231 (38 / 44) | 78 | −0.0003 [−0.0041, +0.0033] | −0.02 | −0.006 (2 / 9) / +0.001 / −0.001 / +0.002 | fail |
| | u | 231 (38 / 61) | 78 | −0.0010 [−0.0051, +0.0030] | −0.05 | −0.006 / +0.001 / −0.003 / +0.001 | fail |
| V_both | m | 359 (65 / 67) | 124 | +0.0001 [−0.0043, +0.0043] | 0.01 | −0.004 / −0.001 / +0.002 / +0.002 | fail |
| | u | 359 (72 / 87) | 123 | +0.0003 [−0.0047, +0.0053] | 0.02 | −0.004 / −0.001 / +0.000 / +0.004 | fail |

In-group diagnostic (descriptive; answer's RR inside its formula group of the window by GLACIER alone, one entry per molecule and
group): G_neg surrogate 0.33 / 0.35 against chance 0.17 / 0.15 (mixed / panel; +0.16 ± 0.03, +0.20 ± 0.04); G_sur surrogate 0.20 /
0.44 against 0.13 / 0.12; native GLACIER on H molecules 0.59 / 0.59 against 0.18 / 0.15. New against `coverage-2026-10-02.md`:
GLACIER on +2 H-shifted negative spectra is above chance where ICEBERG was not. It does not survive the list: about half of
GLACIER's native in-group signal, at the full z weight, swaps answers out of rank 1 as often as into it (c1 2 / 9 in V_neg). Same
outcome as ICE_SURROGATE / GL_SURROGATE on v24a [C anatomy]. Not re-tried at other weights (pre-registered: no rescue).

## 4. The three options

| | (a) FIORA OS v1.0.0, gap term for negative-only molecules | (b) retrain GLACIER / ICEBERG on permitted adducts | (c) adduct shift with existing checkpoints |
|---|---|---|---|
| Status | **Already downloaded with approval and tested** (`fiora-2026-10-07.md`): `D:\casmi-data\input\datasets\ahmedberatozer\casmi26-fiora-os-v1` (66 MB unpacked; Kaggle 64.6 MB, `kaggle datasets download ahmedberatozer/casmi26-fiora-os-v1`), byte-identical to upstream v1.0.0. Prepared wiring in `scratch/fiora/prepared/` | not started; no training code beyond the LightningModules is local | measured here |
| Evidence | in-group on the 19 panel gap molecules 0.069 ± 0.025 vs chance 0.061 (at chance); 0.81 on unseen drug-like Enveda-180 [M-H]-; every gap-term form lost a type beyond 1 SE in the v33b replay [C] | two fine-tuned forward models lost on the leaderboard: v20a ICE_FT −0.030 (third-party `casmi26-ice-ft`, training data not documented, card non-commercial), v24a-fmgl GLACIER fine-tuned on timsTOF / NP / negative spectra −0.023 [C docs/experiments.md rows 24, 30] | fails, ~0 (section 3) |
| Compliance | MIT code and weights, MSnLib v7 (CC BY 4.0, also a train.parquet source); reproducible with the authors' `fiora-train`; no named host answer | clean if trained with the official ms-pred code on train.parquet and our training code is delivered (host, thread 748882 [C stack1-compliance]); no NIST; train.parquet is CC BY-NC (non-commercial allowed) | official checkpoints, no new artefact |
| Cost | stack1 port: FIORA on the gap windows (~5 min GPU at ~50 pred/s), replay; ~half a day | data: train.parquet has [M-H]- 465,688 spectra / 98,485 structures (65,874 enveda-180 synthetic, 21,076 MSnLib), [M+Na]+ 178,092 / 25,112, [M+NH4]+ ~53k / ~9k [V train_meta]; MAGMa labels for GLACIER's targets (CPU, ~0.2-1 s per structure: 4-15 h on 6 workers [I]); training loop around `glacier/joint_model.py` + the ms-pred repo for configs (a download, a few MB). GPU on the RTX 4070 Ti SUPER [I, from 80-150 inference pred/s here, training ~3x slower]: fine-tune of the MassSpecGym checkpoint on ~100k negative spectra × 10 epochs ≈ **5-8 GPU-h**; from scratch on all negative spectra ≈ **40-80 GPU-h**; ICEBERG (generator + intensity, two models; upstream uses two 24 GB GPUs) more | done (3.5 GPU-min) |
| Expected gain | ≈ 0 (at chance where it would act) | ≤ ceiling: 0.25-0.9 public answers even if it matched GLACIER on [M+H]+; realistically below, on natural products, given (a) and (c) | 0 |
| Main risk | — | **memorisation tilt**: a model fine-tuned on train.parquet predicts the spectra of training structures (the pool decoys) better than those of unseen hidden answers, which no local set with train-derived answers can show (v20a was submitted without a local estimate for that reason) [I] | — |

Why v20a lost (its only record is `EXPERIMENTS.md` 2026-10-05 and `docs/experiments.md` row 24; there is no separate review): the run
completed under `STRICT`, so no stage fell back; the loss (−0.030, ~4 public answers) comes from the fine-tuned intensity model itself.
It was fine-tuned by a third party on competition training spectra with no held-out check, and our local sets could not test it
(answers in train). The forward stage is also the most leaderboard-sensitive part of the chain (v34a −0.011 although locally
clean; line search; [C EXPERIMENTS.md 2026-10-07]). The plausible mechanism is the memorisation tilt above [I, not measured].

## 5. Recommendation

- **Close the gap-only forward-model line for stack1.** Three independent forward terms for these molecules (MetFrag-lite today,
  FIORA, GLACIER surrogates) do nothing measurable at stack1's weights, and the arithmetic ceiling is below one public answer.
- **Do not retrain GLACIER / ICEBERG for the gap** (b). If the team still wants it, the pre-condition is an in-group test on
  structures absent from train.parquet with train-structure decoys (the memorisation tilt), on the panel's natural products, before
  any replay; cost 1-2 days plus 5-8 GPU-h for a fine-tune.
- The larger isomer-order prize is on [M+H]+ molecules (0.21-0.25 RR each, 81-86% of molecules), where GLACIER captures 16-22% of a
  perfect order. That is a different question (how the forward evidence is weighted), with its own leaderboard history (v34a).

## 6. Files

`scratch/adductfwd/` (git-ignored): `PREREG.md`, `STATE.md`, `af_lib.py` (replay), `af_quantify.py`, `af_fwd.py` (GLACIER runner jobs),
`af_eval.py`. `C:\Users\hemre\casmi-work\adductfwd\`: `quantify.md/json`, `lists_UO.pkl`, `gl_sur.json`, `gl_neg.json`, `eval.md/json`.
Inputs read: the arb capture, `stack1err/bstar.pkl`, the regime query tables (adducts, spectra), `D:\casmi-data\coverage_study\glacier`
(runner, checkpoint), `train_meta.parquet` (adduct counts). Nothing written to D:.
