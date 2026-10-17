<!-- index: fusion-v4n 2026-10-17 FRIGID encoder pilot: our own spectrum network trained to predict FRIGID's conditioning fingerprint (2 seeds, scaffold-excluded), frozen FRIGID decoder, arms true fingerprint / MIST / ours on set A (50 no-record molecules) and a locked 100-molecule natural-product panel; pre-registered verdict: not PASS, kill (1 additional set-A answer against a bar of 5; panel mixed dMRR +0.0015, 95% CI 0.000 to 0.0045) -->
# FRIGID encoder pilot (2026-10-17)

Offline. Nothing was submitted, pushed or committed; no download, no `kaggle` command; the visible `test.parquet` was not
read; no NIST data. GPU jobs one at a time through `train/guard_run.py` + `gpushare` (duty 0.8 / game 0.25). Rules fixed
before training: `scratch/frigidpilot/PREREG.md`; log: `scratch/frigidpilot/STATE.md`; scripts `scratch/frigidpilot/`
(`prep.py`, `lists.py`, `gen.py`, `eval.py`; git-ignored); model code `clean/frigenc.py`, `clean/train_frigenc.py`,
`tests/test_frigenc.py`; outputs `C:\Users\hemre\casmi-work\frigidpilot\`. Design: `scratch/debate/codex_round2.md` section C.

**Question.** The 10-05 FRIGID study failed because the shipped MIST encoder predicts FRIGID's conditioning fingerprint
badly on our molecules. Can our own spectrum network, trained to predict exactly that fingerprint (Morgan r2, 4,096 bits,
binary), make the frozen FRIGID decoder produce answers our chain does not have?

## 0. Verdict

**Not PASS: killed, as pre-registered. Both decisive numbers miss.**

| decisive number (OURS, L2 formulas, 32 samples) | result | pass bar |
|---|---|---|
| set A: additional answers beyond the existing chain | **1 of 50** | at least 5 |
| panel: mixed dMRR at k = 5 (0.45 gain + 0.55 cost) | **+0.0015, 95% bootstrap 0.000 to 0.0045** (1 answer gained of 99) | > 0 with a positive lower bound |

Category (PREREG section 8): on set A it is **KILL (recipe)**: our fingerprints beat MIST's (Tanimoto mean 0.405 vs
0.322) and still give 1 answer, not 5. On the panel our fingerprints are *worse* than MIST's (0.341 vs 0.605; MIST's lead
comes from the 78 MassSpecGym molecules, section 4), so there the result is "not PASS, not adopted": the recipe does not
produce a fingerprint the decoder can use. With the *true* fingerprint the same decoder recovers 38 of 50 on set A and 70
of 100 on the panel: the decoder is not the limit, the fingerprint is. At Tanimoto 0.3 to 0.4 to the true fingerprint it
generates formula-matched structures that are not the answer (median best Tanimoto of any generated structure to the
answer 0.25 on set A).

**Seen / unseen caveat.** The decoder's SAFE pretraining corpus (`datamol-io/safe-gpt`, up to 230 M structures) is not
local, and the released checkpoint used no exclusion list (`exclude_inchikeys_file: null`). No molecule can be labelled
unseen. Set A (no PubChem record) is labelled **unknown**; panel molecules with a PubChem record **unknown-likely-seen**.
The TRUE-arm recovery (38 of 50) may be partly memorisation of pretraining structures; nothing here is evidence of de novo
generation of unseen structures.

## 1. Setup (fixed in PREREG before training)

| item | value |
|---|---|
| encoder | `clean/frigenc.FrigEnc` = `cd_pairmulti` spectrum transformer + a FRIGID head (BCE on the 4,096 FRIGID bits); 16k steps, seeds 0 and 1, mean of the two seeds' probabilities |
| leak control | held-out folds `hold` / `np` and their identity twins excluded, plus every training structure sharing a generic Murcko scaffold with a set-A or panel molecule (23,588 of 220,443, 10.7%) |
| binarisation | ours: one threshold on DEV (2,655 molecules) maximising mean Tanimoto, t* = 0.26 (DEV mean Tanimoto 0.372); MIST: shipped 0.187; at most 256 bits |
| decoder | FRIGID-base `DLM.ckpt` (EMA), frozen; the 10-05 study's `frigid_local.generate` (length rule, batches of 32, at most 96 samples, stop at 25 distinct formula-matched structures, seeded per molecule and formula, identical for every arm) |
| arms | TRUE (true FRIGID fingerprint, true formula: the decoder's ceiling), MIST (`mist_msg.pt`), OURS |
| formulas | true formula; L2 = first two distinct formulas of the existing list (true formula among them for 41 of 50 on set A) |
| set A | 50 no-PubChem-record, not-in-MassSpecGym molecules of the 10-05 study (fold `hold`); existing chain = that study's truth-removed v13a-like final 25, which already holds 12 of the answers |
| panel | 100 natural products of the joint0 FINAL half (40 timsTOF), locked before training (sha1 `7970a2cb...`), read once at the end |

## 2. Fingerprint quality (binarised prediction vs the true FRIGID fingerprint)

| set | encoder | n | bits set (median) | precision (median) | recall (median) | Tanimoto median | Tanimoto mean |
|---|---|---|---|---|---|---|---|
| DEV (calibration, in-sample threshold) | OURS (2-seed mean) | 2,655 | 33 | 0.615 | 0.429 | 0.333 | 0.372 |
| set A | OURS (2-seed mean) | 50 | 36 | 0.604 | 0.504 | 0.368 | 0.405 |
| set A | ours seed 0 / seed 1 | 50 | 36.5 / 39 | 0.597 / 0.573 | 0.493 / 0.511 | 0.373 / 0.351 | 0.394 / 0.395 |
| set A | MIST (true formula) | 49 | 44 | 0.500 | 0.431 | 0.300 | 0.322 |
| panel | OURS (2-seed mean) | 100 | 32 | 0.594 | 0.399 | 0.317 | 0.341 |
| panel | ours seed 0 / seed 1 | 100 | 33 / 33 | 0.584 / 0.580 | 0.390 / 0.420 | 0.293 / 0.316 | 0.330 / 0.335 |
| panel | MIST (true formula) | 100 | 44.5 | 0.729 | 0.807 | 0.612 | 0.605 |
| panel, in MassSpecGym (78) | OURS / MIST | | | | | | 0.344 / 0.665 |
| panel, not in MassSpecGym (22) | OURS / MIST | | | | | | 0.327 / 0.392 |

True fingerprints have 47 to 48 bits set in the median. `mist_msg.pt` is the MassSpecGym-trained MIST (by its name and the FRIGID
release; its training split is not verifiable from the files); its panel advantage sits on the MassSpecGym molecules. Ours was trained with every panel
scaffold removed.

## 3. Recovery on set A (50 molecules, label unknown)

Answer (metric key, RDKit 2026.03.3) among the decoder's distinct formula-matched structures, ordered by Tanimoto to the
arm's own fingerprint. "top-25" = anywhere in the generated list (generation stops at 25 distinct structures).

| arm | formulas | top-25, 32 samples | top-25, 96 samples | top-1, 32 | top-1, 96 | distinct structures (median) |
|---|---|---|---|---|---|---|
| TRUE fingerprint | true | 34 | 38 | 33 | 37 | 11.5 |
| MIST | true | 0 | 0 | 0 | 0 | 11.5 |
| MIST | L2 | 0 | 0 | 0 | 0 | 15.5 |
| OURS | true | 1 | 1 | 0 | 0 | 9.5 |
| **OURS** | **L2** | **1** | **1** | 0 | 0 | 12.5 |

**Additional answers beyond the existing chain (decisive, OURS / L2 / 32 samples): 1 of 50** (`h_c38f552d06`, 296 Da,
rank 2 in the generated list, within the first 5 new structures; our fingerprint's Tanimoto to the truth 0.71). Same at 96
samples. MIST: 0. TRUE: 30 at 32 samples, 32 at 96. **Pass bar: 5. Not met.**

Why: the decoder needs nearly the exact fingerprint. Of the 8 set-A molecules where our fingerprint reaches Tanimoto
≥ 0.6, the TRUE arm recovers 5 and OURS 1. The molecules TRUE recovers and those it does not have the same mean OURS
Tanimoto (0.41 vs 0.40). Nearest misses with the true formula at 32 samples (best Tanimoto of any generated structure to
the answer, median): OURS 0.25 (≥ 0.7 for 2 molecules), MIST 0.21 (1), TRUE 1.0 (36).

## 4. Locked panel (100 natural products, read once at 17:31 after the set-A analysis was written)

Lock checked before the read (sha1 of the sorted ids `7970a2cb707b827ab6c27aaf2d64d00ae6b55b29`). 96 have a PubChem record
(**unknown-likely-seen**), 4 none (**unknown**); 78 in MassSpecGym; 40 timsTOF. One molecule has no L2 formula (99).

| arm | formulas | top-25, 32 samples | top-25, 96 | top-1, 32 | top-1, 96 | distinct (median) |
|---|---|---|---|---|---|---|
| TRUE fingerprint | true | 63 | 70 | 63 | 70 | 8 |
| MIST | true | 26 | 29 | 22 | 24 | 8.5 |
| MIST | L2 | 25 | 28 | 21 | 23 | 16 |
| OURS | true | 1 | 2 | 0 | 1 | 5 |
| **OURS** | **L2** | **1** | 1 | 0 | 0 | 7 |

By label (32 samples, true formula): OURS 0 of 96 unknown-likely-seen, 1 of 4 unknown; MIST 24 of 96 and 2 of 4; TRUE 59
of 96 and 4 of 4. MIST's hits are 25 of 78 in MassSpecGym against 1 of 22 outside it. OURS's single hit
(`h_ee7395d418`, in MassSpecGym, no PubChem record, our fingerprint Tanimoto 0.31) is the only answer OURS adds on the panel.

Incremental MRR (generated rows take places 21 to 25 of the existing c2J list; gain = truth-removed list, cost = intact
list; mixed = 0.45 gain + 0.55 cost; 95% percentile bootstrap, 2,000 resamples, seed 0):

| arm | formulas | samples | k | gain | cost | **mixed dMRR [95% CI]** | answers gained (removed lists) | answers lost (intact) |
|---|---|---|---|---|---|---|---|---|
| **OURS** | **L2** | **32** | **5** | **+0.0034** | **0.0000** | **+0.0015 [0.0000, 0.0045]** | **1** | 0 |
| OURS | L2 | 96 | 5 | +0.0034 | 0.0000 | +0.0015 [0.0000, 0.0045] | 1 | 0 |
| OURS | true | 96 | 5 | +0.0133 | 0.0000 | +0.0060 [0.0000, 0.0165] | 2 | 0 |
| MIST | L2 | 32 | 5 | +0.0117 | 0.0000 | +0.0053 [0.0034, 0.0070] | 25 | 0 |
| TRUE | true | 32 | 5 | +0.0423 | +0.0014 | +0.0198 [0.0123, 0.0309] | 63 | 0 |

k = 1, 3, 10 change OURS by at most 0.0015 (all in `eval_panel.json`). No arm loses an intact answer because the generated
rows only displace places 21 to 25.

**MIST + FRIGID on the panel (not pre-registered as decisive; not a recommendation).** MIST with L2 formulas gains 25
truth-removed answers with a positive bound. All 25 of these hits are in MassSpecGym, the corpus MIST was (presumably) trained on, and 24 have
a PubChem record (so our chain can already reach them; the truth-removed list is an artificial situation for them). The
10-05 study measured the case that matters (no PubChem record): 0 of 50. This pilot's set A repeats it: MIST 0 of 50.


## 5. Cost

| item | value |
|---|---|
| GPU used (RTX 4070 Ti SUPER, active time) | training 2.23 h (2 seeds x 1.05 h, a failed attempt 0.10 h, smoke 0.03 h); encoding 34 s; generation 2.56 h (722 runs, 191.5 min wall, 37.9 min yielded); **total 4.8 h of the 8 h budget** |
| one decoder call, 96 samples | mean 12.1 s (set A) / 12.9 s (panel), median 9.7 / 8.8 s; 0.13 s per sample |
| one decoder call, first 32 samples | mean 4.0 / 4.3 s, median 3.2 / 2.9 s |
| **per molecule, OURS, L2 (1.76 / 1.88 calls), 32 samples** | **mean 6.4 s (set A), 7.8 s (panel); median 5.4 s** |
| our encoder | 2 seeds, milliseconds per molecule (DEV + set A + panel, 2,805 molecules, in 9 s) |
| GPU memory | peak 1.1 GB allocated (decoder), well inside a T4 |
| T4 equivalent (factor 4 to 6 in FP32, not measured; 10-05 study) | **about 26 to 47 s per molecule** for OURS / L2 / 32 samples |

## 6. Kaggle integration (not proposed; what it would have cost)

Not proposed: the pilot did not pass. If it had, the integration would have cost:

| item | cost |
|---|---|
| generation, about 430 hidden molecules, L2, 32 samples | 430 x 26 to 47 s = **3.1 to 5.6 T4 hours**; beside the present notebook (about 3 to 4 h) this fits the 9 h limit only at the lower end |
| gated to low-confidence lists (half to three quarters), or one formula (L1) | about 1.5 to 3.5 T4 hours |
| our FRIGID head | a second checkpoint of the fingerprint network (or the head added to the shipped one): seconds of T4 time |
| packaging | the 10-05 list (wheel set, SAFE tokenizer in an HF cache layout, length rule, `frigid_local.py`, EMA-only decoder of about 0.7 GB); decoder weights CC BY-NC 4.0, licence question open on the forum |

What a pass would need first is a much better fingerprint: the TRUE arm shows the decoder recovers most answers at
Tanimoto 1.0, while our encoder reaches 0.34 to 0.41, and a 0.6 to 0.7 fingerprint still gave 1 hit in 6 on set A. No
follow-up is proposed from this pilot.

## 7. What can mislead

- Set A is small (50) and the pass bar (5 additional) was set before the run; 1 of 50 is not distinguishable from MIST's 0 of 50 here or the
  10-05 study's 0 of 50. The panel's 1 answer is likewise within noise of 0.
- MIST's panel results cannot be read as generalisation: `mist_msg.pt` is presumably trained on MassSpecGym and 78 of the 100 panel molecules
  are in it; ours was trained with the panel's scaffolds removed.
- `eval.py` was edited after the takeover and before any result was read (per-part fingerprint tables, per-seed DEV
  thresholds for the per-seed rows, top-1 at 32 samples, timing); no criterion changed. `evaluate.py` in the same folder
  is an earlier, truncated draft and was not used.
- TRUE-arm recovery bounds what a perfect fingerprint could do, but may be inflated by pretraining memorisation (section 0).
- Our encoder lost 10.7% of its training structures to the scaffold exclusion; a production encoder trained on everything
  would be somewhat better on scaffold-sharing molecules, but the gap to the TRUE arm (Tanimoto 0.4 vs 1.0) is not a
  10% effect.
- Times are from an RTX 4070 Ti SUPER; the T4 factor (4 to 6 times slower in FP32) is not measured.
