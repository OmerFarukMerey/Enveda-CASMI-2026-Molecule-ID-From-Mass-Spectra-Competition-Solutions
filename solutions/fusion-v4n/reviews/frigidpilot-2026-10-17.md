<!-- index: fusion-v4n 2026-10-17 bounded FRIGID encoder pilot: FRIGID's exact conditioning fingerprint read from the decoder config, our pair+multi spectrum network trained on those 4,096 bits (2 seeds, scaffold-excluded, DEV-calibrated binarisation), frozen FRIGID decoder, arms true fingerprint / shipped MIST / ours on the historical 50 no-record molecules and on a locked 100-molecule natural-product panel, decoder-pretraining overlap labelling, incremental MRR with a bootstrap bound, GPU cost; pre-registered verdict KILL (recipe) -->
# The bounded FRIGID encoder pilot (2026-10-17)

Offline only. Nothing was submitted, uploaded, downloaded, committed or pushed; no `kaggle` command; no package installed; no
network except the local Hugging Face cache (offline flags set); the visible `test.parquet` was not read; no NIST data; notebook
code untouched. GPU work went one job at a time through `train/guard_run.py` + `train/gpushare.py` (duty 0.8, game duty 0.25, yield
above 9 GB, 8 GB cap), no DataLoader orphans after any job. Rules fixed **before** training: `scratch/frigidpilot/PREREG.md`
(design: `scratch/debate/codex_round2.md` section C). Running log for a takeover: `scratch/frigidpilot/STATE.md`. New tracked code:
`clean/frigenc.py`, `clean/train_frigenc.py`, `tests/test_frigenc.py` (data-free, pass). One-off scripts: `scratch/frigidpilot/`
(`prep.py`, `lists.py`, `gen.py`, `eval.py`, `verify.py`; git-ignored). Outputs: `C:\Users\hemre\casmi-work\frigidpilot\`; drive D:
was read only. **Every number below was recomputed from the raw generation records by `verify.py`, written independently of
`eval.py` for that purpose** (section 8 says why). No leaderboard estimate is given.

**Question.** The 10-05 study found that the FRIGID decoder rebuilds 7 of 9 no-record natural products from the *true* Morgan
fingerprint, while the shipped MIST encoder's predicted fingerprint (median Tanimoto 0.30 on our molecules) finds none. Can a
fingerprint network trained by us predict FRIGID's own conditioning bits well enough that the frozen official decoder produces
answers our chain does not have?

## 0. Verdict

**KILL (recipe), as pre-registered.** Our encoder is better than MIST on every fingerprint metric on honest molecules (set A
Tanimoto median 0.368 against 0.300, mean 0.405 against 0.322), and that buys almost nothing from the decoder:

| pre-registered condition | bar | measured | met |
|---|---|---|---|
| additional set-A answers beyond the existing chain (OURS, L2 formulas, 32 samples) | ≥ 5 | **1** of 50 | no |
| fresh-panel incremental MRR (same configuration, k = 5), 95% bootstrap lower bound | > 0 | **+0.0015, bound 0.0000** | no |

With the **true** fingerprint the same decoder and the same sampling recover 34 of 50 set-A molecules at 32 samples (38 at 96,
33 / 37 of them at rank 1) and 63 of 100 panel molecules: the decoder is not the limiting half, and a fingerprint of Tanimoto 0.4
is not close enough for it. The best Tanimoto of any structure our arm generates to the answer is 0.25 in the median (set A, true
formula, 32 samples): the decoder returns plausible neighbours, not the molecule. This is the `fpgen` finding again in another
decoder: "a predicted fingerprint loses 3 to 4 times against a true one".

**Seen / unseen.** The decoder's SAFE pretraining corpus is not local and was not downloaded, and the released checkpoint carries
`exclude_inchikeys_file: null` while the README speaks of "up to 230 million unlabeled structures". **No evaluation molecule can be
labelled unseen**, so nothing here is evidence of de novo generation of an unseen structure; the TRUE arm's 34 of 50 is an upper
bound that may be partly memorised pretraining text.

## 1. The fingerprint FRIGID needs (read from the files)

| source | statement |
|---|---|
| `DLM.ckpt` config (`provenance.json` of the 10-05 study) | `model.fingerprint_bits 4096`, `model.fingerprint_radius 2`, `fingerprint_conditioner_type cross_attention`, `fingerprint_activation_threshold 0.0`, `fingerprint_seq_max_len 256` |
| `dlm/utils/utils_data.py`, `Collator._safe_to_fingerprint` (the training target) | `AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=4096)` of the SMILES decoded from the SAFE string, converted to a 0/1 float array; **bits, not counts**; no chirality, no feature invariants (the legacy call's defaults) |
| `dlm/model_components.py`, `FingerprintSequenceEncoder.forward` | embeds the **set** of bit indices with value > 0 (threshold 0.0), at most 256 of them, no positions; so a continuous prediction has to be binarised, and more than 256 set bits would be truncated arbitrarily |

Our target function is `clean/frigenc.frigid_fp` (tested bit-for-bit against `rdFingerprintGenerator.GetMorganGenerator(radius=2,
fpSize=4096)`, RDKit 2026.03.3; a stereo-free SMILES gives the same bits). Over the 275,810 training structures the median count of
set bits is 46 (47.5 on set A, 47 on the panel). Binarised predictions with more than 256 bits keep the 256 most probable (never
triggered: our largest was 95 bits).

## 2. Data, leak control and the evaluation sets (all fixed before training)

| item | value |
|---|---|
| training data | the competition's `train.parquet` only, through the spectrum cache `D:\casmi-data\directions\run\work` |
| identity leak | the published identity-disjoint split: folds `hold` and `np` and their identity twins (metric key, raw first block, InChIKey14) are not training positives — the printed line of every run: "55,367 held-out structures, 220,434 training structures, 1,974,410 training spectra; identifiers shared: 0; shared pool rows: 0" |
| scaffold leak (new here) | every training structure sharing a generic Murcko scaffold with a set-A or panel molecule removed: 89 scaffolds, **23,588 of 220,443 structures (10.7%), 258,078 of 1,974,410 spectra**; asserted in the trainer ("listed structures among the training positives: 0") |
| DEV (calibration only) | the 2,655 DEV-half molecules of the joint0 benchmark minus set A; one query item each; used for exactly one number, the binarisation threshold |
| set A (diagnostic) | the 50 no-PubChem-record, not-in-MassSpecGym regime molecules of the 10-05 FRIGID review, all in fold `hold`; the existing chain is that review's truth-removed v13a-like final 25, which already holds **12** of the 50 answers (through the rule-based generator) |
| fresh panel (confirmation, locked) | 100 natural products of the joint0 **FINAL** half with a stored c2J list and a MIST-supported positive adduct, excluding every key used by the FRIGID review, set A or the fpgen study; drawn by a salted hash: the first 40 timsTOF panel molecules and 60 other natural products. Lock sha1 of the sorted molecule ids `7970a2cb707b827ab6c27aaf2d64d00ae6b55b29`, written before training; read once, after the set-A analysis |

Panel composition, recorded at the draw: 40 timsTOF (`enveda-np-examples`), **78 of 100 are MassSpecGym structures**, 96 have a
record in the local PubChem tier, masses 31 to 1,175 Da, adducts 82 `[M+H]+` / 10 `[M+NH4]+` / 6 `[M+Na]+` / 2 `[M-H2O+H]+`.
The 78 MassSpecGym structures are the panel's weak point and they favour **MIST**, not us (section 5).

## 3. Decoder-pretraining overlap (done before any generation result was read)

The corpus (`datamol-io/safe-gpt`) is **not local**: `D:\casmi-data\frigid\hf` holds 49 KB, the tokenizer only. It was not
downloaded (no network). The released decoder was trained with `exclude_inchikeys_file: null` (the repository's own pretraining
config points at an exclusion file; the release did not use one) and `remove_stereo: false`, so the authors' "test molecules were
excluded" cannot be checked and does not apply to this checkpoint. Labels used, by the only evidence available offline:

| label | basis | set A | panel |
|---|---|---|---|
| **unseen** | would need the corpus | 0 | 0 |
| **unknown-likely-seen** | has a record in the local PubChem tier (a 230 M-structure public corpus overlaps PubChem heavily) | 0 | 96 |
| **unknown** | no PubChem record | 50 | 4 |

Every generation result below is reported per label. Set A is the honest class (all 50 unknown, no database record, none in
MassSpecGym); the panel is dominated by unknown-likely-seen molecules, and 78 of them are in MIST's training set as well.

## 4. The encoder (ours), its training and its fingerprint quality

`clean/frigenc.FrigEnc` = the `cd_pairmulti` design of the specenc study (m/z tokens + pairwise mass differences + multi-spectrum
attention, d 512, 8 layers, 8 heads, 160-peak budget) with its two existing heads (window rank loss, BCE on the 10,226 pipeline
bits, element loss) and **one new head** (2d → 4d → 4,096, output bias at the mean bit density) trained with BCE weight 1.0 on
FRIGID's exact bits of the spectrum's own structure. From scratch, 16,000 steps, batch 256, lr 3e-4 cosine, two seeds. 60.6 M
parameters, 3.1 to 3.6 steps/s, peak 6.2 to 7.0 GB. Per seed: **1.06 h and 1.04 h of GPU computation** (1.32 / 1.30 h wall).
Trunk sanity, not a gate (hold-out window top-1, true formula): hold all 0.703 / 0.703, NP-catalogue 0.401 / 0.411, panel 0.345 /
0.358, enveda-180 0.724 / 0.724 — a little below `cd_pairmulti` (0.429 / 0.432 NP-catalogue), as expected after removing 10.7% of
the training structures and adding a second objective.

**Binarisation**: one threshold on DEV, the only DEV decision, maximising the mean Tanimoto of the two-seed mean probability:
**t\* = 0.26** (DEV mean Tanimoto 0.372; the curve is flat between 0.20 and 0.32, 0.371 to 0.372); per seed 0.26 and 0.25. MIST
keeps the repository's benchmark threshold 0.187.

| set | encoder | n | bits set (median) | precision (median) | recall (median) | Tanimoto median | Tanimoto mean |
|---|---|---|---|---|---|---|---|
| DEV (threshold fitted here) | OURS, 2-seed mean | 2,655 | 33 | 0.615 | 0.429 | 0.333 | 0.372 |
| DEV | ours seed 0 / seed 1 | 2,655 | 33 / 34 | 0.600 / 0.595 | 0.423 / 0.431 | 0.328 / 0.329 | 0.364 / 0.366 |
| **set A** (unknown) | **OURS, 2-seed mean** | 50 | 36 | 0.604 | 0.504 | **0.368** | **0.405** |
| set A | ours seed 0 / seed 1 | 50 | 36.5 / 39 | 0.597 / 0.573 | 0.493 / 0.511 | 0.373 / 0.351 | 0.394 / 0.395 |
| set A | MIST `mist_msg.pt` | 49 | 44 | 0.500 | 0.431 | **0.300** | 0.322 |
| **fresh panel** | **OURS, 2-seed mean** | 100 | 32 | 0.594 | 0.399 | **0.317** | 0.341 |
| fresh panel | ours seed 0 / seed 1 | 100 | 33 / 33 | 0.584 / 0.580 | 0.390 / 0.420 | 0.293 / 0.316 | 0.330 / 0.335 |
| fresh panel | MIST `mist_msg.pt` | 100 | 44.5 | 0.729 | 0.807 | **0.612** | 0.605 |

Reading: on the honest set A ours is better than MIST by +0.07 Tanimoto (+0.08 mean) at a third fewer predicted bits. On the panel
MIST is far ahead — because 78 of those 100 structures are MassSpecGym molecules it was trained on (the 10-05 review measured
Tanimoto 0.95 for trained structures, 0.30 for honest ones), while our network has never seen any of them. The panel's MIST column
is a memorisation number, not a generalisation number; our 0.317 and MIST's 0.300 of set A are the comparable pair.

## 5. What the frozen decoder produces

Identical sampling for every arm (`scratch/frigid/frigid_local.generate` of the 10-05 study, unchanged: length rule, batches of
32, at most 96 samples, stop at 25 distinct formula-matched structures, temperature 1.0, randomness 10.0, random state
crc32(mid|formula), order by Tanimoto to the arm's own binarised fingerprint). 722 runs, 720 with output (2 formulas MIST cannot
featurise). "found" = the answer's metric key among the distinct formula-matched structures.

| set | arm | formulas | n | found at 32 samples | found at 96 | rank 1 at 32 / 96 | distinct structures (median) |
|---|---|---|---|---|---|---|---|
| set A | TRUE fingerprint | true | 50 | **34** | **38** | 33 / 37 | 11.5 |
| set A | MIST | true | 50 | 0 | 0 | 0 / 0 | 11.5 |
| set A | MIST | L2 | 50 | 0 | 0 | 0 / 0 | 15.5 |
| set A | **OURS** | true | 50 | **1** | 1 | 0 / 0 | 9.5 |
| set A | **OURS** | **L2** | 50 | **1** | 1 | 0 / 0 | 12.5 |
| panel | TRUE fingerprint | true | 100 | 63 | 70 | 63 / 70 | 8.0 |
| panel | MIST | true | 100 | 26 | 29 | 22 / 24 | 8.5 |
| panel | MIST | L2 | 99 | 25 | 28 | 21 / 23 | 16.0 |
| panel | OURS | true | 100 | 1 | 2 | 0 / 1 | 5.0 |
| panel | OURS | L2 | 99 | 1 | 1 | 0 / 0 | 7.0 |

**Additional set-A answers beyond the existing chain** (the answer generated and not already among the existing final 25; 12 of the
50 are already there):

| arm | formulas | 32 samples | 96 samples | within the first 5 new structures |
|---|---|---|---|---|
| TRUE fingerprint | true | 30 | 32 | 30 / 32 (all at generator rank 1) |
| MIST | true / L2 | 0 / 0 | 0 / 0 | 0 |
| **OURS** | **L2** | **1** | **1** | 1 (generator rank 2) |

The one molecule is `h_c38f552d06` (296 Da); our fingerprint for it has Tanimoto 0.71 to the truth. **Bar: 5. Not met.**

**Panel by overlap label** (found at 32 / 96 samples):

| arm | formulas | PubChem record (96) | no record (4) | in MassSpecGym (78) | not in MassSpecGym (22) | timsTOF panel (40) |
|---|---|---|---|---|---|---|
| TRUE | true | 59 / 66 | 4 / 4 | 49 / 55 | 14 / 15 | 27 / 32 |
| MIST | true | 24 / 27 | 2 / 2 | **25 / 28** | **1 / 1** | 22 / 23 |
| OURS | true | 0 / 0 | **1 / 2** | 1 / 1 | 0 / 1 | 0 / 0 |

MIST's panel recall is 25 of 78 on structures it was trained on against 1 of 22 on structures it was not: the same split as the
10-05 review. Ours finds the one panel molecule it finds among the four **unknown** (no-record) ones.

**Why a better fingerprint does not help.** Best Tanimoto of any generated structure to the answer (set A, true formula, 32
samples; median over the 50): OURS 0.246 (≥ 0.7 for 2 molecules, exact for 1), MIST 0.209 (≥ 0.7 for 1), TRUE 1.000 (≥ 0.7 for 36,
exact for 31). The decoder behaves as a near-deterministic inverse of an (almost) exact fingerprint and degrades off a cliff: at
Tanimoto 0.3 to 0.4 it writes molecules of the right formula and the right flavour, essentially never the right structure.

## 6. Incremental MRR on the fresh panel (read once, at the end)

Existing list = the joint0 c2J list ordered by the rank average (weight 0.5) of the public ranker and our out-of-fold ranker,
key-deduplicated, top 25 (96 of 100 answers are in it; 99 of 100 lists give the true formula among their first two distinct
formulas, 1.88 generator calls per molecule). Generated rows: the first k structures whose key is not in the list take the last k
places. (i) **truth-removed** list (the class-3 situation a generator is for), (ii) **intact** list (what the same rows cost where
the answer is already listed). Decisive statistic, fixed in the pre-registration: mixed ΔMRR = mean(0.45 × gain + 0.55 × cost),
2,000-resample percentile bootstrap over the 100 molecules.

| arm | formulas | samples | k | n | gain (truth-removed) | cost (intact) | **mixed ΔMRR** | 95% interval | lists gaining the answer |
|---|---|---|---|---|---|---|---|---|---|
| **OURS** | **L2** | **32** | **5** | 99 | +0.0034 | 0.0000 | **+0.0015** | **[0.0000, 0.0046]** | 1 |
| OURS | L2 | 96 | 5 | 99 | +0.0034 | 0.0000 | +0.0015 | [0.0000, 0.0046] | 1 |
| OURS | true | 32 / 96 | 5 | 100 | +0.0033 / +0.0133 | 0.0000 | +0.0015 / +0.0060 | [0.0000, 0.0045] / [0.0000, 0.0165] | 1 / 2 |
| MIST | L2 | 32 | 5 | 99 | +0.0117 | 0.0000 | +0.0053 | [0.0034, 0.0070] | 25 |
| MIST | true | 32 | 5 | 100 | +0.0212 | 0.0000 | +0.0095 | [0.0037, 0.0193] | 26 |
| TRUE fingerprint | true | 32 | 5 | 100 | +0.0423 | +0.0014 | +0.0198 | [0.0123, 0.0309] | 63 |

The pre-registered lower bound for OURS is **0.0000**: not positive, so the second pass condition fails as well. At k = 1 the
OURS gain is exactly 0 (its one answer sits at generator rank 2). The cost column is 0.0000 for every arm but TRUE because the
generated rows only occupy tail places 21 to 25 and no answer of an intact list was displaced; adding them is not free in general
(the 10-05 review measured the price of extra outside rows), it is just invisible at k = 5 on these lists.

**MIST's positive panel bound is not a reason to integrate MIST**: 25 of its 25 gains are MassSpecGym structures it was trained on
(1 of 22 outside), the hidden test is not, and the 10-05 review's honest-molecule measurement for exactly this arm was 0 of 49.

## 7. Cost

| | |
|---|---|
| GPU computation of the whole pilot | **≈ 4.8 h of the 8 h cap**: training 2.23 h (1.06 + 1.04 + 0.10 of a failed attempt + 0.03 smoke), encoder inference 0.01 h (34 s for 2,805 items plus 284 MIST pairs), generation 2.56 h (722 runs, 191.5 min wall of which 37.9 min slept by `gpushare`) |
| per generation run | 12.6 s mean, 9.0 s median at 96 samples; 4.2 s mean at 32 samples; 0.13 s per sample; peak 1.08 GB allocated |
| per molecule, the decisive configuration (OURS, L2, 32 samples) | **7.3 s mean, 5.4 s median** on an RTX 4070 Ti SUPER (1.84 calls per molecule) |
| sample quality | 96.6% valid SMILES, 12.6% with the requested formula, 5 to 12 distinct formula-matched structures per run (median by set / arm) |
| CPU | 6 workers for the fingerprint and scaffold pass (31 s), 1 worker for the PubChem-record check of the panel (10 min), 3 DataLoader workers per training run; free memory never below 10 GB |
| disk | 0.65 GB under `C:\Users\hemre\casmi-work\frigidpilot` (two 243 MB checkpoints, 722 JSON records, prediction caches); C: stayed above 20 GB free; nothing was extracted from the tarball on D: (the 10-05 study's `ckpt\` was read in place) |

## 8. Incidents, and what can mislead

- **Files of this study were changed from outside this session.** At 16:29, while generation was running, `scratch/frigidpilot/eval.py`
  (the measurement script) was modified, a second script `scratch/frigidpilot/evaluate.py` appeared (11:34), and a draft of **this
  review file** with the set-A numbers and placeholders was written at 16:39 — none of it by this session, which wrote `eval.py` at
  11:50 and touched no review file before 17:40. The repository also received three commits from another worker during the run. The
  draft is kept at `C:\Users\hemre\casmi-work\frigidpilot\review_draft_1639.md.bak`; no number from it was copied. Because the
  provenance of that measurement code cannot be established, **every number in this review was recomputed from the raw per-run JSON
  records by `scratch/frigidpilot/verify.py`**, written for that purpose and independent of `eval.py` (it also re-derives the true
  FRIGID fingerprints from the SMILES and asserts they equal the cached matrix). The two scripts agree where both compute a
  quantity. If another agent is working on this pilot, its results must be compared against `verify.json` before anything is merged.
- `fe_s1` died at step ~1,300 of a CUDA out-of-memory inside the 8 GB cap (allocator fragmentation on one 640 MB block of the window
  loss). The loss is now accumulated in chunks of 64 queries, numerically the same (identical losses at steps 200 and 400 after the
  restart); the run was repeated from scratch. 0.10 h of GPU lost, no orphan processes.
- **Set A is 50 molecules** and the bar (5 additional answers) was fixed beforehand; 1 of 50 cannot be separated from the 10-05
  review's 0. The panel is 100 molecules, 78 of them MassSpecGym structures: that inflates MIST and not us, and it means the panel
  is a weak test of the honest, class-3-like case (only 4 of its molecules have no database record at all). A better panel would
  draw record-less natural products, of which the hold-out has few.
- Our encoder lost 10.7% of its training structures to the scaffold exclusion and its trunk is accordingly 0.02 to 0.06 top-1 below
  `cd_pairmulti`. A production encoder trained on everything would predict somewhat better fingerprints — but the gap that matters
  is 0.4 against 1.0 Tanimoto, not a 10% effect.
- Two seeds, one length (16,000 steps), one architecture, one loss weight (1.0), one threshold rule. A larger FRIGID head, a
  Tanimoto or focal loss, per-bit calibration, or training only on the bits the decoder attends to were not tried; the pilot's
  question was whether the published decoder pays for a better encoder at all, and at Tanimoto 0.4 it does not.
- The 10-05 length rule is still a substitute for the authors' unpublished token-length model (12.6% of samples match the requested
  formula here, 16% there); a better one would give more distinct structures per call, not a better fingerprint.
- Times are from an RTX 4070 Ti SUPER. The T4 factor (4 to 6 times slower in FP32) is not measured here.
- `tools/gen_index.py` was not run and nothing was committed (other sessions are writing in the repository); `docs/experiments.md`
  and `EXPERIMENTS.md` were not touched (no submission).

## 9. Recommendation

**Close the FRIGID encoder line; do not integrate it and do not spend a submission on it.** Both pass conditions fail, and the
reason is not fixable by a better encoder of our kind: the decoder needs a near-exact fingerprint (34 of 50 from the truth, 1 of 50
from a 0.4-Tanimoto prediction, 0 of 50 from MIST's 0.3), while our own best spectrum network, trained directly on its bits with a
calibrated threshold, reaches 0.37 on the honest class — the same wall the `fpgen` study hit with its own decoder and the `ownfp`
study with retrieval. Had it passed, integration into the 0.451 notebook would have needed: the five pieces the 10-05 review lists
(a wheel set of the tested versions on a stage-local path, the SAFE tokenizer in a Hugging Face cache layout, the length rule,
`frigid_local.py`, optionally a slimmed checkpoint), our two 243 MB encoder checkpoints as a dataset, about 1.84 decoder calls per
molecule at 32 samples (7.3 s per molecule here, roughly 30 to 45 s on a T4: 3.5 to 5.5 h for 430 molecules, which does not fit
beside the notebook's 3 h 40 min unless gated to the `lib_max` < 0.7 half), plus the unanswered CC BY-NC licence question of the
checkpoints (forum 745072, 745615, 745841). None of that is worth doing for one answer in 50.

What would change the verdict is not a packaging or a sampling change but a fingerprint that is right to within a few bits on
timsTOF natural products — i.e. the problem the whole pipeline is already stuck on.

## 10. Reproduce

```bash
P=.venv/Scripts/python; S=scratch/frigidpilot; W='C:\Users\hemre\casmi-work\frigidpilot'
$P $S/prep.py --step structs            # FRIGID fingerprints + generic scaffolds of all 275,810 training structures (6 CPU workers, 31 s)
$P $S/prep.py --step sets               # set A, DEV, the locked panel (sha1 7970a2cb...), the scaffold exclusion list (10 min)
$P $S/lists.py                          # existing-chain lists (intact / truth-removed) and the L2 formulas
sh $S/run_train.sh                      # fe_s0, fe_s1: clean/train_frigenc.py, 16k steps each, through train/guard_run.py
$P $S/gen.py encode ; $P $S/gen.py calib    # predictions (ours, both seeds; MIST per (molecule, formula)); DEV threshold t* = 0.26
sh $S/run_gen.sh                        # 722 decoder runs (arms ours / mist / true), resumable
$P $S/verify.py                         # every number of this review, recomputed from the raw records -> verify.json
$P -c "import sys; sys.path.insert(0,'solutions/fusion-v4n/tests'); import test_frigenc as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]"
```

Checkpoints and records: `C:\Users\hemre\casmi-work\frigidpilot\runs\fe_s{0,1}_final.pt` (`clean/frigenc.load_ckpt`; licence: ours,
trained on `train.parquet` only), `gen\<arm>\<mid>__<formula>.json`, `enc_ours.npz`, `enc_mist.npz`, `calib.json`, `verify.json`,
`sets_info.json`. FRIGID provenance unchanged from the 10-05 review: Zenodo 19685145 (CC BY-NC 4.0), `coleygroup/FRIGID@d242a2c`
(Apache-2.0), decoder trained on the `datamol-io/safe-gpt` SAFE corpus, MIST encoders on NPLIB1 / MassSpecGym per the authors; no
NIST stated anywhere.
