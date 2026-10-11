<!-- index: 2026-10-10 rebuild plan for a prize-eligible fusion-v4n chain: the audit's verdicts re-checked at the primary sources (the host answered thread 745072; corrections in both directions), which code in the notebook comes from whom under which licence, a replacement with its cost for every non-clean item, the measured cost of removing engine 2 / the public ranker / our ranker / the forward models on the Enveda panel, the mixed regime and DEV, what was built without a GPU (own pool builder with coverage, chemistry core, specification, golden tests), the schedule with GPU and CPU hours, approvals needed and the drafted question for the host -->
# Clean-chain rebuild plan, 2026-10-10

Continues `licence-audit-2026-10-10.md`. **We are not lawyers. Whether a submission is eligible is decided by the
host; this file sorts facts and costs so that the user can decide.** Nothing was submitted, pushed, committed or
uploaded; no dataset or model was downloaded; read-only `kaggle datasets metadata` calls (19 datasets, saved under
`scratch/clean/meta/`); cards, licences and the forum were read signed out. No GPU was used (a training job of the
joint model was running). CPU: at most 6 processes of mine at below-normal priority. The visible `test.parquet` was
not read. FINAL of the joint benchmark was not read. Notebook code and `THIRD_PARTY_NOTICES.md` were not changed.
New tracked files: `solutions/fusion-v4n/clean/` (5 files), `solutions/fusion-v4n/tests/test_clean_contract.py`, this
file. Scripts and outputs: `scratch/clean/` (git-ignored), `D:\casmi-data\clean\` (6 MB). `INDEX.md` not regenerated.

## 0. Summary

1. **The host answered thread 745072 during the night before this study** (David Healey, shown as "12 hours ago" at
   12:15 on 2026-10-10): "any datasets that are only noncommercial by inheriting from train.parquet are fine to use.
   As are the large molecule libraries like PubChem and COCONUT [...] developing your solutions on the datasets you
   have mentioned here should be fine." The thread names `ahmedberatozer/casmi26-v4b-models`, `-v3-models`,
   `-v2-pool`, `-fpnet-full1`, `-iceberg`, `-glacier`, `-pubchem-tier`. This removes the audit's main "problem" class
   (a non-commercial note inherited from the training file) for those datasets **as data and weights**.
2. **Not answered, and now the whole remaining exposure:** (a) source code without a licence inside datasets (the
   `casmi` / `fe_v4` packages = the v1 engine, 6,500 lines; the ICEBERG / GLACIER runners and shims, 3,000 lines);
   (b) whether winners must deliver training code for third-party checkpoints (Rule 2.8.a; none is public for
   `fpnet_full1`, `fe_A/B/D`, `ranker_v4b`, the v3 nets, prvsiyan's nets); (c) the scope of the compliance check;
   (d) prvsiyan's two datasets (CC BY-NC-SA 4.0, training data not documented, not named in the thread).
3. **Code is the bottleneck, not the GPU.** Everything that needs training adds up to 10 to 17 GPU hours (two own
   fingerprint networks, two engine-2 networks, replay inference). The v1 engine's code has no licence and no
   public replacement; a rewrite from the specification is the long pole (10 to 14 working days for the reduced
   design of section 4; not measured, an estimate).
4. **Measured cost of removal** (unseen-regime chain, 606 molecules, paired against the v33b / v59a design; section
   3): engine 2 off: −0.15 where the answer is only in engine 2's pool, +0.05 to +0.11 where it is only in PubChem,
   ±0.01 elsewhere. Public ranker off (our ranker alone): within ±0.02 everywhere, leaderboard 0.434 against 0.442
   (old scoring, one submission each). Our ranker off: −0.02 to −0.06 on answers outside the pool. Forward models
   off: −0.04 to −0.09 on answers outside the pool, ±0.01 to +0.02 on pool answers.
5. **Built today:** our own pool builder from `train.parquet` and the upstream COCONUT release (CC0), our own
   chemistry core (adducts, cleaning, fragmenter), the specification of every module to be rewritten, golden tests
   (7 of 7 pass: 300 reference pool rows reproduce SMILES, key, formula, mass and all 10,226 fingerprint bits; 400
   cleaned spectra, 100 fragment lists and every adduct's neutral mass reproduce). The full pool build ran (9.2 CPU
   hours): 715,304 rows, **99.92% of the reference pool's keys**, the training-structure table identical in all
   275,810 rows; 576 reference keys missing, the bit selection not reproduced (section 4.1).

## 1. Corrections to the audit

| # | Audit said | Found today | Source |
|---|---|---|---|
| C1 | Thread 745072: "no host answer" | **Answered** (text in section 0). Data and weights that are non-commercial only by inheritance from `train.parquet` are fine; PubChem and COCONUT are fine; disqualification can include removal from the leaderboard; scope of the compliance check still open ("I will get back to you") | forum thread 745072, read signed out 2026-10-10 12:15 |
| C2 | Items 10 to 13 and 15 (`v2-pool`, `fpnet-full1`, `v3-models`, `v4b-models`, fold-safe FPNet): **problem** by label | **allowed by the host as data / weights** (10 to 13 by name in the thread; 15 by the rule stated: its card gives exactly this reason, "derived from the competition training file, which includes components under non-commercial licences"). The labels themselves are unchanged ("Other", non-commercial note). What remains for them is training-code delivery (2.8.a) for 11 to 13 | C1; cards in `scratch/clean/meta/` |
| C3 | Item 14, `fe_D.pt`: "which DreaMS checkpoint is the base is **not stated**" | **Stated twice by the author**: `code/casmi/dreams_model.py`, docstring: "Weights: ssl_model.ckpt (HF roman-bushuiev/DreaMS, MIT; Zenodo 10997887, CC-BY-4.0), pretrained on GeMS/MassIVE only"; and inside `fe_D.pt` the stored training arguments contain `'ssl_ckpt': 'data/external/dreams/ssl_model.ckpt'` (20,000 steps, batch 128, 63 negatives). `ssl_model.ckpt` is the self-supervised checkpoint, not a head fine-tuned on MoNA / NIST20. The fine-tuning data is still described only by the dataset's general note. Not verified: that the tensors descend from that file (the public checkpoint does not load with `weights_only=True`, and it was not unpickled) | local copy of the dataset; `torch.load(..., weights_only=True)` of `fe_D.pt`; Zenodo record API (licence `cc-by-4.0`, files `embedding_model.ckpt`, `ssl_model.ckpt`); Hugging Face card "license: mit" |
| C4 | Item 20, engine-2 code: licence "by Kaggle's notebook default; their pages were not opened" | **opened: Apache 2.0** on prvsiyan's `analog-propagation-casmi-2026-baseline`, megayak's `casmi26-one-engine-nine-scores`, wangpenghua's `casmi26-v29-fusion-pop-glmh`, ahmedberatozer's `casmi26-v4g-inference` ("This Notebook has been released under the Apache 2.0 open source license") | notebook pages, 2026-10-10 |
| C5 | Item 19 groups the PubChem-channel code with seyitkaangunes's notebook | `pc_runner.py` and `probe_core2.py` were written by ahmedberatozer and are **embedded as source strings in his own Apache-2.0 notebooks** (`casmi26-v3-inference`, `-v4f` … `-v4q`; our pulled copies in `scratch/landscape7/nb/`), so the licence comes from the author himself. They import `casmi.fpnet`, `casmi.chem`, `casmi.spectra` from the dataset, which is not embedded in any notebook (0 of 9 notebooks contain the engine source) | `grep` over the nine pulled notebooks |
| C6 | Items 4, 10: COCONUT "CC BY 4.0" (from the Kaggle cards) | Upstream is **CC0**: "COCONUT data is released under the Creative Commons CC0 license, allowing for free use, modification, and distribution without any restrictions. No attribution is required when utilizing this data." The CC BY 4.0 label is the Kaggle packager's | coconut.naturalproducts.net/download, 2026-10-10 |
| C7 | Section 8: upstream licences not re-read | Read: ms-pred `LICENSE` at commit 708148c ("MIT License, Copyright (c) 2023 Samuel Goldman"); its README offers "the open-source MassSpecGym-trained weights for ICERBERG or GLACIER" (Dropbox links, no licence file of their own) and NIST weights only "with a proof of license"; MassSpecGym card "license: mit"; DreaMS as in C3; PubChem: "Information that is created by or for the US government on this site is within the public domain" (NCBI policies page) | raw.githubusercontent.com, huggingface.co, zenodo.org, ncbi.nlm.nih.gov |
| C8 | Item 16 card body "still says CC0-1.0" | the card of `casmi26-fp-models-v2` now says "The current Kaggle dataset license label is **CC BY-NC-SA 4.0**. The license label applies to this published bundle; it does not establish the terms of every upstream training source." Field and card agree. `prvsiyan/casmi26-fp-models-v4` is still CC0-1.0 | metadata of 2026-10-10 |
| C9 | (confirmed, not a correction) `train_fpnet.py` Apache-2.0 | the card: "the training loop and evaluation scripts in this dataset are released under Apache 2.0". But the loop imports the model class, collate and data access of the unlicensed `casmi` package: a network trained with it is ours as weights, and the training code we would have to deliver (2.8.a) still depends on unlicensed code | card; `train/train_fpnet.py` line 82 |

Net effect on the audit's counts for the v59a chain: of its 11 "problem" items, 5 (10 to 13, 15) move to "allowed by
the host as data, open on training code"; 2 (16, 17) stay; the "unclear" code items 22 and 23 stay and are now the
first-order risk; item 14 moves from "unclear" to "documented by the author".

## 2. Every non-clean item: verdict at the primary source, replacement, cost

Licence texts are quoted from `scratch/clean/meta/<owner>__<slug>/dataset-metadata.json` (fetched 2026-10-10) unless
another place is named. (a) = public permissive alternative, (b) = our own rebuild, (c) = removal.

| Item | Licence text found, where | Verdict today | Replacement and cost |
|---|---|---|---|
| **`code/casmi/**`, `code/fe_v4/**`** (v1 engine; inside `-v3-models`, `-v4b-models`) | no licence file, no header in any of the 28 code files (`grep` for licence / copyright / SPDX: only the DreaMS note in `dreams_model.py`); dataset field "other"; `LICENSE-NOTICE.txt` covers DreaMS only | **not clean: code without a licence; host silent on Rule 6.b for datasets** | (a) none exists. (b) **rewrite from `clean/SPEC.md`**: core engine 2,244 lines in 11 modules (`chem`, `spectra`, `library`, `pool`, `frag`, `derive`, `fpnet`, `engine`, `ranker`, `structs`, `build`), families 3,863 lines in 11 files. Recommended reduced design: core + an own `fragnet`-type count model, the other four families dropped, ranker refitted. Done today: `chem`, `pool`, `structs`, cleaning, fragment masses (about 600 of the 2,244 lines' worth). Remaining estimate 10 to 14 working days, 0 GPU hours. (c) not possible: it is the chain. **Cheapest fix: the author adds MIT / Apache-2.0 (the user's business), or the host says 6.b covers datasets** |
| **`ice_runner.py`, `fuse.py`, `gl_runner.py`, `gl_fuse.py`, `shim/**`** (in `-iceberg`, `-glacier`) | `LICENSE-NOTICE.txt` (`D:\casmi-data\coverage_study\iceberg\`, `...\glacier\`): "shim/ : pure-torch stand-ins [...] (own code)"; no licence for the runners (1,955 lines) or the shim (1,085 lines) | **not clean: code without a licence**; the ms-pred source and weights they drive are MIT | (a) `runner/fwd_runner.py` of `prvsiyan/casmi26-forward-reference-stack` (16,254 bytes; card: "`runner/fwd_runner.py` is our own code, MIT"); the whole dataset is 3.8 GB with unmodified DGL / PyG / CUDA wheels, card: "`easydict` is LGPL-3.0 and `pygmtools` is MulanPSL-2.0. The NVIDIA CUDA, cuDNN and cuSPARSELt runtime wheels are under NVIDIA's EULA/SLA" (wheels are "commercially available software" in the sense of 2.5.a second paragraph at most; not tested by us; "Numerical parity against an upstream CPU reference run has **not** been certified"). (b) own runner on unmodified ms-pred + own glue: about 400 lines, 2 to 3 days, 1.5 GPU hours for the parity check against the stored scores (ICEBERG 61 min + GLACIER 21 min, `reviews/fwd-replay-2026-10-05.md`). (c) `USE_ICE = USE_GL = False`: section 3 |
| `fpnet_full1.pt`, `fe_A.pt`, `fe_B.pt`, `ranker_v4b.pkl`, `fe_v4/res/**` (items 11, 13) | "Source note: trained on the Enveda CASMI26 competition training data (which includes CC BY-NC licensed components). Use under the source licenses (non-commercial, with attribution)." | **allowed by the host as data (C1)**; no training code, ranker rows unpublished → open under 2.8.a | (c) **remove the public ranker pass**: `RANKER='fs'` uses none of them below the gate; measured in section 3 (within noise locally, 0.434 against 0.442 on the old leaderboard). Above the gate (`lib_max` ≥ 0.7) the public ranker still orders the list today; our ranker at every `lib_max` needs class-1 lists in its fit (rows exist: `c1N`, `c1J` of the benchmark) and one replay: 1 CPU hour, not measured yet |
| `fe_D.pt` (item 14) | `LICENSE-NOTICE.txt`: "fe_D.pt is derived (fine-tuned for spectrum -> fingerprint prediction) from DreaMS (Bushuiev et al., Nature Biotechnology 2025; weights MIT / CC-BY-4.0, code MIT, re-implemented in code/casmi/dreams_model.py); redistributed under the same terms with this attribution." | documented base (C3), host: "DreaMS weights are allowed" (745604); no training code | (c) removed together with the public ranker pass (`fs` pass puts our network in all three `model_views` slots). No separate flag exists; not measured on its own |
| `fpnet_0.pt`, `fpnet_1.pt` of `-v3-models` (item 12; PubChem channel) | same source note as `-v2-pool` | allowed as data (C1); no training code | (b) two own networks (below). `PC_MODELS` already selects another pair |
| `casmi26-v2-pool` tables (item 10) | "contains data and/or models derived from the Enveda CASMI26 competition training data (which includes CC BY-NC licensed components) and COCONUT (CC BY 4.0). Use under the respective source licenses (non-commercial, with attribution)." | allowed as data (C1) | (b) **`clean/pool_build.py`, running today** (section 4.1): 9 CPU hours, 1.3 GB. Fragment tables: + 2 to 4 CPU hours once the fragmenter is compiled |
| `casmi26-pubchem-tier` (item 3) | field "other"; card: "Mass-sorted PubChem structures (CID-SMILES + CID-Mass, NCBI PubChem public data, stereo stripped, organic CHNOPS+halogens, 150-1250 Da)" | clean by content; named in the thread the host answered | (b) optional rebuild from `ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/`: `CID-SMILES.gz` 1.4 GB + `CID-Mass.gz` 1.3 GB (listing of 2026-09-27; public domain). Stereo stripping and filtering of about 124.7 M rows: estimated 12 to 16 CPU hours (3 hours with 5 workers), 6.4 GB output, about 10 GB of disk during the build (drive C:). **Not recommended before the code question is settled: it changes nothing in the verdict** |
| `dmitriigluzdov/casmi26-fold-safe-fpnet` `fpnet_fold0.pt` (item 15) | "The checkpoint is derived from the competition training file, which includes components under non-commercial licences; use it under the competition data terms, non-commercially and with attribution. [...] the training loop and evaluation scripts in this dataset are released under Apache 2.0." | allowed as data by the rule of C1 (not named in the thread); training code public | (b) **own fold-safe network** with the same split: published run 82,855 steps, "5.0 h on one RTX 6000"; ours runs 13 steps/s at batch 64 unthrottled (`train/README.md`), about 10 at duty 0.8 → 2.3 GPU hours if batch 64 is the published batch (not checked), **budget 3 to 6 GPU hours per network; two networks** (engine bank + second member of the PubChem pair). Clean only after the model class and data access are ours (SPEC 6: 390 lines) |
| our rankers (`casmi26-ours-ranker-fs`, item 18) | ours; card "Other" | inherit the status of the features: computed with `casmi` / `fe_v4` code and `fpnet_fold0.pt` | (b) refit on the own network's features, computed by the rewritten engine: engine passes over the 5,350 hold-fold molecules (the 7,632 runs of `reviews/fsranker3-2026-10-07.md` ran on the CPU; PubChem lists 5.4 s per molecule on 3 workers) about 10 to 20 CPU hours, under 1 GPU hour; the LightGBM fit takes minutes |
| `prvsiyan/casmi26-fp-models-v2` (item 16) | field "CC-BY-NC-SA-4.0"; subtitle "Training-derived artefacts for the CASMI 2026 notebooks (CC BY-NC-SA)"; card: "Checkpoint metadata inspection does not establish a training split, upstream corpus snapshot, or held-out evaluation result." | **not clean**: non-commercial share-alike label, training data not documented, not named in the answered thread, no training code | (a) `megayak/casmi26-simulated-ranker-rows` `ours_fpnet_single_16k.pt`, `ours_fpnet_merged_26k.pt` (CC0-1.0, 144 MB each, already local; training set implied, not named; no training code). (b) own pair with `train/train_eng_fpnet.py` (ours; the model class is `pv_fp.FPNet`, Apache-2.0): 20,000 steps from scratch took 0.30 GPU hours and reached 0.68 to 0.84 on seen libraries against 0.78 to 0.92 for the public model (`train/README.md`, question 4): **2 networks, about 1 GPU hour**, expect a somewhat weaker engine 2. (c) `USE_ENG=False`: section 3 |
| `prvsiyan/casmi26-ranker-features` `rank_train.npz` (item 17) | field "CC-BY-NC-SA-4.0"; card: "a derived training resource, not a raw spectral library" | **not clean** (label; generator not pinned) | (a) megayak's CC0 rows (`sim_rank_rows_fppair.npz`): measured with his networks at stage level, section 3 (ii). (b) regenerate the rows with the embedded Apache-2.0 code and our own networks: the simulation harness is not in the notebook (only the feature builder is); 1 to 2 days, CPU. (c) with engine 2 |
| `megayak/casmi26-simulated-ranker-rows` `sim_rank_rows_nofp.npz` (item 7) | field "CC0-1.0"; README inside the dataset | clean (CC0; train-derived). Known defect, not a licence matter: the answer's row position leaks through tied ranks (`reviews/eng2fs-2026-10-07.md`) | keep, or regenerate with (b) above |

## 3. Measured cost of removing or substituting each component

Harness: the unseen-regime chain of the metafit study (`scratch/metafit/mf_lib.py`; dumps of every input of
`fusion_core.build_submission` for 606 molecules: mixed regime 380, Enveda panel 226), replayed with
`fusion_core.build_submission` itself and real ICEBERG / GLACIER scores. Script `scratch/clean/abl/abl.py`, results
`D:\casmi-data\clean\abl\table.json`. **Reference** = variant A: v33b's design (the preset v59a differs only in the
source of engine 2's ChEBI / LIPID MAPS rows, with identical fingerprints) with the fold-safe network under both
rankers, engine 2 cross-fitted on the panel; it reproduces the stored 0.8280 / 0.7497 / 0.5323 / 0.3306 (mixed) and
0.9260 / 0.8265 / 0.6148 / 0.3511 (panel). List types: `c1` library spectra of other libraries kept, `c2` answer a
pool row without spectra, `dA` answer removed from the v1 pool but still in engine 2's pool, `dB` answer in neither
(PubChem only). Paired difference of MRR@25 ± one standard error (wins / losses). **No leaderboard estimate is
given**; local replays had the leaderboard's sign in 5 to 7 of 11 contrasts (`reviews/uregime-2026-10-07.md`).

| Arm | set | `c1` | `c2` | `dA` | `dB` |
|---|---|---|---|---|---|
| n | mixed / panel | 150 / 226 | 380 / 226 | 325 / 226 | 325 / 226 |
| **(i) engine 2 off** (`eng = {}`) | mixed | −0.002 ± 0.010 (8 / 11) | +0.007 ± 0.009 (38 / 43) | **−0.149 ± 0.014** (10 / 162) | **+0.052 ± 0.011** (75 / 25) |
| | panel | −0.005 ± 0.009 (11 / 9) | +0.010 ± 0.011 (27 / 17) | **−0.151 ± 0.017** (10 / 114) | **+0.112 ± 0.016** (91 / 25) |
| **(iii-a) public ranker off** (our ranker alone below the gate, variant U; drops `ranker_v4b`, `fe_A/B/D` there) | mixed | +0.004 ± 0.008 (8 / 8) | +0.013 ± 0.009 (45 / 25) | +0.003 ± 0.008 (49 / 39) | −0.003 ± 0.007 (32 / 41) |
| | panel | −0.004 ± 0.006 (5 / 8) | +0.017 ± 0.010 (24 / 11) | −0.022 ± 0.014 (26 / 40) | +0.017 ± 0.011 (41 / 36) |
| **(iii-b) our ranker off** (public ranker alone, variant R) | mixed | −0.001 ± 0.002 (2 / 3) | −0.005 ± 0.006 (15 / 22) | **−0.033 ± 0.010** (27 / 43) | **−0.020 ± 0.008** (21 / 51) |
| | panel | −0.004 ± 0.003 (1 / 2) | **−0.013 ± 0.005** (4 / 13) | **−0.056 ± 0.011** (9 / 45) | **−0.042 ± 0.012** (12 / 47) |
| **(v) forward models off**, no scores at all (`ice = gl = {}`; the fragment term then acts on every list) | mixed | −0.008 ± 0.008 (5 / 8) | +0.005 ± 0.010 (31 / 33) | **−0.056 ± 0.012** (32 / 83) | **−0.036 ± 0.011** (29 / 62) |
| | panel | −0.011 ± 0.010 (10 / 14) | −0.007 ± 0.015 (21 / 29) | **−0.091 ± 0.014** (16 / 80) | **−0.072 ± 0.012** (13 / 85) |
| (v') forward weights 0, fragment term as configured | mixed | +0.004 ± 0.005 | +0.020 ± 0.010 | −0.051 ± 0.012 | −0.040 ± 0.011 |
| | panel | +0.008 ± 0.008 | +0.014 ± 0.013 | −0.075 ± 0.014 | −0.061 ± 0.012 |
| **our ranker alone + engine 2 off** (forward models kept) | mixed | +0.009 ± 0.012 | +0.026 ± 0.012 | −0.140 ± 0.016 | +0.062 ± 0.012 |
| | panel | −0.016 ± 0.011 | +0.029 ± 0.016 | −0.155 ± 0.020 | +0.108 ± 0.018 |
| **our ranker alone + engine 2 off + forward models off** | mixed | −0.004 ± 0.014 | +0.016 ± 0.013 | −0.163 ± 0.017 | +0.039 ± 0.014 |
| | panel | −0.020 ± 0.013 | +0.008 ± 0.017 | −0.175 ± 0.022 | +0.089 ± 0.019 |

**(ii) Engine 2 with megayak's CC0 networks and rows** (cited, `reviews/eng2fs-2026-10-07.md`, section 3; engine 2's
own list at stage level, 226 panel molecules unseen by both network pairs, rankers out of fold; **no full replay
exists for these arms**): class-2 lists public chain 0.641, megayak chain 0.590, **−0.050 ± 0.014** (22 / 71); class-1
lists −0.016 ± 0.013. The loss is a row-order artefact of his rows; with position-free ties the megayak ranker is
level: +0.011 ± 0.012 (41 / 34) class 2, −0.022 ± 0.014 class 1; the source of the networks then makes no measurable
difference (+0.006 ± 0.009). Nothing was re-run today.

**(iv) `fe_D.pt` off:** no flag exists (`fe_D` is one of three slots of the `model_views` family of the public
ranker). It is removed together with the public ranker pass, arm (iii-a). Not measured separately. Nearest numbers:
`model_views` permutation in our ranker −0.003 to −0.005 on `c2J` (`reviews/fsranker2-2026-10-07.md` 2.3).

**DEV** (2,675 unseen molecules; ranker level only: no engine 2, no forward scores, no fusion are stored there;
`scratch/clean/abl/dev_rankers.py`, reference `avg` = v33b's base list; FINAL not read):

| Arm | `c2J` (n 1,508) | `dropJ` (n 1,063) | `c2N` (n 2,515) | panel part of `c2J` / `dropJ` (140 / 121) |
|---|---|---|---|---|
| public ranker off (`ours` alone) | +0.016 ± 0.005 (196 / 173) | **+0.066 ± 0.006** (307 / 94) | +0.022 ± 0.003 | +0.012 ± 0.019 / +0.098 ± 0.018 |
| our ranker off (`pub` alone) | **−0.083 ± 0.006** (84 / 477) | **−0.102 ± 0.007** (56 / 506) | −0.077 ± 0.004 | −0.131 ± 0.022 / −0.215 ± 0.023 |

Our ranker on DEV is out of fold by scaffold and the public ranker is fitted elsewhere, so the DEV table flatters
`ours`; the fused chain above is the fairer of the two.

Readings.
- **The chain without the fold-safe network cannot be measured locally in the literal sense**: without it the public
  ranker reads the public networks, which have seen every local molecule. Arm (iii-b) is the nearest honest
  stand-in (public ranker on fold-safe rows). On the leaderboard the step was v24a 0.437 → v33b 0.442.
- **Removing the public ranker costs nothing measurable**, locally or on the old leaderboard (v33a `RANKER='fs'`
  0.434 against 0.442, −0.008 with one submission each resolving about ±0.02 at best). That removes `fpnet_full1`,
  `fe_A`, `fe_B`, `fe_D`, `ranker_v4b.pkl` below the gate without a rebuild.
- **Engine 2 is a trade**, as every earlier study found: it carries answers that only its pool holds (`dA`; on the
  hidden test that class is ChEBI / LIPID MAPS-only structures plus whatever its different candidate logic recovers)
  and costs on PubChem-only answers. The hidden mix decides; the one leaderboard probe on its weight
  (`FUSE_ALPHA` 0.6 → 0.3, a teammate's variant, row 33 of `docs/experiments.md`) lost 0.031. Rebuild rather than
  remove, unless a submission with `USE_ENG=False` says otherwise.
- **Forward models** matter where the answer is not a pool row (−0.04 to −0.09) and not elsewhere; their weights
  and checkpoints are MIT, only the runner is unlicensed, so the runner is worth its 2 to 3 days.
- Priority by measured weight: (1) v1 engine code (no measurement possible: without it there is no chain);
  (2) own fold-safe network + our ranker (−0.02 to −0.10 if lost); (3) forward-model runner; (4) engine 2's networks
  and rows; (5) public ranker pass: drop.

## 4. Built today (no GPU)

Folder `solutions/fusion-v4n/clean/` (README with the provenance of every input):

| File | State |
|---|---|
| `pool_build.py` | written and run: see 4.1 |
| `chemcore.py` | own adduct parser and neutral mass, spectrum cleaning, fragment masses; passes the pins below |
| `golden.py` | fixtures written to `D:\casmi-data\clean\golden\` (6 MB): 3 spectra per adduct of train with the stored neutral mass, 400 raw and cached spectra, 3,000 reference pool rows with fingerprints, 400 stored fragment lists, 2,000 rows of `train_structs` |
| `SPEC.md` | specification of pool tables, chemistry core, spectrum cache, fragmenter (complete, verified) and of library / analog search, fingerprint network, engine, feature families, forward runners (contract level, constants not re-verified) |
| `tests/test_clean_contract.py` | 3 data-free tests + 4 golden tests, **7 of 7 pass** (run 2026-10-10): neutral mass of every adduct within 2e-6 Da; 400 of 400 cleaned spectra identical in m/z and within 1e-7 in intensity; 100 of 100 fragment lists identical in length and within 6e-4 Da; 300 of 300 reference pool rows identical in SMILES, key, formula, mass and all 10,226 reference fingerprint bits |

Clean-room caveat (also in `SPEC.md`): the specification and the implementation of the four small modules were
written by the same author, who had read the unlicensed originals. No code was copied; the implementation is written
against RDKit / NumPy and the file formats. Whether that satisfies the rules is for the host to say.

### 4.1 Pool builder: run and coverage

Run on 2026-10-10, 12:09 to 14:27: `sources` 10 s; `structs` 8,262 s of wall time with 4 workers at below-normal
priority (about 9.2 CPU hours; 189 chunks, resumable); `assemble` and `verify` a few minutes. Inputs: `train.parquet`
(275,810 structures, 2,539,608 spectra) and `coconut_csv-10-2026.zip` (738,794 rows → 479,690 after keeping one row
per first InChIKey block → 479,686 parsed, 6,285 of them still charged after neutralisation and dropped). Output:
`scratch/clean/pool/tables/` (1.5 GB: `pool_meta.parquet`, `pool_fp.npy`, `fp_bits.npy`, `train_structs.parquet`,
`train_fp_sel.npy`, `stats.json`); report `scratch/clean/pool/verify.json`. Reference = the existing tables of
`ahmedberatozer/casmi26-v2-pool`.

| Check against the reference pool | Result |
|---|---|
| rows | ours **715,304** (training only 248,310, COCONUT only 441,118, both 25,876); reference 710,701 |
| reference keys covered | **710,125 of 710,701 = 99.92%**: training-only 248,440 of 248,440, both 25,746 of 25,746, COCONUT-only 435,939 of 436,515 (99.87%) |
| `train_structs.parquet` | **275,810 of 275,810 rows identical in every column** (`sid`, SMILES, key, raw key, `n_spec`, `n_tims`, `libmask`) |
| shared rows (710,125) | formula and heavy-atom count identical in all; mass identical except 12 rows (isotope-labelled variants); SMILES identical in 707,377 (99.61%; the other 2,748 are a different representative of the same metric key); row order (mass-sorted) consistent; `src` identical in 709,789 |
| fingerprints | random 20,000 shared rows with the same SMILES: **20,000 identical** on the 9,908 bits both selections contain; 20,000 of all shared rows: 19,916 identical (the 84 others have a different SMILES for the key) |
| bit selection | ours 11,598 bits (frequency in [0.005, 0.995] over all pool rows), reference 10,226, in common 9,908. **The reference selection is not reproduced** (it was drawn on a sample we do not have); own networks are trained on our selection, `--bits ref` exists for comparisons only |
| only in the reference: 576 keys, 520 between 157 and 1,159 Da | 256 are structures our neutralisation leaves charged (zwitterions such as betaines, which the reference keeps); the other 320 were not traced (most likely lost by keeping one COCONUT row per first InChIKey block before standardisation). Both are fixable in `sources` / `standardise`; a re-run of the affected rows costs about 4 CPU hours. **Not fixed today** |
| only in ours: 5,179 keys, all COCONUT rows | 5,039 are ordinary CHNOPS-halogen formulas, mostly above 1,500 Da (peptides, glycosides, lipids: the reference evidently applies a size limit we did not find), 171 have no carbon; **20** are ordinary carbon compounds between 157 and 1,159 Da |

Reading: for the mass range of the hidden test the own pool holds the reference pool minus 520 keys (0.07% of the
reference) plus 20. Not built: `pool_frag_off.npy` / `pool_frag_mass.npy` (the fragmenter passes its pin but is pure
Python; a compiled version is needed for 715,304 rows), the popularity arrays aligned to the new row order
(`dmitriigluzdov/casmi26-pubchem-popularity-prior` is aligned to the reference rows: re-index by key, minutes).

## 5. Schedule

Dependencies: S1 → S2 → S3 → S4 → S7; S5 and S6 are independent of S2 to S4; S8 needs all. GPU hours are for the
RTX 4070 Ti SUPER at duty 0.8; "days" are working days of code, estimates, not measurements.

| Step | What | Needs | GPU h | CPU h | Code days | Expected effect (section 3) | Decision / approval |
|---|---|---|---|---|---|---|---|
| S0 | questions to the host (section 6) and, in parallel, the request to ahmedberatozer for a code licence | | 0 | 0 | 0 | if either succeeds, S2 shrinks to the network class or disappears | **user posts** |
| S1 | own pool tables (`pool_build.py`: built today), fix of the 576 missing keys, fragment tables | | 0 | 9.2 (done) + 4 + 3 | 1 | none (same content) | none; COCONUT file already local |
| S2 | rewrite of the v1 engine core from `SPEC.md` 5 to 7 (library and analog search, network class + collate + data access, candidates and base features, generator), own `fragnet`-type family | S1 | 0 | 2 (parity and benchmark runs) | 10 to 14 | must hold DEV `c2J` 0.788 / `dropJ` 0.655 of the `ours` baseline after S4 | user: accept the reduced design (four families dropped) |
| S3 | two own fingerprint networks (fold-safe split; second seed for the PubChem pair) | network class of S2 (2 days into S2) | 6 to 12 | 2 | 0.5 | hold-out top-1 0.669 / 0.389 / 0.345 ± 0.013 | **GPU schedule** |
| S4 | ranker rows with the new engine and network on the 5,350 hold-fold molecules, refit of our ranker with class-1 lists so that it serves every `lib_max` | S2, S3 | 1 | 10 to 20 | 1 | replaces (iii-b)'s loss; (iii-a) says the public ranker is not needed | none |
| S5 | engine 2: own pair of networks (`train_eng_fpnet.py`), rows regenerated or megayak's CC0 rows with position-free ties | | 1 | 4 | 2 | level with the public chain at stage level in the best case (ii); weaker networks cost an unknown amount | user: rebuild or remove (after P2) |
| S6 | own forward runner on unmodified ms-pred (or prvsiyan's MIT `fwd_runner.py` as base) + parity on stored scores | | 1.5 | 1 | 2 to 3 | keeps −0.04 to −0.09 on non-pool answers | approval if the 16 KB runner file is downloaded |
| S7 | PubChem channel on the own pair; optional own PubChem tier | S3 | 0.5 | 1 (+ 14 for the tier) | 0.5 | none expected | approval for the 2.7 GB PubChem download, if wanted |
| S8 | full local replay of the clean chain, notebook build with only clean inputs, commit run, notices and README | all | 1.5 | 3 | 2 | | **submission approval** |
| | **total** | | **11.5 to 17.5** | **35 to 45** | **18 to 23** | | |

**Earliest date of a fully clean submission.** GPU: 11.5 to 17.5 hours = two days at 12 hours a day; it is never the
constraint. Code: S2 dominates. With one person-equivalent working S2 and S5 / S6 in sequence: 18 to 23 working days
→ **2026-11-02 to 2026-11-09**; with S5 / S6 done in parallel to S2 by a second worker or agent: 12 to 16 days →
**2026-10-26 to 2026-10-30**. If the code question is settled in our favour (author licence or host ruling), S2
shrinks to nothing and S6 to nothing: S3 can start at once (the Apache-2.0 loop then stands), and the clean chain is
S1 + S3 + S4 + S5 + S7 + S8 → **2026-10-16 to 2026-10-19**. The final deadline is 2026-12-14. These are estimates;
none of the code-day figures is a measurement.

**Partially clean notebooks worth one submission each** (each is v59a with one change, so each measures one
replacement on the leaderboard; every one needs the user's approval):

| # | Change | Ready | Why |
|---|---|---|---|
| P1 | `RANKER='fs'` on the new scoring regime (public ranker off below the gate) | now (preset `v33a` with the CC BY set) | confirms on the leaderboard that items 11, 13, 14 can simply be dropped; old regime: 0.434 against 0.442 |
| P2 | `USE_ENG=False` | now | decides rebuild (S5) against removal of engine 2; local: a trade |
| P3 | `USE_ICE=False`, `USE_GL=False` | now | decides whether S6 is worth 2 to 3 days |
| P4 | own fold-safe network + refitted ranker in place of `fpnet_fold0.pt` (old engine code) | after S3 with the Apache loop, about 3 days from a free GPU | measures the network swap alone, before the engine rewrite |
| P5 | engine 2 on megayak's CC0 networks and rows with position-free ties | needs a flag in `eng_runner.select_fp_models` and in `pv._rank_norm` (not written) | measures the (a) alternative for items 16, 17 |

P1 to P3 cost no work; at four submissions a day they fit into one day. P1 + P2 + P3 together are also the lower
bound of a chain that needs no third-party weights except the fold-safe network and the MIT forward checkpoints.

## 6. Decisions and approvals for the user

1. **Post in thread 745072** (drafted; the user posts):

   > Hi David, thank you for the answer above. Three short follow-ups, each answerable with yes or no:
   >
   > 1. **Code.** The source packages `casmi` / `fe_v4` inside `ahmedberatozer/casmi26-v3-models` and
   >    `casmi26-v4b-models`, and the runners `ice_runner.py` / `gl_runner.py` (plus `shim/`) inside
   >    `casmi26-iceberg` / `casmi26-glacier`, carry no code licence (dataset licence "Other"). Is source code shared
   >    publicly for this competition inside a Kaggle dataset covered by Foundational Rule 6.b, or does a
   >    prize-eligible solution need an explicit OSI licence from its author (Rule 6.c)?
   > 2. **Training code.** For third-party checkpoints whose training scripts are not public (`casmi26-fpnet-full1`,
   >    the `fe_A` / `fe_B` / `fe_D` models and the ranker in `casmi26-v4b-models`, `prvsiyan/casmi26-fp-models-v2`),
   >    is it enough under Rule 2.8 to deliver our own code plus references to the public datasets, or must a winner
   >    be able to retrain them?
   > 3. `prvsiyan/casmi26-fp-models-v2` and `prvsiyan/casmi26-ranker-features` are labelled CC BY-NC-SA 4.0 and
   >    described as training-derived. Do they count as "noncommercial only by inheriting from train.parquet"?
   >
   > Thank you!

2. Ask ahmedberatozer for MIT or Apache-2.0 on the code in his datasets (the user's business; already requested by
   another participant in the thread, unanswered). It would save 12 to 17 code days.
3. Accept or reject the reduced design of S2 (own `fragnet`-type family, the other four families dropped).
4. GPU hours for S3 (6 to 12) after the running job.
5. Submissions P1 to P3 (each needs explicit approval).
6. Downloads, none required now:

| What | Source | Size | Licence | Needed for |
|---|---|---|---|---|
| COCONUT October 2026 CSV | coconut.naturalproducts.net/download | **already local** (247,959,854 bytes, downloaded 2026-10-04; the page lists 207.9 MB today, so the file may have been regenerated: re-download only to pin a hash) | CC0 | S1 |
| megayak's two networks and rows | `megayak/casmi26-simulated-ranker-rows` | already local (288 MB + rows) | CC0-1.0 | P5 / S5 (a) |
| `runner/fwd_runner.py` only | `prvsiyan/casmi26-forward-reference-stack` | 16,254 bytes | MIT per the card (dataset field "Other (mixed)") | S6, as a base |
| `CID-SMILES.gz`, `CID-Mass.gz` | ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/ | 1.4 GB + 1.3 GB | public domain | S7, optional own tier |

## 7. Disk

Drive D: 3.0 GB free at the start and at the end of this study (I wrote 6 MB there); drive **C: has 42 GB free**, and
`scratch/clean/` on it holds the pool build (3.0 GB with the raw-fingerprint chunks; the chunks, 1.5 GB, can go once the tables are accepted). The rebuild needs about
1.3 GB (pool tables), 2 × 0.2 GB (networks), 6.4 GB only if the PubChem tier is rebuilt: it fits on C: without
freeing anything. If D: must hold it, these are the largest deletable caches (nothing was deleted; sizes measured
today; "regenerable" means a script in `scratch/` rebuilds it):

| Folder under `D:\casmi-data\` | GB | What it is |
|---|---|---|
| `valtest_fix`, `valtest_c1`, `valtest_f0`, `valtest_comp` | 4.2 + 4.2 + 4.2 + 4.1 | four derived copies of `train.parquet` for the Kaggle validation runs of 09-30 to 10-02 (regenerable; the runs are scored) |
| `hardneg\win_train`, `win_regime`, `w10_hold`, `win_np` | 6.8 + 3.4 + 3.2 + 2.3 | PubChem-window caches of the hard-negative study (regenerable, 2.6 h each family); `hardneg\w10` (8.3) is read by the joint model's loader: keep |
| `frigid` | 6.2 | FRIGID checkpoints and environment (CC BY-NC, not usable) |
| `np_ours\run` | 2.6 | engine-2 fine-tune runs of 10-05 (soups kept elsewhere in the folder: check before deleting) |
| `rankbank\feat` | 2.5 | first feature bank, superseded by `rankbank4` |
| `ranker_study\ds`, `ranker_study\logits` | 2.3 + 1.0 | dataset copies and logits of the first ranker study (`ds\v4b` is the only full local copy of `-v4b-models`: keep until the rewrite is verified) |
| `extra_db` | 2.4 | natural-product catalogues; **keep `raw\coconut_csv-10-2026.zip`** (input of S1) |

`directions\run\work`, `ranker_study\work` and `work_val1` are hard links to the same files; deleting one frees nothing.

## 8. Not verified

- The body of ahmedberatozer's notebooks was read from our pulled copies, not from the live pages; only the licence
  line of `casmi26-v4g-inference` was read live. megayak's "Two Rankers One Engine" and bobthebot369's notebook pages
  were not opened (another megayak notebook was: Apache 2.0).
- That `fe_D.pt`'s tensors descend from `ssl_model.ckpt`; what `fe_D.pt`, `fe_A/B`, `fpnet_full1`, `ranker_v4b` and
  `fe_v4/res` were fitted on beyond the cards' general note; the training data of prvsiyan's and megayak's networks.
- That the re-saved ICEBERG / GLACIER tensors and the shipped wheels equal the upstream files.
- The licence of the Dropbox-hosted MassSpecGym checkpoints themselves (no licence file; the repository is MIT).
- The batch size of the published fold-safe run, hence the GPU hours of S3 (given as a range).
- Every "code days" figure and every date in section 5: estimates.
- The clean-room question itself (same author read and wrote), and whether the host accepts a rewrite "from a
  specification" of unlicensed code: not asked yet.
- Our ranker at every `lib_max` (needed when the public ranker goes): not replayed.
- The size difference between the local COCONUT file and the file listed today.
- SPEC sections 5 to 9: constants not re-verified against the code.
- Whether the compliance check covers earlier submissions (host: "I will get back to you").
