<!-- index: 2026-10-17 compliance of stack1 (omerfarukmerey/casmi26-om-stack1 v1, 0.451) with the host's weights rule of thread 748882: every file the notebook loads with cell / line, publisher, licence (Kaggle metadata of 2026-10-17) and status (a official checkpoint, b participant weights with public training code, c without, d data, e code); new since 10-10: ahmedberatozer's code is MIT (cards of 10-13), prvsiyan's FPNet training loop is public in his Apache notebook but his data builder and split are not, fragnet stage A re-derived exactly and stage B equivalently from the shipped training data; the eleven (c) items grouped by stage with the score at stake (measured ablations, each with its base), our replacements and their cost; a replacement plan in three leaderboard steps with stop rules; the minimum compliant chain and the removal fallback; step 1 (PubChem channel on our two fold-safe networks, gate rescaled by a label-free constant) prepared in scratch/comply/step1 with tests -->
# stack1 against the weights rule: inventory, replacement plan, step 1 (2026-10-17)

Marks: **[V]** read in the code / metadata / page, or computed here (file named); **[C]** cited from a review of ours or a
publisher's statement; **[I]** inference. **No leaderboard estimate is given.** One answer at rank 1 on the public part
(about 124 molecules) = 0.008. We are not lawyers; the host decides eligibility.

What was done: read the pulled stack1 notebook (`scratch/nb1016/omer/`) and every source string embedded in it
(extracted by `scratch/comply/extract.py` to `scratch/comply/stack1_src/`); read the dataset packages from the local copies
(`D:\casmi-data\ranker_study\ds\v4b`, `D:\casmi-data\input\datasets\...`); `kaggle datasets metadata` for the 14 attached
datasets and the fold-safe set (read-only, `scratch/comply/meta/`, fetched 2026-10-17 about 11:10); `kaggle kernels list`
and `kernels pull` (read-only) of five public notebooks of ahmedberatozer, prvsiyan and megayak (`scratch/comply/nb/`); the
GitHub repository named on prvsiyan's cards, signed out. CPU only (at most 3 processes of mine); no GPU; no Kaggle push,
submission, upload or download of datasets / models. Nothing committed.

The rule (host Marie Killian, thread 748882, as verified by the team on 2026-10-16): "Third party weights that cannot be
reproduced with the original training code and permitted training data are not allowed... We reserve the right to check all
submissions and potentially remove from the competition those that do not abide by this rule." Same thread: code shared by
other competitors, including code files inside public Kaggle datasets, may be used if it comes from permissible sources and
complies with applicable licences; official GLACIER / ICEBERG, DreaMS and FRIGID checkpoints are allowed with a reference to
the authors' training code and data; retraining a participant-shared architecture on permitted data with our own delivered
training code is sufficient; non-commercial datasets are allowed.

## 0. Summary

1. **Inventory** (section 2): stack1 loads **2 official checkpoint items (a)** (ICEBERG `gen.pt` + `inten.pt`, GLACIER
   `glacier.pt`), **0 confirmed (b)**, **11 third-party weight items without (complete) public training code (c)** in
   active use (12 files) plus one inert one, **11 data items (d)** and **12 code items (e)**.
2. **The (c) items fall into three coherent groups** (a network and everything fitted on its outputs):
   - **G-E1, engine 1** (the main list): `fpnet_full1.pt` (engine bank), `fe_A.pt`, `fe_B.pt`, `fe_D.pt` (the
     `model_views` slots), `ranker_v4b.pkl` (orders the list), and three fitted tables of the `fe_v4` feature families
     (fragnet `stageA.npz`, `stageB.txt`, derivation `prior_tables.pkl`).
   - **G-PC, PubChem-only channel**: `casmi26-v3-models/fpnet_0.pt`, `fpnet_1.pt` (A+B), and the gates tuned in their raw
     f.z units (`REL_TH` 600, promotion `rel > 150`).
   - **G-E2, engine 2**: `prvsiyan/casmi26-fp-models-v2` `fp_single_s2.pt`, `fp_merged_m1.pt`, and `rank_train.npz`
     (data, but its fingerprint columns come from those networks); the two engine-2 rankers are fitted in the run with
     public code and need nothing else.
3. **New since the 10-10 audits** [V metadata 2026-10-17]: ahmedberatozer's cards (updated 2026-10-13) release all his code
   (`casmi`, `fe_v4`) under **MIT** and state that weights and tables were "built only from the competition's
   `train.parquet` and COCONUT"; no training code was added. prvsiyan's notebook contains his **FPNet training loop**
   (Apache 2.0) but not the data builder or the split. `prvsiyan/coconut-casmi26-candidates` is now labelled
   CC BY-NC-SA 4.0 (was CC BY 4.0; non-commercial data is allowed by the host).
4. **Step 0, done (no submission):** fragnet stage A is re-derived **exactly** from the shipped count tables with the MIT
   `fit_chain` and our driver (13 of 13 arrays identical); stage B, refitted by us on the shipped stage-B rows with the
   parameters stored in `stageB.txt`, is not bit-identical but equivalent (prediction correlation 0.998 on the author's
   validation part; within-group MRR 0.800 against 0.799) [V `C:\Users\hemre\casmi-work\comply\step0\repro_fragnet.json`].
5. **Plan** (section 4): three leaderboard steps, one coherent group each: **step 1 G-PC** (our two fold-safe networks,
   gates rescaled label-free; two variants built because the pre-registered scale constant over-fires the gates), **step 2 G-E1** (our network in the bank and all three view slots,
   our fragnet refit, derivation family dropped, ranker refitted on the existing P4 rows; CPU only), **step 3 G-E2** (two
   engine-2 networks retrained by us with prvsiyan's public loop, ranker rows regenerated; 1 to 4 GPU hours plus a row
   builder we must write). After step 3 the chain loads no third-party weights except the official forward checkpoints.
6. **Removal fallback** (section 5): G-E1 cannot be removed (no list without a fingerprint network and a ranker; no
   official checkpoint fits), so the cheapest compliant stack1 is step 2 + PubChem channel off + engine 2 without its
   fingerprint networks. Its pieces were measured only on other bases (section 5); the total was never measured.
7. **Step 1 is prepared** in `scratch/comply/step1/` (builder, calibration, dataset folder, commands) with
   `solutions/fusion-v4n/tests/test_comply_step1.py` (8 tests, pass); the switch off reproduces stack1's bytes. Calibration
   finding: dividing the gates by the SD-ratio k (23.9) would promote 45% of panel molecules against stack1's 17%; the
   firing-share-matched thresholds (rel 20.0 / 57.2 in our units) are built as the second variant.

## 1. What stack1 is (for orientation)

stack1 = pub-p2 (ahmedberatozer v7c: canqiang's e006 chain + our MetFrag-lite gap re-rank) + E2R2 (engine 2's first
candidate at rank 2) + PubChem promotion at `rel > 150` [C `docs/landscape/two-notebooks-2026-10-16.md`]. Order of work in
the run: setup and hash check (cell 8) → engine 2 in a subprocess (cell 13) → PubChem-only channel in a subprocess
(cell 15) → engine 1 with its feature families and LightGBM ranker, top 60, pool popularity re-rank (cells 17 to 20) →
ICEBERG + GLACIER re-rank inside formula groups (cell 22) → gated PubChem merge and promotion (cell 24) → engine fusion
`1/(3+r1) + 0.6/(3+r2)` and the e006 overlays (cells 26 to 34) → E2R2 (cell 36). Cell numbers are notebook cell indices
of `casmi26-om-stack1.ipynb`; line numbers are 1-based inside the cell (`scratch/comply/stack1_src/cellNN.py`).

## 2. Inventory: every file the notebook loads

Licence = Kaggle field on 2026-10-17 [V `scratch/comply/meta/*/dataset-metadata.json`]; card text quoted where it matters.
"Training code" = where we looked for it (section 2.6 lists the search).

### 2.1 (a) Official checkpoints

| # | File (dataset) | Loaded where | Role | Publisher, licence | Status and evidence |
|---|---|---|---|---|---|
| A1 | `ckpt/gen.pt`, `ckpt/inten.pt` (`ahmedberatozer/casmi26-iceberg`) | cell 22 lines 59-101 (`ice_fuse.run_ice` → `ice_runner.py`) | ICEBERG spectra for same-formula re-ranking of the base, engine-2 and PubChem lists | ms-pred (Coley group, MIT), MassSpecGym `msg_all` weights re-saved by ahmedberatozer; field "other" | **(a)**: the packager's `LICENSE-NOTICE.txt` names the MassSpecGym checkpoint of ms-pred commit 708148c2a8eb [C licence audit item 8]; training code = ms-pred (public, MIT), data = MassSpecGym; host allows official ICEBERG. Not verified: that the re-saved tensors equal the upstream file |
| A2 | `ckpt/glacier.pt` (`ahmedberatozer/casmi26-glacier`) | cell 22 lines 113-126 (`gl_fuse.run_gl`) | GLACIER, [M+H]+ spectra only | ms-pred MassSpecGym GLACIER checkpoint, sha256 prefix `5a47cecc` recorded independently on prvsiyan's forward stack card; field "other" | **(a)**, same evidence class [C audit item 9] |

### 2.2 (c) Third-party weights without complete public training code

| # | File (dataset, licence field) | Loaded where | Role in the chain | Publisher's statement on training | Where we looked for training code | Status |
|---|---|---|---|---|---|---|
| C1 | `fpnet_full1.pt` (`ahmedberatozer/casmi26-fpnet-full1`, other) | cell 18 lines 11-13 (`fpnet.ModelBank` = engine bank) | engine 1: candidate f.z, the 94 base features (`fz`, ranks, gaps), lead of `frag2` / `fragnet`, frag set of `fragnet` (top 400 by f.z), generated-candidate scoring | card: "trained on ALL train folds incl. np-examples ... Weights only; the model code is the MIT-licensed `casmi` package" | card (no code, no link); `casmi` package: `fpnet.py` (model, `save_ckpt`, `load_model`), `traindata.py` (data access, no optimiser, no loop); his 16 CASMI notebooks listed by `kaggle kernels list --user ahmedberatozer` (none trains; `casmi26-v2-baseline` and `-v3-inference` pulled and grepped: no optimiser, no `backward`, no `fit`); the fold-safe card states: "Its public packages contain the model and the data access, and no training loop" | **(c)** |
| C2 | `fe_A.pt` (`casmi26-v4b-models`, other) | cell 18 line 15 (`MAN['fe_models']` → `v1engine.V1FE`, slot A) | `model_views` family: `fzABD_gap`, `fzABD_rank`, `rank_spread`, `rank_worst` (4 of the ranker's 160 features); the per-spectrum and per-block A+B views are computed but not used by `ranker_v4b` | `MANIFEST.json`: source `work/fpnet_a.pt`; card: built only from `train.parquet` and COCONUT | as C1 | **(c)** |
| C3 | `fe_B.pt` (same) | same, slot B | as C2 | `MANIFEST.json`: `work/fpnet_b.pt` | as C1 | **(c)** |
| C4 | `fe_D.pt` (same) | same, slot D (`fpnet6.ModelBank`, DreaMS architecture) | `model_views`: `fzD_gap`, `fzD_rank`, `fzD_z` and the D part of the four features above (7 features in all) | `LICENSE-NOTICE.txt`: fine-tuned from DreaMS; `dreams_model.py` docstring and the stored arguments name the self-supervised `ssl_model.ckpt` (official, allowed) [C clean-chain plan C3]; fine-tuning data stated only generally | `dreams_model.py` builds the model (`build_dreams_fp`) and has a checkpointing flag for training, but no loss, optimiser or loop; same notebooks as C1 | **(c)** (official base, unpublished fine-tune) |
| C5 | `ranker_v4b.pkl` (same) | cell 18 lines 20-28 (4 LightGBM boosters, 160 features), applied in cell 20 line 29 | orders engine 1's candidate list before the top-60 cut: the base list every later stage starts from | `MANIFEST.json`: `work/analysis/v1fe/ranker_v4b.pkl`, 500 rounds; cell 8 docstring: "a ranker retrained on the v1-engine simulation sim1x" | `casmi/ranker.py` has `RankerBag.fit` (a fit routine) and the pickle holds the parameters, but the training rows and the simulation that made them are not published anywhere we found | **(c)** |
| C6 | `fpnet_0.pt`, `fpnet_1.pt` (`ahmedberatozer/casmi26-v3-models`, other) | cell 15, `pc_runner.py` lines 22-24 (`ModelBank` of both, logits averaged) | PubChem-only channel: pass-1 / pass-2 ranking of the PubChem window, `fz_top`, `best_pool_fz`, hence `rel`, the slot choice (`REL_TH` 600) and the promotion `rel > 150` (cell 24 lines 57-66) | card (10-13): code MIT; "Model weights and tables were built only from the competition's `train.parquet` and COCONUT" | as C1 | **(c)** |
| C7 | `fp_single_s2.pt` (`prvsiyan/casmi26-fp-models-v2`, **CC BY-NC-SA 4.0**) | cell 13, `eng_runner.py` line 16 → `pv_fp.load_fp_models` (`casmi_engine.main`) | engine 2: per-spectrum logits `zlog` → 10 of the 31 columns of prvsiyan's ranker (15-20, 25-30 of `rank_train.npz`); that ranker has weight 0.88 in engine 2's blend | card: "does not document a training split, an upstream corpus snapshot or a held-out evaluation"; earlier card: trained on the competition training data | **found**: the training loop (`TRAIN_SCRIPT_SRC`, raw cell 18, written to `train_fingerprint_model.py` by cell 47; Apache 2.0, "Licence of this notebook's code: Apache 2.0") and the model (`fpmodel.py`) in `prvsiyan/analog-propagation-casmi-2026-baseline` (pulled 2026-10-17); its checkpoint keys (`model`, `step`, `nbits`, `d`, `layers`) match `fp_single_s2.pt` / `fp_merged_m1.pt` (steps 20,000 / 24,000; 6,930 bits, d 512, 6 layers) [V]. **Not found**: the builder of its inputs `ds/spec.npz` (with the `is_val` split) and `up_train/fp_meta.pkl` (the notebook refers to `build_dataset.py` only by hash), the exact arguments of the two runs; his training notebook `prvsiyan/casmi-fp-trainer-merged-spectra` is referenced in the notebook but private (`kernels pull`: 403) and concerns a later 10,395-bit model; GitHub `cubres/enveda-casmi-2026` (no licence; root, `src/`, `casmi-fingerprint-integration-guide/` read: no training code or split for these checkpoints) | **(c), borderline (b)**: loop public, data pipeline and split not |
| C8 | `fp_merged_m1.pt` (same) | same (merged view) | as C7 | as C7 | as C7 | **(c), borderline (b)** |
| C9 | `fe_v4/res/fragnet/stageB.txt` (`casmi26-v4b-models`) | `fragnet.prepare` via `V1FE` (cell 18 line 15) | LightGBM lambdarank (300 trees) giving the 14 `fragnet__fn_*` features of the ranker | docstring: "stage B (--step fitB): LightGBM lambdarank on 31 stage-B features"; driver `res_fragnet.py` not shipped | shipped: the training rows `featB.npz` (383,923 rows, parts B / V), the feature function `stageB_features`, the parameters inside `stageB.txt` | **(c)**; **re-fitted by us, equivalent** (step 0) |
| C10 | `fe_v4/res/fragnet/stageA.npz` (same) | same | count-based fragment log-likelihood tables feeding stage B and the `fn_raw_*` features | docstring: chains A, H, alpha 80, from `countsA` | shipped: `countsA.npz` and the MIT `fit_chain` / `save_stageA`; the counting driver (`res_fragnet --step countA`) is not | **(c) fitted table**; **re-derived exactly by us** from the shipped counts (step 0) |
| C11 | `fe_v4/res/derivation/prior_tables.pkl` (same) | `derivation` family (cell 18 lines 16-19 incl. its self-test) | class-3 priors behind the 19 `derivation__*` features | docstring: "built from random COCONUT parents ... Verified against the original by res_derivation.py"; table meta: 3,378,502 rows, 29,994 parents, built 2026-09-23 | `res_derivation.py` and `class3_v2/priors.py` are not shipped; `derivation.py` holds the lookup and a "verbatim port" of the prior formulas but no table builder | **(c) fitted table** |
| (C0) | `fpnet_0.pt` of `casmi26-v4b-models` | cell 8 lines 47-55 (sha256 only); cell 18 line 12 only if `fpnet_full1` is missing | none in stack1 | | | inert |

### 2.3 (d) Data and tables

| # | Files (dataset, licence) | Loaded where | Status |
|---|---|---|---|
| D1 | `train.parquet`, `test.parquet`, `sample_submission.csv` (competition, CC BY-NC 4.0) | cells 8, 9, 11, 13, 15, 17, 20, 24 | **(d)** permitted |
| D2 | `pool_meta.parquet`, `pool_fp.npy`, `pool_frag_off.npy`, `pool_frag_mass.npy`, `fp_bits.npy`, `train_structs.parquet`, `train_fp_sel.npy` (`ahmedberatozer/casmi26-v2-pool`, other; card: "Built only from the competition's `train.parquet` and COCONUT") | cell 17 lines 4-8; `pc_runner.py` lines 26-29 | **(d)**; our own builder reproduces 99.92% of its keys and all 10,226 bits on shared rows [C clean-chain plan 4.1] |
| D3 | `pc_mass.npy`, `pc_off.npy`, `pc_smiles.npy` (`ahmedberatozer/casmi26-pubchem-tier`, other) | cell 8 line 45; `probe_core2.init_worker` | **(d)** PubChem, public domain |
| D4 | `pc_lsid.npy`, `pc_lpmid.npy`, `pool_lsid.npy`, `pool_lpmid.npy` (`dmitriigluzdov/casmi26-pubchem-popularity-prior`, other; PubChem Extras, build scripts included) | cell 8 line 29, cell 15 (V17 patch), cell 20 line 7 | **(d)** |
| D5 | `coco_fp.npy`, `coco_mass.npy`, `coco_meta.pkl`, `fp_bits.npy` (`prvsiyan/coconut-casmi26-candidates`, now **CC BY-NC-SA 4.0**) | `casmi_engine.build_pool` lines 147-150; `eng_runner.py` line 11 | **(d)**; COCONUT upstream CC0 [C clean-chain C6]; NC labels allowed by the host |
| D6 | `bio_fp.npy`, `bio_mass.npy`, `bio_meta.pkl` (`sergiomalv/casmi-chebi-lmsd-structures`, CC BY 4.0) | `casmi_engine` lines 152-155 | **(d)**, attribution owed |
| D7 | `rank_train.npz` (`prvsiyan/casmi26-ranker-features`, CC BY-NC-SA 4.0) | `casmi_engine.fit_pv_rankers` line 407: 8 `HistGradientBoostingClassifier` fitted in the run | **(d), tied to C7 / C8**: card changelog "v5: 31 features, m1 merged-spectrum model channel" and "The builder is not included" [V]; its fingerprint columns are outputs of the networks being replaced, so it has to be regenerated with them |
| D8 | `sim_rank_rows_nofp.npz` (`megayak/casmi26-simulated-ranker-rows`, CC0-1.0) | `casmi_engine.fit_rankers` line 398 (4 models fitted in the run, weight 0.12) | **(d)**; no fingerprint columns. The dataset's `ours_fpnet_*.pt` are not loaded (glob `fp_*.pt` restricted to `casmi26-fp-models-v2`) |
| D9 | `MANIFEST.json`, `fe_v4/res/derivation/parity_fixture.pkl` (`casmi26-v4b-models`) | cell 8 lines 41-55; cell 18 line 17 | **(d)** |
| D10 | 23 deterministic `fe_v4/res` tables: `analog_struct/*` (14), `frag2/caps.txt`, `pool_cap.npy`, `pool_f1_off.npy`, `pool_f1_mass.npy`, `fragnet/rec_*` (5) | family `prepare()` via `V1FE` | **(d)**: per-structure descriptors and fragment records of the pool (no fitting) [V the families' docstrings] |
| D11 | `fragnet/countsA.npz`, `featB.npz`, `sets.pkl` | cell 8 (sha256 only) | **(d)**: the fragnet training data (32,000 training structures); used by our step 0 |

### 2.4 (e) Code

| # | Code | Where | Licence | Status |
|---|---|---|---|---|
| E1 | `casmi` + `fe_v4` packages of `casmi26-v4b-models` (engine, families, ranker, chemistry) | cells 17-18 | **MIT** on the card since 2026-10-13 ("All source code in this dataset written by Ahmed Berat Özer ... is released under the MIT License") | **(e)** clean |
| E2 | `casmi` snapshot of `casmi26-v3-models` | cell 15 (`pc_runner.py`) | **MIT** (card, 10-13) | clean |
| E3 | `casmi/dreams_model.py`, `fpnet6.py` | via `v1engine` | MIT (DreaMS re-implementation, notice shipped) | clean |
| E4 | `ice_runner.py`, `fuse.py`, `gl_runner.py`, `gl_fuse.py`, `shim/**` | cell 22 | **no licence** (`LICENSE-NOTICE.txt`: "own code"; the 10-13 MIT statement is on the v3 / v4b / full1 / v2-pool cards only; the iceberg and glacier cards were not changed) | **(e) open**: allowed as code in a public dataset only if licensing permits; an MIT replacement exists (`prvsiyan/casmi26-forward-reference-stack/runner/fwd_runner.py`, card: MIT) [C clean-chain plan] |
| E5 | `src/ms_pred/**` | cell 22 (runners) | MIT (ms-pred) | clean |
| E6 | wheels inside `casmi26-iceberg` (RDKit 2025.3.6, pytorch_lightning, torchmetrics, lightning_utilities, einops) | cell 8 lines 57-74, cell 22 lines 25-50 | BSD-3 / Apache-2.0 / MIT per notice | clean (hashes against PyPI not checked) |
| E7 | `rdkit-2026.3.3` wheel (`metric/rdkit-2026-3-3-wheel`, field "unknown") | cell 8 lines 31-37 | BSD-3 | clean |
| E8 | `pv.py`, `pv_fp.py`, `casmi_engine.py`, `eng_runner.py` (embedded strings) | cell 13 | Apache 2.0 notebooks (prvsiyan, megayak; ours) | clean |
| E9 | `probe_core2` (CORE + V17 patch), `pc_runner.py` (embedded) | cell 15 | Apache 2.0 (ahmedberatozer's notebooks) | clean |
| E10 | `frag_rescore.py` (embedded) | cell 33 | ours, Apache 2.0 | clean |
| E11 | the notebook's own cells (e006 / v7c / v11 lineage, metric identity, E2R2) | all | Apache 2.0 (lineage in cell 0) | clean |
| E12 | image packages (torch, LightGBM, scikit-learn, numba, pandas, pyarrow) | | BSD / MIT / Apache | clean |

Inert or absent: `casmi26-v3-models/ranker_0.pkl` (cell 8 line 43 only locates the folder), C0, megayak's two networks,
`dreams_np_mass.npy` (cell 29 globs for a hengck23 dataset that is not attached).

### 2.5 Counts

| Status | Items | Files |
|---|---|---|
| (a) official checkpoint | 2 | 3 |
| (b) participant weights with public training code and permitted data | **0 confirmed** (C7 / C8 borderline; C9 / C10 reproducible by us after step 0) | 0 |
| (c) participant weights without (complete) public training code, in use | **11** (C1 to C11) | 12 |
| (c) inert | 1 (C0) | 1 |
| (d) data / tables | 11 | about 50 |
| (e) code | 12 (one with an open licence question: E4) | |

### 2.6 Where training code was searched

- Dataset cards and file lists of the 14 inputs (metadata 2026-10-17) and the local copies of `casmi26-v4b-models`,
  `-v3-models`, `-fpnet-full1` (grep for `optim`, `backward`, `def train`, `def fit`, `__main__`, `argparse`, `--step`
  over all `code/**`): fit routines exist only in `casmi/ranker.py` (`RankerBag.fit`) and `fe_v4/fams/fragnet.py`
  (`fit_chain`, `save_stageA`); the drivers they name (`res_fragnet.py`, `res_derivation.py`, `res_analog_struct.py`,
  `res_frag2.py`, `work/analysis/...`) are not shipped.
- `kaggle kernels list --user` for ahmedberatozer (17 CASMI notebooks), prvsiyan (3 CASMI notebooks) and megayak (2 CASMI
  notebooks); pulled and grepped `casmi26-v2-baseline`, `casmi26-v3-inference`, all three of prvsiyan's
  (`analog-propagation-casmi-2026-baseline`, `casmi-library-only-probe-v1`, `casmi-calibrated-ranker-v3`) and both of
  megayak's (`casmi26-two-rankers-one-engine`; `casmi26-one-engine-nine-scores` from the eng2fs pull) for optimisers,
  `backward`, `np.savez` (row writers) and fits. Only prvsiyan's baseline holds a training loop (C7 / C8); no notebook writes
  ranker rows (D7, D8 are fitted on, never built, in public code).
  Earlier pulls of nine ahmedberatozer notebooks found no engine source at all [C clean-chain plan C5].
- GitHub `github.com/cubres/enveda-casmi-2026` (linked on prvsiyan's cards), signed out: no licence; no FPNet training
  script or split for these checkpoints in the parts read (root listing, `src/`, `casmi-fingerprint-integration-guide/`,
  whose README says it includes "no weights, inference launcher, or chemical data"). The full tree was not read.
- `dmitriigluzdov/casmi26-fold-safe-fpnet` card: "Its public packages contain the model and the data access, and no
  training loop" (third-party statement about C1 / C2 / C3 / C6).

## 3. The (c) groups: what rides on them, what we have, what it costs

Measured numbers are paired differences ± one standard error unless they are leaderboard scores; the base is named each time.

### 3.1 G-E1 (engine 1): C1 to C5, C9 to C11

**Role.** Everything engine 1 contributes: the candidate set's f.z features (C1), 160 ranker features of which 7 come
from C2 to C4, 14 from C9 / C10 and 19 from C11, and the order of the 60-row base list (C5). The base list enters the
fusion at weight 1 (engine 2: 0.6) and is what ICEBERG / GLACIER, the PubChem merge and the promotion act on.

**What rides on it (no measurement on stack1; nearest ones):**
- v67e, e006r's chain fed with **our** base list (rank average with our fitted ranker below `lib_max` 0.7): **0.411 against
  0.443** (−0.032; base e006r; diag67 found no execution fault, most plausibly a real loss) [C `docs/experiments.md` row 55,
  `reviews/diag67-2026-10-12.md`]. That swapped far more than G-E1 (our whole v59a base list), but it is the only
  leaderboard number for replacing engine 1's list on this chain family.
- public ranker pass off on the fusion-v4n chain (v33a, our ranker alone below the gate): 0.434 against 0.442 (old scoring;
  one submission each); locally within ±0.02 everywhere [C clean-chain plan section 3 (iii-a)].
- our fold-safe network against the public fold-safe one, with the ranker refitted on its rows (same 165 features):
  level on DEV and FINAL (e.g. c2J −0.005 ± 0.004 / +0.005 ± 0.004) [C `reviews/ownfp-2026-10-10.md` section 4]. Base: the
  fusion-v4n second pass, not stack1's `fpnet_full1` + `ranker_v4b`, which cannot be measured locally (the public networks
  have seen every local molecule).
- `fpnet_full1` as engine bank against FPNet A: 0.378 against 0.380 (ahmedberatozer's v4m / v4l, his notebook's docstring,
  cell 8 line 1; not ours).
- `model_views` permutation in our ranker: −0.003 to −0.005 on c2J [C fsranker2]; `derivation`: +0.0035 [−0.003, +0.010]
  overall, +0.048 on his class-3 groups (the author's own benchmark, `derivation.py` docstring; not ours).

**What we have.** Our network `base_s0` (`solutions/fusion-v4n/clean/`: model, data, trainer; same class, same 10,226
bits, same checkpoint layout, loads in the MIT `casmi.fpnet`) and `base_s1` [C ownfp]. **Drop-in as networks**, not as
scores: a fold-safe network is less sharp than one trained on every molecule (the logit scale differs, section 4.1), so
every model fitted on C1 to C4 outputs must be refitted. The P4 rows (`C:\Users\hemre\casmi-work\ownfp\p4\base_s0\feat\`,
6,694 training lists, our network in the bank and all three view slots) contain **all 160 features of `ranker_v4b`**
(the 165 of our ranker = the 160 + 5 join features) [V]; a 160-feature (or 141 without derivation) ranker can be refitted
from them with `scratch/ownfp/p4_bench.py`'s recipe, joined PubChem rows excluded (stack1's engine-1 lists have none).
Fragnet: step 0 (C9, C10). Derivation (C11): no builder; drop the family (19 features) or write one.

**Cost.** GPU 0 (networks exist; a retrain is 2.5 to 3.3 GPU hours each at duty 0.8 [C ownfp]). CPU: ranker refit, 4 seeds ×
500 rounds on about 0.98 M rows, 3 threads (not timed; the P4 fit of the same size ran inside one session); a local
feature-parity check of the patched `V1FE` against the P4 rows on about 50 molecules (a few CPU hours). Code: notebook patch
of cells 8 and 18 (MANIFEST check, bank path, `fe_models` paths, family list, ranker file, `fragnet` resource override);
about one day. If C11 is to be kept: a builder for the prior tables, 1 to 2 days [I].

### 3.2 G-PC (PubChem-only channel): C6

**Role.** Which PubChem-only structures are listed (pass 1 / pass 2 by f.z of A+B), and the two raw-f.z gates: `rel =
fz_top − best_pool_fz` decides aggressive slots (`rel > REL_TH` = 600) and promotion to rank 1 (`rel > 150`). `S` is
z-scored and popularity is not network-dependent.

**What rides on it:** the promotion threshold alone moved +0.003 to +0.006 on four public bases (rel 250 → 150)
[C two-notebooks section 5]; e006's own notebook credits the promotion rule with 0.417 against 0.374 (an upstream comment,
cell 24, not ours). **The one measurement of replacing these networks on stack1**: stack1-v71b (our multi-spectrum
network `cd_pairmulti`, 2-seed mean, per-molecule rescaling of the gate) scored **0.439 against 0.451** (−0.012), while the
local replay said +0.007 ± 0.004 / +0.009 ± 0.003 [C two-notebooks section 7]. Its cause is not known (the v71 diagnosis
is another agent's open work).

**What we have.** `base_s0`, `base_s1` (two seeds, full published length): same class and bits as A / B, so `pc_runner.py`
runs unchanged with a different model folder; **not scale-compatible** with A+B (section 4.1), so the two gates must be
rescaled. No model is fitted on the channel's output, so no refit.

**Cost.** GPU 0; CPU about 1 hour (calibration); one private dataset upload (0.4 GB, user); one commit run.

### 3.3 G-E2 (engine 2): C7, C8 (+ D7)

**Role.** The two networks give `zlog`, which feeds 10 of the 31 columns of prvsiyan's ranker (weight 0.88 of engine 2's
blend); engine 2's candidates (window ±10 ppm of COCONUT + ChEBI / LIPID MAPS + training structures) and its other ranker
(megayak's rows, no fingerprint columns) do not depend on them. Engine 2's first candidate is also E2R2's rank-2 insert.

**What rides on it:** engine 2 off (fusion-v4n design, local): −0.149 ± 0.014 / −0.151 ± 0.017 where the answer is only in
engine 2's pool (dA), +0.052 / +0.112 where it is PubChem-only (dB), ±0.01 elsewhere [C clean-chain plan section 3]; its
fusion weight halved lost 0.031 on the leaderboard (fusion-v4n base, a teammate's variant) [C same]; E2R2 +0.001 to +0.002
on three bases [C two-notebooks]. Replacing the networks only (megayak's leak-free pair, position-free ties): level with the
public chain at stage level, +0.011 ± 0.012 [C `reviews/eng2fs-2026-10-07.md`]; megayak's pair has no public training code
either, so it is not a compliant replacement.

**What we have.** `train/train_eng_fpnet.py` (ours; prvsiyan's model class `pv_fp.FPNet` from his Apache notebook): a
20,000-step run from scratch took 0.30 GPU hours and reached 0.68 to 0.84 on seen libraries against 0.78 to 0.92 for the
public model [C `train/README.md` question 4]; prvsiyan's own public loop (`train_fingerprint_model.py`, needs a `spec.npz`
builder we would write; his note: about 2 hours on a T4 per model). The pool fingerprints of engine 2 exist locally
(`scratch/mixed/eng2_local.py` caches). **Not drop-in**: D7 must be regenerated with the new networks, and its builder is
not public (card); our local regime harness computes the 31-column block (`pv.rank_features`) for local molecules, the
class-1 / class-2 simulation sets would have to be rebuilt [I].

**Cost.** GPU 1 to 4 hours (two networks); code 1 to 2 days for the row builder [I]; CPU several hours for the rows.

## 4. Replacement plan

Rules for every step: one coherent group per submission, built as a switch on stack1 (switch off = stack1's bytes); a
patched-off commit run must reproduce stack1's commit output (sha256 `2b32bc61…`) before the "on" notebook is pushed; a
submission needs the user's approval. Stop rules are about what to do next, since each step is required for compliance:
"accept" = keep as the new base; "retry" = one pre-registered alternative, then keep the better of the two.

### 4.0 Step 0 (done, no submission): fragnet reproducible from the shipped data

`scratch/comply/step0/repro_fragnet.py` [V `C:\Users\hemre\casmi-work\comply\step0\repro_fragnet.json`]:

| Artefact | Re-derived how | Result |
|---|---|---|
| `stageA.npz` | `fragnet.save_stageA(countsA)` (MIT functions, chains A + H, alpha 80) | **identical**: 13 of 13 arrays, maximum difference 0.0, chains equal |
| `stageB.txt` | LightGBM 4.6.0 lambdarank on part B of `featB.npz` (12,743 groups, 308,604 rows; features `stageB_features(agg_AH80)`), the parameters stored in `stageB.txt`, 3 threads | 300 trees, not bit-identical; on part V (3,103 groups): prediction correlation 0.998, maximum difference 0.53, within-group MRR **0.800 (ours) against 0.799 (shipped)** |

Not re-derived: `countsA.npz` itself (the counting driver is not shipped; its inputs `sets.pkl` and the training spectra
are). So stage A is reproducible from shipped train-derived counts, stage B from shipped train-derived rows, each with our
driver on MIT functions.

### 4.1 Step 1: G-PC on our networks (prepared, section 6)

Change: `pc_runner.py` loads `fpnet_own_s0.pt` + `fpnet_own_s1.pt` (`huseyinemreaksoy/casmi26-ours-pcnets`, to be
uploaded) instead of A+B; `REL_TH` 600 → 600 / k and the promotion 150 → 150 / k (primary) or both gates at the
quantile-matched thresholds (retry). k is label-free (pre-registered in
`scratch/comply/step1/PREREG.md`): the median over the Enveda natural-product panel molecules of SD(A+B f.z) / SD(our f.z)
over the pool window (`calib_k.py`).

Calibration [V `C:\Users\hemre\casmi-work\comply\step1_calib\calib_k.json`, 68 CPU minutes; spectra only]:

| Quantity | Enveda panel (262 molecules; 257 with a PubChem-only list under both pairs) | hold-out fold sample (296) |
|---|---|---|
| k = median SD(A+B f.z) / SD(our f.z) over the pool window (IQR) | **23.92** (17.35 to 31.99) | 15.95 (13.15 to 20.22) |
| `rel` quantiles 5 / 25 / 50 / 75 / 95 %, A+B | −331 / −135 / −19 / 92 / 320 | not computed |
| `rel` quantiles, ours | −14.6 / −3.2 / 4.1 / 15.0 / 36.4 | |
| share promoted by `rel`: A+B at 150 / ours at 150 / k = 6.27 / ours at the matched 20.04 | 16.7% / **45.1%** / 16.7% | |
| share in aggressive slots: A+B at 600 / ours at 600 / k = 25.1 / ours at the matched 57.2 | 1.2% / **12.5%** / 1.2% | |
| promotion decision equal under A+B at 150 and ours at 150 / k (same molecule) | 66% | |

Reading. The SD ratio does not transfer the gates: our `rel` distribution is relatively heavier in its upper tail, so
150 / k would promote 2.7 times and 600 / k open the aggressive slots 10 times as often as stack1 does on the same
molecules. The quantile-matched thresholds (20.04, 57.24) differ from 150 / k and 600 / k by more than the factor 1.5 of
`PREREG.md`. k also differs between the two sets (23.9 against 16.0), which fits A+B having seen one of them [I: the A / B
training split is not documented]. Both variants are built: **`casmi26-comply-s1`** (pre-registered primary, gates / k) and
**`casmi26-comply-s1m`** (pre-registered retry, matched thresholds). The pre-registration fixes the order (primary first);
the firing-share table is label-free and was written before any outcome exists, so the user may, with that table, choose to
submit `s1m` first instead; that choice is the user's and is recorded here as a deviation candidate, not taken.

Expected effect: **no measured expectation on this base.** The evidence is v71b (−0.012, a different network and
rescaling) against ownfp's network-level parity; the gate transfer is the main unknown.
Stop rule: Δ ≥ −0.008 (within one answer) → accept. −0.016 ≤ Δ < −0.008 → retry once with the other built variant
(`s1m` after `s1`, or `s1` after `s1m`), keep the better. Δ < −0.016 → do not retry blindly: a full-size validation pair (switch on / off, same
400 molecules) to check the channel's firing counts and lists, then decide; this also feeds the v71 diagnosis.

### 4.2 Step 2: G-E1 on our network, refitted ranker

Change (one switch): engine bank `fpnet_full1` → `base_s0`; `fe_A` / `fe_B` / `fe_D` slots → `base_s0` (the P4 design;
alternative pre-registered for the retry: `base_s0` / `base_s1` / `a0_ctrl_s0`, which needs new rows); `fragnet` reads our
`stageA` / `stageB` (step 0); `derivation` dropped from `MAN['families']` (its self-test then skips); `ranker_v4b.pkl` →
our 141-feature refit on the P4 rows without joined rows; the MANIFEST hash check restricted to the files still read.
Before building: the patched `V1FE` on 50 local molecules must reproduce the P4 feature rows (the families' features to
float tolerance, except fragnet within the step-0 equivalence).
Expected effect: no measurement on stack1 (section 3.1 lists the nearest: −0.032 for a much larger swap, ±0.02 locally for
the ranker pass, level at ranker level for the network).
Stop rule: Δ ≥ −0.008 → accept. Otherwise retry once with derivation kept as our own rebuilt table or with the
three-network view variant, whichever local CV on DEV prefers; Δ < −0.016 after the retry → full-size validation pair to
locate the loss before anything else.
Cost: section 3.1. Commit run exercises the change (the ranker orders all 12 commit molecules); runtime is unchanged.

### 4.3 Step 3: G-E2 on our networks, regenerated rows

Change: `fp_single_s2.pt` / `fp_merged_m1.pt` → two networks trained by us with prvsiyan's public loop (single view
`--merge_p 0`, merged view `--merge_p 0.6`, best-by-validation as his notes say) on `train.parquet` with our own split
(identity-disjoint, Enveda panel held out); `rank_train.npz` → rows regenerated by our builder with those networks (same 31
columns, `pv.rank_features`); `fit_pv_rankers` unchanged. One submission.
Interim, if step 3 is not ready: **3b** `w_pv = 0` (engine 2 ranks with megayak's no-fingerprint rows only; no network at
all) as its own submission; not measured anywhere; megayak's rows carry the row-order artefact [C eng2fs].
Stop rule: Δ ≥ −0.008 → accept; otherwise compare with 3b if it exists, keep the better; Δ < −0.016 → validation pair.
Cost: section 3.3.

### 4.4 Minimum compliant chain

stack1 after steps 1 to 3: our `base_s0` (+ `base_s1` in the channel) wherever A, B, D or `full1` were; our ranker refit;
our fragnet refit; derivation dropped or rebuilt; our engine-2 networks and rows; official ICEBERG / GLACIER checkpoints;
every other part of stack1 (pools, PubChem tier, popularity, COCONUT, ChEBI / LIPID MAPS, megayak's CC0 rows, fusion,
frag gap, E2R2, overlays) unchanged. Open beyond weights: the ICEBERG / GLACIER runner and shim code has no licence (E4);
the MIT runner of `prvsiyan/casmi26-forward-reference-stack` is the replacement if the host requires one (not tested).
Every weight file we ship would then have training code in our repository (`clean/train_fpnet.py`, the step-0 driver, the
ranker fit, the engine-2 trainer, the row builder) and permitted data (`train.parquet`, COCONUT, PubChem).

## 5. Fallback: remove instead of retrain

| Group | Removal | What it costs, measured (base) |
|---|---|---|
| G-PC | `PC = {}`: no PubChem-only rows, no promotion | not measured on any base; related: rel 250 → 150 +0.003 to +0.006 (four public bases); e006's notebook credits the promotion with 0.417 against 0.374 (upstream comment) |
| G-E2 | `w_pv = 0` (no networks) or engine 2 off | `w_pv = 0`: not measured. Engine 2 off: dA −0.15, dB +0.05 to +0.11 (fusion-v4n design, local); fusion weight halved −0.031 (fusion-v4n, leaderboard); E2R2 lost (+0.001 to +0.002 on three bases) |
| G-E1 | **not possible**: engine 1 needs a fingerprint network and a ranker; no official checkpoint (ICEBERG, GLACIER, DreaMS `ssl_model.ckpt`, FRIGID) gives fingerprint logits for this engine | the cheapest compliant engine 1 is step 2 itself |

So the cheapest compliant stack1 is step 2 + PubChem channel off + engine 2 with `w_pv = 0`; its total cost was never
measured and the pieces above come from other bases.

## 6. Step 1, prepared (not run on Kaggle)

| File | What |
|---|---|
| `scratch/comply/step1/PREREG.md` | the calibration rule, written before any number |
| `scratch/comply/step1/calib_k.py` | label-free k and gate firing shares (CPU, 3 workers); output `C:\Users\hemre\casmi-work\comply\step1_calib\` |
| `scratch/comply/step1/build_step1.py` | builder: `off` = stack1's bytes; `patched-off` and `on` = one inserted cell after the commit switch + two one-line substitutions (cell 15: `V3MOD` → `COMPLY_PC_MODELS` in the runner call; cell 24: `rel > 150.0` → `rel > COMPLY_REL_PROMO`). With the switch on, a missing dataset, a hash mismatch or a third `fpnet_*.pt` stops the run (no silent fallback to A+B) |
| `scratch/comply/step1/nb/` | the four built notebooks (`casmi26-om-stack1` = stack1's bytes, `casmi26-comply-s1-off`, `casmi26-comply-s1`, `casmi26-comply-s1m`) with `kernel-metadata.json` (private, T4, stack1's image, inputs + `huseyinemreaksoy/casmi26-ours-pcnets`) |
| `C:\Users\hemre\casmi-work\comply\step1\ds\casmi26-ours-pcnets\` | dataset folder for the user to upload: `fpnet_own_s0.pt` (sha256 `4714a6e7…`), `fpnet_own_s1.pt` (`88d86a36…`), `code/` (our trainer, model, data access), `ledger.json`, README, metadata |
| `scratch/comply/step1/commands.sh` | the commands, including the user's Kaggle lines |
| `solutions/fusion-v4n/tests/test_comply_step1.py` | 8 tests, no competition data: anchors replaced exactly and nothing else, ambiguous anchor refused, switch off = stack1's values (`V3MOD`, 150.0, 600.0), switch on rescales and pins, matched thresholds act only with the switch on, never falls back, gate comparison identical when off, real notebook: `off` bytes identical, exactly two cells changed, every code cell parses. **All pass** (`python solutions/fusion-v4n/tests/test_comply_step1.py`) |

Training commands: none needed (the networks are `base_s0` / `base_s1` of the ownfp study, 2.60 / 2.45 GPU hours); the
retraining command is in the dataset README and `commands.sh`. Kaggle sequence for the user: upload the dataset; push
`casmi26-comply-s1-off` (its `submission.csv` must equal stack1's commit output); push `casmi26-comply-s1` (and
`casmi26-comply-s1m`); read the
`COMPLY_S1` line, `pc_runner done`, `merge stats` and the stage-health lines; submit only with approval. The 12-molecule
commit run exercises the channel but not the gate (all 12 have `lib_max` 1.0: "untouched"), which is a constant change
covered by the tests. The patched channel itself (our two networks through `molecule_logits` and `probe_one`, the
channel's own code) ran locally for 257 molecules during the calibration without an error.

## 7. Not verified

- The forum thread 748882 itself (the rule text is the team's verification of 2026-10-16, quoted from the brief).
- That the re-saved ICEBERG / GLACIER tensors equal the upstream ms-pred files.
- The full tree of `cubres/enveda-casmi-2026` (only the root, `src/` and one guide folder were read); the private trainer
  notebook of prvsiyan (403); whether the host would count prvsiyan's public loop without its data builder as "the original
  training code" (C7 / C8 borderline).
- Whether A / B (C6) saw the Enveda panel molecules (their training split is not documented); this affects k (section 4.1).
- Whether the host treats fitted count tables (C10, C11) as "weights".
- Stage B not bit-identical (thread count or an undocumented sample weight are the likely causes; not tested with 4 threads
  because of the 3-worker limit).
- Ranker-refit time and the feature parity of a patched `V1FE` against the P4 rows (step 2): not run.
- Every cost in days: estimates. No leaderboard number for any step.
- `INDEX.md` not regenerated; nothing committed.
