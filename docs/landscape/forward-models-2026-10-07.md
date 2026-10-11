<!-- index: 2026-10-07 desk survey of open spectrum-prediction (forward) models as a third, independent forward model and for the adducts / negative mode ICEBERG and GLACIER do not cover: training data of each released checkpoint, NIST status, licence, adducts, size, speed, host statements (CFM-ID 4 "is fine", METLIN "to avoid"), verdict table, integration plan for FIORA OS v1.0.0 as a gap-only term for negative-only molecules, downloads to approve, what a local check can and cannot show -->
# Forward models we could add, 2026-10-07

Desk research on 2026-10-07 (web pages, repositories, paper pages, Kaggle listings and metadata through the CLI,
forum topics through `kaggle forums topics show`). **Nothing was downloaded** (no weights, no datasets), nothing was
installed, run, committed, pushed or submitted; no GPU. Pages were read through a fetch tool that returns extracted
text, so a quoted number is "the page says so".

**Marks.** **[P]** read at the source today (repository page, paper page, Kaggle listing / metadata field, forum
post). **[R]** taken from an earlier review of this repository, not re-read today (the review is named). **[S]** search
snippet only: unverified. **[I]** my inference or estimate. "unknown" means I could not establish it; **unknown counts
as unusable** (rule of `docs/competition/rules-and-pitfalls.md`). No leaderboard estimate is given anywhere.

## 1. Summary

1. **There is exactly one open, NIST-free forward model with trained negative-mode weights: FIORA OS v1.0.0**
   (MIT code, trained on MSnLib v7 only, `[M+H]+` and `[M-H]-`, Orbitrap HCD, 54 MB). It is a different architecture
   (bond-neighbourhood GNN, single fragmentation step) **and** different training data from our MassSpecGym ICEBERG /
   GLACIER checkpoints. It is already packaged for offline Kaggle use by the engine's author (65 MB, cp312 wheels,
   scorer). Two public notebooks run it as a third forward model for all molecules; neither moved in the score order.
2. **CFM-ID 4 is the second candidate and the only one with a host statement of its own**: David Healey, 2026-09-27,
   topic 743774: "CFM ID 4 is fine." Its stock models were trained on METLIN, a commercial library the same host
   lists as "to avoid" two days earlier (742193, 742991). It covers `[M+H]+` and `[M-H]-`, is trained on **QTOF** CID
   spectra at 10 / 20 / 40 eV (the instrument class of the test), and is independent of everything we run. It has no
   Python wheel (C++ binary), costs seconds of CPU per molecule, and exists precomputed for natural products (ISDB,
   CC BY 4.0, both modes), which is a lookup, not a model for arbitrary candidates.
3. **Everything else is unusable or adds nothing.** NIST-trained and therefore forbidden: ICEBERG NIST'20 / '23
   weights (the only ICEBERG weights with ten adducts and negative mode), MARASON, FraGNNet (paper), GrAFF-MS,
   3DMolMS (also non-commercial), MassFormer's manuscript models, NEIMS, RASSP. Same family and same data as what we
   have: ICEBERG `msg_simulation`, the other GLACIER packages on Kaggle. No open checkpoint exists for SCARF or
   MARASON.
4. **No open model covers `[M+NH4]+`, `[M+K]+`, `[M+Cl]-`, formate adducts or in-source water loss with trained
   weights.** The only weights that do are ICEBERG's NIST ones. For the negative adducts our own library test says
   an `[M-H]-` model is enough (formate and chloride spectra fragment like `[M-H]-`, `coverage-2026-10-02.md` 2.1) [R].
5. **No forward model has been published as a Kaggle dataset for this competition since 2026-10-04.** The newest is
   still `ahmedberatozer/casmi26-fiora-os-v1` (10-03).
6. **Recommendation**: FIORA OS as a **gap-only** term for molecules whose only spectra are negative (7.2% of the
   Enveda panel [R]), leaving every molecule with ICEBERG / GLACIER scores bit-identical. Section 4 has the plan, the
   downloads to approve and the limits of a local check.

## 2. Candidates

### 2.1 FIORA (BAMeScience)

| Item | Finding | Source |
|---|---|---|
| Released weights | The repository's `fiora/resources/models/` holds only "OS" files: `fiora_OS_v1.0.0_state.pt` (53,814,639 bytes), `fiora_OS_v1.0.0.pt` (53,845,076), `fiora_OS_v1.0.0_params.json`, and the older `fiora_OS_v0.1.0_state.pt` (19,755,411), `fiora_OS_v0.1.0.pt` (39,533,372). README: "By default, an open-source model is selected automatically" | [P] [GitHub API listing](https://api.github.com/repos/BAMeScience/fiora/contents/fiora/resources/models), [README](https://github.com/BAMeScience/fiora) |
| Training data of the released weights | Release v1.0.0 (2025-10-16): "Full open source v1.0.0 release"; trained on the "large MSnLib v7 library (24741 trainings compounds after curation; 3092 for validation; 3092 for testing)". Paper: "FIORA-OS" was "trained on 80% of the MSnLib library". The model's own params file says "trained on the MSnLib v7" | [P] [releases](https://api.github.com/repos/BAMeScience/fiora/releases), [paper, PMC11889238](https://pmc.ncbi.nlm.nih.gov/articles/PMC11889238/); params file [R] `coverage-2026-10-02.md` 2.4 (my fetch of it was refused today) |
| NIST in THAT checkpoint | **No, by the authors' statement.** The paper's *default* model was trained on "three spectral libraries (NIST'17, MS-DIAL, and MSnLib)", 74,401 spectra of 10,692 compounds: that model is **not** among the files in the repository. Release v1.1.0 (2026-04-28) says "Retraining and repository cleanup" without naming data; the model files listed today are still the v0.1.0 and v1.0.0 OS files | [P] same sources. What v0.1.0 OS was trained on: **unknown** (not stated in the release note); do not use v0.1.0 |
| Licence | Code and bundled weights: MIT (LICENSE: "Copyright (c) 2022 [Yannek Nowatzky, Bundesanstalt für Materialforschung und -prüfung (BAM)]"). MSnLib: CC BY 4.0. Both allow commercial use | [P] [LICENSE](https://github.com/BAMeScience/fiora/blob/main/LICENSE), [MSnLib on Zenodo](https://zenodo.org/records/15683784) |
| Adducts, modes | Paper: "both positive and negative ionization modes ([M+H]+ and [M-H]- precursors)"; "[M]+ and [M + Na]+" are "currently not supported by the default FIORA model". Training example in the README: `[M+H]+,[M-H]-,[M]+,[M]-`. **Not covered: `[M+NH4]+`, `[M+Na]+`, `[M+K]+`, `[M+Cl]-`, formate, water loss** | [P] paper, README |
| Instrument, collision energy | MSnLib is Orbitrap ID-X, flow injection; the checkpoint's instrument list is `["HCD"]` [R]. Collision energy is a continuous input, "0 to 100 eV" | [P] Zenodo, paper |
| Size | 53.8 MB state file; Kaggle package 64.6 MB | [P] |
| Dependencies | `pyproject.toml`: Python ≥ 3.10.8; numpy, pandas, torch, `lightning-fabric` 2.6 to 2.7, `torch_geometric` 2.6 to 2.7, torchmetrics, dill, rdkit (unpinned), regex, treelib, `spectrum_utils`, scikit-learn, seaborn, ipython. `requirements.txt` pins rdkit 2024.9.6, torch 2.6.0, numba 0.61.0. **Whether it gives the same output under RDKit 2026.03.3: unknown.** No compiled extension of its own; `spectrum_utils` needs numba | [P] [pyproject.toml](https://raw.githubusercontent.com/BAMeScience/fiora/main/pyproject.toml), [requirements.txt](https://raw.githubusercontent.com/BAMeScience/fiora/main/requirements.txt) |
| Speed | Paper: 0.03 s per prediction on an NVIDIA A100, 0.92 s on a CPU; "predicts around 10,000 spectra within just five minutes"; in the same table CFM-ID 3.43 s (CPU), ICEBERG 0.64 s (CPU). **No T4 number exists** | [P] paper |
| Re-ranking record | Paper: a candidate-ranking experiment, described qualitatively in what I could read (the score without the precursor peak "is better suited to distinguishing the correct compounds from false candidates"); no MassSpecGym retrieval number found. Test-split median cosine, negative mode: FIORA 0.79, CFM-ID 0.57 | [P] paper |
| Independence | High on both axes: architecture (edge-property GNN over single bond breaks, against ICEBERG's fragment-DAG generator + intensity model and GLACIER's detection formulation) and data (MSnLib Orbitrap, both polarities, against MassSpecGym positive). Caveat: MSnLib is one of the eleven sources of `train.parquet` (207,954 negative MSnLib spectra of 24,852 structures [R]), so FIORA has seen structures our rankers and FPNets have also seen | [I] from the sources above |
| On Kaggle | `ahmedberatozer/casmi26-fiora-os-v1` (2026-10-03, 64,597,755 bytes, licence field "Other", 651 downloads). Description: "FIORA OS v1.0.0 (MSnLib v7 only), exact official source/state/params pin 21dc75bf10e76f807414dd2e7370d326fb8149a5. MIT repository notice retained; source MSnLib data CC BY 4.0; no raw spectra. Public offline Linux cp312 Torch2.10+cu128 runtime wheels". Files: `models/fiora_OS_v1.0.0_state.pt` (53,814,639 bytes: the same size as upstream), `models/fiora_OS_v1.0.0_params.json`, `scorer/fiora_score.py` (12 kB), `scorer/fiora_runner.py`, `source/`, manifests. Used by three notebooks: `lehau007/casmi26-sota-v32-fiora-tri-forward-0420`, `nursrijan/forensics-lab-dual-shield-two-ranker-sota`, `arman1o1/enveda-casmi26-molecule-id-v10` | [P] `kaggle datasets files / metadata`, `kaggle kernels list --dataset` |
| Public result | None positive. lehau007 v32 (`FIORA_LAM` 0.5, budget 2,000 s, all molecules) is unscored; nursrijan's copy sits 22nd in the score order, below the same author's lineage without FIORA | [R] `update-2026-10-05b.md`, `update-2026-10-06.md`; order re-read today [P] |
| Host statement | **None on FIORA by name** (forum search "fiora": no topic) [P]. It falls under the general rule: an open-weight model is fine "as long as the open weight model is distributed in compliance with the license for the data it was trained on" (742193), and MSnLib is CC BY 4.0 and inside `train.parquet`, which "can be treated effectively as open-source" (745841 [R]) | [P] [742193](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742193) |

### 2.2 The ms-pred family (coleygroup): ICEBERG, GLACIER, MARASON, SCARF

| Item | Finding | Source |
|---|---|---|
| Public checkpoints | README: ICEBERG 2.1 `msg_all` ("a streamlined ICEBERG workflow for using all MassSpecGym entries in the main retrieval benchmark") and `msg_simulation` ("the official MassSpecGym simulation-challenge subset"), and one GLACIER checkpoint trained on MassSpecGym, all on Dropbox. Changelog: 2026-07-07 ICEBERG 2.1 and GLACIER; 2026-04-04 the ICEBERG atlas. **No MARASON and no SCARF checkpoint is listed.** Dropbox file sizes: not stated | [P] [README](https://github.com/coleygroup/ms-pred) |
| NIST | "If you have an NSIT'20/23/26 license, you can email the maintainer with a proof of license and we will send you the weights." The MassSpecGym checkpoints: no NIST by the README's own separation. MARASON's paper trains on NIST'20 [R] | [P] README |
| Adducts | The ICEBERG 2 paper lists ten: "[M+H]+, [M+Na]+, [M+K]+, [M−H2O+H]+, [M−2H2O+H]+, [M+NH3+H]+, [M−H]−, [M+Cl]−, [M−H2O−H]−, and [M−CO2−H]−", trained on "530,640 spectra across 25,541 unique molecules" of NIST'20, HCD; "The pretrained weights will be made available to those with a license". **These are the NIST weights: unusable.** The MassSpecGym checkpoints have trained input columns for `[M+H]+` and `[M+Na]+` only (ICEBERG) and `[M+H]+` only (GLACIER) [R, measured in `coverage-2026-10-02.md` section 1] | [P] [bioRxiv 2025.05.28.656653](https://www.biorxiv.org/content/10.1101/2025.05.28.656653v1.full) |
| Licence | Code MIT [R]; the GLACIER arXiv page carries CC BY-NC-ND 4.0 for the paper text. Host: "GLACIER/ICEBERG ... are all fine" (745841 [R]) | [P] [arXiv 2606.29161](https://arxiv.org/abs/2606.29161) |
| What is new for us | **Nothing.** `msg_simulation` is the same architecture trained on a subset of the same data as the `msg_all` we run. On Kaggle: `seyitkaangunes/casmi26-glacier` (165 MB, `ckpt/glacier_best.ckpt` 181,602,423 bytes, "MassSpecGym contrastive checkpoint", 2026-09-29) and `shishiradhikari11/casmi-fm-runner` (131 MB, MIT, an ms-pred runner, 2026-10-01): same family, same data; whether the first is a different checkpoint from the 56 MB `ahmedberatozer/casmi26-glacier` is not stated on its card | [P] Kaggle listings |

### 2.3 CFM-ID 4 (Wishart lab) and its precomputed spectra (ISDB)

| Item | Finding | Source |
|---|---|---|
| Training data | Forum 743774 quotes the CFM-ID 4.0 paper: "We used experimental ESI-MS/MS data obtained from the Scripps Research Institute's Metlin MS database to train CFM-ID 4.0" (Wang et al., Anal. Chem. 2021; the ACS page refused my fetch, so the quote is the poster's). The FIORA paper says the same: "the latest model, pre-trained on the METLIN library". The web-server paper: "trained exclusively on QTOF CID data", collision energies "10 eV, 20 eV and 40 eV" | [P] [743774](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743774), [PMC11889238](https://pmc.ncbi.nlm.nih.gov/articles/PMC11889238/), [PMC9252813](https://pmc.ncbi.nlm.nih.gov/articles/PMC9252813/) |
| NIST / non-free data | NIST: no. **METLIN: yes, and METLIN is a commercial library.** | as above |
| Host statements (they pull in two directions) | 742991, David Healey, 09-25: what the rule is meant to avoid is "datasets like NIST ... (or proprietary commercial datasets like METLIN, or those distributed by the mass spectrometer companies) ... if they do exist they would not be eligible". 743774, David Healey, **09-27**, asked exactly whether the METLIN-trained stock models are eligible: "**CFM ID 4 is fine. It's published by a major lab in a major journal, so we'll proceed with the assumption that the authors had permission to distribute the weights.**" 741912, 09-19: "Spectra generated from COCONUT structures with CFM-ID is fine"; "LGPL-licensed runtime dependency is fine also" | [P] forum, read in full |
| Licence | LGPL-2.1 [R, `coverage-2026-10-02.md`; the Bitbucket page returned no content today]. LGPL allows commercial use | [R] |
| Adducts | "Both [M + H]+ and [M-H]- adduct types are fully supported for spectrum prediction tasks, while other adduct types are only partially supported." | [P] PMC9252813 |
| Size, install | Docker image `wishartlab/cfmid`, tag 4.2.6.0, 193.9 MB, last updated over four years ago; contains `cfm-predict` and the 4.0 models for both ESI modes. **No Python wheel.** A Kaggle run would need the Linux binary and its shared libraries shipped as a dataset; nobody has published one (Kaggle search "cfm", "cfmid", "cfm-id": only the ISDB spectra) | [P] [Docker Hub](https://hub.docker.com/r/wishartlab/cfmid), Kaggle listing |
| Speed | 3.43 s per prediction on a CPU (FIORA paper's table); "Typically, a molecule with a MW < 1000 Da will take less than one minute to predict" (web server, three-minute time-out). CPU only | [P] |
| Re-ranking record | Long (CASMI contests); no MassSpecGym retrieval number found. FIORA paper: negative-mode median cosine 0.57 against FIORA's 0.79 on FIORA's MSnLib test split | [P] |
| Independence | Highest of all candidates: a probabilistic fragmentation model, other training data, and **QTOF** spectra at fixed 10 / 20 / 40 eV (timsTOF is a QTOF-type instrument; our two models and FIORA are Orbitrap- or mixed-trained) | [I] |
| Precomputed: ISDB | Zenodo 14887271, v5.0.0 (2025-02-19), CC BY 4.0: about one million natural products of Wikidata / LOTUS fragmented with "cfm-predict 4", both modes; `isdb_lotus_neg_energySum.mgf` 724.3 MB, `isdb_lotus_pos_energySum.mgf` 939.4 MB, `isdb_wikidata_neg_energySum.mgf` 3.4 GB, `isdb_wikidata_pos_energySum.mgf` 4.4 GB. On Kaggle: `n10705013/casmi26-isdb-predicted-spectra` (3.38 GB, four MGF files, 09-21), `n10705013/casmi26-isdb-predicted-index` (1.37 GB; "2,260,984 CFM-ID 4 predicted MS/MS spectra"), `aminvafaei/casmi26-e045-isdb` (251 MB listed; `peaks.npz` 474,549,642 bytes, `index.parquet` 3,369,118 bytes, CC BY 4.0, no card, 10-06). No notebook attaches any of them (`kernels list --dataset`: nothing) | [P] [Zenodo](https://zenodo.org/records/14887271), Kaggle |

A lookup scores only candidates that are in LOTUS / Wikidata. A same-formula group in which some members have a
predicted spectrum and others do not cannot be z-scored fairly; how many of our candidates are covered is **unknown**
and is the first thing to measure (section 4.6).

### 2.4 The rest

| Model | Training data of the released weights | NIST | Licence | Adducts / modes | Verdict |
|---|---|---|---|---|---|
| **FraGNNet** ([arXiv 2404.02360](https://arxiv.org/html/2404.02360v2), [repo](https://github.com/FraGNNet/fragnnet)) | Paper: "NIST20 MS/MS dataset". The repository README names no checkpoint and no download [P]. The MassSpecGym paper reports a FraGNNet baseline on its simulation challenge (cosine 0.52, hit rate@1 46.64% [S]); weights for it were not found | paper model: **yes**; a MassSpecGym checkpoint: **not found** | BSD-2-Clause [P] | paper: "we focus our analysis on [M+H]+ spectra" | unusable (no open checkpoint found). To ask: Adamo Young (address in the README) whether MassSpecGym-trained weights can be released |
| **MassSpecGym simulation baselines** ([repo](https://github.com/pluskal-lab/MassSpecGym)) | the repo points to ICEBERG's MassSpecGym weights; no other simulation weights documented [P] | – | MIT | MassSpecGym holds `[M+H]+` and `[M+Na]+` only, positive mode [R] | nothing beyond what we run |
| **GrAFF-MS** ([repo](https://github.com/murphy17/graff-ms)) | weights in the repo (`epoch=96-step=27257.ckpt`), trained on NIST-20 [P] | **yes** | none stated [P] | `[M+H]+`, `[M-H]-` | **unusable** |
| **MassFormer** ([repo](https://github.com/Roestlab/massformer)) | `demo.pkl`: MoNA, InChIKey split. Manuscript models: NIST 2020; "NIST does not support redistribution of parameters for models trained on this library" [P] | demo: no; manuscript: **yes**, not distributed | BSD-2-Clause | training configs list `[M+H]+`, `[M+H-H2O]+`, `[M+H-2H2O]+`, `[M+2H]2+`, `[M+H-NH3]+`, `[M+Na]+`; FT instruments; **no negative mode** [P]; which of them the demo checkpoint was trained with: unknown | usable in principle, low value: positive only, a demo model, binned spectra, PyTorch 1.12 / DGL stack (will not install on the current image without work [I]) |
| **3DMolMS** ([repo](https://github.com/JosieHong/3DMolMS)) | NIST20, NIST23, MoNA and vendor libraries [P; Agilent PCDL, GNPS, Waters per R] | **yes** | CC BY-NC-SA 4.0, "academic use" [P] | `[M+H]+`, `[M-H]-`, `[M+H-H2O]+`, `[M+Na]+`, `[M+2H]2+`; QTOF and Orbitrap models | **unusable twice** (NIST, non-commercial) |
| **NEIMS**, **RASSP** | NIST 2017 EI library [S] | **yes** | – | electron ionisation, GC-MS | **unusable and irrelevant** (wrong ionisation) |
| **QCxMS / QCxMS2** | no weights: semi-empirical quantum chemistry + molecular dynamics | none | open source | QCxMS has an ESI / CID mode [S]; QCxMS2's documented default is EI | **too slow**: the QCxMS2 example, 2-pentanone (6 heavy atoms), EI, 16 cores, shows stages of 35 s, 314 s and 97 to 158 s each over six iterations [P, [xtb docs](https://xtb-docs.readthedocs.io/en/latest/qcxms2_doc/qcxms2_ex.html)]; QCxMS runs "from minutes to a few hours" per molecule [S]. We need tens of thousands of candidates of 25 to 80 heavy atoms on four CPU cores |
| **GIF** ([arXiv 2511.09571](https://arxiv.org/abs/2511.09571)) | prompts GPT-4o / GPT-4o-mini to simulate spectra (cosine 0.36 / 0.35 on MassSpecGym) [P] | – | – | – | **unusable**: needs a hosted LLM; the run is offline |
| **ICEBERG PubChem atlas** | predictions of the NIST'23-trained ICEBERG [R, forum 745134, unanswered] | **yes** | – | – | **unusable** |
| **MIST, DreaMS, MS-GPT, FRIGID** (packaged on Kaggle, the newest `tamerlanomralinov/casmi26-experiment-production-assets`, 10-07, "public MS-GPT MassSpecGym assets") | – | – | – | – | not forward models (spectrum → fingerprint / embedding / structure); out of scope here |

### 2.5 Kaggle datasets since 2026-10-04 (forward models only)

Searches today [P]: `fiora`, `iceberg`, `glacier`, `ms-pred`, `marason`, `scarf`, `fragnnet`, `massformer`, `3dmolms`,
`cfm-id`, `cfmid`, `cfm`, `isdb`, `graff`, `neims`, `msnlib`, `massspecgym`, `casmi26`, "forward model", "spectrum
prediction", "in silico spectra", and the dataset lists of ahmedberatozer, prvsiyan, megayak, dmitriigluzdov,
goooman, wangpenghua, bobthebot369, seyitkaangunes, denpugovkin, evgendvorkin, lehau007, takumuhata,
shishiradhikari11, tamerlanomralinov, skomuro.

- **No new forward model.** ahmedberatozer's only new dataset is `casmi26-full-pc-backbones-20261006` (10-07; two
  962 MB-in-total DreaMS-backbone retrieval checkpoints, "no E2E or leaderboard improvement certified"): not a
  forward model. wangpenghua, bobthebot369, denpugovkin, skomuro: no datasets. goooman, megayak, prvsiyan,
  dmitriigluzdov, seyitkaangunes, evgendvorkin: nothing forward-model-related since 10-04.
- New but not forward models: `aminvafaei/casmi26-e045-isdb` (10-06, the ISDB lookup of 2.3),
  `tamerlanomralinov/casmi26-production-runtime-wheels` (10-07; RDKit 2026.3.3 wheels for cp311 / cp312 / cp313 and
  RDKit 2025.3.6 cp313 "generator/GLACIER"), `shishiradhikari11/casmi-rdkit2026-cp312` (10-07).

## 3. Verdict table

Ranked by expected value for us: (a) scores adducts or the mode we cannot score today, (b) independence from
ICEBERG / GLACIER, (c) cost inside the run (about 4 h 20 min of 9 h used; forward budgets 5,400 s and 4,000 s).

| # | Model | Training data of the released weights | NIST | Licence, commercial use | Adducts with trained weights | Size | (a) | (b) | (c) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **FIORA OS v1.0.0** | MSnLib v7 (Orbitrap HCD, CC BY 4.0) | **no** (authors' statement) | MIT; yes | `[M+H]+`, `[M-H]-` (`[M]+`, `[M]-`) | 54 MB (Kaggle package 65 MB) | `[M-H]-`: yes; formate / chloride through the `[M-H]-` proxy [R]; nothing else | high | GPU, 0.03 s per prediction on an A100; T4 unknown | **usable** |
| 2 | **CFM-ID 4** (binary) | METLIN (commercial), QTOF CID 10 / 20 / 40 eV | no | LGPL-2.1 [R]; yes | `[M+H]+`, `[M-H]-`; others "partially" | Docker image 194 MB | `[M-H]-`: yes | highest | CPU, 3.4 s to a minute per molecule; no wheel | **usable per the host's explicit answer (743774), with a compliance caveat** (METLIN; the answer rests on "the assumption that the authors had permission"). Engineering unclear |
| 3 | **ISDB** (CFM-ID 4 spectra, precomputed) | as 2 | no | CC BY 4.0; yes | both modes, natural products of LOTUS / Wikidata only | 251 MB to 3.4 GB on Kaggle | partial (covered candidates only) | highest | a lookup: seconds | same status as 2; value depends on a coverage number nobody has |
| 4 | MassFormer `demo.pkl` | MoNA | no | BSD-2; yes | positive only | not stated | no | medium | old PyTorch / DGL stack | usable, not worth it |
| 5 | ICEBERG `msg_simulation`, other GLACIER packages | MassSpecGym (subset / same) | no | MIT; host-approved | `[M+H]+` (`[M+Na]+`) | 56 to 165 MB | no | none | as today | usable, adds nothing |
| – | ICEBERG NIST'20 / '23 (ten adducts, negative mode), MARASON, ICEBERG atlas | NIST | **yes** | by e-mail with a NIST licence | – | – | – | – | – | **unusable** |
| – | GrAFF-MS, 3DMolMS, MassFormer manuscript models, NEIMS, RASSP | NIST (3DMolMS also vendor libraries, CC BY-NC-SA) | **yes** | – | – | – | – | – | – | **unusable** |
| – | FraGNNet | paper: NIST'20; no public checkpoint found | paper: yes; MassSpecGym weights: not found | BSD-2 | `[M+H]+` | – | no | high | – | **unclear → unusable**: ask Adamo Young (University of Toronto, address in the repository README) for MassSpecGym-trained weights |
| – | SCARF | no public checkpoint | – | MIT | – | – | – | – | – | unusable (nothing to download) |
| – | QCxMS / QCxMS2 | none (physics) | no | open | CID mode exists [S] | – | yes in principle | highest | minutes to hours per molecule | **unusable: time** |
| – | GIF | hosted LLM | – | – | – | – | – | – | – | unusable offline |

Questions that would change a verdict, and whom to ask:
- **Host (David Healey, topic 743774 or a new one)**: "Your answers in 742991 (METLIN-trained weights not eligible)
  and 743774 (CFM-ID 4 is fine) differ. Does 743774 also hold at the final compliance check, for the binary and for
  ISDB spectra?" Needed before CFM-ID or ISDB goes into a final selection.
- **Host**: "Is FIORA OS v1.0.0 (MIT, trained only on MSnLib v7, CC BY 4.0) fine?" The general rule covers it; a
  named answer costs one post and removes the question.
- **FIORA authors (BAM, GitHub issue)**: what `fiora_OS_v0.1.0` was trained on; whether v1.1.0 changed the weights
  file; whether anyone has run OS v1.0.0 on QTOF / timsTOF spectra.

## 4. Integration plan for FIORA OS v1.0.0 (no code written)

### 4.1 What to add first: negative-only molecules, as a gap term

Our pipeline already has the right slot. `FRAG_MODE='gap'` gives the MetFrag-parsimony term to molecules "ICE cannot
score" and leaves every molecule with ICEBERG scores untouched (`nb/embed/fusion_core.py::frag_rerank`;
`tests/test_frag_gap.py` checks exactly that). FIORA would enter the same way, for a narrower segment:

- **Segment: molecules without any `[M+H]+` / `[M+Na]+` spectrum that have at least one negative spectrum.** On the
  Enveda panel that is 7.2% of the molecules (18 of 250), half of the 13.6% uncovered [R, `anatomy-2026-10-06.md`
  2.2]. The hidden test's share is not documented.
- **Spectra used**: `[M-H]-` natively. `[M+CH2O2-H]-` and `[M+Cl]-` cut to their fragments and scored against the
  `[M-H]-` prediction (the `frag` transformation of `coverage-2026-10-02.md` 2.1: isomer pick 0.91 to 0.96 on timsTOF
  with real reference spectra [R]; with predicted spectra it is unmeasured).
- **Not in the first step**: the `[M+NH4]+` / water-loss half of the gap (FIORA has no such adduct, and giving these
  molecules `[M+H]+`-surrogate evidence is what failed in the anatomy review: `c1` −0.042 to −0.135 in the segment);
  negative spectra of molecules that already have ICEBERG / GLACIER scores (52% of the panel have both polarities:
  the larger prize and the larger risk, because it changes covered molecules); FIORA on `[M+H]+` for everybody (what
  the two public notebooks do, without a visible gain).
- **Bit-identity**: with the flag on, a molecule that has an ICEBERG or GLACIER score gets no FIORA term at any
  stage. A test of the same form as `test_frag_gap.py` should assert it.
- **Relation to the MetFrag gap term** (v33b runs `FRAG_LAM` 0.5 in gap mode): for the negative-only molecules FIORA
  either replaces it or joins it in one z-sum. That is the one design choice, and it has to be made on the local
  in-group test (4.5), not on a replay.

### 4.2 Where it enters

1. `nb/00_config.py`: `USE_FIORA=False`, `FIORA_LAM=0.0`, `FIORA_MODE='neg_gap'`, `FIORA_BUDGET=900`,
   `FIORA_TOPK=60` (defaults off: the default config stays reproducible).
2. `nb/07_rescore.py`, after the GLACIER block: build items for the segment only (candidates: the same list ICEBERG
   would get, `ICE_UNION` rows and joined PubChem rows included; spectra: the negative ones, one prediction per
   candidate and collision-energy bucket), run the scorer in its **own process with its own `site` directory** as
   ICEBERG does (`site=os.path.join(WORK, 'ice_site')` is the pattern), collect `{molecule: {smiles: score}}`,
   `dump('fiora', FIORA_SCORES)`, and add `FAILS.append(...)` lines for an install failure and a budget stop so
   `STRICT` sees them. Score = mean entropy similarity over the molecule's spectra, the precursor peak removed
   (the FIORA paper's finding, and what our ICEBERG runner does).
3. `nb/embed/fusion_core.py`: `fiora_scores=None` next to `frag_scores`; a `fiora_rerank` that is `frag_rerank` with
   the extra condition "no ICE / GL coverage"; called at the same two places (stage A, and the post-fusion re-rank at
   line 355 ff.); same-formula z-scored term with weight `FIORA_LAM`; respects `FRAG_LIB_OFF`-style library gating.
4. `tools/build_notebook.py`: attach `ahmedberatozer/casmi26-fiora-os-v1` when `USE_FIORA`.
5. `README.md` provenance (MSnLib v7 only, MIT, CC BY 4.0, pin `21dc75bf`), `THIRD_PARTY_NOTICES.md` (FIORA source;
   the Kaggle package's scorer if we reuse it), `EXPERIMENTS.md`, a preset `configs/v36a-v33b-fiora-neg.json`,
   `tests/test_fiora_gap.py`, `python tools/gen_index.py`.

### 4.3 Downloads to approve (none made)

| # | What | Size | URL | Needed for |
|---|---|---|---|---|
| 1 | `ahmedberatozer/casmi26-fiora-os-v1` (weights, pinned source, cp312 Linux wheels, scorer, manifests) | **64.6 MB** | https://www.kaggle.com/datasets/ahmedberatozer/casmi26-fiora-os-v1 | reading the scorer and the wheel manifest; the Kaggle run attaches it without a download |
| 2 | Alternative to 1 for the local test, straight from the authors: `fiora_OS_v1.0.0_state.pt` + `fiora_OS_v1.0.0_params.json` | **53.8 MB** + 1.3 kB | https://raw.githubusercontent.com/BAMeScience/fiora/main/fiora/resources/models/fiora_OS_v1.0.0_state.pt (and `..._params.json`) | local in-group test |
| 3 | FIORA source at commit `21dc75bf10e76f807414dd2e7370d326fb8149a5` | a few MB [I] | https://github.com/BAMeScience/fiora | local test (Windows; the Kaggle wheels are Linux cp312) |
| 4 | Local Python packages FIORA needs and the repo's `.venv` may lack: `torch_geometric`, `lightning-fabric`, `torchmetrics`, `treelib`, `spectrum_utils` (+ numba), `dill` — into a **separate target directory**, not the `.venv` (as the coverage study did for ICEBERG) | not measured | PyPI | local test |

An install into anything, and each download, is a separate approval. The small text files of package 1 (README,
licence notice, manifests, scorer) were **not** read today: the single-file download was refused by the session's
permission layer. What is quoted from that package is its metadata description and file listing only.

### 4.4 Offline install on Kaggle and the RDKit pin

- The package ships "Public offline Linux cp312 Torch2.10+cu128 runtime wheels" [P, description]. Our notebook is
  pinned to the Python 3.12 image because of the ICEBERG wheels, so the Python version matches. **Which RDKit wheel
  the package installs is in `WHEEL_MANIFEST.json`, which I could not read**: check it first.
- FIORA's `pyproject.toml` does not pin RDKit; `requirements.txt` pins 2024.9.6 for reproducibility. The metric's
  RDKit 2026.03.3 must stay in the main process: run FIORA in a subprocess with `pip install --no-index --target
  <site>` into its own directory, as the ICEBERG runner does. FIORA only parses SMILES and enumerates bond breaks, so
  an RDKit difference can change predictions slightly but never a metric key; whether its outputs are identical
  across RDKit versions is unknown and does not need to be.
- Kaggle's image moved to Python 3.13 on 10-01 [R]; the notebook's pinned environment must be kept, and a failed
  install must go to `FAILS` (ICEBERG's silent 3.13 failure is the precedent).
- First Kaggle-side step, before any local work is trusted: a private commit run (no submission) that attaches the
  dataset, installs it, and scores the smoke molecules, printing seconds per prediction on the T4. Nobody has
  published that number.

### 4.5 GPU time

Published rate: 0.03 s per prediction on an A100 ("around 10,000 spectra within just five minutes"). T4 rate:
**unknown**; whether the scorer batches candidates: not read.

| Scope | Predictions | At the A100 rate | If a T4 is 3× slower [I] |
|---|---|---|---|
| **Proposed**: negative-only molecules, 7.2% of 400 ≈ 29 molecules × 60 to 100 candidates × 1 to 3 collision-energy buckets | 1,700 to 8,700 | 1 to 4.5 min | 3 to 13 min |
| All molecules with a negative spectrum (59% [R]) ≈ 236 × 60 to 100 × 1 to 3 | 14,000 to 71,000 | 7 to 35 min | 21 to 106 min |
| Everything: 400 molecules × 60 to 100 candidates × both modes × 1 to 3 buckets | 48,000 to 240,000 | 24 to 120 min | 72 to 360 min |

The proposed scope fits a 900 s budget with room; the full scope does not fit next to ICEBERG and GLACIER. The 3×
is a guess, not a measurement; the Kaggle smoke run of 4.4 replaces it.

### 4.6 What a local check can and cannot tell us

**Can.**
1. *Does FIORA rank isomers on timsTOF negative spectra at all?* The in-group test of `coverage-2026-10-02.md` 2.2:
   for each molecule, the answer's same-formula group, FIORA alone, top-1 and MRR against chance. Three panels, in
   order of honesty:
   - Enveda-180 timsTOF `[M-H]-` structures of `train.parquet` whose key is **not** in MSnLib (185,023 `[M-H]-`
     timsTOF spectra of 65,996 structures exist [R]; how many are outside MSnLib must be counted): the right
     instrument, structures FIORA has not seen, thousands of molecules, pool isomers as decoys. Synthetic drug-like
     chemistry, not natural products.
   - the 18 negative-only panel molecules and the 148 with a negative spectrum. 123 of the 148 are in MSnLib, FIORA's
     training set [R]: the clean subset is about 25 molecules.
   - the uncovered molecules of the 380 mixed regime (11% timsTOF).
   Acceptance, stated before the run: in-group MRR on unseen timsTOF negative-only molecules clearly above chance
   (0.16 on the panel) with a standard error that excludes it, and above the MetFrag term on the same molecules. If
   FIORA is at chance there (an Orbitrap HCD model on QTOF-type spectra can be), the plan ends without a submission.
2. *Does the flag leave covered molecules bit-identical, and does the stage stay inside its budget?* A unit test and
   the Kaggle smoke run.
3. *Which form of the term* (replace or join the MetFrag gap term; precursor removed; the formate / chloride cut) is
   best in the in-group test.

**Cannot.**
- **The size or the sign of the leaderboard effect.** Local replays have understated the forward stage five times
  (weights 0.6 and 0.5, v34a, two fine-tuned checkpoints), and the one local result on giving gap molecules forward
  evidence (the surrogate) was negative locally and sat inside a losing submission (v11). A full replay with FIORA
  scores would be run for the veto only (does any type lose beyond one standard error, does exposure rise), never as
  an estimate.
- **Anything on 18 molecules.** One molecule is 0.06 inside that segment; the replay's standard errors there were
  0.02 to 0.06 in the anatomy review.
- **The segment's share of the hidden test.** If the hidden test has few negative-only molecules, the change touches
  few lists: 7.2% of the 130 public-leaderboard molecules is about nine molecules, so a single submission cannot
  resolve the effect either way (one answer moving between place 1 and 2 is 0.0038 [R]). The case for it has to rest
  on the in-group test on unseen timsTOF structures; a submission would only confirm that nothing broke.
- **Leak direction.** Our rankers and FPNets have seen the panel; FIORA has seen its MSnLib members. A panel result
  mixes both.

### 4.7 Second candidate: CFM-ID 4 through ISDB, as a measurement first

Not for a submission before the host question of section 3 is answered. The cheap first step is a coverage count:
which share of our candidates (pool rows and joined PubChem rows of the stage dumps) has an ISDB spectrum, per
molecule and per same-formula group. It needs `index.parquet` of `aminvafaei/casmi26-e045-isdb` (**3.4 MB**; the
dataset has no card, so its key column is unknown) or `n10705013/casmi26-isdb-predicted-index` (**1.37 GB**,
`isdb_index.pkl` 2.0 GB: a pickle from a third party, which must not be unpickled without review). If most
same-formula groups of natural-product candidates are fully covered, ISDB gives a QTOF-trained, both-polarity forward
term for seconds of run time, entering exactly as in 4.2 with a "group fully covered" condition. If coverage is
patchy, the remaining option is the `cfm-predict` binary: a build problem (static Linux binary as a dataset) and,
at 3.4 s to a minute per molecule on four CPU cores, affordable only for the gap segment (29 molecules × 60 to 100
candidates ≈ 1,700 to 2,900 structures: 25 to 40 minutes at 3.4 s on four cores [I], more for large natural products).

### 4.8 Training our own, for completeness

Allowed (a model trained on `train.parquet`), and `fiora-train` takes `--precursor-modes` [P]. `train.parquet` has
639,968 negative spectra of 113,160 structures, 266k of them timsTOF [R]. Against it: both fine-tuned forward models
tried so far lost on the leaderboard (0.401, 0.414 [R]). Not proposed here.

## 5. Could not be established

- The T4 speed of FIORA, whether its packaged scorer batches, and which RDKit wheel the Kaggle package installs (its
  text files were not readable in this session).
- FIORA on timsTOF / QTOF spectra: no published number. FIORA's quantitative candidate-ranking result: not in the
  text I could read. No MassSpecGym retrieval number for FIORA or CFM-ID 4.
- What `fiora_OS_v0.1.0` was trained on; what "Retraining" in release v1.1.0 refers to.
- `fiora_OS_v1.0.0_params.json` at the source: my fetch was refused; the statement about it is from
  `coverage-2026-10-02.md`.
- The CFM-ID 4.0 paper page (ACS, HTTP 403) and the Bitbucket repository page (no content): the METLIN quote is the
  forum poster's and the FIORA paper's; LGPL-2.1 is from our earlier review.
- Whether the host's "CFM ID 4 is fine" survives the final compliance check, given the same host's METLIN sentence.
- File sizes of the ms-pred Dropbox checkpoints; whether `seyitkaangunes/casmi26-glacier` is a different checkpoint
  from the one we run.
- Whether FraGNNet weights trained on MassSpecGym exist anywhere.
- QCxMS CID timings for molecules of our size (only an EI example for a 6-heavy-atom molecule was readable).
- ISDB coverage of our candidates; the schema of `aminvafaei/casmi26-e045-isdb`.
- The share of negative-only molecules in the hidden test.
- A forward model published between my search terms: the listing is keyword-driven, so a dataset whose title and
  description avoid all 21 terms would be missed.

`INDEX.md` was not regenerated (this task edits no existing file): run `python tools/gen_index.py` when this file is
committed.
