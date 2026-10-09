<!-- index: 2026-10-10 licence and rule audit of everything the fusion-v4n notebook uses: the rule clauses and host statements quoted from the Rules tab and the forum (signed out), an inventory of 37 items (datasets, checkpoints, embedded code, wheels) with licence field, underlying origin, training data, verdict and evidence, the fixes ordered by risk, the gaps of THIRD_PARTY_NOTICES.md, and what the audit means for the final selection -->
# Licence and rule audit of the fusion-v4n notebook, 2026-10-10

Read-only audit. Nothing in the solution was changed, nothing was downloaded from Kaggle except dataset metadata
(`kaggle datasets metadata`, `kaggle datasets files`) into `scratch/licaudit/` (git-ignored). Small text files
(`LICENSE-NOTICE.txt`, `README.md`, `MANIFEST.json`) were read from copies that were already on this machine
(`D:\casmi-data\coverage_study\`, `D:\casmi-data\kaggle_ds\`). Rules and forum threads were read in the browser, signed
out, on 2026-10-10. Marks: **[R]** a rule as written, **[H]** a host comment in the forum, **[O]** our own caution.

## 0. Result in one paragraph

No item is NIST-derived as far as any publisher states, and the hosts have approved by name or by class almost
everything the notebook uses (MassSpecGym ICEBERG / GLACIER, DreaMS, ChEBI / LIPID MAPS, anything trained on
`train.parquet`). But the user's standard is stricter than "the host tolerates it": **seven datasets of the v33b chain
carry a non-commercial licence label or note** (the whole public v1 engine of ahmedberatozer, the fold-safe FPNet, and,
new since 10-07, both of prvsiyan's engine-2 datasets, relabelled from CC0 to CC BY-NC-SA 4.0). The ChEBI / LIPID MAPS
swap of 10-09 removed one such label out of eight. No public, permissively labelled replacement exists for the v1
engine. The code inside those datasets has no code licence at all, and the hosts have not answered that question
(threads 745072, 742265). Counts: **18 clean, 11 problem, 8 unclear** (37 items, section 3).

## 1. The rules as written, and what the hosts said

### 1.1 Rules tab [R] (https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules)

| Clause | Text (quoted) |
|---|---|
| 1.6 Winner licence type | "Open Source - MIT" |
| 1.7 Data access and use | "Competition Use and Non-Commercial & Academic Research (NC) - CC BY-NC 4.0" |
| 2.2 Submission limits | "You may select up to two (2) Final Submissions for judging." |
| 2.4.a Competition data | "You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition and on Kaggle.com forums, and for academic research and education." |
| 2.5.a Winner licence | "You hereby license and will license your winning Submission and the source code used to generate the Submission under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code." |
| 2.5.a, second paragraph | "For generally commercially available software that you used to generate your Submission that is not owned by you, but that can be procured by the Competition Sponsor without undue expense, you do not need to grant the license in the preceding Section for that software." |
| 2.5.a, third paragraph | "In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s)." |
| 2.5.b | "The description should also include a link to a code repository with complete and detailed instructions so that the results obtained can be reproduced." |
| 2.6.a External data | "You may use data other than the Competition Data (“External Data”) to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria as outlined in Section 2.6.b below." |
| 2.6.b | "The use of external data and models is acceptable unless specifically prohibited by the Host. Because of the potential costs or restrictions (e.g., “geo restrictions”) associated with obtaining rights to use external data or certain software and associated tools, their use must be “reasonably accessible to all” and of “minimal cost”." |
| 2.8.a Winner's obligations | "Deliver to the Competition Sponsor the final model's software code as used to generate the winning Submission and associated documentation. [...] For avoidance of doubt, delivered software code should include training code, inference code, and a description of the required computational environment." and "Grant to the Competition Sponsor the license to the winning Submission stated in the Competition Specific Rules above, and represent that you have the unrestricted right to grant that license" |
| 3.5.d Private sharing | "No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums." |
| 3.6.b Public code sharing | "If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code." |
| 3.6.c Use of open source | "Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code." |
| 3.9.b Prizes | "In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions)." |
| 3.14.a Warranty | "You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses." |
| 3.18.c Definition | "A “Final Submission” is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard." |

Overview, Code Requirements [R]: "Freely & publicly available external data is allowed, including pre-trained models".

**Deadline for declaring external data:** none. Neither the Rules tab nor the Overview has a declaration thread or a
date. The only dates are the entry / merger deadline (2026-12-07) and the final submission deadline (2026-12-14).

**What the text does not say.** It does not forbid a non-commercial *input*: 2.5.a third paragraph expressly foresees
"input data or pretrained models with an incompatible license". It does require OSI-approved, commercially usable
licences for *code* (2.5.a, 3.6.c). The stricter line on inputs comes from the welcome post (next table).

### 1.2 Host comments [H]

| Thread | Who, when | Statement (quoted) |
|---|---|---|
| 741359 Welcome post (pinned) | David Healey, about 09-15 | "Per the competition rules, the winning solution cannot include any licenses that would otherwise prohibit commercial use or redistribution. [...] Any external data, models, or tools you rely on must be compatible with that license. We're aware that datasets exist which carry licenses that fall into this category, i.e. terms that restrict commercial use or redistribution. These should be avoided. Solutions built on data or tools that cannot be commercially licensed will not be eligible to win, and we reserve the right to disqualify any submission that does not comply or can’t be reproduced with allowed data and tools. As always, any external data used must be publicly available and equally accessible to all participants." |
| 741844 | David Healey, about 09-18 | "The main question with model weights is whether the license for the data in question allows redistribution (even noncommercial redistribution) of derivatives like model weights. If so, then you should be fine to use it." |
| 742193 | David Healey, about 09-21 and 09-25 | "We can’t blanket approve all open weights models. It should be fine as long as the open weight model is distributed in compliance with the license for the data it was trained on." Later: "Many datasets (MassSpecGym perhaps) are functionally open source and many prominent open source models train on them. These are fine for solutions, as are any models trained on this competitions train.parquet, which is compiled from open-source data. (enveda-np-examples we neglected to assign a specific license to but should also be considered open source)" and "There are also fully commercial datasets like METLIN and those distributed by the mass spec companies (Thermo, Bruker, SciEx). Also to avoid." |
| 742265 | David Healey, about 09-25 | "Yes, any open source models trained on train.parquet are fair game. train.parquet is compiled from open source libraries and we are essentially just redistributing them with some minor touching up and normalization." |
| 742991 | David Healey, about 09-25 | "For the purposes of solution eligibility, we consider MassSpecGym and models trained on it and released with permissive licenses as fair game." |
| 743236 | David Healey, about 09-26 | "The Competition Data in train.parquet is a compilation of open-source databases, and so models trained on it can be open-sourced. [...] the winning solutions should be open-sourced. And the artifacts you describe seem acceptable open-source-derived solutions." (The question described private Kaggle datasets with a PubChem index, a COCONUT table, a library built from `train.parquet` and model weights.) |
| 744338 | David Healey, about 09-30 | "Yes, MassSpecGym is allowed, so is GLACIER" |
| 744470 NIST | David Healey, about 09-30 | "I’m sorry, the NIST dataset is not permitted for solutions, any solution using or trained on NIST will be disqualified. It is not freely available and its license has restrictions on redistribution of itself or derivatives, including model weights." |
| 745604 DreaMS | Marie Killian (host), about 10-06 | "Thanks for clarifying! Yes, DreaMS weights are allowed." |
| 745841 | David Healey, about 10-06 | "GLACIER/ICEBERG, DreaMS and ChEBI/LIPID MAPS are all fine. Regarding leaderboard rank and medals, I will make a broader announcement but I believe the compliance check at the end can remove teams from the leaderboard entirely, not just disqualify from prizes. Regarding the training data: the train.parquet file can be treated effectively as open-source and any model trained on it is fine, redistributed artifacts are fine. The noncommercial provision of the Kaggle data license is primarily to protect the test data; the training data is already a compilation of open source libraries." (The question named "ChEBI/LIPID MAPS candidates under CC BY-NC-SA".) |
| 745832 | David Healey, about 10-08 | First: an annotation set matched against NIST "would be a derivative of NIST and NIST does not allow derivatives". Then: "There is at least one dataset in GNPS that has this form, with fairly old and small NIST data. We'll say if it was on GNPS well before the competition started (say, 6 months), and hasn't been taken down, its fair game." |

**Thread 745072** ("Prize eligibility: code and weights shared in public Kaggle datasets with an 'Other
(non-commercial)' licence", gennexuslabsDDM, about 10-02): **no host answer** as of 2026-10-10. Eleven comments, all
from participants. Open in it: whether 3.6.b covers code shared inside a public Kaggle dataset; whether the
ahmedberatozer datasets (`casmi26-v4b-models`, `-v3-models`, `-v2-pool`, `-fpnet-full1`, `casmi26-pubchem-tier`) are
acceptable by name; whether winners must deliver training code for third-party checkpoints; whether the compliance
check covers only the two final submissions; when the "broader announcement" comes. Also without a host answer:
745615 (non-commercial inputs: prizes only, or leaderboard too), 746430 (FRIGID CC BY-NC checkpoints), the follow-up
in 742265 on code files without a licence header, the `GNPS-NIST14-MATCHES` question in 745832.

### 1.3 Our own caution [O]

- Unknown training data counts as not usable (the host cannot approve what nobody documents).
- A licence label that says non-commercial is treated here as a problem even where a host comment tolerates the
  content, because the label is what a compliance reviewer sees first and the host's by-name confirmation is pending.
- The popularity prior counts every PubChem substance record, including records deposited by the NIST Mass
  Spectrometry Data Center. These are counts of public PubChem records (public domain), no spectra, no NIST file. Not a
  rule problem; noted because the word NIST appears in the source's depositor list.

## 2. What the notebook attaches and loads

Source of the lists: `tools/build_notebook.py` (`DATASETS`, 15 entries, attached to every build, plus flag-dependent
ones), `nb/01_setup.py` (`find()`), `nb/02` to `nb/08`, `nb/embed/**`. The v33b preset is
`configs/v33b-v24a-fsranker-gate07-rankavg.json` (`POP_MU 0.25`, `RANKER 'fs_avg'`, `PC_JOIN_N 100`); `v59a-v33b-ccby`
is the same config built with the CC BY ChEBI / LIPID MAPS dataset. Defaults that matter: `FP_BANK='full1'`,
`USE_ENG / USE_PC / USE_ICE / USE_GL = True`.

Files loaded by the v33b chain:

| Dataset | Files read | By |
|---|---|---|
| competition | `train.parquet` (library, engine 2), `test.parquet` | `nb/02`, `nb/05` (`build_spec_cache`), `eng_runner.py` |
| `metric/rdkit-2026-3-3-wheel` | `rdkit-2026.3.3-cp312-*.whl` (pip, offline) | `nb/01_setup.py` |
| `ahmedberatozer/casmi26-v4b-models` | `MANIFEST.json`, `code/casmi/*.py`, `code/fe_v4/**` (code and `res/` tables), `fe_A.pt`, `fe_B.pt`, `fe_D.pt`, `ranker_v4b.pkl` (pickle: 4 LightGBM boosters, 160 features), `fpnet_0.pt` (sha check; model only with `FP_BANK='A'`) | `nb/01`, `nb/05` |
| `ahmedberatozer/casmi26-fpnet-full1` | `fpnet_full1.pt` | `nb/05` |
| `ahmedberatozer/casmi26-v3-models` | `code/casmi/*.py`, `fpnet_0.pt`, `fpnet_1.pt` (`ranker_0.pkl` only locates the folder) | `nb/embed/pc/pc_runner.py` |
| `ahmedberatozer/casmi26-v2-pool` | `pool_meta.parquet`, `pool_fp.npy`, `pool_frag_off.npy`, `pool_frag_mass.npy`, `fp_bits.npy`, `train_structs.parquet`, `train_fp_sel.npy` | `nb/05`, `pc_runner.py` |
| `ahmedberatozer/casmi26-pubchem-tier` | `pc_mass.npy`, `pc_off.npy`, `pc_smiles.npy` | `probe_core2.py` |
| `dmitriigluzdov/casmi26-pubchem-popularity-prior` | `pool_lsid.npy`, `pool_lpmid.npy` (`POP_MU`); `pc_lsid.npy`, `pc_lpmid.npy` only with `PC_POP_LAM > 0`; `pool_np.npy` only in validation modes | `nb/08`, `nb/04`, `nb/02` |
| `dmitriigluzdov/casmi26-fold-safe-fpnet` | `fpnet_fold0.pt` (`RANKER='fs_avg'`); `split.parquet` only in validation modes | `nb/05`, `nb/02` |
| `huseyinemreaksoy/casmi26-ours-ranker-fs` (private) | `ranker_s0..s3.txt`, `features.json` | `nb/05` |
| `prvsiyan/casmi26-fp-models-v2` | `fp_single_s2.pt`, `fp_merged_m1.pt` (`select_fp_models` keeps only paths of this dataset) | `eng_runner.py` |
| `prvsiyan/casmi26-ranker-features` | `rank_train.npz` | `casmi_engine.py` |
| `megayak/casmi26-simulated-ranker-rows` | `sim_rank_rows_nofp.npz` (its two `ours_fpnet_*.pt` are **not** loaded: the glob is `fp_*.pt`) | `eng_runner.py` |
| `prvsiyan/coconut-casmi26-candidates` | `coco_fp.npy`, `coco_mass.npy`, `coco_meta.pkl`, `fp_bits.npy` | `casmi_engine.py` |
| `sergiomalv/casmi-chebi-lmsd-structures` (before 10-09: `prvsiyan/chebi-lipidmaps-casmi26`) | `bio_fp.npy`, `bio_mass.npy`, `bio_meta.pkl` | `casmi_engine.py` |
| `ahmedberatozer/casmi26-iceberg` | `ice_runner.py`, `fuse.py`, `ckpt/gen.pt`, `ckpt/inten.pt`, `src/ms_pred/**`, `shim/**`, `wheels/*.whl.ice` | `nb/07` |
| `ahmedberatozer/casmi26-glacier` | `gl_runner.py`, `gl_fuse.py`, `ckpt/glacier.pt`, `src/ms_pred/**`, `shim/**` | `nb/07` |

Python packages: `torch`, `lightgbm`, `scikit-learn` (engine 2 fits `HistGradientBoostingClassifier` in the run),
`numba`, `pandas`, `pyarrow` come from the pinned Kaggle image (`build_notebook.DOCKER_IMAGE`); no offline wheel for
them. Offline wheels: RDKit 2026.03.3 (metric dataset); for the forward models RDKit 2025.3.6, pytorch_lightning 2.6.6,
torchmetrics 1.9.0, lightning_utilities 0.15.3, einops 0.8.2 (inside `casmi26-iceberg`). Kaggle Models and notebook
outputs: none (`model_sources`, `kernel_sources` are empty).

## 3. Inventory with verdicts

Verdict classes: **clean** = permissive or public-domain terms, documented origin, no NIST; **problem** = a
non-commercial label or note, unknown training data, or a wrong label of ours; **unclear** = the publisher or the host
has not said. "Licence field" is what `kaggle datasets metadata` returned on 2026-10-10.

### 3.1 In the v33b chain (the notebook as the builder makes it today)

| # | Item | Owner, public? | Licence field | Underlying material and training data (as stated by the publisher) | Verdict and evidence |
|---|---|---|---|---|---|
| 1 | Competition data (`train.parquet` at run time) | Enveda, competition | CC BY-NC 4.0 [R 1.7] | compilation of open libraries (GNPS, MassBank, MoNA, RIKEN, MSnLib, MS-DIAL, Enveda sets) | **clean**: [H] 745841 "can be treated effectively as open-source"; 742193 |
| 2 | `metric/rdkit-2026-3-3-wheel` | Kaggle `metric` account, public | "unknown" | RDKit 2026.03.3 wheels (RDKit is BSD-3-Clause) | **clean**: the metric's own dataset; BSD-3 |
| 3 | `ahmedberatozer/casmi26-pubchem-tier` | public | Other | card: "Mass-sorted PubChem structures (CID-SMILES + CID-Mass, NCBI PubChem public data [...])" | **clean** by content (PubChem, public domain). The field is "Other" with no terms; a by-name confirmation was asked in 745072 and is unanswered |
| 4 | `prvsiyan/coconut-casmi26-candidates` | public | CC BY 4.0 | COCONUT 2.0 structures and fingerprints; card: "Preserve COCONUT attribution" | **clean**; attribution owed |
| 5 | `sergiomalv/casmi-chebi-lmsd-structures` | public | CC BY 4.0 | ChEBI release 255 (CC BY 4.0), LIPID MAPS LMSD of 2026-10-03 (CC BY 4.0), "no competition data"; `ATTRIBUTION.md` | **clean**; attribution owed (citations on the card). Identical fingerprints for 2,000 of 2,000 rows (pubdiff note of 10-09) |
| 6 | `dmitriigluzdov/casmi26-pubchem-popularity-prior` | public | Other | card "Terms": PubChem Compound Extras of 2026-09-27 ("NCBI distributes them for free reuse"), LOTUS flags CC0, NPAtlas flags CC BY 4.0, "No competition data are included"; build scripts included | **clean** by content. [O] counts include deposits by NIST MSDC (public PubChem records) |
| 7 | `megayak/casmi26-simulated-ranker-rows` (`sim_rank_rows_nofp.npz`) | public | CC0-1.0 | README: 2,250 simulated molecules from seven `ingest_lib` sets of train, pool COCONUT + training structures, "only numeric features" | **clean**: train-derived [H 745841], CC0 |
| 8 | `ahmedberatozer/casmi26-iceberg`: `gen.pt`, `inten.pt`, `src/ms_pred`, wheels | public | Other | `LICENSE-NOTICE.txt`: ms-pred commit 708148c2a8eb, MIT, © 2023 Samuel Goldman; "ICEBERG msg_all (MassSpecGym-trained) weights published with ms-pred (README 'open-source MassSpecGym-trained weights')", re-saved without optimiser state | **clean**: [H] 744338, 745841, 742991. The ms-pred README at that commit separates "open-source MassSpecGym-trained weights" from NIST weights sent only "with a proof of license" (read 10-10). Not verified by us: that the re-saved tensors equal the upstream file |
| 9 | `ahmedberatozer/casmi26-glacier`: `glacier.pt`, `src/ms_pred` | public | Other | `LICENSE-NOTICE.txt`: "GLACIER_checkpoint.zip -> best.ckpt, sha256 5a47cecca707d3ab..."; "The download carries no separate license file; used under the ms-pred repository's MIT License" | **clean**: [H] 744338 "so is GLACIER". The same sha256 prefix is recorded independently on the card of `prvsiyan/casmi26-forward-reference-stack` for the MassSpecGym GLACIER checkpoint |
| 10 | `ahmedberatozer/casmi26-v2-pool` | public | Other | card: "contains data and/or models derived from the Enveda CASMI26 competition training data (which includes CC BY-NC licensed components) and COCONUT (CC BY 4.0). Use under the respective source licenses (non-commercial, with attribution)." | **problem** (non-commercial note). Content is train + COCONUT, which [H] 745841 calls fine; the by-name question (745072) is unanswered |
| 11 | `ahmedberatozer/casmi26-fpnet-full1` (`fpnet_full1.pt`, the v1 engine's FPNet) | public | Other | card: "trained on ALL train folds incl. np-examples [...] Use under the source licenses (non-commercial, with attribution)." No training code | **problem** (non-commercial note); training data per card: competition training data only |
| 12 | `ahmedberatozer/casmi26-v3-models` (`fpnet_0.pt`, `fpnet_1.pt`: PubChem channel) | public | Other | same source note as item 10; no training code | **problem** (non-commercial note) |
| 13 | `ahmedberatozer/casmi26-v4b-models`: `fe_A.pt`, `fe_B.pt`, `ranker_v4b.pkl`, `fe_v4/res/**` tables, `fpnet_0.pt` | public | Other | same source note; `MANIFEST.json` names internal sources (`work/fpnet_a.pt`, `work/analysis/v1fe/ranker_v4b.pkl`); no training code, ranker training rows not published | **problem** (non-commercial note) |
| 14 | `fe_D.pt` of item 13 (DreaMS-based feature model) | public | Other | `LICENSE-NOTICE.txt`: "fe_D.pt is derived (fine-tuned for spectrum -> fingerprint prediction) from DreaMS (Bushuiev et al., Nature Biotechnology 2025; weights MIT / CC-BY-4.0, code MIT [...]); redistributed under the same terms with this attribution." `MANIFEST`: source `dreams_ft/s2/pilot_d1_step20k_ema.pt` | **unclear**: which DreaMS checkpoint is the base is **not stated**, and the fine-tuning data is stated only by the dataset's general note (competition data). [H] 745604 "DreaMS weights are allowed"; the DreaMS paper pre-trains on GeMS (GNPS / MassIVE) and trains its embedding model on about 25,000 MoNA spectra "and do not use any spectra from NIST20 for training", but it also fine-tunes other heads on "the combined MoNA and NIST20 datasets". Whether such a head was ever released, and which file ahmedberatozer started from: unknown |
| 15 | `dmitriigluzdov/casmi26-fold-safe-fpnet` (`fpnet_fold0.pt`) | public | Other | card "Terms": "The checkpoint is derived from the competition training file, which includes components under non-commercial licences; use it under the competition data terms, non-commercially and with attribution. [...] the training loop and evaluation scripts in this dataset are released under Apache 2.0." Trained on train without 55,102 + 265 held-out structures; training code, log and split published | **problem** (non-commercial terms on the checkpoint). Best-documented model of the chain; no NIST |
| 16 | `prvsiyan/casmi26-fp-models-v2` (`fp_single_s2.pt`, `fp_merged_m1.pt`: engine 2) | public | **CC BY-NC-SA 4.0** | subtitle: "Training-derived artefacts for the CASMI 2026 notebooks (CC BY-NC-SA)". The card body still says "The existing Kaggle license is CC0-1.0" and "Checkpoint metadata inspection does not establish a training split, upstream corpus snapshot". Our record of 10-05: field CC0-1.0, card empty | **problem**: non-commercial share-alike label (changed from CC0 after 10-07); training corpus not stated beyond "training-derived"; no training code. megayak's notebook describes them as trained on most library structures (our note in `reviews/np-ours-2026-10-05.md`) |
| 17 | `prvsiyan/casmi26-ranker-features` (`rank_train.npz`: rows for an engine-2 ranker fitted in the run) | public | **CC BY-NC-SA 4.0** | same subtitle; body says CC0-1.0; "a derived training resource, not a raw spectral library"; 142,762 rows × 31 features | **problem**: non-commercial share-alike label |
| 18 | `huseyinemreaksoy/casmi26-ours-ranker-fs` | ours, **private** | Other | README: four LightGBM models on 6,687 candidate lists from `train.parquet` only; features from items 10, 13, 15; candidates from items 3 and 10; "No NIST data and no NIST-trained model" | **unclear**: built from allowed inputs as far as items 10, 13, 15 are allowed; it inherits their status. A private dataset with a team's own weights was described to the host in 743236 ("seem acceptable"); no explicit ruling |
| 19 | Embedded code from the reference notebook (`reference/v4n/*`, `nb/0[1-8]_*.py`, `nb/embed/pc/*`, `eng_runner.py`) | seyitkaangunes, public notebook | Apache 2.0 (notebook page, read 10-10) | – | **clean**: [R] 3.6.b, Apache 2.0 |
| 20 | `nb/embed/eng/pv.py`, `pv_fp.py`, `casmi_engine.py` | prvsiyan, megayak (public notebooks) | Apache 2.0 by Kaggle's notebook default; their pages were not opened today | – | **clean** under [R] 3.6.b (code shared in competition notebooks) |
| 21 | Additions from wangpenghua's public notebook (`probe_core2.py` tail, `GL_MH_ONLY`); the popularity-prior formula of bobthebot369 v10 | public notebooks | as 20 | – | **clean** under [R] 3.6.b |
| 22 | `code/casmi/**`, `code/fe_v4/**` (the v1 engine's source, imported from items 12 and 13 at run time) | ahmedberatozer, inside datasets | none for the code; dataset field Other | his notebook `casmi26-v4g-inference` is Apache 2.0 (page read 10-10) but imports this package from the datasets | **unclear**: [R] 3.6.c asks for an OSI licence; 3.6.b speaks of "the discussion forum or notebooks"; whether a public dataset counts is the open question of 745072 and 742265 |
| 23 | `ice_runner.py`, `fuse.py`, `gl_runner.py`, `gl_fuse.py`, `shim/**` (items 8, 9; "own code" per the notices) | ahmedberatozer, inside datasets | none for the code | – | **unclear**: same question as 22 |
| 24 | `src/ms_pred/**` (items 8, 9), `code/casmi/dreams_model.py` (item 13) | coleygroup; re-implementation of DreaMS | MIT (licence texts shipped in the datasets) | – | **clean**; the MIT notices must travel with any redistribution |
| 25 | Our own modules (`fusion_core.py`, `pc_join.py`, `fs_pass.py`, `ranker_infer.py`, `frag_rescore.py`, `conf_gate.py`, `determ.py`, `surrogate.py`, `mass_window.py`, `pop_split.py`, `adduct_prior.py`) | ours | MIT (repo `LICENSE`) | – | **clean** |
| 26 | Wheels inside item 8 | PyPI | per `LICENSE-NOTICE.txt`: RDKit 2025.3.6 BSD-3-Clause; pytorch_lightning, torchmetrics, lightning_utilities Apache-2.0; einops MIT; "unmodified PyPI wheels, renamed" | – | **clean** (the "unmodified" claim was not checked against PyPI hashes) |
| 27 | Kaggle image packages (torch, LightGBM, scikit-learn, numba, pandas, pyarrow) | Kaggle | BSD / MIT / Apache | – | **clean** |

### 3.2 On the shelf, or attached in the past (not in the v33b chain today)

| # | Item | Attached when | Licence field | What it is | Verdict and evidence |
|---|---|---|---|---|---|
| 28 | `prvsiyan/chebi-lipidmaps-casmi26` | **every submission so far**; removed from the builder on 10-09 | CC BY-NC-SA 4.0 | ChEBI + LIPID MAPS candidates; card empty | **problem** (non-commercial share-alike). [H] 745841 answered "ChEBI/LIPID MAPS are all fine" to a question that named this licence. Fixed by item 5 |
| 29 | `huseyinemreaksoy/casmi26-ours-ranker-obs` | preset `v48a` (`RANKER_FS_DS`); private, uploaded | Other | item 18's recipe on 13,092 lists with inverse-fame weights from item 6 | **unclear**: as item 18 |
| 30 | `huseyinemreaksoy/casmi26-ours-adduct-prior` | preset `v50a` (`ADD_JOIN_MU`) | not readable (API 403: not uploaded, or not visible to this token) | docstring of `nb/embed/adduct_prior.py`: seven LightGBM models "fitted on the Enveda timsTOF molecules of train.parquet only" | **unclear**: no card could be read |
| 31 | `huseyinemreaksoy/casmi26-pop-split` | preset `v53a` (`POP_SPLIT`); not uploaded (403), local folder | CC0-1.0 (local `dataset-metadata.json`) | derived from item 32; the notebook uses `sid - spec` only | **clean** |
| 32 | `sergiomalv/casmi-pubchem-popularity` | source of item 31, never attached | CC0-1.0 | PubChem FTP files of 2026-10-01, "no competition data" | **clean** |
| 33 | `huseyinemreaksoy/casmi26-fpnet-pc-hn` | flag `PC_MODELS='hn'` (v22a / v22b); private | **Apache-2.0** | dataset README: `fpnet_fold0.pt` of item 15 fine-tuned by us on train with PubChem negatives. The solution README says "FPNet A and B of `casmi26-v3-models`"; the dataset holds one file | **problem**: our Apache-2.0 label contradicts the non-commercial terms of its base (item 15 or 12), and our two descriptions of the base disagree |
| 34 | `huseyinemreaksoy/casmi26-ours-ranker` | flags `RANKER='ours' / 'blend'`; private | Other | first own ranker on the public models' features, from train | **unclear**: as item 18; not in any current preset |
| 35 | `huseyinemreaksoy/casmi26-fp-np-ours` | flag `ENG_EXTRA_FPNET='np-ours'` | not readable (403) | our fine-tune of an engine-2 FPNet (`train/train_eng_fpnet.py`) | **unclear**: no card could be read; base is item 16 |
| 36 | `goooman/casmi26-fp-npx4-soup` | preset `v25a` (teammate's 0.413 submission) | Apache-2.0 | weight average of five fine-tunes of `fp_np_l4` on the 250 Enveda NP standards | **problem**: the training data of the base model `fp_np_l4` is not documented anywhere and no code is public |
| 37 | `ahmedberatozer/casmi26-ice-ft` | flag `ICE_FT` (v20a) | Other | card: only the general source note (competition data, COCONUT; "non-commercial, with attribution"); one file `inten.pt` | **problem**: non-commercial note; what the fine-tune was trained on is stated only generically |

Counts: **clean 18** (1 to 9, 19 to 21, 24 to 27, 31, 32), **problem 11** (10 to 13, 15 to 17, 28, 33, 36, 37),
**unclear 8** (14, 18, 22, 23, 29, 30, 34, 35).

## 4. Trained weights, one by one

| Checkpoint | Used for | Trained on, per its publisher | NIST or restricted data? | Does the publisher's licence allow our use? |
|---|---|---|---|---|
| `fpnet_full1.pt` (item 11) | v1 engine fingerprint model | "ALL train folds incl. np-examples" | none stated; train only | non-commercial use with attribution: yes inside the competition; not compatible with a commercially usable winner licence as labelled |
| `fpnet_0.pt`, `fpnet_1.pt` of v3-models (item 12) | PubChem channel | "derived from the [...] competition training data [...] and COCONUT"; recipe unpublished | none stated | as above |
| `fe_A.pt`, `fe_B.pt` (item 13) | feature models of the public ranker (`model_views`) | same general note; `MANIFEST` sources `work/fpnet_a.pt`, `work/fpnet_b.pt` | none stated | as above |
| `fe_D.pt` (item 14) | feature model, DreaMS-based | fine-tune of DreaMS; base checkpoint **unknown**; fine-tune data: general note only | DreaMS pre-training: GeMS (GNPS / MassIVE). NIST20 appears in the paper for evaluation and for some fine-tuned heads; whether the base used here is one of them: **unknown** | DreaMS terms (MIT / CC BY 4.0) allow it; the dataset's own note is non-commercial |
| `ranker_v4b.pkl` (item 13) | public LightGBM ranker, 4 boosters, 160 features | training rows not published; general note | none stated | as item 11 |
| `fpnet_fold0.pt` (item 15) | second engine pass (`fs_avg`) | `train.parquet` minus 55,102 + 265 structures; code, log, split published | no | non-commercial with attribution per "Terms" |
| `ranker_s0..s3.txt` (item 18) | our ranker of the second pass | 6,687 lists from `train.parquet`; features through items 10, 13, 15 | no | ours |
| `fp_single_s2.pt`, `fp_merged_m1.pt` (item 16) | engine 2 fingerprint models (6,930 bits) | **not stated** beyond "training-derived artefacts"; no code | **unknown** | CC BY-NC-SA 4.0 as labelled today: non-commercial, share-alike |
| engine-2 rankers | fitted inside the run by scikit-learn | rows of items 7 (CC0) and 17 (CC BY-NC-SA) | rows are numeric features from train-based simulations | item 17: non-commercial share-alike |
| ICEBERG `gen.pt`, `inten.pt` (item 8) | forward re-score | MassSpecGym `msg_all` weights of ms-pred ("open-source MassSpecGym-trained weights") | **not the NIST weights**: evidence is the packager's notice, the ms-pred README, the dataset name of the checkpoint; the host approved by name | MIT (ms-pred) |
| GLACIER `glacier.pt` (item 9) | forward re-score | MassSpecGym weights of ms-pred, sha256 `5a47cecc…` of the upstream file | not the NIST weights; host approved by name | no separate licence file upstream; used under the repository's MIT licence |
| `inten.pt` of ice-ft (item 37), `fp_single_npx4soup.pt` (item 36), `fpnet-pc-hn` (item 33) | off | see section 3.2 | 36: **unknown** base | 37: non-commercial; 36: Apache-2.0 on undocumented weights |

## 5. Fixes, ordered by risk

| Rank | Items | Risk | Fix |
|---|---|---|---|
| 1 | 22, 23: code inside the ahmedberatozer datasets has no licence | highest: the one point where the rule text itself ([R] 3.6.c, 2.5.a) is not met on its face, and it covers the whole v1 engine and both forward-model runners | (a) Ask the host, in 745072, one narrow question: "Is source code published inside a public Kaggle dataset for this competition covered by Foundational Rule 6.b?" (b) Ask ahmedberatozer to add MIT or Apache-2.0 to the code (already requested in 745072 by another participant, unanswered). (c) Fallback for the runners only: `prvsiyan/casmi26-forward-reference-stack` (about 3.8 GB, field Other, card: runner MIT, ms-pred MIT, the same MassSpecGym checkpoints; not downloaded, not tested; its wheels include `easydict` LGPL-3.0 and NVIDIA runtime wheels). There is no licensed replacement for `casmi` / `fe_v4` |
| 2 | 10 to 13: the v1 engine's pool, FPNets, feature models and ranker carry "non-commercial, with attribution" | high by label, low by content ([H] 745841 covers train-derived artefacts) | No permissively labelled public copy exists (`dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets` mirrors them under the same terms). (a) Ask the host for the by-name confirmation already requested in 745072 and 745841. (b) Ask ahmedberatozer to relabel, since the host has said train-derived artefacts can be open-sourced (743236). (c) Own rebuild from `train.parquet` + COCONUT with `train/train_fpnet.py` (Apache-2.0 loop): removes the label for the weights, not for the engine code (rank 1); effect on the score unknown, several GPU days |
| 3 | 16, 17: engine 2's FPNets and ranker rows are now CC BY-NC-SA 4.0 | high by label; the label is newer than our last check and contradicts the card body | (a) Ask prvsiyan which licence holds (field or card body), and the host as in rank 2. (b) Replacement for the FPNets, already attached and CC0: `megayak/casmi26-simulated-ranker-rows` `ours_fpnet_single_16k.pt` and `ours_fpnet_merged_26k.pt` (144 MB each; prvsiyan's architecture retrained on train with 3,150 simulation structures held out) through a new flag in `select_fp_models`; or `prvsiyan/casmi26-fp-models-v4` (CC0-1.0, 4 × 432 MB; its card warns that it is not a drop-in). (c) Replacement for `rank_train.npz`: fit that ranker on megayak's CC0 rows, or regenerate the rows with the embedded code. (d) Removal: `USE_ENG=False`. Effect of (b) to (d): not measured; engine 2 enters the final fusion with weight 0.6, so each needs a replay and a submission |
| 4 | 15: fold-safe FPNet, non-commercial terms | medium: train-derived, fully documented | Retrain it ourselves with the published Apache-2.0 loop and split (5 h on an RTX 6000 per its card) and refit item 18 on the new features; or ask the author to relabel. Until then covered by [H] 745841 only |
| 5 | 14: `fe_D.pt` | medium: DreaMS is approved, the base file is not named | Ask ahmedberatozer which DreaMS checkpoint `fe_D.pt` was fine-tuned from and on what. Removal is not a switch: `fe_D` feeds `model_views` features of `ranker_v4b.pkl` |
| 6 | 18, 29, 30, 34, 35: our private datasets | medium: [R] 2.6.a and the welcome post ask for publicly available external data; whether a team's own weights count is not ruled | Give every one a card with inputs and a licence consistent with its inputs; upload 30 and 31 only when a preset needs them; be ready to publish them (the README already says so for `fpnet-pc-hn`). If the hosts require it, publish before the final deadline |
| 7 | 33: `casmi26-fpnet-pc-hn` labelled Apache-2.0 | low (off) but it is our own wrong label | Change the field to Other with the upstream note, and make the solution README and the dataset README agree on the base model |
| 8 | 36, 37 | low (off) | Keep `ENG_EXTRA_FPNET='goooman/…'` and `ICE_FT` out of any final candidate; 36 only after the author documents `fp_np_l4` |
| 9 | 28 | done in the builder | Keep `sergiomalv/casmi-chebi-lmsd-structures`; add its attribution (rank 10). It has not been in a submission yet |
| 10 | `THIRD_PARTY_NOTICES.md` | documentation | section 6 |

Questions worth putting to the host in one post (745072), each answerable with yes or no: (1) code inside public
datasets and Rule 6.b; (2) the ahmedberatozer and prvsiyan datasets by name, as labelled; (3) a team's own private
dataset of weights trained on `train.parquet`; (4) compliance checked for the two final submissions only?

## 6. `THIRD_PARTY_NOTICES.md` against the inventory

Missing or wrong:

1. The runtime list still names prvsiyan's "ChEBI+LIPID MAPS candidates"; the builder now attaches
   `sergiomalv/casmi-chebi-lmsd-structures` (CC BY 4.0), whose card asks to keep its attribution: ChEBI (Hastings et
   al., NAR 44:D1214, 2016), LIPID MAPS LMSD (Sud et al., NAR 35:D527, 2007; Conroy et al., NAR 52:D1677, 2024).
2. Not listed at all although v33b reads them: `dmitriigluzdov/casmi26-pubchem-popularity-prior` (PubChem; LOTUS CC0;
   NPAtlas CC BY 4.0) and `dmitriigluzdov/casmi26-fold-safe-fpnet` as a runtime dataset (it appears only as the origin
   of `train_fpnet.py`).
3. `train_fpnet.py` row: "(Apache-2.0)" is true for the training loop only. The dataset's field is Other and its
   checkpoint is under non-commercial terms.
4. No licence per runtime dataset. The file says "Each keeps its own licence (see its data card)"; the actual labels
   are: ahmedberatozer all "Other" with a non-commercial note (v2-pool, v3-models, v4b-models, fpnet-full1) or MIT
   content (iceberg, glacier) or PubChem (pubchem-tier); prvsiyan fp-models-v2 and ranker-features CC BY-NC-SA 4.0,
   COCONUT candidates CC BY 4.0; megayak CC0-1.0; metric wheel "unknown" (RDKit BSD-3).
5. Missing upstream attributions that the packages themselves require or carry: ms-pred (MIT, © 2023 Samuel Goldman,
   commit 708148c2a8eb); DreaMS (MIT, © 2024 Roman Bushuiev; weights MIT / CC BY 4.0) for `fe_D.pt` and
   `dreams_model.py`; MassSpecGym as the training set of the forward models; COCONUT; PubChem (Kim et al., NAR
   53:D1516, 2025); RDKit (BSD-3); the wheels of the ICEBERG package (pytorch_lightning, torchmetrics,
   lightning_utilities Apache-2.0; einops MIT; RDKit 2025.3.6).
6. megayak's public notebook ("Two Rankers One Engine", 09-17) is not named as an origin of `casmi_engine.py`
   (`docs/landscape/pubdiff-2026-10-09.md` identifies it); bobthebot369 (popularity prior formula in
   `fusion_core.py`) is named in the code and not in the notices.
7. Flag-dependent datasets are absent: `goooman/casmi26-fp-npx4-soup`, `ahmedberatozer/casmi26-ice-ft`, and our five
   private datasets (only `casmi26-pop-split` is described).
8. `solution.json` → `provenance` and the README's provenance section still describe the state before the relabelling
   of the prvsiyan datasets and do not mention the non-commercial notes of the ahmedberatozer datasets.

## 7. Past submissions and the final selection

- **Every submission so far** (all rows of `docs/experiments.md`, the 0.442 of v33b and v40a included) attached
  `prvsiyan/chebi-lipidmaps-casmi26` (CC BY-NC-SA 4.0); `v59a-v33b-ccby` exists as a config and is not in the log.
- Replacing that dataset does not make a submission label-clean: items 10 to 13 and 15 to 17 are in every one of
  them as well (15 from v33a / v33b on). **Under a strict reading of the labels, none of our existing submissions is
  clean, and none can be made clean today with public replacements**, because the v1 engine has none.
- Under the rules as written and the host comments quoted above, none of them uses anything the host has prohibited:
  no NIST, no METLIN or vendor data; ChEBI / LIPID MAPS, ICEBERG / GLACIER, DreaMS and train-derived artefacts are
  each covered by a host statement ([H] 745841, 744338, 745604, 742265). The remaining exposure is the unanswered
  part: code without a licence in datasets, and the by-name status of the "Other" / CC BY-NC-SA datasets.
- Only the (up to) two Final Submissions are judged for placement ([R] 2.2.b, 3.18.c), and 3.9.b speaks of "the
  compliance of the winning Submissions" with a one-week remediation option. Whether earlier submissions are looked at
  was asked in 745072 and is **unanswered**. Practical consequence: the final pair should be built after the open
  questions are answered, from the cleanest chain available then; at the least it should be a `v59a`-type build (CC
  BY ChEBI / LIPID MAPS), which needs one submission to exist.

## 8. Not verified

- Licence shown on the notebook pages of prvsiyan, megayak, wangpenghua, bobthebot369 (only seyitkaangunes's and
  ahmedberatozer's pages were opened: Apache 2.0).
- That the re-saved ICEBERG / GLACIER tensors equal the upstream ms-pred files, and that the shipped wheels equal
  PyPI's (publisher's statements only).
- The Zenodo record of DreaMS (which checkpoints are released, their licence); the base of `fe_D.pt`.
- Training data of `fp_single_s2.pt` / `fp_merged_m1.pt`, of `ranker_v4b.pkl`'s rows, of `casmi26-ice-ft`, of
  `fp_np_l4`: not published.
- Cards of `casmi26-ours-adduct-prior`, `casmi26-pop-split`, `casmi26-fp-np-ours` on Kaggle (API 403).
- When exactly prvsiyan changed the licence field (CC0-1.0 in our note of 10-05 and on his card text of 10-07; CC
  BY-NC-SA 4.0 on 10-10).
- Upstream licences of COCONUT, ChEBI, LIPID MAPS, LOTUS, NPAtlas were taken from the dataset cards, not re-read at
  the sources.
- The fifth comment of thread 745832 that the forum index announced ("last comment 2h ago by David Healey") was not
  visible signed out; the four visible comments are quoted.
- `INDEX.md` was not regenerated (`python tools/gen_index.py`), since this audit was read-only apart from this file.
