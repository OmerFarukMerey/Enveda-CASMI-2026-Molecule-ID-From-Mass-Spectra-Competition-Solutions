<!-- index: licences and origins of copied code (Apache-2.0 public notebooks) and runtime Kaggle datasets -->
# Third-party notices

This repository is MIT-licensed ([LICENSE](LICENSE)) except for the files below, which keep their original licence.

## solutions/fusion-v4n
Code derived from public Kaggle notebooks for this competition, released under the **Apache License 2.0**
(Kaggle's default for public competition notebooks):

| Files | Origin |
|---|---|
| `reference/v4n/*` (verbatim), `nb/0[1-8]_*.py` (restructured), `nb/embed/pc/*`, `nb/embed/eng/eng_runner.py` (parametrised) | [CASMI26 \| v4n + Engine Fusion + Union [LB 0.399]](https://www.kaggle.com/code/seyitkaangunes/casmi26-v4n-engine-fusion-union-lb-0-399), Seyit Kaan Güneş, building on ahmedberatozer's v3/v4 inference notebooks |
| `nb/embed/eng/pv.py`, `pv_fp.py`, `casmi_engine.py` | prvsiyan's analog-propagation pipeline (public notebook, 2026-09-16 version), as embedded in the notebook above |
| `nb/embed/pc/probe_core2.py` (appended `init_worker_pop` / `probe_one_pop`), `GL_MH_ONLY` filter in `nb/07_rescore.py` | [casmi26 v29 fusion pop glmh](https://www.kaggle.com/code/wangpenghua/casmi26-v29-fusion-pop-glmh), wangpenghua ("v17 patch": popularity prior in the PubChem channel; GLACIER on [M+H]+ only) |
| `train/train_fpnet.py` (threads instead of fork, NP- and timsTOF-weighted sampling, resume, GPU guard, hold-out breakdown) | `train_fpnet.py` of the dataset [dmitriigluzdov/casmi26-fold-safe-fpnet](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-fold-safe-fpnet) (Apache-2.0); it imports the `casmi` package of ahmedberatozer/casmi26-v4b-models at run time (not redistributed) |
| `clean/train_fpnet.py` (the loop of `train/train_fpnet.py` on our own model and data modules `clean/fpnet.py`, `clean/specdata.py`: worker processes, step schedule, resume, GPU sharing, ledger, bucketed forward) | `train_fpnet.py` of the dataset [dmitriigluzdov/casmi26-fold-safe-fpnet](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-fold-safe-fpnet) (Apache-2.0); imports only our own modules (no `casmi` package) |
| `tools/metric.py` (reimplementation) | follows the rules of [metric/casmi-mean-reciprocal-rank](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank) |

Changes are marked in the files ("ours:" comments) and listed in `solutions/fusion-v4n/README.md`.

Kaggle datasets used at runtime are **not** redistributed here. Each keeps its own licence (see its data card):
ahmedberatozer (casmi26-v2-pool, v3-models, v4b-models, fpnet-full1, iceberg, glacier, pubchem-tier; ICEBERG / GLACIER
are ms-pred MassSpecGym checkpoints, MIT), prvsiyan (fp-models-v2, ranker-features, COCONUT and ChEBI+LIPID MAPS
candidates), megayak (simulated ranker rows), metric (rdkit wheel).

Preset `configs/v53a-v33b-popsplit.json` (flag `POP_SPLIT`, off by default) reads our dataset `casmi26-pop-split`, a
derived table of [sergiomalv/casmi-pubchem-popularity](https://www.kaggle.com/datasets/sergiomalv/casmi-pubchem-popularity)
(**CC0 1.0**; counts of PubChem substance records per InChIKey first block from the public PubChem FTP files of
2026-10-01; PubChem data are in the public domain, cite Kim S. et al., Nucleic Acids Res. 53:D1516, 2025). Only
`sid - spec` enters the notebook: the substance records that were *not* deposited by MassBank Europe, MoNA, the NIST
Mass Spectrometry Data Center or RIKEN NPDepo. No spectra and no NIST data are involved
(`solutions/fusion-v4n/reviews/popsplit-2026-10-08.md`).

Option `E006_FUSE` (off by default; presets `configs/v65a-v59a-e006-rrf.json`, `configs/v65b-v59a-e006-fill.json`) runs a second,
complete public notebook inside ours and fuses the two submission tables
(`solutions/fusion-v4n/reviews/e006-2026-10-10.md`):

| Files | Origin |
|---|---|
| `solutions/fusion-v4n/reference/e006/casmi26-e006-window-40-ice-fallback-true.ipynb`, `kernel-metadata.json` (verbatim, as pulled on 2026-10-10; sha256 of the .ipynb `ee55b55c…9ad7e53`); its 30 code cells are embedded unchanged by `tools/build_notebook.py` when the option is set | [CASMI26 E006 Window 40 ICE Fallback True](https://www.kaggle.com/code/canqiang/casmi26-e006-window-40-ice-fallback-true), canqiang, version 1, **Apache License 2.0**. It is flexonafft's "public baseline adaptation experiments" v80 reproduced, on obstacledeveloper's "V1.1 metric audit and safe tail", bobthebot369's v18, wangpenghua's v29, ahmedberatozer's v3 / v4 inference notebooks and prvsiyan's analog-propagation pipeline, with the rule "ICEBERG where GLACIER has no score" from samanyu1808 (all public Apache-2.0 notebooks; the notebook's own credit cells are kept in the snapshot). Its copy of the official metric helpers comes from [metric/casmi-mean-reciprocal-rank](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank) (Apache-2.0) |
| `solutions/fusion-v4n/nb/embed/e006_stage.py`, `nb/09_e006.py` (ours) | our own code. At run time `e006_stage.script_from_cells` applies three marked text patches to the cells above (work folder `/kaggle/working/e006`; engine-2 lists handed over from our own run instead of a second engine-2 run; full run on request) and defines a no-op `display`. Stated here as the notice of modification that the licence asks for |

Inputs: the snapshot's input `prvsiyan/chebi-lipidmaps-casmi26` (CC BY-NC-SA 4.0) is **not** used; the second pipeline reads
`sergiomalv/casmi-chebi-lmsd-structures` (CC BY 4.0), as our own engine 2 does since v59a. `hengck23/hengck23-dreams-enveda-casmi26`
(MIT) and `eveliaveldrine/rdkit-wheels` (Apache-2.0), attached by the original, are not attached to our combined notebook (never read
on the pinned Python 3.12 image); the private stand-alone rerun `casmi26-e006r` (`scratch/e006/e006r/`) keeps them attached.
A third preset of the same option, `configs/v65c-v59a-e006-rrf-w03.json` (the second table at weight 0.3, `E006_RRF_W`), uses the
same snapshot and code path; nothing else is added.


Option `E006_BASE` (off by default; presets `configs/v67a-e006-ours-base.json` and the other `v67*` files; `reviews/e006imp-2026-10-10.md`) runs
the same embedded cells as the main pipeline with our own base lists. `e006_stage.script_improved` applies four more marked text patches to the
cells at run time, each tagged `# ours: patch N` on the changed line and inert unless its environment variable is set: (4) our base list in
place of the notebook's engine call for molecules handed over, (5) another threshold for the `rel > 250` branch of its promotion rule,
(6) GLACIER on every covered spectrum instead of [M+H]+ only, (7) a row promoted by its popularity branch also leads the final list. This
paragraph is the notice of modification for those patches (Apache License 2.0, section 4(b)); no file of the snapshot is changed on disk, and
no dataset is added (the notebook's inputs are those of `configs/v59a-v33b-ccby.json`).