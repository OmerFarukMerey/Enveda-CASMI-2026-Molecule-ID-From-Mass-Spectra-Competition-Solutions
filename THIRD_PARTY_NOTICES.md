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
