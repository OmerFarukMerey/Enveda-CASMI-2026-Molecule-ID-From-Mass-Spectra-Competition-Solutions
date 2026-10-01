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
| `tools/metric.py` (reimplementation) | follows the rules of [metric/casmi-mean-reciprocal-rank](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank) |

Changes are marked in the files ("ours:" comments) and listed in `solutions/fusion-v4n/README.md`.

Kaggle datasets used at runtime are **not** redistributed here. Each keeps its own licence (see its data card):
ahmedberatozer (casmi26-v2-pool, v3-models, v4b-models, fpnet-full1, iceberg, glacier, pubchem-tier; ICEBERG / GLACIER
are ms-pred MassSpecGym checkpoints, MIT), prvsiyan (fp-models-v2, ranker-features, COCONUT and ChEBI+LIPID MAPS
candidates), megayak (simulated ranker rows), metric (rdkit wheel).
