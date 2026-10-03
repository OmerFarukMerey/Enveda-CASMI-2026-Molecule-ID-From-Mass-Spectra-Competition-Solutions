<!-- index: human entry point: solutions table, where things are, quick start, repo rules -->
# Enveda CASMI 2026: Molecule ID from Mass Spectra (team solutions)

Team repo for the Kaggle code competition
[Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra):
rank up to 25 candidate structures (SMILES) per unknown molecule from its LC-MS/MS spectra. Metric: MRR@25 on the
tautomer-canonical InChIKey14. Final submission deadline: **2026-12-14**.

> AI agents: start at [AGENTS.md](AGENTS.md). A generated file map is in [INDEX.md](INDEX.md).

## Solutions

| Solution | Owner | Idea | Public LB | Status |
|---|---|---|---|---|
| [`solutions/fusion-v4n`](solutions/fusion-v4n/) | Hemre | Reproduces the public 0.399 notebook (two engines + PubChem channel + ICEBERG/GLACIER isomer re-scoring + RRF fusion) with every knob in one config, stage dumps and a tested fusion core | **0.431** | v13a: v4b (popularity prior 0.25 + gap-mode MetFrag re-score + library gate) + PubChem join, 100 candidates |
| `research/competition-analysis` (branch) | İlker | Built from scratch: leak-aware validation, library search, PubChem/COCONUT catalog, fragmentation ranking (Stage D) | 0.260 | active on its branch |

See [solutions/README.md](solutions/README.md) for how a solution is laid out.

## Where things are

| You want… | Go to |
|---|---|
| Task, metric, data columns, test novelty classes | [docs/competition/](docs/competition/) |
| What is allowed (NIST banned, offline, 9 h) and known traps | [docs/competition/rules-and-pitfalls.md](docs/competition/rules-and-pitfalls.md) |
| Leaderboard, public notebooks, forum insights | [docs/landscape/](docs/landscape/) |
| The plan and the next experiments | [docs/plan/roadmap.md](docs/plan/roadmap.md) |
| Every submission and its score | [docs/experiments.md](docs/experiments.md) |
| Data layout (not in git) | [data/README.md](data/README.md) |

## Quick start

```bash
# 1. Accept the competition rules on kaggle.com, then `kaggle auth login` (browser OAuth) or put an API token in ~/.kaggle/
# 2. Python env (Python 3.12/3.13; RDKit is pinned to 2026.3.3 = the metric's version)
python -m venv .venv && .venv/Scripts/python -m pip install -r solutions/fusion-v4n/requirements.txt
# 3. Data -> data/input/ (mirrors /kaggle/input)
.venv/Scripts/python solutions/fusion-v4n/tools/download_data.py
# 4. Build and push a notebook
.venv/Scripts/python solutions/fusion-v4n/tools/build_notebook.py
.venv/Scripts/kaggle kernels push -p solutions/fusion-v4n/kaggle
```

(`.venv/bin/...` on Linux/macOS.)

## Rules for this repo

- **Never commit competition data, model weights or run outputs.** `data/`, `runs/`, `*.parquet`, `*.pt`, … are git-ignored.
- One folder per approach under `solutions/`, each with `README.md`, `AGENTS.md`, `solution.json`.
- Log every leaderboard submission in [docs/experiments.md](docs/experiments.md).
- Regenerate the index after adding or removing files: `python tools/gen_index.py`.

Licence: MIT ([LICENSE](LICENSE)). Third-party code keeps its own licence, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
