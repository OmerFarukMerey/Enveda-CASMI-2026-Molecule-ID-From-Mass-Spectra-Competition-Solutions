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
| [`solutions/fusion-v4n`](solutions/fusion-v4n/) | Hemre | The public 0.399 pipeline (two engines + PubChem channel + ICEBERG / GLACIER isomer re-scoring + rank fusion) rebuilt with every knob in one config, stage dumps, a tested fusion core and a local replay of the final stages; on top of it the PubChem join and a ranker fitted on molecules the fingerprint model has not seen | **0.442** | v33b: popularity prior + library gate (0.409) → PubChem join, 100 candidates (0.431) → forward-model weights 0.75 (0.437) → our ranker averaged in below the library gate (0.442). [How we got there](solutions/fusion-v4n/README.md#best-submission-public-lb-0442-v33b) |
| `research/competition-analysis` (branch) | İlker | Built from scratch: leak-aware validation, library search, PubChem/COCONUT catalog, fragmentation ranking (Stage D) | 0.260 | active on its branch |

See [solutions/README.md](solutions/README.md) for how a solution is laid out.

## Submissions and scores

Public leaderboard (MRR@25). Best in the scoring regime in force since 2026-10-07: **0.443** (e006r, a rerun of the best public notebook with a licence-clean input; our own pipeline: v59a 0.434). Before the change: 0.442 (v33b). Folders with the exact notebook, config and a detailed
`README.txt` are in [submissions/](submissions/); every row with its details is in [docs/experiments.md](docs/experiments.md).

| # | Date | Solution | Version | Change | Public LB | Folder |
|---|---|---|---|---|---|---|
| 1 | 09-30 | competition-analysis | Stage F v1 | library search + catalogue fallback | 0.193 | |
| 2 | 09-30 | competition-analysis | Stage F v2 | training + COCONUT pool only | 0.163 | |
| 3 | 09-30 | competition-analysis | Stage D | pool re-ranked by fragmentation + analog propagation | 0.260 | |
| 4 | 09-30 | fusion-v4n | v1 | reproduction of the public 0.399 notebook | 0.399 | |
| 5 | 09-30 | fusion-v4n | v2 | + popularity prior 0.15 + gap-mode MetFrag re-score | 0.404 | |
| 6 | 10-01 | fusion-v4n | v3b | v2 with the MetFrag term on all molecules | 0.390 | |
| 7 | 10-01 | fusion-v4n | v4a | v2 + forward models off for library hits | 0.404 | |
| 8 | 10-01 | fusion-v4n | v4b | v4a + popularity prior 0.25 | 0.409 | |
| 9 | 10-01 | fusion-v4n | v5a | replica of a public 0.415 notebook | 0.393 | |
| 10 | 10-01 | fusion-v4n | v5c | v4b + PubChem-channel popularity + GLACIER on [M+H]+ only | 0.405 | |
| 11 | 10-02 | fusion-v4n (teammate) | np1 | v4b + NP-tuned FPNet in engine 2 | 0.413 | |
| 12 | 10-02 | fusion-v4n | v7b | v4b + GLACIER on [M+H]+ only + engine-2 gate | 0.405 | |
| 13 | 10-02 | fusion-v4n (teammate) | v7a | v4b + isomer-ranker channel | 0.400 | |
| 14 | 10-02 | fusion-v4n | v10a | v4b + order-free ties in the ranker | 0.408 | |
| 15 | 10-02 | fusion-v4n | v11 | v4b + five small changes at once | 0.405 | |
| 16 | 10-03 | fusion-v4n | v12a | v4b + PubChem join, 50 candidates | 0.420 | [folder](submissions/fusion-v4n-v12a-0.420/) |
| 17 | 10-03 | fusion-v4n | v13a | v12a with 100 candidates in the join | 0.431 | [folder](submissions/fusion-v4n-v13a-0.431/) |
| 18 | 10-03 | fusion-v4n | v14a | v13a with 200 candidates | 0.406 | |
| 19 | 10-03 | fusion-v4n | v14c | v13a + GLACIER on [M+H]+ only | 0.431 | |
| 20 | 10-03 | fusion-v4n | v14b | v13a with the join gate at 0.8 | 0.416 | |
| 21 | 10-04 | fusion-v4n | v15b | v13a with a PubChem shortlist of 20,000 | 0.431 | |
| 22 | 10-04 | fusion-v4n | v15a | v13a + our own ranker blended in | 0.417 | |
| 23 | 10-04 | fusion-v4n | v17b | v13a + confidence gate for the join | 0.430 | |
| 24 | 10-04 | fusion-v4n | v20a | v13a + fine-tuned ICEBERG intensity model | 0.401 | |
| 25 | 10-04 | fusion-v4n | v19a | 200 candidates + confidence gate | 0.409 | |
| 26 | 10-05 | fusion-v4n | v22a | v13a, joined candidates selected by a fine-tuned FPNet | 0.423 | |
| 27 | 10-05 | fusion-v4n | v24a | v13a with ICEBERG / GLACIER weights 0.75 | 0.437 | [folder](submissions/fusion-v4n-v24a-0.437/) |
| 28 | 10-06 | fusion-v4n | v24b | v13a with ICEBERG / GLACIER weights 0.5 | 0.433 | |
| 29 | 10-06 | fusion-v4n | v24c | v13a with ICEBERG / GLACIER weights 0.6 | 0.430 | |
| 30 | 10-06 | fusion-v4n (teammate) | v24a-fmgl | v24a + fine-tuned GLACIER weights | 0.414 | |
| 31 | 10-06 | fusion-v4n | **v33b** | v24a + our ranker (fitted on unseen molecules) averaged with the public one below the library gate | **0.442** | [folder](submissions/fusion-v4n-v33b-0.442/) |
| 32 | 10-07 | fusion-v4n | v34a | v33b with the forward models off in the first re-rank of the base list | 0.431 | |
| 33 | 10-07 | fusion-v4n (teammate) | – | v24a with the engine-2 fusion weight 0.6 → 0.3 | 0.406 | |
| 34 | 10-07 | fusion-v4n | v33a | v33b with our ranker alone (no average) below the library gate | 0.434 | |
| 35 | 10-07 | fusion-v4n | v40a | v33b + wider protection of library hits (no engine-2 fusion at lib_max ≥ 0.97, no popularity / fragment term) | 0.442 | |
| 36 | 10-08 | fusion-v4n | v44a | v33b with the popularity prior at 0.35 | 0.432 | |
| 37 | 10-08 | fusion-v4n | v46c | v44a + popularity prior inside same-formula groups below the library gate | 0.426 | |
| 38 | 10-08 | fusion-v4n | v35a | v33b with the engine-2 rank fusion constant 3 → 1 | 0.437 | |
| 39 | 10-08 | fusion-v4n | v47a | v33b with the popularity prior at 0.15 | 0.426 | |
| 40 | 10-08 | fusion-v4n | v48a | v33b with our ranker refitted with inverse-fame weights | 0.436 | |
| 41 | 10-09 | fusion-v4n | v33b again | row 31 unchanged, scored after Kaggle's scoring change of 10-07 | 0.433 | new-regime baseline |
| 42 | 10-09 | fusion-v4n | v59a | v33b with the CC BY ChEBI / LIPID MAPS set (licence fix) | 0.434 | current reference |
| 43 | 10-09 | fusion-v4n | v60a | probe: first row only | 0.377 | |
| 44 | 10-09 | fusion-v4n | v60b | probe: first two rows only | 0.413 | |
| 45 | 10-09 | fusion-v4n | v54a | v59a with fusion constant 1 and fragment weight 0.25 | 0.431 | |
| 46 | 10-10 | fusion-v4n | v62a | v59a + generated analogs, 5 rows from position 6 | 0.433 | |
| 47 | 10-10 | fusion-v4n | v62b | v59a + generated analogs, 5 rows from position 11 | 0.434 | |
| 48 | 10-10 | fusion-v4n | v65c | v59a fused with the public e006 pipeline (weight 0.3) | 0.430 | |
| 49 | 10-10 | fusion-v4n | v66a | v59a + e006's strong-branch promotion | 0.434 | |
| 50 | 10-10 | fusion-v4n | v66c | v66a + popularity 0.10 on joined rows | 0.428 | |
| 51 | 10-11 | public rerun | **e006r** | public e006 notebook unchanged, CC BY ChEBI set | **0.443** | [folder](submissions/e006r-0.443/) |
| 52 | 10-11 | fusion-v4n | v67a | e006's chain fed with our base list (join + rank average) | 0.411 | |
| 53 | 10-11 | fusion-v4n | v67b | v67a + promoted row kept first | 0.410 | |
| 54 | 10-11 | fusion-v4n | v67d | e006's chain + our PubChem join only | 0.398 | |
| 55 | 10-11 | fusion-v4n | v67e | e006's chain + our rank average only | 0.411 | |

## Where things are

| You want… | Go to |
|---|---|
| Task, metric, data columns, test novelty classes | [docs/competition/](docs/competition/) |
| What is allowed (NIST banned, offline, 9 h) and known traps | [docs/competition/rules-and-pitfalls.md](docs/competition/rules-and-pitfalls.md) |
| Leaderboard, public notebooks, forum insights | [docs/landscape/](docs/landscape/) |
| The plan and the next experiments | [docs/plan/roadmap.md](docs/plan/roadmap.md) |
| Every submission and its score | [docs/experiments.md](docs/experiments.md), table above |
| The exact notebook, config and write-up of a submission | [submissions/](submissions/) |
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
