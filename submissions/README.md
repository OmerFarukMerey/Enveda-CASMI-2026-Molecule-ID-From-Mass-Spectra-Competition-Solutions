<!-- index: one folder per notable Kaggle submission: exact notebook, config and a detailed README.txt -->
# Submissions

One folder per submission worth keeping, named `<solution>-<version>-<public LB>`. Each holds:

| File | What |
|---|---|
| `README.txt` | score, date, Kaggle ids, what changed, config, inputs, how to reproduce, the local evidence, what we learned |
| `notebook.ipynb` | the exact notebook that was submitted |
| `config.json` | the config overrides of that version |

| Folder | Public LB | Date | Change |
|---|---|---|---|
| [fusion-v4n-v33b-0.442](fusion-v4n-v33b-0.442/) | **0.442** | 2026-10-06 | v24a + a ranker fitted on unseen molecules, rank-averaged with the public one below the library gate |
| [fusion-v4n-v24a-0.437](fusion-v4n-v24a-0.437/) | 0.437 | 2026-10-05 | v13a with ICEBERG / GLACIER weights 0.75 instead of 1.0 |
| [fusion-v4n-v13a-0.431](fusion-v4n-v13a-0.431/) | 0.431 | 2026-10-03 | v12a with 100 PubChem candidates in the join |
| [fusion-v4n-v12a-0.420](fusion-v4n-v12a-0.420/) | 0.420 | 2026-10-03 | v4b + PubChem join (50 candidates, `lib_max` < 0.7) |

Every submission, including the ones without a folder, is a row in [../docs/experiments.md](../docs/experiments.md) and in the
table of the root [README](../README.md). To add one: create the folder, copy the notebook and the config, write the
`README.txt`, add a row here and in the root README, run `python tools/gen_index.py`.
