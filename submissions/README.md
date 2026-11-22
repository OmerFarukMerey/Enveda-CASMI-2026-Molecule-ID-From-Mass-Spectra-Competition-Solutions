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
| [fpn-0.496](fpn-0.496/) | **0.496** | 2026-11-21 | Ömer's fp5 + p12f + n5f merged on tp5 (FR after P5, 12-parent P5, NetGen), NIST-matched library rows dropped |
| [gfp5-0.500](gfp5-0.500/) | **0.500** | 2026-11-22 | goodpjw2008's 0.498 stack (APRIOR + UNIQ) with FR 3-25 after the P5 tail; NIST-matched library rows kept (compliant gfp5n queued); team best |
| [tp5-0.495](tp5-0.495/) | **0.495** | 2026-11-20 | public notebook taeyangg4 "0.495 Network + FR + P5 Tail" copied, 117 NIST-matched library rows dropped; team best |
| [n0-0.492](n0-0.492/) | **0.492** | 2026-11-16 | public notebook mathleteutd "0.492 Test-Set Network Rerank" copied (N-1c at LAM 0.35); team best |
| [ax13n-0.488](ax13n-0.488/) | **0.488** | 2026-11-13 | public notebook analyticaobscura/casmi-2026 (alexchilton v13 + N-1c in-test network rerank) copied unchanged; team best |
| [e006r-0.443](e006r-0.443/) | **0.443** (new scoring regime) | 2026-10-11 | public notebook canqiang e006 rerun unchanged with the CC BY ChEBI / LIPID MAPS set; baseline from here on |
| [fusion-v4n-v33b-0.442](fusion-v4n-v33b-0.442/) | **0.442** (old regime; 0.433 in the new one) | 2026-10-06 | v24a + a ranker fitted on unseen molecules, rank-averaged with the public one below the library gate |
| [fusion-v4n-v24a-0.437](fusion-v4n-v24a-0.437/) | 0.437 | 2026-10-05 | v13a with ICEBERG / GLACIER weights 0.75 instead of 1.0 |
| [fusion-v4n-v13a-0.431](fusion-v4n-v13a-0.431/) | 0.431 | 2026-10-03 | v12a with 100 PubChem candidates in the join |
| [fusion-v4n-v12a-0.420](fusion-v4n-v12a-0.420/) | 0.420 | 2026-10-03 | v4b + PubChem join (50 candidates, `lib_max` < 0.7) |

Every submission, including the ones without a folder, is a row in [../docs/experiments.md](../docs/experiments.md) and in the
table of the root [README](../README.md). To add one: create the folder, copy the notebook and the config, write the
`README.txt`, add a row here and in the root README, run `python tools/gen_index.py`.
