<!-- index: public notebook lineage (v3 to v4n to the 0.399 fusion), what each adds, and the shared public datasets -->
# Public notebooks

Snapshot 2026-09-30 from Code → sort by score. Almost every top public notebook descends from one lineage.

## Main lineage (ahmedberatozer → forks)
| Notebook | LB | Adds |
|---|---|---|
| [casmi26-v3-inference](https://www.kaggle.com/code/ahmedberatozer/casmi26-v3-inference) | 0.358 | v1 engine (library + pool + FPNet fingerprint model + LightGBM ranker) + gated PubChem-only channel |
| [v4f](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4f-inference) | 0.362 | FPNet A engine bank + ICEBERG post-ranker isomer re-scoring + fe_v4 feature families |
| [v4g](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4g-inference) | 0.370 | |
| [v4h](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4h-inference) | 0.373 | GLACIER as third re-scoring term: z(ranker) + 0.5 z(ICE) + 0.5 z(GL) |
| v4i | – | PubChem pass-1 depth 1000 → 5000 |
| [v4l](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4l-inference) | 0.380 | ICE_LAM = GL_LAM = 1.0 |
| [v4m](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4m-inference) | 0.378 | FP bank `fpnet_full1` |
| [v4n](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4n-inference) | 0.384 | v4i + v4l + v4m |
| [**seyitkaangunes v4n + Engine Fusion + Union**](https://www.kaggle.com/code/seyitkaangunes/casmi26-v4n-engine-fusion-union-lb-0-399) | **0.399** | + prvsiyan's two-ranker engine (BIO + AFIX) as engine-2, RRF-fused (α=0.6, k=3), its candidates added to ICE/GL input. **Our `solutions/fusion-v4n` baseline** |
| [seyitkaangunes e5c union gl a06](https://www.kaggle.com/code/seyitkaangunes/casmi26-e5c-union-gl-a06) | 0.386 | earlier fusion variant |
| evgendvorkin "Enveda CASMI 2026", dmitriigluzdov "From Spectra to Structures", "InferenceSuperNext" | 0.386 | forks with the same 13 inputs |
| wangpenghua v21 / v22 / v24 / v26 | 0.379–0.384 | "pop" variants of v4i / v4m, one without GLACIER (0.379) |

## Other approaches
| Notebook | LB | Idea |
|---|---|---|
| [flexonafft Confidence-Gated Analog Generation](https://www.kaggle.com/code/flexonafft/casmi26-confidence-gated-analog-generation) | 0.378 | analog generation only when confidence is low (class-3 attempt) |
| nursrijan "Dual-Shield Two-Ranker" | 0.377 | two-ranker variant |
| [inversion CASMI denovo tutorial](https://www.kaggle.com/code/inversion/casmi-denovo-tutorial-notebook) | – | host-side de novo tutorial (most-voted notebook) |
| [truthseeking progress report](https://www.kaggle.com/code/truthseeking/casmi-2026-progress-report) | – | 377-page experiment compendium: library search fails once structures are held out, PubChem holds ~96% of answers but ~10k candidates per window |
| forum "[0.339 Top 1 Solution] 4-Channel Mass-Shifted Analog Propagation & Neural Bayes Reranking" | 0.339 (early) | origin of prvsiyan's analog-propagation engine |

## Shared public datasets (the reference notebook's 13 inputs)
prvsiyan: `casmi26-fp-models-v2`, `casmi26-ranker-features`, `coconut-casmi26-candidates`, `chebi-lipidmaps-casmi26`.
ahmedberatozer: `casmi26-v2-pool`, `casmi26-v3-models`, `casmi26-v4b-models`, `casmi26-fpnet-full1`,
`casmi26-iceberg`, `casmi26-glacier`, `casmi26-pubchem-tier`. megayak: `casmi26-simulated-ranker-rows`.
metric: `rdkit-2026-3-3-wheel`. Total ≈ 15.7 GB with the competition data.
