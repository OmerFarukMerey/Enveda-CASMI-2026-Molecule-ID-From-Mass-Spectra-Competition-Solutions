<!-- index: FRR, a learned fragment-explanation isomer reranker (own code), its offline benchmark and the Kaggle overlay cell -->
# FRR: learned fragment-explanation isomer reranker

Overlay for the public MIDX notebook (baidalinadilzhan, `casmi26-midx-exact-dedup-fill-0-479`, LB 0.479). Experiments
rows 120-121 in `docs/experiments.md`.

| File | What it does |
|---|---|
| `fragfeat.py` | v2 features: bond-cut enumeration (bridges, bridge pairs, ring-system cut pairs; own code), peak explanation at several tolerances, scalings and cut depths, chance level from peaks shifted by +-5 / +-9 Da, `model_matrix` (raw + within-group z and rank) |
| `fragfeat_v1.py` | v1 features (the `frrtop` / `frrall` tickets): v2 without `top10`, `hi2`, `mzw2`, `depth` |
| `build_sets.py` | ranking sets from train.parquet: an enveda-180 molecule (up to 6 spectra) against its 10 most Tanimoto-similar and 5 random same-formula train isomers; Murcko-scaffold folds |
| `merge_sets.py`, `eval_sets.py` | merge set tables; 5-fold scaffold CV of a LightGBM lambdarank model vs single scores, `save` fits the final model |
| `frr_cell.py`, `build_frr.py` | the Kaggle overlay cell (placeholders filled by the builder) and the builder that inserts it after MIDX's final layer |
| `harness.py` | runs a built cell locally on fake lists made from held-out sets |

Offline (71,891 sets, scaffold-held-out top-1 among up to 16 isomers): untrained fragment-check score 0.663, v1 0.739,
v2 0.751. The model files live in the private dataset `omerfarukmerey/casmi26-om-frr` (not in git). Paths in the
scripts point to Ömer's machine (train.parquet in `data/input/...`, the molecule cache of the neural trainer).
