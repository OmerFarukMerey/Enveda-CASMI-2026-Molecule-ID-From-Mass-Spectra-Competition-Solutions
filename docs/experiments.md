<!-- index: team-wide append-only log of every Kaggle submission and its public LB score -->
# Experiment log (team-wide, append-only)

One row per Kaggle submission. Details go in the solution's own `EXPERIMENTS.md`.
`Δ` is against the row named in "vs". LB noise band is about ±0.005–0.02.

| # | Date | Solution | Version / Kaggle version | Change | vs | Public LB | Δ | Who |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-30 | research/competition-analysis | Stage F v1 | library search + PubChem/COCONUT catalog fallback | – | 0.193 | – | İlker |
| 2 | 2026-09-30 | research/competition-analysis | Stage F v2 | training ∪ COCONUT pool only | 1 | 0.163 | −0.030 | İlker |
| 3 | pending | fusion-v4n | ours-v1-baseline | reproduction of public v4n + engine fusion (default CFG) | ref 0.399 | – | – | Hemre |
