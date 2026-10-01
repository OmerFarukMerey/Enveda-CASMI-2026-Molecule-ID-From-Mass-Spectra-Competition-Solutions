<!-- index: team-wide append-only log of every Kaggle submission and its public LB score -->
# Experiment log (team-wide, append-only)

One row per Kaggle submission. Details go in the solution's own `EXPERIMENTS.md`.
`Δ` is against the row named in "vs". LB noise band is about ±0.005–0.02.

| # | Date | Solution | Version / Kaggle version | Change | vs | Public LB | Δ | Who |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-30 | research/competition-analysis | Stage F v1 | library search + PubChem/COCONUT catalog fallback | – | 0.193 | – | İlker |
| 2 | 2026-09-30 | research/competition-analysis | Stage F v2 | training ∪ COCONUT pool only | 1 | 0.163 | −0.030 | İlker |
| 3 | 2026-09-30 | research/competition-analysis | Stage D | pool reranked by MetFrag-lite + 4×analog propagation + 0.25×first-stage prior | 1 | 0.260 | +0.067 | İlker |
| 4 | 2026-09-30 | fusion-v4n | ours-v1-baseline, Kaggle v3 (`huseyinemreaksoy/notebook095c91ef58`) | reproduction of public v4n + engine fusion (default CFG). Hidden rerun took ~3–4 h | ref 0.399 | **0.399** | 0.000 vs ref | Hemre |
| 5 | 2026-09-30 | fusion-v4n | v2-pop015-fraggap05, Kaggle v4 (submission 56717888) | + pool popularity prior μ=0.15 (bobthebot v10) + MetFrag-parsimony re-score in `gap` mode λ=0.5 (only molecules ICE/GLACIER cannot score) | 4 | **0.404** | +0.005 | Hemre |
