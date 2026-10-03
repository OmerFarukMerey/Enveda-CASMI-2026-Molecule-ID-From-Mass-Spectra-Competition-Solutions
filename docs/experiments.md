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
| 6 | 2026-10-01 | fusion-v4n | v3b-fragall025 (submission 56745660) | v2 with the frag term in `all` mode λ=0.25 instead of `gap` λ=0.5 | 5 | 0.390 | −0.014 | Hemre |
| 7 | 2026-10-01 | fusion-v4n | v4a-libgate (submission 56751038) | v2 + library gate: ICE/GL weight 0 when lib_max ≥ 0.9 | 5 | 0.404 | 0.000 | Hemre |
| 8 | 2026-10-01 | fusion-v4n | v4b-libgate-pop025 (submission 56751042) | v4a + popularity prior μ=0.25 | 7 | **0.409** | +0.005 | Hemre |
| 9 | 2026-10-01 | fusion-v4n | v5a-v29replica | reference + pool popularity 0.15 + PubChem-channel popularity 0.25/200 + GLACIER on [M+H]+ only (attempted replica of wangpenghua v29, reported 0.415; its popularity datasets are private, ours came from dmitriigluzdov) | ref 0.399 | 0.393 | −0.006 | Hemre |
| 10 | 2026-10-01 | fusion-v4n | v5c-v4b-pcpop-glmh | v4b (0.409) + PubChem-channel popularity + GLACIER [M+H]+ only | 8 | 0.405 | −0.004 | Hemre |
| 11 | 2026-10-02 | fusion-v4n (teammate's variant, not in this tree) | np1 (submission 56767215) | v4b + NP-tuned FPNet (goooman npx4 soup) averaged into engine-2 | 8 | 0.413 | +0.004 | teammate |
| 12 | 2026-10-02 | fusion-v4n | v7b-v4b-glmh-eng2gate05 (submission 56771836) | v4b + GLACIER on [M+H]+ only + engine-2 fused only when lib_max ≥ 0.5. Predicted 0.421 (0.400–0.435) | 8 | 0.405 | −0.004 | Hemre |
| 13 | 2026-10-02 | fusion-v4n (teammate's variant, not in this tree) | "v7a" (submission 56775374) | v4b + isomer-ranker channel (similar negatives + NP oversampling), CL_LAM 0.25, mode all | 8 | 0.400 | −0.009 | teammate |
| 14 | 2026-10-02 | fusion-v4n | v10a-v4b-tiefree-gate07 (submission 56777636) | v4b + public ranker with order-free ties below lib_max 0.7 (`RANKER=public_free`, `RANKER_LIB_GATE=0.7`). Predicted 0.411 (0.404–0.419) | 8 | 0.408 | −0.001 | Hemre |
| 15 | 2026-10-02 | fusion-v4n | v11-v4b-bundle | v4b + `RANKER=blend` (gate 0.7) + `SLOTS_GENTLE=[3,6,9,12,15]` + `PC_FWD_T1=0.0` + `GL_MH_ONLY` + `ICE_SURROGATE=gap`. Predicted 0.415 (0.403–0.428) | 8 | 0.405 | −0.004 | Hemre |
| 16 | 2026-10-03 | fusion-v4n | v12a-v4b-pcjoin50-lib07 (submission 56784505) | v4b + PubChem join: best 50 PubChem-only candidates ranked by the v1 engine when lib_max < 0.7 (`PC_JOIN_N=50`, `PC_JOIN_MAX_LIB=0.7`). Predicted 0.415 (0.403–0.44) | 8 | **0.420** | **+0.011** | Hemre |
| 17 | 2026-10-03 | fusion-v4n | v13a-v12a-n100 (submission 56793694) | v12a with 100 instead of 50 PubChem candidates in the join (`PC_JOIN_N=100`). Predicted 0.423 (0.412–0.432) | 8 | pending | – | Hemre |
