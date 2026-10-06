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
| 17 | 2026-10-03 | fusion-v4n | v13a-v12a-n100 (submission 56793694) | v12a with 100 instead of 50 PubChem candidates in the join (`PC_JOIN_N=100`). Predicted 0.423 (0.412–0.432) | 8 | **0.431** | **+0.011** | Hemre |
| 18 | 2026-10-03 | fusion-v4n | v14a-v13a-n200 | v13a with 200 instead of 100 PubChem candidates in the join (`PC_JOIN_N=200`). Predicted 0.436 (0.425–0.445) | 8 | 0.406 | −0.025 | Hemre |
| 19 | 2026-10-03 | fusion-v4n | v14c-v13a-glmh | v13a + GLACIER on [M+H]+ only. Predicted 0.432 (0.420–0.440) | 8 | 0.431 | 0.000 | Hemre |
| 20 | 2026-10-03 | fusion-v4n | v14b-v13a-lib08 | v13a with the join gate at `lib_max` < 0.8 instead of 0.7. Predicted 0.428 (0.415–0.440) | 8 | 0.416 | −0.015 | Hemre |
| 21 | 2026-10-04 | fusion-v4n | v15b-v13a-pcn1-20000 | v13a with the PubChem first-pass shortlist at 20,000 instead of 5,000 (`PC_N1=20000`). Predicted 0.433 (0.420–0.440) | 8 | 0.431 | 0.000 | Hemre |
| 22 | 2026-10-04 | fusion-v4n | v15a-v13a-blend | v13a + `RANKER=blend`, `RANKER_LIB_GATE=0.7` (our ranker places the joined rows). Predicted 0.433 (0.420–0.445) | 8 | 0.417 | −0.014 | Hemre |
| 23 | 2026-10-04 | fusion-v4n | v17b-v13a-conf07 | v13a + confidence gate: the join acts only when the unjoined list looks wrong (`PC_JOIN_CONF=0.7`). Predicted 0.433 (0.422–0.445) | 8 | 0.430 | −0.001 | Hemre |
| 24 | 2026-10-04 | fusion-v4n | v20a-v13a-iceft | v13a with the fine-tuned ICEBERG intensity model (`ICE_FT`, ahmedberatozer/casmi26-ice-ft) and `STRICT` (the run fails if any stage fell back). No local estimate possible (the model was fine-tuned on the training spectra). Guess 0.432 (0.415–0.445) | 9 | **0.401** | −0.030 | Hemre |
| 25 | 2026-10-04 | fusion-v4n | v19a-n200-conf05 | 200 PubChem candidates joined only when the unjoined list looks wrong (`PC_JOIN_N=200`, `PC_JOIN_CONF=0.5`). Predicted 0.428 (0.405–0.440) | 8 | **0.409** | −0.022 | Hemre |
| 26 | 2026-10-05 | fusion-v4n | v22a-v13a-hn-n100 | v13a with the 100 joined PubChem candidates selected by the fold-safe FPNet fine-tuned against PubChem-window negatives (`PC_MODELS=hn`, second channel pass; fusion list and thresholds keep public A+B). Local, v1 side before fusion: answer-removed +0.028 ± 0.009, c2 +0.001, c1 −0.002, joined row on top unchanged. Expected 0.431–0.437 | 17 | 0.423 | −0.008 | Hemre |
| 27 | 2026-10-05 | fusion-v4n | v24a-v13a-fwd075 | v13a with ICEBERG / GLACIER weights 0.75 instead of 1.0 (`ICE_LAM`, `GL_LAM`). Full local replay: c1 +0.005 ± 0.004, c2 +0.011 ± 0.004 (22 / 5), dA +0.001, dB +0.001, pool-answer lists topped by a joined row 25 → 22. Information probe, expected 0.428–0.437 | 17 | **0.437** | **+0.006** | Hemre |
| 28 | 2026-10-06 | fusion-v4n | v24b-v13a-fwd05 (submission 56864496) | v13a with ICEBERG / GLACIER weights 0.5 (v24a used 0.75). Full local replay against v13a: c1 +0.009 ± 0.006, c2 +0.026 ± 0.007, dA −0.004, dB −0.007, exposed lists 25 → 19. Expected 0.432–0.445 | 27 | 0.433 | −0.004 | Hemre |
| 29 | 2026-10-06 | fusion-v4n | v24c-v13a-fwd06 (submission 56864507) | v13a with ICEBERG / GLACIER weights 0.6; not replayed locally, between v24a and v24b. Expected 0.434–0.442 | 27 | 0.430 | −0.007 | Hemre |
| 30 | 2026-10-06 | fusion-v4n (teammate's variant, not in this tree) | casmi26-v24a-fmgl (submission 56876711; two earlier attempts ended in a Kaggle error) | v24a + GLACIER weights fine-tuned on timsTOF / NP / negative-mode spectra (`GL_CKPT`); one change | 27 | 0.414 | −0.023 | İlker |
| 31 | 2026-10-06 | fusion-v4n | v33b-v24a-fsranker-gate07-rankavg | v24a; below `lib_max` 0.7 the base list is the rank average of the public ranker and a LightGBM ranker fitted to the fold-safe FPNet's features on molecules that bank has not seen (`RANKER=fs_avg`, `RANKER_LIB_GATE=0.7`, dataset casmi26-ours-ranker-fs). Full replay against the leaked v24a: panel c1 −0.009 ± 0.004, c2 +0.009 ± 0.005, dA +0.030 ± 0.008, dB +0.023 ± 0.008; regime −0.001 / +0.007 / +0.009 / +0.009; exposure not raised. Expected 0.435–0.448 | 27 | **0.442** | **+0.005** | Hemre |
