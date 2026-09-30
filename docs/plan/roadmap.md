<!-- index: strategy: what runs on Kaggle vs locally, phases P0-P4, open questions -->
# Roadmap

Written 2026-09-30. Update when a phase finishes or a decision changes.

## Where we stand
- `solutions/fusion-v4n` = a restructured copy of the best public notebook (0.399). Not yet run on Kaggle.
- İlker's `research/competition-analysis` branch = a from-scratch pipeline with leak-aware validation (LB 0.193).
- The public top is 0.464; the 0.40–0.46 band is private work.

## Decision: where to run what
| Work | Where | Why |
|---|---|---|
| Full pipeline (all 9 cells) | **Kaggle** | It is the only place the hidden test exists. The pipeline hardcodes `/kaggle/...` and Linux `fork`, and the ICE/GLACIER wheels are `cp312 manylinux`, so it does not run on native Windows. The commit run is ~20 min on 12 molecules. |
| Fusion and gating knobs (`fusion_core`) | Local (`tools/refuse.py`) + Kaggle LB | Re-fusing stage dumps takes seconds. But the only labels available locally are the visible placeholder test (optimistic, not tunable), so the LB judges. |
| New components (re-scorers, ranker retrain) | **Local GPU** (RTX 4070 Ti SUPER 16 GB ≥ T4) | They need a leak-free panel. Untrained scorers (e.g. MetFrag-lite) can be evaluated on enveda-np-examples isomer panels. |
| Full pipeline locally | Optional, WSL2 Ubuntu | Needed only to build large holdout caches. Put the WSL disk on D: (C: has ~20 GB free). |

**Validation caveat:** the reference pipeline's artefacts (candidate pool, FPNets, rankers) were trained on all of `train.parquet`,
so any holdout drawn from train is leaked for the pipeline as a whole. Pipeline-level changes are judged on the LB,
component changes on leak-free panels.

## Phases
### P0: prove the trigger (today)
1. Accept the rules, `kaggle auth login`, download data (`tools/download_data.py`).
2. `build_notebook.py` → `kaggle kernels push` → the commit run passes (12 molecules).
3. Submit → expect ≈0.399 ± 0.01. Log it in [../experiments.md](../experiments.md).

### P1: cheap LB probes (one knob per submission, default config otherwise)
Noise band ±0.005–0.02, so only chase clear gains and re-test before stacking.
- `FUSE_ALPHA` 0.6 → 0.8, `FUSE_KRR` 3 → 1 (engine-2 weight in the RRF)
- `ICE_UNION_K` 40 → 60, `TOPN` 60 → 80 (more candidates re-scored; watch the 9 h budget)
- `ICE_LAM` / `GL_LAM` rebalancing (e.g. 1.0 / 0.5)
- `LIB_TAU` 0.9 → 0.85, `REL_TH` 600 → 400 (PubChem gate)

### P2: isomer separation (the main lever per the forum)
- Add a **MetFrag-lite parsimony re-scorer** as a third or fourth term in `rerank_multi`
  (2-bond ×0.6, linear intensity, 0.005 Da, H shifts −2..+3). Develop it on a same-formula isomer panel from
  enveda-np-examples. It is untrained, so the panel is leak-free.
- Joint with İlker's Stage D (fragmentation-based ranking): one implementation, shared panel.

### P3: ranker
Retrain the LightGBM ranker with the new re-scoring features on the public simulated ranker rows
(`megayak/casmi26-simulated-ranker-rows`, `prvsiyan/casmi26-ranker-features`). Average over several seeds (±0.007 seed noise).

### P4: class 3 (novel structures)
Confidence-gated analog generation (see the flexonafft notebook). Validate only on genuinely absent molecules; deletion
simulations overstate reach about 3×.

## Open questions
- Is the GLACIER / ICEBERG MassSpecGym provenance fully prize-eligible? Watch the forum thread before the final pick.
- The daily submission limit and how many we reserve for P1 vs P2.
