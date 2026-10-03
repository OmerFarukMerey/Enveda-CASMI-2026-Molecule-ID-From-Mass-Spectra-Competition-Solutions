<!-- index: strategy: what runs on Kaggle vs locally, phases P0-P4, open questions -->
# Roadmap

Written 2026-09-30, target section added 2026-10-02. Update when a phase finishes or a decision changes.

## Where we stand (2026-10-02)
- `solutions/fusion-v4n`: best own LB **0.409** (v4b); a teammate's variant with an NP-tuned FPNet in engine-2 scored
  **0.413**. v7b (v4b + GLACIER on [M+H]+ only + engine-2 gate) is submitted and pending.
- İlker's `research/competition-analysis` branch = a from-scratch pipeline with leak-aware validation (LB 0.260).
- The public top is 0.464, then 0.435 and 0.425. Nobody is at 0.50.

## Target: 0.50 and above (plan of 2026-10-02)

The team wants 0.50 to 0.60. This section says where those points would have to come from, what each step is expected
to give, and how sure we are. Read the track record first: of our last five score predictions three missed on the
optimistic side by 0.015 to 0.02 (v3b, v5a, v5c), and the public LB has about 132 molecules, so one submission carries
a standard error near 0.04 and differences under 0.01 are not resolvable. The ranges below are wide on purpose.

### Where the score is today
Budget from `solutions/fusion-v4n/reviews/directions-2026-10-02.md` section 2.7 (an inference, not a measurement):

| Group of hidden molecules | Share | MRR inside the group | Contribution | Ceiling |
|---|---|---|---|---|
| Reference spectrum in a public library (class 1) | 16–33% | ~0.92 | 0.15–0.30 | ~0.98 |
| Structure in our pool, no reference, truth ranked | 21–46% | 0.50–0.58 | 0.10–0.26 | ~0.95 (perfect isomer pick) |
| Truth never ranked (PubChem-only, novel, outside top-60) | 38–46% | ~0 | ~0 | unknown |

Consequences: class 1 is nearly exhausted. Each +0.10 MRR inside the second group is worth +0.02 to +0.05 on the LB.
Each 10% of the third group brought to rank 1 is worth about +0.04. **0.50 needs both**: a clearly better isomer order
and a real share of the unranked molecules recovered. **0.60 needs about half of the unranked group answered at rank
1 or 2**; we have no component today that does this and no public evidence that anyone has.

### Steps and projected scores
Gain = expected public-LB change against the previous row. "Cumulative central" is the score if this step and every
step above it land at their central figure; it is not the score of that step's notebook alone. "Evidence" says what
the number rests on.

| # | Step | Gain (low / central / high) | Cumulative central | Evidence | Effort | Tested by |
|---|---|---|---|---|---|---|
| 0 | v4b | | 0.409 | LB | done | |
| 1 | GLACIER on [M+H]+ only (`GL_MH_ONLY`) | 0 / +0.007 / +0.010 | 0.416 | another team's LB (0.401 → 0.408–0.411) | ready (v6a) | 1 submission |
| 2 | ~~Engine-2 only where the library knows the compound (`FUSE_MIN_LIB` 0.5)~~ | **failed: v7b 0.405** (v4b 0.409; with step 1 worth +0.004 the gate itself is about −0.008) | – | local +0.13 plain and +0.07 to +0.09 with an honest prior did not transfer. Every local set has the answer in the main pool, so a channel whose value is finding answers outside it looks useless locally | dropped | LB |
| 3 | NP-tuned FPNet averaged into engine-2 | 0 / +0.004 / +0.008 | 0.425 | teammate's LB (0.409 → 0.413) | 1 day (take their weights) | 1 submission |
| 4 | Soft library gate (`GATE_TAU` 0.8) | −0.003 / +0.003 / +0.008 | 0.428 | class-1-like set +0.024, 14 wins 0 losses | rebuild on the current best (the existing v4d notebook sits on POP_MU 0.15 with half weights, so it is not this step) | 1 submission |
| 5 | Confidence-weighted isomer ranking (use ICEBERG/GLACIER only when they agree or the margin is large) | +0.002 / +0.005 / +0.010 | 0.433 | measured (`isomer-2026-10-02.md`): with an honest prior every recombination of the cached scores (gates, rank fusion, robust z, fitted or size-dependent weights, per-spectrum aggregation) moves the in-group rank by less than ±0.01; `z(ranker)+z(ICE)+z(GL)` is a flat optimum; the gains seen on the plain caches were the ranker leak | prototype exists (`prototypes/isomer_rank.py`), not better than existing knobs | not worth a submission on its own |
| 6 | Forward-model evidence for other adducts and negative mode | +0.003 / +0.010 / +0.025 | 0.443 | measured (`coverage-2026-10-02.md`): [M+NH4]+ and water-loss spectra score like [M+H]+ against the [M+H]+ prediction; [M+K]+ and negative mode are at chance and need a model of their own | surrogate done (`ICE_SURROGATE`), negative model 1–2 weeks | in-group test on np-examples, then LB |
| 7 | PubChem-only answers: slots one place earlier, early slots on forward-model evidence | 0 / +0.004 / +0.009 | 0.447 | measured (`pubchem-2026-10-02.md`): +0.054 per PubChem-only molecule, no cost in five regimes; the PubChem list alone scored 0.037 on the LB, so **a perfect merge is worth at most about +0.03** | done (`SLOTS_GENTLE`, `PC_FWD_T1`) | LB, together with another change |
| 8 | Class 3: analog-based generation | 0 / +0.004 / +0.015 | 0.451 | measured (`coverage-2026-10-02.md`): today's generator regenerates 16.7% of hidden structures, the prototype 39.3% (about 6% → 13% after the 3× overstatement) | 2–3 days | `drop_pid` run |
| 9 | Own leak-free ranker and fingerprint model (replace the public v1 models) | −0.005 / +0.002 / +0.010 | 0.461 | measured (`ranker-2026-10-02.md`): the public ranker carries a tie-order code (training structures first) that points the wrong way for hidden class-2 answers; order-free ties and our ranker gain +0.03 to +0.07 before the fusion and about +0.01 after it. NP-weighted FPNet fine-tune: no effect (`train/README.md`) | flags done (`RANKER`, `RANKER_LIB_GATE`), presets v10a / v10b | 1–2 submissions |

Totals if every step lands at its figure: low 0.409 (nothing helps; a step that loses is reverted, so the best
submission stays), **central 0.461**, high 0.544 (revised 2026-10-02 after the coverage, PubChem and isomer studies;
before them 0.476 and 0.595). The high case needs every step at its optimistic end, which has not happened once so far. A fair
statement: **0.44–0.46 is likely within two to three weeks, 0.50 is possible if steps 5 to 8 all work, 0.60 is not
supported by anything we have measured.**

### Where the score is lost (2026-10-02, evening; replaces the estimates above where they differ)
Per-molecule score of v4b by where the answer is (`mixed-regime-2026-10-02.md`; local, optimistic):

| Answer is | v4b MRR | What decides it |
|---|---|---|
| in the pool, with reference spectra | 0.83 to 0.85 | library search; nearly exhausted |
| in the pool (and in engine-2's candidates), no spectra | 0.74 | ranker and forward models; our ranker adds +0.03 here |
| only among engine-2's candidates | 0.28 to 0.40 | the fusion rule: engine-2 has it first, the fusion puts it at rank 5 to 7 |
| in neither candidate set (PubChem-only or novel) | 0.09 to 0.13 | PubChem slots and the generator |

With a leaderboard score of 0.41 this means about half of the hidden molecules are of the last kind. Work that moves
the score has to (1) rank PubChem candidates with the full engine instead of fixed slots (`pcjoin` study, running),
(2) remove the models' preference for structures they were trained on (order-free ties: done as `RANKER`; simulated
spectra for the whole pool: second experiment running), (3) reach novel structures (generator upgrade). Fusion
weights and gates are not on this list.

### v7b result (2026-10-02, 0.405): what it means for the plan
- Fusion knobs are exhausted. Eight submissions of this pipeline sit between 0.390 and 0.409, and the two largest
  local effects we ever measured (the engine-2 gate here, frag for all molecules earlier) both lost on the LB.
- The honest-prior regime fails as well, not only the plain caches. No local number is to be read as an LB
  prediction; this includes the cross-validation numbers of our own ranker.
- Presets v9a / v9b were built on v7b and are withdrawn. The small measured changes (soft library gate, adduct
  surrogate, PubChem slots) stay as flags and ride along with the next real change on the v4b base.
- The next submissions carry a new model (own ranker, NP-tuned fingerprint model), one at a time.

### What the first three studies changed (2026-10-02)
- The molecules that score about zero are **not** mainly PubChem-only answers waiting for a better merge: those are
  about 7 to 14% of the public molecules, and the merge can recover at most +0.03. The rest are answers outside
  PubChem, or answers that the fingerprint model does not rank. For PubChem-only answers the lever is the ranking
  inside the mass window (the answer is first for 19 to 41%), not the merge.
- Isomer order cannot be improved by recombining the evidence we have. Of the in-group errors on natural products,
  46 to 48% are molecules where ICEBERG and GLACIER disagree, 26 to 33% have no forward score and 21 to 26% are
  cases where both agree on a wrong isomer. Only new evidence (a better or additional forward model, more adducts,
  a better first-stage model) reaches them.
- Steps 5 to 8 together are worth about +0.02 at their central figures. They do not carry the score to 0.50.
- What is left as a large lever: new evidence, that is the quality of the first-stage models (fingerprint model and
  ranker, step 9) and of the forward models. The first is being studied (`ranker-2026-10-02.md`), and a survey of
  public work since 30 September (`docs/landscape/update-2026-10-02.md`) checks whether another spectrum encoder
  or candidate database explains the 0.42 to 0.46 scores.

### Order of work
1. Today: read the v7b result. If it is ≥ 0.415 the engine-2 gate is kept and v6a is not needed; if it is < 0.409
   submit v6a (step 1 alone) to separate the two changes.
2. This week: steps 3–5 (cheap, re-fusion only), each as one change against the best submission.
3. In parallel (research agents, reports under `solutions/fusion-v4n/reviews/`): `isomer-2026-10-02.md` (step 5),
   `pubchem-2026-10-02.md` (step 7), `coverage-2026-10-02.md` (steps 6 and 8).
4. Step 9 started 2026-10-02 as a feasibility study (`ranker-2026-10-02.md`): run the v1 engine locally with the
   fold-safe fingerprint bank, measure how much leak is left, build leak-free ranker rows, train our own ranker, and
   measure what happens when the truth is removed from the pool. Its first product is a validation set that orders
   configurations like the LB does, which we do not have (`tools/calibrate.py`: Spearman −0.13 and −0.95 on the two
   current sets). The ranker goes into a submission only if it wins where the comparison with the public one is fair.

Rules for every step: one change per submission against the current best, the prediction written into
`docs/experiments.md` before the score arrives, and no claim of a replica without the same data.

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
### P0: prove the trigger: DONE 2026-09-30, LB 0.399 (identical to the reference)
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
