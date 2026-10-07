<!-- index: fusion-v4n: pipeline stages, what is the same as and different from the 0.399 reference, how to run -->
# fusion-v4n: public 0.399 pipeline, restructured for iteration

Owner: Hemre · Status: **public LB 0.442 (v33b, 2026-10-06)**; the default config reproduces the reference (0.399) · Baseline:
[CASMI26 | v4n + Engine Fusion + Union [LB 0.399]](https://www.kaggle.com/code/seyitkaangunes/casmi26-v4n-engine-fusion-union-lb-0-399)

The default `CFG` reproduces the reference step for step. Every improvement goes in as a config knob or a flagged stage, so
each experiment is one diff against a known-good baseline.

## Pipeline

| Cell | Stage | Dump (`/kaggle/working/stage_cache/`) |
|---|---|---|
| `nb/00_config.py` | all knobs (`CFG`) | `cfg.json` |
| `nb/01_setup.py` | offline RDKit 2026.03.3 wheel, input discovery, MANIFEST sha check, write `nb/embed/**` to `/kaggle/working/ours/` | |
| `nb/02_smoke.py` | commit run → `SMOKE_N`=12 molecules (unless `FULL_ON_COMMIT`); scoring rerun → all | |
| `nb/03_engine2.py` | **engine-2**: prvsiyan two-ranker (BIO + AFIX), subprocess → top-40 lists | `eng.json` |
| `nb/04_pubchem.py` | **PubChem-only channel**: FPNet A+B logits, ±10 ppm window, keys not in the pool, top 25 | `pc.json` |
| `nb/05_v1_engine.py` | **v1 engine**: library + candidate pool + FPNet `fpnet_full1` + fe_v4 families + LightGBM ranker; installs the PubChem join and, with `RANKER='fs*'`, loads the fold-safe FPNet and our ranker | |
| `nb/06_base_lists.py` | per molecule top-`TOPN`=60 (key-deduplicated), `lib_max`, scores, formulas. With `PC_JOIN_N`: the best PubChem-only candidates are ranked together with the pool candidates below the join gate. With `RANKER='fs*'`: a second engine pass below `RANKER_LIB_GATE` (`nb/embed/fs_pass.py`) | `base.json`, `pc_deep.json` |
| `nb/07_rescore.py` | **ICEBERG + GLACIER** forward-model scores on same-formula groups (base ∪ engine-2 top-40 ∪ gated PubChem) | `ice.json`, `gl.json` |
| `nb/08_submit.py` | `fusion_core`: (A) z(ranker)+λ·z(ICE)+λ·z(GL) re-rank → (B) gated PubChem merge → (C) RRF with engine-2 + re-rank | `meta.json`, `submission.csv` |

## Same as the reference vs different

**Same (behaviour):** all 13 input datasets and model artefacts; every algorithm and default value (`TOPN 60, LIB_TAU 0.9,
REL_TH 600, slots [2..10]/[4..20], ICE_LAM = GL_LAM = 1.0, ICE_BUDGET 5400, GL_BUDGET 4000, PC_N1 5000, α 0.6, k 3, top-40
engine-2`); the stage order; smoke / rerun detection (same test signature); the MANIFEST sha check; engine code `pv.py`,
`pv_fp.py`, `casmi_engine.py` verbatim; PubChem runner verbatim.

**Different (structure, behaviour-preserving by default):**
| Change | Why |
|---|---|
| One `CFG` dict; engine-2 knobs via `$CASMI_ENG_CFG`, `PC_N1` via `$CASMI_PC_N1` | experiments = config diffs |
| Final stage moved into `nb/embed/fusion_core.py` (pure function) | same code on Kaggle and locally; **tested identical** to the reference (`tests/test_fusion_equivalence.py`, 8 random cases, all branches) |
| Stage dumps to `stage_cache/*.json` | re-fuse locally in seconds (`tools/refuse.py`) |
| Flags `USE_ENG/USE_PC/USE_ICE/USE_GL`, `ICE_UNION(_K)`, `FP_BANK`, `FULL_ON_COMMIT` | ablations without code edits |
| Notebook generated from `nb/*.py` by `tools/build_notebook.py`, pushed with the Kaggle CLI | reviewable diffs, no copy-paste in the Kaggle UI |
| Embedded sources written to `/kaggle/working/ours/{eng,pc}/` | one place instead of scattered writes |
| `assert rdkit == 2026.03.3` in setup (reference only prints) | fail fast instead of silently mismatched keys |
| `run_manifest.json` records version, CFG and stage stats | trace every LB row back to its config |

**Verified on Kaggle:** commit runs pass every stage (12-molecule smoke: 0.958 local, all exact library hits), and the hidden-test submission scored **0.399**, the same as the reference.

## Best submission: public LB 0.442 (v33b)

Each row is one submission that became the new best; the config of a row contains everything above it.

| Version | What changed against the row above | Config | Public LB | Folder / notebook |
|---|---|---|---|---|
| v1 | reference defaults | [v1-0.399.json](configs/v1-0.399.json) | 0.399 | |
| v2 | popularity prior μ=0.15 on pool candidates + MetFrag-parsimony re-score in `gap` mode (λ=0.5, only molecules ICEBERG / GLACIER cannot score) | [v2-0.404.json](configs/v2-0.404.json) | 0.404 | [kaggle/submitted/v2-0.404.ipynb](kaggle/submitted/v2-0.404.ipynb) |
| v4b | ICEBERG / GLACIER off for library hits + popularity μ=0.25 | [v4b-libgate-pop025.json](configs/v4b-libgate-pop025.json) | 0.409 | public: [Kaggle notebook](https://www.kaggle.com/code/huseyinemreaksoy/casmi26-sub-v4b-0-409-on-the-public) |
| v12a | **PubChem join**: the best 50 PubChem-only candidates enter the v1 engine and are placed by its ranker when `lib_max` < 0.7 (`nb/embed/pc_join.py`) | [v12a-v4b-pcjoin50-lib07.json](configs/v12a-v4b-pcjoin50-lib07.json) | 0.420 | [submissions/fusion-v4n-v12a-0.420](../../submissions/fusion-v4n-v12a-0.420/), public: [Kaggle notebook](https://www.kaggle.com/code/huseyinemreaksoy/casmi26-v4n-fusion-pubchem-on-public-0-421) |
| v13a | 100 instead of 50 PubChem candidates in the join | [v13a-v12a-n100.json](configs/v13a-v12a-n100.json) | 0.431 | [submissions/fusion-v4n-v13a-0.431](../../submissions/fusion-v4n-v13a-0.431/) |
| v24a | forward-model weights 0.75 instead of 1.0 (`ICE_LAM`, `GL_LAM`) | [v24a-v13a-fwd075.json](configs/v24a-v13a-fwd075.json) | 0.437 | [submissions/fusion-v4n-v24a-0.437](../../submissions/fusion-v4n-v24a-0.437/) |
| **v33b** | below `lib_max` 0.7 the base list is the rank average of the public ranker and **a ranker fitted on molecules the fingerprint model has not seen** (`RANKER='fs_avg'`, `nb/embed/fs_pass.py`) | [v33b-v24a-fsranker-gate07-rankavg.json](configs/v33b-v24a-fsranker-gate07-rankavg.json) | **0.442** | [submissions/fusion-v4n-v33b-0.442](../../submissions/fusion-v4n-v33b-0.442/) |

All 31 submissions are in [../../docs/experiments.md](../../docs/experiments.md) (one row each) and in
[EXPERIMENTS.md](EXPERIMENTS.md) (what was measured before and what the result meant).

### What moved the score, and why

1. **The PubChem join (0.409 → 0.431).** An answer that is not in the main pool could only enter the list through
   fixed slots, judged by one fingerprint score. Joined candidates get every feature and the same ranker as pool
   candidates. A gate keeps molecules with library evidence untouched. Study: [reviews/pcjoin-2026-10-02.md](reviews/pcjoin-2026-10-02.md).
2. **A ranker fitted in the regime it is used in (0.437 → 0.442).** The public ranker was fitted to the public
   fingerprint models' features on molecules those models had memorised; on the hidden test every molecule is new to
   them. Our ranker is fitted on the fold-safe FPNet's features for molecules that model has not seen. Studies:
   [reviews/rankbank-2026-10-06.md](reviews/rankbank-2026-10-06.md), [reviews/fsranker-2026-10-06.md](reviews/fsranker-2026-10-06.md).
3. **Smaller steps:** the popularity prior and the library gate (0.399 → 0.409), lower forward-model weights
   (0.431 → 0.437; a line search over 1.0 / 0.75 / 0.6 / 0.5 gave 0.431 / 0.437 / 0.430 / 0.433, so this step is at the
   edge of the noise of one submission).

### What did not work (each measured, most also submitted)

| Idea | Result | Where |
|---|---|---|
| More PubChem candidates in the join (200), a looser gate (0.8) | 0.406, 0.416 | [reviews/next-levers-2026-10-03.md](reviews/next-levers-2026-10-03.md) |
| Our first own ranker, fitted on the public models' features; blended in with the join | 0.405–0.408; 0.417 | [reviews/ranker-2026-10-02.md](reviews/ranker-2026-10-02.md) |
| Fine-tuned forward models (ICEBERG; GLACIER by a teammate) | 0.401; 0.414 | [EXPERIMENTS.md](EXPERIMENTS.md) |
| A better selection model for the joined candidates (fine-tuned against PubChem negatives) | 0.423 | [reviews/hardneg-2026-10-05.md](reviews/hardneg-2026-10-05.md) |
| Confidence gate for the join; GLACIER on [M+H]+ only; larger PubChem shortlist | 0.430; 0.431; 0.431 (neutral) | [reviews/confgate-2026-10-04.md](reviews/confgate-2026-10-04.md) |
| Parameter search over 1,764 configurations; joined-row verifier; their combinations | nothing distinguishable from chance locally | [reviews/param-search-2026-10-05.md](reviews/param-search-2026-10-05.md), [reviews/verifier-2026-10-05.md](reviews/verifier-2026-10-05.md), [reviews/combo-2026-10-05.md](reviews/combo-2026-10-05.md) |
| Better fingerprint models inside the main engine or engine 2 (ensembles, NP-tuned, hard-negative fine-tune) | better alone, no change in the final list | [reviews/fpbank-2026-10-06.md](reviews/fpbank-2026-10-06.md), [reviews/np-ours-2026-10-05.md](reviews/np-ours-2026-10-05.md) |
| Other spectrum models and generators (MIST, DreaMS heads, FRIGID, rule-based generator join) | weaker than what we have, or a trade | [reviews/dreams-2026-10-06.md](reviews/dreams-2026-10-06.md), [reviews/frigid-2026-10-05.md](reviews/frigid-2026-10-05.md), [reviews/genjoin-2026-10-06.md](reviews/genjoin-2026-10-06.md) |
| Narrower mass windows, adduct surrogates, extra data (Enveda-180 Zenodo release), other databases | no decoys removed without losing answers; nothing gained | [reviews/masswin-2026-10-05.md](reviews/masswin-2026-10-05.md), [reviews/anatomy-2026-10-06.md](reviews/anatomy-2026-10-06.md), [reviews/enveda180z-2026-10-06.md](reviews/enveda180z-2026-10-06.md) |

### How to read local numbers

- Every public model was trained on all of `train.parquet`, so any hold-out drawn from it is recognised. Local replays
  overstate the public chain and understate components fitted on unseen molecules.
- No local score orders our leaderboard results ([reviews/calib-2026-10-05.md](reviews/calib-2026-10-05.md),
  [reviews/panel2-2026-10-05.md](reviews/panel2-2026-10-05.md)). One rule has held: a change that lets more joined
  PubChem rows stand above answers that are pool rows loses.
- One submission resolves about ±0.004. A change needs a mechanism and a segment it moves, not a tuned number.
- The full final stage can be replayed locally with real ICEBERG / GLACIER scores
  ([reviews/fwd-replay-2026-10-05.md](reviews/fwd-replay-2026-10-05.md)).

### Flags added on top of the reference (all off by default)

| Flag | What it does | Status |
|---|---|---|
| `POP_MU`, `ICE_LAM_LIB`, `GL_LAM_LIB`, `FRAG_LAM`, `FRAG_MODE` | popularity prior, forward models off for library hits, fragment re-score | in the best config |
| `PC_JOIN_N`, `PC_JOIN_MAX_LIB`, `PC_JOIN_ORDER` | PubChem join | in the best config (100, 0.7) |
| `ICE_LAM`, `GL_LAM` | forward-model weights | in the best config (0.75) |
| `RANKER='fs' / 'fs_avg'`, `RANKER_LIB_GATE` | second engine pass with the fold-safe FPNet and our ranker below the gate | in the best config (`fs_avg`, 0.7) |
| `RANKER='public_free' / 'ours' / 'blend'` | order-free ties; the first own ranker | tried, not better |
| `PC_JOIN_CONF`, `PC_JOIN_CONF_DEEP`, `PC_JOIN_N_DEEP` | confidence gate for the join | neutral |
| `PC_MODELS`, `PC_PPM`, `MASS_OFFSET_BY_ADDUCT`, `PC_N1` | other selection model, mass window, shortlist of the PubChem channel | not better |
| `ICE_FT`, `ICE_SURROGATE`, `GL_SURROGATE`, `GL_MH_ONLY`, `JOIN_FWD_A` | forward-model variants | not better |
| `ENG_EXTRA_FPNET`, `FUSE_INJ_CREDIT`, `PC_FWD_T1`, `FUSE_*` | engine 2 and fusion variants | not better |
| `STRICT` | fail the run if any stage fell back (diagnostic submissions) | tool |

Reproduce the best version: see the `README.txt` in its folder. The current code with the v2 config still reproduces
the v2 Kaggle output exactly, and with every flag off the fusion equals the reference (`tests/test_fusion_equivalence.py`).

## Run

```bash
# from the repo root, with the venv active and Kaggle auth done (`kaggle auth login`)
python solutions/fusion-v4n/tools/download_data.py            # competition + packages for local re-fusion
python solutions/fusion-v4n/tools/build_notebook.py           # -> kaggle/casmi26-ours.ipynb + kernel-metadata.json
kaggle kernels push -p solutions/fusion-v4n/kaggle            # commit run (smoke) on Kaggle, then "Submit" in the UI
python solutions/fusion-v4n/tests/test_fusion_equivalence.py  # offline test

# variants: override CFG at build time, use a separate slug per experiment
python solutions/fusion-v4n/tools/build_notebook.py --slug casmi26-ours-alpha08 --set FUSE_ALPHA=0.8
# full visible-test dumps (plumbing check), then local re-fusion
python solutions/fusion-v4n/tools/build_notebook.py --slug casmi26-ours-full --set FULL_ON_COMMIT=true
python solutions/fusion-v4n/tools/download_data.py --run <user>/casmi26-ours-full --name full1
python solutions/fusion-v4n/tools/make_labels.py
python solutions/fusion-v4n/tools/refuse.py solutions/fusion-v4n/runs/full1 --grid FUSE_ALPHA=0.4,0.6,0.8
```

The visible test is a placeholder drawn from train: local scores on it check the plumbing and do not predict the LB.

## Data provenance
All inputs of the reference are public Kaggle datasets listed in `solution.json` → `kaggle.datasets`. The best
version adds one private dataset of ours, `huseyinemreaksoy/casmi26-ours-ranker-fs`: four LightGBM boosters (165
features) fitted on 6,687 candidate lists built from the competition's `train.parquet` only, with fingerprint
features from `dmitriigluzdov/casmi26-fold-safe-fpnet` (public; its card carries no licence statement) and the
public feature models; no NIST (`reviews/fsranker-2026-10-06.md`). It may be shared inside the team only.

The other inputs: ICEBERG / GLACIER are ms-pred
MassSpecGym checkpoints (MIT). No NIST-derived artefact is known to be used. Re-check before the final selection.

`huseyinemreaksoy/casmi26-fpnet-pc-hn` (flag `PC_MODELS='hn'`, private for now): FPNet A and B of
`ahmedberatozer/casmi26-v3-models`, each fine-tuned by us for 20,000 steps on the competition's `train.parquet` with
PubChem structures of the same mass window as negatives (`train/mine_pc_negatives.py`, `train/train_fpnet.py --pc_neg`;
`reviews/hardneg-2026-10-05.md`). Inputs: competition training data, public weights trained on it, PubChem (public
domain). No NIST, nothing keyed to `test.parquet`. To be made public if the rules or the hosts require external
artefacts to be public.

`goooman/casmi26-fp-npx4-soup` (flag `ENG_EXTRA_FPNET`, off by default; preset `v25a-v13a-np-eng2`): one file,
`fp_single_npx4soup.pt` (144 MB, sha256 `4f4c090a…`), Apache-2.0 (licence field of the dataset). Card: an FPNet of
prvsiyan's architecture, the weight average of five fine-tunes of a model `fp_np_l4`, each on 200 of the "250 Enveda
NP standards (timsTOF rows)", which matches `ingest_lib == 'enveda-np-examples'` of `train.parquet` (250 structures,
1,184 timsTOF rows). **The training data of the base model `fp_np_l4` is not documented and no training code is
public**; its weights are unrelated to the public `fp_single_s2` / `fp_merged_m1` (cosine of the layers about 0), so
it is not a fine-tune of those. Provenance is therefore incomplete: ask the author before a final selection relies on
it (`reviews/np-eng2-2026-10-05.md`).
