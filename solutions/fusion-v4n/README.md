<!-- index: fusion-v4n: pipeline stages, what is the same as and different from the 0.399 reference, how to run -->
# fusion-v4n: public 0.399 pipeline, restructured for iteration

Owner: Hemre · Status: **public LB 0.431 (v13a, 2026-10-03)**; the default config reproduces the reference (0.399) · Baseline:
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
| `nb/05_v1_engine.py` | **v1 engine**: library + candidate pool + FPNet `fpnet_full1` + fe_v4 families + LightGBM ranker | |
| `nb/06_base_lists.py` | per molecule top-`TOPN`=60 (key-deduplicated), `lib_max`, scores, formulas | `base.json` |
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

## Best submission: public LB 0.431 (v13a)

| Version | Config | Public LB | Exact notebook |
|---|---|---|---|
| v1 | [configs/v1-0.399.json](configs/v1-0.399.json) (reference defaults) | 0.399 | |
| v2 | [configs/v2-0.404.json](configs/v2-0.404.json): pool popularity prior μ=0.15 + MetFrag-parsimony re-score in `gap` mode (λ=0.5, only molecules ICEBERG/GLACIER cannot score) | 0.404 | [kaggle/submitted/v2-0.404.ipynb](kaggle/submitted/v2-0.404.ipynb) |
| v4b | [configs/v4b-libgate-pop025.json](configs/v4b-libgate-pop025.json): v2 + ICEBERG / GLACIER off for library hits + popularity μ=0.25 | 0.409 | public: [Kaggle notebook](https://www.kaggle.com/code/huseyinemreaksoy/casmi26-sub-v4b-0-409-on-the-public) |
| v12a | [configs/v12a-v4b-pcjoin50-lib07.json](configs/v12a-v4b-pcjoin50-lib07.json): v4b + **PubChem join**: the best 50 PubChem-only candidates enter the v1 engine and are placed by its ranker when `lib_max` < 0.7 (`nb/embed/pc_join.py`, [reviews/pcjoin-2026-10-02.md](reviews/pcjoin-2026-10-02.md)) | 0.420 | [submissions/fusion-v4n-v12a-0.420](../../submissions/fusion-v4n-v12a-0.420/) |
| **v13a** | [configs/v13a-v12a-n100.json](configs/v13a-v12a-n100.json): v12a with 100 instead of 50 PubChem candidates in the join | **0.431** | [submissions/fusion-v4n-v13a-0.431](../../submissions/fusion-v4n-v13a-0.431/) |

Twelve submissions of this pipeline are logged in [../../docs/experiments.md](../../docs/experiments.md); every change to how
pool candidates are ordered stayed between 0.390 and 0.409. v12a and v13a are the first changes that act on answers outside
the main pool, and each added 0.011. Why: [reviews/mixed-regime-2026-10-02.md](reviews/mixed-regime-2026-10-02.md).

Reproduce v2: either upload `kaggle/submitted/v2-0.404.ipynb` (File → Import Notebook) and attach the 14 inputs listed in
`solution.json`, or build and push it:
```bash
python solutions/fusion-v4n/tools/build_notebook.py --config solutions/fusion-v4n/configs/v2-0.404.json --slug <your-slug> --out scratch/push
kaggle kernels push -p scratch/push
```
The current code with this config reproduces the v2 Kaggle output exactly (checked on the commit run, 0 differing rows).

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
All inputs are public Kaggle datasets listed in `solution.json` → `kaggle.datasets`. ICEBERG / GLACIER are ms-pred
MassSpecGym checkpoints (MIT). No NIST-derived artefact is known to be used. Re-check before the final selection.

`huseyinemreaksoy/casmi26-fpnet-pc-hn` (flag `PC_MODELS='hn'`, private for now): FPNet A and B of
`ahmedberatozer/casmi26-v3-models`, each fine-tuned by us for 20,000 steps on the competition's `train.parquet` with
PubChem structures of the same mass window as negatives (`train/mine_pc_negatives.py`, `train/train_fpnet.py --pc_neg`;
`reviews/hardneg-2026-10-05.md`). Inputs: competition training data, public weights trained on it, PubChem (public
domain). No NIST, nothing keyed to `test.parquet`. To be made public if the rules or the hosts require external
artefacts to be public.
