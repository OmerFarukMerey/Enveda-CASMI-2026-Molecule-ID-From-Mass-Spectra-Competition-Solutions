# Enveda CASMI 2026 — Molecule ID from mass spectra

Team research for the [Enveda CASMI 2026 Kaggle competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra). The task is to rank up to 25 candidate molecular structures for each molecule from its MS/MS spectra.

**Current status:** [Stage A data audit and validation](STAGE_A_REPORT.md) is complete. The [Stage B reference-retrieval baseline](STAGE_B_REPORT.md) reaches MRR@25 0.890 on the 250-query class-1 proxy when each structure's full public library is searchable (0.296 with only one designated reference). [Stage C](STAGE_C_REPORT.md) built a frozen PubChem/COCONUT catalog: 74% of the 5,200 structure-holdout answers fall inside a ±10 ppm candidate window, but spectrum-free ranking reaches only MRR@25 0.157 because a window holds a median of 5,510 candidate structures. The [first Kaggle submission](STAGE_F_REPORT.md) (library search + catalog fallback, offline Notebook) scored **0.193** on the public leaderboard (median team 0.312 on 30 September 2026); a second submission with a training-structure ∪ COCONUT pool scored 0.163. Candidate ranking, not candidate source, is the bottleneck; Stage D adds fragmentation-based ranking. No model has been trained. The [research strategy](RESEARCH_PLAN.md) describes the experiment sequence and competition constraints.

Competition data is subject to the [competition rules](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules) and is not included in this repository. Participants who have accepted those rules can download `train.parquet`, `test.parquet` and `sample_submission.csv` from Kaggle into the git-ignored `data/raw/` directory. The downloaded `test.parquet` contains **placeholder training examples**, not the hidden scoring molecules. See the research strategy for the exact download command and the implications for validation.

## Reproduce

After placing the three competition files in `data/raw/`:

```bash
uv sync --group dev
uv run python scripts/profile_data.py --train data/raw/train.parquet --test data/raw/test.parquet --output data/analysis/profile.json
uv run python scripts/build_validation.py --train data/raw/train.parquet --output data/analysis/validation.json --seed 2026
uv run python scripts/build_validation.py --train data/raw/train.parquet --output data/analysis/validation.json --seed 2026 --check
uv run python scripts/evaluate_retrieval.py --class1-references designated --output data/analysis/retrieval_designated.json
uv run python scripts/evaluate_retrieval.py --class1-references library --output data/analysis/retrieval_library.json
uv run python scripts/build_catalog.py      # needs the PubChem/COCONUT snapshots listed in STAGE_C_REPORT.md
uv run python scripts/evaluate_catalog.py
uv run python scripts/build_pool.py         # training ∪ COCONUT candidate pool (RDKit, ~30 min)
uv run python scripts/predict.py            # submission.csv for data/raw/test.parquet
uv run python scripts/evaluate_pipeline.py  # the submitted pipeline on the fixed proxies
uv run python -m pytest -q
uv run ty check src scripts tests kaggle
```

`data/analysis/` is also ignored by git: the validation manifest contains competition-derived structure keys and row indices. The committed report contains only aggregate findings. `src/metric.py` implements local MRR@25 and stricter submission validation; it does not contain a trained model.