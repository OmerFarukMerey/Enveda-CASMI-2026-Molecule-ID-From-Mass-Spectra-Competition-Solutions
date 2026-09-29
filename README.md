# Enveda CASMI 2026 — Molecule ID from mass spectra

Team research for the [Enveda CASMI 2026 Kaggle competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra). The task is to rank up to 25 candidate molecular structures for each molecule from its MS/MS spectra.

**Current status:** [Stage A data audit and validation](STAGE_A_REPORT.md) is complete. The first [Stage B reference-retrieval baseline](STAGE_B_REPORT.md) reaches MRR@25 0.296 on the 250-query class-1 proxy; the candidate gate, not the ranking, limits it. No model has been trained or submitted. The [research strategy](RESEARCH_PLAN.md) describes the experiment sequence and competition constraints.

Competition data is subject to the [competition rules](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules) and is not included in this repository. Participants who have accepted those rules can download `train.parquet`, `test.parquet` and `sample_submission.csv` from Kaggle into the git-ignored `data/raw/` directory. The downloaded `test.parquet` contains **placeholder training examples**, not the hidden scoring molecules. See the research strategy for the exact download command and the implications for validation.

## Reproduce Stage A

After placing the three competition files in `data/raw/`:

```bash
uv sync --group dev
uv run python scripts/profile_data.py --train data/raw/train.parquet --test data/raw/test.parquet --output data/analysis/profile.json
uv run python scripts/build_validation.py --train data/raw/train.parquet --output data/analysis/validation.json --seed 2026
uv run python scripts/build_validation.py --train data/raw/train.parquet --output data/analysis/validation.json --seed 2026 --check
uv run python scripts/evaluate_retrieval.py --train data/raw/train.parquet --manifest data/analysis/validation.json --output data/analysis/retrieval_baseline.json
uv run python -m pytest -q
uv run ty check src scripts tests
```

`data/analysis/` is also ignored by git: the validation manifest contains competition-derived structure keys and row indices. The committed report contains only aggregate findings. `src/metric.py` implements local MRR@25 and stricter submission validation; it does not contain a trained model.