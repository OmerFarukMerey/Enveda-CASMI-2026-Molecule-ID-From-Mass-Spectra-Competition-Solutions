# Enveda CASMI 2026 — Molecule ID from mass spectra

Team research for the [Enveda CASMI 2026 Kaggle competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra). The task is to rank up to 25 candidate molecular structures for each molecule from its MS/MS spectra.

**Current status:** competition research and initial data inventory. No model has been trained or submitted. Read the [research strategy](RESEARCH_PLAN.md) for verified dataset observations, validation design, experiment priorities, decision gates and competition constraints.

Competition data is subject to the [competition rules](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules) and is not included in this repository. Participants who have accepted those rules can download `train.parquet`, `test.parquet` and `sample_submission.csv` from Kaggle into the git-ignored `data/raw/` directory. The downloaded `test.parquet` contains **placeholder training examples**, not the hidden scoring molecules. See the research strategy for the exact download command and the implications for validation.