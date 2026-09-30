<!-- index: layout of the git-ignored data/ folder (mirrors /kaggle/input) and how to fill it -->
# data/ (git-ignored except this file)

Mirrors Kaggle's `/kaggle/input` so notebook code and local tools see the same paths:

```
data/
  input/
    competitions/enveda-CASMI26-molecule-id-mass-spectra/   train.parquet, test.parquet, sample_submission.csv
    datasets/<owner>/<slug>/                                  public Kaggle datasets (models, pools, wheels)
  visible_labels.csv                                          labels for the placeholder visible test (sanity only)
```

Populate with `python solutions/fusion-v4n/tools/download_data.py` (`--all` for all 13 notebook inputs, ~16 GB).
It needs Kaggle auth (`kaggle auth login`, or a token in ~/.kaggle/) and the competition rules accepted on kaggle.com.
Point elsewhere with `CASMI_DATA=/path/to/data`.

Competition data is CC BY-NC 4.0 and subject to the competition rules. Never commit it.
The research branch uses `data/raw/` for the three competition files. Both layouts can coexist.
