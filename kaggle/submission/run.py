"""Kaggle submission entry point (internet disabled).

Locates the competition files, the private candidate-pool dataset and the code
dataset under /kaggle/input, unpacks the code and runs scripts/predict.py to
write /kaggle/working/submission.csv. Kaggle replaces test.parquet with the
hidden test set when the notebook is rerun for scoring.
"""

from pathlib import Path
import subprocess
import sys
import zipfile

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def find(pattern: str) -> Path:
    matches = sorted(INPUT.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern}, found {matches}")
    return matches[0]


test = find("**/enveda-CASMI26-molecule-id-mass-spectra/test.parquet")
train = test.with_name("train.parquet")
pool = find("**/pool.parquet")

archives = sorted(INPUT.glob("**/casmi_code.zip"))
if archives:
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(WORK / "code")
    predict = WORK / "code" / "scripts" / "predict.py"
else:  # Kaggle may unpack uploaded archives into a directory
    predict = find("**/scripts/predict.py")

print({"test": str(test), "train": str(train), "pool": str(pool), "predict": str(predict)}, flush=True)
subprocess.run(
    [sys.executable, str(predict), "--test", str(test), "--train", str(train), "--pool", str(pool),
     "--output", str(WORK / "submission.csv")],
    check=True,
)
