"""Package the inference code as a Kaggle dataset folder (data/kaggle/code-dataset/).

The archive holds src/ and scripts/predict.py only; nothing under data/ is
included. Upload with:
  uvx --from kaggle kaggle datasets create -p data/kaggle/code-dataset      (first time)
  uvx --from kaggle kaggle datasets version -p data/kaggle/code-dataset -m "<message>"
"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "kaggle" / "code-dataset"
FILES = sorted((ROOT / "src").glob("*.py")) + [ROOT / "scripts" / "predict.py"]
METADATA = {
    "title": "CASMI26 inference code",
    "id": "ilkeryasin8/casmi26-code",
    "subtitle": "Library search and catalog pipeline for the Enveda CASMI 2026 submission",
    "description": "src/ and scripts/predict.py from the team repository (MIT).",
    "licenses": [{"name": "other"}],
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT / "casmi_code.zip"
    # Fixed timestamps keep the archive byte-identical for unchanged code.
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in FILES:
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
            archive.writestr(info, path.read_bytes())
    (OUTPUT / "dataset-metadata.json").write_text(json.dumps(METADATA, indent=2) + "\n")
    print(archive_path, [str(path.relative_to(ROOT)) for path in FILES])


if __name__ == "__main__":
    main()
