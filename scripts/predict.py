"""Write a competition submission.csv from test.parquet with the fixed pipeline.

Runs without RDKit (the Kaggle image does not ship it). The format check below
enforces the submission contract: every test molecule_id exactly once and
1..25 nonempty, semicolon-free, distinct SMILES per row.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import QuerySpectrum, Settings, predict

TEST_COLUMNS = ["molecule_id", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]


def load_test(path: Path) -> list[QuerySpectrum]:
    table = pq.read_table(path, columns=TEST_COLUMNS)
    columns = {name: table.column(name).to_pylist() for name in TEST_COLUMNS}
    return [
        QuerySpectrum(str(molecule), adduct, mz, mzs, intensities)
        for molecule, adduct, mz, mzs, intensities in zip(*(columns[name] for name in TEST_COLUMNS))
    ]


def check_format(molecule_ids: list[str], predictions: dict[str, list[str]], max_rank: int = 25) -> None:
    expected = set(molecule_ids)
    if set(predictions) != expected or len(predictions) != len(expected):
        missing, extra = expected - set(predictions), set(predictions) - expected
        raise ValueError(f"Prediction IDs differ from test IDs: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
    for molecule, guesses in predictions.items():
        if not 1 <= len(guesses) <= max_rank:
            raise ValueError(f"{molecule}: expected 1..{max_rank} guesses, got {len(guesses)}")
        if any(not guess.strip() or ";" in guess for guess in guesses):
            raise ValueError(f"{molecule}: empty guess or ';' inside a SMILES")
        if len(set(guesses)) != len(guesses):
            raise ValueError(f"{molecule}: duplicate SMILES")


def write_submission(path: Path, molecule_ids: list[str], predictions: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule_id", "smiles"])
        for molecule in molecule_ids:
            writer.writerow([molecule, ";".join(predictions[molecule])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=Path, default=Path("data/raw/test.parquet"))
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.parquet"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/pubchem-2026-09-27"))
    parser.add_argument("--smiles", type=Path, default=Path("data/external/pubchem-2026-09-27/CID-SMILES.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/submission/submission.csv"))
    args = parser.parse_args()
    started = perf_counter()
    queries = load_test(args.test)
    predictions, stats = predict(queries, args.train, args.catalog, args.smiles, settings=Settings())
    molecule_ids = list(dict.fromkeys(query.molecule_id for query in queries))
    check_format(molecule_ids, predictions)
    write_submission(args.output, molecule_ids, predictions)
    report: dict[str, float] = {**stats, "runtime_seconds": round(perf_counter() - started, 1)}
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
