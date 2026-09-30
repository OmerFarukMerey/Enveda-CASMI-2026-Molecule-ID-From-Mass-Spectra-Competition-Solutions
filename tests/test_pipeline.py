"""Submission pipeline: library/pool fusion, eligibility, support and format."""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.predict import check_format, load_test, write_submission
from src.adducts import ELECTRON_MASS, ELEMENT_MASSES
from src.pipeline import FILLER_SMILES, QuerySpectrum, Settings, predict

PROTON = ELEMENT_MASSES["H"] - ELECTRON_MASS
POTASSIUM = ELEMENT_MASSES["K"] - ELECTRON_MASS
M1 = 100.0 - PROTON  # neutral mass of molecule m1, observed as [M+H]+ at m/z 100

# (key14, canonical14, smiles, mass, in_train, coconut)
POOL = [
    ("LIBSTRONGAAAAA", "CANSTRONGAAAAA", "L1", M1, True, False),
    ("LIBWEAKAAAAAAA", "CANWEAKAAAAAAA", "P2", M1, True, False),
    ("COCONUTONLYAAA", "CANCOCONUTAAAA", "K1", M1, False, True),
    ("TAUTOMERAAAAAA", "CANSTRONGAAAAA", "L1-tautomer", M1, False, True),  # same official key as L1
    ("TRAINNOSPECTRA", "CANNOSPECTRAAA", "X", M1, True, False),            # no eligible spectrum
    ("SHAREDBYBOTHAA", "CANSHAREDAAAAA", "SY", 45.05790, False, True),     # in both m2 windows
    ("ONESPECTRUMAAA", "CANONEAAAAAAAA", "SX", 45.05740, True, False),     # in one window, 3 spectra
    ("LIBONLYWEAKAAA", "CANLIBONLYAAAA", "PW", 500.0, True, False),        # far from m1 in the pool
]


def _pool(tmp_path: Path) -> Path:
    rows = sorted(POOL, key=lambda row: row[3])
    names = ("key14", "canonical14", "smiles", "mass", "in_train", "coconut")
    path = tmp_path / "pool.parquet"
    pq.write_table(pa.table({name: [row[i] for row in rows] for i, name in enumerate(names)}), path)
    return path


def _train(tmp_path: Path) -> Path:
    def row(key: str | None, smiles: str, mz: float, peaks: tuple[float, ...]) -> dict:
        return {"ingest_lib": "gnps", "inchikey14": key, "normalized_smiles": smiles, "adduct": "[M+H]+",
                "precursor_mz": mz, "ms2_mzs": list(peaks), "ms2_normalized_intensities": [1.0] * len(peaks)}

    rows = [
        row("LIBSTRONGAAAAA", "L1", 100.0, (50.0, 60.0)),        # cosine 1 with the m1 query
        row("LIBWEAKAAAAAAA", "L2", 100.0, (50.0, 70.0, 80.0)),  # cosine 1/sqrt(6)
        row(None, "UNLABELED", 100.0, (50.0, 60.0)),             # no key: never a reference
        row("LIBSTRONGAAAAA", "L1-other", 100.0, (50.0, 90.0)),  # same structure, worse score
        *[row("ONESPECTRUMAAA", "SX", 46.0651, (30.0,)) for _ in range(3)],
        row("LIBONLYWEAKAAA", "LW", 100.0, (50.0, 71.0, 81.0)),  # weak hit, not in m1's pool window
    ]
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    return path


def _queries() -> list[QuerySpectrum]:
    return [
        QuerySpectrum("m1", "[M+H]+", 100.0, [50.0, 60.0], [1.0, 1.0]),
        QuerySpectrum("m2", "[M+K]+", 45.0578 + POTASSIUM, [30.0], [1.0]),  # spectrum A
        QuerySpectrum("m2", "[M+K]+", 45.0580 + POTASSIUM, [30.0], [1.0]),  # spectrum B
        QuerySpectrum("m3", "[M+Xx]+", 150.0, [30.0], [1.0]),               # no neutral mass
    ]


def test_fusion_order_official_dedup_eligibility_support_and_filler(tmp_path: Path) -> None:
    predictions, stats = predict(_queries(), _train(tmp_path), _pool(tmp_path))
    # Strong library hit; then pool by library spectrum count (the tautomer of L1
    # and the training structure without eligible spectra never appear); then the
    # weak library-only structure. The weak hit on LIBWEAK appears once, from the pool.
    assert predictions["m1"] == ["L1", "P2", "K1", "LW"]
    # Two-spectrum support outranks more library spectra.
    assert predictions["m2"] == ["SY", "SX"]
    assert predictions["m3"] == [FILLER_SMILES]
    assert stats == {
        "molecules": 3, "spectra": 4, "spectra_without_neutral_mass": 1,
        "pool_rows": 8, "pool_rows_eligible": 7,
        "molecules_with_strong_library_hit": 1, "molecules_with_library_hit": 1,
        "molecules_with_pool_candidates": 2, "molecules_with_filler_only": 1,
    }
    capped, _ = predict(_queries(), _train(tmp_path), _pool(tmp_path), settings=Settings(max_rank=2))
    assert capped["m1"] == ["L1", "P2"]


def test_library_eligibility_rule_controls_pool_membership(tmp_path: Path) -> None:
    queries = [QuerySpectrum("m1", "[M+H]+", 100.0, [50.0, 60.0], [1.0, 1.0])]
    # Excluding the strong structure's rows removes it from the library and from
    # the training side of the pool; its COCONUT tautomer remains reachable.
    predictions, stats = predict(
        queries, _train(tmp_path), _pool(tmp_path),
        allowed=lambda _row, supplied, _source: bool(supplied) and supplied != "LIBSTRONGAAAAA",
    )
    assert predictions["m1"] == ["P2", "K1", "L1-tautomer", "LW"]
    assert stats["molecules_with_strong_library_hit"] == 0


def test_submission_format_and_csv(tmp_path: Path) -> None:
    ids = ["a", "b"]
    check_format(ids, {"a": ["C"], "b": ["N", "O"]})
    for predictions in ({"a": ["C"]}, {"a": ["C"], "b": []}, {"a": ["C"], "b": ["N;O"]},
                        {"a": ["C"], "b": ["N", "N"]}, {"a": ["C"], "b": ["C"] * 26}, {"a": ["C"], "b": [" "]}):
        with pytest.raises(ValueError):
            check_format(ids, predictions)
    output = tmp_path / "submission.csv"
    write_submission(output, ids, {"a": ["C"], "b": ["N", "O"]})
    with output.open() as handle:
        assert list(csv.reader(handle)) == [["molecule_id", "smiles"], ["a", "C"], ["b", "N;O"]]


def test_load_test_keeps_every_spectrum_with_its_molecule(tmp_path: Path) -> None:
    path = tmp_path / "test.parquet"
    pq.write_table(pa.table({
        "molecule_id": ["x", "x", "y"], "spectrum_id": ["s1", "s2", "s3"],
        "adduct": ["[M+H]+", "[M-H]-", "[M+H]+"], "precursor_mz": [1.0, 2.0, 3.0],
        "ms2_mzs": [[1.0], [2.0], [3.0]], "ms2_normalized_intensities": [[1.0], [1.0], [1.0]],
    }), path)
    queries = load_test(path)
    assert [(q.molecule_id, q.adduct, q.precursor_mz) for q in queries] == [
        ("x", "[M+H]+", 1.0), ("x", "[M-H]-", 2.0), ("y", "[M+H]+", 3.0)]
