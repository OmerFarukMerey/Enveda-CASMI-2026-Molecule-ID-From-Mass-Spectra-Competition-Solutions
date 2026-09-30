"""Submission pipeline: library/catalog fusion, multi-spectrum support and format."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.build_catalog import build
from scripts.predict import check_format, load_test, write_submission
from src.adducts import ELECTRON_MASS, ELEMENT_MASSES
from src.pipeline import FILLER_SMILES, QuerySpectrum, Settings, predict

PROTON = ELEMENT_MASSES["H"] - ELECTRON_MASS
POTASSIUM = ELEMENT_MASSES["K"] - ELECTRON_MASS
M1 = 100.0 - PROTON  # neutral mass of molecule m1, observed as [M+H]+ at m/z 100
STRONG, WEAK, CATALOG_ONLY = "LIBSTRONGAAAAA", "LIBWEAKAAAAAAA", "CATALOGONLYAAA"
SHARED, SINGLE = "SHAREDBYBOTHAA", "ONESPECTRUMAAA"


def _gz(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt") as handle:
        handle.write("".join(f"{line}\n" for line in lines))


def _catalog(tmp_path: Path) -> tuple[Path, Path]:
    pubchem = tmp_path / "data" / "pubchem"
    pubchem.mkdir(parents=True)
    rows = {  # cid: (key, smiles, mass, pubmed links)
        1: (CATALOG_ONLY, "K1", M1, 0),
        2: (WEAK, "K2", M1, 0),
        3: (SINGLE, "SX", 45.05740, 9),   # in spectrum A's window only, better prior
        4: (SHARED, "SY", 45.05790, 0),   # in both spectra's windows
    }
    _gz(pubchem / "CID-Mass.gz", [f"{c}\tX\t{m}\t{m}" for c, (_, _, m, _) in rows.items()])
    _gz(pubchem / "CID-InChI-Key.gz", [f"{c}\tInChI=1S/x\t{k}-UHFFFAOYSA-N" for c, (k, _, _, _) in rows.items()])
    _gz(pubchem / "CID-SMILES.gz", [f"{c}\t{s}" for c, (_, s, _, _) in rows.items()])
    _gz(pubchem / "CID-PMID.gz", [f"{c}\t{i}\t1" for c, (_, _, _, n) in rows.items() for i in range(n)])
    coconut = tmp_path / "data" / "coconut.zip"
    with zipfile.ZipFile(coconut, "w") as archive:
        archive.writestr("c.csv", f"identifier,standard_inchi_key\nC1,{CATALOG_ONLY}-UHFFFAOYSA-N\n")
    catalog = tmp_path / "data" / "catalog"
    build(pubchem, coconut, catalog)
    return catalog, pubchem / "CID-SMILES.gz"


def _train(tmp_path: Path) -> Path:
    def row(key: str | None, smiles: str, peaks: tuple[float, ...]) -> dict:
        return {"ingest_lib": "gnps", "inchikey14": key, "normalized_smiles": smiles, "adduct": "[M+H]+",
                "precursor_mz": 100.0, "ms2_mzs": list(peaks), "ms2_normalized_intensities": [1.0] * len(peaks)}

    path = tmp_path / "train.parquet"
    rows = [
        row(STRONG, "L1", (50.0, 60.0)),            # cosine 1 with the m1 query
        row(WEAK, "L2", (50.0, 70.0, 80.0)),        # cosine 1/sqrt(6)
        row(None, "UNLABELED", (50.0, 60.0)),       # no key: never a reference
        row(STRONG, "L1-other", (50.0, 90.0)),      # same structure, worse score
    ]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    return path


def _queries() -> list[QuerySpectrum]:
    return [
        QuerySpectrum("m1", "[M+H]+", 100.0, [50.0, 60.0], [1.0, 1.0]),
        QuerySpectrum("m2", "[M+K]+", 45.0578 + POTASSIUM, [30.0], [1.0]),  # spectrum A
        QuerySpectrum("m2", "[M+K]+", 45.0580 + POTASSIUM, [30.0], [1.0]),  # spectrum B
        QuerySpectrum("m3", "[M+Xx]+", 150.0, [30.0], [1.0]),               # no neutral mass
    ]


def test_fusion_order_dedup_support_and_filler(tmp_path: Path) -> None:
    catalog, smiles = _catalog(tmp_path)
    predictions, stats = predict(_queries(), _train(tmp_path), catalog, smiles)
    # Strong library hit first; catalog next (COCONUT member first); the weak
    # library structure appears once, at its catalog position.
    assert predictions["m1"] == ["L1", "K1", "K2"]
    # Two-spectrum support outranks a better single-spectrum prior.
    assert predictions["m2"] == ["SY", "SX"]
    assert predictions["m3"] == [FILLER_SMILES]
    assert stats == {
        "molecules": 3, "spectra": 4, "spectra_without_neutral_mass": 1,
        "molecules_with_strong_library_hit": 1, "molecules_with_library_hit": 1,
        "molecules_with_catalog_candidates": 2, "molecules_with_filler_only": 1,
    }
    capped, _ = predict(_queries(), _train(tmp_path), catalog, smiles, settings=Settings(max_rank=2))
    assert capped["m1"] == ["L1", "K1"]


def test_weak_library_hits_fill_after_catalog(tmp_path: Path) -> None:
    catalog, smiles = _catalog(tmp_path)
    queries = [QuerySpectrum("m1", "[M+H]+", 100.0, [50.0, 60.0], [1.0, 1.0])]
    # With a threshold above every cosine, even the perfect match waits for the catalog.
    predictions, stats = predict(queries, _train(tmp_path), catalog, smiles,
                                 settings=Settings(min_library_cosine=1.01))
    assert predictions["m1"] == ["K1", "K2", "L1"]
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
