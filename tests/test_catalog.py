"""Catalog build accounting and holdout retrieval on tiny PubChem/COCONUT snapshots."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdkit import Chem

from scripts.build_catalog import build
from scripts.build_validation import build_validation
from scripts.evaluate_catalog import run

ETHYLAMINE, DIMETHYLAMINE, ETHANOL = "CCN", "CNC", "CCO"  # first two share C2H7N
C2H7N = 45.057849


def _key(smiles: str) -> str:
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def _gz(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt") as handle:
        handle.write("".join(f"{line}\n" for line in lines))


def _catalog(tmp_path: Path) -> tuple[Path, Path, Path]:
    pubchem = tmp_path / "data" / "external" / "pubchem"
    pubchem.mkdir(parents=True)
    structures = {  # cid: (smiles or None, mass)
        10: ("CCN.Cl", C2H7N),      # salt: multi-component, excluded
        15: (None, C2H7N),          # no SMILES record, excluded
        20: (ETHYLAMINE, C2H7N),    # the answer
        30: (DIMETHYLAMINE, C2H7N),  # same-mass decoy with more literature
        40: (ETHYLAMINE, C2H7N),    # second CID of the answer's connectivity
        50: (ETHANOL, 46.041865),   # outside a 10 ppm window
    }
    order = [50, 10, 15, 20, 30, 40]  # CID-Mass deliberately not sorted by CID
    _gz(pubchem / "CID-Mass.gz", [f"{cid}\tX\t{structures[cid][1]}\t{structures[cid][1]}" for cid in order])
    _gz(pubchem / "CID-InChI-Key.gz", [
        f"{cid}\tInChI=1S/x\t{_key(smiles or ETHYLAMINE)}" for cid, (smiles, _) in structures.items()
    ])
    _gz(pubchem / "CID-SMILES.gz", [f"{cid}\t{smiles}" for cid, (smiles, _) in structures.items() if smiles])
    _gz(pubchem / "CID-PMID.gz", ["20\t1\t1"] + [f"30\t{pmid}\t1" for pmid in range(5)])
    coconut = tmp_path / "data" / "external" / "coconut.zip"
    with zipfile.ZipFile(coconut, "w") as archive:
        archive.writestr("coconut.csv", f"identifier,standard_inchi_key\nCNP1,{_key(ETHYLAMINE)}\n")
    return pubchem, coconut, tmp_path / "data" / "catalog"


def _row(smiles: str, source: str, mz: float, peaks: tuple[float, ...], error: float | None = 0.5) -> dict:
    return {
        "ingest_lib": source,
        "inchikey14": _key(smiles)[:14],
        "normalized_smiles": smiles,
        "adduct": "[M+H]+",
        "precursor_mz": mz,
        "precursor_error_ppm": error,
        "ms2_mzs": list(peaks),
        "ms2_normalized_intensities": [1.0] * len(peaks),
    }


def _holdout(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    train = raw / "train.parquet"
    rows = [
        _row(ETHANOL, "enveda-np-examples", 47.049141, (30.0,)),  # class-1 query
        _row(ETHANOL, "riken", 47.049141, (31.0,)),               # its reference
        _row(ETHYLAMINE, "riken", C2H7N + 1.007276, (30.0, 44.0)),  # the only holdout group
    ]
    pq.write_table(pa.Table.from_pylist(rows), train, row_group_size=2)
    manifest = build_validation(train)
    assert [entry["query"]["row_index"] for entry in manifest["structure_holdout"]] == [2]
    path = tmp_path / "data" / "analysis" / "validation.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    return train, path


def test_build_excludes_salts_and_missing_smiles_and_sorts_by_mass(tmp_path: Path) -> None:
    pubchem, coconut, output = _catalog(tmp_path)
    manifest = build(pubchem, coconut, output)
    assert manifest["row_accounting"] == {
        "mass_rows": 6, "missing_or_invalid_mass": 0, "missing_inchikey": 0,
        "missing_smiles": 1, "multi_component": 1, "kept": 4,
    }
    mass = np.load(output / "mass.npy")
    cid = np.load(output / "cid.npy")
    assert np.all(np.diff(mass) >= 0)
    assert sorted(cid.tolist()) == [20, 30, 40, 50]
    pmid = dict(zip(cid.tolist(), np.load(output / "pmid_count.npy").tolist()))
    assert pmid == {20: 1, 30: 5, 40: 0, 50: 0}
    coconut_flag = dict(zip(cid.tolist(), np.load(output / "coconut.npy").tolist()))
    assert coconut_flag == {20: True, 30: False, 40: True, 50: False}  # membership is by key, not CID


def test_holdout_membership_window_dedup_and_prior_ranking(tmp_path: Path) -> None:
    pubchem, coconut, catalog = _catalog(tmp_path)
    build(pubchem, coconut, catalog)
    train, manifest = _holdout(tmp_path)
    output = tmp_path / "data" / "analysis" / "catalog.json"
    report = run(train, manifest, catalog, pubchem / "CID-SMILES.gz", output)
    assert report["answer_in_catalog"] == report["answer_in_coconut"] == 1
    assert report["answer_in_window_at_ppm"] == {"5": 1, "10": 1, "20": 1}
    # CIDs 20 and 40 share a connectivity: two distinct structures in the window, not three.
    assert report["window_distinct_structures_quantiles"]["1"] == 2.0
    results = report["results_by_prior"]
    assert results["cid"]["all"]["mrr_at_25"] == 1.0           # CID 20 < CID 30
    assert results["pmid"]["all"]["mrr_at_25"] == 0.5          # the decoy has more PubMed links
    assert results["coconut_pmid"]["all"]["mrr_at_25"] == 1.0  # natural-product prior wins
    assert results["pmid"]["source_riken"]["count"] == 1
    assert "CCN" not in output.read_text()  # aggregates only

    one = run(train, manifest, catalog, pubchem / "CID-SMILES.gz", output, max_rank=1)
    assert one["results_by_prior"]["pmid"]["all"]["candidate_recall_at_25"] == 0.0


def test_parameter_and_location_validation(tmp_path: Path) -> None:
    pubchem, coconut, catalog = _catalog(tmp_path)
    build(pubchem, coconut, catalog)
    train, manifest = _holdout(tmp_path)
    smiles = pubchem / "CID-SMILES.gz"
    output = tmp_path / "data" / "x.json"
    for tolerance in (0.0, float("nan")):
        with pytest.raises(ValueError):
            run(train, manifest, catalog, smiles, output, tolerance_ppm=tolerance)
    for limit in (0, 26):
        with pytest.raises(ValueError):
            run(train, manifest, catalog, smiles, output, max_rank=limit)
    with pytest.raises(ValueError, match="data/"):
        run(train, manifest, catalog, smiles, tmp_path / "elsewhere.json")
