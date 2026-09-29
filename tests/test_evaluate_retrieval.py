"""End-to-end fixed-fold retrieval over small multi-row-group parquet libraries."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdkit import Chem

from scripts.build_validation import build_validation
from scripts.evaluate_retrieval import run
from src.metric import canonicalize_smiles_to_ik14


def _row(
    smiles: str, source: str, mz: float, peaks: tuple[float, ...],
    *, adduct: str | None = "[M+H]+", supplied: str | None = None,
) -> dict:
    return {
        "ingest_lib": source,
        "inchikey14": supplied or Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14],
        "normalized_smiles": smiles,
        "adduct": adduct,
        "precursor_mz": mz,
        "ms2_mzs": list(peaks),
        "ms2_normalized_intensities": [1.0] * len(peaks),
    }


def _fixture(tmp_path: Path, rows: list[dict], *, seed: int = 2026) -> tuple[Path, Path, Path, dict]:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    train = raw / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), train, row_group_size=2)
    manifest = build_validation(train, seed=seed)
    manifest_path = tmp_path / "data" / "analysis" / "validation.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    return train, manifest_path, manifest_path.with_name("baseline.json"), manifest


def test_exact_adduct_ppm_boundary_holdout_leakage_and_source_metrics(tmp_path: Path) -> None:
    rows = [
        _row("CCO", "enveda-np-examples", 100.0, (51.0, 52.0)),
        _row("CCN", "enveda-np-examples", 200.0, (61.0, 62.0)),
        _row("CCC", "enveda-np-examples", 300.0, (71.0, 72.0)),
        # Physical row 3 sits in another row group. The ppm boundary is inclusive.
        _row("CCO", "riken", 100.002, (51.0, 53.0)),
        _row("CCN", "gnps", 200.0, (61.0, 63.0), adduct="[M+Na]+"),
        _row("CCC", "massbank", 300.0, (71.0, 73.0)),
        # Only the explicit row of a class-1 group may serve as its reference.
        _row("CCO", "gnps", 100.0, (51.0, 52.0)),
        # This is the only remaining group, so the held-out key and all its
        # aliases must be excluded despite perfect peaks/adduct/precursor.
        _row("CCCl", "gnps", 100.0, (51.0, 52.0)),
        _row("CCCl", "riken", 100.0, (51.0, 52.0)),
        # Invalid arrays on an unrelated row must never enter peak preparation.
        {**_row("CCBr", "other", 500.0, (44.0,)), "ms2_normalized_intensities": []},
    ]
    train, manifest_path, output, manifest = _fixture(tmp_path, rows)
    assert len(manifest["class1_single_spectrum"]) == 3
    assert manifest["structure_holdout"][0]["query"]["inchikey14"] == rows[7]["inchikey14"]
    report = run(train, manifest_path, output)
    assert report["eligible_reference_rows"] == 4  # two explicit + one mismatched + unrelated
    assert report["holdout_excluded_rows"] == 2
    assert report["gated_reference_rows"] == 2
    assert report["gated_query_count"] == 2
    assert report["explicit_reference_gated_count"] == 2
    assert report["explicit_reference_gate_failures"] == 1
    assert report["predicted_query_count"] == 2
    assert report["candidate_recall_at_25"] == 2 / 3
    assert report["top1"] == 2 / 3
    assert report["mrr_at_25"] == 2 / 3
    assert {source: report["by_reference_source"][source]["count"] for source in ("riken", "gnps", "massbank")} == {
        "riken": 1, "gnps": 1, "massbank": 1,
    }
    assert report["by_reference_source"]["gnps"]["mrr_at_25"] == 0.0
    assert report["by_reference_source"]["riken"]["mrr_at_25"] == 1.0
    assert report["by_reference_source"]["massbank"]["mrr_at_25"] == 1.0
    assert json.loads(output.read_text()) == report
    assert "CCO" not in output.read_text()  # no labels or raw spectra written


def test_official_identity_dedup_keeps_best_row_and_deterministic_ties(tmp_path: Path) -> None:
    lactam, lactim = "O=c1cccc[nH]1", "Oc1ccccn1"
    assert canonicalize_smiles_to_ik14(lactam) == canonicalize_smiles_to_ik14(lactim)
    rows = [
        _row("CCO", "enveda-np-examples", 100, (50.0,)),
        _row("CCO", "riken", 100, (55.0,)),
        _row(lactam, "gnps", 100, (50.0,)),
        _row(lactim, "massbank", 100, (50.0,), supplied="AAAAAAAAAAAAAA"),
        _row("CCCl", "gnps", 400, (90.0,)),
    ]
    assert rows[2]["inchikey14"] != rows[3]["inchikey14"]
    # A held-out group is selected from the two unrelated groups. Choose a
    # fixed deterministic seed where the tautomer group remains in the library.
    for seed in range(2026, 2040):
        train, manifest_path, output, manifest = _fixture(tmp_path / str(seed), rows, seed=seed)
        if all(rows[2]["inchikey14"] not in entry["identity_aliases"] for entry in manifest["structure_holdout"]):
            break
    else:
        pytest.fail("No seed selected the independent control holdout")
    report = run(train, manifest_path, output, max_rank=2)
    assert report["candidate_recall_at_25"] == 1.0
    assert report["top1"] == 0.0
    assert report["mrr_at_25"] == 0.5  # two tautomer aliases occupy one rank
    assert report["gated_reference_rows"] == 3
    assert report["predicted_query_count"] == 1
    # With a single rank, the deterministic tied canonical candidate beats CCO.
    one = run(train, manifest_path, output, max_rank=1)
    assert one["mrr_at_25"] == 0.0
    assert one["candidate_recall_at_25"] == 0.0


def test_equal_cosines_and_ppm_rank_by_original_row_index(tmp_path: Path) -> None:
    query = _row("CCO", "enveda-np-examples", 100, (50.0, 51.0))
    known = _row("CCO", "riken", 100, (50.0, 52.0))
    decoy = _row("CCN", "gnps", 100, (50.0, 53.0))
    holdout = _row("CCCl", "riken", 400, (90.0,))
    for seed in range(2026, 2040):
        train, manifest_path, output, manifest = _fixture(
            tmp_path / str(seed) / "first", [query, known, decoy, holdout], seed=seed
        )
        if all(decoy["inchikey14"] not in entry["identity_aliases"]
               for entry in manifest["structure_holdout"]):
            break
    else:
        pytest.fail("No seed selected the independent control holdout")
    assert run(train, manifest_path, output, max_rank=1)["top1"] == 1.0
    train, manifest_path, output, _ = _fixture(
        tmp_path / str(seed) / "second", [query, decoy, known, holdout], seed=seed
    )
    assert run(train, manifest_path, output, max_rank=1)["top1"] == 0.0
    train, manifest_path, output, _ = _fixture(
        tmp_path / str(seed) / "ppm",
        [query, {**known, "precursor_mz": 100.001}, decoy, holdout],
        seed=seed,
    )
    # Equal cosine: the lower ppm error wins even when its row comes later.
    assert run(train, manifest_path, output, max_rank=1)["top1"] == 0.0


def test_no_gated_candidate_gives_zero_without_invalid_empty_submission(tmp_path: Path) -> None:
    rows = [
        _row("CCO", "enveda-np-examples", 100.0, (51.0,)),
        _row("CCO", "riken", 100.0021, (52.0,)),
    ]
    train, manifest_path, output, _ = _fixture(tmp_path, rows)
    report = run(train, manifest_path, output)
    assert report["explicit_reference_gated_count"] == 0
    assert report["gated_query_count"] == 0
    assert report["predicted_query_count"] == 0
    assert report["candidate_recall_at_25"] == report["mrr_at_25"] == 0.0
    assert output.exists()


def test_manifest_identity_row_groups_filename_and_parameter_validation(tmp_path: Path) -> None:
    rows = [
        _row("CCO", "enveda-np-examples", 100.0, (51.0,)),
        _row("CCO", "riken", 100.0, (52.0,)),
    ]
    train, manifest_path, output, manifest = _fixture(tmp_path, rows)
    with pytest.raises(ValueError, match="filename"):
        run(train.with_name("renamed.parquet"), manifest_path, output)
    for field, replacement in (("rows", 3), ("row_groups", 2)):
        altered = {**manifest, "input": {**manifest["input"], field: replacement}}
        manifest_path.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="row"):
            run(train, manifest_path, output)
    manifest_path.write_text(json.dumps(manifest))
    for limit in (0, 26):
        with pytest.raises(ValueError):
            run(train, manifest_path, output, max_rank=limit)
    with pytest.raises(ValueError):
        run(train, manifest_path, output, max_peaks=0)
    with pytest.raises(ValueError):
        run(train, manifest_path, output, precursor_tolerance_ppm=float("nan"))
    with pytest.raises(ValueError):
        run(train, manifest_path, output, fragment_tolerance_da=float("inf"))
    # An alternate file with identical layout but changed selected row identity
    # must not be mistaken for the validated training file.
    modified = list(rows)
    modified[1] = _row("CCN", "riken", 100.0, (52.0,))
    pq.write_table(pa.Table.from_pylist(modified), train, row_group_size=2)
    with pytest.raises(ValueError, match="identity"):
        run(train, manifest_path, output)
