"""Consumer-visible split invariants on small, physically distinct parquet rows."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from scripts.build_validation import _canonical_key, build_validation, check_manifest, reference_filter


def _key(smiles: str) -> str:
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14]


def _row(smiles: str, source: str, peaks: tuple[float, ...] = (101.0, 102.0)) -> dict:
    return {
        "ingest_lib": source,
        "inchikey14": _key(smiles),
        "normalized_smiles": smiles,
        "ms2_mzs": list(peaks),
        "ms2_normalized_intensities": [0.4, 1.0][: len(peaks)],
    }


def _parquet(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
    return path


def test_deterministic_cross_source_query_and_grouped_holdout(tmp_path: Path) -> None:
    rows = [
        _row("CCO", "enveda-np-examples"),
        _row("CCO", "enveda-np-examples", (103.0, 104.0)),
        _row("CCO", "gnps", (105.0, 106.0)),
        _row("CCN", "enveda-np-examples"),
        _row("CCN", "enveda-np-examples", (105.0, 106.0)),
        _row("CCC", "gnps"),
        _row("CCC", "riken", (107.0, 108.0)),
        _row("CCCl", "riken"),
    ]
    path = _parquet(tmp_path, rows)
    first = build_validation(path, 2026)
    second = build_validation(path, 2026)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(first["class1_single_spectrum"]) == 1
    pair = first["class1_single_spectrum"][0]
    assert pair["query"]["row_index"] == 0
    assert pair["reference"]["row_index"] == 2
    # Repeating a key within the query source never creates a reference.
    assert _key("CCN") not in {entry["query"]["inchikey14"] for entry in first["class1_single_spectrum"]}
    assert first["structure_holdout"]
    for entry in first["structure_holdout"]:
        key = entry["query"]["inchikey14"]
        assert all(
            row["inchikey14"] not in entry["identity_aliases"]
            for row_index, row in enumerate(rows)
            if row_index == pair["reference"]["row_index"]
        )
        assert {i for i, row in enumerate(rows) if row["inchikey14"] == key}.isdisjoint(
            {pair["reference"]["row_index"]}
        )
    check_manifest(first)


def test_requested_holdout_source_excludes_same_key_from_every_source(tmp_path: Path) -> None:
    rows = [
        _row("CCN", "enveda-180"),
        _row("CCN", "riken", (103.0, 104.0)),
        _row("CCN", "gnps", (105.0, 106.0)),
        _row("CCO", "enveda-np-examples"),
        _row("CCO", "riken", (107.0, 108.0)),
    ]
    path = _parquet(tmp_path, rows)
    manifest = build_validation(path, query_sources=("gnps",))
    holdout = manifest["structure_holdout"]
    assert len(holdout) == 1
    assert holdout[0]["query"]["row_index"] == 2
    assert holdout[0]["query"]["source"] == "gnps"
    assert manifest["counts"]["structure_holdout_eligible_groups"] == 1
    assert manifest["counts"]["structure_holdout_query_source_counts"] == {"gnps": 1}
    eligible = reference_filter(manifest)
    assert all(not eligible(i, rows[i]["inchikey14"]) for i in (0, 1, 2))
    assert not eligible(3, rows[3]["inchikey14"])
    assert eligible(4, rows[4]["inchikey14"])
    default_manifest = build_validation(path)
    assert default_manifest["structure_holdout"][0]["query"]["row_index"] == 1
    assert default_manifest["structure_holdout"][0]["query"]["source"] == "riken"


def test_natural_product_reference_priority_over_parquet_row_order(tmp_path: Path) -> None:
    rows = [
        _row("CCO", "enveda-np-examples"),
        _row("CCO", "drug_plus", (105.0, 106.0)),
        _row("CCO", "riken", (107.0, 108.0)),
    ]
    manifest = build_validation(_parquet(tmp_path, rows))
    pair = manifest["class1_single_spectrum"][0]
    assert pair["reference"]["source"] == "riken"
    assert pair["reference"]["row_index"] == 2



def test_exact_peak_copy_is_rejected_and_next_independent_reference_used(tmp_path: Path) -> None:
    rows = [
        _row("CCO", "enveda-np-examples"),
        _row("CCO", "gnps", (111.0, 112.0)),
        _row("CCO", "riken"),
        _row("CCN", "enveda-np-examples"),
        _row("CCN", "riken"),
        _row("CCC", "gnps"),
    ]
    manifest = build_validation(_parquet(tmp_path, rows))
    pairs = manifest["class1_single_spectrum"]
    assert len(pairs) == 1
    assert pairs[0]["query"]["row_index"] == 0
    assert pairs[0]["reference"]["row_index"] == 1
    assert manifest["counts"]["rejected_exact_peak_pairs"] == 2
    assert _key("CCN") not in {entry["query"]["inchikey14"] for entry in pairs}


def test_tautomer_aliases_never_split_across_holdout_and_reference(tmp_path: Path) -> None:
    lactam, lactim = "O=c1cccc[nH]1", "Oc1ccccn1"
    provided_alias = "AAAAAAAAAAAAAA"
    assert _key(lactam) != provided_alias
    assert _canonical_key(lactam) == _canonical_key(lactim)
    rows = [
        _row("CCO", "enveda-np-examples"),
        _row("CCO", "gnps", (111.0, 112.0)),
        _row(lactam, "gnps"),
        {**_row(lactim, "riken", (111.0, 112.0)), "inchikey14": provided_alias},
    ]
    manifest = build_validation(_parquet(tmp_path, rows))
    assert len(manifest["class1_single_spectrum"]) == 1
    assert len(manifest["structure_holdout"]) == 1
    heldout = manifest["structure_holdout"][0]
    assert {_key(lactam), provided_alias} <= set(heldout["identity_aliases"])
    assert {2, 3}.isdisjoint(
        {entry["reference"]["row_index"] for entry in manifest["class1_single_spectrum"]}
    )
    assert heldout["query"]["row_index"] in {2, 3}


def test_invalid_canonical_label_excludes_its_group(tmp_path: Path) -> None:
    rows = [
        _row("CCO", "enveda-np-examples"),
        _row("CCO", "gnps", (111.0, 112.0)),
        {**_row("CCO", "riken"), "normalized_smiles": "not-a-smiles"},
        _row("CCN", "gnps"),
    ]
    manifest = build_validation(_parquet(tmp_path, rows))
    assert manifest["class1_single_spectrum"] == []
    assert _key("CCO") in manifest["excluded_invalid_identity_aliases"]
    assert manifest["counts"]["invalid_canonical_label_rows"] == 1
