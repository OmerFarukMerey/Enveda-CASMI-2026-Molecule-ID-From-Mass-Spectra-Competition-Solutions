"""Behavioral checks for the aggregate audit, using only synthetic structures."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.profile_data import main, profile_paths


def _parquets(tmp_path: Path) -> tuple[Path, Path]:
    train = pa.table(
        {
            "ingest_lib": ["enveda-np-examples", "riken", "riken", "gnps", "gnps"],
            "normalized_smiles": ["C", "C", None, "O", "Cl"],
            "inchikey": ["A", "A", "B", "C", "D"],
            "inchikey14": ["SHARED", "SHARED", "R_ONLY", "AMBIG", "AMBIG"],
            "molecular_formula": ["CH4", "CH4", None, "H2O", "HCl"],
            "ionization_mode": ["positive", "positive", "negative", "positive", None],
            "instrument_type": ["timsTOF", "orbitrap", "orbitrap", "qtof", "qtof"],
            "adduct": ["[M+H]+", "[M+H]+", "[M-H]-", "[M+H]+", None],
            "adduct_orig": [None] * 5,
            "precursor_mz": [100.0, 100.0, 20.0, 30.0, float("inf")],
            "precursor_error_ppm": [1.0, -11.0, 30.0, None, float("nan")],
            "ms2_mzs": [[100.0, 102.0], [103.0], [], None, [float("inf"), 2.0]],
            "ms2_normalized_intensities": [[0.5, 1.0], [1.000002, -0.1], [], [0.9], [None, 1.0]],
            "num_peaks": [2, 1, 2, None, 2],
            "base_peak_intensity": [None, 1.0, None, 2.0, None],
            "collision_energy_ev": [None, [], [10.0], [10.0, 20.0], [None]],
            "collision_energy_orig": [None] * 5,
            "collision_energy_orig_units": ["eV", "eV", "NCE", None, "eV"],
        }
    )
    test = pa.table(
        {
            "molecule_id": ["example-1", "example-1"],
            "spectrum_id": ["s1", "s2"],
            "ms2_mzs": [[1.0, 2.0, 3.0], [2.0]],
            "ms2_normalized_intensities": [[0.5, 0.7, 1.0], [1.0]],
            "base_peak_intensity": [None, 2.0],
            "adduct": ["[M+H]+", "[M+H]+"],
            "ionization_mode": ["positive", "positive"],
            "instrument_type": ["timsTOF", "timsTOF"],
            "precursor_mz": [100.0, 110.0],
            "collision_energy_orig": ["20", "30"],
            "collision_energy_ev": [[], [20.0]],
            "collision_energy_orig_units": ["eV", "eV"],
        }
    )
    train_path, test_path = tmp_path / "train.parquet", tmp_path / "test.parquet"
    pq.write_table(train, train_path, row_group_size=2)
    pq.write_table(test, test_path)
    return train_path, test_path


def test_source_overlaps_ambiguity_and_quality_counts(tmp_path):
    train_path, test_path = _parquets(tmp_path)
    report = profile_paths(train_path, test_path)
    train = report["train"]
    assert (train["rows"], train["row_groups"], train["distinct_inchikey14"]) == (5, 3, 3)
    assert train["sources"]["riken"]["rows"] == 2
    assert train["sources"]["riken"]["distinct_inchikey14"] == 2
    assert train["enveda_np_examples_overlap"]["by_other_source"] == {"gnps": 0, "riken": 1}
    assert train["cross_source_distinct_inchikey14_overlaps"]["enveda-np-examples"]["riken"] == 1
    assert train["observed_same_key_ambiguity"]["smiles_keys"] == 1
    assert train["observed_same_key_ambiguity"]["formula_keys"] == 1
    assert train["column_nulls"]["collision_energy_ev"] == 1
    assert train["column_nulls"]["precursor_error_ppm"] == 1
    assert train["column_nonfinite_numeric"]["precursor_error_ppm"] == 1
    assert train["column_nonfinite_numeric"]["ms2_mzs"] == 1
    assert train["list_element_nulls"]["collision_energy_ev"] == 1
    assert train["overall"]["array_length_integrity"]["length_mismatch_rows"] == 1
    assert train["overall"]["array_length_integrity"]["num_peaks_mismatch_rows"] == 1
    assert train["sources"]["gnps"]["base_peak_intensity"]["missing_rows"] == 1
    assert train["sources"]["enveda-np-examples"]["collision_energy_ev"]["null_rate"] == 1
    assert train["sources"]["riken"]["collision_energy_ev"]["single_value_lists"] == 1
    ppm = train["overall"]["distributions"]["precursor_error_ppm"]
    assert ppm["finite_count"] == 3
    assert ppm["absolute_percentiles"]["0.5"] == 11
    assert ppm["absolute_outlier_counts_gt_ppm"]["10"] == 2


def test_per_source_null_counts_preserve_absent_versus_nonfinite(tmp_path):
    train_path, test_path = _parquets(tmp_path)
    sources = profile_paths(train_path, test_path)["train"]["sources"]
    assert sources["riken"]["column_nulls"] == {
        "normalized_smiles": 1,
        "molecular_formula": 1,
        "precursor_error_ppm": 0,
        "adduct": 0,
        "precursor_mz": 0,
        "num_peaks": 0,
        "ms2_mzs": 0,
        "ms2_normalized_intensities": 0,
        "collision_energy_ev": 0,
    }
    assert sources["gnps"]["column_nulls"] == {
        "normalized_smiles": 0,
        "molecular_formula": 0,
        "precursor_error_ppm": 1,
        "adduct": 1,
        "precursor_mz": 0,  # +infinity is nonfinite, not null
        "num_peaks": 1,
        "ms2_mzs": 1,
        "ms2_normalized_intensities": 0,
        "collision_energy_ev": 0,
    }
    assert sources["enveda-np-examples"]["column_nulls"]["collision_energy_ev"] == 1


def test_source_spectral_quality_flags_contaminated_rows(tmp_path):
    train_path, test_path = _parquets(tmp_path)
    report = profile_paths(train_path, test_path)["train"]
    riken = report["sources"]["riken"]["peak_quality"]
    assert riken["empty_mz_arrays"] == 1
    assert riken["empty_intensity_arrays"] == 1
    assert riken["mz_above_precursor_plus_2_da"] == 1
    assert riken["negative_intensity"] == 1
    assert riken["intensity_above_one"] == 1
    assert riken["intensity_max_not_one"] == 1
    assert report["sources"]["enveda-np-examples"]["peak_quality"]["mz_above_precursor_plus_2_da"] == 0
    gnps = report["sources"]["gnps"]["peak_quality"]
    assert gnps["nonfinite_intensity"] == 1  # null child is not a null list
    assert gnps["intensity_max_not_one"] == 1
    assert gnps["mz_above_precursor_plus_2_da"] == 0  # +infinite precursor is ineligible
    assert report["overall"]["peak_quality"]["intensity_max_not_one"] == 2


def test_placeholder_stats_and_json_exclude_raw_identifiers(tmp_path):
    train_path, test_path = _parquets(tmp_path)
    output = tmp_path / "analysis" / "profile.json"
    main(["--train", str(train_path), "--test", str(test_path), "--output", str(output)])
    data = json.loads(output.read_text())
    assert data["test"]["rows"] == 2
    assert data["test"]["overall"]["distributions"]["num_peaks"]["percentiles"]["0.5"] == 2
    assert data["test"]["overall"]["categories"]["adduct"] == {"[M+H]+": 2}
    assert data["test"]["overall"]["base_peak_intensity"]["missing_rate"] == 0.5
    assert "NOT the hidden scoring distribution" in data["test"]["placeholder_warning"]
    text = output.read_text()
    assert "SHARED" not in text
    assert "example-1" not in text
    assert str(tmp_path) not in text
