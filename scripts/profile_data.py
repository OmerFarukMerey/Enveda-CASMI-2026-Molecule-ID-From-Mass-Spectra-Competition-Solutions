"""Aggregate a reproducible, structure-safe Stage A audit from Parquet batches.

Only counts and numerical summaries leave this process; no structure, spectrum, or
per-identifier data are written. Percentiles are exact over finite observations.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


CATEGORIES = ("ionization_mode", "adduct", "instrument_type", "collision_energy_orig_units")
NUMERIC = ("precursor_mz", "precursor_error_ppm", "num_peaks")
LISTS = ("ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev")
SOURCE_NULL_COLUMNS = (
    "normalized_smiles", "molecular_formula", "precursor_error_ppm", "adduct",
    "precursor_mz", "num_peaks", "ms2_mzs", "ms2_normalized_intensities",
    "collision_energy_ev",
)
QUANTILES = (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)
PPM_THRESHOLDS = (5, 10, 20, 50)
BATCH_SIZE = 8192


def _count_true(values: pa.Array) -> int:
    return int(pc.sum(pc.fill_null(values, pa.scalar(False))).as_py() or 0)


def _distribution(values: array, *, ppm: bool = False) -> dict:
    """Compute exact linear-interpolation percentiles without sorting Python floats."""
    if not values:
        return {"finite_count": 0, "percentiles": None}
    data = np.frombuffer(values, dtype=np.float64)
    percentiles = np.quantile(data, QUANTILES).tolist()
    result = {
        "finite_count": len(values),
        "percentiles": {f"{q:g}": float(value) for q, value in zip(QUANTILES, percentiles)},
    }
    if ppm:
        absolute = np.abs(data)
        result["absolute_percentiles"] = {
            f"{q:g}": float(value)
            for q, value in zip(QUANTILES, np.quantile(absolute, QUANTILES).tolist())
        }
        result["absolute_outlier_counts_gt_ppm"] = {
            str(threshold): int(np.count_nonzero(absolute > threshold))
            for threshold in PPM_THRESHOLDS
        }
    return result


def _peak_quality(fields: dict[str, pa.Array], lengths: dict, precursor: list) -> dict[str, np.ndarray]:
    """Flag rows via flat Arrow child arrays; peak values never become Python lists."""
    count = len(precursor)
    flags = {
        name: np.zeros(count, dtype=np.bool_)
        for name in ("mz_above_precursor_plus_2_da", "negative_intensity", "nonfinite_intensity",
                     "intensity_above_one", "intensity_max_not_one")
    }

    def flattened(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sizes = np.fromiter((size or 0 for size in lengths[name]), dtype=np.int64, count=count)
        offsets = np.empty(count + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(sizes, out=offsets[1:])
        occupied = np.flatnonzero(sizes)
        children = pc.list_flatten(fields[name])
        values = cast(pa.Array, pc.fill_null(children, pa.scalar(float("nan")))).to_numpy(
            zero_copy_only=False
        )
        return values, occupied, offsets[occupied]

    mzs, rows, starts = flattened("ms2_mzs")
    if rows.size:
        maximum = np.maximum.reduceat(np.where(np.isfinite(mzs), mzs, -np.inf), starts)
        masses = np.asarray(precursor, dtype=np.float64)
        flags["mz_above_precursor_plus_2_da"][rows] = np.isfinite(masses[rows]) & (maximum > masses[rows] + 2)

    intensities, rows, starts = flattened("ms2_normalized_intensities")
    if rows.size:
        finite = np.isfinite(intensities)
        flags["negative_intensity"][rows] = np.logical_or.reduceat(finite & (intensities < 0), starts)
        flags["nonfinite_intensity"][rows] = np.logical_or.reduceat(~finite, starts)
        flags["intensity_above_one"][rows] = np.logical_or.reduceat(finite & (intensities > 1), starts)
        maximum = np.maximum.reduceat(np.where(finite, intensities, -np.inf), starts)
        flags["intensity_max_not_one"][rows] = np.isfinite(maximum) & (np.abs(maximum - 1) > 1e-6)
    return flags


@dataclass
class Slice:
    rows: int = 0
    categories: dict[str, Counter] = field(default_factory=lambda: {name: Counter() for name in CATEGORIES})
    numeric: dict[str, array] = field(default_factory=lambda: {name: array("d") for name in NUMERIC})
    integrity: Counter = field(default_factory=Counter)
    column_nulls: Counter = field(default_factory=Counter)
    peak_quality: Counter = field(default_factory=Counter)
    energy_lengths: Counter = field(default_factory=Counter)
    energy_exact_lengths: Counter = field(default_factory=Counter)
    base_peak_missing: int = 0
    formulas: set[str] = field(default_factory=set)

    def report(self, *, train: bool, columns: set[str]) -> dict:
        energy = self.energy_lengths
        report = {
            "rows": self.rows,
            "column_nulls": {name: self.column_nulls[name] for name in SOURCE_NULL_COLUMNS if name in columns},
            "categories": {name: dict(sorted(counts.items())) for name, counts in self.categories.items()},
            "distributions": {
                name: _distribution(values, ppm=name == "precursor_error_ppm")
                for name, values in self.numeric.items()
                if train or name != "precursor_error_ppm"
            },
            "array_length_integrity": dict(sorted(self.integrity.items())),
            "peak_quality": {
                **{name: self.peak_quality[name] for name in (
                    "empty_mz_arrays", "empty_intensity_arrays", "mz_above_precursor_plus_2_da",
                    "negative_intensity", "nonfinite_intensity", "intensity_above_one",
                    "intensity_max_not_one",
                )},
                "rules": "Empty excludes null arrays. m/z threshold requires finite precursor and finite peak. Intensity negative/>1 requires finite values; nonfinite includes null child values. Maximum test uses finite values only and excludes all-nonfinite or empty arrays.",
            },
            "collision_energy_ev": {
                "null_lists": energy["null"],
                "empty_lists": energy["empty"],
                "single_value_lists": energy["single"],
                "multiple_value_lists": energy["multiple"],
                "list_length_counts": dict(sorted((str(length), count) for length, count in self.energy_exact_lengths.items())),
                "null_rate": energy["null"] / self.rows if self.rows else None,
                "empty_rate": energy["empty"] / self.rows if self.rows else None,
                "single_value_rate": energy["single"] / self.rows if self.rows else None,
                "multiple_value_rate": energy["multiple"] / self.rows if self.rows else None,
            },
            "base_peak_intensity": {
                "missing_rows": self.base_peak_missing,
                "missing_rate": self.base_peak_missing / self.rows if self.rows else None,
            },
        }
        if train:
            report["distinct_observed_molecular_formulas"] = len(self.formulas)
        return report


class Profile:
    def __init__(self, path: Path, *, train: bool):
        self.path = path
        self.train = train
        self.parquet = pq.ParquetFile(path)
        self.schema = self.parquet.schema_arrow
        self.column_nulls = Counter()
        self.column_nonfinite = Counter()
        self.list_element_nulls = Counter()
        self.overall = Slice()
        self.sources: dict[str, Slice] = {}
        self.source_keys: dict[str, set[str]] = {}
        # A first non-null string and two ambiguity flags per distinct key; no
        # RDKit canonicalization or inferred equivalence is performed here.
        self.key_labels: dict[str, list] = {}
        self.missing_key_rows = 0

    def scan(self) -> None:
        columns = set(self.schema.names)
        for batch in self.parquet.iter_batches(batch_size=BATCH_SIZE, use_threads=False):
            fields = {name: batch.column(batch.schema.get_field_index(name)) for name in columns}
            for name, field_type in zip(batch.schema.names, batch.schema.types):
                values = fields[name]
                self.column_nulls[name] += values.null_count
                if pa.types.is_floating(field_type):
                    self.column_nonfinite[name] += _count_true(pc.invert(pc.is_finite(values)))
                elif pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
                    children = pc.list_flatten(values)
                    self.list_element_nulls[name] += children.null_count
                    if pa.types.is_floating(field_type.value_type):
                        self.column_nonfinite[name] += _count_true(pc.invert(pc.is_finite(children)))

            names = CATEGORIES + ("ingest_lib", "base_peak_intensity", "molecular_formula", "inchikey14", "normalized_smiles")
            scalars = {name: fields[name].to_pylist() for name in names if name in columns}
            scalars.update({name: fields[name].to_pylist() for name in NUMERIC if name in columns})
            lengths = {name: pc.list_value_length(fields[name]).to_pylist() for name in LISTS if name in columns}
            quality = _peak_quality(fields, lengths, scalars["precursor_mz"])
            for i in range(batch.num_rows):
                source = scalars["ingest_lib"][i] if self.train else None
                source_name = source if source is not None else "(null)"
                stats = self.overall
                if self.train:
                    stats = self.sources.get(source_name)
                    if stats is None:
                        stats = self.sources[source_name] = Slice()
                self._add_row(stats, scalars, lengths, quality, i)
                if self.train:
                    self._add_row(self.overall, scalars, lengths, quality, i)
                    key = scalars["inchikey14"][i]
                    if not key:
                        self.missing_key_rows += 1
                    else:
                        source_set = self.source_keys.get(source_name)
                        if source_set is None:
                            source_set = self.source_keys[source_name] = set()
                        source_set.add(key)
                        labels = self.key_labels.get(key)
                        if labels is None:
                            labels = self.key_labels[key] = [None, None, False, False]
                        for j, name in enumerate(("normalized_smiles", "molecular_formula")):
                            text = scalars[name][i]
                            if text is not None:
                                if labels[j] is None:
                                    labels[j] = text
                                elif labels[j] != text:
                                    labels[j + 2] = True

    @staticmethod
    def _add_row(stats: Slice, scalars: dict, lengths: dict, quality: dict, i: int) -> None:
        stats.rows += 1
        for name in SOURCE_NULL_COLUMNS:
            values = scalars if name in scalars else lengths
            if name in values and values[name][i] is None:
                stats.column_nulls[name] += 1
        for name in CATEGORIES:
            stats.categories[name][scalars[name][i] if scalars[name][i] is not None else "(null)"] += 1
        for name in NUMERIC:
            if name in scalars:
                value = scalars[name][i]
                if value is not None and math.isfinite(value):
                    stats.numeric[name].append(value)
        if "molecular_formula" in scalars and scalars["molecular_formula"][i] is not None:
            stats.formulas.add(scalars["molecular_formula"][i])
        if scalars["base_peak_intensity"][i] is None:
            stats.base_peak_missing += 1
        mz = lengths["ms2_mzs"][i]
        intensity = lengths["ms2_normalized_intensities"][i]
        if mz == 0:
            stats.peak_quality["empty_mz_arrays"] += 1
        if intensity == 0:
            stats.peak_quality["empty_intensity_arrays"] += 1
        for name, values in quality.items():
            if values[i]:
                stats.peak_quality[name] += 1
        peaks = scalars.get("num_peaks")
        if mz is None:
            stats.integrity["null_mz_arrays"] += 1
        if intensity is None:
            stats.integrity["null_intensity_arrays"] += 1
        if mz is not None and intensity is not None and mz != intensity:
            stats.integrity["length_mismatch_rows"] += 1
        if peaks is not None:
            count = peaks[i]
            if count is None:
                stats.integrity["null_num_peaks_rows"] += 1
            elif mz is not None and count != mz:
                stats.integrity["num_peaks_mismatch_rows"] += 1
        else:
            # Test has no num_peaks column: its aligned m/z array defines the peak count.
            if mz is not None:
                stats.numeric["num_peaks"].append(mz)
        energy = lengths["collision_energy_ev"][i]
        bucket = "null" if energy is None else "empty" if energy == 0 else "single" if energy == 1 else "multiple"
        stats.energy_lengths[bucket] += 1
        if energy is not None:
            stats.energy_exact_lengths[energy] += 1

    def report(self) -> dict:
        result = {
            "file": self.path.name,
            "rows": self.parquet.metadata.num_rows,
            "row_groups": self.parquet.num_row_groups,
            "schema": {field.name: str(field.type) for field in self.schema},
            "column_nulls": {name: self.column_nulls[name] for name in self.schema.names},
            "column_nonfinite_numeric": {
                name: self.column_nonfinite[name]
                for name in self.schema.names
                if name in NUMERIC or name in LISTS or name == "base_peak_intensity"
            },
            "list_element_nulls": {name: self.list_element_nulls[name] for name in LISTS if name in self.schema.names},
            "overall": self.overall.report(train=self.train, columns=set(self.schema.names)),
        }
        if self.train:
            keys = self.source_keys
            sources = sorted(self.sources)
            result["sources"] = {
                source: {**self.sources[source].report(train=True, columns=set(self.schema.names)), "distinct_inchikey14": len(keys.get(source, ()))}
                for source in sources
            }
            result["distinct_inchikey14"] = len(self.key_labels)
            result["key_identity_note"] = (
                "All distinct-key and source-overlap counts use the supplied inchikey14 literally. "
                "Official scoring applies RDKit tautomer normalization first, which can change connectivity keys; "
                "these supplied-key counts are NOT official-scoring identity counts."
            )
            result["missing_inchikey14_rows"] = self.missing_key_rows
            result["observed_same_key_ambiguity"] = {
                "smiles_keys": sum(entry[2] for entry in self.key_labels.values()),
                "formula_keys": sum(entry[3] for entry in self.key_labels.values()),
                "either_keys": sum(entry[2] or entry[3] for entry in self.key_labels.values()),
                "method": "Different non-null literal strings for one InChIKey14; not RDKit-canonical equivalence or chemical ground truth.",
            }
            result["cross_source_distinct_inchikey14_overlaps"] = {
                left: {right: len(keys.get(left, set()) & keys.get(right, set())) for right in sources if right > left}
                for left in sources
            }
            np_keys = keys.get("enveda-np-examples", set())
            result["enveda_np_examples_overlap"] = {
                "distinct_inchikey14": len(np_keys),
                "by_other_source": {source: len(np_keys & keys.get(source, set())) for source in sources if source != "enveda-np-examples"},
            }
        else:
            result["placeholder_warning"] = (
                "Downloaded test.parquet is an organizer-provided placeholder, NOT the hidden scoring distribution; "
                "do not tune thresholds or infer novelty-class proportions from these rows."
            )
        return result


def profile_paths(train: Path, test: Path) -> dict:
    profiles = []
    for path, is_train in ((train, True), (test, False)):
        profile = Profile(path, train=is_train)
        profile.scan()
        profiles.append(profile.report())
    return {
        "method": "Arrow streaming batches of 8192 rows; exact counts and finite-value linear-interpolation percentiles; test peak count from m/z array length; no raw keys, SMILES, or spectra in JSON.",
        "train": profiles[0],
        "test": profiles[1],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = profile_paths(args.train, args.test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
