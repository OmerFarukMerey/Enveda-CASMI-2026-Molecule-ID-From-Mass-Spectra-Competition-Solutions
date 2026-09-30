"""Adduct- and precursor-gated reference selection over the training parquet.

Shared by the fixed-fold evaluation (scripts/evaluate_retrieval.py) and the
submission pipeline (src/pipeline.py). Nothing here imports RDKit, so it runs in
the Kaggle image, which does not ship it.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite, nextafter
from typing import Any, Callable, Iterator, Mapping

import pyarrow.parquet as pq

METADATA_COLUMNS = ["ingest_lib", "inchikey14", "adduct", "precursor_mz"]
SPECTRUM_COLUMNS = ["normalized_smiles", "ms2_mzs", "ms2_normalized_intensities"]
BATCH_SIZE = 8192
PEAK_BATCH_SIZE = 128


def positive_float(value: object) -> float | None:
    """A finite positive float, or None for anything else (including bools)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if isfinite(number) and number > 0 else None


def selected_rows(
    parquet: pq.ParquetFile, indices: set[int], columns: list[str]
) -> Iterator[tuple[int, list[Any]]]:
    """Yield (row index, column values) for selected rows only, in row order.

    Arrow buffers hold at most one small batch; only selected rows are converted
    to Python, so unselected peak arrays and SMILES are never retained.
    """
    wanted = sorted(indices)
    position = 0
    base = 0
    for group in range(parquet.num_row_groups):
        end = base + parquet.metadata.row_group(group).num_rows
        if position == len(wanted) or wanted[position] >= end:
            base = end
            continue
        offset = base
        for batch in parquet.iter_batches(
            row_groups=[group], columns=columns, batch_size=PEAK_BATCH_SIZE, use_threads=False,
        ):
            batch_end = offset + batch.num_rows
            while position < len(wanted) and wanted[position] < batch_end:
                relative = wanted[position] - offset
                yield wanted[position], [batch.column(i)[relative].as_py() for i in range(len(columns))]
                position += 1
            offset = batch_end
            if position == len(wanted) or wanted[position] >= end:
                break
        base = end
    if position != len(wanted):
        raise ValueError("Selected row index outside parquet bounds")


@dataclass(frozen=True, slots=True)
class GatedReference:
    """A library row that passed the gate for at least one query."""

    precursor_mz: float
    source: str | None
    supplied_key: str | None
    matches: tuple[tuple[int, float], ...]  # (query id, ppm error)


def gate_references(
    parquet: pq.ParquetFile,
    queries: Mapping[int, tuple[str | None, float | None]],
    allowed: Callable[[int, str | None, str | None], bool],
    tolerance_ppm: float,
    key_counts: Counter[str] | None = None,
) -> tuple[dict[int, GatedReference], int]:
    """Find library rows with the same adduct label and precursor within tolerance.

    ``queries`` maps a query id to its (adduct, precursor m/z). ``allowed`` is
    called with (row index, supplied inchikey14, source) and decides library
    eligibility. The ppm error is relative to the query precursor, and the
    boundary is inclusive. If ``key_counts`` is given, it is incremented once
    per eligible row with a supplied key. Returns the gated rows and the number
    of eligible rows.
    """
    fraction = tolerance_ppm / 1_000_000
    by_adduct: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for query_id, (adduct, mz) in queries.items():
        if adduct and mz is not None:
            by_adduct[adduct].append((mz, query_id))
    for items in by_adduct.values():
        items.sort()
    masses = {adduct: [mz for mz, _ in items] for adduct, items in by_adduct.items()}

    gated: dict[int, GatedReference] = {}
    eligible = 0
    offset = 0
    for batch in parquet.iter_batches(columns=METADATA_COLUMNS, batch_size=BATCH_SIZE, use_threads=False):
        sources, supplied, adducts, precursors = (batch.column(i) for i in range(4))
        for local in range(batch.num_rows):
            index = offset + local
            supplied_key = supplied[local].as_py()
            source = sources[local].as_py()
            if not allowed(index, supplied_key, source):
                continue
            eligible += 1
            if key_counts is not None and supplied_key:
                key_counts[supplied_key] += 1
            label = adducts[local].as_py()
            items = by_adduct.get(label) if isinstance(label, str) else None
            mz = positive_float(precursors[local].as_py())
            if not items or mz is None:
                continue
            values = masses[label]
            first = bisect_left(values, nextafter(mz / (1 + fraction), -float("inf")))
            last = bisect_right(values, nextafter(mz / (1 - fraction), float("inf"))) if fraction < 1 else len(items)
            matches = tuple(
                (query_id, abs(mz - query_mz) / query_mz * 1_000_000)
                for query_mz, query_id in items[first:last]
                if abs(mz - query_mz) <= fraction * query_mz
            )
            if matches:
                gated[index] = GatedReference(mz, source, supplied_key, matches)
        offset += batch.num_rows
    if offset != parquet.metadata.num_rows:
        raise ValueError("Training parquet changed during library gating")
    return gated, eligible
