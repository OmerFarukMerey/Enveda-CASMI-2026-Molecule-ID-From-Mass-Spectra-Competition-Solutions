"""Evaluate fixed class-1 spectral retrieval without materializing the training library.

The manifest is a private validation split, and all output belongs under ignored data/.
No holdout structure is a reference; this is not a class-2 or class-3 estimate.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
from math import isfinite, nextafter
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterator

import pyarrow.parquet as pq

# Also allow `python scripts/evaluate_retrieval.py` from the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_validation import check_manifest, reference_filter
from src.metric import canonicalize_smiles_to_ik14, evaluate
from src.spectral_search import PreparedSpectrum, prepare_spectrum, spectral_cosine


METADATA_COLUMNS = ["ingest_lib", "inchikey14", "adduct", "precursor_mz"]
SPECTRUM_COLUMNS = ["normalized_smiles", "ms2_mzs", "ms2_normalized_intensities"]
BATCH_SIZE = 8192
PEAK_BATCH_SIZE = 128


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if isfinite(number) and number > 0 else None


def _positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _selected_rows(
    parquet: pq.ParquetFile, indices: set[int], *, smiles_only: bool = False
) -> Iterator[tuple[int, str | None, list[float] | None, list[float] | None]]:
    """Yield only selected rows, respecting row-group-relative batch offsets.

    Arrow buffers contain at most one small batch. Only a selected row is
    converted to Python; unselected peaks and SMILES are never retained.
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
            row_groups=[group],
            columns=SPECTRUM_COLUMNS[:1] if smiles_only else SPECTRUM_COLUMNS,
            batch_size=PEAK_BATCH_SIZE, use_threads=False,
        ):
            batch_end = offset + batch.num_rows
            while position < len(wanted) and wanted[position] < batch_end:
                relative = wanted[position] - offset
                if smiles_only:
                    yield wanted[position], batch.column(0)[relative].as_py(), None, None
                else:
                    yield (wanted[position], batch.column(0)[relative].as_py(),
                           batch.column(1)[relative].as_py(), batch.column(2)[relative].as_py())
                position += 1
            offset = batch_end
            if position == len(wanted) or wanted[position] >= end:
                break
        base = end
    if position != len(wanted):
        raise ValueError("Selected row index outside parquet bounds")


def _passes_gate(
    query: tuple[str | None, float | None],
    reference: tuple[str | None, float | None],
    fraction: float,
) -> bool:
    query_adduct, query_mz = query
    reference_adduct, reference_mz = reference
    return (
        query_adduct is not None and query_adduct == reference_adduct
        and query_mz is not None and reference_mz is not None
        and abs(reference_mz - query_mz) <= fraction * query_mz
    )


def _metrics(ids: list[str], ranks: dict[str, float]) -> dict[str, float | int]:
    count = len(ids)
    return {
        "count": count,
        "candidate_recall_at_25": sum(ranks[identifier] > 0 for identifier in ids) / count if count else 0.0,
        "top1": sum(ranks[identifier] == 1.0 for identifier in ids) / count if count else 0.0,
        "mrr_at_25": sum(ranks[identifier] for identifier in ids) / count if count else 0.0,
    }


def run(
    train: Path,
    manifest_path: Path,
    output: Path,
    *,
    precursor_tolerance_ppm: float = 20.0,
    fragment_tolerance_da: float = 0.02,
    max_peaks: int = 100,
    max_rank: int = 25,
) -> dict:
    """Score every adduct/precursor-gated reference against each class-1 query."""
    started = perf_counter()
    if _positive_float(precursor_tolerance_ppm) is None:
        raise ValueError("precursor_tolerance_ppm must be finite and positive")
    if _positive_float(fragment_tolerance_da) is None:
        raise ValueError("fragment_tolerance_da must be finite and positive")
    _positive_int(max_peaks, "max_peaks")
    _positive_int(max_rank, "max_rank")
    if max_rank > 25:
        raise ValueError("max_rank must not exceed the official 25-candidate limit")

    train, manifest_path, output = Path(train), Path(manifest_path), Path(output)
    if "data" not in output.resolve().parts:
        raise ValueError("Private evaluation output must be placed under data/")
    manifest = json.loads(manifest_path.read_text())
    check_manifest(manifest)
    info = manifest["input"]
    if info["filename"] != train.name:
        raise ValueError("Manifest filename does not match training parquet")
    parquet = pq.ParquetFile(train)
    if info["rows"] != parquet.metadata.num_rows or info["row_groups"] != parquet.num_row_groups:
        raise ValueError("Manifest row count or row groups do not match training parquet")
    missing = set(METADATA_COLUMNS + SPECTRUM_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    pairs = manifest["class1_single_spectrum"]
    if not pairs:
        raise ValueError("Manifest has no class-1 queries to evaluate")
    expected_rows = {
        item["row_index"]: item
        for pair in pairs for item in (pair["query"], pair["reference"])
    }
    expected_rows.update(
        (entry["query"]["row_index"], entry["query"])
        for entry in manifest["structure_holdout"]
    )
    if any(not isinstance(index, int) or index < 0 or index >= info["rows"]
           for index in expected_rows):
        raise ValueError("Manifest row index outside training parquet")
    query_rows = {pair["query"]["row_index"]: pair for pair in pairs}
    reference_rows = {pair["reference"]["row_index"]: pair for pair in pairs}
    query_metadata: dict[int, tuple[str | None, float | None]] = {}
    reference_metadata: dict[int, tuple[str | None, float | None]] = {}
    allowed = reference_filter(manifest)

    # First pass reads only scalar metadata, never peak arrays or SMILES.
    offset = 0
    for batch in parquet.iter_batches(columns=METADATA_COLUMNS, batch_size=BATCH_SIZE, use_threads=False):
        source, supplied, adduct, mz = (batch.column(i) for i in range(4))
        for local in range(batch.num_rows):
            index = offset + local
            expected = expected_rows.get(index)
            if expected is None:
                continue
            if source[local].as_py() != expected["source"] or supplied[local].as_py() != expected["inchikey14"]:
                raise ValueError(f"Manifest row {index} source or identity differs from training parquet")
            label = adduct[local].as_py()
            metadata = (label if isinstance(label, str) and label else None,
                        _positive_float(mz[local].as_py()))
            if index in query_rows:
                query_metadata[index] = metadata
            if index in reference_rows:
                reference_metadata[index] = metadata
        offset += batch.num_rows
    if offset != info["rows"] or len(query_metadata) != len(query_rows) or len(reference_metadata) != len(pairs):
        raise ValueError("Manifest rows missing from training parquet")

    fraction = precursor_tolerance_ppm / 1_000_000
    by_adduct: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for index, (adduct, mz) in query_metadata.items():
        if adduct is not None and mz is not None:
            by_adduct[adduct].append((mz, index))
    for queries in by_adduct.values():
        queries.sort()
    query_masses = {adduct: [mz for mz, _ in queries] for adduct, queries in by_adduct.items()}
    explicit_gated = sum(
        _passes_gate(
            query_metadata[pair["query"]["row_index"]],
            reference_metadata[pair["reference"]["row_index"]],
            fraction,
        )
        for pair in pairs
    )

    # Second metadata pass retains only the row indices and ppm errors of gated
    # references. No candidate is truncated before its score is known.
    gated: dict[int, tuple[float, list[tuple[int, float]]]] = {}
    eligible_count = excluded_holdout = gated_pairs = 0
    excluded_aliases = {
        alias for entry in manifest["structure_holdout"] for alias in entry["identity_aliases"]
    }
    offset = 0
    for batch in parquet.iter_batches(columns=METADATA_COLUMNS[1:], batch_size=BATCH_SIZE, use_threads=False):
        supplied, adducts, masses = (batch.column(i) for i in range(3))
        for local in range(batch.num_rows):
            index = offset + local
            supplied_key = supplied[local].as_py()
            if supplied_key in excluded_aliases:
                excluded_holdout += 1
            if not allowed(index, supplied_key):
                continue
            eligible_count += 1
            label = adducts[local].as_py()
            queries = by_adduct.get(label) if isinstance(label, str) else None
            mz = _positive_float(masses[local].as_py())
            if not queries or mz is None:
                continue
            query_values = query_masses[label]
            first = bisect_left(query_values, nextafter(mz / (1 + fraction), -float("inf")))
            last = (
                bisect_right(query_values, nextafter(mz / (1 - fraction), float("inf")))
                if fraction < 1 else len(queries)
            )
            matches = [
                (query_index, abs(mz - query_mz) / query_mz * 1_000_000)
                for query_mz, query_index in queries[first:last]
                if abs(mz - query_mz) <= fraction * query_mz
            ]
            if matches:
                gated[index] = mz, matches
                gated_pairs += len(matches)
        offset += batch.num_rows
    if offset != info["rows"]:
        raise ValueError("Training parquet changed during evaluation")

    # Verify every manifest identity against official chemistry, including
    # holdout queries and references that never pass the precursor gate. Their
    # SMILES are selected without loading any unneeded peak column.
    unchecked = set(expected_rows) - set(query_rows) - set(gated)
    for index, smiles, _, _ in _selected_rows(parquet, unchecked, smiles_only=True):
        key = canonicalize_smiles_to_ik14(smiles) if isinstance(smiles, str) else None
        if key != expected_rows[index]["canonical_inchikey14"]:
            raise ValueError(f"Manifest row {index} canonical identity differs from parquet")

    # Query spectra are the only spectra retained; all candidate peak arrays are
    # converted and released one selected row at a time in the final pass.
    query_spectra: dict[int, PreparedSpectrum] = {}
    solutions: dict[str, str] = {}
    for index, smiles, mzs, intensities in _selected_rows(parquet, set(query_rows)):
        if not isinstance(smiles, str) or canonicalize_smiles_to_ik14(smiles) != query_rows[index]["query"]["canonical_inchikey14"]:
            raise ValueError(f"Manifest query row {index} canonical identity differs from parquet")
        solutions[str(index)] = smiles
        mz = query_metadata[index][1]
        if mz is not None and mzs is not None and intensities is not None:
            query_spectra[index] = prepare_spectrum(mz, mzs, intensities, max_peaks=max_peaks)

    # Score all gated references, then keep the best row for each official
    # tautomer-canonical connectivity key (not the supplied parquet label).
    best: dict[int, dict[str, tuple[tuple[float, float, int, str], str]]] = defaultdict(dict)
    for index, smiles, mzs, intensities in _selected_rows(parquet, set(gated)):
        key = canonicalize_smiles_to_ik14(smiles) if isinstance(smiles, str) else None
        if index in reference_rows and key != reference_rows[index]["reference"]["canonical_inchikey14"]:
            raise ValueError(f"Manifest reference row {index} canonical identity differs from parquet")
        if key is None or not isinstance(smiles, str) or mzs is None or intensities is None:
            continue
        precursor, matched_queries = gated[index]
        reference_spectrum = prepare_spectrum(precursor, mzs, intensities, max_peaks=max_peaks)
        for query_index, ppm_error in matched_queries:
            spectrum = query_spectra.get(query_index)
            if spectrum is None:
                continue
            score = spectral_cosine(spectrum, reference_spectrum, tolerance_da=fragment_tolerance_da)
            ordering = (-score, ppm_error, index, key)
            previous = best[query_index].get(key)
            if previous is None or ordering < previous[0]:
                best[query_index][key] = ordering, smiles

    predictions: dict[str, list[str]] = {}
    for index, candidates in best.items():
        ranked = sorted(candidates.values(), key=lambda candidate: candidate[0])[:max_rank]
        if ranked:
            predictions[str(index)] = [smiles for _, smiles in ranked]
    if predictions:
        result = evaluate(solutions, predictions, max_rank=max_rank)
        ranks = result.reciprocal_ranks
    else:
        ranks = {identifier: 0.0 for identifier in solutions}
    by_source: dict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        by_source[pair["reference"]["source"]].append(str(pair["query"]["row_index"]))
    for source in ("gnps", "riken", "massbank"):
        by_source[source]
    gated_query_count = len({
        query_index for _, matches in gated.values() for query_index, _ in matches
    })
    summary = {
        "class1_count": len(pairs),
        "eligible_reference_rows": eligible_count,
        "holdout_excluded_rows": excluded_holdout,
        "gated_reference_rows": len(gated),
        "gated_reference_query_pairs": gated_pairs,
        "gated_query_count": gated_query_count,
        "gated_query_coverage": gated_query_count / len(pairs),
        "explicit_reference_gated_count": explicit_gated,
        "explicit_reference_gate_failures": len(pairs) - explicit_gated,
        "explicit_reference_gate_coverage": explicit_gated / len(pairs),
        "predicted_query_count": len(predictions),
        **{key: value for key, value in _metrics(list(solutions), ranks).items() if key != "count"},
        "by_reference_source": {
            source: _metrics(ids, ranks) for source, ids in sorted(by_source.items())
        },
        "parameters": {
            "precursor_tolerance_ppm": precursor_tolerance_ppm,
            "fragment_tolerance_da": fragment_tolerance_da,
            "max_peaks": max_peaks,
            "max_rank": max_rank,
        },
        "runtime_seconds": round(perf_counter() - started, 3),
        "scope": "Class-1 single-spectrum retrieval only; holdout aliases excluded, no class-2/3 estimate.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/analysis/validation.json"))
    parser.add_argument("--output", type=Path, default=Path("data/analysis/retrieval_baseline.json"))
    parser.add_argument("--precursor-tolerance-ppm", type=float, default=20.0)
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    parser.add_argument("--max-peaks", type=int, default=100)
    parser.add_argument("--max-rank", type=int, default=25)
    args = parser.parse_args()
    summary = run(
        args.train, args.manifest, args.output,
        precursor_tolerance_ppm=args.precursor_tolerance_ppm,
        fragment_tolerance_da=args.fragment_tolerance_da,
        max_peaks=args.max_peaks, max_rank=args.max_rank,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
