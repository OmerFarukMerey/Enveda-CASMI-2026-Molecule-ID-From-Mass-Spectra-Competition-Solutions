"""Evaluate fixed class-1 spectral retrieval without materializing the training library.

The manifest is a private validation split, and all output belongs under ignored data/.
No holdout structure is a reference; this is not a class-2 or class-3 estimate.

Two class-1 reference policies are available. ``designated`` keeps only the one
reference row chosen by the manifest. ``library`` keeps every row of a class-1
structure from a source other than the query source, minus exact copies of the
query spectrum, which is closer to the library available at inference.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq

# Also allow `python scripts/evaluate_retrieval.py` from the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_validation import (
    check_manifest, class1_library_filter, peak_fingerprint, reference_filter,
)
from src.metric import canonicalize_smiles_to_ik14, evaluate
from src.library_search import (
    BATCH_SIZE, METADATA_COLUMNS, SPECTRUM_COLUMNS, gate_references, positive_float, selected_rows,
)
from src.spectral_search import PreparedSpectrum, prepare_spectrum, spectral_cosine


CLASS1_POLICIES = ("designated", "library")
NEAR_IDENTICAL_COSINE = 0.99
COSINE_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


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
    class1_references: str = "designated",
) -> dict:
    """Score every adduct/precursor-gated reference against each class-1 query."""
    started = perf_counter()
    if positive_float(precursor_tolerance_ppm) is None:
        raise ValueError("precursor_tolerance_ppm must be finite and positive")
    if positive_float(fragment_tolerance_da) is None:
        raise ValueError("fragment_tolerance_da must be finite and positive")
    _positive_int(max_peaks, "max_peaks")
    _positive_int(max_rank, "max_rank")
    if max_rank > 25:
        raise ValueError("max_rank must not exceed the official 25-candidate limit")
    if class1_references not in CLASS1_POLICIES:
        raise ValueError(f"class1_references must be one of {CLASS1_POLICIES}")

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
    designated_allowed = reference_filter(manifest)
    library_allowed = class1_library_filter(manifest)

    def allowed(index: int, supplied: str | None, source: str | None) -> bool:
        if class1_references == "library":
            return library_allowed(index, supplied, source)
        return designated_allowed(index, supplied)

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
                        positive_float(mz[local].as_py()))
            if index in query_rows:
                query_metadata[index] = metadata
            if index in reference_rows:
                reference_metadata[index] = metadata
        offset += batch.num_rows
    if offset != info["rows"] or len(query_metadata) != len(query_rows) or len(reference_metadata) != len(pairs):
        raise ValueError("Manifest rows missing from training parquet")

    fraction = precursor_tolerance_ppm / 1_000_000
    explicit_gated = sum(
        _passes_gate(
            query_metadata[pair["query"]["row_index"]],
            reference_metadata[pair["reference"]["row_index"]],
            fraction,
        )
        for pair in pairs
    )

    # Second metadata pass (shared core) retains only the row indices, sources
    # and ppm errors of gated references. No candidate is truncated before scoring.
    excluded_aliases = {
        alias for entry in manifest["structure_holdout"] for alias in entry["identity_aliases"]
    }
    excluded_holdout = 0

    def counted(index: int, supplied: str | None, source: str | None) -> bool:
        nonlocal excluded_holdout
        if supplied in excluded_aliases:
            excluded_holdout += 1
        return allowed(index, supplied, source)

    gated, eligible_count = gate_references(parquet, query_metadata, counted, precursor_tolerance_ppm)
    gated_pairs = sum(len(reference.matches) for reference in gated.values())

    # Verify every manifest identity against official chemistry, including
    # holdout queries and references that never pass the precursor gate. Their
    # SMILES are selected without loading any unneeded peak column.
    unchecked = set(expected_rows) - set(query_rows) - set(gated)
    for index, (smiles,) in selected_rows(parquet, unchecked, SPECTRUM_COLUMNS[:1]):
        key = canonicalize_smiles_to_ik14(smiles) if isinstance(smiles, str) else None
        if key != expected_rows[index]["canonical_inchikey14"]:
            raise ValueError(f"Manifest row {index} canonical identity differs from parquet")

    # Query spectra are the only spectra retained; all candidate peak arrays are
    # converted and released one selected row at a time in the final pass.
    query_spectra: dict[int, PreparedSpectrum] = {}
    query_fingerprints: dict[int, bytes | None] = {}
    solutions: dict[str, str] = {}
    for index, (smiles, mzs, intensities) in selected_rows(parquet, set(query_rows), SPECTRUM_COLUMNS):
        if not isinstance(smiles, str) or canonicalize_smiles_to_ik14(smiles) != query_rows[index]["query"]["canonical_inchikey14"]:
            raise ValueError(f"Manifest query row {index} canonical identity differs from parquet")
        solutions[str(index)] = smiles
        query_fingerprints[index] = peak_fingerprint(mzs, intensities)
        mz = query_metadata[index][1]
        if mz is not None and mzs is not None and intensities is not None:
            query_spectra[index] = prepare_spectrum(mz, mzs, intensities, max_peaks=max_peaks)

    # Score all gated references, then keep the best row for each official
    # tautomer-canonical connectivity key (not the supplied parquet label).
    # A same-structure reference whose peaks exactly copy the query is not an
    # independent measurement and is dropped before scoring.
    best: dict[int, dict[str, tuple[tuple[float, float, int, str], str]]] = defaultdict(dict)
    best_same_structure: dict[int, float] = {}
    exact_copy_pairs = 0
    for index, (smiles, mzs, intensities) in selected_rows(parquet, set(gated), SPECTRUM_COLUMNS):
        key = canonicalize_smiles_to_ik14(smiles) if isinstance(smiles, str) else None
        if index in reference_rows and key != reference_rows[index]["reference"]["canonical_inchikey14"]:
            raise ValueError(f"Manifest reference row {index} canonical identity differs from parquet")
        if key is None or not isinstance(smiles, str) or mzs is None or intensities is None:
            continue
        precursor, matched_queries = gated[index].precursor_mz, gated[index].matches
        reference_spectrum = prepare_spectrum(precursor, mzs, intensities, max_peaks=max_peaks)
        reference_fingerprint: bytes | None = None
        for query_index, ppm_error in matched_queries:
            spectrum = query_spectra.get(query_index)
            if spectrum is None:
                continue
            same_structure = key == query_rows[query_index]["query"]["canonical_inchikey14"]
            if same_structure:
                if reference_fingerprint is None:
                    reference_fingerprint = peak_fingerprint(mzs, intensities)
                query_fingerprint = query_fingerprints[query_index]
                if query_fingerprint is not None and query_fingerprint == reference_fingerprint:
                    exact_copy_pairs += 1
                    continue
            score = spectral_cosine(spectrum, reference_spectrum, tolerance_da=fragment_tolerance_da)
            if same_structure:
                best_same_structure[query_index] = max(score, best_same_structure.get(query_index, 0.0))
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
        query_index for reference in gated.values() for query_index, _ in reference.matches
    })
    reachable = [str(index) for index in best_same_structure]
    hit_sources: dict[str, int] = defaultdict(int)
    for index, candidates in best.items():
        answer = candidates.get(query_rows[index]["query"]["canonical_inchikey14"])
        if answer is not None and ranks.get(str(index), 0.0) > 0:
            hit_sources[str(gated[answer[0][2]].source)] += 1
    summary = {
        "class1_count": len(pairs),
        "class1_reference_policy": class1_references,
        "eligible_reference_rows": eligible_count,
        "holdout_excluded_rows": excluded_holdout,
        "gated_reference_rows": len(gated),
        "gated_reference_query_pairs": gated_pairs,
        "gated_query_count": gated_query_count,
        "gated_query_coverage": gated_query_count / len(pairs),
        "explicit_reference_gated_count": explicit_gated,
        "explicit_reference_gate_failures": len(pairs) - explicit_gated,
        "explicit_reference_gate_coverage": explicit_gated / len(pairs),
        # A query is reachable when at least one same-structure reference passed
        # the gate and was scored; this caps candidate recall.
        "reachable_query_count": len(reachable),
        "reachable_query_coverage": len(reachable) / len(pairs),
        "reachable_metrics": _metrics(reachable, ranks),
        "exact_copy_pairs_removed": exact_copy_pairs,
        # Possible near-duplicate measurements: identical labels do not prove
        # independent acquisition, so very high same-structure scores are flagged.
        "reachable_best_same_structure_cosine_ge_0_99": sum(
            score >= NEAR_IDENTICAL_COSINE for score in best_same_structure.values()
        ),
        "reachable_best_same_structure_cosine_quantiles": (
            {f"{q:g}": float(value) for q, value in zip(
                COSINE_QUANTILES, np.quantile(list(best_same_structure.values()), COSINE_QUANTILES)
            )}
            if best_same_structure else None
        ),
        "predicted_query_count": len(predictions),
        **{key: value for key, value in _metrics(list(solutions), ranks).items() if key != "count"},
        # Grouped by the manifest's designated reference source. Under the
        # library policy a hit can come from another source: see hit counts.
        "by_reference_source": {
            source: _metrics(ids, ranks) for source, ids in sorted(by_source.items())
        },
        "correct_hit_reference_source_counts": dict(sorted(hit_sources.items())),
        "parameters": {
            "precursor_tolerance_ppm": precursor_tolerance_ppm,
            "fragment_tolerance_da": fragment_tolerance_da,
            "max_peaks": max_peaks,
            "max_rank": max_rank,
            "class1_references": class1_references,
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
    parser.add_argument(
        "--class1-references", choices=CLASS1_POLICIES, default="designated",
        help="designated: the manifest's single reference row; library: all non-query-source rows",
    )
    args = parser.parse_args()
    summary = run(
        args.train, args.manifest, args.output,
        precursor_tolerance_ppm=args.precursor_tolerance_ppm,
        fragment_tolerance_da=args.fragment_tolerance_da,
        max_peaks=args.max_peaks, max_rank=args.max_rank,
        class1_references=args.class1_references,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
