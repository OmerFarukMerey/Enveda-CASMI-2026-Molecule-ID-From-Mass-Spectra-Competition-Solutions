"""Measure structure-catalog coverage and simple candidate ranking on the structure holdout.

For each held-out query spectrum, the neutral mass implied by its precursor m/z
and adduct selects catalog structures within a ppm window. Candidates are
deduplicated by the first standard InChIKey block and ranked by one of several
fixed priors (no spectral evidence). The top candidates' PubChem SMILES are then
scored with the official metric. Output holds aggregates only and belongs under
ignored data/.

Holdout answers are structures with public spectra and are likely better
studied than genuine class-2 molecules, so literature-based priors are
optimistic here. This is a structure-retrieval proxy, not a class-2 estimate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import isfinite
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_validation import check_manifest
from src.adducts import neutral_mass
from src.catalog import PRIORS, fetch_smiles, load_catalog, rank_candidates, window
from src.metric import evaluate

REPORT_TOLERANCES_PPM = (5.0, 10.0, 20.0)
LABEL_CONSISTENT_PPM = 20.0
MEMBERSHIP_CHUNK = 5_000_000
QUERY_COLUMNS = ["ingest_lib", "inchikey14", "adduct", "precursor_mz", "precursor_error_ppm", "normalized_smiles"]


def _metrics(ids: list[str], ranks: dict[str, float]) -> dict[str, float | int]:
    count = len(ids)
    return {
        "count": count,
        "candidate_recall_at_25": sum(ranks[i] > 0 for i in ids) / count if count else 0.0,
        "top1": sum(ranks[i] == 1.0 for i in ids) / count if count else 0.0,
        "mrr_at_25": sum(ranks[i] for i in ids) / count if count else 0.0,
    }


def _quantiles(values: list[int]) -> dict[str, float] | None:
    if not values:
        return None
    return {f"{q:g}": float(v) for q, v in zip((0.1, 0.5, 0.9, 1.0), np.quantile(values, (0.1, 0.5, 0.9, 1.0)))}


def run(
    train: Path, manifest_path: Path, catalog_dir: Path, smiles_path: Path, output: Path,
    *, tolerance_ppm: float = 10.0, max_rank: int = 25,
) -> dict:
    started = perf_counter()
    if not isfinite(tolerance_ppm) or tolerance_ppm <= 0:
        raise ValueError("tolerance_ppm must be finite and positive")
    if isinstance(max_rank, bool) or not isinstance(max_rank, int) or not 1 <= max_rank <= 25:
        raise ValueError("max_rank must be an integer in 1..25")
    if "data" not in output.resolve().parts:
        raise ValueError("Private evaluation output must be placed under data/")
    manifest = json.loads(manifest_path.read_text())
    check_manifest(manifest)
    parquet = pq.ParquetFile(train)
    info = manifest["input"]
    if (info["filename"], info["rows"], info["row_groups"]) != (
        train.name, parquet.metadata.num_rows, parquet.num_row_groups
    ):
        raise ValueError("Manifest does not match the training parquet layout")
    holdout = {entry["query"]["row_index"]: entry for entry in manifest["structure_holdout"]}
    if not holdout:
        raise ValueError("Manifest has no structure-holdout queries")

    # One pass over the training file; only the held-out rows are converted.
    queries: dict[int, dict] = {}
    offset = 0
    for batch in parquet.iter_batches(columns=QUERY_COLUMNS, batch_size=8192, use_threads=False):
        for local in range(batch.num_rows):
            entry = holdout.get(offset + local)
            if entry is None:
                continue
            row = {name: batch.column(i)[local].as_py() for i, name in enumerate(QUERY_COLUMNS)}
            if row["ingest_lib"] != entry["query"]["source"] or row["inchikey14"] != entry["query"]["inchikey14"]:
                raise ValueError(f"Manifest row {offset + local} source or identity differs from parquet")
            queries[offset + local] = row
        offset += batch.num_rows
    if len(queries) != len(holdout):
        raise ValueError("Structure-holdout rows missing from training parquet")

    catalog = load_catalog(catalog_dir)
    mass, keys = catalog["mass"], catalog["key14"]

    # Answer membership anywhere in the catalog, independent of mass.
    alias_of: dict[bytes, set[int]] = defaultdict(set)
    for row_index, entry in holdout.items():
        for alias in entry["identity_aliases"]:
            alias_of[alias.encode()].add(row_index)
    alias_array = np.array(sorted(alias_of), dtype="S14")
    answer_masses: dict[int, list[float]] = defaultdict(list)
    answer_coconut: set[int] = set()
    for start in range(0, keys.shape[0], MEMBERSHIP_CHUNK):
        chunk = np.asarray(keys[start:start + MEMBERSHIP_CHUNK])
        for position in np.flatnonzero(np.isin(chunk, alias_array)):
            for row_index in alias_of[bytes(chunk[position])]:
                answer_masses[row_index].append(float(mass[start + position]))
                if catalog["coconut"][start + position]:
                    answer_coconut.add(row_index)

    counts: dict[str, int] = defaultdict(int)
    window_sizes: list[int] = []
    min_answer_ppm: dict[int, float] = {}
    top_rows: dict[str, dict[int, np.ndarray]] = {prior: {} for prior in PRIORS}
    label_consistent: list[str] = []
    for row_index, row in queries.items():
        error = row["precursor_error_ppm"]
        if error is not None and isfinite(error) and abs(error) <= LABEL_CONSISTENT_PPM:
            label_consistent.append(str(row_index))
        try:
            neutral = neutral_mass(float(row["precursor_mz"]), row["adduct"])
        except (TypeError, ValueError):
            counts["unsupported_adduct_or_mass"] += 1
            continue
        if answer_masses.get(row_index):
            min_answer_ppm[row_index] = min(abs(m - neutral) / neutral * 1e6 for m in answer_masses[row_index])
        low, high = window(mass, neutral, tolerance_ppm)
        if high <= low:
            counts["empty_window"] += 1
            window_sizes.append(0)
            continue
        rows = {name: np.asarray(catalog[name][low:high]) for name in ("key14", "cid", "pmid_count", "coconut")}
        window_sizes.append(int(np.unique(rows["key14"]).size))
        for prior in PRIORS:
            chosen = rank_candidates(rows["key14"], rows["cid"], rows["pmid_count"], rows["coconut"], prior, max_rank)
            top_rows[prior][row_index] = rows["cid"][chosen]

    # Fetch SMILES only for chosen CIDs, then score every prior in one call so
    # the official canonicalization cache is shared.
    smiles_of = fetch_smiles(smiles_path, (int(c) for per_prior in top_rows.values()
                                           for cids in per_prior.values() for c in cids))
    solutions: dict[str, str] = {}
    predictions: dict[str, list[str]] = {}
    for prior in PRIORS:
        for row_index, row in queries.items():
            solutions[f"{prior}:{row_index}"] = row["normalized_smiles"]
        for row_index, cids in top_rows[prior].items():
            guesses = [smiles_of[int(c)] for c in cids if int(c) in smiles_of]
            if guesses:
                predictions[f"{prior}:{row_index}"] = guesses
    ranks = evaluate(solutions, predictions, max_rank=max_rank).reciprocal_ranks if predictions else {
        identifier: 0.0 for identifier in solutions}

    in_pubchem = [str(r) for r in queries if answer_masses.get(r)]
    in_window = [str(r) for r, ppm in min_answer_ppm.items() if ppm <= tolerance_ppm]
    by_source: dict[str, list[str]] = defaultdict(list)
    for row_index, row in queries.items():
        by_source[row["ingest_lib"]].append(str(row_index))
    subsets = {
        "all": [str(r) for r in queries],
        "answer_in_catalog": in_pubchem,
        "answer_in_window": in_window,
        "label_consistent_precursor": label_consistent,
        **{f"source_{source}": ids for source, ids in sorted(by_source.items())},
    }
    results = {
        prior: {
            name: _metrics(ids, {i: ranks[f"{prior}:{i}"] for i in ids})
            for name, ids in subsets.items()
        }
        for prior in PRIORS
    }
    summary = {
        "structure_holdout_queries": len(queries),
        "unsupported_adduct_or_mass": counts["unsupported_adduct_or_mass"],
        "empty_window": counts["empty_window"],
        "label_inconsistent_precursor_gt_20ppm_or_null": len(queries) - len(label_consistent),
        "answer_in_catalog": len(in_pubchem),
        "answer_not_in_catalog": len(queries) - len(in_pubchem),
        "answer_in_coconut": len(answer_coconut),
        "answer_in_window_at_ppm": {
            f"{t:g}": sum(ppm <= t for ppm in min_answer_ppm.values()) for t in REPORT_TOLERANCES_PPM
        },
        "window_distinct_structures_quantiles": _quantiles(window_sizes),
        "window_distinct_structures_le_25": sum(size <= 25 for size in window_sizes),
        "results_by_prior": results,
        "parameters": {"tolerance_ppm": tolerance_ppm, "max_rank": max_rank, "priors": list(PRIORS),
                       "label_consistent_ppm": LABEL_CONSISTENT_PPM},
        "catalog": json.loads((catalog_dir / "manifest.json").read_text())["inputs"],
        "runtime_seconds": round(perf_counter() - started, 1),
        "scope": ("Structure-holdout retrieval by mass window plus a fixed prior; no spectral "
                  "evidence. Not a certified class-2 estimate."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/analysis/validation.json"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/pubchem-2026-09-27"))
    parser.add_argument("--smiles", type=Path, default=Path("data/external/pubchem-2026-09-27/CID-SMILES.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/analysis/catalog_baseline.json"))
    parser.add_argument("--tolerance-ppm", type=float, default=10.0)
    args = parser.parse_args()
    summary = run(args.train, args.manifest, args.catalog, args.smiles, args.output, tolerance_ppm=args.tolerance_ppm)
    print(json.dumps({key: value for key, value in summary.items() if key != "catalog"}, sort_keys=True))


if __name__ == "__main__":
    main()
