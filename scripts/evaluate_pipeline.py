"""Score the exact submission pipeline on the fixed Stage A proxies.

Each class-1 query and structure-holdout query is treated as a one-spectrum
molecule. The library follows the class-1 "library" rule (every other-source
spectrum of a class-1 structure; all structure-holdout aliases and invalid
labels excluded), so holdout answers can only come from the pool's COCONUT part. Answers
are scored with the official metric. Output holds aggregates only and belongs
under ignored data/.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from time import perf_counter

import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_validation import check_manifest, class1_library_filter
from src.library_search import selected_rows
from src.metric import evaluate
from src.pipeline import QuerySpectrum, Settings, predict
from src.pool import load_pool

COLUMNS = ["ingest_lib", "inchikey14", "adduct", "precursor_mz", "ms2_mzs",
           "ms2_normalized_intensities", "normalized_smiles"]


def _metrics(ids: list[str], ranks: dict[str, float]) -> dict[str, float | int]:
    count = len(ids)
    return {
        "count": count,
        "candidate_recall_at_25": sum(ranks[i] > 0 for i in ids) / count if count else 0.0,
        "top1": sum(ranks[i] == 1.0 for i in ids) / count if count else 0.0,
        "mrr_at_25": sum(ranks[i] for i in ids) / count if count else 0.0,
    }


def run(train: Path, manifest_path: Path, pool_path: Path, output: Path) -> dict:
    started = perf_counter()
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
    proxies = {
        **{pair["query"]["row_index"]: ("class1", pair["query"]) for pair in manifest["class1_single_spectrum"]},
        **{entry["query"]["row_index"]: ("holdout", entry["query"]) for entry in manifest["structure_holdout"]},
    }
    queries: list[QuerySpectrum] = []
    solutions: dict[str, str] = {}
    subsets: dict[str, list[str]] = defaultdict(list)
    for row, values in selected_rows(parquet, set(proxies), COLUMNS):
        source, supplied, adduct, mz, mzs, intensities, answer = values
        proxy, expected = proxies[row]
        if source != expected["source"] or supplied != expected["inchikey14"]:
            raise ValueError(f"Manifest row {row} source or identity differs from parquet")
        molecule = str(row)
        queries.append(QuerySpectrum(molecule, adduct, mz, mzs, intensities))
        solutions[molecule] = answer
        subsets[proxy].append(molecule)
        subsets[f"{proxy}_source_{source}"].append(molecule)

    predictions, stats = predict(queries, train, pool_path, allowed=class1_library_filter(manifest))
    ranks = evaluate(solutions, predictions).reciprocal_ranks
    # Held-out answers can enter the pool only through COCONUT (their training
    # rows are excluded), so COCONUT membership bounds their reachability.
    pool = load_pool(pool_path)
    coconut_keys = set(pool.canonical14[pool.coconut].tolist()) | set(pool.key14[pool.coconut].tolist())
    holdout_in_coconut = sum(
        any(alias in coconut_keys for alias in entry["identity_aliases"]) for entry in manifest["structure_holdout"]
    )
    summary = {
        "results": {name: _metrics(ids, ranks) for name, ids in sorted(subsets.items())},
        "pipeline_stats": stats,
        "holdout_answers_in_pool_coconut": holdout_in_coconut,
        "settings": Settings().__repr__(),
        "library_rule": "class1_library_filter (holdout aliases and invalid labels excluded)",
        "runtime_seconds": round(perf_counter() - started, 1),
        "scope": "One-spectrum proxies; not a hidden-test estimate.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/analysis/validation.json"))
    parser.add_argument("--pool", type=Path, default=Path("data/pool/train-coconut-2026-09/pool.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/analysis/pipeline_proxy.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.train, args.manifest, args.pool, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
