"""Build single-spectrum, connectivity-disjoint validation proxies from training data.

Rows are zero-based positions in the original parquet file. The JSON manifest is
sensitive competition-derived data: keep it under ignored data/, never in git.
"""

from __future__ import annotations

import argparse
from array import array
from collections.abc import Callable
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import NotRequired, TypedDict

import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


QUERY_SOURCE = "enveda-np-examples"
REFERENCE_SOURCE_PRIORITY = (
    "riken", "gnps", "massbank", "mona", "msdial",
    "pluskal_ms2", "drug_plus", "masaryk", "spectraverse", "enveda-180",
)
BATCH_SIZE = 8192
_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()


class SpectrumRow(TypedDict):
    row_index: int
    inchikey14: str
    canonical_inchikey14: str
    source: str


class Class1Pair(TypedDict):
    group: str
    canonical_inchikey14: str
    query: SpectrumRow
    reference: SpectrumRow
    identity_aliases: NotRequired[list[str]]


class HoldoutGroup(TypedDict):
    group: str
    query: SpectrumRow
    identity_aliases: list[str]




class ConnectivityGroups:
    """Union supplied and official tautomer-canonical connectivity labels."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        parent = self.parent
        parent.setdefault(key, key)
        node = key
        while parent[node] != node:
            node = parent[node]
        while key != node:
            old = parent[key]
            parent[key] = node
            key = old
        return node

    def join(self, supplied: str, canonical: str) -> None:
        left, right = self.find(supplied), self.find(canonical)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


@lru_cache(maxsize=500_000)
def _canonical_key(smiles: str) -> str | None:
    # Match the organizer's metric exactly: canonicalize before InChIKey14.
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = _TAUTOMER_ENUMERATOR.Canonicalize(mol)
        key = Chem.MolToInchiKey(canonical)[:14]
        return key if len(key) == 14 else None
    except (RuntimeError, ValueError, TypeError):
        return None


def _batches(parquet: pq.ParquetFile, columns: list[str]):
    for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns, use_threads=False):
        yield batch.to_pydict()


def _fingerprint(mzs: list[float] | None, intensities: list[float] | None) -> bytes | None:
    if mzs is None or intensities is None or len(mzs) != len(intensities):
        return None
    if any(value is None for value in mzs) or any(value is None for value in intensities):
        return None
    digest = hashlib.sha256()
    for values in (mzs, intensities):
        digest.update(len(values).to_bytes(8, "big"))
        digest.update(array("d", values).tobytes())
    return digest.digest()


def _measurement_fingerprints(parquet: pq.ParquetFile, indices: set[int]) -> dict[int, bytes | None]:
    """Read peak columns only, one small batch at a time, skipping irrelevant row groups."""
    wanted = sorted(indices)
    result: dict[int, bytes | None] = {}
    cursor = 0
    base = 0
    for group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        end = base + count
        if cursor == len(wanted) or wanted[cursor] >= end:
            base = end
            continue
        offset = base
        for batch in parquet.iter_batches(
            batch_size=1024,
            row_groups=[group],
            columns=["ms2_mzs", "ms2_normalized_intensities"],
            use_threads=False,
        ):
            batch_end = offset + batch.num_rows
            if cursor < len(wanted) and wanted[cursor] < batch_end:
                mzs = batch.column(0)
                intensities = batch.column(1)
                while cursor < len(wanted) and wanted[cursor] < batch_end:
                    row = wanted[cursor]
                    result[row] = _fingerprint(
                        mzs[row - offset].as_py(), intensities[row - offset].as_py()
                    )
                    cursor += 1
            offset = batch_end
            if cursor == len(wanted) or wanted[cursor] >= end:
                break
        base = end
    if cursor != len(wanted):
        raise ValueError("Requested peak row outside parquet bounds")
    return result


def _holdout_order(seed: int, group: str) -> tuple[bytes, str]:
    return hashlib.sha256(f"{seed}:{group}".encode("ascii")).digest(), group


def build_validation(
    train: Path, seed: int = 2026, query_sources: tuple[str, ...] = ("riken", "gnps")
) -> dict:
    if not query_sources or not all(query_sources) or len(query_sources) != len(set(query_sources)):
        raise ValueError("Query sources must be distinct nonempty labels")
    source_priority = {source: index for index, source in enumerate(query_sources)}
    parquet = pq.ParquetFile(train)
    required = {
        "ingest_lib", "inchikey14", "normalized_smiles",
        "ms2_mzs", "ms2_normalized_intensities",
    }
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    groups = ConnectivityGroups()
    query_rows: list[tuple[int, str, str, str]] = []
    # Per supplied identity, keep the best allowed query without retaining all rows.
    # Tie-break: source priority, canonical key, SMILES, then original row index.
    first_rows: dict[str, tuple[int, str, str, int, str]] = {}
    invalid_keys: set[str] = set()
    invalid_labels = unidentifiable_rows = 0
    offset = 0
    for batch in _batches(parquet, ["ingest_lib", "inchikey14", "normalized_smiles"]):
        for source, supplied, smiles in zip(
            batch["ingest_lib"], batch["inchikey14"], batch["normalized_smiles"]
        ):
            row = offset
            offset += 1
            if not supplied or not isinstance(supplied, str) or len(supplied) != 14:
                unidentifiable_rows += 1
                continue
            if not smiles or not isinstance(smiles, str):
                invalid_keys.add(supplied)
                invalid_labels += 1
                continue
            canonical = _canonical_key(smiles)
            if canonical is None:
                invalid_keys.add(supplied)
                invalid_labels += 1
                continue
            groups.join(supplied, canonical)
            if source in source_priority:
                choice = (source_priority[source], canonical, smiles, row, source)
                if supplied not in first_rows or choice < first_rows[supplied]:
                    first_rows[supplied] = choice
            if source == QUERY_SOURCE:
                query_rows.append((row, supplied, canonical, source))
    if offset != parquet.metadata.num_rows:
        raise ValueError("Row count changed while reading the training parquet")

    bad_groups = {groups.find(key) for key in invalid_keys}
    queries_by_group: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)
    for item in query_rows:
        root = groups.find(item[1])
        if root not in bad_groups:
            queries_by_group[root].append(item)
    eligible_canonical = {
        root: {item[2] for item in items} for root, items in queries_by_group.items()
    }

    # Only rows with a possible query identity enter the reference shortlist.
    # Exact peak comparisons below remove measurement copies even across sources.
    candidate_refs: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)
    offset = 0
    for batch in _batches(parquet, ["ingest_lib", "inchikey14", "normalized_smiles"]):
        for source, supplied, smiles in zip(
            batch["ingest_lib"], batch["inchikey14"], batch["normalized_smiles"]
        ):
            row = offset
            offset += 1
            if source == QUERY_SOURCE or not supplied or supplied not in groups.parent:
                continue
            root = groups.find(supplied)
            if root not in eligible_canonical or not smiles:
                continue
            canonical = _canonical_key(smiles)
            if canonical in eligible_canonical[root]:
                candidate_refs[root].append((row, supplied, canonical, source))

    shortlisted_queries = [item for root in candidate_refs for item in queries_by_group[root]]
    shortlisted_refs = [item for items in candidate_refs.values() for item in items]
    fingerprints = _measurement_fingerprints(
        parquet, {item[0] for item in shortlisted_queries + shortlisted_refs}
    )
    class1: list[Class1Pair] = []
    duplicate_pairs = invalid_peak_rows = 0
    reference_priority = {source: index for index, source in enumerate(REFERENCE_SOURCE_PRIORITY)}
    for root in sorted(candidate_refs):
        # Natural-product references first; Parquet row order is source-sorted and
        # would otherwise choose drug_plus for many of these query structures.
        candidate_refs[root].sort(
            key=lambda item: (reference_priority.get(item[3], len(reference_priority)), item[3], item[0])
        )
        refs_by_canonical: dict[str, list[tuple[int, str, str, str]]] = defaultdict(list)
        for item in candidate_refs[root]:
            refs_by_canonical[item[2]].append(item)
        for query in queries_by_group[root]:
            query_fingerprint = fingerprints[query[0]]
            if query_fingerprint is None:
                invalid_peak_rows += 1
                continue
            for ref in refs_by_canonical[query[2]]:
                reference_fingerprint = fingerprints[ref[0]]
                if reference_fingerprint is None:
                    invalid_peak_rows += 1
                    continue
                if query_fingerprint == reference_fingerprint:
                    duplicate_pairs += 1
                    continue
                class1.append({
                    "group": root,
                    "canonical_inchikey14": query[2],
                    "query": {
                        "row_index": query[0], "inchikey14": query[1],
                        "canonical_inchikey14": query[2], "source": query[3],
                    },
                    "reference": {
                        "row_index": ref[0], "inchikey14": ref[1],
                        "canonical_inchikey14": ref[2], "source": ref[3],
                    },
                })
                break
            if class1 and class1[-1]["group"] == root:
                break

    class1.sort(key=lambda item: item["group"])
    class1_groups = {entry["group"] for entry in class1}
    first_by_group: dict[str, tuple[int, str, str, str, int, str]] = {}
    for supplied, choice in first_rows.items():
        root = groups.find(supplied)
        group_choice = (choice[0], choice[1], supplied, choice[2], choice[3], choice[4])
        if root not in bad_groups and (
            root not in first_by_group or group_choice < first_by_group[root]
        ):
            first_by_group[root] = group_choice
    remaining = sorted(set(first_by_group) - class1_groups, key=lambda key: _holdout_order(seed, key))
    holdout_count = max(1, len(remaining) // 10) if remaining else 0
    heldout_groups = set(remaining[:holdout_count])

    aliases: dict[str, set[str]] = defaultdict(set)
    for key in groups.parent:
        root = groups.find(key)
        if root in heldout_groups or root in class1_groups:
            aliases[root].add(key)
    for entry in class1:
        entry["identity_aliases"] = sorted(aliases[entry["group"]])
    structure_holdout: list[HoldoutGroup] = [
        {
            "group": root,
            "query": {
                "row_index": first_by_group[root][4],
                "inchikey14": first_by_group[root][2],
                "canonical_inchikey14": first_by_group[root][1],
                "source": first_by_group[root][5],
            },
            "identity_aliases": sorted(aliases[root]),
        }
        for root in sorted(heldout_groups)
    ]
    manifest = {
        "format_version": 2,
        "input": {"filename": train.name, "rows": offset, "row_groups": parquet.num_row_groups},
        "seed": seed,
        "identity": "union of supplied inchikey14 and official RDKit tautomer-canonical InChIKey14",
        "structure_holdout_query_sources": list(query_sources),
        "class1_reference_source_priority": list(REFERENCE_SOURCE_PRIORITY),
        "structure_holdout_query_tiebreak": (
            "Requested source priority, canonical InChIKey14, supplied InChIKey14, "
            "normalized SMILES, then original row index."
        ),
        "class1_single_spectrum": class1,
        "structure_holdout": structure_holdout,
        "reference_policy": (
            "Exclude every identity_aliases group in structure_holdout, all invalid-label "
            "groups and unidentifiable rows. For each class1 group retain ONLY its "
            "explicit reference row, never another same-group row. Other valid "
            "non-holdout groups may contribute all rows."
        ),
        "supervised_training_policy": (
            "Exclude every structure_holdout identity_aliases group, all invalid-label "
            "groups and unidentifiable rows; for class1 groups retain ONLY their "
            "explicit reference row. Apply these filters before feature fitting, "
            "model selection, and external spectral indexing."
        ),
        "excluded_invalid_identity_aliases": sorted(
            key for key in groups.parent if groups.find(key) in bad_groups
        ),
        "counts": {
            "class1_query_keys": len(class1),
            "structure_holdout_groups": len(structure_holdout),
            "structure_holdout_identity_aliases": sum(len(aliases[root]) for root in heldout_groups),
            "structure_holdout_eligible_groups": len(remaining),
            "structure_holdout_query_source_counts": {
                source: sum(item["query"]["source"] == source for item in structure_holdout)
                for source in query_sources
            },
            "distinct_identity_groups": len({
                groups.find(key) for key in groups.parent if groups.find(key) not in bad_groups
            }),
            "rejected_exact_peak_pairs": duplicate_pairs,
            "invalid_peak_candidate_rows": invalid_peak_rows,
            "invalid_canonical_label_rows": invalid_labels,
            "unidentifiable_key_rows": unidentifiable_rows,
            "invalid_label_groups": len(bad_groups),
        },
        "limitations": [
            "Class 1 is one spectrum per group; source labels and nonidentical peak "
            "arrays do not prove different acquisition provenance or rule out near-duplicates.",
            "Structure holdout is an unseen-spectrum/structure retrieval proxy, NOT "
            "certified class 2 without a frozen, independently checked licensed catalog.",
            "Default RIKEN/GNPS selection enriches a natural-product-focused proxy "
            "but does not reproduce the hidden test distribution.",
            "This is NOT a class 3 novelty estimate without excluding held-out "
            "structures from generative supervision and a fixed candidate catalog "
            "and verifying provenance; no PubChem membership is claimed.",
            "Apply the group alias exclusions to external libraries and model-training "
            "corpora as well; those sources are not inspected by this manifest.",
        ],
    }
    check_manifest(manifest)
    return manifest


def reference_filter(manifest: dict) -> Callable[[int, str | None], bool]:
    """Compile the exact row eligibility rule for spectral reference and supervision.

    Call with the original zero-based parquet row index and supplied inchikey14.
    Apply the same identity exclusions to any external library separately.
    """
    excluded = set(manifest["excluded_invalid_identity_aliases"])
    excluded.update(
        alias for entry in manifest["structure_holdout"] for alias in entry["identity_aliases"]
    )
    class1_rows = {
        alias: entry["reference"]["row_index"]
        for entry in manifest["class1_single_spectrum"]
        for alias in entry["identity_aliases"]
    }
    total_rows = manifest["input"]["rows"]

    def eligible(row_index: int, supplied: str | None) -> bool:
        if row_index < 0 or row_index >= total_rows or not supplied or supplied in excluded:
            return False
        reference_index = class1_rows.get(supplied)
        return reference_index is None or row_index == reference_index

    return eligible


def check_manifest(manifest: dict) -> None:
    class1 = manifest["class1_single_spectrum"]
    holdout = manifest["structure_holdout"]
    query_groups = [item["group"] for item in class1]
    holdout_groups = [item["group"] for item in holdout]
    if len(query_groups) != len(set(query_groups)) or len(holdout_groups) != len(set(holdout_groups)):
        raise ValueError("Duplicate group in proxy manifest")
    if set(query_groups) & set(holdout_groups):
        raise ValueError("Class-1 and structure-holdout groups overlap")
    all_group_aliases = [set(item["identity_aliases"]) for item in class1 + holdout]
    if sum(map(len, all_group_aliases)) != len(set().union(*all_group_aliases)):
        raise ValueError("Identity alias appears in multiple proxy groups")
    excluded = {alias for item in holdout for alias in item["identity_aliases"]}
    excluded.update(manifest["excluded_invalid_identity_aliases"])
    query_rows = {item["query"]["row_index"] for item in class1}
    query_rows.update(item["query"]["row_index"] for item in holdout)
    if len(query_rows) != len(class1) + len(holdout):
        raise ValueError("Duplicate query rows in proxy manifest")
    for item in class1:
        query, reference = item["query"], item["reference"]
        if query["source"] != QUERY_SOURCE or reference["source"] == QUERY_SOURCE:
            raise ValueError("Class-1 reference must be from an independent source label")
        if query["canonical_inchikey14"] != reference["canonical_inchikey14"]:
            raise ValueError("Class-1 canonical identities do not match")
        if query["inchikey14"] not in item["identity_aliases"] or (
            reference["inchikey14"] not in item["identity_aliases"]
        ):
            raise ValueError("Class-1 group omits a query or reference identity")
        if query["row_index"] == reference["row_index"] or reference["row_index"] in query_rows:
            raise ValueError("Query row reused as reference")
        if query["inchikey14"] in excluded or reference["inchikey14"] in excluded:
            raise ValueError("Excluded structure in class-1 reference/query")
    for item in holdout:
        if item["query"]["inchikey14"] not in item["identity_aliases"] or (
            item["query"]["canonical_inchikey14"] not in item["identity_aliases"]
        ):
            raise ValueError("Held-out query missing from excluded identity aliases")
    if any(item["query"]["source"] not in manifest["structure_holdout_query_sources"] for item in holdout):
        raise ValueError("Structure-holdout query source not requested")
    counts = manifest["counts"]
    if counts["structure_holdout_eligible_groups"] < len(holdout) or (
        counts["structure_holdout_groups"] != len(holdout)
    ):
        raise ValueError("Structure-holdout group counts inconsistent")
    if counts["structure_holdout_query_source_counts"] != {
        source: sum(item["query"]["source"] == source for item in holdout)
        for source in manifest["structure_holdout_query_sources"]
    }:
        raise ValueError("Structure-holdout source counts inconsistent")
    eligible = reference_filter(manifest)
    for item in class1:
        if eligible(item["query"]["row_index"], item["query"]["inchikey14"]) or not eligible(
            item["reference"]["row_index"], item["reference"]["inchikey14"]
        ):
            raise ValueError("Class-1 query/reference eligibility violated")
    if any(eligible(item["query"]["row_index"], item["query"]["inchikey14"]) for item in holdout):
        raise ValueError("Structure-holdout query eligible for reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/analysis/validation.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--query-sources", default="riken,gnps",
        help="Comma-separated priority order for structure-holdout query sources",
    )
    parser.add_argument("--check", action="store_true", help="Rebuild and compare a saved manifest against the training parquet")
    args = parser.parse_args()
    query_sources = tuple(source.strip() for source in args.query_sources.split(","))
    if args.check:
        manifest = json.loads(args.output.read_text())
        check_manifest(manifest)
        if manifest != build_validation(args.train, args.seed, query_sources):
            raise ValueError("Manifest does not match training data and seed")
    else:
        manifest = build_validation(args.train, args.seed, query_sources)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
