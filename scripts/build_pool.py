"""Build the candidate structure pool: training-label structures ∪ COCONUT.

One row per standard InChIKey first block (training `inchikey14`, COCONUT
`standard_inchi_key`), with a SMILES (training label preferred), the RDKit
monoisotopic mass, the official tautomer-canonical InChIKey14 used by the
metric, and membership flags. Multi-component SMILES ('.') and unparseable
structures are dropped. RDKit runs here, locally, so inference needs no RDKit.

Which training structures are usable at inference is decided at run time by
the library eligibility rule (src/pipeline.py), so held-out proxy structures can
enter the pool only through COCONUT.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter
import zipfile

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pool import POOL_COLUMNS

CHUNK = 2000


def _init_worker() -> None:
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")


def _describe(smiles_list: list[str]) -> list[tuple[float | None, str | None]]:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    from src.metric import canonicalize_smiles_to_ik14

    result: list[tuple[float | None, str | None]] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result.append((None, None))
            continue
        result.append((rdMolDescriptors.CalcExactMolWt(mol), canonicalize_smiles_to_ik14(smiles)))
    return result


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def training_structures(train: Path) -> dict[str, str]:
    """First non-empty SMILES per training inchikey14, in file order."""
    structures: dict[str, str] = {}
    parquet = pq.ParquetFile(train)
    for batch in parquet.iter_batches(columns=["inchikey14", "normalized_smiles"], batch_size=65536, use_threads=False):
        for key, smiles in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if key and len(key) == 14 and smiles and key not in structures:
                structures[key] = smiles
    return structures


def coconut_structures(path: Path) -> dict[str, str]:
    """Canonical SMILES per COCONUT standard InChIKey first block (first row wins)."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"Expected one CSV in {path.name}, found {names}")
            data = archive.read(names[0])
    else:
        data = path.read_bytes()
    table = pv.read_csv(
        io.BytesIO(data),
        parse_options=pv.ParseOptions(newlines_in_values=True),
        convert_options=pv.ConvertOptions(
            include_columns=["standard_inchi_key", "canonical_smiles"],
            column_types={"standard_inchi_key": pa.string(), "canonical_smiles": pa.string()},
        ),
    )
    structures: dict[str, str] = {}
    for key, smiles in zip(table["standard_inchi_key"].to_pylist(), table["canonical_smiles"].to_pylist()):
        if key and len(key) >= 14 and smiles and key[:14] not in structures:
            structures[key[:14]] = smiles
    return structures


def build(train: Path, coconut: Path, output: Path, workers: int | None = None) -> dict:
    started = perf_counter()
    if "data" not in output.resolve().parts:
        raise ValueError("Pool output must be placed under data/")
    from_train = training_structures(train)
    from_coconut = coconut_structures(coconut)
    keys = sorted(from_train.keys() | from_coconut.keys())
    smiles = [from_train.get(key) or from_coconut[key] for key in keys]
    multi = [("." in s) for s in smiles]
    candidates = [s for s, is_multi in zip(smiles, multi) if not is_multi]
    chunks = [candidates[i:i + CHUNK] for i in range(0, len(candidates), CHUNK)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        described = [item for chunk in pool.map(_describe, chunks) for item in chunk]

    rows = {name: [] for name in POOL_COLUMNS}
    position = 0
    unparseable = 0
    for key, structure, is_multi in zip(keys, smiles, multi):
        if is_multi:
            continue
        mass, canonical = described[position]
        position += 1
        if mass is None or canonical is None:
            unparseable += 1
            continue
        rows["mass"].append(mass)
        rows["key14"].append(key)
        rows["canonical14"].append(canonical)
        rows["smiles"].append(structure)
        rows["in_train"].append(key in from_train)
        rows["coconut"].append(key in from_coconut)
    table = pa.table(rows).sort_by("mass")
    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output / "pool.parquet", compression="zstd")
    manifest = {
        "format_version": 1,
        "inputs": {
            "train": {"file": train.name, "bytes": train.stat().st_size},
            "coconut": {"file": coconut.name, "bytes": coconut.stat().st_size, "md5": _md5(coconut)},
        },
        "counts": {
            "training_structures": len(from_train),
            "coconut_structures": len(from_coconut),
            "union": len(keys),
            "multi_component_dropped": sum(multi),
            "unparseable_dropped": unparseable,
            "rows": table.num_rows,
            "rows_in_train": sum(rows["in_train"]),
            "rows_in_coconut": sum(rows["coconut"]),
            "rows_in_both": sum(t and c for t, c in zip(rows["in_train"], rows["coconut"])),
            "distinct_canonical14": len(set(rows["canonical14"])),
        },
        "sorted_by": "mass",
        "mass": "RDKit Descriptors.ExactMolWt (monoisotopic)",
        "identity": "key14: standard InChIKey first block; canonical14: official metric key",
        "build_seconds": round(perf_counter() - started, 1),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.parquet"))
    parser.add_argument("--coconut", type=Path, default=Path("data/external/coconut-2026-09/coconut_csv_lite-09-2026.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/pool/train-coconut-2026-09"))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    manifest = build(args.train, args.coconut, args.output, args.workers)
    print(json.dumps(manifest["counts"] | {"build_seconds": manifest["build_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
