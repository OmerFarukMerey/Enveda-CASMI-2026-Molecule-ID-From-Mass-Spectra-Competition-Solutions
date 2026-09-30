"""Build a mass-sorted structure catalog from frozen PubChem and COCONUT snapshots.

Inputs (downloaded, never committed):
  <pubchem-dir>/CID-Mass.gz, CID-InChI-Key.gz, CID-SMILES.gz, CID-PMID.gz
  <coconut>  COCONUT CSV export (zip or csv) containing a standard InChIKey column

Output (under ignored data/): numpy arrays aligned row-by-row and sorted by
monoisotopic mass: mass (float64), key14 (S14, first block of the standard
InChIKey), cid (uint32), pmid_count (uint32), coconut (bool), plus a JSON
manifest with input checksums and row accounting.

Only single-component structures (no '.' in the PubChem SMILES) with a mass,
an InChIKey and a SMILES are kept. The standard InChIKey first block is used
for candidate deduplication because running the official tautomer
canonicalization over the whole catalog is not feasible; the official metric is
applied later to the short candidate lists only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
from typing import Iterator, Literal, cast
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

BLOCK_SIZE = 64 << 20
PUBCHEM_FILES = {
    "mass": ("CID-Mass.gz", ["cid", "formula", "monoisotopic_mass", "exact_mass"]),
    "inchikey": ("CID-InChI-Key.gz", ["cid", "inchi", "inchikey"]),
    "smiles": ("CID-SMILES.gz", ["cid", "smiles"]),
    "pmid": ("CID-PMID.gz", ["cid", "pmid", "type"]),
}
ARRAYS = ("mass", "key14", "cid", "pmid_count", "coconut")
COCONUT_KEY_COLUMNS = {"standard_inchi_key", "standard_inchikey", "inchikey", "inchi_key"}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(
    path: Path, names: list[str], columns: list[str], types: dict[str, pa.DataType],
    skipped: list[int],
) -> Iterator[pa.RecordBatch]:
    """Stream a headerless, tab-separated PubChem file (optionally gzipped) in batches."""

    def skip(_row: pv.InvalidRow) -> Literal["skip"]:
        skipped[0] += 1
        return "skip"

    # open_csv decompresses by file extension (.gz).
    reader = pv.open_csv(
        str(path),
        read_options=pv.ReadOptions(column_names=names, block_size=BLOCK_SIZE),
        parse_options=pv.ParseOptions(delimiter="\t", quote_char=False, invalid_row_handler=skip),
        convert_options=pv.ConvertOptions(column_types=types, include_columns=columns),
    )
    yield from reader


def key14_array(keys: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """First 14 characters of InChIKeys as S14, plus a validity mask."""
    block = cast(pa.StringArray, pc.utf8_slice_codeunits(keys, 0, 14))
    length = cast(pa.Array, pc.utf8_length(block))
    valid_array = cast(pa.Array, pc.fill_null(cast(pa.Array, pc.equal(length, pa.scalar(14))), pa.scalar(False)))
    valid = valid_array.to_numpy(zero_copy_only=False)
    values = block.filter(valid_array).to_pylist()
    return np.asarray(values, dtype="S14"), valid


def _sorted_by_cid(cid: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Sort rows by CID if needed; the first row wins on a duplicated CID."""
    if cid.size and not np.all(cid[1:] > cid[:-1]):
        order = np.argsort(cid, kind="stable")
        cid = cid[order]
        values = tuple(value[order] for value in values)
        keep = np.ones(cid.size, dtype=bool)
        keep[1:] = cid[1:] != cid[:-1]
        cid = cid[keep]
        values = tuple(value[keep] for value in values)
    return (cid, *values)


def _lookup(base: np.ndarray, cid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions of base CIDs in a sorted CID array, and whether each was found."""
    if not cid.size:
        return np.zeros(base.size, dtype=np.int64), np.zeros(base.size, dtype=bool)
    index = np.minimum(np.searchsorted(cid, base), cid.size - 1)
    return index, cid[index] == base


def coconut_keys(path: Path) -> np.ndarray:
    """Sorted unique first InChIKey blocks from a COCONUT CSV export (zip or csv).

    Only the standard InChIKey column is parsed; the export has dozens of wide
    descriptor columns that are not needed.
    """
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"Expected one CSV in {path.name}, found {names}")
            data = archive.read(names[0])
    else:
        data = path.read_bytes()
    header = data[:data.find(b"\n")].decode().strip().split(",")
    candidates = [name for name in header if name.strip('"').lower() in COCONUT_KEY_COLUMNS]
    if len(candidates) != 1:
        raise ValueError(f"No unique standard InChIKey column in COCONUT export: {header}")
    column = candidates[0].strip('"')
    table = pv.read_csv(
        io.BytesIO(data),
        parse_options=pv.ParseOptions(newlines_in_values=True),
        convert_options=pv.ConvertOptions(include_columns=[column], column_types={column: pa.string()}),
    )
    keys, _ = key14_array(pc.drop_null(table[column]).combine_chunks())
    return np.unique(keys)


def build(pubchem_dir: Path, coconut: Path, output: Path) -> dict:
    started = perf_counter()
    if "data" not in output.resolve().parts:
        raise ValueError("Catalog output must be placed under data/")
    paths = {name: pubchem_dir / filename for name, (filename, _) in PUBCHEM_FILES.items()}
    missing = [str(path) for path in [*paths.values(), coconut] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing catalog inputs: {missing}")
    skipped = {name: [0] for name in PUBCHEM_FILES}
    rows: dict[str, int] = {}

    cids, masses = [], []
    for batch in read_tsv(paths["mass"], PUBCHEM_FILES["mass"][1], ["cid", "monoisotopic_mass"],
                          {"cid": pa.uint32(), "monoisotopic_mass": pa.float64()}, skipped["mass"]):
        cids.append(batch.column(0).to_numpy(zero_copy_only=False))
        masses.append(cast(pa.Array, pc.fill_null(batch.column(1), pa.scalar(float("nan")))).to_numpy(zero_copy_only=False))
    cid, mass = _sorted_by_cid(np.concatenate(cids), np.concatenate(masses))
    del cids, masses
    rows["mass"] = int(cid.size)

    key_cids, keys = [], []
    for batch in read_tsv(paths["inchikey"], PUBCHEM_FILES["inchikey"][1], ["cid", "inchikey"],
                          {"cid": pa.uint32(), "inchikey": pa.string()}, skipped["inchikey"]):
        block, valid = key14_array(batch.column(1))
        key_cids.append(batch.column(0).to_numpy(zero_copy_only=False)[valid])
        keys.append(block)
    key_cid, key14 = _sorted_by_cid(np.concatenate(key_cids), np.concatenate(keys))
    del key_cids, keys
    rows["inchikey"] = int(key_cid.size)

    smiles_cids, multi = [], []
    for batch in read_tsv(paths["smiles"], PUBCHEM_FILES["smiles"][1], ["cid", "smiles"],
                          {"cid": pa.uint32(), "smiles": pa.string()}, skipped["smiles"]):
        smiles_cids.append(batch.column(0).to_numpy(zero_copy_only=False))
        contains_dot = cast(pa.Array, pc.match_substring(batch.column(1), "."))
        flag = cast(pa.Array, pc.fill_null(contains_dot, pa.scalar(True)))
        multi.append(flag.to_numpy(zero_copy_only=False))
    smiles_cid, multi_component = _sorted_by_cid(np.concatenate(smiles_cids), np.concatenate(multi))
    del smiles_cids, multi
    rows["smiles"] = int(smiles_cid.size)

    pmid_counts = np.zeros(int(cid.max()) + 1 if cid.size else 1, dtype=np.uint32)
    pmid_rows = 0
    for batch in read_tsv(paths["pmid"], PUBCHEM_FILES["pmid"][1], ["cid"],
                          {"cid": pa.uint32()}, skipped["pmid"]):
        values = batch.column(0).to_numpy(zero_copy_only=False)
        unique, counts = np.unique(values[values < pmid_counts.size], return_counts=True)
        pmid_counts[unique] += counts.astype(np.uint32)
        pmid_rows += batch.num_rows
    rows["pmid"] = pmid_rows

    key_index, has_key = _lookup(cid, key_cid)
    smiles_index, has_smiles = _lookup(cid, smiles_cid)
    finite = np.isfinite(mass) & (mass > 0)
    is_multi = has_smiles & multi_component[smiles_index]
    keep = finite & has_key & has_smiles & ~is_multi
    accounting = {
        "mass_rows": int(cid.size),
        "missing_or_invalid_mass": int((~finite).sum()),
        "missing_inchikey": int((~has_key).sum()),
        "missing_smiles": int((~has_smiles).sum()),
        "multi_component": int(is_multi.sum()),
        "kept": int(keep.sum()),
    }
    cid, mass, key14 = cid[keep], mass[keep], key14[key_index[keep]]
    del key_cid, smiles_cid, multi_component, key_index, smiles_index, is_multi, keep
    pmid_count = pmid_counts[cid]
    del pmid_counts
    np_keys = coconut_keys(coconut)
    position = np.minimum(np.searchsorted(np_keys, key14), max(np_keys.size - 1, 0))
    in_coconut = np_keys[position] == key14 if np_keys.size else np.zeros(key14.size, dtype=bool)

    order = np.argsort(mass, kind="stable")
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"mass": mass, "key14": key14, "cid": cid, "pmid_count": pmid_count, "coconut": in_coconut}
    for name in ARRAYS:
        np.save(output / f"{name}.npy", arrays[name][order])
    manifest = {
        "format_version": 1,
        "inputs": {
            **{name: {"file": path.name, "bytes": path.stat().st_size, "md5": _md5(path)}
               for name, path in paths.items()},
            "coconut": {"file": coconut.name, "bytes": coconut.stat().st_size, "md5": _md5(coconut)},
        },
        "rows_read": rows,
        "rows_skipped_malformed": {name: count[0] for name, count in skipped.items()},
        "row_accounting": accounting,
        "coconut_distinct_key14": int(np_keys.size),
        "catalog_rows_in_coconut": int(in_coconut.sum()),
        "distinct_key14": int(np.unique(key14).size),
        "arrays": list(ARRAYS),
        "sorted_by": "mass",
        "identity": "first 14 characters of the PubChem standard InChIKey",
        "build_seconds": round(perf_counter() - started, 1),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pubchem-dir", type=Path, default=Path("data/external/pubchem-2026-09-27"))
    parser.add_argument("--coconut", type=Path, default=Path("data/external/coconut-2026-09/coconut_csv_lite-09-2026.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/catalog/pubchem-2026-09-27"))
    args = parser.parse_args()
    manifest = build(args.pubchem_dir, args.coconut, args.output)
    print(json.dumps({key: manifest[key] for key in ("rows_read", "rows_skipped_malformed", "row_accounting",
                                                    "distinct_key14", "catalog_rows_in_coconut",
                                                    "build_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
