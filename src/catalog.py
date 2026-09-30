"""Mass-sorted structure catalog built by scripts/build_catalog.py.

Shared by the catalog evaluation and the submission pipeline. No RDKit import:
structures are identified by the first block of PubChem's standard InChIKey.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

ARRAYS = ("mass", "key14", "cid", "pmid_count", "coconut")
PRIORS = ("cid", "pmid", "coconut_pmid")
BLOCK_SIZE = 64 << 20


def read_tsv(
    path: Path, names: list[str], columns: list[str], types: dict[str, pa.DataType],
    skipped: list[int],
) -> Iterator[pa.RecordBatch]:
    """Stream a headerless, tab-separated PubChem file (optionally gzipped) in batches.

    Malformed rows are skipped and counted in ``skipped[0]``.
    """

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


def load_catalog(directory: Path) -> dict[str, np.ndarray]:
    """Memory-map the catalog arrays after checking the manifest describes them."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("sorted_by") != "mass" or manifest.get("arrays") != list(ARRAYS):
        raise ValueError("Catalog manifest does not describe a mass-sorted catalog")
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
    if len({array.shape[0] for array in arrays.values()}) != 1:
        raise ValueError("Catalog arrays have different lengths")
    return arrays


def window(mass: np.ndarray, neutral_mass: float, tolerance_ppm: float) -> tuple[int, int]:
    """Row range [low, high) of catalog masses within ±tolerance_ppm (inclusive)."""
    delta = neutral_mass * tolerance_ppm / 1e6
    return (int(np.searchsorted(mass, neutral_mass - delta, "left")),
            int(np.searchsorted(mass, neutral_mass + delta, "right")))


def prior_order(cid: np.ndarray, pmid: np.ndarray, coconut: np.ndarray, prior: str) -> np.ndarray:
    """Row order under a fixed prior.

    cid: smallest CID first. pmid: most PubMed links first, then smallest CID.
    coconut_pmid: COCONUT natural products first, then as pmid.
    """
    if prior == "cid":
        return np.argsort(cid, kind="stable")
    if prior == "pmid":
        return np.lexsort((cid, -pmid.astype(np.int64)))
    if prior == "coconut_pmid":
        return np.lexsort((cid, -pmid.astype(np.int64), ~coconut.astype(bool)))
    raise ValueError(f"Unknown prior {prior!r}; expected one of {PRIORS}")


def rank_candidates(
    keys: np.ndarray, cid: np.ndarray, pmid: np.ndarray, coconut: np.ndarray,
    prior: str, limit: int,
) -> np.ndarray:
    """Row positions of the best row per distinct key, in prior order, at most limit.

    A key's rank is its best row's rank under the prior.
    """
    order = prior_order(cid, pmid, coconut, prior)
    _, first = np.unique(keys[order], return_index=True)
    return order[np.sort(first)][:limit]


def fetch_smiles(path: Path, cids: Iterable[int]) -> dict[int, str]:
    """PubChem SMILES for the requested CIDs from a CID-SMILES file (one stream pass)."""
    wanted = np.unique(np.fromiter(cids, dtype=np.uint32))
    found: dict[int, str] = {}
    if not wanted.size:
        return found
    value_set = pa.array(wanted, type=pa.uint32())
    for batch in read_tsv(path, ["cid", "smiles"], ["cid", "smiles"],
                          {"cid": pa.uint32(), "smiles": pa.string()}, [0]):
        selected = pc.is_in(batch.column(0), value_set=value_set)
        for cid_value, smiles in zip(batch.column(0).filter(selected).to_pylist(),
                                     batch.column(1).filter(selected).to_pylist()):
            if cid_value is not None and smiles is not None:
                found[cid_value] = smiles
    return found
