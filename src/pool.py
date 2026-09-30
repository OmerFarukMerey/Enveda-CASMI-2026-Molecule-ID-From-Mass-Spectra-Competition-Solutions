"""Candidate structure pool built by scripts/build_pool.py (training ∪ COCONUT).

No RDKit import: masses and official keys are precomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

POOL_COLUMNS = ("mass", "key14", "canonical14", "smiles", "in_train", "coconut")


@dataclass(frozen=True)
class Pool:
    mass: np.ndarray         # float64, ascending
    key14: np.ndarray        # standard InChIKey first block (<U14)
    canonical14: np.ndarray  # official metric key (<U14)
    smiles: np.ndarray       # str objects
    in_train: np.ndarray     # bool
    coconut: np.ndarray      # bool

    def canonical_of_key(self) -> dict[str, str]:
        """Official key for each training structure's standard key."""
        rows = np.flatnonzero(self.in_train)
        return dict(zip(self.key14[rows].tolist(), self.canonical14[rows].tolist()))


def load_pool(path: Path) -> Pool:
    table = pq.read_table(path, columns=list(POOL_COLUMNS))
    arrays = {name: table.column(name).to_numpy(zero_copy_only=False) for name in POOL_COLUMNS}
    if arrays["mass"].size and not np.all(np.diff(arrays["mass"]) >= 0):
        raise ValueError("Pool is not sorted by mass")
    for name in ("key14", "canonical14"):
        arrays[name] = arrays[name].astype("<U14")
    return Pool(**arrays)
