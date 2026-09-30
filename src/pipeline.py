"""Molecule-level structure ranking for a submission: library search, then pool.

For each molecule (all spectra sharing a molecule_id):
1. Library: every spectrum is searched in the eligible training library with
   its own adduct and a precursor tolerance; a structure's score is its best
   cosine over the molecule's spectra.
2. Pool: every spectrum's neutral mass selects candidate structures within a
   ppm window from the pool (training structures that have at least one
   eligible library spectrum, plus COCONUT). Structures supported by more of the
   molecule's spectra rank first, then those with more eligible library
   spectra, then by key.
3. Fusion: library structures with cosine >= min_library_cosine first (by
   cosine), then pool structures, then weaker library structures, until
   max_rank distinct structures under the official metric key (precomputed in
   the pool, so no RDKit is required).

A molecule with no candidate from either source receives FILLER_SMILES so the
row stays valid; the count is reported.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pyarrow.parquet as pq

from src.adducts import neutral_mass
from src.catalog import window
from src.library_search import SPECTRUM_COLUMNS, gate_references, positive_float, selected_rows
from src.pool import Pool, load_pool
from src.spectral_search import PreparedSpectrum, prepare_spectrum, spectral_cosine

FILLER_SMILES = "C"


@dataclass(frozen=True, slots=True)
class QuerySpectrum:
    molecule_id: str
    adduct: str | None
    precursor_mz: float | None
    mzs: list[float] | None
    intensities: list[float] | None


@dataclass(frozen=True, slots=True)
class Settings:
    precursor_tolerance_ppm: float = 20.0
    pool_tolerance_ppm: float = 10.0
    fragment_tolerance_da: float = 0.02
    max_peaks: int = 100
    max_rank: int = 25
    min_library_cosine: float = 0.7


LibraryHit = tuple[float, float, int, str]  # cosine, ppm error, library row, SMILES


def _prepared(query: QuerySpectrum, max_peaks: int) -> PreparedSpectrum | None:
    mz = positive_float(query.precursor_mz)
    if mz is None or query.mzs is None or query.intensities is None or len(query.mzs) != len(query.intensities):
        return None
    return prepare_spectrum(mz, query.mzs, query.intensities, max_peaks=max_peaks)


def library_candidates(
    queries: Sequence[QuerySpectrum], parquet: pq.ParquetFile,
    allowed: Callable[[int, str | None, str | None], bool], settings: Settings,
    key_counts: Counter[str],
) -> dict[str, dict[str, LibraryHit]]:
    """Best library hit per molecule and supplied inchikey14; fills key_counts
    with the number of eligible library rows per supplied key."""
    metadata = {i: (q.adduct or None, positive_float(q.precursor_mz)) for i, q in enumerate(queries)}
    spectra = {i: s for i, q in enumerate(queries) if (s := _prepared(q, settings.max_peaks)) is not None}
    gated, _ = gate_references(parquet, metadata, allowed, settings.precursor_tolerance_ppm, key_counts)
    best: dict[str, dict[str, LibraryHit]] = defaultdict(dict)
    for row, (smiles, mzs, intensities) in selected_rows(parquet, set(gated), SPECTRUM_COLUMNS):
        reference = gated[row]
        key = reference.supplied_key
        if not key or not isinstance(smiles, str) or not smiles or mzs is None or intensities is None:
            continue
        if len(mzs) != len(intensities):
            continue
        reference_spectrum = prepare_spectrum(reference.precursor_mz, mzs, intensities, max_peaks=settings.max_peaks)
        for query_id, ppm_error in reference.matches:
            spectrum = spectra.get(query_id)
            if spectrum is None:
                continue
            score = spectral_cosine(spectrum, reference_spectrum, tolerance_da=settings.fragment_tolerance_da)
            molecule = queries[query_id].molecule_id
            current = best[molecule].get(key)
            if current is None or (-score, ppm_error, row) < (-current[0], current[1], current[2]):
                best[molecule][key] = (score, ppm_error, row, smiles)
    return best


def pool_candidates(
    queries: Sequence[QuerySpectrum], pool: Pool, library_counts: np.ndarray, settings: Settings, limit: int,
) -> tuple[dict[str, list[tuple[str, str]]], int]:
    """Ranked (official key, SMILES) pool candidates per molecule, and the count
    of spectra whose adduct or precursor could not give a neutral mass.

    A pool row is eligible if it is a COCONUT structure or a training structure
    with at least one eligible library spectrum (library_counts > 0).
    """
    eligible = pool.coconut | (library_counts > 0)
    by_molecule: dict[str, list[tuple[int, int]]] = defaultdict(list)
    unsupported = 0
    for query in queries:
        mz = positive_float(query.precursor_mz)
        try:
            mass = neutral_mass(mz, query.adduct or "") if mz is not None else None
        except ValueError:
            mass = None
        if mass is None:
            unsupported += 1
            continue
        low, high = window(pool.mass, mass, settings.pool_tolerance_ppm)
        if high > low:
            by_molecule[query.molecule_id].append((low, high))

    result: dict[str, list[tuple[str, str]]] = {}
    for molecule, windows in by_molecule.items():
        rows = np.concatenate([np.arange(low, high) for low, high in windows])
        spectrum = np.concatenate([np.full(high - low, number) for number, (low, high) in enumerate(windows)])
        keep = eligible[rows]
        rows, spectrum = rows[keep], spectrum[keep]
        if not rows.size:
            continue
        keys, inverse = np.unique(pool.canonical14[rows], return_inverse=True)
        pairs = np.unique(inverse.astype(np.int64) * len(windows) + spectrum)
        support = np.bincount(pairs // len(windows), minlength=keys.size)
        counts = library_counts[rows]
        best_count = np.zeros(keys.size, dtype=np.int64)
        np.maximum.at(best_count, inverse, counts)
        # Representative row per key: most library spectra, then smallest standard key.
        order = np.lexsort((pool.key14[rows], -counts))
        _, first = np.unique(inverse[order], return_index=True)
        representative = rows[order[first]]
        chosen = np.lexsort((keys, -best_count, -support))[:limit]
        result[molecule] = [(str(keys[k]), str(pool.smiles[representative[k]])) for k in chosen]
    return result, unsupported


def fuse(
    library: dict[str, LibraryHit], pool: list[tuple[str, str]], settings: Settings,
) -> list[str]:
    """Strong library hits, then pool, then weak library hits; distinct official keys.

    ``library`` must already be keyed by official key.
    """
    hits = sorted(((-h[0], h[1], h[2], key, h[3]) for key, h in library.items()))
    strong = [(key, smiles) for negative, _, _, key, smiles in hits if -negative >= settings.min_library_cosine]
    weak = [(key, smiles) for negative, _, _, key, smiles in hits if -negative < settings.min_library_cosine]
    guesses: list[str] = []
    seen: set[str] = set()
    for key, smiles in [*strong, *pool, *weak]:
        if key in seen or not smiles:
            continue
        seen.add(key)
        guesses.append(smiles)
        if len(guesses) == settings.max_rank:
            break
    return guesses


def _by_official_key(hits: dict[str, LibraryHit], canonical_of: dict[str, str]) -> dict[str, LibraryHit]:
    """Merge library hits whose supplied keys share an official key; keep the best."""
    merged: dict[str, LibraryHit] = {}
    for key, hit in hits.items():
        official = canonical_of.get(key, key)
        current = merged.get(official)
        if current is None or (-hit[0], hit[1], hit[2]) < (-current[0], current[1], current[2]):
            merged[official] = hit
    return merged


def predict(
    queries: Sequence[QuerySpectrum], train: Path, pool_path: Path,
    *, allowed: Callable[[int, str | None, str | None], bool] | None = None,
    settings: Settings = Settings(),
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Return up to max_rank SMILES per molecule_id (first-seen order) and run counts."""
    if not 1 <= settings.max_rank <= 25:
        raise ValueError("max_rank must be in 1..25")
    parquet = pq.ParquetFile(train)
    key_counts: Counter[str] = Counter()
    library = library_candidates(
        queries, parquet, allowed or (lambda _row, supplied, _source: bool(supplied)), settings, key_counts,
    )
    pool = load_pool(pool_path)
    library_counts = np.fromiter((key_counts.get(key, 0) for key in pool.key14.tolist()),
                                 dtype=np.int64, count=pool.key14.size)
    ranked, unsupported = pool_candidates(queries, pool, library_counts, settings, limit=2 * settings.max_rank)
    canonical_of = pool.canonical_of_key()

    molecules = list(dict.fromkeys(query.molecule_id for query in queries))
    predictions: dict[str, list[str]] = {}
    stats = {
        "molecules": len(molecules), "spectra": len(queries),
        "spectra_without_neutral_mass": unsupported,
        "pool_rows": int(pool.mass.size),
        "pool_rows_eligible": int((pool.coconut | (library_counts > 0)).sum()),
        "molecules_with_strong_library_hit": 0, "molecules_with_library_hit": 0,
        "molecules_with_pool_candidates": 0, "molecules_with_filler_only": 0,
    }
    for molecule in molecules:
        hits = _by_official_key(library.get(molecule, {}), canonical_of)
        pool_items = ranked.get(molecule, [])
        stats["molecules_with_library_hit"] += bool(hits)
        stats["molecules_with_strong_library_hit"] += any(h[0] >= settings.min_library_cosine for h in hits.values())
        stats["molecules_with_pool_candidates"] += bool(pool_items)
        guesses = fuse(hits, pool_items, settings)
        if not guesses:
            stats["molecules_with_filler_only"] += 1
            guesses = [FILLER_SMILES]
        predictions[molecule] = guesses
    return predictions, stats
