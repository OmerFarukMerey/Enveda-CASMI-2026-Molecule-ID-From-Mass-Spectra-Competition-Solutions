"""Molecule-level structure ranking for a submission: library search, then catalog.

For each molecule (all spectra sharing a molecule_id):
1. Library: every spectrum is searched in the training library with its own
   adduct and a precursor tolerance; a structure's score is its best cosine
   over the molecule's spectra.
2. Catalog: every spectrum's neutral mass selects catalog structures within a
   ppm window; structures supported by more spectra rank first, then the fixed
   COCONUT/PubMed/CID prior.
3. Fusion: library structures with cosine >= min_library_cosine first (by
   cosine), then catalog structures, then weaker library structures, until
   max_rank distinct structures. Structures are identified by the first block of
   the standard InChIKey, which both sources carry, so no RDKit is required.

A molecule with no candidate from either source receives FILLER_SMILES so the
row stays valid; the count is reported.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pyarrow.parquet as pq

from src.adducts import neutral_mass
from src.catalog import fetch_smiles, load_catalog, prior_order, window
from src.library_search import SPECTRUM_COLUMNS, gate_references, positive_float, selected_rows
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
    catalog_tolerance_ppm: float = 10.0
    fragment_tolerance_da: float = 0.02
    max_peaks: int = 100
    max_rank: int = 25
    min_library_cosine: float = 0.7
    catalog_prior: str = "coconut_pmid"


LibraryHit = tuple[float, float, int, str]  # cosine, ppm error, library row, SMILES


def _prepared(query: QuerySpectrum, max_peaks: int) -> PreparedSpectrum | None:
    mz = positive_float(query.precursor_mz)
    if mz is None or query.mzs is None or query.intensities is None or len(query.mzs) != len(query.intensities):
        return None
    return prepare_spectrum(mz, query.mzs, query.intensities, max_peaks=max_peaks)


def library_candidates(
    queries: Sequence[QuerySpectrum], parquet: pq.ParquetFile,
    allowed: Callable[[int, str | None, str | None], bool], settings: Settings,
) -> dict[str, dict[str, LibraryHit]]:
    """Best library hit per molecule and structure (supplied inchikey14)."""
    metadata = {i: (q.adduct or None, positive_float(q.precursor_mz)) for i, q in enumerate(queries)}
    spectra = {i: s for i, q in enumerate(queries) if (s := _prepared(q, settings.max_peaks)) is not None}
    gated, _ = gate_references(parquet, metadata, allowed, settings.precursor_tolerance_ppm)
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


def catalog_candidates(
    queries: Sequence[QuerySpectrum], catalog: dict[str, np.ndarray], settings: Settings, limit: int,
) -> tuple[dict[str, list[tuple[str, int]]], int]:
    """Ranked (inchikey14, CID) catalog candidates per molecule, and the count of
    spectra whose adduct or precursor could not give a neutral mass."""
    by_molecule: dict[str, list[tuple[int, int, int]]] = defaultdict(list)  # (spectrum, low, high)
    unsupported = 0
    for index, query in enumerate(queries):
        mz = positive_float(query.precursor_mz)
        try:
            mass = neutral_mass(mz, query.adduct or "") if mz is not None else None
        except ValueError:
            mass = None
        if mass is None:
            unsupported += 1
            continue
        low, high = window(catalog["mass"], mass, settings.catalog_tolerance_ppm)
        if high > low:
            by_molecule[query.molecule_id].append((index, low, high))

    result: dict[str, list[tuple[str, int]]] = {}
    for molecule, windows in by_molecule.items():
        parts = {name: np.concatenate([np.asarray(catalog[name][low:high]) for _, low, high in windows])
                 for name in ("key14", "cid", "pmid_count", "coconut")}
        spectrum = np.concatenate([np.full(high - low, number) for number, (_, low, high) in enumerate(windows)])
        keys, inverse = np.unique(parts["key14"], return_inverse=True)
        pairs = np.unique(inverse.astype(np.int64) * len(windows) + spectrum)
        support = np.bincount(pairs // len(windows), minlength=keys.size)
        order = prior_order(parts["cid"], parts["pmid_count"], parts["coconut"], settings.catalog_prior)
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(order.size)
        best_rank = np.full(keys.size, np.iinfo(np.int64).max)
        np.minimum.at(best_rank, inverse, rank)
        chosen = np.lexsort((best_rank, -support))[:limit]
        result[molecule] = [(keys[k].decode(), int(parts["cid"][order[best_rank[k]]])) for k in chosen]
    return result, unsupported


def fuse(
    library: dict[str, LibraryHit], catalog: list[tuple[str, str]], settings: Settings,
) -> list[str]:
    """Strong library hits, then catalog, then weak library hits; distinct structures."""
    hits = sorted(((-h[0], h[1], h[2], key, h[3]) for key, h in library.items()))
    strong = [(key, smiles) for negative, _, _, key, smiles in hits if -negative >= settings.min_library_cosine]
    weak = [(key, smiles) for negative, _, _, key, smiles in hits if -negative < settings.min_library_cosine]
    guesses: list[str] = []
    seen: set[str] = set()
    for key, smiles in [*strong, *catalog, *weak]:
        if key in seen or not smiles:
            continue
        seen.add(key)
        guesses.append(smiles)
        if len(guesses) == settings.max_rank:
            break
    return guesses


def predict(
    queries: Sequence[QuerySpectrum], train: Path, catalog_dir: Path, smiles_path: Path,
    *, allowed: Callable[[int, str | None, str | None], bool] | None = None,
    settings: Settings = Settings(),
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Return up to max_rank SMILES per molecule_id (first-seen order) and run counts."""
    if not 1 <= settings.max_rank <= 25:
        raise ValueError("max_rank must be in 1..25")
    parquet = pq.ParquetFile(train)
    library = library_candidates(
        queries, parquet, allowed or (lambda _row, supplied, _source: bool(supplied)), settings,
    )
    catalog = load_catalog(catalog_dir)
    ranked, unsupported = catalog_candidates(queries, catalog, settings, limit=2 * settings.max_rank)
    smiles_of = fetch_smiles(smiles_path, (cid for items in ranked.values() for _, cid in items))

    molecules = list(dict.fromkeys(query.molecule_id for query in queries))
    predictions: dict[str, list[str]] = {}
    stats = {
        "molecules": len(molecules), "spectra": len(queries),
        "spectra_without_neutral_mass": unsupported,
        "molecules_with_strong_library_hit": 0, "molecules_with_library_hit": 0,
        "molecules_with_catalog_candidates": 0, "molecules_with_filler_only": 0,
    }
    for molecule in molecules:
        hits = library.get(molecule, {})
        catalog_items = [(key, smiles_of[cid]) for key, cid in ranked.get(molecule, []) if cid in smiles_of]
        stats["molecules_with_library_hit"] += bool(hits)
        stats["molecules_with_strong_library_hit"] += any(h[0] >= settings.min_library_cosine for h in hits.values())
        stats["molecules_with_catalog_candidates"] += bool(catalog_items)
        guesses = fuse(hits, catalog_items, settings)
        if not guesses:
            stats["molecules_with_filler_only"] += 1
            guesses = [FILLER_SMILES]
        predictions[molecule] = guesses
    return predictions, stats
