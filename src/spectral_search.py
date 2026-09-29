"""Fixed peak preparation and unshifted spectral cosine for reference retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PreparedSpectrum:
    mzs: tuple[float, ...]
    weights: tuple[float, ...]
    norm: float


def prepare_spectrum(
    precursor_mz: float,
    mzs: Sequence[float],
    intensities: Sequence[float],
    max_peaks: int = 100,
) -> PreparedSpectrum:
    """Keep the strongest valid peaks and prepare square-root intensity weights."""
    if isinstance(precursor_mz, bool) or not isfinite(precursor_mz) or precursor_mz <= 0:
        raise ValueError("precursor_mz must be finite and positive")
    if isinstance(max_peaks, bool) or not isinstance(max_peaks, int) or max_peaks <= 0:
        raise ValueError("max_peaks must be a positive integer")
    if len(mzs) != len(intensities):
        raise ValueError("mzs and intensities must have equal lengths")

    limit = precursor_mz + 2.0
    peaks = [
        (intensity, mz, index)
        for index, (mz, intensity) in enumerate(zip(mzs, intensities))
        if isfinite(mz) and 0 < mz <= limit and isfinite(intensity) and 0 < intensity <= 1
    ]
    peaks.sort(key=lambda peak: (-peak[0], peak[1], peak[2]))
    kept = peaks[:max_peaks]
    kept.sort(key=lambda peak: (peak[1], peak[2]))
    return PreparedSpectrum(
        mzs=tuple(float(mz) for _, mz, _ in kept),
        weights=tuple(sqrt(intensity) for intensity, _, _ in kept),
        norm=sqrt(fsum(intensity for intensity, _, _ in kept)),
    )


def spectral_cosine(
    query: PreparedSpectrum,
    reference: PreparedSpectrum,
    tolerance_da: float = 0.02,
) -> float:
    """Greedily match each peak once, prioritizing its weight product.

    Only peaks within the unshifted, inclusive mass window form candidate edges.
    Equal products prefer the smaller m/z error, then input peak indices.
    """
    if not isfinite(tolerance_da) or tolerance_da < 0:
        raise ValueError("tolerance_da must be finite and nonnegative")
    if not query.mzs or not reference.mzs or not query.norm or not reference.norm:
        return 0.0

    edges: list[tuple[float, float, int, int]] = []
    first_reference = 0
    for query_index, (query_mz, query_weight) in enumerate(zip(query.mzs, query.weights)):
        while (
            first_reference < len(reference.mzs)
            and reference.mzs[first_reference] < query_mz - tolerance_da
        ):
            first_reference += 1
        reference_index = first_reference
        while (
            reference_index < len(reference.mzs)
            and reference.mzs[reference_index] <= query_mz + tolerance_da
        ):
            distance = abs(query_mz - reference.mzs[reference_index])
            if distance <= tolerance_da:
                edges.append(
                    (-query_weight * reference.weights[reference_index], distance,
                     query_index, reference_index)
                )
            reference_index += 1

    edges.sort()
    matched_queries: set[int] = set()
    matched_references: set[int] = set()
    dot_product = 0.0
    for negative_product, _, query_index, reference_index in edges:
        if query_index not in matched_queries and reference_index not in matched_references:
            matched_queries.add(query_index)
            matched_references.add(reference_index)
            dot_product -= negative_product
    return min(1.0, max(0.0, dot_product / (query.norm * reference.norm)))
