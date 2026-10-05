"""Calibrated precursor-mass window: per-adduct ppm offset of the target mass and one window width per channel (flags MASS_OFFSET_BY_ADDUCT, PPM_WIN, ENG_POOL_PPM, PC_PPM).

Every channel collects candidates within +-10 ppm of the target neutral mass (median over the molecule's spectra of
precursor m/z minus the labelled adduct). Measured on the two Enveda timsTOF libraries of train.parquet, the hidden
test's instrument (reviews/masscal-2026-10-04.md):
  * the neutral mass carries a systematic offset per adduct: [M+H]+ +2.2 ppm, [M-H]- +0.8 ppm (table below). It is not
    an electron (the adduct tables are right); it is closer to constant in ppm than in Da across 150 to 1000 Da, so the
    correction is multiplicative: mass / (1 + ppm * 1e-6);
  * per molecule, with the test's adducts and number of spectra, the corrected target is outside +-5 ppm for 0.2%
    (enveda-180) and 0.9% (enveda-np-examples), outside +-3 ppm for 0.7% and 7%. Uncorrected: 5 ppm loses 0.03% / 2.8%.

    off = offsets('enveda')                                  # CFG['MASS_OFFSET_BY_ADDUCT']: None | 'enveda' | {adduct: ppm}
    target = target_mass(neutral_masses, adducts, off)       # replaces float(np.median(nms)) in nb/06, pc_runner, engine-2
    a, b = window(sorted_masses, target, ppm=5.0)

Defaults reproduce today's behaviour (no offset, 10 ppm everywhere). Pure numpy; tests in test_mass_window.py need no
competition data. The table is fitted on training libraries only (never on test.parquet).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

DEFAULTS = dict(MASS_OFFSET_BY_ADDUCT=None, PPM_WIN=10.0, ENG_POOL_PPM=10.0, PC_PPM=10.0)

# median of (neutral mass from the precursor - exact mass of the formula) in ppm, enveda-180 rows of train.parquet with a
# test adduct and at least 1,000 spectra (scratch/masscal/s1b_testlike.py). enveda-np-examples: [M+H]+ +2.01, [M-H]- +0.56,
# [M+Na]+ +1.67, [M+NH4]+ +1.25, [M+CH2O2-H]- +0.22 (60 to 569 spectra each).
ENVEDA_PPM: Dict[str, float] = {
    '[M+H]+': 2.22, '[M+NH4]+': 1.50, '[M+Na]+': 1.03, '[M+K]+': 0.23,
    '[M-H]-': 0.80, '[M+CH2O2-H]-': 0.35, '[M+Cl]-': -0.15,
}


def offsets(flag: Union[None, str, Mapping[str, float]]) -> Dict[str, float]:
    """CFG['MASS_OFFSET_BY_ADDUCT'] -> {adduct: ppm}. None / '' / 'none' = no correction; 'enveda' = ENVEDA_PPM."""
    if flag is None or flag == '' or flag == 'none':
        return {}
    if isinstance(flag, str):
        if flag != 'enveda':
            raise ValueError(f'unknown offset table {flag!r}')
        return dict(ENVEDA_PPM)
    return {str(k): float(v) for k, v in flag.items()}


def corrected(neutral_masses: Sequence[float], adducts: Sequence[str], off: Optional[Mapping[str, float]] = None) -> np.ndarray:
    """Neutral mass per spectrum with the adduct's ppm offset removed; adducts without an entry are unchanged."""
    m = np.asarray(neutral_masses, np.float64)
    if not off:
        return m.copy()
    p = np.array([float(off.get(a, 0.0)) for a in adducts], np.float64)
    return m / (1.0 + 1e-6 * p)


def target_mass(neutral_masses: Sequence[float], adducts: Sequence[str], off: Optional[Mapping[str, float]] = None) -> Optional[float]:
    """Median over the spectra of the corrected neutral masses; None when no mass is finite. off empty = today's target."""
    m = corrected(neutral_masses, adducts, off)
    m = m[np.isfinite(m)]
    return float(np.median(m)) if len(m) else None


def window(mass_sorted: np.ndarray, target: float, ppm: float = 10.0) -> Tuple[int, int]:
    """[a, b) rows of an ascending mass array within +-ppm of target (the rule of Pool.window and probe_core2._window)."""
    tol = float(target) * float(ppm) * 1e-6
    return int(np.searchsorted(mass_sorted, target - tol, 'left')), int(np.searchsorted(mass_sorted, target + tol, 'right'))


def window_with_fallback(mass_sorted: np.ndarray, target: float, ppm: float, fallbacks: Sequence[float] = (10.0, 30.0)) -> Tuple[int, int, float]:
    """window(), widened step by step while it is empty (a narrowed pool window must not fall straight to 30 ppm)."""
    for p in (ppm,) + tuple(f for f in fallbacks if f > ppm):
        a, b = window(mass_sorted, target, p)
        if b > a:
            return a, b, float(p)
    return a, b, float(p)


def fit_offsets(delta_ppm: Sequence[float], adducts: Sequence[str], min_n: int = 1000) -> Dict[str, float]:
    """Median ppm offset per adduct from reference rows (delta = (neutral mass - exact mass) / exact mass * 1e6).
    Adducts with fewer than min_n finite rows get no entry."""
    d = np.asarray(delta_ppm, np.float64); a = np.asarray(adducts, object)
    out = {}
    for ad in sorted(set(a.tolist())):
        v = d[(a == ad) & np.isfinite(d)]
        if len(v) >= min_n:
            out[ad] = float(np.median(v))
    return out


def loss_share(abs_ppm_error: Sequence[float], ppm: float) -> float:
    """Share of molecules whose answer a +-ppm window loses, given |target - exact mass| in ppm per molecule."""
    e = np.asarray(abs_ppm_error, np.float64); e = e[np.isfinite(e)]
    return float((e > ppm).mean()) if len(e) else float('nan')


def pass2_rows(n_window: int, n1: int = 5000) -> int:
    """Rows of the PubChem second pass (full fingerprints): the shortlist binds only when the window is larger."""
    return int(min(int(n_window), int(n1)))
