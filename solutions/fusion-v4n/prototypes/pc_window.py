"""PubChem-channel mass window: width (PC_PPM), per-adduct offset of the target mass (PC_MASS_SHIFT) and shortlist size (PC_N1).

Today the channel takes every PubChem structure within +-10 ppm of the target neutral mass (median over the molecule's
spectra of precursor m/z minus the labelled adduct), keeps the best PC_N1 = 5000 by a partial fingerprint score and ranks
those by the full FPNet f.z. Measured on the two Enveda timsTOF libraries of train.parquet (the hidden test's instrument;
reviews/next-levers-2026-10-03.md section 2):
  * the target mass is never more than 10 ppm from the structure's mass (0 of 183,191 molecules), more than 5 ppm for
    0.04% (enveda-180) and 2.4% (enveda-np-examples);
  * it carries a constant offset per adduct: [M+H]+ +0.0007 Da (+2.2 ppm, standard deviation 0.9 ppm), [M-H]- +0.0003 Da.
    With the offset removed the error is above 5 ppm for 0.3 to 0.4% and above 3 ppm for 1.5 to 7%.
So half of the window is empty of answers. A +-5 ppm window around the offset-corrected target removes the structures of
neighbouring formulas (19% of the best 200 by f.z, about half of the window), which halves pass 2 and makes PC_N1 bind
half as often.

    shifts = ENVEDA_SHIFT                                  # or estimate_shifts(nm_minus_mass, adducts) inside the run
    target = shifted_target(neutral_masses, adducts, shifts)
    a, b = window(pc_mass_sorted, target, ppm=5.0)

Flags (defaults = today): PC_PPM=10.0, PC_MASS_SHIFT=None ('enveda' = the table below), PC_N1=5000.
Pure functions (numpy), tests in test_pc_window.py. The table is a property of the instrument's calibration in the
training libraries, not of the visible test.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

DEFAULTS = dict(PC_PPM=10.0, PC_MASS_SHIFT=None, PC_N1=5000)

# median of (neutral mass from the precursor - exact mass of the structure), Da, enveda-180 rows of train.parquet
# (scratch/next_levers/q2_mass_error.py; adducts with at least 2,000 spectra). enveda-np-examples agrees within 0.0001 Da
# for [M+H]+ and [M-H]-.
ENVEDA_SHIFT: Dict[str, float] = {
    '[M+H]+': 0.00072, '[M+Na]+': 0.00034, '[M+NH4]+': 0.00049, '[2M+H]+': 0.00023, '[2M+Na]+': 0.00026,
    '[M-H]-': 0.00026, '[M+CH2O2-H]-': 0.00012, '[M+Cl]-': -0.00005, '[2M-H]-': 0.00000,
}


def estimate_shifts(delta_da: Sequence[float], adducts: Sequence[str], min_n: int = 2000) -> Dict[str, float]:
    """Median offset per adduct from reference rows (delta = neutral mass computed from the precursor - exact mass).
    Adducts with fewer than min_n finite rows get no entry (no correction)."""
    d = np.asarray(delta_da, np.float64); a = np.asarray(adducts, object)
    out = {}
    for ad in sorted(set(a.tolist())):
        v = d[(a == ad) & np.isfinite(d)]
        if len(v) >= min_n:
            out[ad] = float(np.median(v))
    return out


def shifted_target(neutral_masses: Sequence[float], adducts: Sequence[str], shifts: Optional[Dict[str, float]] = None) -> Optional[float]:
    """Median over the spectra of (neutral mass - offset of the spectrum's adduct); None when no mass is finite.
    shifts None or empty = the plain median (today)."""
    m = np.asarray(neutral_masses, np.float64)
    if shifts:
        m = m - np.array([float(shifts.get(a, 0.0)) for a in adducts], np.float64)
    m = m[np.isfinite(m)]
    return float(np.median(m)) if len(m) else None


def window(mass_sorted: np.ndarray, target: float, ppm: float = 10.0) -> Tuple[int, int]:
    """[a, b) rows of an ascending mass array within +-ppm of target (the +-10 ppm rule of probe_core2._window)."""
    tol = float(target) * float(ppm) * 1e-6
    return int(np.searchsorted(mass_sorted, target - tol, 'left')), int(np.searchsorted(mass_sorted, target + tol, 'right'))


def loss_share(abs_ppm_error: Sequence[float], ppm: float) -> float:
    """Share of molecules whose answer a +-ppm window loses, given |target - exact mass| in ppm per molecule."""
    e = np.asarray(abs_ppm_error, np.float64); e = e[np.isfinite(e)]
    return float((e > ppm).mean()) if len(e) else float('nan')
