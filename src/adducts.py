"""Neutral monoisotopic mass from an observed precursor m/z and adduct label.

Adducts follow the training data's notation, for example ``[M+H]+``,
``[M+CH2O2-H]-``, ``[2M+Na-2H]-`` or ``[M+2H]2+``. Each group after ``M`` is
an optional count followed by an elemental formula; the ion mass is
``k*M + Σ(±groups) − charge·electron``, and ``m/z = ion / |charge|``.
"""

from __future__ import annotations

from math import isfinite
import re

ELECTRON_MASS = 0.00054857990946
# Monoisotopic masses of the most abundant isotope (NIST/IUPAC).
ELEMENT_MASSES = {
    "H": 1.00782503207, "C": 12.0, "N": 14.0030740048, "O": 15.99491461956,
    "Na": 22.9897692809, "K": 38.96370668, "Cl": 34.96885268, "Ca": 39.96259098,
    "S": 31.97207100, "P": 30.97376163, "F": 18.99840322, "Br": 78.9183371,
    "I": 126.904473, "Li": 7.01600455, "Mg": 23.9850417, "Fe": 55.9349375,
}

_ADDUCT = re.compile(r"^\[(\d*)M((?:[+-]\d*[A-Z][A-Za-z0-9]*)*)\](\d*)([+-])$")
_GROUP = re.compile(r"([+-])(\d*)([A-Z][A-Za-z0-9]*)")
_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula: str) -> float:
    """Monoisotopic mass of a plain elemental formula such as ``CH2O2``."""
    position = 0
    total = 0.0
    for match in _ELEMENT.finditer(formula):
        if match.start() != position or match.group(1) not in ELEMENT_MASSES:
            raise ValueError(f"Unsupported formula: {formula}")
        total += ELEMENT_MASSES[match.group(1)] * int(match.group(2) or 1)
        position = match.end()
    if position != len(formula) or not formula:
        raise ValueError(f"Unsupported formula: {formula}")
    return total


def neutral_mass(precursor_mz: float, adduct: str) -> float:
    """Return the neutral monoisotopic mass M implied by an adduct and m/z.

    Raises ValueError for a nonpositive/nonfinite m/z or an adduct that is not
    in the ``[kM±groups]z±`` notation with known elements.
    """
    if isinstance(precursor_mz, bool) or not isfinite(precursor_mz) or precursor_mz <= 0:
        raise ValueError("precursor_mz must be finite and positive")
    match = _ADDUCT.match(adduct.strip()) if isinstance(adduct, str) else None
    if match is None:
        raise ValueError(f"Unsupported adduct: {adduct!r}")
    multiplier = int(match.group(1) or 1)
    charge = int(match.group(3) or 1) * (1 if match.group(4) == "+" else -1)
    groups = sum(
        (1 if sign == "+" else -1) * int(count or 1) * formula_mass(formula)
        for sign, count, formula in _GROUP.findall(match.group(2))
    )
    mass = (abs(charge) * precursor_mz + charge * ELECTRON_MASS - groups) / multiplier
    if mass <= 0:
        raise ValueError(f"Adduct {adduct!r} implies a nonpositive neutral mass")
    return mass
