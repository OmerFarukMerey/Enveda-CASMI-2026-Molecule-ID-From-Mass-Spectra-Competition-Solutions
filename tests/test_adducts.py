"""Neutral masses recovered from published adduct m/z values of caffeine (C8H10N4O2)."""

import pytest

from src.adducts import formula_mass, neutral_mass

CAFFEINE = 194.080376  # monoisotopic neutral mass


@pytest.mark.parametrize(
    ("mz", "adduct"),
    [
        (195.087652, "[M+H]+"),
        (193.073100, "[M-H]-"),
        (217.069594, "[M+Na]+"),
        (212.114199, "[M+NH4]+"),
        (233.043534, "[M+K]+"),
        (229.049778, "[M+Cl]-"),
        (239.078577, "[M+CH2O2-H]-"),
        (253.094227, "[M+C2H4O2-H]-"),
        (177.077087, "[M-H2O+H]+"),
        (389.168028, "[2M+H]+"),
        (194.079827, "[M]+"),
        (98.047464, "[M+2H]2+"),
        (409.135420, "[2M+Na-2H]-"),
    ],
)
def test_neutral_mass_matches_caffeine(mz: float, adduct: str) -> None:
    assert neutral_mass(mz, adduct) == pytest.approx(CAFFEINE, abs=2e-5)


def test_formula_mass_counts_elements() -> None:
    assert formula_mass("CH2O2") == pytest.approx(46.005479, abs=1e-6)
    assert formula_mass("HNa") == pytest.approx(1.007825 + 22.989769, abs=1e-6)


@pytest.mark.parametrize("adduct", ["M+H", "[M+Xx]+", "[M+H]", "[M+h]+", "", "[M+H]+ extra"])
def test_malformed_or_unknown_adducts_are_rejected(adduct: str) -> None:
    with pytest.raises(ValueError):
        neutral_mass(195.0, adduct)


@pytest.mark.parametrize("mz", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_precursor_is_rejected(mz: float) -> None:
    with pytest.raises(ValueError):
        neutral_mass(mz, "[M+H]+")


def test_nonpositive_neutral_mass_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonpositive"):
        neutral_mass(10.0, "[M+C2H4O2-H]-")
