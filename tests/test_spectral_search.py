"""Behavioral checks for fixed spectral peak processing and matching."""

import math
import unittest
from dataclasses import FrozenInstanceError
from typing import cast

from src.spectral_search import PreparedSpectrum, prepare_spectrum, spectral_cosine


class SpectralSearchTests(unittest.TestCase):
    def test_identical_and_disjoint_spectra(self) -> None:
        spectrum = prepare_spectrum(250, [60, 120, 200], [0.25, 1.0, 0.5])
        self.assertAlmostEqual(spectral_cosine(spectrum, spectrum), 1.0)
        far_away = prepare_spectrum(250, [61, 121, 201], [0.25, 1.0, 0.5])
        self.assertEqual(spectral_cosine(spectrum, far_away), 0.0)
        self.assertEqual(spectral_cosine(spectrum, prepare_spectrum(250, [], [])), 0.0)

    def test_inclusive_mass_window_and_no_shift(self) -> None:
        query = prepare_spectrum(250, [100], [1.0])
        self.assertEqual(spectral_cosine(query, prepare_spectrum(250, [100.02], [1.0])), 1.0)
        self.assertEqual(spectral_cosine(query, prepare_spectrum(250, [100.021], [1.0])), 0.0)
        self.assertEqual(
            spectral_cosine(query, prepare_spectrum(250, [100], [1.0]), tolerance_da=0),
            1.0,
        )

    def test_a_reference_peak_cannot_match_two_query_peaks(self) -> None:
        query = prepare_spectrum(250, [100, 100], [1.0, 1.0])
        reference = prepare_spectrum(250, [100], [1.0])
        self.assertAlmostEqual(spectral_cosine(query, reference), 1 / math.sqrt(2))
        self.assertAlmostEqual(spectral_cosine(reference, query), 1 / math.sqrt(2))

    def test_square_root_intensity_and_normalization(self) -> None:
        query = prepare_spectrum(250, [100, 150], [0.16, 0.64])
        reference = prepare_spectrum(250, [100, 150], [0.64, 0.16])
        self.assertAlmostEqual(spectral_cosine(query, reference), 0.8)
        scaled_query = prepare_spectrum(250, [100, 150], [0.08, 0.32])
        self.assertAlmostEqual(spectral_cosine(scaled_query, reference), 0.8)
        self.assertAlmostEqual(query.norm, math.sqrt(0.8))
        self.assertAlmostEqual(query.weights[0], 0.4)

    def test_greedy_matches_highest_product_first(self) -> None:
        query = prepare_spectrum(250, [100, 100.03], [1.0, 0.36])
        reference = prepare_spectrum(250, [100.01, 100.02], [0.64, 0.49])
        expected = (0.8 + 0.6 * 0.7) / math.sqrt(1.36 * 1.13)
        self.assertAlmostEqual(spectral_cosine(query, reference), expected)

    def test_top_n_tie_breaking_and_noise_filtering(self) -> None:
        spectrum = prepare_spectrum(
            200,
            [100, 150, 199, 202, 202.001, float("nan"), -1, 0, 170, 180, 181, 120],
            [0.3, 0.8, 0.8, 0.9, 1.0, 0.9, 0.9, 0.9, float("inf"), -0.1, 1.1, 0],
            max_peaks=3,
        )
        self.assertEqual(spectrum.mzs, (150.0, 199.0, 202.0))
        self.assertAlmostEqual(spectrum.norm, math.sqrt(2.5))
        self.assertEqual(prepare_spectrum(200, [20, 10], [0.5, 0.5], 1).mzs, (10.0,))
        self.assertEqual(prepare_spectrum(200, [float("inf")], [1.0]).norm, 0.0)

    def test_input_validation_and_immutable_preparation(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            prepare_spectrum(200, [100], [])
        for precursor in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(precursor=precursor), self.assertRaisesRegex(ValueError, "precursor_mz"):
                prepare_spectrum(precursor, [], [])
        for max_peaks in (0, -1, 1.5, True):
            with self.subTest(max_peaks=max_peaks), self.assertRaisesRegex(ValueError, "max_peaks"):
                prepare_spectrum(200, [], [], max_peaks=cast(int, max_peaks))
        empty = prepare_spectrum(200, [], [])
        self.assertEqual(empty, PreparedSpectrum((), (), 0.0))
        with self.assertRaises(FrozenInstanceError):
            setattr(empty, "norm", 1.0)
        for tolerance in (-0.01, float("nan"), float("inf")):
            with self.subTest(tolerance=tolerance), self.assertRaisesRegex(ValueError, "tolerance_da"):
                spectral_cosine(empty, empty, tolerance_da=tolerance)


if __name__ == "__main__":
    unittest.main()
