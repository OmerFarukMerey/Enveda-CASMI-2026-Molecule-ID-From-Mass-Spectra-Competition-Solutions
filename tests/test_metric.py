"""Small synthetic checks for the locally reproduced CASMI metric."""

import unittest

from src.metric import canonicalize_smiles_to_ik14, evaluate, validate_submission


class MetricTests(unittest.TestCase):
    def test_stereo_and_tautomer_equivalence(self) -> None:
        solution = {
            "stereo": "C[C@H](N)C(=O)O",
            "tautomer": "CC(=O)C",
        }
        predictions = {
            "stereo": ["CC(N)C(=O)O"],
            "tautomer": ["CC(O)=C"],
        }
        result = evaluate(solution, predictions)
        self.assertEqual(result.mrr, 1.0)
        self.assertEqual(result.reciprocal_ranks, {"stereo": 1.0, "tautomer": 1.0})
        self.assertEqual(
            canonicalize_smiles_to_ik14("CC(O)=C"),
            canonicalize_smiles_to_ik14("CC(=O)C"),
        )

    def test_first_hit_uses_order_and_multiple_accepted_answers(self) -> None:
        result = evaluate(
            {"a": "CC(=O)O,CCO", "b": "c1ccccc1"},
            {"a": ["C", "CCO", "CC(=O)O"], "b": ["CCO", "c1ccccc1"]},
        )
        self.assertEqual(result.reciprocal_ranks, {"a": 0.5, "b": 0.5})
        self.assertEqual(result.mrr, 0.5)

    def test_invalid_guesses_count_as_ranks_but_empty_tokens_do_not(self) -> None:
        result = evaluate({"a": "CCO"}, {"a": ["", "not valid!", "   ", "OCC"]})
        self.assertEqual(result.reciprocal_ranks, {"a": 0.5})
        self.assertIsNone(canonicalize_smiles_to_ik14("not valid!"))
        self.assertIsNone(canonicalize_smiles_to_ik14(""))

    def test_missing_scores_zero_and_unknown_id_is_ignored(self) -> None:
        result = evaluate(
            {"missing": "CCO", "hit": "C"},
            {"hit": ["C"], "unrecognized": ["CCO"]},
        )
        self.assertEqual(result.reciprocal_ranks, {"missing": 0.0, "hit": 1.0})
        self.assertEqual(result.mrr, 0.5)
        with self.assertRaisesRegex(ValueError, "missing=.*missing"):
            validate_submission({"missing": "CCO", "hit": "C"}, {"hit": ["C"]})
        with self.assertRaisesRegex(ValueError, "extra=.*unrecognized"):
            validate_submission({"hit": "C"}, {"hit": ["C"], "unrecognized": ["CCO"]})

    def test_no_hits_or_missing_ids_score_zero(self) -> None:
        result = evaluate({"a": "CCO", "b": "C"}, {"a": ["N"]})
        self.assertEqual(result.reciprocal_ranks, {"a": 0.0, "b": 0.0})
        self.assertEqual(result.mrr, 0.0)
        with self.assertRaisesRegex(ValueError, "No parseable answer"):
            evaluate({"a": "not valid!"}, {"a": ["CCO"]})

    def test_official_row_checks_apply_even_to_unknown_ids(self) -> None:
        for predictions, message in (
            ({}, "Submission is empty"),
            ({"a": []}, "No guesses"),
            ({"a": ["", "   "]}, "No guesses"),
            ({"a": ["C"], "unknown": []}, "No guesses.*unknown"),
        ):
            with self.subTest(predictions=predictions), self.assertRaisesRegex(ValueError, message):
                evaluate({"a": "C"}, predictions)
        with self.assertRaisesRegex(ValueError, "More than 1 guesses.*unknown"):
            evaluate({"a": "C"}, {"a": ["C"], "unknown": ["C", "N"]}, max_rank=1)

    def test_strict_validation_rejects_invalid_and_duplicate_connectivity(self) -> None:
        solution = {"a": "CCO"}
        validate_submission(solution, {"a": ["CCO", "C"]})
        for predictions, message in (
            ({"a": []}, "Expected 1"),
            ({"a": [" "]}, "Empty"),
            ({"a": ["not valid!"]}, "Invalid"),
            ({"a": ["CCO", "OCC"]}, "Duplicate"),
            ({"a": ["C[C@H](N)C(=O)O", "CC(N)C(=O)O"]}, "Duplicate"),
            ({"a": ["C"] * 26}, "Expected 1"),
        ):
            with self.subTest(predictions=predictions), self.assertRaisesRegex(ValueError, message):
                validate_submission(solution, predictions)

    def test_rank_cap_counts_only_nonempty_guesses(self) -> None:
        self.assertEqual(evaluate({"a": "C"}, {"a": ["", "C"]}, max_rank=1).mrr, 1.0)
        with self.assertRaisesRegex(ValueError, "More than 1 guesses"):
            evaluate({"a": "C"}, {"a": ["N", "C"]}, max_rank=1)
        with self.assertRaisesRegex(ValueError, "at most 25"):
            validate_submission({"a": "C"}, {"a": ["C"]}, max_rank=26)


if __name__ == "__main__":
    unittest.main()
