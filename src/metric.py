"""Local CASMI mean reciprocal rank evaluation using RDKit 2026.03.3.

The official notebook accepts semicolon-delimited submission strings; this API accepts
already-separated, ordered candidate sequences instead. A solution value may contain
comma-separated accepted SMILES. Neither a supplied InChIKey nor a molecular-formula
filter substitutes for canonicalizing both answers and guesses with RDKit.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


_TAUTOMER_CANONICALIZER = rdMolStandardize.TautomerEnumerator()


class EvaluationResult(NamedTuple):
    """Overall mean reciprocal rank and individual scores keyed by solution ID."""

    mrr: float
    reciprocal_ranks: dict[str, float]


def canonicalize_smiles_to_ik14(smiles: str) -> str | None:
    """Return the tautomer-canonical first InChIKey block, or None if invalid.

    Uses the competition's RDKit 2026.03.3 chemistry: parse, canonicalize the
    tautomer, generate an InChIKey, then take its first 14 characters. This
    deliberately drops stereo/protonation distinctions in the first block.
    """
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical_mol = _TAUTOMER_CANONICALIZER.Canonicalize(mol)
        return Chem.MolToInchiKey(canonical_mol)[:14] or None
    except (ValueError, RuntimeError, TypeError):
        return None


def _check_max_rank(max_rank: int) -> None:
    if isinstance(max_rank, bool) or not isinstance(max_rank, int) or max_rank < 1:
        raise ValueError("max_rank must be a positive integer")


def evaluate(
    solution: Mapping[str, str],
    predictions: Mapping[str, Sequence[str]],
    max_rank: int = 25,
) -> EvaluationResult:
    """Compute MRR over all solution IDs, with first matching rank per ID.

    Missing IDs score zero; unknown prediction IDs do not affect MRR but their
    rows must still pass the official row-level checks. Empty candidate tokens
    are discarded *before* counting ranks, as in the official notebook; invalid
    nonempty SMILES occupy a rank but never match. An empty predictions mapping,
    an explicit row without nonempty guesses, or a row with more than max_rank
    nonempty guesses raises ValueError. Invalid answer SMILES raise ValueError
    unless another comma-separated answer is valid. An empty solution cannot
    define a mean and raises ValueError.

    This scoring API intentionally does not enforce strict submission validity:
    the official scorer ignores unknown IDs and empty semicolon tokens in an
    otherwise nonempty row, and accepts invalid guesses as nonmatches. Use
    validate_submission separately to check an export before sending it.
    """
    _check_max_rank(max_rank)
    if not solution:
        raise ValueError("solution must contain at least one molecule")
    if not predictions:
        raise ValueError("Submission is empty")
    for molecule_id, guesses in predictions.items():
        if isinstance(guesses, (str, bytes)):
            raise ValueError(f"Expected a sequence of guesses for molecule_id {molecule_id}")
        count = sum(bool(guess.strip()) for guess in guesses)
        if not count:
            raise ValueError(f"No guesses for molecule_id {molecule_id}")
        if count > max_rank:
            raise ValueError(f"More than {max_rank} guesses for molecule_id {molecule_id}")

    answer_keys: dict[str, set[str]] = {}
    for molecule_id, answer in solution.items():
        accepted = {
            key
            for smiles in (answer.split(",") if isinstance(answer, str) else ())
            if smiles
            if (key := canonicalize_smiles_to_ik14(smiles)) is not None
        }
        if not accepted:
            raise ValueError(f"No parseable answer SMILES for molecule_id {molecule_id}")
        answer_keys[molecule_id] = accepted

    reciprocal_ranks: dict[str, float] = {}
    guess_keys: dict[str, str | None] = {}
    for molecule_id, accepted in answer_keys.items():
        guesses = predictions.get(molecule_id, ())
        score = 0.0
        rank = 0
        for candidate in guesses:
            guess = candidate.strip()
            if not guess:
                continue
            rank += 1
            if guess not in guess_keys:
                guess_keys[guess] = canonicalize_smiles_to_ik14(guess)
            if guess_keys[guess] in accepted:
                score = 1.0 / rank
                break
        reciprocal_ranks[molecule_id] = score

    return EvaluationResult(sum(reciprocal_ranks.values()) / len(solution), reciprocal_ranks)


def validate_submission(
    solution: Mapping[str, str],
    predictions: Mapping[str, Sequence[str]],
    max_rank: int = 25,
) -> None:
    """Reject non-submittable predictions; stricter than competition scoring.

    Requires exactly the solution ID set, 1..max_rank (at most 25) nonblank
    candidate SMILES per ID, parseable guesses and unique tautomer-canonical
    InChIKey14s per ID. Unlike evaluate(), no invalid/empty guesses are permitted.
    Raises ValueError on a violation; does not inspect the answer SMILES.
    """
    _check_max_rank(max_rank)
    if max_rank > 25:
        raise ValueError("strict submission max_rank must be at most 25")

    missing = solution.keys() - predictions.keys()
    extra = predictions.keys() - solution.keys()
    if missing or extra:
        raise ValueError(f"Prediction IDs differ from solution IDs: missing={sorted(missing)}, extra={sorted(extra)}")

    for molecule_id, candidates in predictions.items():
        if isinstance(candidates, (str, bytes)) or not 1 <= len(candidates) <= max_rank:
            raise ValueError(f"Expected 1..{max_rank} candidates for molecule_id {molecule_id}")
        seen: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError(f"Empty or non-string candidate for molecule_id {molecule_id}")
            key = canonicalize_smiles_to_ik14(candidate.strip())
            if key is None:
                raise ValueError(f"Invalid candidate SMILES for molecule_id {molecule_id}")
            if key in seen:
                raise ValueError(f"Duplicate canonical candidate for molecule_id {molecule_id}")
            seen.add(key)
