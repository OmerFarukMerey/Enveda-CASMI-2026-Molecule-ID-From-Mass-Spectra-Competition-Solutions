"""Generator join: the best M products of analog_derive.propose enter the v1 engine through the pc_join door (GEN_JOIN_M; 0 = off).

For answers that are in no database. The v1 engine's own generator (6 spectral parents, first-come cap of 150 products)
regenerates a hidden truth for 17% of hide-and-regenerate molecules; analog_derive (20 parents, products ranked by
multi-parent support) for 39%, 30% within its first 25 (reviews/coverage-2026-10-02.md section 4). Here its first M
products are handed to the engine as extra candidates next to the PubChem candidates and are placed by the ranker; the
engine's generator keeps running and treats them as pool keys (pc_join order 'first').

    an = E.analogs(spectra[:E.cfg.max_query_spectra], target, None, -1, -1)       # the engine's analog search, once more
    recs_gen = records(parents(an, L.struct_smiles, E.cfg.gen_min_sim, n=20), target, window_keys | pc_keys, m=30,
                       candidate_record=lambda s: pc_join.candidate_record(s, None, chem, fragmod, P.bits))
    pc_join.set_injection(E, pc_records + recs_gen)

Flags: GEN_JOIN_M=0, GEN_JOIN_PARENTS=20, GEN_JOIN_MAX_LIB=0.7 (as PC_JOIN_MAX_LIB). Measured in
reviews/next-levers-2026-10-03.md section 5. Pure helpers; analog_derive is imported lazily. Tests: test_gen_join.py.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

DEFAULTS = dict(GEN_JOIN_M=0, GEN_JOIN_PARENTS=20, GEN_JOIN_MAX_LIB=0.7)


def parents(analogs: Sequence[tuple], struct_smiles: Sequence[str], min_sim: float = 0.3, n: int = 20,
            skip_sids: Iterable[int] = ()) -> List[Tuple[str, float]]:
    """(parent SMILES, similarity) of the first n analogs with similarity >= min_sim. analogs: the engine's list of
    (structure id, similarity, ...) in descending similarity."""
    skip = set(skip_sids); out = []
    for a in analogs:
        sid, sim = int(a[0]), float(a[1])
        if sim < min_sim or sid in skip:
            continue
        out.append((struct_smiles[sid], sim))
        if len(out) >= n:
            break
    return out


def records(par: Sequence[Tuple[str, float]], target: float, exclude_keys: Set[str], m: int, candidate_record: Callable[[str], Optional[dict]],
            propose: Optional[Callable] = None, ppm: float = 10.0, max_ppm: float = 30.0, budget: int = 400) -> List[dict]:
    """The first m products (by support) as engine records. exclude_keys: keys already in the candidate set (pool window,
    PubChem records); a product whose metric key is among them, or repeats, is skipped. candidate_record(smiles) builds
    the record (fingerprint, fragments, metric key) and may return None."""
    if m <= 0 or not par:
        return []
    if propose is None:
        import analog_derive
        propose = analog_derive.propose
    have = set(exclude_keys); out = []
    for j, c in enumerate(propose(par, target, ppm=ppm, top=None, exclude_keys=exclude_keys, budget=budget)):
        if len(out) >= m:
            break
        r = candidate_record(c.smiles)
        if r is None or r['key'] in have or abs(r['mass'] - target) > target * max_ppm * 1e-6:
            continue
        have.add(r['key']); out.append(dict(r, src='gen', gen_rank=j, gen_support=float(getattr(c, 'support', 0.0))))
    return out
