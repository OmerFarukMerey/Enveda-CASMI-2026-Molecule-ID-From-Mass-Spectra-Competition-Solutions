"""Deep PubChem join: helpers for PC_JOIN_N of several hundred (cache clearing, runtime and memory budget, depth read-out).

The join (pc_join.py) is a two-stage ranker: FPNet f.z shortlists the PubChem window, the v1 engine places the best N.
Leaderboard: N 0 / 50 / 100 / 200 = 0.409 / 0.420 / 0.431 / 0.406. N 100 is the best known depth; N 200 lost 0.025, which
the local regime did not predict (it shows the mechanism, lists with a pool answer losing their first place to injected
rows, at a fifth of that size; reviews/next-levers-2026-10-03.md section 1). Do not go deeper without a ranker that
was fitted with injected rows. The helpers here are what any depth above 50 needs:

  clear_every(n)            how often nb/06 must clear the per-SMILES caches of fragnet / frag2 (0.66 MB per injected row)
  budget(n, ...)            added seconds per molecule and for the run, peak cache memory; raises nothing, returns a dict
  depth_readout(pc_keys, base_keys, final_keys)   where the injected candidates of each depth band ended up: the only
                            label-free signal of "is depth still used" a scoring run can log

Config: PC_JOIN_N (existing flag), PC_JOIN_CLEAR_MB=1600 (new, cache budget in MB), PC_N1 (existing flag: the first-pass
shortlist the N are drawn from). Pure functions, tests in test_join_depth.py.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

DEFAULTS = dict(PC_JOIN_CLEAR_MB=1600)

MB_PER_ROW = 0.66          # per-SMILES records of fragnet / frag2 for one injected row (33 MB per molecule at N 50)
SEC_PER_ROW_LOCAL = 0.0095  # engine seconds per injected row on one desktop thread: (4.88 - 3.1) s at 185 rows
SEC_KEY_LOCAL = 0.0077      # one metric key in the channel's walk; about 1.2 keys per listed structure


def clear_every(n: int, budget_mb: float = 1600.0) -> int:
    """Molecules between two cache clears so that the caches stay under budget_mb. N 50 -> 48 (the notebook clears every
    50 today), N 100 -> 24, N 200 -> 12, N 1000 -> 2. Never less than 1."""
    if n <= 0:
        return 50
    return max(1, int(budget_mb / (MB_PER_ROW * n)))


def budget(n: int, n_molecules: int = 430, joined_share: float = 0.75, kaggle_slowdown: float = 2.0, pc_workers: int = 4,
           listed_share: float = 0.93, clear_mb: float = 1600.0) -> Dict[str, float]:
    """What N costs on the scoring run. joined_share: molecules below the PC_JOIN_MAX_LIB gate (0.72 locally);
    listed_share: rows the window really has per requested row (small windows list fewer than N);
    kaggle_slowdown: Kaggle CPU core against the desktop thread the constants were measured on."""
    rows = n * listed_share
    eng = rows * SEC_PER_ROW_LOCAL * kaggle_slowdown                       # per joined molecule
    chan = max(0.0, rows - 25) * 1.2 * SEC_KEY_LOCAL * kaggle_slowdown / max(1, pc_workers)   # per molecule, all molecules
    total = n_molecules * (joined_share * eng + chan)
    return dict(engine_s_per_joined_molecule=eng, channel_s_per_molecule=chan, run_minutes=total / 60.0,
                clear_every=clear_every(n, clear_mb), cache_mb_peak=min(clear_mb, clear_every(n, clear_mb) * MB_PER_ROW * n) if n > 0 else 0.0)


def depth_readout(pc_keys: Sequence[str], base_keys: Sequence[str], final_keys: Optional[Sequence[str]] = None,
                  bands: Sequence[int] = (25, 50, 100, 200, 500, 1000)) -> Dict[str, int]:
    """Counts per depth band of the PubChem list: how many injected candidates the ranker put into the base top 25 and
    how many are in the final 25. A run whose deepest band never reaches the lists has no use for more depth; one whose
    deepest band is as busy as the first has not hit the end of the useful list. No label is needed."""
    pos = {k: i for i, k in enumerate(pc_keys)}
    top = set(list(base_keys)[:25]); fin = set(list(final_keys)[:25]) if final_keys is not None else set()
    out, lo = {}, 0
    for hi in bands:
        if lo >= len(pc_keys):
            break
        ks = [k for k, i in pos.items() if lo <= i < hi]
        out[f'base25_{lo + 1}_{hi}'] = sum(k in top for k in ks)
        if final_keys is not None:
            out[f'final25_{lo + 1}_{hi}'] = sum(k in fin for k in ks)
        lo = hi
    return out
