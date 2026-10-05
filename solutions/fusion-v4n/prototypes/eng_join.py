"""Engine-2 candidates outside the main pool: an early fusion slot (FUSE_ENG_SLOT) or the join door (ENG_JOIN_K); both off by default.

Engine-2's candidate set = COCONUT + ChEBI / LIPID MAPS + training structures; its surplus over the v1 pool is the
ChEBI / LIPID MAPS file (62.7k structures). When the answer is such a structure, engine-2 lists it first or second, but
the reciprocal-rank fusion gives a candidate that only engine-2 lists the weight of v1 rank 3.7, and it lands at rank 5 to
7 (reviews/mixed-regime-2026-10-02.md section 5.3). Two ways to use it, measured in reviews/next-levers-2026-10-03.md:

  FUSE_ENG_SLOT = s   stage C of fusion_core: engine-2's best candidate whose key is not a key of the main pool, if it
                      is among engine-2's first FUSE_ENG_SLOT_TOP candidates, is moved up to position s of the fused
                      list (never down). `slot_fuse(fusion_core.fuse_rrf, pool_keys, s, top)` returns the replacement
                      for fusion_core.fuse_rrf; with s = None it returns the original function.
  ENG_JOIN_K = k      nb/06: engine-2's first k candidates that are not candidates of the v1 window enter the v1 engine
                      through pc_join (before the PubChem candidates) and are placed by the ranker.
                      `outside_window(entry, window_keys, k)` picks them, `join_records(...)` builds the records and
                      merges them with the PubChem records.

Pure helpers only (numpy); the engine side is pc_join.install / set_injection, unchanged. Tests: test_eng_join.py.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Set

DEFAULTS = dict(FUSE_ENG_SLOT=None,      # 1-based slot for engine-2's best outside-pool candidate; None = off (today)
                FUSE_ENG_SLOT_TOP=3,     # ... only when it is among engine-2's first this many candidates
                ENG_JOIN_K=0,            # engine-2 candidates outside the v1 window that enter the v1 engine; 0 = off
                ENG_JOIN_MAX_LIB=None)   # no engine-2 injection when the engine's lib_max >= this; None = always


def outside_pool(e_keys: Sequence[Optional[str]], pool_keys: Set[str], top: int = 3, n: int = 1) -> List[str]:
    """Keys of engine-2's first `n` candidates that are not main-pool keys, looking only at its first `top` candidates."""
    out = []
    for k in list(e_keys)[:max(0, int(top))]:
        if k and k not in pool_keys and k not in out:
            out.append(k)
            if len(out) >= n:
                break
    return out


def move_up(order: Sequence[str], key: str, slot: int) -> List[int]:
    """Permutation (indices into `order`) that moves `key` to 1-based position `slot` if it sits later; identity otherwise."""
    idx = list(range(len(order)))
    if key in order:
        i = list(order).index(key); t = max(0, int(slot) - 1)
        if i > t:
            idx.insert(t, idx.pop(i))
    return idx


def slot_fuse(fuse_rrf: Callable, pool_keys: Iterable[str], slot: Optional[int], top: int = 3, n: int = 1) -> Callable:
    """A drop-in for fusion_core.fuse_rrf. After the reciprocal-rank fusion, engine-2's best `n` outside-pool candidates
    (see outside_pool) are moved up to positions slot, slot + 1, ... The returned scores are re-sorted so that the
    forward-model re-rank that follows sees a monotone score column (it z-scores this column)."""
    if slot is None:
        return fuse_rrf
    pool_keys = pool_keys if isinstance(pool_keys, (set, frozenset)) else set(pool_keys)

    def fuse(v_smis, e_smis, e_keys, score_key, alpha, krr, n_out):
        smis, order, sc = fuse_rrf(v_smis, e_smis, e_keys, score_key, alpha, krr, n_out)
        moved = False
        for j, k in enumerate(outside_pool(e_keys, pool_keys, top, n)):
            p = move_up(order, k, slot + j)
            if p != list(range(len(order))):
                smis = [smis[i] for i in p]; order = [order[i] for i in p]; moved = True
        if moved:
            sc = sorted(sc, reverse=True)
        return smis, order, sc
    return fuse


def outside_window(entry: Optional[dict], window_keys: Set[str], k: int) -> List[int]:
    """Indices into an engine-2 list entry {smiles, keys} of its first k candidates whose key is not a candidate key of
    the v1 window (pool rows of the mass window). Entries without a key are skipped."""
    if not entry or k <= 0:
        return []
    out, seen = [], set()
    for i, kk in enumerate(entry.get('keys', [])):
        if not kk or kk in window_keys or kk in seen:
            continue
        seen.add(kk); out.append(i)
        if len(out) >= k:
            break
    return out


def merge_records(first: Sequence[dict], second: Sequence[dict], target: Optional[float] = None, ppm: float = 30.0) -> List[dict]:
    """Records of two sources in injection order, one per key (the first source wins), optionally only those whose mass
    is within `ppm` of the target (an engine-2 candidate of another adduct hypothesis must not enter the window)."""
    out, seen = [], set()
    for r in list(first) + list(second):
        if r is None or r['key'] in seen:
            continue
        if target is not None and abs(r['mass'] - target) > target * ppm * 1e-6:
            continue
        seen.add(r['key']); out.append(r)
    return out


def join_records(entry: Optional[dict], window_keys: Set[str], k: int, build_records: Callable, pc_records: Sequence[dict] = (),
                 target: Optional[float] = None, lib_max: Optional[float] = None, max_lib: Optional[float] = None) -> List[dict]:
    """Injection list of nb/06 with ENG_JOIN_K: engine-2's outside-window candidates first, then the PubChem records.
    build_records(smiles, keys) -> records (pc_join.build_records with chem / fragmod / bits bound)."""
    recs = []
    if not (max_lib is not None and lib_max is not None and lib_max >= max_lib):
        ix = outside_window(entry, window_keys, k)
        if ix:
            recs = [dict(r, src='eng') for r in build_records([entry['smiles'][i] for i in ix], [entry['keys'][i] for i in ix])]
    return merge_records(recs, pc_records, target)
