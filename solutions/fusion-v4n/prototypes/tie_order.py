"""Order-dependent rank features of the v1 engine: recompute them with another tie rule or another candidate order.

casmi/engine.py `_rank_norm` ranks with a stable argsort, so tied candidates get consecutive ranks in candidate order =
pool row order (same-mass rows: training structures in order of first appearance in train.parquet, then COCONUT-only
structures; generated candidates last). Candidates of one formula tie exactly on mass error, subformula score and
element-head distance, and all candidates without a library spectrum tie on lib_max, so these features tell the ranker
where a candidate sits in the pool: a provenance signal, and in a purged hold-out the purged truth is the first
training structure among the lib_max == 0 ties.

    Xn = rerank(X, names, start, mode='average')                 # order-free: ties share the average rank
    Xc = rerank(X, names, start, mode='stable', order=new_order) # stable ties under another candidate order
    o  = class2_order(pid, y, start, fg, is_train, rng)          # truth moved among the COCONUT-only rows of its group

Only numpy; covered by test_tie_order.py.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

# rank feature -> (source feature, sign so that larger = better, absolute value first)
RANK_SOURCES: Dict[str, tuple] = {
    'lib_max_rank': ('lib_max', 1, False), 'ap_rank': ('ap', 1, False), 'fz_rank': ('fz', 1, False), 'el_rank': ('el_sum', -1, False),
    'fr_rank': ('fr_int', 1, False), 'sf_rank': ('sf_int', 1, False), 'mass_ppm_rank': ('mass_ppm', -1, True),
}


def rank_norm_stable(x: np.ndarray, order: Optional[np.ndarray] = None) -> np.ndarray:
    """== engine._rank_norm: 0 = best (largest x); ties in candidate order, or in `order` (smaller = earlier)."""
    n = len(x)
    if order is None:
        o = np.argsort(-x, kind='stable')
    else:
        o = np.lexsort((order, -x))
    r = np.empty(n); r[o] = np.arange(n)
    return r / max(1, n - 1)


def rank_norm_average(x: np.ndarray) -> np.ndarray:
    """0 = best .. 1 = worst, ties share the average rank (== fe_v4 ctxlib._rank_norm)."""
    n = len(x)
    if n == 0:
        return np.zeros(0)
    o = np.argsort(-x, kind='mergesort'); xs = x[o]
    b = np.r_[True, xs[1:] != xs[:-1]]; bid = np.cumsum(b) - 1
    st = np.flatnonzero(b); en = np.r_[st[1:], n]
    avg = (st + en - 1) / 2.0
    r = np.empty(n); r[o] = avg[bid]
    return r / max(1, n - 1)


def rerank(X: np.ndarray, names: Sequence[str], start: np.ndarray, mode: str = 'average', order: Optional[np.ndarray] = None,
           features: Optional[Sequence[str]] = None) -> np.ndarray:
    """Copy of X with the rank features recomputed per molecule. mode 'stable' (with `order`, one value per row) or
    'average'. Source values are the stored float32 features."""
    names = list(names); X2 = X.copy()
    todo = [f for f in (features or RANK_SOURCES) if f in names and RANK_SOURCES[f][0] in names]
    for f in todo:
        src, sgn, ab = RANK_SOURCES[f]
        v = X[:, names.index(src)].astype(np.float64)
        v = sgn * (np.abs(v) if ab else v)
        j = names.index(f)
        for m in range(len(start) - 1):
            a, b = start[m], start[m + 1]
            if b > a:
                X2[a:b, j] = rank_norm_average(v[a:b]) if mode == 'average' else rank_norm_stable(v[a:b], None if order is None else order[a:b])
    return X2


def class2_order(pid: np.ndarray, y: np.ndarray, start: np.ndarray, fg: np.ndarray, is_train: np.ndarray, rng) -> np.ndarray:
    """Candidate order in which the truth sits where a COCONUT-only structure would: directly after a random
    COCONUT-only row of its formula group (after the group's last training row if it has none). Others keep their
    place. Returns a float order key per row (smaller = earlier)."""
    order = np.arange(len(pid), dtype=np.float64)
    for m in range(len(start) - 1):
        a, b = start[m], start[m + 1]
        yy = y[a:b]
        if b == a or not yy.any():
            continue
        t = int(np.flatnonzero(yy)[0])
        if pid[a + t] < 0:
            continue                                       # a generated truth is already last
        g = np.flatnonzero((fg[a:b] == fg[a + t]) & (pid[a:b] >= 0))
        coco = g[~is_train[a:b][g]]; coco = coco[coco != t]
        if len(coco):
            order[a + t] = a + coco[rng.randint(len(coco))] + 0.5
        else:
            order[a + t] = a + g.max() + 0.5
    return order
