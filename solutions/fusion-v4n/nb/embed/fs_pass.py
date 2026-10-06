"""Second v1-engine pass with the fold-safe fingerprint model and a ranker fitted to it (RANKER 'fs' / 'fs_avg'; off by default).

Why. The public FPNets and the public ranker were fitted on molecules the FPNets had memorised; every molecule of the hidden test is
unseen by them. The fold-safe FPNet (dmitriigluzdov/casmi26-fold-safe-fpnet) gives features in the unseen regime for our hold-out
molecules, and the ranker of casmi26-ours-ranker-fs is fitted on exactly those features (reviews/fsranker-2026-10-06.md).

What. A molecule whose library evidence is below the gate (lib_max < RANKER_LIB_GATE) is run through the engine a second time with
  * the engine FP bank          = fpnet_fold0.pt
  * the three model_views slots = fpnet_fold0.pt (one forward pass, repeated; as scratch/ranker/run_engine.py built the training rows)
  * the same joined PubChem rows as the first pass
and its base list is ordered by the fitted ranker ('fs') or by the average of the two chains' ranks ('fs_avg': public chain as in the
first pass, fold-safe chain, half weight each). Molecules at or above the gate keep the first pass untouched. Nothing of the public
`casmi` package is edited: `swapped` replaces three attributes of the engine / V1FE objects and restores them.

    with swapped(E, V, fs_bank, fs_views):
        C2, X2, F2, names2, info2 = V.run(spectra, target)
    s2 = fs_score(ranker, C2.formula, X2, F2, FEATURES, names2, inj2, pcrank2)
    rows = entries(C2.key, smiles2, C2.formula, pids2, s2)                     # key-deduplicated, best first
    rows = rank_average(entries(<first pass>), rows)                           # 'fs_avg' only
    smis, keys, scs, forms, pids = to_base(rows, TOPN)

Pure helpers (entries, rank_average, to_base, fs_score with a stub ranker, swapped with stub objects): tests/test_fs_pass.py.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional, Sequence

import numpy as np

MODES = ('fs', 'fs_avg')


@contextmanager
def swapped(E, V, fs_bank, fs_views):
    """Engine bank and model_views banks replaced by the fold-safe ones inside the block; one model fills the A, B and D slots."""
    had = 'model_views' in V.__dict__
    old = (E.bank, V.banks, V.__dict__.get('model_views'))
    base_mv = type(V).model_views

    def model_views(prep):
        zs, es, zm, em, modes = base_mv(V, prep)
        rep = lambda x: np.concatenate([x, x, x])
        return rep(zs), rep(es), rep(zm), rep(em), modes
    E.bank = fs_bank; V.banks = list(fs_views); V.model_views = model_views
    try:
        yield
    finally:
        E.bank, V.banks = old[0], old[1]
        if had:
            V.model_views = old[2]
        else:
            del V.__dict__['model_views']


def fs_score(ranker, formulas, X, F, names_x, names_f, inj: Optional[np.ndarray] = None, pcrank: Optional[np.ndarray] = None, threads: int = 4) -> np.ndarray:
    """Fitted ranker on one molecule: rank features order-free with joined rows last among ties (pc_join.order_free_join), then the
    model's columns (x__, family, j__)."""
    import pc_join
    Xo = pc_join.order_free_join(X, list(names_x), inj)
    return np.asarray(ranker.score(formulas, X, F, list(names_x), list(names_f), threads=threads, inj=inj, pcrank=pcrank, Xo=Xo), np.float64)


def entries(keys: Sequence, smiles: Sequence, formulas: Sequence, pids: Sequence, score: Sequence) -> List[tuple]:
    """Key-deduplicated candidate list by descending score (stable): [(key, smiles, formula, pool id, score)] (nb/06's order)."""
    out, seen = [], set()
    for i in np.argsort(-np.asarray(score, np.float64), kind='stable'):
        k = keys[i]
        if k in seen:
            continue
        seen.add(k); out.append((k, smiles[i], formulas[i], int(pids[i]), float(score[i])))
    return out


def _z(v):
    v = np.asarray(v, np.float64)
    if len(v) < 2:
        return np.zeros(len(v))
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else np.zeros(len(v))


def rank_average(pub: List[tuple], fs: List[tuple], w: float = 0.5) -> List[tuple]:
    """Half weight to each chain: a key's rank is (1 - w) x rank in the public chain's list + w x rank in the fold-safe chain's list
    (1-based; a key missing from a list gets that list's length + 1). Ties: the fold-safe rank, then the public rank. The row
    (SMILES, formula, pool id) is the fold-safe pass's, or the public pass's for keys only it holds. Scores: the two lists' z-scored
    score profiles (sorted, best first), averaged with the same weights and handed out in the new order, so the list keeps a ranker-like
    score shape for the popularity prior and the forward-model terms of fusion_core."""
    rp = {e[0]: i + 1 for i, e in enumerate(pub)}; rf = {e[0]: i + 1 for i, e in enumerate(fs)}
    row = {e[0]: e for e in pub}; row.update({e[0]: e for e in fs})
    keys = list(dict.fromkeys([e[0] for e in fs] + [e[0] for e in pub]))
    np_, nf = len(pub) + 1, len(fs) + 1
    keys.sort(key=lambda k: ((1 - w) * rp.get(k, np_) + w * rf.get(k, nf), rf.get(k, nf), rp.get(k, np_)))
    n = len(keys)

    def profile(lst):
        z = _z([e[4] for e in lst]) if lst else np.zeros(0)
        return np.r_[z, np.full(max(0, n - len(z)), z[-1] if len(z) else 0.0)][:n]
    prof = (1 - w) * profile(pub) + w * profile(fs)
    prof = np.minimum.accumulate(prof) if n else prof                # non-increasing whatever the padding did
    return [(k, row[k][1], row[k][2], row[k][3], float(prof[i])) for i, k in enumerate(keys)]


def to_base(rows: List[tuple], topn: int):
    """-> (smiles, keys, scores, formulas, pool ids) of the first topn rows: the five lists nb/06 stores per molecule."""
    r = rows[:topn]
    return [e[1] for e in r], [e[0] for e in r], [e[4] for e in r], [e[2] for e in r], [e[3] for e in r]
