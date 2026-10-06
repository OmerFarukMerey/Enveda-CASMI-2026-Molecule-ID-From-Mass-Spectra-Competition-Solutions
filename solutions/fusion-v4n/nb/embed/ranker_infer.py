"""Inference for our v1-engine ranker (drop-in for nb/06 `rank_score`): order-free ranks, confidence features, boosters.

Self-contained (numpy + lightgbm), so it can be embedded in the Kaggle notebook like nb/embed/fusion_core.py. One
molecule at a time, from what `V.run(spectra, target)` returns: C (candidates), X (94 engine features, names
`casmi.engine.FEATURES`), F (fe_v4 family features, names `names`), info.

    R = OurRanker.load(model_dir)                                   # features.json + ranker_s*.txt
    s_pub = rank_score(np.concatenate([X, F], 1), FEATURES + names) # public ranker, unchanged
    s     = choose_score(CFG['RANKER'], R, C.formula, X, F, FEATURES, names, info['lib_max'], rank_score, CFG['RANKER_LIB_GATE'])

Feature list of the packaged model (features.json, 250 names), all computed here:
  x__<name>   the engine feature <name> (82 of the 94: without the eight constant xs_* and without target_mass,
              mass_ppm, fp_bits, fr_n), where lib_max_rank, ap_rank, fz_rank, el_rank, fr_rank, sf_rank, mass_ppm_rank are
              RECOMPUTED with average ties from their float32 sources (order_free): the engine's stable-argsort ranks
              encode the pool row order (training structures first), see reviews/ranker-2026-10-02.md section 2;
  <family>__<feature>   the 158 fe_v4 family features exactly as V.run returns them;
  e__<name>   10 confidence features (confidence_features): formula-group size and share, f.z gap / z / rank / margin
              inside the candidate's formula group, library maximum of the group, number of library hits, library margin.
  j__<name>   5 join features (join_features; only in models fitted with joined lists, casmi26-ours-ranker-fs): injected flag,
              PubChem rank / 50 (-1 for other rows), f.z minus the best f.z of a row that is not injected, share of injected
              rows above that best row, log1p(number of rows that are not injected). Needs `inj` (and `pcrank`) from pc_join.
Covered by test_ranker_infer.py (no competition data). Parity with the training tables: scratch/ranker/check_infer_parity.py;
for the j__ features and the fold-safe model: scratch/rankbank2/r2_parity.py, tests/test_fs_pass.py.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# rank feature -> (source feature, sign so that larger = better, absolute value first); == prototypes/tie_order.py
RANK_SOURCES = {'lib_max_rank': ('lib_max', 1, False), 'ap_rank': ('ap', 1, False), 'fz_rank': ('fz', 1, False), 'el_rank': ('el_sum', -1, False),
                'fr_rank': ('fr_int', 1, False), 'sf_rank': ('sf_int', 1, False), 'mass_ppm_rank': ('mass_ppm', -1, True)}


def _rank_avg(x: np.ndarray) -> np.ndarray:
    """0 = best (largest x) .. 1 = worst; ties share the average rank."""
    n = len(x)
    if n == 0:
        return np.zeros(0)
    o = np.argsort(-x, kind='mergesort'); xs = x[o]
    b = np.r_[True, xs[1:] != xs[:-1]]; bid = np.cumsum(b) - 1
    st = np.flatnonzero(b); en = np.r_[st[1:], n]
    r = np.empty(n); r[o] = ((st + en - 1) / 2.0)[bid]
    return r / max(1, n - 1)


def order_free(X: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Copy of one molecule's engine features with the seven rank features recomputed order-free (float32 sources)."""
    names = list(names); X2 = np.array(X, np.float32, copy=True)
    for f, (src, sgn, ab) in RANK_SOURCES.items():
        if f in names and src in names:
            v = X2[:, names.index(src)].astype(np.float64)
            X2[:, names.index(f)] = _rank_avg(sgn * (np.abs(v) if ab else v))
    return X2


def _groups(labels: Sequence) -> np.ndarray:
    ix: Dict = {}
    return np.array([ix.setdefault(l, len(ix)) for l in labels], np.int64)


def confidence_features(X: np.ndarray, names: Sequence[str], formulas: Sequence[str]) -> Dict[str, np.ndarray]:
    """The 10 `e__*` confidence features of one molecule (== prototypes/ranker_rows.extra_features without popularity)."""
    names = list(names); n = len(X)
    fz = X[:, names.index('fz')].astype(np.float64) * 100.0
    lib = X[:, names.index('lib_max')].astype(np.float64)
    g = _groups(['?' if f is None else f for f in formulas]); G = int(g.max()) + 1 if n else 0
    size = np.bincount(g, minlength=G).astype(float)
    gmax = np.full(G, -np.inf); np.maximum.at(gmax, g, fz)
    gsum = np.bincount(g, weights=fz, minlength=G); gmean = gsum / np.maximum(size, 1)
    gss = np.bincount(g, weights=(fz - gmean[g]) ** 2, minlength=G)
    gsd = np.sqrt(gss / np.maximum(size - 1, 1))                           # sample standard deviation (pandas default)
    z = np.where((gsd[g] > 1e-9) & (size[g] > 1), (fz - gmean[g]) / np.where(gsd[g] > 1e-9, gsd[g], 1.0), 0.0)
    rank = np.zeros(n); top2 = np.zeros(G)
    for j in range(G):
        idx = np.flatnonzero(g == j)
        if len(idx) > 1:
            rank[idx] = _rank_avg(fz[idx]); top2[j] = np.sort(fz[idx])[-2]
        else:
            top2[j] = fz[idx[0]]
    libg = np.zeros(G); np.maximum.at(libg, g, lib)
    hit = (lib > 0).astype(float)
    lib2 = float(np.sort(lib)[-2]) if n > 1 else 0.0
    E = {'fg_size': np.log(size[g]), 'fg_share': size[g] / max(n, 1), 'fz_fg_gap': (fz - gmax[g]) / 100.0, 'fz_fg_z': z, 'fz_fg_rank': rank,
         'fz_fg_margin': (gmax[g] - top2[g]) / 100.0, 'lib_fg_max': libg[g], 'n_lib_hit': np.full(n, np.log1p(hit.sum())),
         'n_lib_hit_fg': np.log1p(np.bincount(g, weights=hit, minlength=G))[g], 'lib_margin': np.full(n, (lib.max() if n else 0.0) - lib2)}
    return {k: np.asarray(v, np.float32) for k, v in E.items()}


JFEATS = ['inj', 'pcrank', 'fz_rel_pool', 'inj_above_pool', 'n_pool']


def join_features(X: np.ndarray, names: Sequence[str], inj: Optional[np.ndarray] = None, pcrank: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """The 5 `j__*` features of one molecule (== scratch/rankbank/rb_lib.join_features): how a row entered the list.
    inj: True for joined PubChem rows (None = no joined row); pcrank: 0-based rank of a joined row in the PubChem list."""
    n = len(X); inj = np.zeros(n, bool) if inj is None else np.asarray(inj).astype(bool)
    pcrank = np.full(n, -1) if pcrank is None else np.asarray(pcrank)
    f = X[:, list(names).index('fz')].astype(np.float64)
    if n == 0:
        return {k: np.zeros(0, np.float32) for k in JFEATS}
    best_pool = f[~inj].max() if (~inj).any() else f.max()
    J = {'inj': inj.astype(np.float64), 'pcrank': np.where(inj, np.maximum(pcrank, 0) / 50.0, -1.0), 'fz_rel_pool': f - best_pool,
         'inj_above_pool': np.full(n, float((f[inj] > best_pool).mean()) if inj.any() else 0.0), 'n_pool': np.full(n, np.log1p((~inj).sum()))}
    return {k: np.asarray(v, np.float32) for k, v in J.items()}


class OurRanker:
    def __init__(self, features: Sequence[str], boosters: Sequence):
        self.features = list(features); self.boosters = list(boosters)

    @classmethod
    def load(cls, model_dir: str) -> 'OurRanker':
        import lightgbm as lgb
        feats = json.load(open(os.path.join(model_dir, 'features.json')))
        files = sorted(glob.glob(os.path.join(model_dir, 'ranker_s*.txt')))
        assert files, f'no ranker_s*.txt in {model_dir}'
        return cls(feats, [lgb.Booster(model_file=f) for f in files])

    def matrix(self, formulas, X, F, names_x, names_f, inj=None, pcrank=None, Xo=None) -> np.ndarray:
        """Feature matrix of one molecule in the model's column order. Xo: engine features whose rank features are already
        order-free (pc_join.order_free_join for lists with joined rows); inj / pcrank: for the j__ features."""
        names_x = list(names_x); names_f = list(names_f)
        if Xo is None:
            Xo = order_free(X, names_x)
        E = confidence_features(Xo, names_x, formulas) if any(f.startswith('e__') for f in self.features) else {}
        Jf = join_features(Xo, names_x, inj, pcrank) if any(f.startswith('j__') for f in self.features) else {}
        px = {n: i for i, n in enumerate(names_x)}; pf = {n: i for i, n in enumerate(names_f)}
        cols = []
        for f in self.features:
            if f.startswith('x__'):
                cols.append(Xo[:, px[f[3:]]])
            elif f.startswith('e__'):
                cols.append(E[f[3:]])
            elif f.startswith('j__'):
                cols.append(Jf[f[3:]])
            else:
                cols.append(np.asarray(F[:, pf[f]], np.float32))
        return np.ascontiguousarray(np.stack(cols, 1), dtype=np.float32)

    def score(self, formulas, X, F, names_x, names_f, threads: int = 4, inj=None, pcrank=None, Xo=None) -> np.ndarray:
        M = self.matrix(formulas, X, F, names_x, names_f, inj=inj, pcrank=pcrank, Xo=Xo)
        return np.mean([b.predict(M, num_threads=threads) for b in self.boosters], axis=0)


def _z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    if len(v) < 2:
        return np.zeros(len(v))
    s = v.std(ddof=1)
    return (v - v.mean()) / s if s > 1e-9 else np.zeros(len(v))


def choose_score(mode: str, ranker: Optional[OurRanker], formulas, X, F, names_x, names_f, lib_max: float,
                 public_score: Callable[[np.ndarray, List[str]], np.ndarray], lib_gate: Optional[float] = None) -> np.ndarray:
    """The base-list score of one molecule (replaces the `rank_score(...)` call of nb/06).
    mode 'public'       the public ranker exactly as today (lib_gate is ignored);
    mode 'public_free'  the public ranker on order-free rank features;
    mode 'ours'         our ranker;
    mode 'blend'        z(ours) + z(public ranker on order-free rank features).
    lib_gate = tau: molecules with library evidence (lib_max >= tau, class-1-like) keep the public ranker as today,
    whatever the mode; None = the mode applies to every molecule."""
    names = list(names_x) + list(names_f)
    if mode == 'public' or (lib_gate is not None and lib_max >= lib_gate):
        return public_score(np.concatenate([X, F], 1), names)
    if mode == 'public_free':
        return public_score(np.concatenate([order_free(X, names_x), F], 1), names)
    ours = ranker.score(formulas, X, F, names_x, names_f)
    if mode == 'ours':
        return ours
    if mode == 'blend':
        return _z(ours) + _z(public_score(np.concatenate([order_free(X, names_x), F], 1), names))
    raise ValueError(mode)
