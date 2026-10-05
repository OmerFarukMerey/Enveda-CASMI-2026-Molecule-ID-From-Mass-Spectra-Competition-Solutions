"""Confidence gate for the PubChem join: list-shape features of one v1 base list (engine pass WITHOUT injection) and a 9-feature logistic model for "the answer is a candidate of this list"; numpy only, coefficients in this file.

Why (reviews/confgate-2026-10-04.md): the join helps where the answer is outside the pool and hurts where the pool
already had it. `lib_max` does not tell these apart below 0.7; the shape of the base list does (how far the first
candidate is ahead, how strong its evidence is).

    f = list_features(score, X, names_x, F, names_f, formulas)    # one molecule, no injected rows
    p = p_present(f)                                              # 0..1, high = the list probably holds the answer
    n = join_depth(p, lib_max, conf=0.5, n=100, max_lib=0.7)      # 0 = leave the list alone, else how many to inject

Config (nb/00_config.py; all off by default): PC_JOIN_CONF (join only when p < this), PC_JOIN_CONF_DEEP + PC_JOIN_N_DEEP
(a deeper join for the least confident lists). Operating points measured in the mixed regime, against the join for
every list with lib_max < 0.7 at N 100 (share of the truth-outside-the-pool gain kept / share of truth-in-pool lists
still touched): conf 0.3: 77% / 0.12; 0.5: 88% / 0.23; 0.7: 94% / 0.40.

`score` is the public ranker's score as the notebook computes it (stable ties); the model was fitted on lists in which
a present answer sits where a COCONUT-only pool row sits (the class-2 emulation of prototypes/tie_order.py), never on
lists where the answer is first among ties. No rank feature with pool-order ties is used. Covered by test_conf_gate.py.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Sequence

import numpy as np

# evidence of the first candidate (engine features; none of the seven order-coded *_rank features)
TOP1_X = ['fz_z', 'fz_gap', 'fz_softmax', 'fz_norm', 'fp_tan', 'fp_cos', 'ap', 'best_tan', 'lib_max', 'fr_int', 'sf_int', 'formula_share',
          'is_gen', 'el_sum', 'agree_ap', 'ap_gap', 'fr_gap']
# evidence of the first candidate (fe_v4 families; average ties)
TOP1_F = ['fragnet__fn_rank', 'fragnet__fn_score', 'frag2__mix_rank', 'analog_struct__st_rank']
# list-level engine features (the same value on every row)
MOL_X = ['n_cand', 'n_formula', 'fz_grp_ent', 'fz_grp_gap', 'ap_grp_max', 'top_sim', 'n_analog_hi', 'lib_grp_max', 'n_query', 'n_gen', 'q_npeaks',
         'n_pos', 'n_neg']


def _groups(formulas: Sequence) -> np.ndarray:
    seen: Dict[object, int] = {}
    return np.array([seen.setdefault('?' if f is None else f, len(seen)) for f in formulas], np.int64)


def list_features(score: np.ndarray, X: np.ndarray, names_x: Sequence[str], F: Optional[np.ndarray] = None,
                  names_f: Optional[Sequence[str]] = None, formulas: Optional[Sequence] = None) -> Dict[str, float]:
    """Features of one candidate list. `score` one value per row (larger = better), `X` the engine features, `F` the
    family features (optional: their features are NaN without it), `formulas` a formula string or group id per row."""
    score = np.asarray(score, np.float64); n = len(score)
    if n == 0:
        return {}
    px = {c: i for i, c in enumerate(names_x)}
    o = np.argsort(-score, kind='stable'); top = int(o[0]); s = score[o]
    sd = float(score.std())
    f = dict(s1=float(s[0]), m12=float(s[0] - s[1]) if n > 1 else 5.0, m15=float(s[0] - s[min(4, n - 1)]), m1_10=float(s[0] - s[min(9, n - 1)]),
             s_mean5=float(s[:5].mean()), s_sd=sd, z1=float((s[0] - score.mean()) / sd) if sd > 1e-9 else 0.0,
             n_close=float(np.log1p((s[0] - s < 1.0).sum() - 1)), log_n=float(np.log(n)))
    for c in TOP1_X:
        f['t1_' + c] = float(X[top, px[c]]) if c in px else float('nan')
    for c in MOL_X:
        f[c] = float(X[top, px[c]]) if c in px else float('nan')
    pf = {c: i for i, c in enumerate(names_f)} if (F is not None and names_f is not None) else {}
    for c in TOP1_F:
        f['t1_' + c] = float(F[top, pf[c]]) if c in pf else float('nan')
    fz = X[:, px['fz']].astype(np.float64) * 100.0
    f['t1_is_fz_top'] = float(fz[top] >= fz.max() - 1e-9)
    f['fz_z_max'] = float(X[:, px['fz_z']].max())
    g = _groups(formulas) if formulas is not None else np.zeros(n, np.int64)      # the first candidate inside its formula group
    idx = np.flatnonzero(g == g[top]); v = fz[idx]
    f['t1_fg_size'] = float(np.log(len(idx)))
    if len(idx) > 1:
        gsd = float(v.std(ddof=1))
        f['t1_fz_fg_z'] = float((fz[top] - v.mean()) / gsd) if gsd > 1e-9 else 0.0
        f['t1_fz_fg_margin'] = float((v.max() - np.sort(v)[-2]) / 100.0)
        f['t1_fz_fg_gap'] = float((fz[top] - v.max()) / 100.0)
    else:
        f['t1_fz_fg_z'] = 0.0; f['t1_fz_fg_margin'] = 0.0; f['t1_fz_fg_gap'] = 0.0
    return f


# Fitted by scratch/confgate/cg_fit.py (2026-10-04) on 3,711 hold-out lists of molecules outside the mixed regime:
# present = 2,530 class-2-like lists scored with the class-2 emulation + 767 class-1-like lists as run, absent = 414
# truth-removed runs; classes weighted equally. p = sigmoid(sum(coef * (clip(x, clip_lo, clip_hi) - mean) / scale) + intercept).
# Regime AUC 0.82 (natural-product catalogue structures 0.73, others 0.95). s1 = score of the first candidate,
# m1_10 = its margin to rank 10; t1_* = engine features of the first candidate; n_formula, q_npeaks = list-level.
MODEL = {
    'features': ['s1', 't1_is_gen', 't1_formula_share', 'm1_10', 't1_fp_cos', 't1_best_tan', 't1_lib_max', 'n_formula', 'q_npeaks'],
    'mean': [2.62653, 0.0274859, 0.683155, 5.45236, 0.426242, 0.749482, 0.207599, 4.60119, 4.07086],
    'scale': [1.56866, 0.163494, 0.330545, 2.51219, 0.09189, 0.231798, 0.39066, 3.38422, 1.33176],
    'coef': [1.04922, -0.622781, 0.700641, 0.795762, -0.542309, 0.469465, -0.486874, 0.351933, -0.206127],
    'intercept': 1.00542,
    'fill': [2.70611, 0, 0.815789, 5.50475, 0.438187, 0.811395, 0, 4, 3.93183],
    'clip_lo': [-1.62969, 0, 0.0123457, 0, 0.170646, 0.215596, 0, 1, 1.09861],
    'clip_hi': [5.58732, 1, 1, 11.4667, 0.621297, 1, 1, 15, 8.10198],
}


def load_model(path: Optional[str] = None) -> dict:
    """The built-in model, or one read from a JSON file with the same keys (a refit; no pickle)."""
    M = MODEL
    if path:
        with open(path, encoding='utf-8') as fh:
            M = json.load(fh)
    n = len(M['features'])
    assert all(len(M[k]) == n for k in ('mean', 'scale', 'coef', 'fill')), 'confidence model: array lengths differ'
    return M


def p_present(feats: Dict[str, float], M: Optional[dict] = None) -> float:
    """Probability-like score that the list holds the answer; a missing or non-finite feature takes the model's fill value.
    An empty list (no candidates) returns 0.0: nothing to protect."""
    if not feats:
        return 0.0
    M = MODEL if M is None else M
    x = np.array([feats.get(c, np.nan) for c in M['features']], np.float64)
    fill = np.asarray(M['fill'], np.float64); x = np.where(np.isfinite(x), x, fill)
    if 'clip_lo' in M:
        x = np.clip(x, np.asarray(M['clip_lo'], np.float64), np.asarray(M['clip_hi'], np.float64))
    z = float(np.dot((x - np.asarray(M['mean'])) / np.asarray(M['scale']), np.asarray(M['coef'])) + M['intercept'])
    return float(1.0 / (1.0 + np.exp(-max(-50.0, min(50.0, z)))))


def join_depth(p: float, lib_max: float, conf: Optional[float], n: int, max_lib: Optional[float] = None,
               conf_deep: Optional[float] = None, n_deep: int = 0) -> int:
    """How many PubChem candidates to inject for one molecule. 0 when lib_max >= max_lib or when the list is confident
    (p >= conf); `n_deep` when p < conf_deep; `n` otherwise. conf None = no confidence gate (today's behaviour)."""
    if max_lib is not None and lib_max >= max_lib:
        return 0
    if conf is None:
        return int(n)
    if p >= conf:
        return 0
    if conf_deep is not None and n_deep > 0 and p < conf_deep:
        return int(n_deep)
    return int(n)


def quantile_threshold(ps: Sequence[float], share: float) -> float:
    """Threshold that flags the `share` least confident lists of a run (label-free; a guard against a shift of the
    score scale between the fold-safe models of the study and the notebook's models)."""
    ps = np.sort(np.asarray(ps, np.float64))
    if len(ps) == 0 or share <= 0:
        return 0.0
    return float(ps[min(len(ps) - 1, int(np.ceil(share * len(ps))) - 1)]) + 1e-12
