"""Leaderboard calibration of the mixed regime: fit type weights to logged submissions and predict a new config's public-LB difference with an interval.

The mixed regime (mixed_regime.py, reviews/mixed-regime-2026-10-02.md) gives per-molecule reciprocal ranks of a fusion
config for a few molecule TYPES (c1, c2, dA, dB with / without a PubChem record). The public leaderboard gives one
number per submission. This module holds the arithmetic that connects the two (numpy only, no competition data):

    type_stats(rr_cfg, rr_base)        per type: mean paired difference, its standard error, per-molecule sd, n
    fit_weights(D, y)                  non-negative type weights with sum <= 1 (the rest = molecules no channel reaches)
    weight_set(D, y, tol)              every grid weight vector that fits within `tol` of the best: how well determined
    leave_one_out(D, y)                fit without a submission, predict it; RMSE against the "no change" RMSE
    calibrate(D, y, names, types)      -> Calibration (weights, weight set, residuals, leave-one-out table, skill)
    predict(cal, rr_cfg, rr_base)      -> dict(delta, lo, hi, ...): predicted LB difference to the base and an interval

The interval has three parts, all reported: (1) the local sampling error of the type means, (2) the public leaderboard's
own sampling error (about 130 molecules drawn from the mixture: per-molecule sd / sqrt(n_public)), (3) the spread over
all weight vectors that fit the logged submissions equally well. The half-width is never below the leave-one-out RMSE
of the calibration (its measured error on held-out submissions), and `reliable` is False unless the leave-one-out error
is clearly below the error of predicting "no change" (cal.skill >= MIN_SKILL): a fit is then not a prediction.
Result on the 13 replayable submissions (reviews/lbcal-2026-10-04.md): skill +0.04 with five type weights, +0.22 with
the single weight of PubChem-only answers (0.065, range 0.045 to 0.085); neither reproduces the two drops (N 200, gate 0.8).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

N_PUBLIC = 130          # public-LB molecules: one molecule at rank 1 is worth about 0.0076
MIN_SKILL = 0.25        # leave-one-out SSE at most 75% of the "no change" SSE before a prediction is called reliable


def type_stats(rr_cfg: Dict[str, Dict[str, float]], rr_base: Dict[str, Dict[str, float]]) -> Dict[str, dict]:
    """{type: dict(mean, se, sd, n)} of the paired per-molecule difference cfg - base (molecules present in both)."""
    out = {}
    for t, a in rr_cfg.items():
        b = rr_base.get(t, {})
        d = np.array([a[m] - b[m] for m in a if m in b], float)
        if len(d) == 0:
            continue
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        out[t] = dict(mean=float(d.mean()), se=sd / np.sqrt(len(d)) if len(d) > 1 else float('nan'), sd=sd, n=int(len(d)))
    return out


def _project(w: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w) <= cap}."""
    w = np.maximum(w, 0.0)
    if w.sum() <= cap:
        return w
    u = np.sort(w)[::-1]; css = np.cumsum(u) - cap
    k = np.nonzero(u - css / (np.arange(len(u)) + 1) > 0)[0][-1]
    return np.maximum(w - css[k] / (k + 1.0), 0.0)


def fit_weights(D: np.ndarray, y: np.ndarray, cap: float = 1.0, iters: int = 20000) -> np.ndarray:
    """argmin |D w - y|^2 over w >= 0, sum(w) <= cap (projected gradient; D: submissions x types)."""
    D = np.asarray(D, float); y = np.asarray(y, float)
    if D.size == 0:
        return np.zeros(D.shape[1] if D.ndim == 2 else 0)
    A = D.T @ D; b = D.T @ y
    L = float(np.linalg.eigvalsh(A).max()) if A.size else 0.0
    w = np.zeros(D.shape[1])
    if L <= 0:
        return w
    v = w.copy(); tk = 1.0
    for _ in range(iters):                         # FISTA
        w2 = _project(v - (A @ v - b) / L, cap)
        t2 = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        v = w2 + (tk - 1) / t2 * (w2 - w);
        if np.abs(w2 - w).max() < 1e-12:
            w = w2; break
        w, tk = w2, t2
    return w


def sse(D, y, w) -> float:
    r = np.asarray(D, float) @ np.asarray(w, float) - np.asarray(y, float)
    return float(r @ r)


def simplex_grid(k: int, step: float = 0.05, cap: float = 1.0) -> np.ndarray:
    """All weight vectors of length k on a grid with sum <= cap."""
    n = int(round(cap / step))
    return np.array([c for c in itertools.product(range(n + 1), repeat=k) if sum(c) <= n], float) * step


def weight_set(D, y, tol: float, step: float = 0.05, cap: float = 1.0) -> np.ndarray:
    """Grid weight vectors whose SSE is within `tol` of the constrained optimum (tol = sigma^2 is one unit of chi-square
    for a leaderboard noise of sigma per point)."""
    D = np.asarray(D, float); y = np.asarray(y, float)
    G = simplex_grid(D.shape[1], step, cap)
    s = ((G @ D.T - y) ** 2).sum(1)
    best = min(float(s.min()), sse(D, y, fit_weights(D, y, cap)))
    return G[s <= best + tol]


def leave_one_out(D, y, cap: float = 1.0) -> np.ndarray:
    """Prediction of every row from a fit without it."""
    D = np.asarray(D, float); y = np.asarray(y, float); out = np.zeros(len(y))
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        out[i] = D[i] @ fit_weights(D[k], y[k], cap)
    return out


@dataclass
class Calibration:
    types: List[str]
    names: List[str]
    weights: np.ndarray
    wset: np.ndarray                      # equally well fitting weight vectors (rows)
    fitted: np.ndarray
    loo: np.ndarray
    y: np.ndarray
    sigma: float
    rmse_fit: float = field(init=False)
    rmse_loo: float = field(init=False)
    rmse_null: float = field(init=False)
    skill: float = field(init=False)      # 1 - LOO SSE / "no change" SSE; <= 0 means: not better than predicting no change

    def __post_init__(self):
        self.rmse_fit = float(np.sqrt(np.mean((self.fitted - self.y) ** 2)))
        self.rmse_loo = float(np.sqrt(np.mean((self.loo - self.y) ** 2)))
        self.rmse_null = float(np.sqrt(np.mean(self.y ** 2)))
        self.skill = float(1.0 - np.sum((self.loo - self.y) ** 2) / max(np.sum(self.y ** 2), 1e-18))

    def ranges(self) -> Dict[str, tuple]:
        return {t: (float(self.wset[:, j].min()), float(self.wset[:, j].max())) for j, t in enumerate(self.types)}

    def table(self) -> List[dict]:
        return [dict(name=n, lb=float(a), fitted=float(f), loo=float(l), resid=float(f - a), loo_err=float(l - a))
                for n, a, f, l in zip(self.names, self.y, self.fitted, self.loo)]


def calibrate(D, y, names: Sequence[str], types: Sequence[str], sigma: float = 0.01, step: float = 0.05, cap: float = 1.0) -> Calibration:
    """D[i, j] = mean paired difference to the base of submission i on type j; y[i] = its leaderboard difference."""
    D = np.asarray(D, float); y = np.asarray(y, float)
    w = fit_weights(D, y, cap)
    return Calibration(list(types), list(names), w, weight_set(D, y, sigma ** 2, step, cap), D @ w, leave_one_out(D, y, cap), y, sigma)


def predict(cal: Calibration, rr_cfg: Dict[str, Dict[str, float]], rr_base: Dict[str, Dict[str, float]], n_public: int = N_PUBLIC,
            z: float = 1.0) -> dict:
    """Predicted public-LB difference of a config to the base, with a +-z interval.
    Types of the calibration that are missing from rr_cfg count as 'no change'."""
    st = type_stats(rr_cfg, rr_base)
    mu = np.array([st[t]['mean'] if t in st else 0.0 for t in cal.types])
    se = np.array([st[t]['se'] if t in st and np.isfinite(st[t]['se']) else 0.0 for t in cal.types])
    sd = np.array([st[t]['sd'] if t in st else 0.0 for t in cal.types])
    w = cal.weights
    delta = float(w @ mu)
    se_local = float(np.sqrt(((w * se) ** 2).sum()))
    # one public molecule: type j with probability w_j (else unreachable, change 0); variance of its change
    var1 = float((w * (sd ** 2 + mu ** 2)).sum() - delta ** 2)
    se_public = float(np.sqrt(max(var1, 0.0) / n_public))
    alt = cal.wset @ mu if len(cal.wset) else np.array([delta])
    lo_w, hi_w = float(min(alt.min(), delta)), float(max(alt.max(), delta))
    half = z * max(float(np.sqrt(se_local ** 2 + se_public ** 2)), cal.rmse_loo)     # never narrower than the held-out error
    reliable = cal.skill >= MIN_SKILL
    return dict(delta=delta, lo=lo_w - half, hi=hi_w + half, se_local=se_local, se_public=se_public, weight_lo=lo_w, weight_hi=hi_w,
                reliable=bool(reliable), skill=cal.skill, rmse_loo=cal.rmse_loo, rmse_null=cal.rmse_null,
                per_type={t: dict(weight=float(w[j]), mean=float(mu[j]), se=float(se[j])) for j, t in enumerate(cal.types)})


def public_z(observed: float, expected: float, w: Sequence[float], sd: Sequence[float], mu: Sequence[float], se: Sequence[float],
             n_public: int = N_PUBLIC) -> float:
    """How many standard errors the observed LB difference is from the regime's expectation, counting the local error and
    the draw of the public molecules. Large |z| = the regime lacks a mechanism; it is not an unlucky draw."""
    w, sd, mu, se = (np.asarray(x, float) for x in (w, sd, mu, se))
    var1 = float((w * (sd ** 2 + mu ** 2)).sum() - expected ** 2)
    s = np.sqrt(max(var1, 0.0) / n_public + ((w * se) ** 2).sum())
    return float((observed - expected) / s) if s > 0 else float('nan')
