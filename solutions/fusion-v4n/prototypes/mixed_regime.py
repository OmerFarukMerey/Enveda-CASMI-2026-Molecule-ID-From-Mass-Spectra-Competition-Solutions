"""Mixed validation regime: mixtures of molecule types (class 1, class 2 in the pool, truth outside the pool) scored against the LB points.

Every earlier local validation set had the true structure in the main candidate pool, so channels that bring answers
from outside (engine-2's own candidates, the PubChem channel) could only disturb. A regime here is a set of molecule
TYPES, each with its own per-molecule reciprocal ranks for every fusion config, and a weight vector over the types.
This module holds the arithmetic only (no competition data, no engine):

    base_from_rows(...)            candidate rows of the v1 engine -> a BASE entry of fusion_core (key-deduplicated)
    reciprocal_rank(keys, truth)   RR@25 of one final list
    mixture_mean / mixture_delta   weighted mean over types, with a molecule-clustered standard error
    simplex_grid(types, step)      every weight vector on a grid
    lb_agreement(lb, local)        Spearman, correctly ordered pairs with an LB gap >= min_gap
    scan_mixtures(...)             all of the above for every weight vector, plus named sign checks

Used by scratch/mixed/regime.py; reviews/mixed-regime-2026-10-02.md describes the data behind the types.
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def base_from_rows(smiles: Sequence[str], keys: Sequence[str], formulas: Sequence[str], pids: Sequence[int], score: Sequence[float],
                   lib_max: float, topn: int = 80) -> list:
    """[smiles, keys, lib_max, scores, formulas, pool ids] in descending score (stable), first occurrence of a key only:
    the BASE[mid] entry nb/06_base_lists.py writes."""
    seen, idx = set(), []
    for i in np.argsort(-np.asarray(score, float), kind='stable'):
        k = keys[i]
        if k in seen:
            continue
        seen.add(k); idx.append(int(i))
        if len(idx) >= topn:
            break
    return [[smiles[i] for i in idx], [keys[i] for i in idx], float(lib_max), [float(score[i]) for i in idx],
            [formulas[i] for i in idx], [int(pids[i]) for i in idx]]


def reciprocal_rank(keys: Sequence[Optional[str]], truth: str, max_rank: int = 25) -> float:
    """1 / position of the first entry equal to `truth` among the first max_rank entries, else 0."""
    for r, k in enumerate(list(keys)[:max_rank], 1):
        if k == truth:
            return 1.0 / r
    return 0.0


def _norm(weights: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(weights.values()))
    if s <= 0:
        raise ValueError('weights sum to zero')
    return {t: w / s for t, w in weights.items()}


def mixture_mean(values: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Tuple[float, float]:
    """values[type][molecule] -> (weighted mean of the type means, standard error clustered by molecule).
    A molecule that appears in several types (its truth-in-pool and truth-removed versions) is one cluster.
    Types with weight 0 are ignored; a type named in `weights` but absent from `values` counts as all zeros
    (an 'unreachable' type: no channel can find the answer)."""
    w = _norm(weights)
    mean, u = 0.0, {}
    for t, wt in w.items():
        if wt == 0 or t not in values or not values[t]:
            continue
        v = values[t]; x = np.array(list(v.values()), float); mu = x.mean(); n = len(x)
        mean += wt * mu
        for m, xi in v.items():
            u[m] = u.get(m, 0.0) + wt * (xi - mu) / n
    if len(u) < 2:
        return mean, float('nan')
    a = np.array(list(u.values()))
    return float(mean), float(np.sqrt((a ** 2).sum() * len(a) / (len(a) - 1)))


def mixture_delta(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Tuple[float, float]:
    """Paired difference a - b under the mixture: (mean, clustered standard error)."""
    d = {t: {m: a[t][m] - b[t][m] for m in a[t]} for t in a if t in b}
    return mixture_mean(d, weights)


def simplex_grid(types: Sequence[str], step: float = 0.1, fixed: Optional[Dict[str, float]] = None) -> List[Dict[str, float]]:
    """Every weight vector over `types` whose entries are multiples of `step` and sum to 1 - sum(fixed)."""
    fixed = dict(fixed or {})
    free = [t for t in types if t not in fixed]
    n = int(round((1.0 - sum(fixed.values())) / step))
    out = []
    for c in itertools.product(range(n + 1), repeat=max(len(free) - 1, 0)):
        if sum(c) > n:
            continue
        w = dict(fixed)
        for t, k in zip(free[:-1], c):
            w[t] = k * step
        if free:
            w[free[-1]] = (n - sum(c)) * step
        out.append({t: round(w.get(t, 0.0), 10) for t in types})
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    def rank(x):
        x = np.asarray(x, float); o = np.argsort(x, kind='stable'); r = np.empty(len(x)); r[o] = np.arange(len(x))
        for v in np.unique(x):                      # average ties
            m = x == v
            r[m] = r[m].mean()
        return r
    ra, rb = rank(a), rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def pair_agreement(lb: Sequence[float], local: Sequence[float], min_gap: float = 0.008) -> Tuple[int, int]:
    """(pairs ordered like the LB, pairs considered) over all pairs whose LB values differ by at least min_gap."""
    ok = tot = 0
    for i, j in itertools.combinations(range(len(lb)), 2):
        if abs(lb[i] - lb[j]) >= min_gap - 1e-12:
            tot += 1
            ok += int(np.sign(lb[i] - lb[j]) == np.sign(local[i] - local[j]))
    return ok, tot


def lb_agreement(lb: Sequence[float], local: Sequence[float], min_gap: float = 0.008) -> Dict[str, float]:
    ok, tot = pair_agreement(lb, local, min_gap)
    return dict(spearman=spearman(lb, local), pairs_ok=ok, pairs=tot)


def scan_mixtures(type_means: Dict[str, Sequence[float]], names: Sequence[str], lb: Sequence[float], grid: Iterable[Dict[str, float]],
                  checks: Optional[Dict[str, Tuple[str, str]]] = None, min_gap: float = 0.008) -> List[dict]:
    """type_means[type][i] = mean reciprocal rank of config names[i] on that type. For every weight vector: the mixture
    value of each config, Spearman and pair agreement with the LB, and for every check (a, b) the difference a - b
    (the caller decides which sign it wants). A type that is missing from type_means contributes zeros; a config whose
    LB value is NaN is left out of the agreement and can still be used in a check."""
    pos = {n: i for i, n in enumerate(names)}
    lb = np.asarray(lb, float); fin = np.isfinite(lb)            # configs without an LB score (NaN) only take part in the checks
    out = []
    for w in grid:
        wn = _norm(w)
        loc = np.zeros(len(names))
        for t, wt in wn.items():
            if t in type_means:
                loc += wt * np.asarray(type_means[t], float)
        row = dict(weights=dict(w), local=loc.tolist(), **lb_agreement(lb[fin], loc[fin], min_gap))
        for cname, (x, y) in (checks or {}).items():
            row[cname] = float(loc[pos[x]] - loc[pos[y]])
        out.append(row)
    return out
