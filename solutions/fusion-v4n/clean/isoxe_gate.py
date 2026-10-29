"""List-aware gate for the isoxe v2 same-formula re-order (isoxe3): per formula group, a small logistic model decides whether to move.

Team code (Huseyin Emre Aksoy, Apache-2.0). Pure Python + numpy so a notebook cell can exec it. The re-order itself is fz_core's rule
(scratch/land1027/fz_core.py; library skip + training-structure freeze, idea credit heliosli): candidates that are not frozen and have
a v2 score are grouped by molecular formula; a group's own slots are re-assigned by 1/(3+b) + a/(3+x) (b = list index, x = v2 rank in
the group). Here each group is re-ordered only if `gate(features)` says yes. `Gate()` without coefficients never moves (= switch off);
`gated_fz_list(..., gate=None)` moves every group and equals fz_core.fz_list.

    feats = group_features(smis, idx, xe, ev, lib_max)       # 14 floats, FEATURES order
    new, info = gated_fz_list(smis, xe, frozen, formula, ev, lib_max, a=1.0, lock1=True, gate=Gate(**params))
`ev`: {smiles: {'main_rank', 'engine2_rank', 'pubchem_rank', 'gl', 'ice'}} (the pipeline's frozen_candidate_evidence.parquet columns);
missing entries count as rank 0 / no score.
"""
import math

import numpy as np

FEATURES = ('top1', 'lpos_top', 'lsize', 'zmargin', 'fav_hold', 'rho', 'lib_max', 'hold_main', 'fav_main', 'fav_pconly',
            'gl_sign', 'ice_sign', 'eng2', 'lpos_fav')


def groups_of(smis, xe, frozen, formula, lock1=False):
    """{formula: [list indices]} exactly as fz_core.fz_list groups them (size >= 2 only)."""
    g = {}
    for i, s in enumerate(smis):
        if frozen[i] or formula[i] is None or (lock1 and i == 0):
            continue
        v = xe.get(s)
        if v is None or v != v:
            continue
        g.setdefault(formula[i], []).append(i)
    return {f: idx for f, idx in g.items() if len(idx) >= 2}


def reorder(smis, idx, xe, a):
    """New occupant of each of the group's slots (fz_core's RRF; stable ties)."""
    order = sorted(idx, key=lambda i: -xe[smis[i]])
    xr = {i: r + 1 for r, i in enumerate(order)}
    new = sorted(idx, key=lambda i: (-(1.0 / (3 + i + 1) + a / (3 + xr[i])), i))
    return list(zip(sorted(idx), new))


def _num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _rank(e, k):
    v = _num((e or {}).get(k))
    return int(v) if v is not None else 0


def _sign(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.sign(a - b))


def group_features(smis, idx, xe, ev, lib_max):
    """The 14 FEATURES of one group (idx in list order)."""
    idx = sorted(idx)
    v = np.array([xe[smis[i]] for i in idx], np.float64)
    h = idx[0]
    f = idx[int(np.argmax(v))]                      # first maximum = list order on ties
    sd = float(v.std())
    zm = 0.0 if f == h else float((xe[smis[f]] - xe[smis[h]]) / (sd + 1e-6))
    n = len(idx)
    # Spearman between list position and v2 order (1 = v2 agrees with the list)
    xr = np.empty(n); xr[np.argsort(-v, kind='stable')] = np.arange(n)
    pr = np.arange(n, dtype=np.float64)
    rho = float(np.corrcoef(pr, xr)[0, 1]) if n > 1 and xr.std() > 0 else 1.0
    eh, ef = ev.get(smis[h]) or {}, ev.get(smis[f]) or {}
    same = f == h
    return [float(h == 0), math.log1p(h), math.log(n), zm, float(same), rho, float(lib_max or 0.0),
            float(_rank(eh, 'main_rank') > 0), float(_rank(ef, 'main_rank') > 0),
            float(_rank(ef, 'main_rank') == 0 and _rank(ef, 'pubchem_rank') > 0),
            0.0 if same else _sign(_num(ef.get('gl')), _num(eh.get('gl'))),
            0.0 if same else _sign(_num(ef.get('ice')), _num(eh.get('ice'))),
            float(_rank(ef, 'engine2_rank') > 0) - float(_rank(eh, 'engine2_rank') > 0), math.log1p(f)]


class Gate:
    """Logistic gate p = sigmoid(w . (x - mean) / scale + b); moves iff p >= threshold. No coefficients -> never moves."""

    def __init__(self, coef=None, intercept=0.0, mean=None, scale=None, threshold=0.5, features=FEATURES):
        assert tuple(features) == FEATURES, 'feature list changed'
        self.coef = None if coef is None else np.asarray(coef, np.float64)
        self.intercept = float(intercept)
        self.mean = np.zeros(len(FEATURES)) if mean is None else np.asarray(mean, np.float64)
        self.scale = np.ones(len(FEATURES)) if scale is None else np.asarray(scale, np.float64)
        self.threshold = float(threshold)

    def proba(self, x):
        if self.coef is None:
            return 0.0
        z = float(np.dot(self.coef, (np.asarray(x, np.float64) - self.mean) / self.scale) + self.intercept)
        return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))

    def __call__(self, x):
        return self.coef is not None and self.proba(x) >= self.threshold

    def params(self):
        return dict(coef=None if self.coef is None else [float(c) for c in self.coef], intercept=self.intercept,
                    mean=[float(m) for m in self.mean], scale=[float(s) for s in self.scale], threshold=self.threshold,
                    features=list(FEATURES))


def gated_fz_list(smis, xe, frozen, formula, ev, lib_max, a=1.0, lock1=False, gate=None):
    """Re-order the groups the gate accepts (gate None = all, = fz_core.fz_list). Returns (new list, info)."""
    smis = list(smis)
    out = list(smis)
    info = dict(groups=0, moved_groups=0)
    for fm, idx in groups_of(smis, xe, frozen, formula, lock1).items():
        info['groups'] += 1
        if gate is not None and not gate(group_features(smis, idx, xe, ev, lib_max)):
            continue
        pairs = reorder(smis, idx, xe, a)
        if any(s != i for s, i in pairs):
            info['moved_groups'] += 1
        for slot, i in pairs:
            out[slot] = smis[i]
    return out, info
