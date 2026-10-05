"""Selection of the PubChem candidates that enter the v1 engine: f.z plus a structure prior (PC_SEL_LAM; 0 = plain f.z, today).

The join hands the best N PubChem-only candidates by FPNet f.z to the engine, and the engine's placement of an injected
truth hardly depends on its f.z rank. So the selection step decides: which N of the first PC_SEL_POOL (1,000) by f.z
enter. The hidden test is natural-product chemistry and the PubChem window is mostly synthetic compounds, so the
selection score is z(f.z) + PC_SEL_LAM * z(prior) with prior = RDKit's natural-product-likeness (Contrib/NP_Score,
Ertl et al. 2008; BSD, model file shipped with RDKit). The prior acts in the selection only: the selected candidates are
handed over in f.z order and the ranker never sees the prior. It is not the popularity prior (documentation counts)
that lost on the leaderboard. Measured in reviews/next-levers-2026-10-03.md section 6.

    prior = np_likeness(smiles_list)                         # NaN where RDKit cannot parse
    idx = select(pc_fz, prior, n=100, lam=0.5)               # indices into the deep list, ascending (= f.z order)
    idx = select(pc_fz, prior, n=100, reserve=0.25)          # alternative: a quarter of the N for the highest prior

Flags: PC_SEL_LAM=0.0, PC_SEL_POOL=1000 (list depth requested from the channel when PC_SEL_LAM > 0), PC_SEL_RESERVE=0.0.
Pure numpy except np_likeness (RDKit, lazy). Tests: test_pc_select.py.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

DEFAULTS = dict(PC_SEL_LAM=0.0, PC_SEL_POOL=1000, PC_SEL_RESERVE=0.0)
_NP = {}


def np_likeness(smiles: Sequence[str]) -> np.ndarray:
    """RDKit Contrib NP_Score per SMILES (about -5 synthetic .. +5 natural product); NaN when unparsable."""
    if 'fs' not in _NP:
        import os
        import sys
        from rdkit import RDConfig
        sys.path.insert(0, os.path.join(RDConfig.RDContribDir, 'NP_Score'))
        import npscorer
        _NP['mod'] = npscorer; _NP['fs'] = npscorer.readNPModel()
    from rdkit import Chem
    out = np.full(len(smiles), np.nan)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            try:
                out[i] = float(_NP['mod'].scoreMol(m, _NP['fs']))
            except Exception:
                pass
    return out


def _z(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, np.float64)
    if not len(v) or np.all(np.isnan(v)):
        return np.zeros(len(v))
    s = np.nanstd(v)
    if not np.isfinite(s) or s < 1e-9:
        return np.zeros(len(v))
    return np.nan_to_num((v - np.nanmean(v)) / s, nan=0.0)


def select(fz: Sequence[float], prior: Optional[Sequence[float]], n: int, lam: float = 0.0, reserve: float = 0.0, pool: int = 1000) -> List[int]:
    """Indices (ascending, i.e. in f.z order) of the n candidates chosen among the first `pool` of a list sorted by f.z.
    lam: weight of z(prior) next to z(f.z); a candidate without a prior counts as average.
    reserve: share of the n taken by the highest prior among the candidates the first rule left out.
    lam = 0 and reserve = 0, or prior None: the first n (today)."""
    m = min(len(fz), int(pool)); n = max(0, min(int(n), m))
    if n == 0:
        return []
    if prior is None or (lam == 0 and reserve == 0):
        return list(range(n))
    p = np.asarray(prior, np.float64)[:m]
    score = _z(np.asarray(fz, np.float64)[:m]) + lam * _z(p)
    k = n - int(round(n * reserve))
    first = np.argsort(-score, kind='stable')[:k]
    chosen = set(int(i) for i in first)
    if k < n:
        for i in np.argsort(-np.nan_to_num(p, nan=-np.inf), kind='stable'):
            if int(i) not in chosen:
                chosen.add(int(i))
                if len(chosen) >= n:
                    break
    return sorted(chosen)
