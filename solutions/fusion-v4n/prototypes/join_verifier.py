"""Joined-row verifier (prototype, NOT wired into the notebook): P("this joined PubChem row is the answer") from quantities nb/06 + nb/07 already hold, as a 56-feature logistic model with fixed coefficients (numpy only), and its two exposure-aware uses: promotion of one high-scored row and demotion of low-scored rows.

Why and what it measured: reviews/verifier-2026-10-05.md. Short version: demotion lowers the exposure and loses the
answer-removed gain at every setting (off by default here). Promotion of the best joined row at a high threshold, with
fusion credit for that row only, left the exposure where v13a has it and gained a little on answer-removed lists
(regime dB +0.009 +- 0.004 at p >= 0.7; Enveda panel, stages A + B, +0.018 to +0.026). The review does not ask for it
to be wired in before the open points of its section 7 are settled.

    f = row_features(smis, keys, scs, forms, pids, lib_max, pc_keys, pc_fz, ice, gl, n_spec, n_adduct, has_mh, n_neg)
    p = p_answer(f)                       # one value per joined row, in list order; f['li'] = their positions
    entry = apply([smis, keys, lib_max, scs, forms, pids], f)   # BASE[mid] for fusion_core, pool id -2 only on a promoted row

Inputs are one BASE[mid] entry of nb/06 (top TOPN after the ranker, key-deduplicated; pool id -2 = joined row, -1 =
generated), the selection model's PubChem list the join took its rows from (keys and f.z, best first, at least the first
100), the ICEBERG / GLACIER dicts of nb/07 ({smiles: score | None}), and four numbers about the query spectra.

Where it would hook in: after nb/07 has the forward scores and before `fusion_core.build_submission(BASE, ...)`, for every
molecule whose entry holds a pool id -2: `BASE[mid] = join_verifier.apply(BASE[mid], f)` and `FUSE_INJ_CREDIT = 1.0` in the
fusion config (the credit then reaches only the promoted row, because apply() re-tags every other joined row as -1; both
are < 0, which is all popularity_reorder looks at). nb/06 would have to keep per molecule: the number of spectra, of
distinct adducts, whether an [M+H]+ spectrum exists, the number of negative-mode spectra (all in `sub`), and the keys and
f.z of the PubChem list (PC_DEEP[mid]). No feature uses the label, the list type, popularity or the molecule's identity.

Model: fitted by scratch/verifier/export_model.py on 21,925 joined rows (154 answers) of the 380 mixed-regime molecules,
v13a selection (fold-safe FPNet), lists c1 + c2 + answer-removed; x -> fill non-finite -> clip -> standardise -> dot.
The constants of the fit on the fine-tuned selection are in D:/casmi-data/verifier/model_v22a_noconf.json (same keys;
pass it as M). Covered by test_join_verifier.py.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

DEMOTE_THR, DEMOTE_K = None, 10       # demotion is off by default (it loses the answer-removed gain); the study used 0.03 / 10
PROMOTE_THR = 0.70                    # regime CV: 0 of 280 pool-answer lists at or above it (1 list in 1 of 5 repeats); Enveda panel: 0 of 159


def _fwd(sc: np.ndarray, forms: np.ndarray, pool: np.ndarray, joined: np.ndarray, rows: np.ndarray) -> Dict[str, np.ndarray]:
    """Forward-model features of `rows` inside same-formula groups of the list; sc = score or nan per list row."""
    out = {k: np.full(len(rows), np.nan) for k in ('s', 'rank', 'rankf', 'margin', 'z', 'gap_pool', 'gap_join', 'top')}
    out['has'] = np.zeros(len(rows)); ok = np.isfinite(sc)
    for t, i in enumerate(rows):
        if not ok[i]:
            continue
        g = np.flatnonzero((forms == forms[i]) & ok); v = sc[g]; s = sc[i]; oth = g[g != i]
        out['has'][t] = 1.0; out['s'][t] = s
        out['rank'][t] = 1 + (v > s).sum(); out['rankf'][t] = (v > s).sum() / max(1, len(g) - 1)
        if len(oth):
            sd = v.std()
            out['margin'][t] = s - sc[oth].max(); out['z'][t] = (s - v.mean()) / sd if sd > 1e-9 else 0.0
            out['top'][t] = float(s >= sc[oth].max())
        gp = oth[pool[oth]]; gj = oth[joined[oth]]
        if len(gp):
            out['gap_pool'][t] = s - sc[gp].max()
        if len(gj):
            out['gap_join'][t] = s - sc[gj].max()
    return out


def row_features(smis: Sequence[str], keys: Sequence[str], scs: Sequence[float], forms: Sequence[str], pids: Sequence[int], lib_max: float,
                 pc_keys: Sequence[str], pc_fz: Sequence[float], ice: Optional[dict], gl: Optional[dict],
                 n_spec: float = 1.0, n_adduct: float = 1.0, has_mh: float = 1.0, n_neg: float = 0.0) -> Dict[str, np.ndarray]:
    """Features of the joined rows (pool id -2) of one base entry; {} when there is none. 'li' = their list positions (0-based)."""
    s = np.asarray(scs, np.float64); pid = np.asarray(pids); forms = np.asarray(list(forms), object); n = len(s)
    J = pid == -2; P = pid >= 0; rows = np.flatnonzero(J)
    if not len(rows) or n < 2:
        return {}
    one = np.ones(len(rows)); sd = s.std() + 1e-9; z = (s - s.mean()) / sd
    pool_s = s[P]; join_s = s[J]; best_pool = pool_s.max() if len(pool_s) else np.nan; ps = np.sort(pool_s)
    f = dict(pos=rows + 1.0, score=s[rows], z=z[rows], gap_top=(s[rows] - s[0]) / sd, gap_pool=s[rows] - best_pool, gap_pool_sd=(s[rows] - best_pool) / sd,
             gap_join=s[rows] - join_s.max(), rank_join=np.arange(1, len(rows) + 1, dtype=float), gap_next_join=np.r_[join_s[:-1] - join_s[1:], 0.0],
             pool_above=np.array([P[:i].sum() for i in rows], float), n_join60=one * len(rows), n_join25=one * J[:25].sum(), n_pool60=one * P.sum(),
             list_sd=one * sd, pool_top=one * best_pool, top_is_join=one * float(J[0]),
             pool_m12=one * ((ps[-1] - ps[-2]) / sd if len(ps) > 1 else np.nan), join_m12=one * ((join_s[0] - join_s[1]) / sd if len(join_s) > 1 else np.nan),
             f_top=(forms[rows] == forms[0]).astype(float),
             f_pool_top=(forms[rows] == forms[np.flatnonzero(P)[0]]).astype(float) if P.any() else one * np.nan,
             grp_n=np.array([(forms == forms[i]).sum() for i in rows], float), grp_pool=np.array([((forms == forms[i]) & P).sum() for i in rows], float),
             grp_pool_above=np.array([((forms[:i] == forms[i]) & P[:i]).sum() for i in rows], float))
    f['grp_has_pool'] = (f['grp_pool'] > 0).astype(float)
    # selection model: rank and f.z margins in units of the spread of the first 100 f.z (the f.z scale differs between models)
    fz = np.asarray(pc_fz, np.float64); pk = {k: j for j, k in enumerate(pc_keys)}
    r = np.array([pk.get(keys[i], -1) for i in rows]); has = r >= 0; rr = np.maximum(r, 0)
    if len(fz):
        f100 = fz[:100]; fsd = f100.std() + 1e-9; nxt = np.r_[fz[1:], fz[-1:]]
        f['sel_rank'] = np.where(has, rr + 1.0, np.nan); f['sel_lnrank'] = np.log(f['sel_rank'])
        f['sel_fz_z'] = np.where(has, (fz[rr] - f100.mean()) / fsd, np.nan)
        f['sel_gap_next'] = np.where(has, (fz[rr] - nxt[rr]) / fsd, np.nan); f['sel_gap_top'] = np.where(has, (fz[rr] - fz[0]) / fsd, np.nan)
        f['sel_m12'] = one * ((fz[0] - fz[1]) / fsd if len(fz) > 1 else np.nan)
    else:
        for k in ('sel_rank', 'sel_lnrank', 'sel_fz_z', 'sel_gap_next', 'sel_gap_top', 'sel_m12'):
            f[k] = one * np.nan
    for name, d in (('ice_', ice), ('gl_', gl)):
        v = np.array([(d.get(x) if d and d.get(x) is not None else np.nan) for x in smis], np.float64)
        for k, a in _fwd(v, forms, P, J, rows).items():
            f[name + k] = a
    zz = np.stack([f['ice_z'], f['gl_z']]); anyz = np.isfinite(zz).any(0)
    f['fwd_z'] = np.where(anyz, np.where(np.isfinite(zz), zz, 0.0).sum(0), np.nan)
    f['fwd_gap_pool_min'] = np.fmin(f['ice_gap_pool'], f['gl_gap_pool'])
    f['fwd_top_both'] = ((f['ice_top'] == 1) & (f['gl_top'] == 1)).astype(float)
    f.update(lib_max=one * float(lib_max), n_spec=one * float(n_spec), n_adduct=one * float(n_adduct), has_mh=one * float(has_mh), n_neg=one * float(n_neg))
    f['li'] = rows
    return f


MODEL = {
    'features': ['pos', 'score', 'z', 'gap_top', 'gap_pool', 'gap_pool_sd', 'gap_join', 'rank_join', 'gap_next_join', 'pool_above', 'n_join60', 'n_join25', 'n_pool60', 'list_sd', 'pool_top', 'top_is_join', 'pool_m12', 'join_m12', 'f_top', 'f_pool_top', 'grp_n', 'grp_pool', 'grp_has_pool', 'grp_pool_above', 'sel_rank', 'sel_lnrank', 'sel_fz_z', 'sel_gap_next', 'sel_gap_top', 'sel_m12', 'ice_s', 'ice_rank', 'ice_rankf', 'ice_margin', 'ice_z', 'ice_gap_pool', 'ice_gap_join', 'ice_top', 'ice_has', 'gl_s', 'gl_rank', 'gl_rankf', 'gl_margin', 'gl_z', 'gl_gap_pool', 'gl_gap_join', 'gl_top', 'gl_has', 'fwd_z', 'fwd_gap_pool_min', 'fwd_top_both', 'lib_max', 'n_spec', 'n_adduct', 'has_mh', 'n_neg'],
    'fill': [32.0, -2.27097, -0.33191, -3.44833, -3.60033, -2.73559, -3.62188, 23.0, 0.0461796, 3.0, 51.0, 20.0, 7.0, 1.44511, 1.9073, 0.0, 0.575498, 0.352332, 1.0, 1.0, 50.0, 2.0, 1.0, 1.0, 36.0, 3.58352, 0.1175, 0.0232139, -3.32458, 0.536063, 0.163029, 15.0, 0.448628, -0.141744, -0.0220378, -0.102071, -0.116125, 0.0, 1.0, 0.148704, 15.0, 0.45614, -0.165776, -0.106465, -0.12437, -0.1342, 0.0, 1.0, -0.0912178, -0.147627, 0.0, 0.300493, 3.0, 1.0, 1.0, 0.0],
    'clip_lo': [2.0, -4.92723, -1.34998, -6.14169, -9.32488, -5.98382, -8.42823, 1.0, 0.0, 0.0, 11.0, 2.0, 0.0, 0.714225, -4.12043, 0.0, 0.00367216, 0.00371276, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.27659, 0.000316544, -6.09841, 0.00304924, 0.0, 1.0, 0.0, -0.578309, -1.89362, -0.544347, -0.571494, 0.0, 0.0, 0.0, 1.0, 0.0, -0.645711, -1.87098, -0.627188, -0.6414, 0.0, 0.0, -3.48569, -0.627574, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    'clip_hi': [60.0, 2.94235, 2.70412, -0.0432213, 3.68539, 2.44455, 0.0, 56.0, 1.1471, 38.0, 60.0, 25.0, 48.0, 2.40816, 5.64363, 1.0, 5.32157, 3.28031, 1.0, 1.0, 60.0, 46.0, 1.0, 36.0, 99.0, 4.59512, 3.59445, 0.810307, 0.0, 3.01743, 0.590766, 58.0, 1.0, 0.0527939, 2.49772, 0.323756, 0.0656974, 1.0, 1.0, 0.637027, 58.0, 1.0, 0.0588619, 2.56031, 0.353554, 0.0698251, 1.0, 1.0, 4.67418, 0.290324, 1.0, 0.696171, 6.0, 4.0, 1.0, 6.0],
    'mean': [31.8535, -2.03576, -0.100105, -3.35665, -3.56117, -2.4679, -3.73372, 24.0752, 0.106111, 6.35129, 47.2229, 18.3935, 10.4733, 1.44849, 1.52069, 0.450308, 1.16649, 0.591432, 0.630285, 0.640502, 38.1686, 6.91895, 0.669145, 4.3751, 40.7381, 3.31691, 0.332964, 0.0622496, -3.15791, 0.700636, 0.184523, 18.4598, 0.459274, -0.167598, 0.0270575, -0.114755, -0.145667, 0.075439, 0.786773, 0.174679, 18.4847, 0.46297, -0.199116, -0.00179117, -0.142506, -0.171622, 0.0697834, 0.786864, 0.0320817, -0.164835, 0.04187, 0.297659, 3.60055, 1.45227, 0.799909, 0.936967],
    'scale': [16.9022, 1.60583, 0.887291, 1.26279, 2.78839, 1.79001, 2.02295, 15.2077, 0.176959, 7.70011, 11.7654, 6.23971, 10.5629, 0.373516, 2.31592, 0.497525, 1.28572, 0.68953, 0.482727, 0.479853, 23.3221, 10.1233, 0.470521, 7.07564, 28.3974, 1.06294, 1.09579, 0.120209, 1.39558, 0.619127, 0.123302, 14.9334, 0.281544, 0.137165, 0.820878, 0.126817, 0.130141, 0.264098, 0.409587, 0.132434, 14.8842, 0.280447, 0.163695, 0.826182, 0.150456, 0.154972, 0.254782, 0.409523, 1.49577, 0.144536, 0.200292, 0.220941, 2.07435, 0.693718, 0.400068, 1.54165],
    'coef': [-0.15619, 0.18316, 0.598945, 0.690155, -0.0390192, -0.238537, 0.958078, -0.145253, 0.088567, 0.0527183, -0.153128, -0.276498, -0.25039, 0.146818, -0.604494, -0.141712, -0.443914, 0.366385, 0.623399, 0.163378, 0.170798, 0.265909, -0.166931, 0.292804, 0.19787, -0.107885, 0.0586607, -0.127533, 0.503054, 0.367286, 0.138415, 0.362889, 0.0328716, 0.535521, 0.281211, -0.0135598, 0.218654, -0.109395, -0.0383124, -0.0810136, -0.0644923, -0.123565, 0.588936, -0.0491437, 0.19528, 0.404288, 0.13655, -0.0381549, -0.00684166, -0.00164431, 0.0640468, -0.00999424, 0.0538841, -0.0525414, -0.416945, 0.0535608],
    'intercept': -10.369,
}


def p_answer(feats: Dict[str, np.ndarray], M: Optional[dict] = None) -> np.ndarray:
    """One probability-like score per joined row (list order). {} -> empty array."""
    if not feats:
        return np.zeros(0)
    M = MODEL if M is None else M
    X = np.stack([np.asarray(feats[c], np.float64) for c in M['features']], 1)
    X = np.clip(np.where(np.isfinite(X), X, np.asarray(M['fill'])), np.asarray(M['clip_lo']), np.asarray(M['clip_hi']))
    z = ((X - np.asarray(M['mean'])) / np.asarray(M['scale'])) @ np.asarray(M['coef']) + M['intercept']
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _take(e: list, order: Sequence[int], scs=None, pids=None) -> list:
    return [[e[0][i] for i in order], [e[1][i] for i in order], e[2], list(scs) if scs is not None else [e[3][i] for i in order],
            [e[4][i] for i in order], list(pids) if pids is not None else [e[5][i] for i in order]]


def _monotone(scs: Sequence[float]) -> list:
    s = np.array(scs, float)
    for i in range(1, len(s)):
        if s[i] > s[i - 1] - 1e-6:
            s[i] = s[i - 1] - 1e-6
    return s.tolist()


def demote(entry: list, p_of: Dict[str, float], thr: float = 0.03, k: int = DEMOTE_K) -> list:
    """Joined rows (pool id -2) with p < thr that stand above the k-th non-joined row are moved to just behind it, in
    their order; their scores are lowered to keep the list sorted. No row moves up past a row that was above it, except
    non-joined rows and kept joined rows passing the moved ones: a joined row can only go down."""
    pid = np.array(entry[5]); keys = entry[1]; n = len(pid)
    low = np.array([pid[i] == -2 and p_of.get(keys[i], 0.0) < thr for i in range(n)], bool)
    if n == 0 or not low.any():
        return entry
    nj = np.cumsum(pid != -2); cut = int(np.searchsorted(nj, k)) if nj[-1] >= k else n - 1
    moved = [i for i in range(cut + 1) if low[i]]
    if not moved:
        return entry
    order = [i for i in range(cut + 1) if not low[i]] + moved + list(range(cut + 1, n))
    return _take(entry, order, _monotone([entry[3][i] for i in order]))


def promote(entry: list, p_of: Dict[str, float], thr: float = PROMOTE_THR) -> list:
    """The best-scored joined row goes to the first place when its p >= thr and keeps pool id -2; every other joined row
    is re-tagged -1, so that FUSE_INJ_CREDIT of fusion_core reaches the promoted row only."""
    pid = np.array(entry[5]); keys = entry[1]; j = [i for i in range(len(pid)) if pid[i] == -2]
    if not j:
        return entry
    pid = np.where(pid == -2, -1, pid)
    b = max(j, key=lambda i: p_of.get(keys[i], 0.0))
    if p_of.get(keys[b], 0.0) < thr:
        return _take(entry, list(range(len(pid))), pids=pid.tolist())
    pid[b] = -2; order = [b] + [i for i in range(len(pid)) if i != b]
    scs = [entry[3][0] + 1e-3] + [entry[3][i] for i in order[1:]] if b != 0 else list(entry[3])
    return _take(entry, order, scs, pid[order].tolist())


def apply(entry: list, feats_or_p, demote_thr: Optional[float] = DEMOTE_THR, demote_k: int = DEMOTE_K,
          promote_thr: Optional[float] = PROMOTE_THR, M: Optional[dict] = None) -> list:
    """demote() then promote() on one BASE entry [smis, keys, lib_max, scores, formulas, pool ids]. `feats_or_p`: the dict
    of row_features() or {key: p}. None switches a step off. The input is not mutated."""
    if isinstance(feats_or_p, dict) and 'li' in feats_or_p:
        p = p_answer(feats_or_p, M); p_of = {entry[1][int(i)]: float(v) for i, v in zip(feats_or_p['li'], p)}
    else:
        p_of = dict(feats_or_p or {})
    e = [list(entry[0]), list(entry[1]), entry[2], list(entry[3]), list(entry[4]), list(entry[5])]
    if demote_thr is not None:
        e = demote(e, p_of, demote_thr, demote_k)
    if promote_thr is not None:
        e = promote(e, p_of, promote_thr)
    return e
