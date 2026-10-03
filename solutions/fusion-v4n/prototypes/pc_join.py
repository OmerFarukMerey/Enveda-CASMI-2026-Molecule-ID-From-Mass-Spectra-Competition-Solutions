"""PubChem join: hand the best N PubChem-only candidates to the v1 engine as extra candidates (flag PC_JOIN_N; off = 0).

Today the PubChem channel (nb/04) lists 25 structures outside the main pool by FPNet f.z and stage B of fusion_core
puts them into fixed slots (4, 8, 12, ...). Here the first N of that list enter the v1 engine's candidate set, get every
feature a candidate gets (analog propagation, f.z views, fragmentation, subformula, the fe_v4 families) and are placed
by the ranker. Nothing inside the public `casmi` package is edited: `install` wraps two methods of the engine object.

How a candidate enters. `Engine.candidates(target, drop_pid, generated)` appends a list of dicts (smiles, key, formula,
mass, n_heavy, fp, frags, sim, steps, tan) behind the pool rows; that is how the derivative generator's products enter
(pool id -1). An injected PubChem candidate is one more such dict (`candidate_record`), with two differences:
  * it is NOT flagged as generated: is_gen, gen_parent_sim, gen_steps, gen_parent_tan are 0, n_gen and gen_rank_fz count
    real generator products only, and the families see is_gen = 0 (derivation: the dp_* "pool candidate" columns). It
    has no library spectrum (train_sid -1: lib_* = 0), no pool id (per-structure records of frag2 / fragnet /
    analog_struct are computed from the SMILES, as for generated candidates);
  * order = 'first' (default): the generator treats the injected keys as pool keys, so a derivative of an analog that
    equals a PubChem candidate is logged as a derivable pool row (dp_hit and the other dp_* features) instead of
    becoming a generated twin. order = 'after': the generator runs as today and an injected candidate whose key was
    generated is dropped (the generated copy stays).
Injected rows sit last in the candidate order, so under the public ranker's stable-tie ranks (reviews/ranker-2026-10-02.md
section 2) they carry the worst order code; RANKER 'public_free' / 'blend' remove that.

    install(E, V)                                       # once, after v1engine.V1FE(E, ...)
    if library_max(E, spectra, target) < MAX_LIB: ...   # optional gate: no injection for molecules with library evidence
    recs = build_records(smiles[:N], keys[:N], chem, fragmod, E.pool.bits)
    set_injection(E, recs)                              # before V.run(spectra, target); set_injection(E, []) = today
    C, X, F, names, info = V.run(spectra, target)
    inj = injected_mask(E, C)                           # which rows of C came from PubChem
    score = choose_score(CFG['RANKER'], OUR_RK, C.formula, X, F, FEATURES, names, lib_max, rank_score, CFG['RANKER_LIB_GATE'], inj)

Ranking. `choose_score` is ranker_infer.choose_score with one change: where that function recomputes the seven rank
features order-free (modes 'public_free', 'ours', 'blend'), ties are averaged inside a tier and pool / generated rows come
before injected rows of the same value (`order_free_join`). Without it an injected decoy tops the list of a quarter of
the molecules whose answer is a pool row (reviews/pcjoin-2026-10-02.md section 3); with it the public ranker's accidental
"pool rows first" order code is kept for the join and dropped inside the pool. With no injected row it equals
ranker_infer.choose_score exactly.

Pure helpers (no engine, covered by test_pc_join.py): select_injection (N and the gates), split_for_slots (the PubChem
dict that is left for the slot merge), inject_last, rank_two_tier, order_free_join, choose_score.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

DEFAULTS = dict(PC_JOIN_N=0,            # how many PubChem candidates enter the v1 engine; 0 = off (today's notebook)
                PC_JOIN_ORDER='first',  # 'first' | 'after' (see the module docstring)
                PC_JOIN_MAX_LIB=None,   # inject only when the molecule's best library similarity (library_max, the engine's
                                        #   lib_max) is below this; 0.7 keeps class-1-like lists as today; None = no gate
                PC_JOIN_MIN_REL=None,   # inject only candidates with f.z - best pool f.z above this (None = no gate)
                PC_JOIN_SLOTS='all')    # slot merge afterwards: 'all' = the first 25 of the list as today, fusion_core
                                        #   unchanged (proposals the ranker placed in the top 25 are skipped by merge()),
                                        #   'rest' = only proposals that were not injected, 'none'; the three differ by
                                        #   less than 0.006 MRR on every molecule type (review section 3)


# ------------------------------------------------------------------------------------------------ pure helpers
def select_injection(pc: Optional[dict], n: int, lib_max: Optional[float] = None, max_lib: Optional[float] = None,
                     min_rel: Optional[float] = None, absent: Optional[bool] = None) -> List[int]:
    """Indices into pc['pc'] to inject: the first n of the f.z order, gated.
    max_lib: nothing when lib_max >= max_lib. min_rel: only candidates with pc_fz - best_pool_fz > min_rel (when the pool
    window is empty every candidate passes). absent: an external "truth probably absent" flag; False injects nothing."""
    if not pc or not pc.get('pc') or n <= 0:
        return []
    if max_lib is not None and lib_max is not None and lib_max >= max_lib:
        return []
    if absent is False:
        return []
    idx = list(range(min(n, len(pc['pc']))))
    if min_rel is not None:
        bp = pc.get('best_pool_fz')
        if bp is not None and np.isfinite(bp):
            idx = [i for i in idx if pc['pc_fz'][i] - bp > min_rel]
    return idx


def split_for_slots(pc: Optional[dict], injected: Sequence[int], mode: str = 'all', k: int = 25) -> Optional[dict]:
    """The PubChem dict stage B of fusion_core should see after an injection. 'all': unchanged (first k). 'rest': the
    injected entries are removed and the next ones move up. 'none': None. The gate quantity keeps the ORIGINAL top
    f.z (`fz_top`), so the early-slot decision does not move."""
    if not pc or not pc.get('pc') or mode == 'none':
        return None
    drop = set(injected) if mode == 'rest' else set()
    keep = [i for i in range(len(pc['pc'])) if i not in drop][:k]
    if not keep:
        return None
    out = {kk: v for kk, v in pc.items() if kk not in ('pc', 'pc_fz', 'pc_keys', 'pc_form')}
    out['pc'] = [pc['pc'][i] for i in keep]; out['pc_fz'] = [pc['pc_fz'][i] for i in keep]; out['pc_keys'] = [pc['pc_keys'][i] for i in keep]
    if out.get('fz_top') is None:
        out['fz_top'] = pc['pc_fz'][0]
    return out


def inject_last(generated: Optional[list], records: Sequence[dict], pool_keys: set) -> List[dict]:
    """Generated products + the injected records whose key is neither a pool key of the window, nor generated, nor repeated."""
    g = list(generated or [])
    seen = set(pool_keys) | {x['key'] for x in g}
    for r in records:
        if r['key'] in seen:
            continue
        seen.add(r['key']); g.append(r)
    return g


# ------------------------------------------------------------------------------------------------ ranking
# rank feature -> (source feature, sign so that larger = better, absolute value first); == ranker_infer.RANK_SOURCES
RANK_SOURCES = {'lib_max_rank': ('lib_max', 1, False), 'ap_rank': ('ap', 1, False), 'fz_rank': ('fz', 1, False), 'el_rank': ('el_sum', -1, False),
                'fr_rank': ('fr_int', 1, False), 'sf_rank': ('sf_int', 1, False), 'mass_ppm_rank': ('mass_ppm', -1, True)}


def rank_two_tier(x: np.ndarray, tier: np.ndarray) -> np.ndarray:
    """0 = best (largest x) .. 1 = worst. Ties share the average rank inside a tier; among equal values tier 0 comes
    before tier 1. With a single tier this is the average-tie rank of ranker_infer._rank_avg."""
    n = len(x)
    if n == 0:
        return np.zeros(0)
    x = np.asarray(x, np.float64); tier = np.asarray(tier, np.int64)
    o = np.lexsort((tier, -x)); xs = x[o]; ts = tier[o]
    b = np.r_[True, (xs[1:] != xs[:-1]) | (ts[1:] != ts[:-1])]; bid = np.cumsum(b) - 1
    st = np.flatnonzero(b); en = np.r_[st[1:], n]
    r = np.empty(n); r[o] = ((st + en - 1) / 2.0)[bid]
    return r / max(1, n - 1)


def order_free_join(X: np.ndarray, names: Sequence[str], inj: Optional[np.ndarray] = None) -> np.ndarray:
    """Copy of one molecule's engine features with the seven rank features recomputed from their float32 sources:
    order-free among pool and generated rows, injected rows (inj True) last among ties."""
    names = list(names); X2 = np.array(X, np.float32, copy=True)
    tier = np.zeros(len(X2), np.int64) if inj is None else np.asarray(inj).astype(np.int64)
    for f, (src, sgn, ab) in RANK_SOURCES.items():
        if f in names and src in names:
            v = X2[:, names.index(src)].astype(np.float64)
            X2[:, names.index(f)] = rank_two_tier(sgn * (np.abs(v) if ab else v), tier)
    return X2


def _z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    if len(v) < 2:
        return np.zeros(len(v))
    s = v.std(ddof=1)
    return (v - v.mean()) / s if s > 1e-9 else np.zeros(len(v))


def choose_score(mode: str, ranker, formulas, X, F, names_x, names_f, lib_max: float, public_score: Callable, lib_gate: Optional[float] = None,
                 inj: Optional[np.ndarray] = None, bias: float = 0.0) -> np.ndarray:
    """Base-list score of one molecule whose candidate set may hold injected rows (drop-in for ranker_infer.choose_score).
    mode 'public': the public ranker as it is (injected rows are last in candidate order, so its stable-tie ranks already
    put them behind pool rows of the same value). 'public_free' / 'ours' / 'blend': as in ranker_infer, on the features
    of order_free_join. lib_gate: lib_max >= gate keeps the public ranker as it is. bias: injected rows lose
    bias x (standard deviation of the list's scores); 0 = off (a stronger pool prior, not needed locally)."""
    names_x = list(names_x); names_f = list(names_f); names = names_x + names_f
    if mode == 'public' or (lib_gate is not None and lib_max >= lib_gate):
        s = np.asarray(public_score(np.concatenate([X, F], 1), names), np.float64)
    else:
        Xo = order_free_join(X, names_x, inj)
        pub = lambda: np.asarray(public_score(np.concatenate([Xo, F], 1), names), np.float64)
        if mode == 'public_free':
            s = pub()
        elif mode in ('ours', 'blend'):
            import ranker_infer as RI
            E = RI.confidence_features(Xo, names_x, formulas)
            px = {n: i for i, n in enumerate(names_x)}; pf = {n: i for i, n in enumerate(names_f)}
            M = np.ascontiguousarray(np.stack([Xo[:, px[f[3:]]] if f.startswith('x__') else (E[f[3:]] if f.startswith('e__') else np.asarray(F[:, pf[f]], np.float32))
                                               for f in ranker.features], 1), dtype=np.float32)
            ours = np.mean([b.predict(M, num_threads=4) for b in ranker.boosters], axis=0)
            s = np.asarray(ours, np.float64) if mode == 'ours' else _z(ours) + _z(pub())
        else:
            raise ValueError(mode)
    if bias and inj is not None and np.any(inj) and len(s) > 1:
        s = s - bias * float(np.std(s, ddof=1)) * np.asarray(inj, np.float64)
    return s


# ------------------------------------------------------------------------------------------------ engine side
def candidate_record(smiles: str, key: Optional[str], chem, fragmod, bits: np.ndarray, rank: int = -1, fz: float = float('nan'),
                     standardize: bool = True) -> Optional[dict]:
    """Everything Engine.candidates needs for one extra structure, computed like Engine.generate does for a product:
    standardised SMILES, formula / exact mass / heavy atoms, selected fingerprint bits (float32 0/1), depth-2 fragment
    masses. `key` = the metric key the PubChem channel already computed (None: computed here, about 8 ms)."""
    std = chem.standardize_smiles(smiles) if standardize else smiles
    if std is None:
        std = smiles                                   # stays charged: keep the PubChem form
    p = chem.mol_props(std)
    if p is None:
        return None
    fp = chem.raw_fingerprint(std)
    if fp is None:
        return None
    if key is None:
        key = chem.score_key(std)
        if key is None:
            return None
    return dict(smiles=std, key=key, formula=p[0], mass=p[1], n_heavy=p[2], fp=fp[bits].astype(np.float32),
                frags=fragmod.fragments_for_smiles(std), sim=0.0, steps=0, tan=0.0, parent=-1, raw=None, combo=None,
                pc_smiles=smiles, pc_rank=int(rank), pc_fz=float(fz))


def build_records(smiles: Sequence[str], keys: Sequence[Optional[str]], chem, fragmod, bits: np.ndarray, fz: Optional[Sequence[float]] = None,
                  cache: Optional[Dict[str, dict]] = None) -> List[dict]:
    out = []
    for i, (s, k) in enumerate(zip(smiles, keys)):
        r = cache.get(s) if cache is not None else None
        if r is None:
            r = candidate_record(s, k, chem, fragmod, bits)
            if cache is not None and r is not None:
                cache[s] = r
        if r is not None:
            r = dict(r, pc_rank=i, pc_fz=float(fz[i]) if fz is not None else float('nan'))
            out.append(r)
    return out


def library_max(E, spectra: Sequence[dict], target: float, exclude=None, drop_pid: int = -1) -> float:
    """The engine's `lib_max` of a molecule (best library similarity of any candidate) without running the engine:
    the L stage of Engine.run only (library window, entropy search per query spectrum, candidates of the pool window).
    Injected rows have no library spectrum, so the value is the same with and without an injection; it lets the
    PC_JOIN_MAX_LIB gate decide before the one engine pass. Checked against info['lib_max'] in scratch/pcjoin/check_dropin.py."""
    from casmi import spectra as spmod
    cfg, L, P = E.cfg, E.L, E.pool
    cand = P.window(target, cfg.ppm_win)
    if len(cand) == 0:
        cand = P.window(target, cfg.ppm_fallback)
    if drop_pid >= 0:
        cand = cand[cand != drop_pid]
    sids = set(int(x) for x in P.train_sid[cand] if x >= 0)
    best = 0.0
    if not sids:
        return best
    for q in list(spectra)[:cfg.max_query_spectra]:
        qm, qp = L.prep(q['mz'], q['it'], cfg.search)
        if len(qm) == 0:
            continue
        lc = L.window(target, cfg.search.ppm_win, exclude, q['mode'])
        if len(lc) == 0:
            continue
        sc = spmod.search(qm, qp, lc, L.off, L.mz, L.it, cfg.search.tol, cfg.search.ppm_tol, cfg.search.floor, cfg.search.topk, cfg.search.power,
                          cfg.search.ent_weight, 0)
        for j, x in enumerate(lc):
            if sc[j] > best and int(L.sid[x]) in sids:
                best = float(sc[j])
    return best


def install(E, V=None, order: str = 'first'):
    """Wrap E.generate / E.candidates (and V.view) so that the records of `set_injection` join the candidate set.
    Call after hooks_v1.install (i.e. after V1FE was built). With no injection set the engine behaves exactly as before."""
    assert order in ('first', 'after')
    if getattr(E, '_pc_installed', False):
        E._pc_order = order
        return
    E._pc_installed = True; E._pc_order = order; E._pc_inject = []; E._pc_last = []
    gen0, cand0 = E.generate, E.candidates

    def generate(analogs, target, exclude_keys):
        if E._pc_inject and E._pc_order == 'first':
            exclude_keys = set(exclude_keys) | {r['key'] for r in E._pc_inject}
        return gen0(analogs, target, exclude_keys)

    def candidates(target, drop_pid=-1, generated=None):
        E._pc_last = []
        if not E._pc_inject:
            return cand0(target, drop_pid, generated)
        P = E.pool
        win = P.window(target, E.cfg.ppm_win)
        if len(win) == 0:
            win = P.window(target, E.cfg.ppm_fallback)
        if drop_pid >= 0:
            win = win[win != drop_pid]
        n_gen = len(generated or [])
        g = inject_last(generated, E._pc_inject, set(P.key[c] for c in win))
        C = cand0(target, drop_pid, g)
        n_inj = len(g) - n_gen
        if n_inj:
            C.is_gen[len(C.pid) - n_inj:] = 0.0          # not a generator product: gen_* stay 0, n_gen counts real products
            E._pc_last = g[n_gen:]
        return C

    E.generate = generate; E.candidates = candidates
    if V is not None:
        view0 = V.view

        def view(rec, prep, target, C, X, features_v1):
            v = view0(rec, prep, target, C, X, features_v1)
            if E._pc_last:
                v.is_gen = np.asarray(C.is_gen, np.float32).copy()      # families: injected rows are pool-like, not generated
            return v
        V.view = view


def set_injection(E, records: Optional[Sequence[dict]]):
    E._pc_inject = list(records or [])


def injected_mask(E, C) -> np.ndarray:
    """Boolean mask over the candidates of the last run: True for injected PubChem rows (they are the last rows)."""
    m = np.zeros(len(C.pid), bool)
    n = len(getattr(E, '_pc_last', []) or [])
    if n:
        m[len(m) - n:] = True
    return m


def injected_info(E) -> List[dict]:
    """The records that actually entered the last run (after de-duplication), in candidate order."""
    return list(getattr(E, '_pc_last', []) or [])
