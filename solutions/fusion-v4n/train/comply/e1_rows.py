"""Ranker rows of our engine-1 ranker (stack1 compliance step 2): stored engine runs -> the 141-feature matrix, CPU only.

Input: one chunked engine run as written by the engine-row runner (`e1_engine_rows.py`; the P4 runs of the ownfp study were written by
its predecessors scratch/ranker/run_engine.py = format 'eng', scratch/pcjoin/run_join.py = format 'join', with the same engine call):
per chunk `w<i>_c<k>.npz` (X = the v1 engine's 94 features, F = fe_v4 family features, pid, y, g [, inj, strs | gen_json]) and
`w<i>_c<k>.parquet` (one row per molecule). Per list:
  1. rows marked injected (PubChem join) are removed; a no-join run has none;
  2. the seven tie-dependent rank features of the v1 engine are recomputed with average ties from their float32 sources (the engine's
     stable ties encode pool row order: a provenance signal that does not exist at test time);
  3. the 14 `fragnet__*` features are recomputed with the given stage A / stage B (default: ours, `fit_fragnet.py`) by calling the MIT
     `fe_v4.fams.fragnet.compute` on a view built from the stored candidates (pool rows: pool SMILES and formula; generated rows: the
     stored strings) and the molecule's spectra (the set's queries table);
  4. the matrix holds the 94 engine features + the family features of `MANIFEST.json` without the 19 `derivation__*` = 141 columns.

Parity mode (`parity`): steps 1 to 3 with the SHIPPED stage A / B must reproduce the stored fragnet columns of a no-join run.
usage:
  python e1_rows.py build  --rows DIR --queries Q.parquet --out OUT.npz [--stage_a A.npz --stage_b B.txt] [--mids FILE]
  python e1_rows.py parity --rows DIR --queries Q.parquet --n 60
Defaults: --code D:/casmi-data/ranker_study/ds/v4b/code (MIT fe_v4 + casmi), --pool_meta D:/casmi-data/directions/run/work/pool_meta.parquet.
"""
from __future__ import annotations

import argparse, glob, json, os, sys, time
from types import SimpleNamespace

import numpy as np
import pandas as pd

CODE = 'D:/casmi-data/ranker_study/ds/v4b/code'
POOL_META = 'D:/casmi-data/directions/run/work/pool_meta.parquet'
OURS_A = 'C:/Users/hemre/casmi-work/comply/step0/stageA_repro.npz'
OURS_B = 'C:/Users/hemre/casmi-work/comply/step0/stageB_repro.txt'

# rank feature -> (source feature, sign so that larger = better, absolute value first); == prototypes/tie_order.RANK_SOURCES
RANK_SOURCES = {'lib_max_rank': ('lib_max', 1, False), 'ap_rank': ('ap', 1, False), 'fz_rank': ('fz', 1, False), 'el_rank': ('el_sum', -1, False),
                'fr_rank': ('fr_int', 1, False), 'sf_rank': ('sf_int', 1, False), 'mass_ppm_rank': ('mass_ppm', -1, True)}


def rank_norm_average(x):
    """0 = best (largest) .. 1 = worst; ties share the average rank."""
    x = np.asarray(x, np.float64); n = len(x)
    if n == 0:
        return np.zeros(0)
    o = np.argsort(-x, kind='mergesort'); xs = x[o]
    b = np.r_[True, xs[1:] != xs[:-1]]; bid = np.cumsum(b) - 1
    st = np.flatnonzero(b); en = np.r_[st[1:], n]
    r = np.empty(n); r[o] = ((st + en - 1) / 2.0)[bid]
    return r / max(1, n - 1)


def rerank_average(X, names, start):
    """Copy of X (float32) with the seven rank features recomputed per list from the float32 source columns."""
    names = list(names); X2 = np.array(X, np.float32, copy=True)
    for f, (src, sgn, ab) in RANK_SOURCES.items():
        if f not in names or src not in names:
            continue
        v = X2[:, names.index(src)].astype(np.float32).astype(np.float64)
        v = sgn * (np.abs(v) if ab else v); j = names.index(f)
        for m in range(len(start) - 1):
            a, b = int(start[m]), int(start[m + 1])
            if b > a:
                X2[a:b, j] = rank_norm_average(v[a:b])
    return X2


def feature_names(names_x, manifest):
    """The 141 ranker features: the engine's 94 + the MANIFEST family features without derivation, in that order."""
    fam = [f for f in manifest['family_features'] if not f.startswith('derivation__')]
    return list(names_x) + fam


def load_run(rows_dir, mids=None, pool_meta=None):
    """Concatenate the chunks of one run (either format). Molecules in first-chunk order; a molecule run twice counts once."""
    pm = pool_meta if pool_meta is not None else pd.read_parquet(POOL_META, columns=['key', 'smiles', 'formula'])
    X, F, pid, y, inj, smi, form, key, start, mids_, lib, empty = [], [], [], [], [], [], [], [], [0], [], [], []
    names_x = names_f = None; seen = set()
    for f in sorted(glob.glob(os.path.join(rows_dir, '*.parquet'))):
        m = pd.read_parquet(f); z = np.load(f[:-8] + '.npz')
        if len(z['names_f']) == 0:
            for mid in m.mid:
                if mid not in seen and (mids is None or mid in mids):
                    seen.add(mid); empty.append(mid)
            continue
        if names_x is None:
            names_x = [str(s) for s in z['names_x']]; names_f = [str(s) for s in z['names_f']]
        g = z['g']; st = np.r_[0, np.cumsum(np.bincount(g, minlength=len(m)))]
        strs = {}
        for d in json.loads(str(z['strs'] if 'strs' in z.files else z['gen_json'])):
            strs[(d['mid'], int(d['j']))] = d
        zinj = z['inj'] if 'inj' in z.files else np.zeros(len(g), bool)
        for i, mid in enumerate(m.mid):
            if mid in seen or (mids is not None and mid not in mids):
                continue
            seen.add(mid); sl = slice(int(st[i]), int(st[i + 1])); n = sl.stop - sl.start
            if n == 0:
                empty.append(mid); continue
            p = z['pid'][sl]
            for c, v in ((X, z['X'][sl]), (F, z['F'][sl]), (pid, p), (y, z['y'][sl]), (inj, zinj[sl])):
                c.append(v)
            for j, q in enumerate(p):
                if q >= 0:
                    smi.append(pm.smiles.values[q]); form.append(pm.formula.values[q]); key.append(pm.key.values[q])
                else:
                    d = strs[(mid, j)]; smi.append(d['smiles']); form.append(d['formula']); key.append(d['key'])
            start.append(start[-1] + n); mids_.append(mid); lib.append(float(m.lib_max.values[i]))
    R = SimpleNamespace(names_x=names_x, names_f=names_f, mids=mids_, empty=empty, start=np.array(start, np.int64), lib_max=np.array(lib))
    if mids_:
        R.X = np.concatenate(X).astype(np.float32); R.F = np.concatenate(F).astype(np.float32); R.pid = np.concatenate(pid).astype(np.int64)
        R.y = np.concatenate(y).astype(bool); R.inj = np.concatenate(inj).astype(bool)
        R.smiles = np.array(smi, object); R.formula = np.array(form, object); R.key = np.array(key, object)
    return R


def strip_injected(R):
    """The run without injected rows (molecules left without a row move to `empty`)."""
    if not R.mids or not R.inj.any():
        return R
    keep = ~R.inj; n = np.diff(R.start); mol = np.repeat(np.arange(len(R.mids)), n)
    cnt = np.bincount(mol[keep], minlength=len(R.mids)); ok = cnt > 0
    S = SimpleNamespace(names_x=R.names_x, names_f=R.names_f, mids=[m for m, o in zip(R.mids, ok) if o],
                        empty=list(R.empty) + [m for m, o in zip(R.mids, ok) if not o], lib_max=R.lib_max[ok],
                        start=np.r_[0, np.cumsum(cnt[ok])].astype(np.int64))
    for c in ('X', 'F', 'pid', 'y', 'inj', 'smiles', 'formula', 'key'):
        setattr(S, c, getattr(R, c)[keep])
    return S


def spectra_from_rows(rows):
    """Query rows (train.parquet layout) -> the engine's query dicts (== prototypes/local_engine.spectra_from_rows)."""
    out = []
    for r in rows.itertuples():
        ce = r.collision_energy_ev
        ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
        out.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                        mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct, prec=float(r.precursor_mz),
                        ce=float(ce_arr.mean()) if len(ce_arr) else np.nan, ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
    return out


class Fragnet:
    """fe_v4's fragnet family (MIT) with a chosen stage A table and stage B model, on stored candidate lists."""

    def __init__(self, code=CODE, stage_a=OURS_A, stage_b=OURS_B, pool_smiles=None):
        for p in (os.path.join(code, 'fe_v4'), code):
            if p not in sys.path:
                sys.path.insert(0, p)
        import lightgbm as lgb
        import ctxlib
        from fams import fragnet as FN
        self.FN, self.ctx = FN, ctxlib
        FN._R.clear(); FN._GEN.clear()
        FN._R['A'] = FN.StageA(stage_a)
        FN._R['B'] = lgb.Booster(model_file=stage_b)
        assert list(FN._R['B'].feature_name()) == FN.BNAMES, 'stage B was trained with another feature list'
        FN._R['rec'] = FN.PoolRecords(pool_smiles)
        self.res = SimpleNamespace(cfg=SimpleNamespace(max_query_spectra=16))
        self.names = ['fragnet__' + k for k in FN.NAMES]

    def prepare(self, spectra):
        """== ctxlib.Resources.prepare (the engine's query cleaning)."""
        return self.ctx.Resources.prepare(self.res, spectra)

    def compute(self, prep, X, names_x, pid, smiles, formula):
        v = SimpleNamespace(n=len(pid), spectra=prep, X=self.ctx.to_v6_layout(np.asarray(X, np.float32), list(names_x)),
                            pid=np.asarray(pid, np.int64), smiles=list(smiles), formula=list(formula), res=self.res)
        d = self.FN.compute(v)
        return np.stack([np.asarray(d[k], np.float32) for k in self.FN.NAMES], 1)


def recompute_fragnet(R, Q, fr, log_every=200):
    """(n_rows, 14) fragnet features of every list of R; Q = queries table (molecule_id, train.parquet columns)."""
    QG = {m: g for m, g in Q.groupby('molecule_id', sort=False)}
    out = np.zeros((len(R.y), 14), np.float32); t0 = time.time()
    for i, mid in enumerate(R.mids):
        a, b = int(R.start[i]), int(R.start[i + 1])
        prep = fr.prepare(spectra_from_rows(QG[mid]))
        out[a:b] = fr.compute(prep, R.X[a:b], R.names_x, R.pid[a:b], R.smiles[a:b], R.formula[a:b])
        if log_every and (i + 1) % log_every == 0:
            print(f'  fragnet {i + 1}/{len(R.mids)} lists {time.time() - t0:.0f}s', flush=True)
    return out


def build(rows, queries, out, stage_a=OURS_A, stage_b=OURS_B, code=CODE, mids=None, manifest=None):
    """Write the 141-feature cache of one run; returns a small report."""
    t0 = time.time()
    pm = pd.read_parquet(POOL_META, columns=['key', 'smiles', 'formula'])
    R = strip_injected(load_run(rows, mids, pm))
    if not R.mids:
        raise SystemExit(f'no rows in {rows}')
    man = manifest or json.load(open(os.path.join(os.path.dirname(code), 'MANIFEST.json')))
    names = feature_names(R.names_x, man)
    Q = pd.read_parquet(queries); Q = Q[Q.molecule_id.isin(set(R.mids))]
    fr = Fragnet(code, stage_a, stage_b, pm.smiles.values.astype(object))
    FNW = recompute_fragnet(R, Q, fr)
    X2 = rerank_average(R.X, R.names_x, R.start)
    px = {n: i for i, n in enumerate(R.names_x)}; pf = {n: i for i, n in enumerate(R.names_f)}; pn = {n: i for i, n in enumerate(fr.names)}
    cols = [X2[:, px[f]] if f in px else (FNW[:, pn[f]] if f in pn else R.F[:, pf[f]]) for f in names]
    M = np.ascontiguousarray(np.stack(cols, 1), np.float32)
    old = R.F[:, [pf[f] for f in fr.names]]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out + '.tmp.npz', X=M, names=np.array(names), y=R.y, start=R.start, mids=np.array(R.mids), empty=np.array(R.empty, dtype='U64'),
                        key=np.array([str(k) for k in R.key]), pid=R.pid, lib_max=R.lib_max, smiles=np.array([str(s) for s in R.smiles]),
                        formula=np.array([str(s) for s in R.formula]), fn_stored=old)
    os.replace(out + '.tmp.npz', out)
    has = np.array([R.y[R.start[i]:R.start[i + 1]].any() for i in range(len(R.mids))])
    rep = dict(rows=rows, lists=len(R.mids), empty=len(R.empty), n_rows=int(len(R.y)), answer_listed=float(has.mean()), features=len(names),
               fragnet_corr_vs_stored=float(np.corrcoef(FNW[:, 0], old[:, 0])[0, 1]) if len(FNW) > 2 else None, secs=round(time.time() - t0, 1))
    print(json.dumps(rep), flush=True)
    return rep


def parity(rows, queries, n=60, code=CODE, seed=0):
    """Shipped stage A / B through this module against the stored fragnet columns of a no-join run."""
    RES = os.path.join(code, 'fe_v4', 'res', 'fragnet')
    pm = pd.read_parquet(POOL_META, columns=['key', 'smiles', 'formula'])
    R0 = load_run(rows, None, pm)
    pick = set(np.random.RandomState(seed).choice(R0.mids, min(n, len(R0.mids)), replace=False))
    R = strip_injected(load_run(rows, pick, pm))
    Q = pd.read_parquet(queries); Q = Q[Q.molecule_id.isin(set(R.mids))]
    fr = Fragnet(code, os.path.join(RES, 'stageA.npz'), os.path.join(RES, 'stageB.txt'), pm.smiles.values.astype(object))
    W = recompute_fragnet(R, Q, fr, log_every=0)
    pf = {nm: i for i, nm in enumerate(R.names_f)}; old = R.F[:, [pf[f] for f in fr.names]]
    d = np.abs(W.astype(np.float64) - old.astype(np.float64))
    per = {f: float(d[:, j].max()) for j, f in enumerate(fr.names)}
    bad = [R.mids[i] for i in range(len(R.mids)) if d[R.start[i]:R.start[i + 1]].max(initial=0) > 1e-4]
    X2 = rerank_average(R.X, R.names_x, R.start)
    rep = dict(rows=rows, lists=len(R.mids), n_rows=int(len(R.y)), max_abs_diff=float(d.max()), per_feature=per, lists_over_1e4=len(bad), bad=bad[:10],
               rank_features_changed_share={f: float((X2[:, R.names_x.index(f)] != R.X[:, R.names_x.index(f)]).mean()) for f in RANK_SOURCES})
    print(json.dumps(rep, indent=1), flush=True)
    return rep


if __name__ == '__main__':
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ.setdefault(_v, '1')
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['build', 'parity']); ap.add_argument('--rows', required=True)
    ap.add_argument('--queries', required=True); ap.add_argument('--out', default=None); ap.add_argument('--stage_a', default=OURS_A)
    ap.add_argument('--stage_b', default=OURS_B); ap.add_argument('--code', default=CODE); ap.add_argument('--mids', default=None)
    ap.add_argument('--n', type=int, default=60); ap.add_argument('--report', default=None); a = ap.parse_args()
    if a.cmd == 'build':
        rep = build(a.rows, a.queries, a.out, a.stage_a, a.stage_b, a.code, set(open(a.mids).read().split()) if a.mids else None)
    else:
        rep = parity(a.rows, a.queries, a.n, a.code)
    if a.report:
        json.dump(rep, open(a.report, 'w'), indent=1)
