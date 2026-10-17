"""Our engine-1 ranker for stack1 compliance step 2 (replaces ahmedberatozer's ranker_v4b.pkl): LightGBM lambdarank on 141 features.

Rows: engine runs without the PubChem join of the production ranker's molecules (hold-out c2 / c1, training plan, mixed regime and
Enveda panel in versions c2 / c1 / drop), our base_s0 network in the engine bank and in all three model_views slots; features by
`e1_rows.py` (order-free rank features, fragnet with OUR stage A / B, derivation dropped). Recipe: the public ranker's stored
hyper-parameters (lambdarank, lr 0.03, 63 leaves, ...), 500 rounds, seeds 0..3, every list whose answer is a candidate trains, no folds.
Out-of-fold scores (5 scaffold folds = `cvfold` of the benchmark molecules, seeds 0 and 4) are written for the local replay only.

  python fit_ranker_e1.py feat [--workers 2]          feature caches <work>/feat/<set>.npz (e1_rows.build) for every set whose rows exist
  python fit_ranker_e1.py fit  --out DIR [--threads 4] final ranker: ranker_s<k>.txt, features.json, meta.json (refuses to overwrite)
  python fit_ranker_e1.py cv   [--threads 4]           out-of-fold scores <work>/cv/<set>.npz for the sets in CV_SETS
Data: train.parquet only (rows are engine runs on train.parquet molecules against the pool of casmi26-v2-pool, built from train.parquet
and COCONUT); no NIST; no third-party weights (the network is ours, stage A / B are ours, the public ranker is not read).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1_rows as ER  # noqa: E402

P4 = 'C:/Users/hemre/casmi-work/ownfp/p4/base_s0'; WORK = 'C:/Users/hemre/casmi-work/comply/step2'
D = 'D:/casmi-data'; BENCH = D + '/joint/bench/mols.parquet'
Q = dict(hold=D + '/ranker_study/queries.parquet', tp=D + '/trainplan/queries.parquet', mx=D + '/mixed_regime/queries.parquet',
         p2=D + '/panel2/regime/queries.parquet')
SETS = {'hold_c2': (P4 + '/rs/rows/c2_fold0', Q['hold']), 'hold_c1': (P4 + '/rs/rows/c1_fold0', Q['hold'])}
for _reg in ('tp', 'mx', 'p2'):
    for _ver in ('c2', 'c1', 'drop'):
        SETS[f'{_reg}_{_ver}_n0'] = (f'{P4}/{_reg}/rows/{_ver}_n0', Q[_reg])
# a run is complete when its stage marker exists (P4 queue names for the runs of the ownfp study, s2_* for the runs added here)
DONE = dict(hold_c2='eng_c2', hold_c1='eng_c1', p2_c2_n0='p2_c2n0', p2_c1_n0='p2_c1n0')
for _s in SETS:
    DONE.setdefault(_s, 's2_' + _s.replace('_n0', 'n0').replace('_', '_', 1))
TRAIN = list(SETS)                                   # every set trains (drop lists only where the engine generated the answer)
CV_SETS = [s for s in SETS if s.startswith(('mx_', 'p2_'))]
# the public ranker's stored hyper-parameters (ranker_v4b.pkl 'params' / 'rounds'; numbers only, its boosters are never read)
PARAMS = dict(objective='lambdarank', metric='None', learning_rate=0.03, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_depth=-1, lambdarank_truncation_level=30, label_gain=[0, 1], verbose=-1)
ROUNDS = 500


def feat_path(st):
    return f'{WORK}/feat/{st}.npz'


def _one(st):
    rows, q = SETS[st]
    return st, ER.build(rows, q, feat_path(st))


def do_feat(workers=2, only=None):
    todo = [s for s in SETS if (only is None or s in only) and not os.path.exists(feat_path(s)) and os.path.exists(f'{P4}/done/{DONE[s]}')]
    print('to build', todo, flush=True)
    rep = {}
    with ProcessPoolExecutor(workers) as ex:
        for st, r in ex.map(_one, todo):
            rep[st] = r
    if rep:
        old = json.load(open(f'{WORK}/feat/report.json')) if os.path.exists(f'{WORK}/feat/report.json') else {}
        old.update(rep); json.dump(old, open(f'{WORK}/feat/report.json', 'w'), indent=1)


class Cache:
    def __init__(self, st):
        z = np.load(feat_path(st), allow_pickle=False)
        self.st = st; self.X = z['X']; self.names = [str(x) for x in z['names']]; self.y = z['y']; self.start = z['start']
        self.mids = [str(x) for x in z['mids']]; self.n = np.diff(self.start); self.key = z['key']
        self.has = np.array([self.y[self.start[i]:self.start[i + 1]].any() for i in range(len(self.mids))])


def training_matrix(caches, keep_mol=None):
    """X, y, group sizes over the lists with a listed answer (and keep_mol(cache) True)."""
    Xs, ys, gs, names = [], [], [], None
    for c in caches:
        s = c.has.copy() if keep_mol is None else c.has & keep_mol(c)
        rm = np.repeat(s, c.n)
        if names is None:
            names = c.names
        assert c.names == names, ('feature order differs', c.st)
        Xs.append(c.X[rm]); ys.append(c.y[rm]); gs.append(c.n[s])
    return np.concatenate(Xs), np.concatenate(ys).astype(np.int32), np.concatenate(gs), names


def do_fit(out, seeds=4, threads=4):
    import lightgbm as lgb
    assert not os.path.exists(os.path.join(out, 'ranker_s0.txt')), 'never overwrite an exported ranker'
    os.makedirs(out, exist_ok=True)
    cs = [Cache(s) for s in TRAIN if os.path.exists(feat_path(s))]
    missing = [s for s in TRAIN if not os.path.exists(feat_path(s))]
    assert not missing, f'feature caches missing: {missing}'
    X, y, g, names = training_matrix(cs)
    print(f'fit: {len(g)} lists, {len(y)} rows, {len(names)} features', flush=True)
    for s in range(seeds):
        t0 = time.time(); p = dict(PARAMS, num_threads=threads, seed=s, bagging_seed=s, feature_fraction_seed=s)
        m = lgb.train(p, lgb.Dataset(X, label=y, group=g, feature_name=[f'f{i}' for i in range(len(names))], free_raw_data=False), num_boost_round=ROUNDS)
        m.save_model(os.path.join(out, f'ranker_s{s}.txt')); print(f'seed {s} {time.time() - t0:.0f}s', flush=True)
    json.dump(names, open(os.path.join(out, 'features.json'), 'w'), indent=1)
    sets = {c.st: dict(lists=int(c.has.sum()), of=len(c.mids), rows=int(np.repeat(c.has, c.n).sum())) for c in cs}
    json.dump(dict(what='engine-1 ranker of stack1 compliance step 2 (replaces ranker_v4b.pkl)', recipe='lambdarank, public hyper-parameters, '
                   f'{ROUNDS} rounds, seeds 0..{seeds - 1}, no folds; prediction = mean of the boosters', params=PARAMS, rounds=ROUNDS, seeds=seeds,
                   lists=int(len(g)), rows=int(len(y)), features=len(names), sets=sets,
                   rows_from='engine runs without the PubChem join, base_s0 in the engine bank and all three model_views slots (ownfp study)',
                   features_from='e1_rows.py: 94 engine features with the seven rank features recomputed with average ties + family features of '
                                 'MANIFEST.json without derivation; fragnet recomputed with our stage A / B (fit_fragnet.py)',
                   data='train.parquet (competition) and the casmi26-v2-pool candidate pool (train.parquet + COCONUT); no NIST',
                   lightgbm=lgb.__version__), open(os.path.join(out, 'meta.json'), 'w'), indent=1)


def do_cv(seeds=(0, 4), threads=4):
    import lightgbm as lgb
    fold = dict(pd.read_parquet(BENCH, columns=['mid', 'cvfold']).itertuples(index=False, name=None))
    cs = [Cache(s) for s in TRAIN]
    for c in cs:
        c.fold = np.array([fold.get(m, -1) for m in c.mids]); c.sc = np.zeros(len(c.y))
    os.makedirs(f'{WORK}/cv', exist_ok=True)
    for k in range(5):
        X, y, g, names = training_matrix(cs, keep_mol=lambda c: c.fold != k)
        for s in seeds:
            t0 = time.time(); p = dict(PARAMS, num_threads=threads, seed=s, bagging_seed=s, feature_fraction_seed=s)
            m = lgb.train(p, lgb.Dataset(X, label=y, group=g), num_boost_round=ROUNDS)
            for c in cs:
                if c.st in CV_SETS:
                    te = np.repeat(c.fold == k, c.n)
                    if te.any():
                        c.sc[te] += m.predict(c.X[te], num_threads=threads) / len(seeds)
            print(f'fold {k} seed {s}: {len(g)} lists {len(y)} rows {time.time() - t0:.0f}s', flush=True)
    for c in cs:
        if c.st in CV_SETS:
            assert (c.fold >= 0).all(), ('molecule without a fold', c.st)
            np.savez_compressed(f'{WORK}/cv/{c.st}.npz', score=c.sc.astype(np.float32), fold=c.fold)


if __name__ == '__main__':
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['feat', 'fit', 'cv']); ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--only', default=None)
    a = ap.parse_args()
    if a.cmd == 'feat':
        do_feat(a.workers, a.only.split(',') if a.only else None)
    elif a.cmd == 'fit':
        do_fit(a.out, threads=a.threads)
    else:
        do_cv(threads=a.threads)
