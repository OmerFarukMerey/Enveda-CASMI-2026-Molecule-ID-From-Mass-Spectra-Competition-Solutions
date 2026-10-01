"""Learned fusion ("stacking"): rank every candidate of a run's stage cache with a LightGBM LambdaRank model over all
channel signals, evaluated by grouped K-fold CV on labelled molecules, paired against the hand-tuned fusion_core.

Candidate universe per molecule = base lists ∪ engine-2 ∪ PubChem-only (+ what the default fusion output contains).
Features: base rank / score / z / generated flag / popularity, engine-2 rank, PubChem rank / relative f.z, ICE / GL /
frag scores (raw + z within the candidate's formula group), formula-group size and rank, default-fusion rank, lib_max.
Label: metric key (tautomer-canonical InChIKey14) equals the answer's.

usage: python tools/stack_eval.py <run_dir> [--folds 5] [--seeds 3] [--set KEY=VALUE ...]
Only meaningful on VALIDATION runs (held-out molecules); prints oracle recall, default fusion MRR and CV stacking MRR.
"""
import argparse, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refuse
import metric

FEATS = ['b_rank', 'b_score', 'b_z', 'b_gen', 'b_pop', 'e_rank', 'p_rank', 'p_rel', 'ice', 'gl', 'frag', 'ice_z', 'gl_z',
         'frag_z', 'b_z_grp', 'grp_n', 'grp_rank', 'f_rank', 'lib_max', 'n_in_lists']


def _z(x):
    x = np.asarray(x, np.float64); ok = np.isfinite(x)
    out = np.full(len(x), 0.0)
    if ok.sum() >= 2:
        s = x[ok].std()
        out[ok] = (x[ok] - x[ok].mean()) / (s if s > 0 else 1.0)
    return out


def molecule_table(m, S, fused, pool_pop, formula, key_of):
    b = S['base'][m]; e = S['eng'].get(m) or {}; p = S['pc'].get(m) or {}
    ice = S['ice'].get(m) or {}; gl = S['gl'].get(m) or {}; fr = (S['frag'] or {}).get(m) or {}
    rows = {}
    def row(smi):
        k = key_of(smi)
        if k is None:
            return None
        if k not in rows:
            rows[k] = dict(smiles=smi, key=k, b_rank=999, b_score=np.nan, b_z=np.nan, b_gen=np.nan, b_pop=np.nan,
                           e_rank=999, p_rank=999, p_rel=np.nan, f_rank=999)
        return rows[k]
    bz = _z(b[3]) if len(b[3]) else []
    pids = b[5] if len(b) > 5 else [0] * len(b[0])
    for i, smi in enumerate(b[0]):
        r = row(smi)
        if r and r['b_rank'] == 999:
            r.update(b_rank=i, b_score=b[3][i], b_z=bz[i], b_gen=float(pids[i] < 0),
                     b_pop=float(pool_pop[pids[i]]) if (pool_pop is not None and pids[i] >= 0) else np.nan)
    for i, smi in enumerate(e.get('smiles', [])):
        r = row(smi)
        if r and r['e_rank'] == 999:
            r['e_rank'] = i
    bp = p.get('best_pool_fz')
    for i, smi in enumerate(p.get('pc', [])):
        r = row(smi)
        if r and r['p_rank'] == 999:
            r['p_rank'] = i
            r['p_rel'] = p['pc_fz'][i] - bp if bp is not None and np.isfinite(bp) else np.nan
    for i, smi in enumerate(fused):
        r = row(smi)
        if r and r['f_rank'] == 999:
            r['f_rank'] = i
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return df
    val = lambda d, s: (d.get(s) if d.get(s) is not None else np.nan)
    df['ice'] = [val(ice, s) for s in df.smiles]; df['gl'] = [val(gl, s) for s in df.smiles]
    df['frag'] = [val(fr, s) for s in df.smiles]
    df['formula'] = [formula(s) for s in df.smiles]
    for c in ['ice', 'gl', 'frag', 'b_score']:
        df[c + '_z' if c != 'b_score' else 'b_z_grp'] = df.groupby('formula')[c].transform(lambda x: pd.Series(_z(x.values), index=x.index))
    df['grp_n'] = df.groupby('formula').smiles.transform('size')
    df['grp_rank'] = df.groupby('formula').b_rank.rank(method='first')
    df['lib_max'] = b[2]
    df['n_in_lists'] = (df.b_rank < 999).astype(int) + (df.e_rank < 999) + (df.p_rank < 999)
    df['molecule_id'] = m
    return df


def rr_of(order_keys, truth_key):
    for i, k in enumerate(order_keys[:25], 1):
        if k == truth_key:
            return 1.0 / i
    return 0.0


if __name__ == '__main__':
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--folds', type=int, default=5); ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--set', action='append', default=[])
    a = ap.parse_args()
    S = refuse.load_run(a.run)
    assert S['val_labels'], 'needs a VALIDATION run (stage_cache/val_labels.csv)'
    lab = pd.read_csv(S['val_labels']).set_index('molecule_id')
    cfg = dict(S['meta'].get('cfg', {}))
    for kv in a.set:
        k, v = kv.split('=', 1); cfg[k] = refuse.parse_val(v)
    mods = refuse.load_modules(); pool_pop = refuse.load_pool_pop()
    sub, _, rr_def = refuse.run_one(S, mods, cfg, lab.reset_index(), pool_pop)
    fused = dict(zip(sub.molecule_id, sub.smiles.str.split(';')))
    key_of = metric.ik14; formula = mods[3]
    tabs = []
    for m in S['meta']['mol_order']:
        t = molecule_table(m, S, fused.get(m, []), pool_pop, formula, key_of)
        if len(t):
            t['y'] = (t.key == key_of(lab.smiles[m])).astype(int)
            tabs.append(t)
    D = pd.concat(tabs, ignore_index=True)
    mids = np.array(S['meta']['mol_order'])
    has = D.groupby('molecule_id').y.max()
    print(f'{len(mids)} molecules | {len(D)} candidate rows | oracle (truth anywhere in the lists): {has.reindex(mids).fillna(0).mean():.3f}')
    print(f'default fusion (run cfg + --set): MRR@25 = {rr_def.mean():.4f}')
    preds = np.zeros(len(D))
    for seed in range(a.seeds):
        perm = np.random.RandomState(seed).permutation(mids); folds = np.array_split(perm, a.folds)
        for f in folds:
            te = D.molecule_id.isin(f).values; tr = ~te & D.molecule_id.isin(has[has == 1].index).values
            T = D[tr].sort_values('molecule_id', kind='stable')
            model = lgb.LGBMRanker(objective='lambdarank', n_estimators=300, learning_rate=0.03, num_leaves=15,
                                   min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   random_state=seed, verbose=-1, lambdarank_truncation_level=25)
            model.fit(T[FEATS], T.y, group=T.groupby('molecule_id', sort=False).size().loc[T.molecule_id.unique()].values)
            preds[te] += model.predict(D.loc[te, FEATS]) / a.seeds
    D['p'] = preds
    rr_st = {}
    for m, g in D.groupby('molecule_id'):
        rr_st[m] = rr_of(list(g.sort_values('p', ascending=False).key), key_of(lab.smiles[m]))
    rr_st = pd.Series(rr_st).reindex(mids).fillna(0.0); rr_d = rr_def.reindex(mids).fillna(0.0)
    d = (rr_st - rr_d).values
    print(f'stacking CV ({a.folds} folds x {a.seeds} seeds): MRR@25 = {rr_st.mean():.4f} | delta vs default '
          f'{d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f} | W/L {(d > 0).sum()}/{(d < 0).sum()}')
    imp = pd.Series(model.booster_.feature_importance('gain'), index=FEATS).sort_values(ascending=False)
    print('feature gain (last fold):', ', '.join(f'{k}={v:.0f}' for k, v in imp.head(10).items()))
