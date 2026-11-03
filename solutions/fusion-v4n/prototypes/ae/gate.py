"""Learned gate for analog edits: P(top site pick = the true structure) from eval2.py rows.
usage: gate.py OUT_MODEL EVAL_PARQUET [EVAL_PARQUET ...]
5-fold cross-validation grouped by molecule; per molecule the candidate with the highest out-of-fold probability is
the one an overlay would use. Prints fire rate and precision per threshold, then fits on everything."""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

FEATS = ['score', 'abs_delta', 'is_add', 'n_qspec', 'n_spec_hit', 'mean_score', 'second', 'top_score', 'n_prod', 'site1',
         'site_margin', 'prec', 'grp', 'nq', 'na', 'n_match', 'n_shift', 'w_shift', 'w_unshift', 'cand_rank']
PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
ROUNDS = 400

df = pd.concat([pd.read_parquet(p) for p in sys.argv[2:]], ignore_index=True)
n_mol = df.sidx.nunique()
df = df[df.cand_rank >= 0].drop_duplicates(['sidx', 'analog', 'kind', 'direction']).reset_index(drop=True)
df['is_add'] = (df.direction == 'add').astype(int)
y = df.pick_ok.astype(int).to_numpy()
print(f'{n_mol} molecules, {df.sidx.nunique()} with a candidate, {len(df)} rows, positives {y.sum()} '
      f'({df[df.pick_ok].sidx.nunique()} molecules), truth among edits {df[df.truth_in].sidx.nunique()} molecules')
mol = df.sidx.to_numpy(); um = np.unique(mol)
fold_of = dict(zip(um, np.random.default_rng(0).integers(0, 5, len(um))))
fold = np.array([fold_of[m] for m in mol])
oof = np.zeros(len(df))
for k in range(5):
    tr, te = fold != k, fold == k
    b = lgb.train(PARAMS, lgb.Dataset(df.loc[tr, FEATS], y[tr]), ROUNDS)
    oof[te] = b.predict(df.loc[te, FEATS])
df['p'] = oof
best = df.sort_values('p', ascending=False).drop_duplicates('sidx')
print('per molecule (best candidate by out-of-fold p):')
for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6):
    f = best[best.p >= t]
    if len(f):
        print(f'  p >= {t:.2f}: fire {len(f) / n_mol:.4f} ({len(f)}) | precision {f.pick_ok.mean():.3f} | hits {int(f.pick_ok.sum())}')
print('calibration (deciles of p):')
best['bin'] = pd.qcut(best.p, 10, duplicates='drop')
print(best.groupby('bin', observed=True).agg(n=('p', 'size'), p=('p', 'mean'), hit=('pick_ok', 'mean')).round(3).to_string())
b = lgb.train(PARAMS, lgb.Dataset(df[FEATS], y), ROUNDS)
b.save_model(sys.argv[1])
imp = pd.Series(b.feature_importance('gain'), FEATS).sort_values(ascending=False)
print('gain importance:', (imp / imp.sum()).round(3).head(10).to_dict())
