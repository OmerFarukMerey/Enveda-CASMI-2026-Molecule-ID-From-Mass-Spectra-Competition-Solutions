"""Scaffold-fold CV of a LightGBM lambdarank isomer reranker on fragment-explanation features vs single scores.
Model inputs come from fragfeat.model_matrix per candidate group, the function the Kaggle cell uses.
usage: eval_sets.py SETS_DIR [save]   (save: also fit on every fold and write model.txt)"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
sys.path.insert(0, str(Path(__file__).parent))
import fragfeat as F

d = Path(sys.argv[1])
df = pd.read_parquet(d / 'sets.parquet')
df = df.dropna(subset=F.FEATURES)
df = df[df.groupby('qkey').label.transform('sum') == 1]
df = df[df.groupby('qkey').label.transform('size') >= 4].sort_values(['qkey', 'label'], ascending=[True, False], kind='stable').reset_index(drop=True)
M = np.zeros((len(df), len(F.MODEL_FEATURES)))
X = df[F.FEATURES].to_numpy()
starts = np.flatnonzero(np.r_[True, df.qkey.values[1:] != df.qkey.values[:-1]]); ends = np.r_[starts[1:], len(df)]
for a, b in zip(starts, ends):
    M[a:b] = F.model_matrix(X[a:b])
sizes = ends - starts
PARAMS = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambdarank_truncation_level=16, verbose=-1, seed=1, num_threads=16)
ROUNDS = 400


def metrics(score, mask=None):
    x = df.assign(_s=score)
    if mask is not None:
        x = x[mask]
    r = x.groupby('qkey')._s.rank(ascending=False, method='average')
    rt = r[x.label == 1]
    return dict(n=int(len(rt)), top1=round(float((rt <= 1).mean()), 4), mrr=round(float((1 / rt).mean()), 4))


def cv(cols):
    pred = np.zeros(len(df)); gid = np.repeat(np.arange(len(starts)), sizes); fold_g = df.fold.values[starts]
    for k in range(5):
        trg = np.flatnonzero(fold_g != k); tr = np.isin(gid, trg); te = ~tr
        m = lgb.train(PARAMS, lgb.Dataset(M[tr][:, cols], df.label.values[tr], group=sizes[trg]), ROUNDS)
        pred[te] = m.predict(M[te][:, cols])
    return pred


hard = df.kind.isin(['true', 'sim']).values
rep = {}
for f in ['rd_mean', 'lin1_mean', 'sq2_mean', 'clin2_mean', 'csq2_mean']:
    rep['single ' + f] = dict(all=metrics(df[f].values), hard=metrics(df[f].values, hard))
allc = list(range(len(F.MODEL_FEATURES)))
pred = cv(allc)
rep['lgb'] = dict(all=metrics(pred), hard=metrics(pred, hard))
# pairwise view: how often the model's top pick differs from rd's, and who is right then
x = df.assign(p=pred)
top_p = x.loc[x.groupby('qkey').p.idxmax()]; top_r = x.loc[x.groupby('qkey').rd_mean.idxmax()]
diff = top_p.qkey.values[top_p.cand.values != top_r.set_index('qkey').loc[top_p.qkey].cand.values]
rep['disagree with rd'] = dict(sets=int(len(diff)), lgb_right=int(top_p.set_index('qkey').loc[diff].label.sum()),
                               rd_right=int(top_r.set_index('qkey').loc[diff].label.sum()))
for k, v in rep.items():
    print(k, json.dumps(v), flush=True)
df.assign(pred=pred)[['qkey', 'fold', 'cand', 'label', 'kind', 'sim', 'rd_mean', 'pred']].to_parquet(d / 'cv_pred.parquet')
if len(sys.argv) > 2 and sys.argv[2] == 'save':
    m = lgb.train(PARAMS, lgb.Dataset(M, df.label.values, group=sizes), ROUNDS)
    m.save_model(str(d / 'model.txt'))
    imp = sorted(zip(m.feature_importance('gain'), F.MODEL_FEATURES), reverse=True)[:12]
    print('saved', d / 'model.txt', [f'{f} {g:.0f}' for g, f in imp])
