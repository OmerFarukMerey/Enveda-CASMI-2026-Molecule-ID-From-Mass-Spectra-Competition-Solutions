"""Our fragnet stage A table and stage B model (stack1 compliance, replaces ahmedberatozer's fe_v4/res/fragnet/stageA.npz and stageB.txt).

Training data: the fragnet training data shipped in ahmedberatozer/casmi26-v4b-models (MIT code; card: built only from train.parquet and
COCONUT): res/fragnet/countsA.npz (stage-A class counts of part A: 16,144 train-fold structures x one library spectrum, decoys from the
pool) and res/fragnet/featB.npz (stage-B aggregates of parts B / V). Their builder (`res_fragnet --step sets / countA`) is not shipped:
stated as the limit of this reproduction. Model functions (save_stageA / fit_chain, stageB_features, BNAMES) come from the MIT
fe_v4.fams.fragnet; the driver (data selection, grouping, LightGBM call and its parameters as stored in the shipped stageB.txt) is ours.
Step 0 of the compliance plan: stage A comes out identical to the shipped table; stage B is equivalent (prediction correlation 0.998 on
part V, within-group MRR 0.800 against 0.799), not bit-identical. Deterministic given LightGBM version and thread count (3).
usage: python fit_fragnet.py --code D:/casmi-data/ranker_study/ds/v4b/code --out DIR      (writes stageA.npz, stageB.txt, fit_fragnet.json)
"""
import argparse, hashlib, json, os, sys, time

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--code', default='D:/casmi-data/ranker_study/ds/v4b/code'); ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--exclude_sids', default=None, help='parquet with a `sid` column (train_structs row): stage-B groups whose training '
                    'structure (res/fragnet/sets.pkl) is listed are left out (clean instrument: structures of the exclusion set E)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in (os.path.join(a.code, 'fe_v4'), a.code):
        sys.path.insert(0, p)
    from fams import fragnet as FN
    import lightgbm as lgb
    RES = os.path.join(a.code, 'fe_v4', 'res', 'fragnet'); rep = dict(lightgbm=lgb.__version__, threads=a.threads)
    # stage A: count tables -> smoothed LLR tables (chains A and H, alpha 80: the configuration named in the module's decision log)
    FN.save_stageA(os.path.join(a.out, 'stageA.npz'), np.load(os.path.join(RES, 'countsA.npz')))
    # stage B: LightGBM lambdarank on part B (groups = training sets of one structure and spectrum), 31 features of the AH80 aggregates
    t0 = time.time(); f = np.load(os.path.join(RES, 'featB.npz'), allow_pickle=True)

    keep = np.ones(len(f['set']), bool)
    if a.exclude_sids:
        import pickle, pandas as pd
        set_sid = np.array([d['sid'] for d in pickle.load(open(os.path.join(RES, 'sets.pkl'), 'rb'))])
        ex = set(pd.read_parquet(a.exclude_sids, columns=['sid']).sid.astype(int))
        keep = ~np.isin(set_sid[f['set']], list(ex))
        rep.update(exclude_sids=a.exclude_sids, excluded_rows=int((~keep).sum()), excluded_groups_B=int(len(np.unique(f['set'][~keep & (f['part'] == 'B')]))),
                   excluded_groups_V=int(len(np.unique(f['set'][~keep & (f['part'] == 'V')]))))
    def part(name):
        sel = (f['part'] == name) & keep
        st, y, agg = f['set'][sel], f['y'][sel], f['agg_AH80'][sel]
        o = np.argsort(st, kind='stable'); st, y, agg = st[o], y[o], agg[o]
        _, start, cnt = np.unique(st, return_index=True, return_counts=True)
        X = np.concatenate([FN.stageB_features(agg[s:s + c]) for s, c in zip(start, cnt)]).astype(np.float64)
        return X, y, start, cnt
    X, y, _, cnt = part('B')
    params = dict(objective='lambdarank', metric='None', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, bagging_fraction=0.8,
                  bagging_freq=1, feature_fraction=0.7, lambda_l2=1.0, seed=0, deterministic=True, force_row_wise=True, num_threads=a.threads,
                  lambdarank_truncation_level=30, label_gain=[0, 1], verbose=-1)
    bst = lgb.train(params, lgb.Dataset(X, label=y, group=cnt, feature_name=list(FN.BNAMES), free_raw_data=False), num_boost_round=300)
    bst.save_model(os.path.join(a.out, 'stageB.txt'))
    XV, yv, sv, cv = part('V'); pv = bst.predict(XV, num_threads=a.threads); rr = []
    for s, c in zip(sv, cv):
        p, t = pv[s:s + c], yv[s:s + c]
        if t.max() > 0:
            i = int(np.argmax(t)); rr.append(1.0 / (1 + int((p > p[i]).sum())))
    rep.update(stageB_rows=int(len(y)), stageB_groups=int(len(cnt)), params=params, rounds=300, within_group_mrr_V=float(np.mean(rr)), groups_V=len(rr),
               secs=round(time.time() - t0, 1),
               sha256={n: hashlib.sha256(open(os.path.join(a.out, n), 'rb').read()).hexdigest() for n in ('stageA.npz', 'stageB.txt')},
               data_sha256={n: hashlib.sha256(open(os.path.join(RES, n), 'rb').read()).hexdigest() for n in ('countsA.npz', 'featB.npz')})
    json.dump(rep, open(os.path.join(a.out, 'fit_fragnet.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'params'}), flush=True)


if __name__ == '__main__':
    main()
