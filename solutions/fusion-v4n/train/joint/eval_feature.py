"""Phase-1 evaluation of a joint-model score file: stand-alone ranks, the as-feature ranker refit, the primary metric with molecule-bootstrap errors.

Wraps bench.py (the benchmark, `refit`, `score_features` are used unchanged). For one score file it stores the rank of the answer in every
benchmark list under five orders (`ranks_<half>.npz`): the score alone, the refit ranker without and with the score's four features
(LightGBM seeds averaged), and both inside the rank average with the public ranker. From the stored ranks:
  primary   gain in MRR@25 of the refit with the score over the refit without it, mean of the c2J gain and the dropJ gain; the standard
            error resamples MOLECULES (a molecule's c2J and dropJ lists stay together), 2,000 resamples
  compare   the same paired statistics between two runs (run A's refit against run B's refit)
  strata    the gain per (view, stratum) of bench.Bench.strata plus spectra-per-molecule and candidate-count bands
On 'final' the refit is fitted on all of DEV (with DEV's scores) and applied to FINAL: the score file must cover both halves.

usage: python train/joint/eval_feature.py --scores S.parquet [--half dev] [--seeds 0,4] [--threads 3] --out <run folder>
       python train/joint/eval_feature.py --null fz --out <folder>          (a ranker input passed through the same refit: the null control)
       python train/joint/eval_feature.py --compare A/ranks_dev.npz B/ranks_dev.npz
"""
import argparse, json, os, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as BN

ORDERS = ('alone', 'fz', 'base', 'new', 'base_avg', 'new_avg')


def compute(scores, half='dev', seeds=(0, 4), threads=3, B=None, row_vec=None, log=print):
    """Ranks of the answer per benchmark list of `half` under every order. `scores`: parquet path(s) (several are concatenated) or None with row_vec."""
    B = B or BN.Bench(); lists = B.lists(half)
    if row_vec is None:
        s = pd.concat([pd.read_parquet(p) for p in ([scores] if isinstance(scores, str) else scores)], ignore_index=True); s = B.row_scores(s)
    else:
        s = np.asarray(row_vec, np.float64)
    cov = float(np.mean([np.isfinite(s[B.start[i]:B.start[i] + B.n[i]]).mean() for i in lists])); log(f'{half}: {len(lists)} lists, share of candidates scored {cov:.4f}')
    tag = '_'.join(map(str, seeds)); os.makedirs(B.root + '/refit', exist_ok=True); t0 = time.time()
    base = BN.refit(B, half, None, seeds, threads, cache=f'{B.root}/refit/base_{half}_s{tag}.npy', log=log)
    new = BN.refit(B, half, BN.score_features(B, s), seeds, threads, log=log); pub = B.R['pub'].values
    r = dict(alone=B.ranks(lists, score=s, tie='random'), fz=B.ranks(lists, name='fz'), base=B.ranks(lists, score=base), new=B.ranks(lists, score=new),
             base_avg=B.ranks(lists, score=base, avg_with=pub), new_avg=B.ranks(lists, score=new, avg_with=pub))
    L = B.L.iloc[lists]
    r.update(list=lists.astype(np.int64), mid=L.mid.values, view=L.view.values.astype(str), covered=np.float64(cov), refit_s=np.float64(time.time() - t0))
    return r


def _boot_pair(mid, view, d, n_boot=2000, seed=0):
    """Mean of the c2J mean and the dropJ mean of d (one value per list), and its SE resampling molecules."""
    m = np.isin(view, ('c2J', 'dropJ')); u, inv = np.unique(mid[m], return_inverse=True); n = len(u)
    S = np.zeros((2, n)); C = np.zeros((2, n))
    for j, v in enumerate(('c2J', 'dropJ')):
        k = view[m] == v; np.add.at(S[j], inv[k], d[m][k]); np.add.at(C[j], inv[k], 1.0)
    est = 0.5 * (S[0].sum() / C[0].sum() + S[1].sum() / C[1].sum())
    rng = np.random.default_rng(seed); ix = rng.integers(0, n, (n_boot, n))
    bs = 0.5 * (S[0][ix].sum(1) / np.maximum(C[0][ix].sum(1), 1) + S[1][ix].sum(1) / np.maximum(C[1][ix].sum(1), 1))
    return float(est), float(bs.std(ddof=1))


def _view(view, v, d):
    m = view == v; return float(d[m].mean()), BN.boot_se(d[m]), int(m.sum()), int((d[m] > 1e-9).sum()), int((d[m] < -1e-9).sum())


def summary(r):
    """Numbers of one run from its stored ranks."""
    rr = {k: BN.rr_of(r[k]) for k in ORDERS}; view = r['view'].astype(str); mid = r['mid']; out = dict(covered=float(r['covered']))
    for v in BN.VIEWS:
        m = view == v
        out[v] = dict(n=int(m.sum()), alone_top1=float((r['alone'][m] == 1).mean()), alone_mrr=float(rr['alone'][m].mean()), fz_mrr=float(rr['fz'][m].mean()),
                      base_top1=float((r['base'][m] == 1).mean()), base_mrr=float(rr['base'][m].mean()), new_top1=float((r['new'][m] == 1).mean()), new_mrr=float(rr['new'][m].mean()))
        for nm, a, b in (('gain', 'new', 'base'), ('gain_avg', 'new_avg', 'base_avg')):
            g, se, n, w, l = _view(view, v, rr[a] - rr[b]); out[v].update({nm: g, nm + '_se': se, nm + '_wl': [w, l]})
            g, se, n, w, l = _view(view, v, (r[a] == 1).astype(float) - (r[b] == 1)); out[v].update({nm + '_top1': g, nm + '_top1_se': se})
    out['primary'], out['primary_se'] = _boot_pair(mid, view, rr['new'] - rr['base'])
    out['primary_avg'], out['primary_avg_se'] = _boot_pair(mid, view, rr['new_avg'] - rr['base_avg'])
    out['primary_top1'], out['primary_top1_se'] = _boot_pair(mid, view, (r['new'] == 1).astype(float) - (r['base'] == 1))
    out['alone_mrr_J'] = 0.5 * (out['c2J']['alone_mrr'] + out['dropJ']['alone_mrr'])
    return out


def compare(ra, rb, key='new'):
    """Run A minus run B on the same lists: primary (mean of c2J and dropJ) of the refit with each score, the c2N difference, the stand-alone difference."""
    assert (ra['list'] == rb['list']).all(); view = ra['view'].astype(str); mid = ra['mid']; out = {}
    d = BN.rr_of(ra[key]) - BN.rr_of(rb[key]); out['primary_diff'], out['primary_diff_se'] = _boot_pair(mid, view, d)
    out['c2N_diff'], out['c2N_diff_se'] = _view(view, 'c2N', d)[:2]
    d = BN.rr_of(ra['alone']) - BN.rr_of(rb['alone']); out['alone_J_diff'], out['alone_J_diff_se'] = _boot_pair(mid, view, d)
    return out


def strata_table(r, B=None, key=('new', 'base')):
    """Gain per (view, stratum): the strata of the benchmark plus the number of query spectra and candidate-count bands."""
    B = B or BN.Bench(); lists = r['list']; S = B.strata(lists); L = B.L.iloc[lists]; view = L.view.values
    ns = B.spectra().groupby('mid').size().reindex(L.mid.values).fillna(0).values; nc = L.n.values
    for v in BN.VIEWS:
        b = view == v
        for lab, m in (('spectra 1', ns == 1), ('spectra 2-3', (ns >= 2) & (ns <= 3)), ('spectra 4+', ns >= 4), ('candidates <= 50', nc <= 50), ('candidates 51-150', (nc > 50) & (nc <= 150)),
                       ('candidates 151-400', (nc > 150) & (nc <= 400)), ('candidates > 400', nc > 400)):
            S[f'{v} / {lab}'] = b & m
    d = BN.rr_of(r[key[0]]) - BN.rr_of(r[key[1]]); dt = (r[key[0]] == 1).astype(float) - (r[key[1]] == 1); a = BN.rr_of(r['alone']); rows = []
    for name, m in S.items():
        if m.sum() < 20:
            continue
        v, s = name.split(' / ')
        rows.append(dict(view=v, stratum=s, n=int(m.sum()), base_mrr=float(BN.rr_of(r[key[1]])[m].mean()), gain=float(d[m].mean()), gain_se=BN.boot_se(d[m]), gain_top1=float(dt[m].mean()),
                         gain_top1_se=BN.boot_se(dt[m]), wins=int((d[m] > 1e-9).sum()), losses=int((d[m] < -1e-9).sum()), alone_mrr=float(a[m].mean())))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--scores', default=''); ap.add_argument('--null', default=''); ap.add_argument('--half', default='dev'); ap.add_argument('--seeds', default='0,4')
    ap.add_argument('--threads', type=int, default=3); ap.add_argument('--out', default=''); ap.add_argument('--compare', nargs=2, default=None); a = ap.parse_args()
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if a.compare:
        print(json.dumps(compare(np.load(a.compare[0], allow_pickle=True), np.load(a.compare[1], allow_pickle=True)), indent=1)); sys.exit(0)
    B = BN.Bench(); seeds = tuple(int(x) for x in a.seeds.split(','))
    r = compute(a.scores.split(',') if ',' in a.scores else a.scores, a.half, seeds, a.threads, B, row_vec=B.R[a.null].values if a.null else None, log=lambda *x: print(*x, flush=True))
    os.makedirs(a.out, exist_ok=True); np.savez(f'{a.out}/ranks_{a.half}.tmp.npz', **r); os.replace(f'{a.out}/ranks_{a.half}.tmp.npz', f'{a.out}/ranks_{a.half}.npz')
    s = summary(r); s['seeds'] = list(seeds); s['refit_s'] = float(r['refit_s']); json.dump(s, open(f'{a.out}/eval_{a.half}.json', 'w'), indent=1); print(json.dumps(s, indent=1))
