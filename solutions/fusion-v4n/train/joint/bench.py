"""The joint-model benchmark: load it, rank answers, strata with bootstrap errors, and `evaluate(scores_parquet)` (alone and as a ranker feature).

A score file is a parquet of (molecule_id, candidate_key, score): one score per candidate key of a molecule, used in every list of that
molecule. Higher is better; a missing candidate ranks last. `evaluate` returns one row per (view, stratum) with top-1 and MRR@25 of the
answer inside its key-deduplicated candidate list, a bootstrap standard error (lists resampled; one list per molecule in a view), and the
paired difference to a baseline. `evaluate(..., as_feature=True)` instead refits the ranker of the fsranker studies (the 165 production
inputs, LightGBM lambdarank, public hyper-parameters) out of fold by scaffold with and without four features derived from the score, and
reports the paired gain of the refit alone and inside the rank average with the public ranker (v33b's base list).

Baselines on the same lists: `random`, `fz` (fold-safe FPNet f.z), `fp_cos` (its fingerprint cosine), `pub` (public ranker; has seen the
molecules through its fit), `ours` (ranker fitted out of fold on the 5,202 molecules), `avg` (rank average of pub and ours, weight 0.5).
Halves: 'dev' for every decision, 'final' only when a phase ends. On 'final' the as-feature refit is fitted on DEV and applied to FINAL.

usage: python train/joint/bench.py --baselines [--half dev]
       python train/joint/bench.py --scores <scores.parquet> [--baseline avg] [--as_feature] [--half dev] [--out table.md]
"""
import argparse, json, os, sys, zlib

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jpaths as P

VIEWS = ['c2J', 'dropJ', 'c2N']
K = 25
PUB_PARAMS = dict(objective='lambdarank', metric='None', learning_rate=0.03, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, max_depth=-1, lambdarank_truncation_level=30, label_gain=[0, 1], verbose=-1)
PUB_ROUNDS = 500
NEWF = ['jm__score', 'jm__z', 'jm__rank', 'jm__gap']


# ---------------------------------------------------------------- ranking
def rank_answer(score, y, key, tie=None):
    """1-based rank of the answer in the key-deduplicated list by descending score; 0 = absent. Ties: list order, or `tie` (smaller first)."""
    score = np.asarray(score, np.float64)
    o = np.argsort(-score, kind='stable') if tie is None else np.lexsort((tie, -score))
    k = key[o]; _, first = np.unique(k, return_index=True); oo = o[np.sort(first)]
    pos = np.flatnonzero(y[oo]); return int(pos[0]) + 1 if len(pos) else 0


def rank_answer_avg(sa, sb, y, key, w=0.5, tie=None):
    """Rank of the answer in the rank average of two score vectors over the same rows (nb/embed/fs_pass.rank_average: weight w on `sb`)."""
    def order(s):
        s = np.asarray(s, np.float64); o = np.argsort(-s, kind='stable') if tie is None else np.lexsort((tie, -s))
        k = key[o]; _, first = np.unique(k, return_index=True); return k[np.sort(first)]
    ra = {int(k): i + 1 for i, k in enumerate(order(sa))}; rb = {int(k): i + 1 for i, k in enumerate(order(sb))}
    ks = sorted(rb, key=lambda k: ((1 - w) * ra[k] + w * rb[k], rb[k], ra[k])); yk = set(int(k) for k in key[y])
    return next((i + 1 for i, k in enumerate(ks) if k in yk), 0)


def boot_se(v, n_boot=1000, seed=0):
    v = np.asarray(v, np.float64); n = len(v)
    if n < 2:
        return float('nan')
    rng = np.random.default_rng(seed); return float(v[rng.integers(0, n, (n_boot, n))].mean(1).std(ddof=1))


# ---------------------------------------------------------------- data
class Bench:
    def __init__(self, root=None):
        r = root or P.BENCH
        self.root = r; self.M = pd.read_parquet(r + '/mols.parquet'); self.L = pd.read_parquet(r + '/lists.parquet')
        self.R = pd.read_parquet(r + '/rows.parquet'); self.C = pd.read_parquet(r + '/cands.parquet')
        self.fused = pd.read_parquet(r + '/fused.parquet') if os.path.exists(r + '/fused.parquet') else None
        self.y = self.R.y.values; self.cand = self.R.cand.values; self.keyc = pd.factorize(self.C.key)[0][self.cand]
        self.start = self.L.start.values; self.n = self.L.n.values; self.lid = self.R.list.values
        rng = np.random.default_rng(0); self.tie = rng.random(len(self.R))        # fixed order-free tie-break for external scores
        self.random = np.random.default_rng(1).random(len(self.R))

    def spectra(self):
        return pd.read_parquet(self.root + '/spectra.parquet')

    def lists(self, half, bench_only=True):
        m = (self.L.half == half).values
        return np.flatnonzero(m & self.L.bench.values) if bench_only else np.flatnonzero(m)

    def pairs(self, half=None):
        """Unique (molecule, candidate) pairs a model has to score: mid, cand, key, smiles, pid (pool index, -1 = not a pool row)."""
        L = self.L if half is None else self.L[self.L.half == half]
        m = np.zeros(len(self.L), bool); m[L.list.values] = True
        d = pd.DataFrame(dict(mid=self.L.mid.values[self.lid], cand=self.cand))[m[self.lid]].drop_duplicates()
        c = self.C.set_index('cand'); d['key'] = c.key.values[d.cand.values]; d['smiles'] = c.smiles.values[d.cand.values]; d['pid'] = c.pid.values[d.cand.values]
        return d.reset_index(drop=True)

    def row_scores(self, scores):
        """Score file -> one value per table row (NaN = candidate not scored). The best score of a key is used for every row with that key."""
        s = pd.read_parquet(scores) if isinstance(scores, str) else scores
        s = s.groupby(['molecule_id', 'candidate_key'], sort=False).score.max()
        ix = pd.MultiIndex.from_arrays([self.L.mid.values[self.lid], self.C.key.values[self.cand]])
        return s.reindex(ix).values.astype(np.float64)

    def strata(self, lists):
        """{view/stratum: boolean mask over `lists`}."""
        L = self.L.iloc[lists]; mi = self.M.set_index('mid').loc[L.mid.values]; S = {}
        for v in VIEWS:
            b = (L.view == v).values
            S[f'{v} / all'] = b
            S[f'{v} / NP catalogue'] = b & mi.np_catalogue.values; S[f'{v} / not NP catalogue'] = b & ~mi.np_catalogue.values
            S[f'{v} / Enveda spectra'] = b & mi.enveda.values; S[f'{v} / Enveda NP panel'] = b & mi.panel.values
            S[f'{v} / nitrogen-free'] = b & mi.nfree.values; S[f'{v} / with nitrogen'] = b & ~mi.nfree.values
            S[f'{v} / lib_max < 0.7'] = b & (L.lib_max < 0.7).values; S[f'{v} / lib_max >= 0.7'] = b & (L.lib_max >= 0.7).values
            for lab, lo, hi in (('isomers <= 5', 0, 5), ('isomers 6-20', 6, 20), ('isomers 21-100', 21, 100), ('isomers > 100', 101, 10 ** 9)):
                S[f'{v} / {lab}'] = b & (L.n_isomers >= lo).values & (L.n_isomers <= hi).values
        return S

    def ranks(self, lists, score=None, name=None, avg_with=None, w=0.5, tie='order'):
        """Rank of the answer per list for a row-score vector (or a stored baseline by name)."""
        R = self.R
        if name is not None:
            if name == 'avg':
                return self.ranks(lists, score=R.ours.values, avg_with=R['pub'].values, w=0.5)
            score = self.random if name == 'random' else R[name].values
        out = np.zeros(len(lists), np.int32)
        for j, i in enumerate(lists):
            sl = slice(self.start[i], self.start[i] + self.n[i]); t = self.tie[sl] if tie == 'random' else None
            s = np.nan_to_num(np.asarray(score[sl], np.float64), nan=-np.inf, neginf=-1e300)
            if avg_with is None:
                out[j] = rank_answer(s, self.y[sl], self.keyc[sl], t)
            else:
                out[j] = rank_answer_avg(avg_with[sl], s, self.y[sl], self.keyc[sl], w, t)
        return out


def rr_of(rank):
    return np.where((rank > 0) & (rank <= K), 1.0 / np.maximum(rank, 1), 0.0)


def table(B, lists, rank, base_rank=None, min_n=20):
    """One row per (view, stratum): n, top-1 and MRR@25 with bootstrap SE; with `base_rank` also the paired differences."""
    rr = rr_of(rank); top = (rank == 1).astype(float); rows = []
    for name, m in B.strata(lists).items():
        if m.sum() < min_n:
            continue
        v, s = name.split(' / ')
        d = dict(view=v, stratum=s, n=int(m.sum()), top1=top[m].mean(), top1_se=boot_se(top[m]), mrr=rr[m].mean(), mrr_se=boot_se(rr[m]))
        if base_rank is not None:
            dr = rr[m] - rr_of(base_rank)[m]; dt = top[m] - (base_rank[m] == 1)
            d.update(d_top1=dt.mean(), d_top1_se=boot_se(dt), d_mrr=dr.mean(), d_mrr_se=boot_se(dr), wins=int((dr > 1e-9).sum()), losses=int((dr < -1e-9).sum()))
        rows.append(d)
    return pd.DataFrame(rows)


def coverage(B, half):
    """Per view: molecules of the half the engine was run for, lists with candidates, lists holding the answer (only those are stored)."""
    R = pd.read_parquet(B.root + '/runs.parquet'); L = B.L[B.L.bench]; v = L.drop_duplicates('set').set_index('set').view
    R = R[R.set.isin(v.index) & R.mid.isin(set(B.M.mid[B.M.half == half]))].copy(); R['view'] = R.set.map(v)
    g = R.groupby('view').agg(molecules_run=('mid', 'nunique'), with_candidates=('candidates', 'sum'), answer_listed=('listed', 'sum')).reindex(VIEWS)
    g['coverage'] = g.answer_listed / g.molecules_run; return g.reset_index()


# ---------------------------------------------------------------- the ranker refit with the score as a feature
def score_features(B, s):
    """Four per-row features of a score inside its list: raw, z-score, rank fraction (0 = best), gap to the list's best."""
    F = np.full((len(s), 4), np.nan, np.float32); F[:, 0] = s
    for i in range(len(B.L)):
        sl = slice(B.start[i], B.start[i] + B.n[i]); v = s[sl]; ok = np.isfinite(v)
        if ok.sum() < 1:
            continue
        mu, sd = v[ok].mean(), v[ok].std(); z = np.full(len(v), np.nan); z[ok] = (v[ok] - mu) / sd if sd > 1e-9 else 0.0
        r = np.full(len(v), np.nan); r[ok] = (-v[ok]).argsort().argsort() / max(ok.sum() - 1, 1)
        F[sl, 1] = z; F[sl, 2] = r; F[sl, 3] = v - v[ok].max()
    return F


def refit(B, half='dev', extra=None, seeds=(0,), threads=4, cache=None, log=print):
    """Scores of the production-recipe ranker for the rows of `half`: out of fold by scaffold on DEV; fitted on all of DEV for FINAL.
    `extra` (rows x k) is appended to the 165 inputs. Every list of the training molecules trains (class-1 lists too), as in production."""
    import lightgbm as lgb
    if cache and os.path.exists(cache):
        return np.load(cache)
    X = np.load(P.TABLE + '/X.npy', mmap_mode='r'); L = B.L; dev = (L.half == 'dev').values; out = np.zeros(len(B.R), np.float32)

    def mat(lm):
        rows = np.repeat(lm, B.n); x = np.asarray(X[rows], np.float32)
        return (x if extra is None else np.hstack([x, extra[rows]])), rows
    jobs = [(dev & (L.cvfold != f).values, dev & (L.cvfold == f).values) for f in range(5)] if half == 'dev' else [(dev, (L.half == 'final').values)]
    for tr, te in jobs:
        xt, rt = mat(tr); g = np.bincount(B.lid[rt], minlength=len(L)); g = g[g > 0]; xe, re = mat(te)
        for sd in seeds:
            ds = lgb.Dataset(xt, label=B.y[rt].astype(np.int32), group=g, free_raw_data=False)
            m = lgb.train(dict(PUB_PARAMS, num_threads=threads, seed=sd, bagging_seed=sd, feature_fraction_seed=sd), ds, num_boost_round=PUB_ROUNDS)
            out[re] += m.predict(xe, num_threads=threads).astype(np.float32) / len(seeds)
        log(f'refit: {int(tr.sum())} lists train, {int(te.sum())} lists scored, {xt.shape[1]} features')
        del xt, xe
    if cache:
        np.save(cache + '.tmp.npy', out); os.replace(cache + '.tmp.npy', cache)
    return out


# ---------------------------------------------------------------- the one function
def evaluate(scores_parquet, baseline='avg', half='dev', as_feature=False, seeds=(0,), threads=4, root=None, bench=None, log=print):
    """Table of the score file on the benchmark half. Plain: the score ranks the lists alone; paired difference to `baseline`.
    as_feature: two tables keyed 'alone' / 'avg': the ranker refitted with the score's features against the same refit without them
    (alone, and inside the rank average with the public ranker)."""
    B = bench or Bench(root); lists = B.lists(half); s = B.row_scores(scores_parquet)
    covered = float(np.mean([np.isfinite(s[B.start[i]:B.start[i] + B.n[i]]).mean() for i in lists]))
    log(f'{half}: {len(lists)} lists, mean share of candidates scored {covered:.4f}')
    if not as_feature:
        t = table(B, lists, B.ranks(lists, score=s, tie='random'), B.ranks(lists, name=baseline)); t.attrs['baseline'] = baseline; return t
    tag = '_'.join(map(str, seeds)); os.makedirs(B.root + '/refit', exist_ok=True)
    base = refit(B, half, None, seeds, threads, cache=f'{B.root}/refit/base_{half}_s{tag}.npy', log=log)
    new = refit(B, half, score_features(B, s), seeds, threads, log=log)
    pub = B.R['pub'].values; out = {}
    out['alone'] = table(B, lists, B.ranks(lists, score=new), B.ranks(lists, score=base))
    out['avg'] = table(B, lists, B.ranks(lists, score=new, avg_with=pub), B.ranks(lists, score=base, avg_with=pub))
    return out


def md(d, nd=3):
    """Markdown of a DataFrame (pandas.to_markdown needs `tabulate`, which is not installed)."""
    f = lambda x: f'{x:.{nd}f}' if isinstance(x, (float, np.floating)) else (f'{x:,}' if isinstance(x, (int, np.integer)) and not isinstance(x, bool) else str(x))
    return '\n'.join(['| ' + ' | '.join(map(str, d.columns)) + ' |', '|---' * len(d.columns) + '|'] + ['| ' + ' | '.join(f(x) for x in r) + ' |' for r in d.itertuples(index=False)])


def fmt(t, diff=True):
    """Markdown of a table."""
    has = diff and 'd_mrr' in t
    h = '| view | stratum | n | top-1 | MRR@25 |' + (' Δ top-1 | Δ MRR@25 (wins / losses) |' if has else ''); out = [h, '|---' * (7 if has else 5) + '|']
    for r in t.itertuples():
        x = f'| {r.view} | {r.stratum} | {r.n:,} | {r.top1:.3f} ± {r.top1_se:.3f} | {r.mrr:.3f} ± {r.mrr_se:.3f} |'
        if has:
            x += f' {r.d_top1:+.3f} ± {r.d_top1_se:.3f} | {r.d_mrr:+.3f} ± {r.d_mrr_se:.3f} ({r.wins} / {r.losses}) |'
        out.append(x)
    return '\n'.join(out)


def baselines(half='dev', bench=None, names=('random', 'fp_cos', 'fz', 'pub', 'ours', 'avg')):
    """Wide table: MRR@25 and top-1 (± bootstrap SE) of every stored baseline per (view, stratum); the fused v33b list where it was stored."""
    B = bench or Bench(); lists = B.lists(half); T = None
    for nm in names:
        t = table(B, lists, B.ranks(lists, name=nm))
        t[nm] = [f'{a:.3f} ± {b:.3f} / {c:.3f} ± {d:.3f}' for a, b, c, d in zip(t.top1, t.top1_se, t.mrr, t.mrr_se)]
        T = t[['view', 'stratum', 'n', nm]] if T is None else T.merge(t[['view', 'stratum', nm]], on=['view', 'stratum'])
    F = None
    if B.fused is not None:
        L = B.L.iloc[lists]; f = L.merge(B.fused, on=['mid', 'view']); rows = []; rk = {nm: B.ranks(f.list.values, name=nm) for nm in ('avg',)}
        for v in ('c2J', 'dropJ'):
            m = (f.view == v).values
            if m.sum():
                rows.append(dict(view=v, n=int(m.sum()), fused_top1=f.top[m].mean(), fused_top1_se=boot_se(f.top[m].astype(float)), fused_mrr=f.rr[m].mean(), fused_mrr_se=boot_se(f.rr[m]),
                                 avg_top1=(rk['avg'][m] == 1).mean(), avg_mrr=rr_of(rk['avg'][m]).mean()))
        F = pd.DataFrame(rows)
    return T, F


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--scores', default=''); ap.add_argument('--baseline', default='avg'); ap.add_argument('--half', default='dev')
    ap.add_argument('--baselines', action='store_true'); ap.add_argument('--as_feature', action='store_true'); ap.add_argument('--seeds', default='0')
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--out', default=''); a = ap.parse_args()
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    pd.set_option('display.width', 250); B = Bench(); txt = []
    if a.baselines:
        txt.append(f'## coverage ({a.half})\n\n' + md(coverage(B, a.half)))
        T, F = baselines(a.half, B)
        txt.append(f'## baselines ({a.half}): top-1 ± SE / MRR@25 ± SE\n\n' + md(T))
        if F is not None:
            txt.append('## fused v33b list where stored (same lists; `avg` = its base list)\n\n' + md(F))
    if a.scores:
        r = evaluate(a.scores, a.baseline, a.half, a.as_feature, tuple(int(x) for x in a.seeds.split(',')), a.threads, bench=B)
        if a.as_feature:
            for k, t in r.items():
                txt.append(f'## score as a ranker feature, refit {k} ({a.half}); difference to the refit without it\n\n' + fmt(t))
        else:
            txt.append(f'## score alone ({a.half}); difference to `{a.baseline}`\n\n' + fmt(r))
    print('\n\n'.join(txt))
    if a.out:
        open(a.out, 'w', encoding='utf-8').write('\n\n'.join(txt) + '\n')
