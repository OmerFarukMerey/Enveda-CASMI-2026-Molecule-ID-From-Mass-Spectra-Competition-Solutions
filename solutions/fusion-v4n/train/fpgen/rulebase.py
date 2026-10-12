"""Rule-based baseline of fpgen: the production analog generator (nb/embed/gen_analog.py, gen3) run on the same evaluation molecules with the same hit test, as the number the learned generator has to beat.

Configuration = the production one (one rule step or a positional move from the three best pool parents per formula, caps of gen3), with
the answer's own pool rows banned as parents, as `gen3` replays it. The answer's scaffold mates stay in the pool, which is generous to
this baseline on the scaffold hold-out; it is the channel that exists today, so it is the baseline.
out: $FPGEN_WORK/eval/rule_<forms>/{rows.parquet, summary.json}.   usage: python rulebase.py [--forms pipe|true] [--workers 4]
"""
import argparse, json, os, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpaths as P, sets as S

_W = {}


def _init():
    P.low_priority(); P.watch_parent()
    sys.path.insert(0, P.CODE)
    import gen_analog as GA
    from casmi import chem
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    PM = pd.read_parquet(P.WORK + '/pool_meta.parquet')
    _W['GA'] = GA; _W['chem'] = chem; _W['Chem'] = Chem; _W['bits'] = np.load(P.WORK + '/fp_bits.npy')
    _W['pool'] = GA.PoolIndex(PM.smiles.values, PM.key.values, PM.formula.values, np.load(P.WORK + '/pool_fp.npy', mmap_mode='r'))
    _W['pkey'] = PM.key.values; _W['pscaf'] = pd.read_parquet(P.CHEM + '/pool.parquet').scaf.values
    z = {}
    for f in ('z_dev.npz', 'z_fair.npz'):
        d = np.load(P.G2 + '/' + f); z.update(zip([str(x) for x in d['mid']], d['z']))
    _W['Z'] = z


def one(task):
    if not _W:
        _init()
    mid, forms, key, taut, cand_keys, scaf = task
    GA = _W['GA']; chem = _W['chem']; bits = _W['bits']; z = _W['Z'][mid]; pool = _W['pool']
    banned = set(np.flatnonzero(_W['pkey'] == key).tolist())
    if scaf:
        banned |= set(np.flatnonzero(_W['pscaf'] == scaf).tolist())   # --ban_scaffold: the baseline gets the same scaffold blindness as the learned generator
    kc = {}; fc = {}

    def key_fn(s):
        if s not in kc:
            kc[s] = chem.score_key(s)
        return kc[s]

    def fp_fn(s):
        if s not in fc:
            f = chem.raw_fingerprint(s); fc[s] = None if f is None else f[bits]
        return fc[s]
    t0 = time.time()
    per = [GA.enumerate_formula(pool, z, f, GA.M_PARENTS, banned, t0 + GA.T_ENUM) for f in forms]
    prods = GA.interleave(per)
    rows = GA.finish(prods, pool, z, set(cand_keys), fp_fn, key_fn, removed_keys={key})
    hit_pos = [i + 1 for i, p in enumerate(prods) if p['smiles'] in taut]
    row_pos = [i + 1 for i, r in enumerate(rows) if r['smiles'] in taut or r['key'] == key]
    return dict(mid=mid, n_products=len(prods), n_rows=len(rows), generated=bool(hit_pos), product_pos=hit_pos[0] if hit_pos else 0,
                rank_rows=row_pos[0] if row_pos else 0, wall_s=round(time.time() - t0, 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--forms', default='pipe', choices=['pipe', 'true']); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--sets', default='N,F,SH'); ap.add_argument('--ban_scaffold', action='store_true'); a = ap.parse_args()
    import multiprocessing as mp
    P.mk(); P.low_priority()
    out = P.EVAL + '/rule_' + a.forms + ('_noscaf' if a.ban_scaffold else ''); os.makedirs(out, exist_ok=True)
    M, _ = S.load(); M = M[M.eset.isin(a.sets.split(','))].reset_index(drop=True)
    import pickle
    L = pickle.load(open(P.G3 + '/lists.pkl', 'rb')); cands = {}
    for r in L:
        if (r['set'], r['view']) in (('dev', 'N'), ('F', 'F'), ('dev', 'c2J')):
            cands.setdefault(r['mid'], set(r['cand_keys']) | set(r['final']))
    tasks = [(r.mid, [f for f in (r.forms3.split('|') if a.forms == 'pipe' and r.forms3 else [r.formula]) if f],
              r.key, set(r.tsmi.split('|')), cands.get(r.mid, set()) - {r.key}, r.scaf if a.ban_scaffold else '') for r in M.itertuples()]
    t0 = time.time(); res = []
    with mp.Pool(a.workers, initializer=_init) as pool:
        for i, x in enumerate(pool.imap_unordered(one, tasks, chunksize=2)):
            res.append(x)
            if i % 100 == 0:
                print(i, len(tasks), round(time.time() - t0), flush=True)
    R = M[['mid', 'eset', 'key']].merge(pd.DataFrame(res), on='mid', how='left'); R.to_parquet(out + '/rows.parquet', index=False)
    summ = {es: dict(n=int(len(g)), generated=float(g.generated.mean()), products_mean=float(g.n_products.mean()), rows_mean=float(g.n_rows.mean()),
                     **{'rows_top%d' % k: float(g.rank_rows.between(1, k).mean()) for k in (1, 5, 25)}) for es, g in R.groupby('eset')}
    json.dump(dict(forms=a.forms, ban_scaffold=a.ban_scaffold, summary=summ, wall_s=round(time.time() - t0), t=time.strftime('%Y-%m-%d %H:%M')), open(out + '/summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == '__main__':
    main()
