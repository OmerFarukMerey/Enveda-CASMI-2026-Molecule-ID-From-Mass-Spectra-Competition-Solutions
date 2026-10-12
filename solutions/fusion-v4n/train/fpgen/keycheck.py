"""Check of fpgen's cheap hit test: recompute the competition metric key (RDKit 2026.03.3 tautomer-canonical InChIKey first block) for the top ranked rows of an evaluation run and compare with the tautomer-set test.

The evaluation counts a sample as the answer when its stereo-free canonical SMILES is in the answer's enumerated tautomer set (cheap).
Here the first --top ranked structures of each molecule get their real metric key: a disagreement in either direction is printed.
usage: python keycheck.py --dir $FPGEN_WORK/eval/<name> [--top 25] [--order fz] [--workers 4]
"""
import argparse, json, os, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpaths as P, sets as S

_W = {}


def _init():
    P.low_priority(); P.watch_parent(); sys.path.insert(0, P.CODE)
    from casmi import chem
    _W['chem'] = chem


def _keys(smis):
    if not _W:
        _init()
    return [_W['chem'].score_key(s) for s in smis]


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--order', default='fz'); ap.add_argument('--workers', type=int, default=4); a = ap.parse_args()
    P.low_priority(); t0 = time.time()
    M, _ = S.load(); M = M.set_index('mid')
    D = pd.read_parquet(a.dir + '/samples.parquet')
    if not len(D):
        print('no samples'); return
    zcol = 'fz' if a.order == 'fz' else 'll'
    if zcol not in D.columns:                                   # samples.parquet holds n and ll; fz is recomputed in rows.parquet, so rank by ll there
        zcol = 'll'
    D = D.sort_values(['mid', zcol], ascending=[True, False]).groupby('mid', sort=False).head(a.top).reset_index(drop=True)
    with mp.Pool(a.workers, initializer=_init) as pool:
        keys = sum(pool.map(_keys, np.array_split(list(D.smiles), max(1, len(D) // 500 + 1))), [])
    D['key_metric'] = keys
    D['cheap'] = [s in set(M.loc[m].tsmi.split('|')) for m, s in zip(D.mid, D.smiles)]
    D['exact'] = [k == M.loc[m].key for m, k in zip(D.mid, D.key_metric)]
    out = dict(dir=a.dir, order=zcol, top=a.top, rows=len(D), molecules=int(D.mid.nunique()),
               cheap_hits=int(D.cheap.sum()), metric_key_hits=int(D.exact.sum()),
               cheap_only=int((D.cheap & ~D.exact).sum()), metric_only=int((~D.cheap & D.exact).sum()), wall_s=round(time.time() - t0))
    json.dump(out, open(a.dir + '/keycheck.json', 'w'), indent=1); print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
