"""Offline check of the analog-edit idea on held-out training structures.
usage: eval_ae.py LIB_DIR N_QUERIES SEED OUT_PARQUET [SOURCE]
For each sampled structure (spectra from SOURCE, default enveda-180), its own spectra are the queries and every other
structure is the library. Records the best analog, whether the truth is one of its edits, and the truth's rank by site
score."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ae_core as A

lib, n_q, seed, out = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
source = sys.argv[5] if len(sys.argv) > 5 else 'enveda-180'
L = A.Library(lib / 'lib_peaks.npz', pd.read_parquet(lib / 'lib_struct.parquet'))
src = np.load(lib / 'lib_peaks.npz')['src']
code = json.loads((lib / 'lib_sources.json').read_text())[source]
pool = np.unique(L.sidx[src == code])
rng = np.random.default_rng(seed)
picks = rng.choice(pool, min(n_q, len(pool)), replace=False)
rows_of = pd.Series(np.arange(len(L.sidx))).groupby(L.sidx).apply(np.array)
t0 = time.time(); recs = []
for qi, s in enumerate(picks):
    best = (0.0, -1, None, None, 0.0, -1)
    for r in rows_of[s]:
        if src[r] != code:
            continue
        b = A.analog_search(L.mz[r], L.it[r], L.n[r], L.prec[r], int(L.grp[r]), L, exclude=(s,))
        if b[0] > best[0]:
            best = b + (int(r),)
    rec = dict(sidx=int(s), key=L.keys[s], score=best[0], kind=best[2], direction=best[3], delta=best[4])
    if best[1] >= 0:
        a = int(L.sidx[best[1]]); rec.update(analog_key=L.keys[a])
        try:
            prods = A.edits(L.smiles[a], best[2], best[3])
        except Exception as e:  # noqa: BLE001
            prods = {}; rec['err'] = repr(e)[:80]
        rec.update(n_prod=len(prods), truth_in=L.keys[s] in prods)
        if prods and rec['truth_in']:
            q, ar = best[5], best[1]
            ranked = A.rank_products(L.smiles[a], prods, L.mz[ar][:L.n[ar]], L.it[ar][:L.n[ar]], L.mz[q][:L.n[q]],
                                     L.it[q][:L.n[q]], best[4], int(L.grp[q]))
            ks = [k for _, k, _ in ranked]
            sc = {k: v for v, k, _ in ranked}
            rec.update(truth_rank=ks.index(L.keys[s]) + 1, truth_ties=sum(1 for v, _, _ in ranked if v == sc[L.keys[s]]))
    recs.append(rec)
    if (qi + 1) % 100 == 0:
        print(f'{qi + 1} queries, {time.time() - t0:.0f}s', flush=True)
df = pd.DataFrame(recs); df.to_parquet(out, index=False)
print(f'done {len(df)} in {time.time() - t0:.0f}s')
for tau in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
    f = df[df.score >= tau]
    if not len(f):
        continue
    tin = f.truth_in.fillna(False).astype(bool)
    top1 = (f.truth_rank == 1).sum(); top3 = (f.truth_rank <= 3).sum()
    print(f'tau {tau}: fire {len(f) / len(df):.3f} ({len(f)}) | truth in edits {tin.mean():.3f} | top-1 {top1 / len(f):.3f} '
          f'| top-3 {top3 / len(f):.3f} | median products {f.n_prod.median():.0f}')
