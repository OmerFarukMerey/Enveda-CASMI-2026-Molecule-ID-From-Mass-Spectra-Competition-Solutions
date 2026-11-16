"""End-to-end GEN simulation on a val replay: lists + keep-truth candidate dump (deploy-like) -> MRR before/after."""
import sys, glob, numpy as np, pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
SUB, VD, EV, PAT = sys.argv[1:5]
V = pd.read_parquet(f'{VD}/val_truth.parquet'); tr = dict(zip(V.molecule_id, V.inchikey14)); fo = dict(zip(V.molecule_id, V.fold))
S = pd.read_csv(SUB, dtype=str, keep_default_na=False)
lib = pd.read_parquet(EV).groupby('molecule_id').library_max.max().to_dict()
D = pd.concat([pd.read_parquet(f) for f in glob.glob(PAT) if 'keep1' in f])
D = D.drop_duplicates(['molecule_id', 'key'])
k14 = lambda s: (lambda m: Chem.MolToInchiKey(m)[:14] if m else None)(Chem.MolFromSmiles(s))
L = {m: [k14(x) for x in s.split(';')] for m, s in zip(S.molecule_id, S.smiles) if m in set(D.molecule_id) or True}
def rr_of(keys, t):
    r = next((i + 1 for i, k in enumerate(keys[:25]) if k == t), None); return 1 / r if r else 0.0
G = {}
for m, d in D.groupby('molecule_id'):
    rr = d.rr.values.astype(float); rr = np.nan_to_num(rr, nan=np.nanmin(rr) if np.isfinite(rr).any() else 0)
    z = (rr - rr.mean()) / (rr.std() if rr.std() > 1e-9 else 1)
    G[m] = (d.key.values, z, d.nsrc.values.astype(float))
def run(W=4.0, START=5, M=10, PROM=99, TAU=0.9):
    out = []
    for m, keys in L.items():
        t = tr[m]; base = rr_of(keys, t); new = base
        if m in G and lib.get(m, 0) < TAU and len(keys) >= START - 1:
            ks, z, ns = G[m]; f = z + W * ns; o = np.argsort(-f, kind='stable')[:M]
            add = [ks[i] for i in o if ks[i] not in keys]
            if add and PROM <= ns[o[0]] and ks[o[0]] == add[0]:
                l = keys[:1] + add[:1] + keys[1:START - 1] + add[1:] + keys[START - 1:]
            else:
                l = keys[:START - 1] + add + keys[START - 1:]
            new = rr_of(l, t)
        out.append((m, fo[m], t in keys[:25], base, new))
    return pd.DataFrame(out, columns=['m', 'fold', 'inlist', 'base', 'new'])
print('molecules with dump', len(G), 'of', len(L), '; truth generated', D.groupby('molecule_id').y.max().mean().round(3))
cfgs = [dict(W=0, PROM=99), dict(W=4, PROM=99), dict(W=4, PROM=3), dict(W=4, PROM=4), dict(W=4, PROM=3, START=8), dict(W=4, PROM=3, M=5),
        dict(W=4, PROM=3, START=11, M=5), dict(W=4, PROM=3, M=5, START=15), dict(W=4, PROM=2)]
for c in cfgs:
    R = run(**c); R = R[R.m.isin(G)] if 'ONLYG' in sys.argv else R
    g = R.groupby('fold').apply(lambda x: pd.Series(dict(n=len(x), base=x.base.mean(), new=x.new.mean(), d=x.new.mean() - x.base.mean(),
        d_in=(x.new - x.base)[x.inlist].sum() / len(x), d_out=(x.new - x.base)[~x.inlist].sum() / len(x)))).round(4)
    print(c); print(g.to_string())
