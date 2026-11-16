"""Test romanrozen's timsTOF mass prior on a val replay's final lists."""
import sys, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
OUT, VD = sys.argv[1], sys.argv[2]
ADD = {'[M+H]+': 1.007276, '[M-H]-': -1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823,
       '[M+K]+': 38.963158, '[M+CH2O2-H]-': 44.998201, '[M+Cl]-': 34.969402}
T = pd.read_parquet(f'{VD}/val_test.parquet', columns=['molecule_id', 'precursor_mz', 'adduct'])
T['nm'] = pd.to_numeric(T.precursor_mz, errors='coerce') - T.adduct.map(ADD)
NM = T[np.isfinite(T.nm) & (T.nm > 50)].groupby('molecule_id').nm.median().to_dict()
V = pd.read_parquet(f'{VD}/val_truth.parquet'); tr = dict(zip(V.molecule_id, V.inchikey14)); fo = dict(zip(V.molecule_id, V.fold))
S = pd.read_csv(f'{OUT}/submission.csv', dtype=str, keep_default_na=False)
E = pd.read_parquet(f'{OUT}/frozen_candidate_evidence.parquet'); LIB = E.groupby('molecule_id').library_max.max().to_dict()
_c = {}
def info(s):
    if s not in _c:
        m = Chem.MolFromSmiles(s); _c[s] = (Chem.MolToInchiKey(m)[:14], Descriptors.ExactMolWt(m)) if m else (None, None)
    return _c[s]
lists = {m: s.split(';')[:25] for m, s in zip(S.molecule_id, S.smiles)}
# truth ppm distribution
tp = [];
for m, c in lists.items():
    for s in c:
        k, w = info(s)
        if k == tr[m] and m in NM: tp.append((fo[m], (w - NM[m]) / NM[m] * 1e6)); break
tp = pd.DataFrame(tp, columns=['fold', 'ppm']); print('truth ppm', tp.groupby('fold').ppm.describe(percentiles=[.05, .5, .95]).round(2))
allp = [(fo[m], (info(s)[1] - NM[m]) / NM[m] * 1e6, info(s)[0] == tr[m]) for m, c in lists.items() if m in NM for s in c if info(s)[1]]
A = pd.DataFrame(allp, columns=['fold', 'ppm', 'y'])
A['off'] = (A.ppm + 1.5).abs() > 3; print('share >3ppm off centre: truth / others'); print(A.groupby(['fold', 'y']).off.mean().round(3))
def mrr(L):
    r = {}
    for m, c in L.items():
        ks = [info(s)[0] for s in c]; rk = next((i + 1 for i, k in enumerate(ks) if k == tr[m]), None); r.setdefault(fo[m], []).append(1 / rk if rk else 0)
    return {f: round(np.mean(v), 4) for f, v in r.items()}
def apply(L, C=-1.5, FREE=3., FULL=8., W=0.15, LEAD=5., LIBT=0.9, k=3., mid=True, top=True):
    out = {}
    for m, c in L.items():
        nm = NM.get(m); c = list(c)
        if nm is None: out[m] = c; continue
        d = [abs((info(s)[1] - nm) / nm * 1e6 - C) if info(s)[1] else 0 for s in c]
        if mid and len(c) > 2:
            pen = [-W * min(1., max(0., (x - FREE) / (FULL - FREE))) for x in d]
            idx = sorted(range(1, len(c)), key=lambda i: -(1 / (k + i + 1) + pen[i]))
            c2 = [c[0]] + [c[i] for i in idx]; d = [d[0]] + [d[i] for i in idx]; c = c2
        if top and LIB.get(m, 0) < LIBT and d[0] > LEAD:
            j = next((i for i in (1, 2) if i < len(c) and d[i] <= FREE), None)
            if j is not None: c = [c[j], c[0]] + [x for i, x in enumerate(c) if i not in (0, j)]
        out[m] = c
    return out
print('base     ', mrr(lists))
print('mass both', mrr(apply(lists)))
print('mass mid ', mrr(apply(lists, top=False)))
print('mass top ', mrr(apply(lists, mid=False)))
