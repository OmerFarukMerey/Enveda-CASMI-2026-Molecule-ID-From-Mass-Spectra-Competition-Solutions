"""Why did engine 2 skip a molecule? For each molecule of a synthetic test: engine 2's own neutral-mass target (pv.neutral_mass,
median over spectra) against the answer's monoisotopic mass, and the number of pool structures (v2 pool masses) within 10 / 30 ppm.
usage: python -I scratch/land1123/vh2_preflight_check.py <comp_dir_with_test.parquet> <answers.csv>"""
import sys, os
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, 'frozen', 'eng'))
import pv
from rdkit import Chem
from rdkit.Chem import Descriptors
te = pd.read_parquet(os.path.join(sys.argv[1], 'test.parquet')); ans = pd.read_csv(sys.argv[2], dtype=str).set_index('molecule_id')
pm = np.sort(pd.read_parquet(os.path.join(REPO, 'data', 'input', 'datasets', 'dmitriigluzdov', 'casmi26-attributed-v4g-reproduction-assets', 'casmi26-v2-pool', 'pool_meta.parquet'), columns=['mass']).mass.values)
rows = []
for mid, sub in te.groupby('molecule_id'):
    nms = pv.neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.astype(str).values)
    fin = nms[np.isfinite(nms)]
    t = float(np.median(fin)) if len(fin) else np.nan
    m = Chem.MolFromSmiles(ans.smiles[mid]); true = Descriptors.ExactMolWt(m) if m else np.nan
    n10 = int(np.searchsorted(pm, t * (1 + 1e-5)) - np.searchsorted(pm, t * (1 - 1e-5))) if np.isfinite(t) else 0
    n30 = int(np.searchsorted(pm, t * (1 + 3e-5)) - np.searchsorted(pm, t * (1 - 3e-5))) if np.isfinite(t) else 0
    rows.append(dict(mid=mid, group=ans.group[mid], n_spec=len(sub), finite=len(fin), target=t, true=true, ppm=(t - true) / true * 1e6 if np.isfinite(t) else np.nan,
                     pool10=n10, pool30=n30, adducts=','.join(sorted(set(sub.adduct)))))
R = pd.DataFrame(rows)
bad = R[(R.finite == 0) | (R.pool30 == 0)]
print('molecules', len(R), '| no finite neutral mass', int((R.finite == 0).sum()), '| empty 30 ppm pool window', int((R.pool30 == 0).sum()))
print(bad.to_string(index=False))
print('abs ppm error by group (median / 95th / max):')
print(R.groupby('group').ppm.agg(lambda x: f'{x.abs().median():.1f} / {x.abs().quantile(.95):.1f} / {x.abs().max():.1f}').to_string())
print(R.sort_values('ppm', key=abs, ascending=False).head(8).to_string(index=False))
