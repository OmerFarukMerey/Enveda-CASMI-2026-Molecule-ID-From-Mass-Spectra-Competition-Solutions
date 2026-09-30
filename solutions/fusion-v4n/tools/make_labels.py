"""Labels for the VISIBLE test.parquet (a placeholder the organisers drew from train.parquet), used ONLY as a
smoke/sanity check of the full pipeline and of tools/refuse.py wiring.

WARNING: do not tune on this. The hidden test is different molecules and these are all in the spectral library,
so scores here are wildly optimistic (see docs/competition/rules-and-pitfalls.md).

train.parquet has no spectrum_id / molecule_id, so rows are matched on (precursor_mz, adduct, peak-list hash).
usage: python tools/make_labels.py  -> <repo>/data/visible_labels.csv (molecule_id, smiles, inchikey14, ingest_lib)
"""
import hashlib
import os
import sys

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import COMP, DATA


def peak_hash(mzs):
    return hashlib.md5(np.round(np.asarray(mzs, dtype=np.float64), 4).tobytes()).hexdigest()


te = pq.read_table(os.path.join(COMP, 'test.parquet'),
                   columns=['molecule_id', 'precursor_mz', 'adduct', 'ms2_mzs']).to_pandas()
cols = ['precursor_mz', 'adduct', 'ms2_mzs', 'normalized_smiles', 'inchikey14', 'ingest_lib']
tr = ds.dataset(os.path.join(COMP, 'train.parquet')).to_table(
    columns=cols, filter=ds.field('precursor_mz').isin(te.precursor_mz.unique().tolist())).to_pandas()
print(f'{len(tr)} train rows share an exact precursor_mz with the {len(te)} visible test spectra')
for d in (te, tr):
    d['_k'] = [f'{p!r}|{a}|{peak_hash(m)}' for p, a, m in zip(d.precursor_mz, d.adduct, d.ms2_mzs)]
m = te[['molecule_id', '_k']].merge(tr.drop(columns='ms2_mzs').drop_duplicates('_k'), on='_k', how='left')
print(f'spectrum match: {m.normalized_smiles.notna().mean():.1%} of test spectra')
g = m.dropna(subset=['normalized_smiles']).groupby('molecule_id')
lab = g.agg(smiles=('normalized_smiles', 'first'), inchikey14=('inchikey14', 'first'),
            n_ik=('inchikey14', 'nunique'), ingest_lib=('ingest_lib', 'first')).reset_index()
print(f'{len(lab)} / {te.molecule_id.nunique()} molecules labelled; {int((lab.n_ik > 1).sum())} with conflicting keys')
os.makedirs(DATA, exist_ok=True)
out = os.path.join(DATA, 'visible_labels.csv')
lab.drop(columns='n_ik').to_csv(out, index=False)
print('wrote', out)
