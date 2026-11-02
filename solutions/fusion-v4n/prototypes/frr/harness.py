"""Run the built FRR cell locally on fake lists (true structure + same-formula isomers, shuffled) from held-out sets.
usage: harness.py KERNEL.ipynb SETS_DIR N"""
import sys, json, shutil, hashlib, os
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa

nbp, sets, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
TRAIN = Path('C:/Users/omerm/Desktop/Kaggle/enveda-casmi-2026/data/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet')
W = Path('C:/Users/omerm/Desktop/Kaggle/casmi-local-work/frr_harness'); shutil.rmtree(W, ignore_errors=True)
(W / 'input/comp').mkdir(parents=True); (W / 'root').mkdir()
df = pd.read_parquet(sets / 'sets.parquet')
df = df[df.fold == 0]
qs = df.qkey.drop_duplicates().sample(n, random_state=3).tolist()
d = df[df.qkey.isin(qs)]
tab = pq.read_table(TRAIN, columns=['ingest_lib', 'inchikey14', 'adduct', 'ionization_mode', 'ms2_mzs', 'ms2_normalized_intensities'],
                    filters=[('ingest_lib', '=', 'enveda-180')])
tab = tab.filter(pc.is_in(tab['inchikey14'], value_set=pa.array(qs))).to_pandas()
tab = tab.groupby('inchikey14').head(6)
rng = np.random.default_rng(0)
rows, sub, ev = [], [], []
truth = {}
for i, q in enumerate(qs):
    mid = f'm_{i:04d}'
    g = d[d.qkey == q]
    smis = g.smiles.tolist(); rng.shuffle(smis)
    truth[mid] = g[g.label == 1].smiles.iloc[0]
    sub.append(dict(molecule_id=mid, smiles=';'.join(smis[:12])))
    for r in tab[tab.inchikey14 == q].itertuples(index=False):
        rows.append(dict(molecule_id=mid, adduct=r.adduct, ionization_mode=r.ionization_mode, ms2_mzs=r.ms2_mzs,
                         ms2_normalized_intensities=r.ms2_normalized_intensities))
    ev.append(dict(molecule_id=mid, library_max=0.5, processed=True))
pd.DataFrame(rows).to_parquet(W / 'input/comp/test.parquet')
# training keys: everything except the held-out queries' candidates (so nothing is frozen), plus one decoy per list frozen
frozen_keys = d[d.label == 0].groupby('qkey').cand.first().tolist()
pd.DataFrame(dict(inchikey14=frozen_keys + ['ZZZZZZZZZZZZZZ'])).to_parquet(W / 'input/comp/train.parquet')
pd.DataFrame(sub).to_csv(W / 'root/submission.csv', index=False)
pd.DataFrame(ev).to_parquet(W / 'root/frozen_candidate_evidence.parquet')
nb = json.loads(nbp.read_text(encoding='utf-8'))
src = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code' and 'FRR_ON = True' in ''.join(c['source'])][0]
src = src.replace("'/kaggle/input/**/test.parquet'", repr(str(W / 'input') + '/**/test.parquet'))
shutil.copy2(sys.argv[4] if len(sys.argv) > 4 else 'C:/Users/omerm/Desktop/Kaggle/casmi-local-work/fragrr2_80k/model.txt', W / 'input/frr_model_v2.txt')
src = src.replace("'/kaggle/input/**/frr_model_v2.txt'", repr(str(W / 'input') + '/**/frr_model_v2.txt'))
g = dict(ROOT=W / 'root', pd=pd, Path=Path, hashlib=hashlib, shutil=shutil, os=os, json=json, sys=sys, IS_RERUN=True,
         MERGE_REPORT=dict(moved_ids=[]))
exec(compile(src, 'frr_cell', 'exec'), g)
rep = g['FRR_REPORT']
print({k: v for k, v in rep.items() if k != 'fired'})
before = pd.read_csv(W / 'root/submission_pre_frr.csv'); after = pd.read_csv(W / 'root/submission.csv')


def mrr(f):
    out = []
    for mid, s in zip(f.molecule_id, f.smiles):
        l = s.split(';'); out.append(1 / (l.index(truth[mid]) + 1) if truth[mid] in l else 0)
    return round(float(np.mean(out)), 4), round(float(np.mean([x == 1 for x in out])), 4)


print('before (shuffled) mrr/top1', mrr(before), 'after', mrr(after))
