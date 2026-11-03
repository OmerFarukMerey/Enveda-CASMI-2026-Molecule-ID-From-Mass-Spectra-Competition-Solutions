"""Run the filled overlay cell locally on a mock: N training molecules as the 'test', random 25-candidate lists.
usage: test_cell.py FILLED_CELL MOCK_DIR TRAIN_PARQUET GATE N"""
import hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

cell, mock, train, gate, n = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
inp, root = mock / 'input', mock / 'root'
for d in (inp / 'comp', inp / 'gate', root):
    d.mkdir(parents=True, exist_ok=True)
if not (inp / 'comp' / 'train.parquet').exists():
    os.link(train, inp / 'comp' / 'train.parquet')
shutil.copy2(gate, inp / 'gate' / 'ae_gate_v1.txt')
t = pq.ParquetFile(train).read_row_group(3, columns=['ingest_lib', 'inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz',
                                                       'ms2_mzs', 'ms2_normalized_intensities']).to_pandas()
t = t[t.ingest_lib == 'enveda-180']
rng = np.random.default_rng(3)
keys = rng.choice(t.inchikey14.unique(), n, replace=False)
te = t[t.inchikey14.isin(keys)].copy()
mid = {k: f'm_{i:06d}' for i, k in enumerate(keys)}
te['molecule_id'] = te.inchikey14.map(mid)
te[['molecule_id', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']].to_parquet(inp / 'comp' / 'test.parquet', index=False)
pool = t.normalized_smiles.drop_duplicates().to_numpy()
truth = te.groupby('molecule_id').normalized_smiles.first()
lists = []
for m in sorted(mid.values()):
    rest = [s for s in rng.choice(pool, 30, replace=False) if s != truth[m]][:24]
    lists.append(';'.join([truth[m]] + rest))
sub = pd.DataFrame(dict(molecule_id=sorted(mid.values()), smiles=lists))
sub.to_csv(root / 'submission.csv', index=False)
pd.DataFrame(dict(molecule_id=sub.molecule_id, library_max=rng.uniform(0.3, 1.0, len(sub)), processed=True)).to_parquet(
    root / 'frozen_candidate_evidence.parquet', index=False)
g = dict(ROOT=root, SAMPLE_IDS=sub.molecule_id.tolist(), NB_T0=time.time(), IS_RERUN=False, Path=Path, pd=pd, shutil=shutil,
         os=os, json=json, hashlib=hashlib)
src = cell.read_text(encoding='utf-8').replace("AE_INPUT = '/kaggle/input'", f"AE_INPUT = {str(inp).replace(chr(92), '/')!r}")
t0 = time.time()
exec(compile(src, 'ae_cell', 'exec'), g)
rep = json.loads((root / 'ae_overlay_report.json').read_text())
print('seconds', round(time.time() - t0), '| report', {k: v for k, v in rep.items() if k not in ('fired_list', 'commit_dry_run')})
print('dry run', {k: v for k, v in rep['commit_dry_run'].items() if k != 'fired_list'})
for f in rep['commit_dry_run']['fired_list'][:8]:
    print('  ', f)
after = pd.read_csv(root / 'submission.csv', dtype=str, keep_default_na=False)
before = pd.read_csv(root / 'submission_pre_ae.csv', dtype=str, keep_default_na=False)
ch = (after.smiles != before.smiles)
print('lists changed', int(ch.sum()), '| rank 1 kept everywhere', all(a.split(';')[0] == b.split(';')[0] for a, b in zip(after.smiles, before.smiles)),
      '| lengths ok', after.smiles.str.split(';').map(len).between(1, 25).all())
