# Lab: are the validation truths that our lists miss present in the public PubChem mass index at all?
import glob, sys, json, importlib.util, time, subprocess
from pathlib import Path
import numpy as np, pandas as pd
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rdkit'])
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog('rdApp.*')
root = Path(sorted(glob.glob('/kaggle/input/**/INDEX.json', recursive=True))[0]).parent
print('root', root, sorted(p.name for p in root.iterdir())[:40])
print(open(root / 'INDEX.json').read()[:3000])
sp = importlib.util.spec_from_file_location('fingerprint_index', str(root / 'fingerprint_index.py'))
mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod); sys.modules['fingerprint_index'] = mod
cv = mod.CandidateView(str(root))
def k14(s):
    try:
        m = Chem.MolFromSmiles(s); return Chem.MolToInchiKey(m)[:14] if m is not None else None
    except Exception: return None
out = []
for tag, sub in (('v3', 'submission3.csv'), ('v2', 'submission.csv')):
    tr = pd.read_parquet(glob.glob(f'/kaggle/input/**/casmi-val-{tag}/**/val_truth.parquet', recursive=True)[0])
    S = pd.read_csv(glob.glob(f'/kaggle/input/**/casmi-lab/**/{sub}', recursive=True)[0]).fillna('').set_index('molecule_id').smiles
    for r in tr.itertuples():
        t0 = time.time()
        m = Chem.MolFromSmiles(r.smiles); tk = Chem.MolToInchiKey(m)[:14]; mass = ExactMolWt(m); form = CalcMolFormula(m)
        lst = [k14(x) for x in S.get(r.molecule_id, '').split(';') if x][:25]
        rank = lst.index(tk) + 1 if tk in lst else 0
        w = cv.window(mass, ppm=10, fields=('lsid',), smiles=True)
        smi = [str(x) for x in w['smiles']]; origin = np.asarray(w['origin'])
        # cheap prefilter: same formula via atom count string is costly; just key everything
        hit = -1; nform = 0
        for i, s in enumerate(smi):
            mm = Chem.MolFromSmiles(s)
            if mm is None: continue
            if abs(ExactMolWt(mm) - mass) > 0.0008: continue
            nform += 1
            if Chem.MolToInchiKey(mm)[:14] == tk: hit = i; break
        out.append(dict(set=tag, molecule_id=r.molecule_id, fold=getattr(r, 'fold', ''), rank=rank, window=len(smi), in_window=hit >= 0,
                        origin=int(origin[hit]) if hit >= 0 else -1, sec=round(time.time() - t0, 2)))
        if len(out) % 25 == 0: print(len(out), out[-1], flush=True)
d = pd.DataFrame(out); d.to_csv('/kaggle/working/pcwin.csv', index=False)
d['absent'] = d['rank'] == 0
print(d.groupby(['set', 'fold', 'absent']).agg(n=('in_window', 'size'), in_window=('in_window', 'sum'), med_window=('window', 'median')))
print(d.groupby(['absent', 'origin']).size())
