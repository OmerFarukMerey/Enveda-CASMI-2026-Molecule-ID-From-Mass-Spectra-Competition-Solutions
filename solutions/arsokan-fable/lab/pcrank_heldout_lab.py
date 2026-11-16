# Lab: rank of the true structure when the public FPNet scores the whole PubChem mass window (not a submission).
import glob, sys, json, importlib.util, time, subprocess, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rdkit'])
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog('rdApp.*')
F = lambda pat: sorted(glob.glob('/kaggle/input/**/' + pat, recursive=True))
root = Path(F('INDEX.json')[0]).parent
sp = importlib.util.spec_from_file_location('fingerprint_index', str(root / 'fingerprint_index.py'))
mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod); sys.modules['fingerprint_index'] = mod
cv = mod.CandidateView(str(root)); fi = mod.FingerprintIndex(str(root), collection='pubchem', cache_parts=2)
NB = int(fi.spec['bits'])
code = [str(Path(p).parent.parent) for p in F('code/casmi/__init__.py')]; print('code dirs', code)
for c in code:
    if Path(c, 'casmi', 'fpnet.py').is_file() and Path(c, 'casmi', 'chem.py').is_file(): sys.path.insert(0, c); break
else:
    for c in code: sys.path.insert(0, c)
from casmi import chem, fpnet
from casmi.spectra import merge_spectra
M0 = fpnet.load_model(F('fpnet_fold0.pt')[0], 'cuda'); assert M0.nbits == NB
BANKS = {'fold0': [M0]}
sp_ = pd.read_parquet(F('split.parquet')[0]); hold = set(sp_[sp_.fold == 'hold'].inchikey14)
import pyarrow.dataset as pads, pyarrow as pa
ds = pads.dataset(F('train.parquet')[0])
meta = ds.to_table(columns=['ingest_lib', 'inchikey14'], filter=pads.field('ingest_lib') == 'enveda-180').to_pandas()
bundle = Path(F('casmi_spectrum_model_v1/inference.py')[0]).parent; sys.path.insert(0, str(bundle))
from inference import Scorer
import features as sfeat
SC = Scorer(str(bundle), checkpoint='best_validation_model.pt', device='cuda')
smol = pd.read_parquet(bundle / 'molecules.parquet'); sval = set(smol[smol.split == 'validation'].key)
keys = sorted(set(meta.inchikey14) & hold & sval); print('doubly held-out enveda keys', len(keys))
rng = np.random.default_rng(5); keys = list(rng.choice(keys, 1300, replace=False))
from multiprocessing import Pool
def _fp(smi_):
    try: return sfeat.structure(smi_, 2048)[-1]
    except Exception: return None
POOL = Pool(4); print('hold enveda keys sampled', len(keys))
TR = ds.to_table(columns=['inchikey14', 'normalized_smiles', 'ionization_mode', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities', 'collision_energy_orig', 'collision_energy_orig_units', 'collision_energy_ev'],
                 filter=(pads.field('ingest_lib') == 'enveda-180') & pads.field('inchikey14').isin(pa.array(keys))).to_pandas()
print('rows', len(TR), TR.collision_energy_orig.head(3).tolist(), TR.collision_energy_ev.head(3).tolist(), TR.collision_energy_orig_units.value_counts().to_dict())
dtype = fpnet.autocast_dtype('cuda')
@torch.no_grad()
def logits(models, items):
    inp = fpnet.collate(items, 'cuda')
    with torch.autocast('cuda', dtype=dtype):
        return torch.stack([m(*inp)[0].float() for m in models]).mean(0).mean(0)
def k14(s):
    try:
        m = Chem.MolFromSmiles(s); return Chem.MolToInchiKey(m)[:14] if m is not None else None
    except Exception: return None
out = []; CAND = []; M0s = [M0]
G = dict(tuple(TR.groupby('inchikey14')))
if True:
    for tk in keys:
        g = G[tk]; t0 = time.time(); rec = dict(key=tk, set='hold', fold='enveda', rank=0)
        try:
            m = Chem.MolFromSmiles(g.normalized_smiles.iloc[0]); mass = ExactMolWt(m); rec['nspec'] = len(g)
            w = cv.window(mass, ppm=10, fields=('lsid', 'lpmid'), smiles=True)
            smi = [str(x) for x in w['smiles']]; origin = np.asarray(w['origin']); src = np.asarray(w['source_row'])
            pos = np.flatnonzero(origin == 0); rec.update(window=len(smi), base=len(pos))
            hit = -1
            for j, i in enumerate(pos):
                mm = Chem.MolFromSmiles(smi[i])
                if mm is None or abs(ExactMolWt(mm) - mass) > 0.0008: continue
                if Chem.MolToInchiKey(mm)[:14] == tk: hit = j; break
            rec['in_window'] = hit >= 0
            if hit >= 0:
                spectra = []
                for s in g.itertuples():
                    try: ce = float(s.collision_energy_orig)
                    except Exception: ce = np.nan
                    spectra.append(dict(mz=np.asarray(s.ms2_mzs, np.float64), it=np.asarray(s.ms2_normalized_intensities, np.float64), prec=float(s.precursor_mz),
                                        adduct=str(s.adduct), mode=1 if 'pos' in str(s.ionization_mode).lower() else -1, ce=ce, ce_ev=s.collision_energy_ev))
                items = []
                for s in spectra[:8]:
                    pm, pi = fpnet.prep_peaks(s['mz'], s['it'], s['prec']); ok = not np.isnan(s['ce'])
                    items.append(dict(mz=pm, it=pi, prec=s['prec'], adduct_ix=chem.adduct_index(s['adduct']), ce=s['ce'] if ok else 0.0, ce_known=1.0 if ok else 0.0, n_merged=1, mode=float(s['mode'])))
                for md in (1, -1):
                    grp = [s for s in spectra if s['mode'] == md][:8]
                    if not grp: continue
                    mm_, ii_ = merge_spectra([(s['mz'], s['it']) for s in grp]); prec = float(np.median([s['prec'] for s in grp]))
                    pm, pi = fpnet.prep_peaks(mm_, ii_, prec); ces = [s['ce'] for s in grp if not np.isnan(s['ce'])]; ads = [s['adduct'] for s in grp]
                    items.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(max(set(ads), key=ads.count)), ce=float(np.mean(ces)) if ces else 0.0,
                                      ce_known=1.0 if ces else 0.0, n_merged=min(len(grp), 8), mode=float(md)))
                o = fi.get(src[pos].astype(np.int64), view='public'); fp = torch.as_tensor(np.asarray(o[0]), device='cuda'); form = np.asarray(o[1])
                bits = fpnet.unpack_gpu(fp, NB); same = torch.as_tensor(form == form[hit], device='cuda'); rec['n_formula'] = int(same.sum())
                sc = bits @ logits(M0s, items)
                rec['r_fold0'] = int((sc > sc[hit]).sum()) + 1; rec['rf_fold0'] = int(((sc > sc[hit]) & same).sum()) + 1
                mu, sd = float(sc.mean()), float(sc.std()) + 1e-6
                top = torch.topk(sc, min(200, len(sc))).indices.cpu().numpy()
                if hit not in top: top = np.append(top, hit)
                scn = sc.cpu().numpy(); lsid = np.asarray(w['lsid'])[pos]; lpm = np.asarray(w['lpmid'])[pos]
                fcount = pd.Series(form).map(pd.Series(form).value_counts()).to_numpy()
                csm = [smi[pos[j]] for j in top]
                for s_, f_ in zip(csm, POOL.map(_fp, csm, chunksize=16)): SC.cache[s_] = f_
                sps = [dict(mzs=x['mz'], intensities=x['it'], precursor_mz=x['prec'], adduct=x['adduct'], ionization_mode='positive' if x['mode'] == 1 else 'negative',
                            collision_energy=x['ce_ev']) for x in spectra[:16]]
                ssc = SC.score_candidates(sps, csm); SC.cache.clear()
                for rk, j in enumerate(top):
                    CAND.append((tk, int(rk), float(scn[j]), float((scn[j] - mu) / sd), float(lsid[j]), float(lpm[j]), (form[j].decode() if isinstance(form[j], bytes) else str(form[j])), int(fcount[j]), int(j == hit), len(pos), float(ssc[rk]), smi[pos[j]] if rk < 60 or j == hit else ''))
        except Exception as e:
            rec['error'] = traceback.format_exc()[-300:]
        rec['sec'] = round(time.time() - t0, 2); out.append(rec)
        if len(out) == 20: pd.DataFrame(CAND).to_parquet('/kaggle/working/pccand_probe.parquet'); print('probe rows', len(CAND), flush=True)
        if len(out) % 250 == 0 or len(out) < 4: print(len(out), rec, flush=True)
d = pd.DataFrame(out); d.to_csv('/kaggle/working/pcrank.csv', index=False)
pd.DataFrame(CAND, columns=['key', 'rk', 'fz', 'z', 'lsid', 'lpmid', 'form', 'fcount', 'y', 'nwin', 'sanp', 'smiles']).to_parquet('/kaggle/working/pccand.parquet')
x = d[d.in_window == True]
print('n', len(d), 'in_window', len(x), 'median window', d.window.median(), 'median n_formula', x.n_formula.median())
for col in ('r_fold0', 'rf_fold0'):
    v = x[col]; print(col, {k: round(float((v <= k).mean()), 3) for k in (1, 5, 25, 100, 500, 2000)}, 'median', v.median())
print(d.error.dropna().head().tolist() if 'error' in d else '')
