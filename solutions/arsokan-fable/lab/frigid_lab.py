# Lab: FRIGID-base de novo generation on our validation lists (internet on; not a submission).
# Method follows lszlst's public "de novo hedge" cell: MIST(msg) fingerprint -> FRIGID DLM with formula filter.
import os, sys, subprocess, glob, json, time, re, itertools, traceback
T0 = time.time()
def sh(c): print('$', c, flush=True); return subprocess.run(c, shell=True, capture_output=True, text=True)
r = sh('git clone --depth 1 https://github.com/coleygroup/FRIGID /kaggle/working/FRIGID'); print(r.stderr[-300:])
r = sh(sys.executable + ' -m pip install -q --no-deps bionemo-moco==0.0.2.1 safe-mol datamol selfies loguru jaxtyping hydra-core omegaconf antlr4-python3-runtime==4.9.3 lightning pytorch-lightning lightning-utilities torchmetrics einops easydict ngboost'); print(r.stdout[-500:], r.stderr[-1500:])
r = sh(sys.executable + ' -m pip install -q rdkit'); print(r.stderr[-300:])
r = sh(sys.executable + ' -m pip install -q transformers==4.52.4'); print(r.stdout[-300:], r.stderr[-800:])
import transformers; print('transformers', transformers.__version__, flush=True)
sys.path.insert(0, '/kaggle/working/FRIGID/src')
import numpy as np, pandas as pd, torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True)
try:
    import transformers.utils as _tu
    if not hasattr(_tu, 'download_url'):
        def _d(*a, **k): raise RuntimeError('download_url unavailable')
        _tu.download_url = _d
except Exception as e: print('shim', e)
CHAIN = ['import safe', 'from safe.converter import SAFEConverter', 'import datamol', 'import selfies',
         'from bionemo.moco.interpolants import MDLM', 'from dlm.utils.benchmark_utils import binarize_fingerprint, generate_with_formula_filter',
         'from dlm.model import DLM', 'from dlm.sampler import Sampler', 'from mist.models import SpectraEncoderGrowing',
         'from mist.data.featurizers import PeakFormula', 'from mist.data import data as mdata']
for st in CHAIN:
    try: exec(st, globals()); print('ok  ', st, flush=True)
    except Exception as e:
        print('FAIL', st, type(e).__name__, str(e)[:600], flush=True); traceback.print_exc(); raise
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog('rdApp.*')
ck = glob.glob('/kaggle/input/**/DLM.ckpt', recursive=True)[0]; enc_ck = glob.glob('/kaggle/input/**/mist_msg.pt', recursive=True)[0]
print(ck, enc_ck, flush=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
_tl = torch.load
def _tload(*a, **k):
    k['weights_only'] = False
    return _tl(*a, **k)
torch.load = _tload   # the public FRIGID checkpoint stores its omegaconf config
sampler = Sampler(ck); sampler.model.eval()
enc = SpectraEncoderGrowing(hidden_size=640, magma_modulo=2048, peak_attn_layers=2, num_heads=8, spectra_dropout=0.0, output_size=4096,
    form_embedder='pos-cos', set_pooling='cls', refine_layers=4, pairwise_featurization=True, embed_instrument=False, inten_transform='float',
    inten_prob=0.0, remove_prob=0.0, cls_type='ms1', spec_features='peakformula', mol_features='fingerprint', top_layers=1, max_peaks=None).to(dev).eval()
mck = torch.load(enc_ck, map_location='cpu', weights_only=False)
msd = {re.sub(r'^model\.', '', k): v for k, v in mck.get('state_dict', mck).items()}
print('encoder load', enc.load_state_dict(msd, strict=False), flush=True)
print('models loaded', round(time.time() - T0), flush=True)
MASS = {'C': 12.0, 'H': 1.0078250319, 'N': 14.0030740052, 'O': 15.9949146221, 'P': 30.97376151, 'S': 31.97207069, 'F': 18.99840320,
        'Cl': 34.96885271, 'Br': 78.9183376, 'I': 126.904468, 'Na': 22.98976928, 'K': 38.9637069, 'Si': 27.9769265, 'B': 11.0093054,
        'As': 74.9215965, 'Se': 79.9165218}
TOK = re.compile(r'([A-Z][a-z]?)(\d*)'); PROTON = 1.00727646 - 0.00054857990
ION = {'[M+H]+': '[M+H]+', '[M+Na]+': '[M+Na]+', '[M+K]+': '[M+K]+', '[M+NH4]+': '[M+H3N+H]+', '[M+H-H2O]+': '[M-H2O+H]+',
       '[M-H]-': '[M-H2O+H]+', '[M+CH2O2-H]-': '[M-H2O+H]+', '[M+Cl]-': '[M-H2O+H]+'}
def parse_formula(f):
    m = TOK.findall(f)
    if not m or ''.join(a + (b or '') for a, b in m) != f: return None
    d = {}
    for e, n in m:
        if e not in MASS: return None
        d[e] = d.get(e, 0) + (int(n) if n else 1)
    return d
def assign(form, mzs, ppm=20.0, cap=300_000):
    c = parse_formula(form)
    if c is None: return None
    els = list(c); tot = 1
    for e in els: tot *= c[e] + 1
    if tot > cap: return None
    subs = []
    for combo in itertools.product(*[range(c[e] + 1) for e in els]):
        if not any(combo): continue
        subs.append((''.join(f'{e}{k}' if k > 1 else e for e, k in zip(els, combo) if k), sum(k * MASS[e] for k, e in zip(combo, els))))
    sm = np.array([m for _, m in subs]); o = np.argsort(sm); sms = sm[o]; names = [subs[i][0] for i in o]; fr = []
    for mz in mzs:
        for probe in (float(mz) - PROTON, float(mz)):
            lo, hi = probe * (1 - ppm * 1e-6), probe * (1 + ppm * 1e-6); i0, i1 = np.searchsorted(sms, lo), np.searchsorted(sms, hi)
            if i1 > i0: fr.append(names[i0 + int(np.argmin(np.abs(sms[i0:i1] - probe)))]); break
    return fr
N_REQ = int(os.environ.get('N_REQ', '10')); LIMIT = float(os.environ.get('LIMIT_H', '10.5')) * 3600
JOBS = [('v3', 'submission3.csv'), ('v2', 'submission.csv')]
lab = os.path.dirname(glob.glob('/kaggle/input/**/submission3.csv', recursive=True)[0])
out = open('/kaggle/working/frigid_gen.jsonl', 'w'); wd = '/kaggle/working/io'; sd, fd = wd + '/spec', wd + '/subform'
os.makedirs(sd, exist_ok=True); os.makedirs(fd, exist_ok=True)
SMOKE = os.environ.get('SMOKE_N')
for tag, subf in JOBS:
    vd = os.path.dirname([p for p in glob.glob('/kaggle/input/**/val_truth.parquet', recursive=True) if ('val-v3' in p) == (tag == 'v3')][0])
    V = pd.read_parquet(vd + '/val_truth.parquet'); T = pd.read_parquet(vd + '/val_test.parquet'); S = pd.read_csv(f'{lab}/{subf}', dtype=str, keep_default_na=False)
    S = S[S.molecule_id.isin(set(V.molecule_id))]
    print(tag, 'molecules', len(S), flush=True)
    T['pref'] = (T.adduct != '[M+H]+').astype(int) * 10 - T.ms2_mzs.map(len) / 1000.0
    spec = {m: g.sort_values('pref').iloc[0] for m, g in T.groupby('molecule_id')}
    tform = dict(zip(V.molecule_id, V.formula)); fold = dict(zip(V.molecule_id, V.fold)); tkey = dict(zip(V.molecule_id, V.inchikey14))
    rows = list(S.itertuples()); rows = rows[:int(SMOKE)] if SMOKE else rows
    PRE = {}
    for r in rows:
        mid = r.molecule_id; s = spec[mid]
        try:
            m1 = Chem.MolFromSmiles(r.smiles.split(';')[0]); f = CalcMolFormula(m1) if m1 else tform[mid]; f = re.sub(r'[+-]\d*$', '', f)
            key = f'{tag}_{mid}'; ion = ION.get(str(s.adduct), '[M+H]+'); fr = assign(f, s.ms2_mzs)
            if fr is None: raise ValueError('formula not assignable ' + f)
            json.dump({'cand_form': f, 'cand_ion': ion, 'output_tbl': {'formula': fr, 'ms2_inten': [1.0] * len(fr), 'ions': [ion] * len(fr)}}, open(f'{fd}/{key}.json', 'w'))
            with open(f'{sd}/{key}.ms', 'w') as fh:
                fh.write(f'>{key} {float(s.precursor_mz):.5f}\n')
                for mz, it in zip(s.ms2_mzs, s.ms2_normalized_intensities): fh.write(f'{float(mz):.5f} {float(it):.5f}\n')
            PRE[mid] = (f, key, len(fr))
        except Exception as e:
            PRE[mid] = f'{type(e).__name__}: {str(e)[:200]}'
    feat = PeakFormula(subform_folder=fd, augment_data=False, cls_type='ms1', inten_transform='float', magma_modulo=2048, max_peaks=None)
    for n, r in enumerate(rows):
        if time.time() - T0 > LIMIT: print('time limit', flush=True); break
        mid = r.molecule_id; s = spec[mid]; t1 = time.time(); rec = dict(set=tag, molecule_id=mid, fold=fold[mid], adduct=s.adduct, truth_formula=tform[mid])
        try:
            if isinstance(PRE[mid], str): raise ValueError(PRE[mid])
            f, key, _np = PRE[mid]; rec['formula'] = f; fr = [0] * _np
            sp2 = mdata.Spectra(spectra_name=key, spectra_file=f'{sd}/{key}.ms', spectra_formula=f, instrument='timsTOF')
            b = feat.collate_fn([feat.featurize(sp2)]); b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.no_grad(): fp, _ = enc(b)
            pfp = binarize_fingerprint(fp.cpu().numpy()[0], 0.187)
            o = generate_with_formula_filter(sampler=sampler, fingerprint_array=pfp, target_formula=f, target_smiles=None, n_required=N_REQ,
                                             max_attempts=16, batch_size=16, softmax_temp=1.0, randomness=0.5)
            g = [x for x in (o[0] if isinstance(o, tuple) else o) if isinstance(x, str) and x]
            sc = [float(x) for x in o[1]] if isinstance(o, tuple) and len(o) > 1 else []
            keys = [(lambda mm: Chem.MolToInchiKey(mm)[:14] if mm else None)(Chem.MolFromSmiles(x)) for x in g]
            rec.update(gen=g, score=sc, keys=keys, nbits=int(pfp.sum()), npeaks=len(fr), hit=(keys.index(tkey[mid]) + 1) if tkey[mid] in keys else 0)
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {str(e)[:200]}'
            if n < 3: traceback.print_exc()
        rec['sec'] = round(time.time() - t1, 1); out.write(json.dumps(rec) + '\n'); out.flush()
        if n % 10 == 0: print(tag, n, len(rows), rec.get('hit'), rec.get('error'), rec['sec'], round(time.time() - T0), flush=True)
out.close(); print('DONE', round(time.time() - T0), flush=True)
