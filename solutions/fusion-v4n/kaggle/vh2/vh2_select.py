"""Selection for harness v2 (casmi26-vh2-gnx11): a hard natural-product view plus the two v1 views.
Groups (300 molecules):
  nphard  150  natural-product-table structures (COCONUT / LOTUS / NPAtlas, plain or metric key) with NO reference spectrum in
               the external libraries the stack reads (MassBank-harmonized, extlib gnps-full, knowledge-table external refs),
               not an enveda-np-examples / enveda-180 structure, in fold 0 (the fold-safe FPNet's split.parquet 'hold' list: never seen
               by fpnet_fold0.pt, which VH_CLEAN_MAIN also loads into the main engine), with >= 1 TOF-type query spectrum (instrument_type matching
               qtof / q-tof / tof / maxis, not ion-trap TOF) in a visible-test adduct, collision energy 20-60 eV, >= 5 peaks.
               Molecules also in the FR 3-25 validation split (unseen by FR) are taken first.
  npref0  35   the enveda-np-examples answers with no external reference spectrum (v1 view)
  e180    115  enveda-180 structures (not NP) from the same fold 0, seed 20261124 (stop-loss)
Hold-out: every train row of the 400 inchikey14s and of plain keys sharing their metric key.
out: scratch/land1123/vh2_selection.json      usage: python -I scratch/land1123/vh2_select.py"""
import hashlib, json, os, pickle, re, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
TRAIN = os.path.join(REPO, 'data', 'input', 'competitions', 'enveda-CASMI26-molecule-id-mass-spectra', 'train.parquet')
SEED, N_HARD, N_CTRL, MAX_PPM = 20261124, 150, 115, 5.0
TOF_RE = r'(?i)(q-?tof|timstof|maxis|^esi-tof$|^lc-esi-tof$|lc-q-tof)'
CE_LO, CE_HI, MIN_PEAKS = 20.0, 60.0, 5
v1 = json.load(open(os.path.join(HERE, 'vh_selection.json')))
k = pd.read_pickle(os.path.join(HERE, 'cache', 'v2_keys.pkl'))       # built in the exploration step: nat / ref / frval / e180 per key
# fold 0 = the 55,102 'hold' structures the fold-safe FPNet fold0 model never saw (its own split.parquet; identity by InChIKey-14)
spl = pd.read_parquet(os.path.join(REPO, 'data', 'input', 'datasets', 'dmitriigluzdov', 'casmi26-fold-safe-fpnet', 'split.parquet'))
HOLD = set(spl.loc[spl.fold == 'hold', 'inchikey14']) | set(spl.loc[spl.fold == 'hold', 'key'])
k['f0'] = k.ik.isin(HOLD) | k.mk.isin(HOLD)
aux = pickle.load(open(os.path.join(HERE, 'cache', 'vh_aux.pkl'), 'rb'))
ans = pd.read_csv(os.path.join(HERE, 'vh_out', 'vh', 'answers.csv'), dtype=str)
npref0 = sorted(ans.set_index('molecule_id').loc[aux['strata']['np no ext. reference spectrum'], 'inchikey14'])
t = pq.read_table(TRAIN, columns=['inchikey14', 'instrument_type', 'adduct', 'collision_energy_ev', 'num_peaks', 'precursor_mz', 'normalized_smiles', 'ingest_lib']).to_pandas()
# precursor accuracy as engine 2 computes it (pv.ADDUCTS deltas): v2 run 1 lost a molecule whose spectra were 19-157 ppm off
DELTA = {'[M+H]+': 1.00727645, '[M-H]-': -1.00727645, '[M+CH2O2-H]-': 44.99820285, '[M+Na]+': 22.9892207,
         '[M+NH4]+': 18.03382555, '[M+K]+': 38.9631581, '[M+Cl]-': 34.96940126}
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
_sm = t.drop_duplicates('inchikey14').set_index('inchikey14').normalized_smiles
_need = set(k.loc[k.nat & ~k.ref & ~k.npx, 'ik']) | set(t.loc[t.ingest_lib == 'enveda-180', 'inchikey14'])
EX = {}
for _k in _need:
    _m = Chem.MolFromSmiles(_sm.get(_k, '') or '')
    EX[_k] = Descriptors.ExactMolWt(_m) if _m is not None else np.nan
_mass = t.inchikey14.map(EX)
ppm = ((t.precursor_mz - t.adduct.map(DELTA) - _mass) / _mass * 1e6).abs()
ce = t.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
ok = t.instrument_type.fillna('').str.contains(TOF_RE, regex=True) & t.adduct.isin(v1['vis_adducts']) & (t.num_peaks >= MIN_PEAKS) & ce.between(CE_LO, CE_HI) & (ppm <= MAX_PPM)
good = set(t.loc[ok, 'inchikey14'])
pm = pd.read_parquet(os.path.join(REPO, 'data', 'input', 'datasets', 'dmitriigluzdov', 'casmi26-attributed-v4g-reproduction-assets', 'casmi26-v2-pool', 'pool_meta.parquet'), columns=['key', 'src'])
COCO = set(pm.loc[pm.src >= 2, 'key'])
c = k[k.nat & ~k.ref & ~k.npx & ~k.e180 & k.f0 & k.ik.isin(good) & (k.ik.isin(COCO) | k.mk.isin(COCO))]   # answer in the COCONUT part of both engines' pools
print('eligible nphard', len(c), '| also FR-val', int(c.frval.sum()))
rng = np.random.default_rng(SEED)
first = sorted(c.loc[c.frval, 'ik']); rest = sorted(c.loc[~c.frval, 'ik'])
hard = first + sorted(rng.choice(rest, N_HARD - len(first), replace=False).tolist())
e = t[t.instrument_type.fillna('') == 'timsTOF']
ectl = k[k.e180 & ~k.nat & k.f0 & k.ik.isin(set(t.loc[(t.ingest_lib == 'enveda-180') & t.adduct.isin(v1['vis_adducts']) & (ppm <= MAX_PPM), 'inchikey14']))]
ctrl = sorted(rng.choice(sorted(ectl.ik), N_CTRL, replace=False).tolist())
print('e180 eligible in fold 0', len(ectl))
# engine 2 covers a molecule only if its own pool (COCONUT + PubChemLite, both capped at 1,300 Da, + ChEBI/LIPID MAPS + train)
# has a structure within 30 ppm of the target (run 3 lost m_a1c907, mass 1,362). Rule on the answer's exact mass (targets are
# within 5 ppm of it): >= 1 such structure within 25 ppm, and the mass inside the envelope of molecules that passed (<= 1,292.66).
sys.path.insert(0, HERE); from eng2_window import n_within
MAX_TARGET, MAX_PRECURSOR = 1292.6554, 1291.6481
eng_ok = lambda x: np.isfinite(EX.get(x, np.nan)) and EX[x] <= MAX_TARGET and n_within(EX[x], 25) > 0
rng2 = np.random.default_rng(SEED + 1)
bad_h = [x for x in hard if not eng_ok(x)]
spare = [x for x in sorted(rest) if x not in set(hard) and eng_ok(x)]
hard = sorted([x for x in hard if eng_ok(x)] + rng2.choice(spare, len(bad_h), replace=False).tolist())
bad_c = [x for x in ctrl if not eng_ok(x)]
ctrl = sorted([x for x in ctrl if eng_ok(x)] + rng2.choice([x for x in sorted(ectl.ik) if x not in set(ctrl) and eng_ok(x)], len(bad_c), replace=False).tolist())
print('replaced: nphard', bad_h, '| e180', bad_c)   # npref0 masses are added later; all 35 passed in runs 1 and 3
held = set(hard) | set(npref0) | set(ctrl)
ts = pd.read_parquet(os.path.join(REPO, 'scratch', 'clean', 'pool', 'tables', 'train_structs.parquet'), columns=['inchikey14', 'key'])
twins = sorted(set(ts.loc[ts.key.isin(set(ts.loc[ts.inchikey14.isin(held), 'key'])), 'inchikey14']) - held)
tall = pq.read_table(TRAIN, columns=['inchikey14']).column(0).to_pandas()
rows = int(tall.isin(held | set(twins)).sum())
for _k in held - set(EX):
    _m = Chem.MolFromSmiles(_sm.get(_k, '') or '')
    EX[_k] = Descriptors.ExactMolWt(_m) if _m is not None else np.nan
MKD = k.set_index('ik').mk.to_dict()
sel_mass = {x: round(float(EX[x]), 6) for x in sorted(held)}
assert all(np.isfinite(v) for v in sel_mass.values())
sel = dict(max_target=MAX_TARGET, max_precursor=MAX_PRECURSOR, mass=sel_mass, mkey={x: MKD.get(x) for x in sorted(held) if isinstance(MKD.get(x), str)}, seed=SEED, n_train_rows=int(len(tall)), vis_adducts=v1['vis_adducts'], twin_keys=twins,
           groups=[dict(name='nphard', keys=sorted(hard), libs=None, inst_re=TOF_RE, ce=[CE_LO, CE_HI], min_peaks=MIN_PEAKS, cap=4, max_ppm=MAX_PPM),
                   dict(name='npref0', keys=npref0, libs=['enveda-np-examples'], inst_re=None, ce=None, min_peaks=0, cap=None),
                   dict(name='e180', keys=ctrl, libs=['enveda-180'], inst_re=None, ce=None, min_peaks=0, cap=4, max_ppm=MAX_PPM)],
           expect=dict(rows_removed=rows), fr_val_in_nphard=len(first))
json.dump(sel, open(os.path.join(HERE, 'vh2_selection.json'), 'w'), separators=(',', ':'))
print('groups', {g['name']: len(g['keys']) for g in sel['groups']}, 'twins', len(twins), 'rows removed', rows)
