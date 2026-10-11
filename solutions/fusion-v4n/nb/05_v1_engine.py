# v1 engine (casmi package of casmi26-v4b-models) + fe_v4 families + LightGBM ranker
sys.path.insert(0, V4CODE)
os.makedirs('work', exist_ok=True)
for f in ['pool_meta.parquet', 'pool_fp.npy', 'pool_frag_off.npy', 'pool_frag_mass.npy', 'fp_bits.npy', 'train_structs.parquet', 'train_fp_sel.npy']:
    if not os.path.exists(f'work/{f}'):
        os.symlink(os.path.join(POOL_DIR, f), f'work/{f}')
from casmi.build import build_spec_cache
build_spec_cache(os.path.join(COMP, 'train.parquet'), 'work')
import casmi; log('casmi from', casmi.__file__)
sys.path.insert(1, os.path.join(V4CODE, 'fe_v4'))

import numpy as np, pandas as pd, torch, lightgbm as lgb
from casmi.library import Library
from casmi.pool import Pool
from casmi.engine import Engine, EngineCfg, FEATURES
from casmi import fpnet, chem
import v1engine
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
L = Library('work'); P = Pool('work')
tfp = np.load('work/train_fp_sel.npy')
_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)
_A = os.path.join(V4MOD, 'fpnet_0.pt')
if CFG['FP_BANK'] == 'ens' and _FULL:        # average fpnet_full1 (v4m) and FPNet A (v4b) logits in one ModelBank
    _fp = [_FULL[0], _A]
elif CFG['FP_BANK'] == 'full1' and _FULL:
    _fp = [_FULL[0]]
elif CFG['FP_BANK'] == 'fold0':              # validation only: never saw the fold-0 hold-out structures
    _fp = [find('casmi26-fold-safe-fpnet/**/fpnet_fold0.pt')]
else:
    _fp = [_A]
bank = fpnet.ModelBank(_fp, device=dev)
log('engine FP bank:', _fp)
E = Engine(L, P, tfp, EngineCfg(generate=True), bank)
V = v1engine.V1FE(E, [os.path.join(V4MOD, m['file']) for m in MAN['fe_models']], MAN['families'], device=dev)
if 'derivation' in MAN['families']:
    n_bad = V.mods['derivation'].self_test()
    log('derivation prior self-test mismatches:', n_bad)
    assert n_bad == 0, 'RDKit build differs from the one the derivation priors were built with'
RK = pickle.load(open(os.path.join(V4MOD, MAN['ranker']['file']), 'rb'))
RK_FEATS = list(RK['features']); RK_B = [lgb.Booster(model_str=s) for s in RK['boosters']]

def rank_score(X, names):
    pos = {n: i for i, n in enumerate(names)}
    miss = [f for f in RK_FEATS if f not in pos]
    assert not miss, f'ranker features not produced (would be zero-filled): {miss}'
    Xs = np.ascontiguousarray(X[:, [pos[f] for f in RK_FEATS]], dtype=np.float32)
    return np.mean([b.predict(Xs, num_threads=4) for b in RK_B], axis=0)
log('engine ready; ranker features', len(RK_FEATS), 'boosters', len(RK_B))
import ranker_infer
OUR_RK = None
if CFG['RANKER'] in ('ours', 'blend'):
    OUR_RK = ranker_infer.OurRanker.load(os.path.dirname(find('casmi26-ours-ranker/**/features.json')))
    log('our ranker:', len(OUR_RK.features), 'features,', len(OUR_RK.boosters), 'boosters')
import fs_pass
FS_BANK = FS_VIEWS = None
if CFG['RANKER'] in fs_pass.MODES:           # second engine pass below RANKER_LIB_GATE: fold-safe FPNet + the ranker fitted to it
    assert CFG['RANKER_LIB_GATE'] is not None, "RANKER 'fs' / 'fs_avg' needs RANKER_LIB_GATE (0.7)"
    from casmi import fpnet6
    _fs = find(CFG['RANKER_FS_NET'] or 'casmi26-fold-safe-fpnet/**/fpnet_fold0.pt')
    FS_BANK = fpnet.ModelBank([_fs], device=dev); FS_VIEWS = [fpnet6.ModelBank([_fs], device=dev)]
    OUR_RK = ranker_infer.OurRanker.load(os.path.dirname(find((CFG['RANKER_FS_DS'] or 'casmi26-ours-ranker-fs') + '/**/ranker_s0.txt')))
    _miss = [f for f in OUR_RK.features if not f.startswith(('x__', 'j__')) and f not in MAN['family_features']] +             [f for f in OUR_RK.features if f.startswith('x__') and f[3:] not in FEATURES]
    assert not _miss, f'fold-safe ranker features the engine does not produce: {_miss}'
    log('fold-safe pass:', CFG['RANKER'], 'below lib_max', CFG['RANKER_LIB_GATE'], '|', _fs, '| ranker', len(OUR_RK.features), 'features,',
        len(OUR_RK.boosters), 'boosters')
if CFG['ADDUCT_TIE'] is not None:            # merged-adduct tie of casmi/engine.py by an explicit key instead of set order (hash seed)
    import casmi.engine as _engmod
    determ.install(_engmod, E, CFG['ADDUCT_TIE'], merged=v1engine.merged_adduct)
    if CFG['ADDUCT_TIE'] == 'mean':
        for _b in (bank, FS_BANK):
            if _b is not None:
                determ.wrap_bank(_b, chem.adduct_index)
    log('ADDUCT_TIE:', CFG['ADDUCT_TIE'], '| the engine holds the set-order expression exactly once')
import pc_join
if CFG['PC_JOIN_N'] > 0:
    from casmi import frag as fragmod
    pc_join.install(E, V, CFG['PC_JOIN_ORDER'])
    log('PubChem join: N', CFG['PC_JOIN_N'], 'order', CFG['PC_JOIN_ORDER'], 'max lib', CFG['PC_JOIN_MAX_LIB'])
import conf_gate
if CFG['PC_JOIN_CONF'] is not None:
    assert CFG['PC_JOIN_N'] > 0, 'PC_JOIN_CONF needs PC_JOIN_N > 0'
    _need = [f[3:] if f.startswith('t1_') else f for f in conf_gate.MODEL['features'] if f not in ('s1', 'm1_10')]
    assert all(f in FEATURES for f in _need), 'engine features the confidence model reads are missing'
    log('confidence gate: join when p_present <', CFG['PC_JOIN_CONF'], '| deep:', CFG['PC_JOIN_CONF_DEEP'], CFG['PC_JOIN_N_DEEP'])
