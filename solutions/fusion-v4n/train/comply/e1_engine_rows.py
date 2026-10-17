"""Engine-row runner for our engine-1 ranker (stack1 compliance step 2): the v1 engine + fe_v4 families of casmi26-v4b-models (MIT code)
with OUR fingerprint network in the engine bank and in all three model_views slots, run on train.parquet molecules with the answer hidden,
every candidate row stored. No PubChem join (stack1's engine 1 has none).

Per molecule (one row of the set table: mid, key = metric key of the answer, pid = its pool row, qlib = its query library) the query
spectra are the train.parquet rows of the set's queries table (column `row`); the library is masked like a hidden-test molecule:
  c2    every library spectrum of the answer's metric key hidden; the answer stays a pool candidate
  c1    only the query library's spectra of the answer hidden (other libraries keep theirs)
  drop  as c2, and the answer's pool row is removed (the engine can only reach it by generation)
The engine is built as stack1's cell 18 with the step-2 switch on builds it (our network in the bank and the three slots, families of
MANIFEST.json without derivation, fragnet on OUR stage A / B): no third-party weight is loaded (no public ranker, no fpnet_full1 /
fe_A/B/D, no derivation tables, no shipped stage A / B). The ownfp study's P4 rows were written by the no-join path of
scratch/pcjoin/run_join.py through scratch/ownfp/p4.py (same engine, masks and network; it also loaded the public ranker for a column the
step-2 ranker does not use, and the derivation family whose 19 features step 2 drops); e1_rows.py turns either into the same 141 features.
Output chunks (format 'join' of e1_rows.load_run): w<i>_c<k>.npz (X, F, rk = 0, pid, y, g, inj = False, pcrank, pcfz, names_x, names_f,
strs = strings of generated rows) + w<i>_c<k>.parquet (one row per molecule). Resumable; one process per worker (--worker i/n).
usage: python e1_engine_rows.py --mols MOLS.parquet --queries Q.parquet --ver c2|c1|drop --net NET.pt --stage_a stageA.npz
       --stage_b stageB.txt --out DIR [--worker i/n] [--v4b D:/casmi-data/ranker_study/ds/v4b] [--work D:/casmi-data/ranker_study/work]
"""
import argparse, json, os, sys, time
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); SOL = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(SOL, 'prototypes'))


def build_engine(v4b, work, net, stage_a, stage_b, device='cpu'):
    """v1 engine + V1FE as stack1's cell 18 builds them with the step-2 switch on (see the module docstring)."""
    code = os.path.join(v4b, 'code')
    for p in (os.path.join(code, 'fe_v4'), code):
        if p not in sys.path:
            sys.path.insert(0, p)
    import lightgbm as lgb
    from casmi.library import Library
    from casmi.pool import Pool
    from casmi.engine import Engine, EngineCfg, FEATURES
    from casmi import fpnet, chem
    import v1engine
    import fams.fragnet as FN
    man = json.load(open(os.path.join(v4b, 'MANIFEST.json')))
    np_load = np.load

    def _mm(path, *x, **k):                                  # big tables memory-mapped (workers share them through the page cache)
        if isinstance(path, str) and path.endswith('.npy') and 'mmap_mode' not in k and os.path.getsize(path) > (32 << 20):
            k['mmap_mode'] = 'r'
        return np_load(path, *x, **k)
    np.load = _mm
    try:
        L = Library(work); P = Pool(work); tfp = np.load(os.path.join(work, 'train_fp_sel.npy'))
    finally:
        np.load = np_load
    E = Engine(L, P, tfp, EngineCfg(generate=True), fpnet.ModelBank([net], device=device))

    def _prepare(res):                                       # == fams.fragnet.prepare with our stage A / B
        assert abs(chem.MASS['H'] - FN.H_MASS) < 1e-12
        FN._R['A'] = FN.StageA(stage_a); FN._R['B'] = lgb.Booster(model_file=stage_b)
        assert list(FN._R['B'].feature_name()) == FN.BNAMES
        FN._R['rec'] = FN.PoolRecords(res.P.smiles)
    FN._R.clear(); FN.prepare = _prepare
    V = v1engine.V1FE(E, [net] * 3, [f for f in man['families'] if f != 'derivation'], device=device)
    return SimpleNamespace(E=E, V=V, L=L, P=P, chem=chem, FEATURES=list(FEATURES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mols', required=True); ap.add_argument('--queries', required=True); ap.add_argument('--ver', required=True, choices=['c2', 'c1', 'drop'])
    ap.add_argument('--net', required=True); ap.add_argument('--stage_a', required=True); ap.add_argument('--stage_b', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--worker', default='0/1'); ap.add_argument('--chunk', type=int, default=40)
    ap.add_argument('--v4b', default='D:/casmi-data/ranker_study/ds/v4b'); ap.add_argument('--work', default='D:/casmi-data/ranker_study/work')
    ap.add_argument('--device', default='cpu'); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    import local_engine as LE                                # spectra_from_rows, target_mass, truth_mask (ours)
    wi, wn = map(int, a.worker.split('/')); os.makedirs(a.out, exist_ok=True)
    T = pd.read_parquet(a.mols)
    if a.ver == 'c1':
        T = T[T.c1] if 'c1' in T.columns else T[T.c1_ok]
    T = T.reset_index(drop=True)
    if a.limit:
        T = T.iloc[:a.limit]
    T = T.iloc[wi::wn].reset_index(drop=True)
    done = set()
    for f in os.listdir(a.out):
        if f.endswith('.parquet'):
            done |= set(pd.read_parquet(os.path.join(a.out, f), columns=['mid']).mid)
    todo = T[~T.mid.isin(done)].reset_index(drop=True)
    print(f'{len(T)} molecules, {len(done)} done, {len(todo)} to do', flush=True)
    if not len(todo):
        return
    import torch
    torch.set_num_threads(1)
    H = build_engine(a.v4b, a.work, a.net, a.stage_a, a.stage_b, a.device)
    Q = pd.read_parquet(a.queries); Q = Q[Q.molecule_id.isin(set(todo.mid))]
    Q['nm'] = H.chem.neutral_mass(Q.precursor_mz.values.astype(np.float64), Q.adduct.values)
    QG = {m: g for m, g in Q.groupby('molecule_id', sort=False)}
    skey, ssid, slib = H.L.struct_key, H.L.sid, H.L.lib
    st = dict(k=len([f for f in os.listdir(a.out) if f.startswith(f'w{wi}_') and f.endswith('.parquet')]), buf=[], meta=[], strs=[], names=None)

    def flush():
        if not st['meta']:
            return
        z = dict(names_x=np.array(H.FEATURES), names_f=np.array(st['names'] or []), strs=np.array(json.dumps(st['strs'])))
        for c in ('X', 'F', 'rk', 'pid', 'y', 'g', 'inj', 'pcrank', 'pcfz'):
            z[c] = np.concatenate([b[c] for b in st['buf']]) if st['buf'] else np.zeros(0)
        tmp = os.path.join(a.out, f"w{wi}_c{st['k']:04d}.tmp.npz"); np.savez_compressed(tmp, **z)
        os.replace(tmp, os.path.join(a.out, f"w{wi}_c{st['k']:04d}.npz"))
        pd.DataFrame(st['meta']).to_parquet(os.path.join(a.out, f"w{wi}_c{st['k']:04d}.parquet"), index=False)
        st.update(buf=[], meta=[], strs=[], k=st['k'] + 1)
    t0 = time.time()
    for i, r in enumerate(todo.itertuples()):
        sub = QG.get(r.mid); rec = dict(mid=r.mid, ver=a.ver, n_cand=0, lib_max=0.0, n_gen=0, sec=0.0, truth_in=False, err='')
        t1 = time.time()
        try:
            target = LE.target_mass(sub.nm.values) if sub is not None else None
            if target is None:
                raise ValueError('no usable query spectrum')
            ex = LE.truth_mask(skey, ssid, slib, r.key, lib=int(r.qlib) if a.ver == 'c1' else None)
            C, X, F, names, info = H.V.run(LE.spectra_from_rows(sub), target, exclude=ex, drop_pid=int(r.pid) if a.ver == 'drop' else -1)
            n = len(C.pid)
            if st['names'] is None and n:
                st['names'] = list(names)
            y = np.array([kk == r.key for kk in C.key], bool) if n else np.zeros(0, bool)
            if n:
                for j in np.flatnonzero(np.asarray(C.pid) < 0):
                    st['strs'].append(dict(mid=r.mid, j=int(j), smiles=C.smiles[j], key=C.key[j], formula=C.formula[j]))
                st['buf'].append(dict(X=np.asarray(X, np.float32), F=np.asarray(F, np.float32), rk=np.zeros(n, np.float32), pid=np.asarray(C.pid, np.int32),
                                      y=y, g=np.full(n, len(st['meta']), np.int32), inj=np.zeros(n, bool), pcrank=np.full(n, -1, np.int16),
                                      pcfz=np.full(n, np.nan, np.float32)))
            rec.update(n_cand=n, lib_max=float(info.get('lib_max', 0.0)), n_gen=int(info.get('n_gen', 0)), truth_in=bool(y.any()), target=target)
        except Exception as e:
            rec['err'] = repr(e)[:300]; print('ERROR', r.mid, repr(e)[:300], flush=True)
        rec['sec'] = time.time() - t1; st['meta'].append(rec)
        if len(st['meta']) >= a.chunk:
            flush()
        if (i + 1) % 20 == 0:
            print(f'{i + 1}/{len(todo)} molecules, {(time.time() - t0) / (i + 1):.2f} s/mol', flush=True)
    flush(); print('done', len(todo), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ.setdefault(_v, '1')
    main()
