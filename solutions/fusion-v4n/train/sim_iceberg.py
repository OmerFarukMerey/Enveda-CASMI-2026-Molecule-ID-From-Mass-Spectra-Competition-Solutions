"""Simulate [M+H]+ spectra of pool structures with the packaged ICEBERG checkpoint, as extra FPNet training data.

ICEBERG (ms-pred, MassSpecGym checkpoint, MIT; package of ahmedberatozer/casmi26-iceberg) predicts a spectrum for any
structure. This script runs it for a table of structures at several collision energies and writes the result in the
layout of the FPNet spectrum cache, so that train_fpnet.py --sim can mix it into training. Only [M+H]+ (and [M+Na]+)
are trained adducts of the checkpoint; every non-Orbitrap instrument uses the token QTOF.

Sharing: below-normal priority, a few CPU threads, a GPU memory cap, and train/gpushare.py after every batch (duty
cycle, smaller while a game runs). When other processes need the GPU memory it exits with code 75 between chunks;
finished chunks stay on disk and the next start continues after them (train/guard_run.py restarts it).

usage:
  python train/guard_run.py -- python train/sim_iceberg.py --structs <parquet: pid, smiles, mass[, set]> --out <dir> \
      --pkg <iceberg package dir> --site <dir with the pure wheels> --code <dir with casmi/>
  python train/sim_iceberg.py ... --assemble      # chunks -> sim_mz.npy, sim_it.npy, sim_off.npy, sim_meta.parquet
RDKit is the environment's (runner switch any_rdkit): predictions are not bit-identical to the Kaggle runner.
"""
import argparse, glob, json, os, sys, time, types

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--structs', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--pkg', default=r'D:\casmi-data\coverage_study\iceberg')
ap.add_argument('--site', default=r'D:\casmi-data\coverage_study\ice_site_win')
ap.add_argument('--code', default=r'D:\casmi-data\directions\code', help='dir with the casmi package (peak cleaning, adduct table)')
ap.add_argument('--ces', default='20,40,60'); ap.add_argument('--adduct', default='[M+H]+'); ap.add_argument('--instr', default='QTOF')
ap.add_argument('--bs', type=int, default=32); ap.add_argument('--chunk', type=int, default=256, help='structures per chunk file')
ap.add_argument('--mem_gb', type=float, default=4.0); ap.add_argument('--threads', type=int, default=3)
ap.add_argument('--duty', type=float, default=0.8); ap.add_argument('--game_duty', type=float, default=0.25)
ap.add_argument('--yield_gb', type=float, default=9.0); ap.add_argument('--limit', type=int, default=0, help='only the first N structures')
ap.add_argument('--resume', action='store_true', help='accepted for guard_run.py; finished chunks are always skipped')
ap.add_argument('--assemble', action='store_true')
A = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpushare
gpushare.below_normal()
os.environ.setdefault('OMP_NUM_THREADS', str(A.threads)); os.environ.setdefault('MKL_NUM_THREADS', str(A.threads))
import pandas as pd

YIELD_CODE = 75
CES = [float(x) for x in A.ces.split(',')]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def chunk_path(c):
    return os.path.join(A.out, 'chunks', f'chunk_{c:05d}.npz')


def assemble(S):
    """Chunks -> arrays in the layout of the spectrum cache (cleaned like casmi.build: <= precursor + 2, floor 0.001)."""
    sys.path.insert(0, A.code)
    from casmi import chem
    from casmi.spectra import clean_peaks
    files = sorted(glob.glob(os.path.join(A.out, 'chunks', 'chunk_*.npz')))
    mass = dict(zip(S.pid.values, S.mass.values)); sset = dict(zip(S.pid.values, S.set.values)) if 'set' in S else {}
    rows, mzs, its, fails, secs, npred = [], [], [], [], 0.0, 0
    for f in files:
        z = np.load(f, allow_pickle=False)
        secs += float(z['seconds']); npred += int(z['n_pred'])
        fails += [(int(p), str(r)) for p, r in zip(z['fail_pid'], z['fail_why'])]
        off = z['off']
        for j in range(len(z['pid'])):
            pid = int(z['pid'][j]); prec = chem.precursor_mz_from_mass(float(mass[pid]), A.adduct)
            m, v = clean_peaks(z['mz'][off[j]:off[j + 1]].astype(np.float64), z['it'][off[j]:off[j + 1]].astype(np.float64), prec, 0.001, 512, 2.0)
            if len(m) == 0:
                fails.append((pid, f'no peak at CE {z["ce"][j]}')); continue
            rows.append((pid, sset.get(pid, ''), float(z['ce'][j]), prec, len(m), int(off[j + 1] - off[j]))); mzs.append(m); its.append(v)
    meta = pd.DataFrame(rows, columns=['pid', 'set', 'ce', 'precursor_mz', 'n_clean', 'n_raw'])
    meta['adduct_ix'] = chem.adduct_index(A.adduct); meta['mode'] = 1
    off = np.concatenate([[0], np.cumsum(meta.n_clean.values)]).astype(np.int64)
    np.save(os.path.join(A.out, 'sim_off.npy'), off); np.save(os.path.join(A.out, 'sim_mz.npy'), np.concatenate(mzs).astype(np.float32))
    np.save(os.path.join(A.out, 'sim_it.npy'), np.concatenate(its).astype(np.float32))
    meta.to_parquet(os.path.join(A.out, 'sim_meta.parquet'), index=False)
    pd.DataFrame(fails, columns=['pid', 'why']).to_csv(os.path.join(A.out, 'sim_fail.csv'), index=False)
    done = sum(len(np.load(f)['chunk_pid']) for f in files)
    info = dict(chunks=len(files), structures_done=int(done), structures_with_spectrum=int(meta.pid.nunique()), spectra=len(meta),
                predictions=npred, gpu_seconds=round(secs, 1), pred_per_s=round(npred / max(secs, 1e-9), 2),
                peaks_median=float(meta.n_clean.median()), peaks_mean=round(float(meta.n_clean.mean()), 1), ces=CES, adduct=A.adduct, instr=A.instr)
    json.dump(info, open(os.path.join(A.out, 'sim_info.json'), 'w'), indent=1)
    log('assembled', json.dumps(info))


def main():
    S = pd.read_parquet(A.structs)
    if A.limit:
        S = S.iloc[:A.limit]
    os.makedirs(os.path.join(A.out, 'chunks'), exist_ok=True)
    if A.assemble:
        return assemble(S)
    n_chunks = (len(S) + A.chunk - 1) // A.chunk
    todo = [c for c in range(n_chunks) if not os.path.exists(chunk_path(c))]
    log(f'{len(S)} structures, {n_chunks} chunks, {len(todo)} to do, CE {CES}, {A.adduct} {A.instr}')
    if not todo:
        return
    sys.path.insert(0, A.site); sys.path.insert(0, A.pkg)
    import ice_runner as R
    rlog = R.Log(quiet=True)
    R.setup_env(types.SimpleNamespace(pkg=A.pkg, site='none', any_rdkit=True), rlog)
    import torch
    torch.set_num_threads(A.threads)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.cuda.set_per_process_memory_fraction(min(1.0, A.mem_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    model = R.load_model(A.pkg, dev, rlog)
    pred = R.Predictor(model, dev, A.bs, lambda *a: log('ice:', *a))
    share = gpushare.Share(A.duty, A.game_duty, A.yield_gb, log=log)
    t_run = time.time(); n_run = 0
    for c in todo:
        share.poll()
        if share.must_yield():
            log(f'YIELD: other processes hold {share.other:.1f} GB of GPU memory -> exiting before chunk {c}')
            sys.stdout.flush(); os._exit(YIELD_CODE)
        d = S.iloc[c * A.chunk:(c + 1) * A.chunk]
        ok, fail_pid, fail_why = [], [], []
        for pid, smi in zip(d.pid.values, d.smiles.values):
            good, why = R.smiles_ok(smi)
            if good:
                ok.append((smi, int(pid)))
            else:
                fail_pid.append(int(pid)); fail_why.append(why)
        ok.sort(key=lambda x: (len(x[0]), x[0]))                         # the runner's order: batch size by SMILES length
        pids, ces, lens, mzs, its = [], [], [], [], []
        secs, npred = 0.0, 0
        for ce in CES:
            i = 0
            while i < len(ok):
                L = len(ok[min(i + pred.bs, len(ok)) - 1][0])
                bs = pred.bs if L <= 60 else (pred.bs // 2 if L <= 90 else (pred.bs // 4 if L <= 140 else pred.bs // 8))
                sub = ok[i:i + max(1, bs)]; i += len(sub)
                ts = time.time()
                outs = pred._safe([s for s, _ in sub], (A.adduct, A.instr, ce))
                if dev == 'cuda':
                    torch.cuda.synchronize()
                dt = time.time() - ts; secs += dt; npred += len(sub)
                for (smi, pid), o in zip(sub, outs):
                    if o is None or len(o[0]) == 0:
                        fail_pid.append(pid); fail_why.append(f'prediction failed at CE {ce}')
                        continue
                    pids.append(pid); ces.append(ce); lens.append(len(o[0])); mzs.append(o[0]); its.append(o[1])
                share.tick(dt)                                           # leave the rest of the GPU time to the desktop / the game
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        tmp = os.path.join(A.out, 'chunks', f'tmp_{c:05d}.npz')
        np.savez(tmp, chunk_pid=d.pid.values.astype(np.int64), pid=np.array(pids, np.int64), ce=np.array(ces, np.float32), off=off,
                 mz=np.concatenate(mzs).astype(np.float32) if mzs else np.zeros(0, np.float32),
                 it=np.concatenate(its).astype(np.float32) if its else np.zeros(0, np.float32),
                 fail_pid=np.array(fail_pid, np.int64), fail_why=np.array(fail_why, dtype='U200'), seconds=secs, n_pred=npred)
        os.replace(tmp, chunk_path(c))
        n_run += npred
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == 'cuda' else 0.0
        log(f'chunk {c + 1}/{n_chunks}: {len(d)} structures, {len(pids)} spectra, {len(set(fail_pid))} failed, {npred / max(secs, 1e-9):.1f} pred/s on the GPU, '
            f'{n_run / (time.time() - t_run):.1f} pred/s wall, duty {share.duty:.2f}, gpu-mem {mem:.1f} GB')
    log('ALL_DONE')


if __name__ == '__main__':
    main()
