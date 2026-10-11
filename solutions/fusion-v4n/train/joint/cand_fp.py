"""Fingerprints of the benchmark candidates that are not pool rows (joined PubChem rows, generated structures), for the fingerprint variant.

The pool's fingerprint definition (`casmi.chem.raw_fingerprint`, the 10,226 selected bits of fp_bits.npy, packed) so that pool rows and
other candidates are encoded identically. Chunked and resumable; at most --workers processes at below-normal priority.
out: $CASMI_DATA/joint/bench/candfp_<half>.npz (cand ids, packed fingerprints; an unparseable SMILES gets all zeros and ok = False)
usage: python train/joint/cand_fp.py --half dev [--workers 4]
"""
import argparse, glob, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jpaths as P

_W = {}


def _init():
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    sys.path.insert(0, P.CODE)
    from casmi import chem
    _W['chem'] = chem; _W['bits'] = np.load(P.WORK + '/fp_bits.npy')


def _chunk(task):
    path, ids, smiles = task
    if os.path.exists(path):
        return 0
    nb = (len(_W['bits']) + 7) // 8; fp = np.zeros((len(ids), nb), np.uint8); ok = np.zeros(len(ids), bool)
    for i, s in enumerate(smiles):
        f = _W['chem'].raw_fingerprint(s)
        if f is not None:
            fp[i] = np.packbits(f[_W['bits']]); ok[i] = True
    np.savez(path + '.tmp.npz', cand=ids, fp=fp, ok=ok); os.replace(path + '.tmp.npz', path); return len(ids)


def load(half):
    """(sorted cand ids, packed fingerprints, ok) of the non-pool candidates of a half."""
    z = np.load(f'{P.BENCH}/candfp_{half}.npz'); return z['cand'], z['fp'], z['ok']


if __name__ == '__main__':
    import multiprocessing as mp
    import bench
    ap = argparse.ArgumentParser(); ap.add_argument('--half', default='dev'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--chunk', type=int, default=2000); a = ap.parse_args()
    out = f'{P.BENCH}/candfp_{a.half}.npz'
    if os.path.exists(out):
        sys.exit('exists: ' + out)
    p = bench.Bench().pairs(a.half); p = p[p.pid < 0].drop_duplicates('cand').sort_values('cand'); d = f'{P.BENCH}/candfp_{a.half}_parts'; os.makedirs(d, exist_ok=True)
    tasks = [(f'{d}/c{i:05d}.npz', p.cand.values[i:i + a.chunk], list(p.smiles.values[i:i + a.chunk])) for i in range(0, len(p), a.chunk)]
    t0 = time.time(); done = 0
    with mp.Pool(a.workers, initializer=_init) as pool:
        for n in pool.imap_unordered(_chunk, tasks):
            done += n
    cand, fp, ok = [], [], []
    for t in tasks:
        with np.load(t[0]) as z:
            cand.append(z['cand']); fp.append(z['fp']); ok.append(z['ok'])
    cand = np.concatenate(cand); ok = np.concatenate(ok); assert (cand == p.cand.values).all()
    np.savez(out + '.tmp.npz', cand=cand, fp=np.concatenate(fp), ok=ok); os.replace(out + '.tmp.npz', out)
    for f in glob.glob(d + '/*.npz'):
        os.remove(f)
    os.rmdir(d); print(f'{len(cand)} candidates, {done} computed now in {time.time() - t0:.0f}s ({a.workers} workers), unparseable {int((~ok).sum())}', flush=True)
