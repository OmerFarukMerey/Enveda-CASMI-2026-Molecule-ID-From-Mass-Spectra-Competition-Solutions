"""Hard-negative tables of the ownneural model: for every positive pool row, its most similar same-formula pool isomers and its most similar
pool rows of ANOTHER formula inside +-10 ppm (the two kinds of candidate that s1's final ranks 2-25 consist of: 77-83% share the formula of
rank 1; the rest are near-mass rows).

Similarity = Tanimoto on the pool's packed selected fingerprint bits (pool_fp.npy). Banned pool rows (structures whose role in split.parquet
is not 'train', and every pool row sharing a metric key with them) are never drawn. Isomer groups larger than --cap are sub-sampled
(seeded) before ranking. Writes $CASMI_DATA/ownneural/negs.npz: pid [n] (positive pool rows), iso [n, M_ISO], win [n, M_WIN] (pool rows,
-1 = none), banned [n_pool] bool.

usage: python build_negs.py [--workers 4] [--cap 3000]
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import onpaths as P  # noqa: E402

M_ISO, M_WIN, PPM = 16, 8, 10.0
_W = {}


def _lowprio():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def as_u64(fp):
    """Packed uint8 [n, nbytes] -> uint64 [n, ceil(nbytes / 8)] (zero padded) for popcounts."""
    fp = np.asarray(fp, np.uint8); n, b = fp.shape; w = (b + 7) // 8
    out = np.zeros((n, w * 8), np.uint8); out[:, :b] = fp
    return out.view(np.uint64)


def tanimoto(q, X):
    """q [w] uint64, X [n, w] uint64 -> Tanimoto [n]."""
    inter = np.bitwise_count(X & q).sum(1); uni = np.bitwise_count(X | q).sum(1)
    return inter / np.maximum(uni, 1)


def top_similar(q, cand, X, m):
    """The m candidate rows most similar to q (descending), as an int32 array padded with -1."""
    out = np.full(m, -1, np.int32)
    if len(cand) == 0:
        return out
    t = tanimoto(q, X); k = min(m, len(cand)); o = np.argsort(-t, kind='stable')[:k]; out[:k] = cand[o]
    return out


def _init(cap):
    _lowprio()
    L = lambda n: np.load(f'{P.JIDX}/{n}.npy', mmap_mode='r')
    _W.update(fp=np.load(P.WORK + '/pool_fp.npy', mmap_mode='r'), form=np.asarray(L('pool_form')), fptr=np.asarray(L('form_ptr')), frows=np.asarray(L('form_rows')),
              mass=np.asarray(L('pool_mass')), banned=np.load(P.ON + '/banned.npy'), cap=cap)


def _job(args):
    i0, pids = args; W = _W; rng = np.random.default_rng(1000 + i0); fp = W['fp']
    iso_o = np.full((len(pids), M_ISO), -1, np.int32); win_o = np.full((len(pids), M_WIN), -1, np.int32)
    for r, p in enumerate(pids):
        q = as_u64(fp[p:p + 1])[0]; f = W['form'][p]
        iso = W['frows'][W['fptr'][f]:W['fptr'][f + 1]]; iso = iso[(iso != p) & ~W['banned'][iso]]
        if len(iso) > W['cap']:
            iso = np.sort(rng.choice(iso, W['cap'], replace=False))
        if len(iso):
            iso_o[r] = top_similar(q, iso, as_u64(fp[iso]), M_ISO)
        m = W['mass'][p]; lo = np.searchsorted(W['mass'], m * (1 - PPM * 1e-6)); hi = np.searchsorted(W['mass'], m * (1 + PPM * 1e-6))
        near = np.arange(lo, hi); near = near[(W['form'][lo:hi] != f) & ~W['banned'][lo:hi]]
        if len(near) > W['cap']:
            near = np.sort(rng.choice(near, W['cap'], replace=False))
        if len(near):
            win_o[r] = top_similar(q, near, as_u64(fp[near]), M_WIN)
    return i0, iso_o, win_o


def banned_rows(split, pool, st):
    """Pool rows that must never be drawn: the pool rows of non-'train' structures and every pool row with one of their metric keys."""
    bad_sid = split.sid.values[split.role.values != 'train']; bad_keys = set(st.key.values[bad_sid]) | set(st.inchikey14.values[bad_sid])
    b = pool.key.isin(bad_keys).values | pool.train_sid.isin(set(bad_sid)).values
    return b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--cap', type=int, default=3000); ap.add_argument('--chunk', type=int, default=2000)
    a = ap.parse_args(); _lowprio(); t0 = time.time()
    split = pd.read_parquet(P.ON + '/split.parquet'); st = pd.read_parquet(P.WORK + '/train_structs.parquet', columns=['sid', 'key', 'inchikey14'])
    pool = pd.read_parquet(P.WORK + '/pool_meta.parquet', columns=['key', 'train_sid']); sid_pid = np.load(P.JIDX + '/sid_pid.npy')
    banned = banned_rows(split, pool, st); np.save(P.ON + '/banned.npy', banned)
    use = split.role.isin(['train', 'val']).values & (sid_pid >= 0); pids = np.unique(sid_pid[use]).astype(np.int64)
    jobs = [(i, pids[i:i + a.chunk]) for i in range(0, len(pids), a.chunk)]
    iso = np.full((len(pids), M_ISO), -1, np.int32); win = np.full((len(pids), M_WIN), -1, np.int32); done = 0
    with Pool(a.workers, initializer=_init, initargs=(a.cap,)) as pl:
        for i0, io, wo in pl.imap_unordered(_job, jobs):
            iso[i0:i0 + len(io)] = io; win[i0:i0 + len(wo)] = wo; done += len(io)
            print(json.dumps(dict(done=done, of=len(pids), s=round(time.time() - t0))), flush=True)
    np.savez(P.ON + '/negs.tmp.npz', pid=pids, iso=iso, win=win); os.replace(P.ON + '/negs.tmp.npz', P.ON + '/negs.npz')
    st_ = dict(positives=len(pids), banned_pool_rows=int(banned.sum()), iso_mean=float((iso >= 0).sum(1).mean()), iso_none=float((iso[:, 0] < 0).mean()),
               win_mean=float((win >= 0).sum(1).mean()), win_none=float((win[:, 0] < 0).mean()), seconds=round(time.time() - t0))
    json.dump(st_, open(P.ON + '/negs_stats.json', 'w'), indent=1); print(json.dumps(st_))


if __name__ == '__main__':
    main()
