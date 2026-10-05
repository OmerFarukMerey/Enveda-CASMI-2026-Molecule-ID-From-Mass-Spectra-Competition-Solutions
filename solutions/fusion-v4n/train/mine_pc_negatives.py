"""Mine PubChem mass-window negatives for the FPNet ranking loss (train_fpnet.py --pc_neg): hard and random isomers.

For every training structure (pool entry of a structure whose fold is not held out) the PubChem tier rows within
+-ppm of its exact mass are sampled (at most --scan rows per formula, one row per InChIKey14), and two kinds of
negatives are kept:
  hard    the --hard rows with the highest ECFP4 Tanimoto similarity to the true structure (not the structure itself:
          rows with one of its InChIKey14s or an identical ECFP4 are dropped)
  random  --rand rows drawn uniformly from the window sample, shared by the structures of one formula
Only the fingerprints (the model's selected bits, packed) and tier row indices are stored. Work is done per formula in
chunks, resumable (one npz per chunk); --assemble writes the tables train_fpnet.py reads:
  fp.npy (U, nbytes) uint8   packed selected-bit fingerprints of the unique tier rows
  pid.npy (S,)               pool index of the structure
  hard.npy (S, hard) int32   rows of fp.npy, -1 = none;  rand.npy (S, rand) int32 likewise
  stats.parquet              per structure: window size, sample size, similarity and same-formula share of the negatives
Every --keycheck-th formula also computes the metric key of its hard negatives, to report how many are the true
structure under the metric (tautomer twins), which the cheap InChIKey14 test cannot see.

usage: python train/mine_pc_negatives.py --work <run>/work --code <dir with casmi/> --tier <tier dir> --ik14 <pc_ik14.npy> \
           --out D:/casmi-data/hardneg/w10 [--holdout hold,np | --only hold,np] [--workers 2];  then the same with --assemble
"""
import argparse, glob, os, sys, time

import numpy as np

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
_W = {}


def below_normal():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def init_worker(a):
    below_normal()
    sys.path.insert(0, a.code)
    from casmi import chem
    _W['a'] = a; _W['chem'] = chem
    for n in ('pc_mass', 'pc_off', 'pc_smiles'):
        _W[n] = np.load(os.path.join(a.tier, n + '.npy'), mmap_mode='r')
    _W['ik'] = np.load(a.ik14, mmap_mode='r')
    _W['bits'] = np.load(os.path.join(a.work, 'fp_bits.npy'))
    r = chem._rdkit()
    _W['gen'] = r['gen'].GetMorganGenerator(radius=2, fpSize=4096); _W['Chem'] = r['Chem']


def _smiles(row):
    o0, o1 = int(_W['pc_off'][row]), int(_W['pc_off'][row + 1])
    return bytes(_W['pc_smiles'][o0:o1]).decode('ascii')


def _ecfp(smi):
    m = _W['Chem'].MolFromSmiles(smi)
    return None if m is None else _W['gen'].GetFingerprintAsNumPy(m).astype(np.float32)


def mine_chunk(task):
    """task = (chunk id, [(group id, mass, [(pid, smiles, [ik14 bytes])])]). Writes chunk_<id>.npz, returns a summary."""
    cid, groups = task
    a = _W['a']; chem = _W['chem']; bits = _W['bits']; t0 = time.time()
    rows_all, fps = [], []                               # unique tier rows of the chunk and their packed fingerprints
    row_ix = {}
    S = dict(pid=[], hard=[], rand=[], tan=[], hsame=[], rsame=[], n_window=[], n_scan=[], twin=[], gid=[])

    def fp_index(row, smi):
        if row in row_ix:
            return row_ix[row]
        f = chem.raw_fingerprint(smi)
        i = -1
        if f is not None:
            i = len(rows_all); rows_all.append(row); fps.append(np.packbits(f[bits]))
        row_ix[row] = i
        return i

    for gid, mass, structs in groups:
        rng = np.random.default_rng([a.seed, gid])
        tol = mass * a.ppm * 1e-6
        lo = int(np.searchsorted(_W['pc_mass'], mass - tol, 'left')); hi = int(np.searchsorted(_W['pc_mass'], mass + tol, 'right'))
        n = hi - lo
        rows = np.arange(lo, hi) if n <= a.scan else lo + np.sort(rng.choice(n, a.scan, replace=False))
        ik = np.asarray(_W['ik'][rows]) if n else np.zeros(0, 'S14')
        _, first = np.unique(ik, return_index=True); keep = np.sort(first)          # one row per InChIKey14
        rows, ik = rows[keep], ik[keep]
        smis = [_smiles(int(r)) for r in rows]
        E = [_ecfp(s) for s in smis]
        ok = np.array([e is not None for e in E], bool)
        rows, ik = rows[ok], ik[ok]; smis = [s for s, o in zip(smis, ok) if o]
        E = np.stack([e for e in E if e is not None]) if ok.any() else np.zeros((0, 4096), np.float32)
        same = np.abs(np.asarray(_W['pc_mass'][rows]) - mass) < 3e-5 if len(rows) else np.zeros(0, bool)
        ne = E.sum(1)
        perm = rng.permutation(len(rows))[:a.rand]                                  # the formula's random negatives
        r_idx = np.array([fp_index(int(rows[j]), smis[j]) for j in perm], np.int64)
        do_key = a.keycheck > 0 and gid % a.keycheck == 0
        for pid, smi, iks in structs:
            hard = np.full(a.hard, -1, np.int64); tan_h = np.full(a.hard, np.nan, np.float32); hs = np.zeros(a.hard, bool)
            rr = np.full(a.rand, -1, np.int64); rs = np.zeros(a.rand, bool); twin = -1
            own = np.isin(ik, np.array(sorted(iks), 'S14')) if len(rows) else np.zeros(0, bool)
            t = _ecfp(smi)
            if len(rows) and t is not None:
                inter = E @ t; tan = inter / np.maximum(ne + t.sum() - inter, 1.0)
                bad = own | (tan >= 0.999)
                order = np.argsort(-np.where(bad, -1.0, tan), kind='stable')
                k = 0; twin = 0 if do_key else -1
                tkey = chem.score_key(smi) if do_key else None
                for j in order:
                    if bad[j] or k >= a.hard:
                        break
                    i = fp_index(int(rows[j]), smis[j])
                    if i < 0:
                        continue
                    if do_key and chem.score_key(smis[j]) == tkey:
                        twin += 1
                    hard[k] = i; tan_h[k] = tan[j]; hs[k] = same[j]; k += 1
                m = ~bad[perm]
                rr[:len(perm)] = np.where(m, r_idx, -1); rs[:len(perm)] = same[perm] & m & (r_idx >= 0)
            S['pid'].append(pid); S['hard'].append(hard); S['rand'].append(rr); S['tan'].append(tan_h); S['hsame'].append(hs)
            S['rsame'].append(rs); S['n_window'].append(n); S['n_scan'].append(len(rows)); S['twin'].append(twin); S['gid'].append(gid)
    nb = (len(bits) + 7) // 8
    out = os.path.join(a.out, f'chunk_{cid:05d}.npz')
    np.savez(out + '.tmp.npz', rows=np.array(rows_all, np.int64), fp=np.stack(fps) if fps else np.zeros((0, nb), np.uint8),
             **{k: np.array(v) for k, v in S.items()})
    os.replace(out + '.tmp.npz', out)
    return cid, len(S['pid']), len(rows_all), time.time() - t0


def structures(a):
    """Training structures by formula group: [(group id, mass, [(pid, smiles, {ik14})])], lightest first."""
    import pandas as pd
    st = pd.read_parquet(os.path.join(a.work, 'train_structs.parquet'), columns=['sid', 'key', 'inchikey14'])
    sp = pd.read_parquet(os.path.join(a.work, 'split.parquet'))
    pool = pd.read_parquet(os.path.join(a.work, 'pool_meta.parquet'), columns=['key', 'smiles', 'mass'])
    fold = np.array(['train'] * len(st), dtype=object); fold[sp.sid.values] = sp.fold.values
    hold = [h for h in a.holdout.split(',') if h]
    st = st[np.isin(fold, a.only.split(',')) if a.only else ~np.isin(fold, hold)]
    k2p = dict(zip(pool.key.values, range(len(pool))))
    st = st.assign(pid=[k2p.get(k, -1) for k in st.key.values]); st = st[st.pid >= 0]
    iks = st.groupby('pid').inchikey14.agg(lambda s: set(s)).to_dict()
    pids = np.array(sorted(iks)); mass = pool.mass.values[pids].astype(np.float64)
    gkey = np.round(mass, 5); um, inv = np.unique(gkey, return_inverse=True)
    groups = [(g, float(um[g]), []) for g in range(len(um))]
    for p, g in zip(pids.tolist(), inv.tolist()):
        groups[g][2].append((p, pool.smiles.values[p], {x.encode() for x in iks[p]} | {pool.key.values[p].encode()}))
    return groups


def assemble(a):
    import pandas as pd
    from numpy.lib.format import open_memmap
    files = sorted(glob.glob(os.path.join(a.out, 'chunk_*.npz')))
    nrow = [len(np.load(f)['rows']) for f in files]; base = np.concatenate([[0], np.cumsum(nrow)])
    nb = np.load(files[0])['fp'].shape[1]
    FP = open_memmap(os.path.join(a.out, 'fp.npy'), mode='w+', dtype=np.uint8, shape=(int(base[-1]), nb))
    P = {k: [] for k in ('pid', 'hard', 'rand', 'tan', 'hsame', 'rsame', 'n_window', 'n_scan', 'twin', 'gid')}; rows = []
    for f, b in zip(files, base[:-1]):
        z = np.load(f)
        FP[b:b + len(z['rows'])] = z['fp']; rows.append(z['rows'])
        for k in P:
            v = z[k]
            if k in ('hard', 'rand'):
                v = np.where(v >= 0, v + b, -1)
            P[k].append(v)
    FP.flush(); del FP
    P = {k: np.concatenate(v) for k, v in P.items()}
    np.save(os.path.join(a.out, 'rows.npy'), np.concatenate(rows)); np.save(os.path.join(a.out, 'pid.npy'), P['pid'].astype(np.int64))
    np.save(os.path.join(a.out, 'hard.npy'), P['hard'].astype(np.int32)); np.save(os.path.join(a.out, 'rand.npy'), P['rand'].astype(np.int32))
    hv = P['hard'] >= 0; rv = P['rand'] >= 0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)                             # structures without any negative
        S = pd.DataFrame(dict(pid=P['pid'], gid=P['gid'], n_window=P['n_window'], n_scan=P['n_scan'], n_hard=hv.sum(1), n_rand=rv.sum(1),
                              tan_max=np.nanmax(np.where(hv, P['tan'], np.nan), 1), tan_mean=np.nanmean(np.where(hv, P['tan'], np.nan), 1),
                              tan_min=np.nanmin(np.where(hv, P['tan'], np.nan), 1),
                              hard_same=(P['hsame'] & hv).sum(1), rand_same=(P['rsame'] & rv).sum(1), twin=P['twin']))
    S.to_parquet(os.path.join(a.out, 'stats.parquet'), index=False)
    q = lambda x: ' / '.join(f'{v:.3g}' for v in np.nanpercentile(x, [10, 50, 90]))
    kc = S[S.twin >= 0]
    print(f'structures {len(S):,} in {S.gid.nunique():,} formulas; unique tier rows {int(base[-1]):,} ({base[-1] * nb / 2**30:.1f} GB)')
    print(f'window rows (10 / 50 / 90%): {q(S.n_window)}; mean {S.n_window.mean():.0f}; scanned sample (one row per InChIKey14) {q(S.n_scan)}')
    print(f'hard negatives per structure: mean {S.n_hard.mean():.1f}; none {np.mean(S.n_hard == 0):.3%}; random: mean {S.n_rand.mean():.1f}; none {np.mean(S.n_rand == 0):.3%}')
    print(f'ECFP4 Tanimoto of the hard negatives: hardest {q(S.tan_max)}; mean of the {a.hard} {q(S.tan_mean)}; weakest {q(S.tan_min)}')
    print(f'same formula as the answer: hard {S.hard_same.sum() / max(1, S.n_hard.sum()):.3f}; random {S.rand_same.sum() / max(1, S.n_rand.sum()):.3f}')
    print(f'metric-key check on {len(kc):,} structures: hard negatives with the answer\'s key {kc.twin.sum()} of {kc.n_hard.sum()} '
          f'({kc.twin.sum() / max(1, kc.n_hard.sum()):.3%}); structures with at least one {np.mean(kc.twin > 0):.3%}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True); ap.add_argument('--code', required=True); ap.add_argument('--tier', required=True)
    ap.add_argument('--ik14', required=True, help='InChIKey14 of the tier rows (row-aligned, S14)'); ap.add_argument('--out', required=True)
    ap.add_argument('--ppm', type=float, default=10.0); ap.add_argument('--scan', type=int, default=1024)
    ap.add_argument('--hard', type=int, default=16); ap.add_argument('--rand', type=int, default=32)
    ap.add_argument('--holdout', default='hold,np', help='folds whose structures get no negatives (empty = all structures)')
    ap.add_argument('--only', default=None, help='mine only the structures of these folds (e.g. hold,np: the complement of the default, '
                    'for a fine-tune on all structures; give train_fpnet.py both directories)')
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--chunk', type=int, default=300, help='formulas per chunk')
    ap.add_argument('--keycheck', type=int, default=20); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='only the first N chunks (smoke test)'); ap.add_argument('--assemble', action='store_true')
    a = ap.parse_args()
    below_normal(); os.makedirs(a.out, exist_ok=True)
    if a.assemble:
        return assemble(a)
    groups = structures(a)
    order = np.random.default_rng(a.seed).permutation(len(groups))                  # chunks of mixed masses: even run time
    chunks = [(c, [groups[g] for g in order[s:s + a.chunk]]) for c, s in enumerate(range(0, len(groups), a.chunk))]
    todo = [c for c in chunks if not os.path.exists(os.path.join(a.out, f'chunk_{c[0]:05d}.npz'))]
    if a.limit:
        todo = [c for c in todo if c[0] < a.limit]
    print(time.strftime('%H:%M:%S'), f'{sum(len(g[2]) for g in groups):,} structures, {len(groups):,} formulas, {len(chunks)} chunks, {len(todo)} to do', flush=True)
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers, initializer=init_worker, initargs=(a,)) as pool:
        for k, (cid, ns, nr, secs) in enumerate(pool.imap_unordered(mine_chunk, todo)):
            print(time.strftime('%H:%M:%S'), f'chunk {cid}: {ns} structures, {nr} rows, {secs:.0f}s | {k + 1}/{len(todo)} '
                  f'eta {(time.time() - t0) / (k + 1) * (len(todo) - k - 1) / 60:.0f} min', flush=True)
    print('MINED', flush=True)


if __name__ == '__main__':
    main()
