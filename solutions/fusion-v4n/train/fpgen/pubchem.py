"""PubChem natural-product-like subset of fpgen (the training-set composition arm): draw rows of the PubChem tier flagged as natural products, run the chemistry pass and their pool-bit fingerprints, and apply the same leak removals.

Selection: `pc_np > 0` and monoisotopic mass between --mmin and --mmax, a salted-hash sample of --n rows, deduplicated by stereo-free
canonical SMILES against each other and against the pool. Removed afterwards, exactly as for the pool: identity or tautomer SMILES / first
block of an evaluation molecule, the generic Murcko scaffold of a scaffold-hold-out molecule, a FINAL-half benchmark key.
out: $FPGEN_WORK/chem/pc.parquet (can, rnd, nH), pc_fp.npy (packed pool bits).   usage: python pubchem.py [--n 400000] [--workers 5]
"""
import argparse, hashlib, json, os, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpaths as P, tok as T, chemrows as CR, sets as S

SALT = 'fpgen-pc-1'
_W = {}


def _init():
    CR._init(); sys.path.insert(0, P.CODE)
    from casmi import chem
    _W['chem'] = chem; _W['bits'] = np.load(P.WORK + '/fp_bits.npy')


def _fp_chunk(task):
    path, smis = task
    if os.path.exists(path):
        return 0
    if not _W:
        _init()
    chem = _W['chem']; b = _W['bits']; nb = (len(b) + 7) // 8
    out = np.zeros((len(smis), nb), np.uint8)
    for i, s in enumerate(smis):
        f = chem.raw_fingerprint(s)
        if f is not None:
            out[i] = np.packbits(f[b])
    np.save(path + '.tmp.npy', out); os.replace(path + '.tmp.npy', path); return len(smis)


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=400000); ap.add_argument('--mode', default='np', choices=['np', 'mix']); ap.add_argument('--mmin', type=float, default=150.0)
    ap.add_argument('--mmax', type=float, default=1200.0); ap.add_argument('--workers', type=int, default=5); a = ap.parse_args()
    P.mk(); P.low_priority(); t0 = time.time()
    name = 'pc' if a.mode == 'np' else 'pc_mix'
    if os.path.exists('%s/%s.parquet' % (P.CHEM, name)) and os.path.exists('%s/%s_fp.npy' % (P.CHEM, name)):
        sys.exit('exists')
    mass = np.load(P.TIER + '/pc_mass.npy', mmap_mode='r'); off = np.load(P.TIER + '/pc_off.npy', mmap_mode='r'); buf = np.load(P.TIER + '/pc_smiles.npy', mmap_mode='r')
    npf = np.load(P.PCNP, mmap_mode='r')
    lo, hi = np.searchsorted(mass, [a.mmin, a.mmax])
    flag = np.asarray(npf[lo:hi]) > 0; ids = np.arange(lo, hi)[flag if a.mode == 'np' else slice(None)]
    print('PubChem tier: %d rows %s between %.0f and %.0f Da' % (len(ids), 'flagged natural product' if a.mode == 'np' else 'in the mass range', a.mmin, a.mmax), flush=True)
    rng = np.random.default_rng(int(hashlib.md5(SALT.encode()).hexdigest()[:8], 16))
    ids = np.sort(rng.choice(ids, min(a.n, len(ids)), replace=False))
    smis = [bytes(buf[off[i]:off[i + 1]]).decode() for i in ids]
    print('read %d SMILES, %.0fs' % (len(smis), time.time() - t0), flush=True)
    D = CR.run('%s_raw' % name, ids, smis, workers=a.workers)
    D = D[D.ok & (D.can != '')].reset_index(drop=True)
    PC = pd.read_parquet(P.CHEM + '/pool.parquet'); M, _ = S.load()
    tsmi = set(); tik = set()
    for r in M.itertuples():
        tsmi |= set(r.tsmi.split('|')); tik |= set(r.tik.split('|'))
    B = pd.read_parquet(P.BENCH + '/mols.parquet')
    bad = D.can.isin(tsmi) | D.ik.isin(tik) | D.scaf.isin(set(M.scaf[(M.eset == 'SH') & (M.scaf != '')]))
    n_scaf = int(D.scaf.isin(set(M.scaf[(M.eset == 'SH') & (M.scaf != '')])).sum())
    D = D[~bad]
    D = D[~D.can.isin(set(PC.can[PC.ok]))].drop_duplicates('can').reset_index(drop=True)
    print('after the removals and the deduplication against the pool: %d structures (%d removed as an evaluation identity or scaffold, of them %d by scaffold)'
          % (len(D), int(bad.sum()), n_scaf), flush=True)
    with mp.Pool(a.workers, initializer=_init) as pool:
        d = '%s/%s_fp_parts' % (P.CHEM, name); os.makedirs(d, exist_ok=True)
        tasks = [('%s/f%06d.npy' % (d, i), list(D.can.values[i:i + 5000])) for i in range(0, len(D), 5000)]
        for k, _ in enumerate(pool.imap_unordered(_fp_chunk, tasks)):
            if k % 10 == 0:
                print('fingerprints %d/%d %.0fs' % (k, len(tasks), time.time() - t0), flush=True)
    FP = np.concatenate([np.load(t[0]) for t in tasks]); np.save('%s/%s_fp.npy' % (P.CHEM, name), FP)
    for t in tasks:
        os.remove(t[0])
    os.rmdir(d)
    hs = []
    from rdkit import Chem
    for s in D.can.values:
        m = Chem.MolFromSmiles(s); hs.append(sum(x.GetTotalNumHs() for x in m.GetAtoms()) if m is not None else 0)
    D['nH'] = hs; D[['rid', 'can', 'rnd', 'nH']].to_parquet('%s/%s.parquet' % (P.CHEM, name), index=False)
    json.dump(dict(mode=a.mode, rows=len(D), drawn=int(a.n), bits=FP.shape[1] * 8, mmin=a.mmin, mmax=a.mmax, wall_s=round(time.time() - t0)), open('%s/%s.json' % (P.CHEM, name), 'w'), indent=1)
    print('done: %d structures, %.0f s' % (len(D), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
