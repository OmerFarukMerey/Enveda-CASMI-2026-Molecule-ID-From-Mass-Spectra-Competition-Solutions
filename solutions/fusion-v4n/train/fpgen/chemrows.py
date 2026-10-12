"""Per-structure chemistry pass of fpgen: stereo-free canonical SMILES, randomised SMILES, generic Murcko scaffold, InChIKey first block (parallel, resumable)."""
import argparse, glob, os, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpaths as P, tok as T

R_RANDOM = 8
MAX_TOKENS = 160
_W = {}


def _init():
    P.low_priority(); P.watch_parent()
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    _W['Chem'] = Chem; _W['MS'] = MurckoScaffold


def scaffold(mol):
    """Generic Murcko scaffold SMILES ('' for an acyclic molecule or on failure)."""
    Chem, MS = _W['Chem'], _W['MS']
    try:
        s = MS.GetScaffoldForMol(mol)
        if s.GetNumAtoms() == 0:
            return ''
        return Chem.MolToSmiles(MS.MakeScaffoldGeneric(s))
    except Exception:
        return ''


def one(smiles, n_random=R_RANDOM):
    """-> dict(can, ik, scaf, rnd, ok); ok = parses, one component, tracked elements only, at most MAX_TOKENS tokens."""
    if not _W:
        _init()
    Chem = _W['Chem']; r = dict(can='', ik='', scaf='', rnd='', ok=False)
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return r
        Chem.RemoveStereochemistry(m)
        r['can'] = Chem.MolToSmiles(m, isomericSmiles=False); r['ik'] = (Chem.MolToInchiKey(m) or '')[:14]; r['scaf'] = scaffold(m)
        t = T.tokenize(r['can'])
        if t is None or not T.usable(t) or len(t) > MAX_TOKENS:
            return r
        out = []
        for _ in range(n_random):
            s = Chem.MolToSmiles(m, isomericSmiles=False, doRandom=True, canonical=False); ts = T.tokenize(s)
            if ts is not None and T.usable(ts) and len(ts) <= MAX_TOKENS + 20:
                out.append(s)
        r['rnd'] = ' '.join(out); r['ok'] = True
    except Exception:
        pass
    return r


def _chunk(task):
    path, ids, smiles, n_random = task
    if os.path.exists(path):
        return 0
    rows = [one(s, n_random) for s in smiles]
    d = pd.DataFrame(rows); d.insert(0, 'rid', ids)
    d.to_parquet(path + '.tmp', index=False); os.replace(path + '.tmp', path)
    return len(d)


def run(name, ids, smiles, workers=5, chunk=5000, n_random=R_RANDOM, log=print):
    """Chemistry pass over (ids, smiles) -> $FPGEN_WORK/chem/<name>.parquet (parts under <name>_parts while running)."""
    import multiprocessing as mp
    assert workers <= 5
    out = f'{P.CHEM}/{name}.parquet'
    if os.path.exists(out):
        return pd.read_parquet(out)
    d = f'{P.CHEM}/{name}_parts'; os.makedirs(d, exist_ok=True)
    tasks = [(f'{d}/c{i:07d}.parquet', np.asarray(ids[i:i + chunk]), list(smiles[i:i + chunk]), n_random) for i in range(0, len(ids), chunk)]
    t0 = time.time(); done = 0
    with mp.Pool(workers, initializer=_init) as pool:
        for k, n in enumerate(pool.imap_unordered(_chunk, tasks)):
            done += n
            if k % 20 == 0:
                log(f'chem {name}: {k}/{len(tasks)} chunks, {time.time() - t0:.0f}s', flush=True)
    D = pd.concat([pd.read_parquet(t[0]) for t in tasks], ignore_index=True)
    assert (D.rid.values == np.asarray(ids)).all()
    D.to_parquet(out + '.tmp', index=False); os.replace(out + '.tmp', out)
    for f in glob.glob(d + '/*.parquet'):
        os.remove(f)
    os.rmdir(d); log(f'chem {name}: {len(D)} rows, ok {int(D.ok.sum())}, {time.time() - t0:.0f}s', flush=True)
    return D


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=5); a = ap.parse_args()
    P.mk(); P.low_priority()
    M = pd.read_parquet(P.WORK + '/pool_meta.parquet', columns=['smiles'])
    run('pool', np.arange(len(M)), M.smiles.values, a.workers)
