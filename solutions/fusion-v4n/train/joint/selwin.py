"""Whole-window selector test on a sub-sample: the joint model ranks EVERY PubChem record within +-10 ppm of the answer's mass (no ECFP4 shortlist).

Sub-sample of the DEV record answers of reviews/budget-2026-10-10.md (answer removed from the pool, a PubChem record exists). For each sampled
molecule all tier rows of the mass window are fingerprinted (the pool's definition, 2.55 ms per row) and keyed by InChIKey first block; the
answer's rows are those sharing a first block with the answer's SMILES or with the truth rows of the stored deep list. Pool rows are NOT
removed from the window (a few rows per window), so the recall is slightly pessimistic. Reported: R@k of the model over the whole window, of
the stored selection on the same molecules, on the CPU.

  build   python train/joint/selwin.py --build --dir scratch/joint1/selwin [--n 150] [--workers 2]     (resumable per molecule)
  score   python train/joint/selwin.py --ckpt <run>/last.pt --dir scratch/joint1/selwin
"""
import argparse, json, os, sys, time

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import jpaths as P

KS = (1, 5, 10, 25, 50, 100, 200, 500)
_W = {}


def _init():
    import cand_fp
    cand_fp._init(); _W.update(cand_fp._W)
    _W['mass'] = np.load(P.TIER + '/pc_mass.npy', mmap_mode='r'); _W['off'] = np.load(P.TIER + '/pc_off.npy', mmap_mode='r'); _W['smi'] = np.load(P.TIER + '/pc_smiles.npy', mmap_mode='r')


def ik14(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    try:
        return Chem.MolToInchiKey(m)[:14] if m is not None else ''
    except Exception:
        return ''


def _mol(task):
    path, mid, mass, smiles = task
    if os.path.exists(path):
        return 0
    lo = int(np.searchsorted(_W['mass'], mass * (1 - 1e-5))); hi = int(np.searchsorted(_W['mass'], mass * (1 + 1e-5))); nb = (len(_W['bits']) + 7) // 8
    fp = np.zeros((hi - lo, nb), np.uint8); keys = []
    for i, r in enumerate(range(lo, hi)):
        s = bytes(_W['smi'][int(_W['off'][r]):int(_W['off'][r + 1])]).decode('ascii'); f = _W['chem'].raw_fingerprint(s); keys.append(ik14(s))
        if f is not None:
            fp[i] = np.packbits(f[_W['bits']])
    np.savez(path + '.tmp.npz', fp=fp, ik=np.array(keys), lo=lo, answer_ik=ik14(smiles)); os.replace(path + '.tmp.npz', path); return hi - lo


def sample(n):
    D = pd.read_parquet(P.DATA + '/budget/dev_drop.parquet').reset_index(); D = D[D.in_tier.astype(bool)].sort_values('mid')
    M = pd.read_parquet(P.BENCH + '/mols.parquet').set_index('mid'); D = D.iloc[np.sort(np.random.default_rng(0).permutation(len(D))[:n])].copy()
    D['mass'] = M.mass.reindex(D.mid).values; D['smiles'] = M.smiles.reindex(D.mid).values; return D


def build(d, n, workers):
    import multiprocessing as mp
    os.makedirs(d, exist_ok=True); D = sample(n); t0 = time.time()
    with mp.Pool(workers, initializer=_init) as pool:
        tot = sum(pool.imap_unordered(_mol, [(f'{d}/{r.mid}.npz', r.mid, float(r.mass), r.smiles) for r in D.itertuples()]))
    print(json.dumps(dict(molecules=len(D), rows_computed=int(tot), s=time.time() - t0)))


def score(ckpt, d, n, threads=2, out=''):
    import torch
    import data as JD, train_joint as TJ
    torch.set_num_threads(threads); ms = []
    for c in ckpt.split(','):                                                    # several checkpoints: the mean of their scores
        ck = torch.load(c, map_location='cpu', weights_only=False); a = argparse.Namespace(**ck['args']); TJ.model_args_from(a, ck['args']); m, _ = TJ.build(a, 'cpu'); m.load_state_dict(ck['model']); m.eval(); ms.append((m, a))
    ckpt = ckpt.split(',')[0]; D = sample(n); D = D[[os.path.exists(f'{d}/{x}.npz') for x in D.mid]]
    T = pd.read_parquet(os.path.dirname(d.rstrip('/')) + '/sel/sel_dev.parquet'); truth = T[T.y].groupby('mid').ik14.agg(set).to_dict()
    S = pd.read_parquet(P.BENCH + '/spectra.parquet'); JDd = JD.JointData(); rk = []; found = []; nwin = []
    with torch.no_grad():
        for r in D.itertuples():
            z = np.load(f'{d}/{r.mid}.npz'); rows = S.row.values[S.mid.values == r.mid]; fp = z['fp']; sc = 0.0
            for m, a in ms:
                zm = TJ.embed_molecules(m, JDd, np.sort(rows), np.zeros(len(rows), np.int64), 1, a, 'cpu')
                sc = sc + (torch.cat([m.struct(torch.as_tensor(fp[s:s + 8192])) for s in range(0, len(fp), 8192)]) @ zm[0]).numpy() / len(ms)
            ok = truth.get(r.mid, set()) | {str(z['answer_ik'])}; ik = z['ik']; y = np.isin(ik, list(ok)); nwin.append(len(ik)); found.append(bool(y.any()))
            if not y.any():
                rk.append(0); continue
            o = np.argsort(-sc, kind='stable'); _, first = np.unique(ik[o], return_index=True); oo = o[np.sort(first)]; rk.append(int(np.flatnonzero(y[oo])[0]) + 1)
    rk = np.array(rk); n = len(D); pos = D.pos.values
    res = dict(molecules=n, answer_found_in_window=int(np.sum(found)), window_rows_median=float(np.median(nwin)), step=ck['step'], checkpoints=len(ms),
               model_whole_window={f'R@{k}': float(((rk > 0) & (rk <= k)).mean()) for k in KS}, stored_selection={f'R@{k}': float(((pos > 0) & (pos <= k)).mean()) for k in KS if k <= 200},
               se_at_half=float(0.5 / np.sqrt(n)))
    o = out or os.path.dirname(ckpt) + '/selwin_dev.json'; json.dump(res, open(o, 'w'), indent=1); print(json.dumps(res)); return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--build', action='store_true'); ap.add_argument('--dir', required=True); ap.add_argument('--ckpt', default=''); ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--threads', type=int, default=2); ap.add_argument('--out', default=''); a = ap.parse_args()
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if a.build:
        build(a.dir, a.n, a.workers)
    if a.ckpt:
        score(a.ckpt, a.dir, a.n, a.threads, a.out)
