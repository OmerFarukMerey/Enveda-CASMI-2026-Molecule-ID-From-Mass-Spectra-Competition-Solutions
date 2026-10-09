"""The joint model as SELECTOR of PubChem-only candidates: recall of the answer when the stored deep PubChem lists of DEV are re-ranked by the model.

Specification from reviews/budget-2026-10-10.md, section 4.1: among the DEV molecules whose answer has a PubChem record within +-10 ppm
(1,215 `record answers`; the answer was removed from the pool), the current selection (fold-safe FPNet f.z over an ECFP4 shortlist) holds the
answer in its first 25 / 100 rows for 0.643 / 0.798 of them; targets 0.75 / 0.88. What can be measured without fingerprinting whole windows:
the model re-ranks the stored first 200 rows of that selection (ceiling 0.842). Orders reported: the stored selection (must reproduce 0.643 /
0.798), the model alone, and the rank average of both. Runs on the CPU (no GPU contention with a trainer).

  build   candidate table and fingerprints of the deep lists of the DEV record answers -> <dir>/sel_dev.parquet, sel_fp.npy
  score   R@1 / 5 / 10 / 25 / 50 / 100 / 200 of a checkpoint -> <run folder>/sel_dev.json
usage: python train/joint/selbench.py --build --dir scratch/joint1/sel [--workers 3]
       python train/joint/selbench.py --ckpt <run>/last.pt --dir scratch/joint1/sel
"""
import argparse, json, os, sys, time

import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import jpaths as P

DEEP = ['pcjoin/pc', 'trainplan/pc', 'panel2/join/ctrl/pc', 'rankbank3/hold2/pc']
KS = (1, 5, 10, 25, 50, 100, 200)


def build(d, workers=3):
    import multiprocessing as mp
    import cand_fp
    os.makedirs(d, exist_ok=True); D = pd.read_parquet(P.DATA + '/budget/dev_drop.parquet').reset_index(); rec = set(D.mid[D.in_tier.astype(bool)]); rows = {}
    for f in DEEP:
        for ln in open(f'{P.DATA}/{f}/rows.jsonl'):
            try:
                x = json.loads(ln)
            except Exception:
                continue
            if x['mid'] in rec:
                rows[x['mid']] = [(x['mid'], i, r[0], float(r[1]), r[2], bool(r[3])) for i, r in enumerate(x['list'])]
    T = pd.DataFrame([r for v in rows.values() for r in v], columns=['mid', 'pos', 'smiles', 'sel_score', 'ik14', 'y'])
    us = T.smiles.drop_duplicates().values; T['ci'] = pd.Series(np.arange(len(us)), index=us).reindex(T.smiles.values).values
    parts = f'{d}/parts'; os.makedirs(parts, exist_ok=True)
    tasks = [(f'{parts}/c{i:05d}.npz', np.arange(i, min(i + 2000, len(us))), list(us[i:i + 2000])) for i in range(0, len(us), 2000)]; t0 = time.time()
    with mp.Pool(workers, initializer=cand_fp._init) as pool:
        n = sum(pool.imap_unordered(cand_fp._chunk, tasks))
    fp = np.concatenate([np.load(t[0])['fp'] for t in tasks]); ok = np.concatenate([np.load(t[0])['ok'] for t in tasks])
    np.save(d + '/sel_fp.npy', fp); T.to_parquet(d + '/sel_dev.parquet', index=False)
    for t in tasks:
        os.remove(t[0])
    os.rmdir(parts)
    print(json.dumps(dict(record_answers=len(rec), with_list=len(rows), rows=len(T), distinct_smiles=len(us), computed=int(n), unparseable=int((~ok).sum()), answer_in_list=int(T.groupby('mid').y.any().sum()), s=time.time() - t0)))


def recall(T, score, n_rec, by=None):
    """R@k of the answer per molecule when the rows are ordered by descending `score` (first row of an InChIKey block counts)."""
    T = T.assign(s=score).sort_values(['mid', 's'], ascending=[True, False], kind='stable').drop_duplicates(['mid', 'ik14'])
    T['rank'] = T.groupby('mid').cumcount() + 1; r = T[T.y].groupby('mid')['rank'].min()
    return {f'R@{k}': float((r <= k).sum() / n_rec) for k in KS}, r


def score(ckpt, d, threads=2, out=''):
    import torch
    import data as JD, train_joint as TJ
    torch.set_num_threads(threads); ck = torch.load(ckpt, map_location='cpu', weights_only=False); a = argparse.Namespace(**ck['args']); TJ.model_args_from(a, ck['args'])
    m, _ = TJ.build(a, 'cpu'); m.load_state_dict(ck['model']); m.eval(); T = pd.read_parquet(d + '/sel_dev.parquet'); fp = np.load(d + '/sel_fp.npy', mmap_mode='r')
    import bench
    DD = pd.read_parquet(P.DATA + '/budget/dev_drop.parquet').reset_index(); rec = DD[DD.in_tier.astype(bool)]; n_rec = len(rec)
    S = pd.read_parquet(P.BENCH + '/spectra.parquet'); mids = np.array(sorted(T.mid.unique())); mix = {x: i for i, x in enumerate(mids)}; S = S[S.mid.isin(mix)]
    rows = S.row.values; own = S.mid.map(mix).values; o = np.argsort(own, kind='stable'); D = JD.JointData(); t0 = time.time()
    with torch.no_grad():
        zm = TJ.embed_molecules(m, D, rows[o], own[o], len(mids), a, 'cpu'); E = torch.cat([m.struct(torch.as_tensor(np.asarray(fp[s:s + 8192]))) for s in range(0, len(fp), 8192)])
        sc = (zm[torch.as_tensor(T.mid.map(mix).values)] * E[torch.as_tensor(T.ci.values)]).sum(1).numpy()
    res = dict(n_record_answers=n_rec, molecules_with_list=int(len(mids)), step=ck['step'], s=time.time() - t0)
    res['stored_selection'], r0 = recall(T, -T.pos.values.astype(float), n_rec); res['model'], r1 = recall(T, sc, n_rec)
    T2 = T.assign(s=sc); rk = T2.groupby('mid').s.rank(ascending=False, method='first').values; res['rank_average'], r2 = recall(T, -(rk + T.pos.values + 1.0), n_rec)
    for lab, col in (('timsTOF', 'tims'), ('panel', 'panel'), ('nitrogen-free', 'nfree'), ('gate open (lib_max < 0.7)', None)):
        mm = set(rec.mid[rec[col].astype(bool)] if col else rec.mid[rec.lib_max < 0.7]); n = len(mm)
        res[lab] = dict(n=n, stored={f'R@{k}': float((r0[r0.index.isin(mm)] <= k).sum() / max(n, 1)) for k in (25, 100)}, model={f'R@{k}': float((r1[r1.index.isin(mm)] <= k).sum() / max(n, 1)) for k in (25, 100)},
                        rank_average={f'R@{k}': float((r2[r2.index.isin(mm)] <= k).sum() / max(n, 1)) for k in (25, 100)})
    o = out or os.path.dirname(ckpt) + '/sel_dev.json'; json.dump(res, open(o, 'w'), indent=1); print(json.dumps({k: res[k] for k in ('stored_selection', 'model', 'rank_average')})); return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--build', action='store_true'); ap.add_argument('--dir', required=True); ap.add_argument('--ckpt', default=''); ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--out', default=''); a = ap.parse_args()
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if a.build:
        build(a.dir, a.workers)
    if a.ckpt:
        score(a.ckpt, a.dir, a.threads, a.out)
