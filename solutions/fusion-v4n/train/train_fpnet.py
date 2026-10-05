"""Train or fine-tune the fingerprint network (FPNet) locally, with natural-product-weighted sampling and resume.

Derived from the public fold-safe training loop (dmitriigluzdov/casmi26-fold-safe-fpnet, train_fpnet.py, Apache-2.0)
through the Windows port of the directions study (threads instead of fork). Same model (casmi.fpnet.FPNet), same loss
(cross-entropy over the true structure + 63 pool structures of its 10 ppm window on f.z, BCE on the bits, an
element-count head). Additions:
  --np_p     share of the spectrum-uniform half of every batch drawn from natural-product-catalogue structures
             (pool src >= 2, training folds only); the hidden test is natural-product chemistry and the public
             checkpoints are weakest there (hold-out top-1 0.39 against 0.70 for other structures)
  --tims_p   same for timsTOF spectra
  --sim      directory written by train/sim_iceberg.py (simulated [M+H]+ spectra of pool structures); --sim_p is the
             share of every batch drawn from them, structure-uniform. A simulated structure gets its own training
             id, so real spectra of a held-out structure stay held out while its simulated spectra are trained on.
             --sim_sets restricts to some sets of the table. --sim_pow 2 squares the simulated intensities: ms-pred
             trains ICEBERG on sqrt intensities, so its output is on the sqrt scale (squared, the peak-count profile
             matches measured timsTOF spectra and the similarity to real spectra rises); --sim_floor then drops
             peaks below a share of the base peak (the spectrum cache uses 0.001)
             --sim_w T=0.1 gives the structures of a set a relative sampling weight (default 1 each): the way to
             test how often a structure's simulated spectra must be seen (a full pool gives few passes per structure)
  --pc_neg   directory (or comma list) written by train/mine_pc_negatives.py (PubChem mass-window negatives of the training structures:
             fp.npy, pid.npy, hard.npy, rand.npy). The ranking loss then runs over the true structure, the pool
             negatives and --pc_k PubChem negatives per sample, --pc_hard of them the structure's hard ones (highest
             similarity), the rest drawn from its formula's random window rows. A structure without mined negatives
             (or with fewer) has the missing columns masked out of the softmax. --pool_k (default 63 = all) keeps
             only that many pool negatives, e.g. --pool_k 31 --pc_k 32 replaces half of them at equal list size
  --extra_sets  parquet (sid, set): extra evaluation sets "<set> all / positive / [M+H]+ / negative"
  --ranks_out   with --eval_only: write the rank of every query group to <tag>_ranks.parquet (paired comparisons)
  --resume   continue from <tag>_resume.pt (model, optimiser, step); the schedule is step-based (--max_steps)
  sharing    the machine is shared with the user's desktop and games. The run sleeps between steps so that it uses
             only a share of the GPU: --duty normally, --game_duty while a game is running (train/gpushare.py).
             Only when other processes hold more than --yield_gb of GPU
             memory does it write <tag>_resume.pt and exit with code 75; train/guard_run.py restarts it later.
The process runs at below-normal priority with a few data threads.

usage (run dir needs work/ with the spectrum cache, pool tables and split.parquet; see train/README.md):
  python train/guard_run.py -- python train/train_fpnet.py --code <dir with casmi/> --run <run dir> \
      --init <ckpt.pt> --np_p 0.5 --bs 64 --lr 1e-4 --max_steps 20000 --tag ft_np50
  python train/train_fpnet.py --code ... --run ... --eval_only <ckpt.pt> --tag <name>
"""
import argparse, copy, json, math, os, queue, sys, threading, time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--run', required=True)
ap.add_argument('--max_steps', type=int, default=20000)
ap.add_argument('--bs', type=int, default=64); ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--warmup', type=int, default=2000)
ap.add_argument('--merge_p', type=float, default=0.3); ap.add_argument('--w_bce', type=float, default=1.0)
ap.add_argument('--w_el', type=float, default=0.1); ap.add_argument('--threads', type=int, default=3)
ap.add_argument('--eval_every', type=int, default=5000); ap.add_argument('--eval_groups', type=int, default=3000)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--layers', type=int, default=8); ap.add_argument('--d', type=int, default=512)
ap.add_argument('--holdout', default='hold,np'); ap.add_argument('--tag', default='fpnet_own')
ap.add_argument('--tims_p', type=float, default=0.0); ap.add_argument('--np_p', type=float, default=0.0)
ap.add_argument('--init', default=None, help='warm start from a checkpoint (state dict loaded with weights_only=True)')
ap.add_argument('--resume', action='store_true', help='continue from <tag>_resume.pt when it exists')
ap.add_argument('--yield_gb', type=float, default=9.0, help='exit(75) when other processes hold more GPU memory than this; 0 = never')
ap.add_argument('--duty', type=float, default=0.8, help='share of wall time spent training when no game runs')
ap.add_argument('--game_duty', type=float, default=0.25, help='the same while a game is running')
ap.add_argument('--eval_only', default=None)
ap.add_argument('--sim', default=None, help='directory (or comma list) with sim_mz.npy, sim_it.npy, sim_off.npy, sim_meta.parquet (train/sim_iceberg.py)')
ap.add_argument('--sim_p', type=float, default=0.0, help='share of every batch drawn from the simulated spectra')
ap.add_argument('--sim_sets', default=None, help='comma list of values of the set column to use (default: all)')
ap.add_argument('--sim_floor', type=float, default=0.0, help='drop simulated peaks below this share of the base peak')
ap.add_argument('--sim_pow', type=float, default=1.0, help='power applied to the simulated intensities (2 = sqrt scale back to linear)')
ap.add_argument('--sim_w', default=None, help='relative sampling weights of simulated structures by set, e.g. T=0.1,Tp=0.1')
ap.add_argument('--pc_neg', default=None, help='directory of train/mine_pc_negatives.py: PubChem window negatives added to the ranking loss')
ap.add_argument('--pc_k', type=int, default=32, help='PubChem negatives per sample (with --pc_neg)')
ap.add_argument('--pc_hard', type=int, default=16, help='how many of them are the hard ones; the rest are random window rows')
ap.add_argument('--pool_k', type=int, default=63, help='pool negatives per sample used in the ranking loss (63 = all)')
ap.add_argument('--extra_sets', default=None, help='parquet (sid, set) of extra evaluation sets')
ap.add_argument('--ranks_out', action='store_true', help='with --eval_only: write per-group ranks to <tag>_ranks.parquet')
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpushare
gpushare.below_normal()                                # the machine is shared
sys.path.insert(0, args.code)
os.chdir(args.run)
import pandas as pd
import torch
import torch.nn.functional as F
from casmi import fpnet, traindata
from casmi.spectra import merge_spectra

YIELD_CODE = 75
SHARE = None                                           # gpushare.Share of this run; evaluation batches tick it too


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_state(path, dev):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    m = fpnet.FPNet(ck['nbits'], d=ck['d'], layers=ck['layers'], heads=ck.get('heads', 8))
    m.load_state_dict(ck['model'])
    return m.to(dev)


def groups_of(D, sel, n, rng):
    """(sid, mode) query groups (<= 8 spectra) built from the selected spectra only; at most n groups."""
    g = {}
    for i in np.where(sel)[0]:
        g.setdefault((int(D.sid[i]), int(D.mode[i])), []).append(int(i))
    keys = sorted(g)
    if len(keys) > n:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), n, replace=False))]
    return [g[k][:8] for k in keys]


@torch.no_grad()
def window_eval(model, D, PFP, entries, dtype, bs=128, detail=False):
    model.eval(); ranks = []; nwin = []
    for s in range(0, len(entries), bs):
        chunk = entries[s:s + bs]; items = []; ts = time.time()
        for idxs in chunk:
            i = idxs[0]
            if len(idxs) == 1:
                mz = D.mz[D.off[i]:D.off[i + 1]]; it = D.it[D.off[i]:D.off[i + 1]]; ce = D.ce[i]; nm = max(1, D.ce_n[i])
            else:
                mz, it = merge_spectra([(D.mz[D.off[j]:D.off[j + 1]], D.it[D.off[j]:D.off[j + 1]]) for j in idxs])
                ces = D.ce[idxs]; ces = ces[~np.isnan(ces)]; ce = ces.mean() if len(ces) else np.nan
                nm = int(sum(max(1, D.ce_n[j]) for j in idxs))
            pm, pi = fpnet.prep_peaks(mz, it, D.prec[i])
            items.append(dict(mz=pm, it=pi, prec=float(D.prec[i]), adduct_ix=int(D.adduct[i]), ce=0.0 if np.isnan(ce) else float(ce),
                              ce_known=0.0 if np.isnan(ce) else 1.0, n_merged=min(nm, 8), mode=float(D.mode[i])))
        inp = fpnet.collate(items, 'cuda')
        with torch.autocast('cuda', dtype=dtype):
            z = model(*inp)[0].float()
        for r, idxs in enumerate(chunk):
            pid = D.sid2pid[D.sid[idxs[0]]]; m = D.pmass[pid]; tol = m * 10e-6
            lo = np.searchsorted(D.pmass, m - tol, 'left'); hi = np.searchsorted(D.pmass, m + tol, 'right')
            sc = fpnet.unpack_gpu(PFP[lo:hi], D.nbits) @ z[r]
            ranks.append(int((sc > sc[pid - lo]).sum().item()) + 1); nwin.append(int(hi - lo))
        if SHARE is not None:
            SHARE.tick(time.time() - ts)
            if args.eval_only and SHARE.must_yield():      # an evaluation has nothing to save: free the memory, run it again later
                log(f'YIELD: other processes hold {SHARE.other:.1f} GB of GPU memory -> evaluation stopped'); sys.stdout.flush(); os._exit(YIELD_CODE)
    model.train()
    ranks = np.array(ranks)
    if not len(ranks):
        res = dict(n=0)
    else:
        res = dict(n=len(ranks), mrr25=round(float(np.where(ranks <= 25, 1.0 / ranks, 0).mean()), 4), top1=round(float((ranks == 1).mean()), 4),
                   top5=round(float((ranks <= 5).mean()), 4), top25=round(float((ranks <= 25).mean()), 4))
    return (res, ranks, np.array(nwin)) if detail else res


def np_flags(D):
    """Per spectrum: structure is in the pool, and it is in a natural-product catalogue (pool src 2/3 = COCONUT)."""
    pool = pd.read_parquet('work/pool_meta.parquet', columns=['src'])
    in_pool = D.sid2pid[D.sid] >= 0
    npcat = np.zeros(len(D.sid), bool); npcat[in_pool] = pool.src.values[D.sid2pid[D.sid[in_pool]]] >= 2
    return in_pool, npcat


def eval_sets(D, n, rng):
    """Hold-out queries by instrument / polarity / natural-product catalogue membership."""
    instr = pd.read_parquet('work/spec_meta.parquet', columns=['instr']).instr.values
    in_pool, npcat = np_flags(D)
    f = D.fold[D.sid]
    hold = (f == 'hold') & in_pool; npx = (f == 'np') & in_pool
    S = {'hold all': hold, 'hold timsTOF': hold & (instr == 0), 'hold other instruments': hold & (instr != 0),
         'hold NP-catalogue': hold & npcat, 'hold NP-catalogue positive': hold & npcat & (D.mode > 0),
         'hold NP-catalogue negative': hold & npcat & (D.mode < 0), 'hold not NP-catalogue': hold & ~npcat,
         'np panel (timsTOF)': npx & (D.lib == 9), 'np panel positive': npx & (D.lib == 9) & (D.mode > 0),
         'np panel negative': npx & (D.lib == 9) & (D.mode < 0), 'np structures, other libraries': npx & (D.lib != 9)}
    out = {k: groups_of(D, v, n, rng) for k, v in S.items()}
    if args.extra_sets:                                 # after the standard sets, uncapped: their random draws stay the same
        ex = pd.read_parquet(args.extra_sets, columns=['sid', 'set'])
        ok = in_pool & np.isin(f, ('hold', 'np'))       # never a structure that is trained on
        for name in sorted(ex.set.unique()):
            m = ok & np.isin(D.sid, ex.sid.values[ex.set.values == name])
            for sub, sel in (('all', m), ('positive', m & (D.mode > 0)), ('[M+H]+', m & (D.mode > 0) & (D.adduct == 0)), ('negative', m & (D.mode < 0))):
                out[f'{name} {sub}'] = groups_of(D, sel, 10 ** 9, rng)
    return out


def add_sim(D, path, sets=None, floor=0.0, power=1.0, weights=None):
    """Append simulated spectra to the data arrays. Returns the group ids (one per simulated structure) to sample from
    and their sampling probabilities (None = uniform).

    Every simulated structure gets a new structure id (fold 'sim') that maps to its pool entry, so the real spectra of
    the same structure keep their own id and fold: a held-out structure can be simulated without leaking its spectra."""
    dirs = path.split(',')                              # several directories: concatenated
    meta = pd.concat([pd.read_parquet(os.path.join(d, 'sim_meta.parquet')) for d in dirs], ignore_index=True)
    ln = np.concatenate([np.diff(np.load(os.path.join(d, 'sim_off.npy'))) for d in dirs])
    mz = np.concatenate([np.load(os.path.join(d, 'sim_mz.npy')) for d in dirs]); it = np.concatenate([np.load(os.path.join(d, 'sim_it.npy')) for d in dirs])
    if power != 1.0:
        it = (it.astype(np.float64) ** power).astype(np.float32)
    keep = np.ones(len(meta), bool) if sets is None else np.isin(meta.set.values, sets)
    pk = np.repeat(keep, ln)
    if floor > 0:                                       # intensities are base-peak normalised
        pk &= it >= floor
    sp = np.repeat(np.arange(len(meta)), ln)[pk]
    ln2 = np.bincount(sp, minlength=len(meta))
    keep &= ln2 > 0
    meta = meta[keep].reset_index(drop=True); ln2 = ln2[keep]; mz = mz[pk]; it = it[pk]
    upid, inv = np.unique(meta.pid.values, return_inverse=True)
    n0 = len(D.sid); s0 = len(D.sid2pid)
    D.sid2pid = np.concatenate([D.sid2pid, upid.astype(np.int64)])
    D.fold = np.concatenate([D.fold, np.array(['sim'] * len(upid), dtype=object)])
    D.off = np.concatenate([D.off, D.off[-1] + np.cumsum(ln2)]).astype(np.int64)
    D.mz = np.concatenate([D.mz, mz.astype(D.mz.dtype)]); D.it = np.concatenate([D.it, it.astype(D.it.dtype)])
    D.sid = np.concatenate([D.sid, s0 + inv]); D.adduct = np.concatenate([D.adduct, meta.adduct_ix.values.astype(np.int64)])
    D.mode = np.concatenate([D.mode, meta['mode'].values.astype(np.int64)]); D.prec = np.concatenate([D.prec, meta.precursor_mz.values.astype(np.float64)])
    D.ce = np.concatenate([D.ce, meta.ce.values.astype(np.float64)]); D.ce_n = np.concatenate([D.ce_n, np.ones(len(meta), np.int64)])
    D.lib = np.concatenate([D.lib, np.full(len(meta), -1, np.int64)])
    # (structure, mode) groups again, as in casmi.traindata.Data (merge augmentation and structure-uniform sampling)
    key = D.sid * 2 + (D.mode > 0)
    order = np.argsort(key, kind='mergesort'); ks = key[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    D.gs = np.concatenate([[0], bounds]); D.ge = np.concatenate([bounds, [len(ks)]]); D.grp_order = order
    D.gk = ks[D.gs]; D.key2g = dict(zip(D.gk.tolist(), range(len(D.gk))))
    D.tr_groups = np.array([D.key2g[k] for k in np.unique(key[D.tr])])
    sim_groups = np.array([D.key2g[k] for k in np.unique(key[n0:])])
    prob = None
    if weights:                                         # sim_groups is ordered by structure id = order of upid
        set_of = dict(zip(meta.pid.values, meta.set.values))
        w = np.array([float(weights.get(set_of[p], 1.0)) for p in upid]); prob = w / w.sum()
        log('simulated sampling weights', weights, '| expected share of samples by set',
            {k: round(float(prob[[set_of[p] == k for p in upid]].sum()), 4) for k in sorted(set(set_of.values()))})
    log(f'simulated spectra {len(meta):,} of {len(upid):,} structures (sets {sorted(set(meta.set))}), power {power} floor {floor}: '
        f'peaks per spectrum median {int(np.median(ln2))} mean {ln2.mean():.1f}')
    return sim_groups, prob


class PcNeg:
    """PubChem window negatives of train/mine_pc_negatives.py: packed fingerprints and a validity mask per sample."""

    def __init__(self, path, npool, k, n_hard):
        dirs = path.split(',')                                                   # several directories: concatenated
        self.fps = [np.load(os.path.join(d, 'fp.npy'), mmap_mode='r') for d in dirs]   # several GB: left on disk, rows read per batch
        self.base = np.concatenate([[0], np.cumsum([len(f) for f in self.fps])])
        cat = lambda n: np.concatenate([np.where(x >= 0, x + b, -1) for x, b in zip((np.load(os.path.join(d, n)) for d in dirs), self.base)])
        self.hard = cat('hard.npy'); self.rand = cat('rand.npy')
        self.row = np.full(npool, -1, np.int64); self.row[np.concatenate([np.load(os.path.join(d, 'pid.npy')) for d in dirs])] = np.arange(len(self.hard))
        self.k = k; self.nh = min(n_hard, k, self.hard.shape[1]); self.nr = k - self.nh
        assert self.nr <= self.rand.shape[1], 'more random negatives asked for than were mined'

    def sample(self, pos, rng):
        idx = np.full((len(pos), self.k), -1, np.int64)
        for r, s in enumerate(self.row[pos]):
            if s < 0:
                continue
            h = self.hard[s]; h = h[h >= 0]
            if len(h) > self.nh:
                h = rng.choice(h, self.nh, replace=False)
            idx[r, :len(h)] = h
            q = self.rand[s]; q = q[q >= 0]; q = q[~np.isin(q, h)]
            n = min(len(q), self.k - len(h))                                     # random rows also fill up missing hard ones
            if n:
                idx[r, len(h):len(h) + n] = rng.choice(q, n, replace=False)
        ok = idx >= 0
        flat = np.where(ok, idx, 0).ravel(); o = np.argsort(flat); fs = flat[o]
        fp = np.empty((len(flat), self.fps[0].shape[1]), np.uint8)
        cut = np.searchsorted(fs, self.base)                                     # sorted reads, one stretch per directory
        for i, f in enumerate(self.fps):
            if cut[i + 1] > cut[i]:
                fp[o[cut[i]:cut[i + 1]]] = f[fs[cut[i]:cut[i + 1]] - self.base[i]]
        return fp.reshape(len(pos), self.k, -1), ok


def main():
    global SHARE
    SHARE = gpushare.Share(args.duty, args.game_duty, args.yield_gb, log=log)
    hold = tuple(args.holdout.split(','))
    D = traindata.Data('cpu', seed=args.seed, holdout_folds=hold)
    rng = np.random.default_rng(12345)
    dev = 'cuda'; dtype = fpnet.autocast_dtype(dev)
    PFP = D.PFP.to(dev); ELEM = D.elem.to(dev)
    sets = eval_sets(D, args.eval_groups, rng)
    log({k: len(v) for k, v in sets.items()}, 'train spectra', len(D.tr))
    if args.eval_only:
        model = load_state(args.eval_only, dev)
        res, rows = {}, []
        for k, ent in sets.items():
            res[k], rk, nw = window_eval(model, D, PFP, ent, dtype, detail=True)
            log('EVAL', k, json.dumps(res[k]))
            if len(ent):
                rows.append(pd.DataFrame(dict(set=k, sid=[int(D.sid[e[0]]) for e in ent], mode=[int(D.mode[e[0]]) for e in ent],
                                              i0=[e[0] for e in ent], n_spec=[len(e) for e in ent], rank=rk, window=nw)))
        json.dump(res, open(f'{args.tag}_eval.json', 'w'), indent=1)
        if args.ranks_out:
            pd.concat(rows, ignore_index=True).to_parquet(f'{args.tag}_ranks.parquet', index=False)
        return
    instr = pd.read_parquet('work/spec_meta.parquet', columns=['instr']).instr.values
    tims_tr = D.tr[instr[D.tr] == 0]
    np_tr = D.tr[np_flags(D)[1][D.tr]]
    log('training spectra', len(D.tr), '| timsTOF', len(tims_tr), '| NP-catalogue', len(np_tr))
    sim_w = {k: float(v) for k, v in (x.split('=') for x in args.sim_w.split(','))} if args.sim_w else None
    sim_groups, sim_prob = add_sim(D, args.sim, args.sim_sets.split(',') if args.sim_sets else None, args.sim_floor, args.sim_pow,
                                   sim_w) if args.sim and args.sim_p > 0 else (None, None)
    n_sim = int(round(args.bs * args.sim_p)) if sim_groups is not None else 0
    pcn = PcNeg(args.pc_neg, len(D.pmass), args.pc_k, args.pc_hard) if args.pc_neg else None
    if pcn is not None:
        log(f'PubChem negatives: {len(pcn.hard):,} structures, {int(pcn.base[-1]):,} fingerprints; per sample {pcn.nh} hard + {pcn.nr} random, '
            f'pool negatives {min(args.pool_k, 63)}')

    def sample_ids(Dw, B):
        sim = []
        if n_sim:                                      # structure-uniform over the simulated structures, one spectrum each
            B = B - n_sim
            sim = [np.array([Dw.rng.choice(Dw.grp_order[Dw.gs[g]:Dw.ge[g]]) for g in Dw.rng.choice(sim_groups, size=n_sim, replace=False, p=sim_prob)])]
        h = B // 2
        nt, nn = int(round(h * args.tims_p)), int(round(h * args.np_p))
        parts = [Dw.rng.choice(Dw.tr, size=h - nt - nn, replace=False)]
        if nt:
            parts.append(Dw.rng.choice(tims_tr, size=nt, replace=False))
        if nn:
            parts.append(Dw.rng.choice(np_tr, size=nn, replace=False))
        gs = Dw.rng.choice(Dw.tr_groups, size=B - h, replace=False)
        b = np.array([Dw.rng.choice(Dw.grp_order[Dw.gs[g]:Dw.ge[g]]) for g in gs])
        return np.concatenate(parts + [b] + sim)

    torch.manual_seed(args.seed)
    model = load_state(args.init, dev).train() if args.init else fpnet.FPNet(D.nbits, d=args.d, layers=args.layers).to(dev)
    log('parameters', sum(p.numel() for p in model.parameters()))
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]; no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=no_decay, weight_decay=0.0)], lr=args.lr, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    step, hist, active = 0, [], 0.0
    rfile = f'{args.tag}_resume.pt'
    if args.resume and os.path.exists(rfile):
        ck = torch.load(rfile, map_location=dev, weights_only=False)             # our own file
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        step, hist, active = ck['step'], ck['hist'], ck['active']
        log('resumed at step', step)

    def save_resume():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, hist=hist,
                        active=active), rfile + '.tmp')
        os.replace(rfile + '.tmp', rfile)

    q = queue.Queue(maxsize=32); stop = threading.Event()

    def worker(wid):
        Dw = copy.copy(D); Dw.rng = np.random.default_rng([args.seed, wid, step])       # shared arrays, own RNG
        while not stop.is_set():
            ids = sample_ids(Dw, args.bs)
            items = [Dw.item(int(i), args.merge_p, True) for i in ids]
            pos = Dw.sid2pid[Dw.sid[ids]]
            q.put((fpnet.collate(items, 'cpu'), pos, Dw.negatives(pos)[:, :args.pool_k], pcn.sample(pos, Dw.rng) if pcn is not None else None))

    for w in range(args.threads):
        threading.Thread(target=worker, args=(w,), daemon=True).start()
    t0 = time.time(); step0 = step; wait = 0.0
    share = SHARE
    ema = {k: 0.0 for k in ('loss', 'rank', 'bce', 'el', 'acc')}
    target = torch.zeros(args.bs, dtype=torch.long, device=dev)
    while step < args.max_steps:
        prog = step / args.max_steps
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g['lr'] = lr
        ts = time.time()
        share.poll()
        if share.must_yield():
            active += time.time() - t0 - share.slept; save_resume()
            log(f'YIELD: other processes hold {share.other:.1f} GB of GPU memory -> saved {rfile}, exiting')
            stop.set(); sys.stdout.flush(); os._exit(YIELD_CODE)
        tw = time.time(); arrs, pos, negs, pcb = q.get(); wait += time.time() - tw
        inp = tuple(a.to(dev, non_blocking=True) for a in arrs)
        pos_t = torch.as_tensor(pos, device=dev)
        cand = torch.as_tensor(np.concatenate([pos[:, None], negs], 1), device=dev)
        with torch.autocast('cuda', dtype=dtype):
            z, el = model(*inp)
        z = z.float(); el = el.float()
        fp = fpnet.unpack_gpu(PFP[cand], D.nbits)
        sc = torch.einsum('bkn,bn->bk', fp, z)
        if pcb is not None:                             # PubChem window negatives: extra columns of the same softmax
            fpp = fpnet.unpack_gpu(torch.as_tensor(pcb[0], device=dev), D.nbits)
            scp = torch.einsum('bkn,bn->bk', fpp, z).masked_fill(~torch.as_tensor(pcb[1], device=dev), -1e4)
            sc = torch.cat([sc, scp], 1)
        l_rank = F.cross_entropy(sc, target)
        l_bce = F.binary_cross_entropy_with_logits(z, fp[:, 0])
        l_el = F.smooth_l1_loss(el, ELEM[pos_t])
        loss = l_rank + args.w_bce * l_bce + args.w_el * l_el
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        torch.cuda.synchronize()
        share.tick(time.time() - ts)                    # leave the rest of the GPU time to the desktop / the game
        vals = dict(loss=loss.item(), rank=l_rank.item(), bce=l_bce.item(), el=l_el.item(), acc=(sc.argmax(1) == 0).float().mean().item())
        for k, v in vals.items():
            ema[k] = v if step == step0 + 1 else 0.98 * ema[k] + 0.02 * v
        if not math.isfinite(vals['loss']):
            log('non-finite loss at step', step); break
        if step % 100 == 0:
            el_t = time.time() - t0
            log(f"step {step}/{args.max_steps} lr {lr:.2e} " + ' '.join(f'{k} {v:.4f}' for k, v in ema.items()) +
                f" {(step - step0) / el_t:.2f} it/s duty {share.duty:.2f} gpu-mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if step % 1000 == 0:
            save_resume()
        if step % args.eval_every == 0 and step < args.max_steps:
            ev = {name: window_eval(model, D, PFP, ent, dtype) for name, ent in sets.items()}
            hist.append(dict(step=step, eval=ev)); log('EVAL', json.dumps(hist[-1]))
    active += time.time() - t0 - share.slept
    ev = {name: window_eval(model, D, PFP, ent, dtype) for name, ent in sets.items()}
    hist.append(dict(step=step, hours=active / 3600, final=True, eval=ev)); log('EVAL', json.dumps(hist[-1]))
    fpnet.save_ckpt(f'{args.tag}_final.pt', model, step, extra=dict(args=vars(args)))
    json.dump(hist, open(f'{args.tag}_history.json', 'w'), indent=1)
    save_resume(); stop.set(); log('ALL_DONE', f'{active / 3600:.2f} h of training')
    sys.stdout.flush(); os._exit(0)


if __name__ == '__main__':
    main()
