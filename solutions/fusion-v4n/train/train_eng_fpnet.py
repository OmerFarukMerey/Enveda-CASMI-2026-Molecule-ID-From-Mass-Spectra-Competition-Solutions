"""Fine-tune engine 2's fingerprint network (pv_fp.FPNet, prvsiyan's architecture) on train.parquet, with oversampled libraries, cross-fitted folds and a weight soup.

Engine 2's FPNet is not the v1 network of train_fpnet.py (6,930 bits, an instrument embedding, raw collision energy, no
element head), so this is a separate trainer; the GPU sharing, resume and logging are the same (gpushare.py, exit code 75,
guard_run.py). The model class and the peak preparation are imported from nb/embed/eng/pv_fp.py, the file the notebook runs.

Data (nothing but train.parquet and public structure lists):
  spectra    the spectrum cache of train.parquet (--work: spec_mz / spec_it / spec_off, row-aligned with train.parquet; peaks
             above precursor + 2 dropped, 0.1% floor, base peak 1: what casmi_engine hands to pv_fp.prep_peaks)
  inputs     per row from train.parquet as the engine computes them: adduct index, instrument family, ionisation mode,
             collision energy = casmi_engine.parse_ce (rows WITHOUT a collision energy are not trained on: at inference the
             engine feeds NaN for them and the fingerprint features are switched off, reviews/np-eng2-2026-10-05.md section 3)
  pool       engine 2's candidate pool, COCONUT + ChEBI / LIPID MAPS + training structures with prvsiyan's 6,930-bit
             fingerprints (--ds, --train_fp: the caches of scratch/mixed/eng2_local.py), sorted by mass
Loss: cross-entropy over the true structure and --k pool rows of its +-10 ppm window on f.z / T, plus --w_bce * BCE on the
bits. T is fitted ONCE on the base checkpoint (--temp auto: the temperature that minimises the base's own ranking loss), so
the loss is scale-free with respect to the checkpoint's logit scale: a public checkpoint whose f.z spread is ten times
another's starts at the same loss (reviews/hardneg-2026-10-05.md: the raw recipe started at 38 and destroyed models A / B).
The start-up prints the logit scale, the ranking loss at T = 1 and at the fitted T, and the BCE of the base.
  --boost    "lib:enveda-np-examples=4" or "instr:timsTOF=4" (comma list): sampling weight of those rows (1 = spectrum-uniform)
  --boost_lambda   loss weight of the boosted rows (1 = none)
  --np_folds / --hold_np_fold   the structures of enveda-np-examples are split into folds by structure; the fold's
             structures are never trained on (all their rows, every library) and are the validation set "np val"
  --hold_keys  parquet with a `key` column (metric keys): structures never trained on (the molecules of the local replay)
  --soup     average the weights of several checkpoints (--soup a.pt,b.pt --out soup.pt [--soup_base base.pt --soup_alpha 0.5])
  --eval_only  the evaluation table of a checkpoint; --mix adds "1/2 (checkpoint + mix)" rows, the single-spectrum group of
             engine 2 when the checkpoint is passed as ENG_EXTRA_FPNET next to the public fp_single_s2.pt
Evaluation = the engine's use of the model: mean logits over the (at most 8) spectra of a structure in one library, rank of
the answer by f.z among the pool rows of its +-10 ppm window (tautomer twins of the answer excluded).

usage (see train/README.md):
  python train/guard_run.py -- python train/train_eng_fpnet.py --run <dir> --init <fp_single_s2.pt> \
      --boost lib:enveda-np-examples=4 --hold_np_fold 0 --max_steps 3000 --lr 3e-5 --tag np4_f0
  python train/train_eng_fpnet.py --run <dir> --soup a.pt,b.pt,c.pt --out soup.pt
"""
import argparse, json, math, os, queue, sys, threading, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True, help='output directory (logs, checkpoints, the table cache eng_meta.parquet)')
ap.add_argument('--eng', default=os.path.join(os.path.dirname(HERE), 'nb', 'embed', 'eng'), help='directory with pv_fp.py')
ap.add_argument('--work', default='D:/casmi-data/directions/run/work', help='spectrum cache + train_structs.parquet')
ap.add_argument('--train', default='D:/casmi-data/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet')
ap.add_argument('--ds', default='D:/casmi-data/mixed_regime/ds', help='coco/ and bio/ of prvsiyan (fingerprints, masses, keys)')
ap.add_argument('--train_fp', default='D:/casmi-data/mixed_regime/eng2/train_fp.npz', help='fingerprints of the training structures (eng2_local.py)')
ap.add_argument('--init', default=None); ap.add_argument('--tag', default='eng_fp')
ap.add_argument('--max_steps', type=int, default=3000); ap.add_argument('--bs', type=int, default=48)
ap.add_argument('--lr', type=float, default=3e-5); ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--warmup', type=int, default=200)
ap.add_argument('--k', type=int, default=63, help='pool negatives per sample'); ap.add_argument('--w_bce', type=float, default=1.0)
ap.add_argument('--temp', default='auto', help="'auto' = fitted on the base checkpoint, or a number")
ap.add_argument('--boost', default=''); ap.add_argument('--boost_lambda', type=float, default=1.0)
ap.add_argument('--np_folds', type=int, default=5); ap.add_argument('--hold_np_fold', type=int, default=-1); ap.add_argument('--fold_seed', type=int, default=0)
ap.add_argument('--hold_keys', default=None); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--threads', type=int, default=3); ap.add_argument('--eval_every', type=int, default=1000); ap.add_argument('--eval_n', type=int, default=400)
ap.add_argument('--resume', action='store_true'); ap.add_argument('--eval_only', default=None); ap.add_argument('--mix', default=None)
ap.add_argument('--ranks_out', action='store_true')
ap.add_argument('--yield_gb', type=float, default=11.0); ap.add_argument('--duty', type=float, default=0.8); ap.add_argument('--game_duty', type=float, default=0.25)
ap.add_argument('--soup', default=None); ap.add_argument('--out', default=None); ap.add_argument('--soup_base', default=None); ap.add_argument('--soup_alpha', type=float, default=1.0)
ap.add_argument('--cpu', action='store_true', help='run on the CPU (tests, smoke)')
args = ap.parse_args()
sys.path.insert(0, HERE); sys.path.insert(0, args.eng)
import gpushare
gpushare.below_normal()
os.makedirs(args.run, exist_ok=True)
import pandas as pd
import torch
import torch.nn.functional as F
import pv_fp

YIELD_CODE = 75
NBITS = 6930
NP_LIB = 'enveda-np-examples'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_ckpt(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def load_model(path, dev):
    ck = load_ckpt(path)
    m = pv_fp.FPNet(ck['nbits'], d=ck['d'], layers=ck['layers']); m.load_state_dict(ck['model'])
    return m.to(dev)


def save_model(path, model, step, **extra):
    torch.save(dict(step=step, nbits=NBITS, d=model.d, layers=len(model.blocks), model={k: v.cpu() for k, v in model.state_dict().items()}, **extra), path + '.tmp')
    os.replace(path + '.tmp', path)


def soup(paths, out, base=None, alpha=1.0):
    """Uniform average of the weights of `paths`; with a base: base + alpha * (average - base)."""
    cks = [load_ckpt(p) for p in paths]
    assert len({(c['nbits'], c['d'], c['layers']) for c in cks}) == 1
    sd = {}
    for k, v in cks[0]['model'].items():
        sd[k] = torch.stack([c['model'][k].double() for c in cks]).mean(0).to(v.dtype) if v.is_floating_point() else v.clone()
    if base:
        b = load_ckpt(base)['model']
        sd = {k: (b[k].double() + alpha * (v.double() - b[k].double())).to(v.dtype) if v.is_floating_point() else v for k, v in sd.items()}
    torch.save(dict(step=cks[0].get('step', 0), nbits=cks[0]['nbits'], d=cks[0]['d'], layers=cks[0]['layers'], model=sd,
                    soup_of=[os.path.basename(p) for p in paths], soup_base=os.path.basename(base) if base else '', soup_alpha=float(alpha)), out + '.tmp')
    os.replace(out + '.tmp', out)
    log('soup of', len(paths), '->', out)


def unpack(fp, dev=None):
    """packed uint8 (..., 867) -> float bits (..., 6930) on the tensor's device (np.unpackbits order)."""
    sh = torch.arange(7, -1, -1, device=fp.device, dtype=torch.uint8)
    return ((fp.unsqueeze(-1) >> sh) & 1).flatten(-2)[..., :NBITS].float()


# --------------------------------------------------------------------------------------------------------------- data
def parse_ce_column(col):
    """casmi_engine.parse_ce for a pyarrow list column: mean of |values|; an empty list -> 25.0; null (or NaN values) -> NaN."""
    import pyarrow as pa
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    off = col.offsets.to_numpy().astype(np.int64); val = np.abs(col.values.to_numpy(zero_copy_only=False).astype(np.float64))
    n = np.diff(off); cs = np.concatenate([[0.0], np.cumsum(val)])
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(n > 0, (cs[off[1:]] - cs[off[:-1]]) / np.maximum(n, 1), 25.0)
    bad = np.add.reduceat(np.concatenate([np.isnan(val), [False]]).astype(np.int64), np.minimum(off[:-1], len(val)))[: len(n)] if len(val) else np.zeros(len(n), np.int64)
    out[(n > 0) & (bad > 0)] = np.nan                      # NaN inside a list: the engine's mean is NaN too
    out[~np.asarray(col.is_valid())] = np.nan
    return out


def build_meta(path):
    """Per row of train.parquet: the engine's model inputs and the row's library; cached in <run>/eng_meta.parquet."""
    import pyarrow as pa, pyarrow.parquet as pq
    f = pq.ParquetFile(args.train); parts = []
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=['ingest_lib', 'instrument_type', 'adduct', 'precursor_mz', 'ionization_mode', 'collision_energy_ev', 'inchikey14'])
        ins = t.column('instrument_type').cast(pa.string()).to_pylist(); ad = t.column('adduct').cast(pa.string()).to_pylist()
        parts.append(pd.DataFrame(dict(
            lib=t.column('ingest_lib').cast(pa.string()).to_pylist(), ik=t.column('inchikey14').cast(pa.string()).to_pylist(),
            instr=np.array([pv_fp.instr_family(s) for s in ins], np.int8), tims=np.array([isinstance(s, str) and 'timstof' in s.lower() for s in ins]),
            adduct=np.array([pv_fp.ADDUCT_IX.get(str(a), pv_fp.ADDUCT_IX['<unk>']) for a in ad], np.int16),
            prec=t.column('precursor_mz').to_numpy(zero_copy_only=False).astype(np.float64),
            mode=np.where(np.array(t.column('ionization_mode').cast(pa.string()).to_pylist(), dtype=object) == 'positive', 1, -1).astype(np.int8),
            ce=parse_ce_column(t.column('collision_energy_ev')))))
    m = pd.concat(parts, ignore_index=True); m.to_parquet(path, index=False)
    return m


class Data:
    def __init__(self):
        mp = os.path.join(args.run, 'eng_meta.parquet')
        M = pd.read_parquet(mp) if os.path.exists(mp) else build_meta(mp)
        self.off = np.load(args.work + '/spec_off.npy'); assert len(self.off) == len(M) + 1, 'spectrum cache is not row-aligned with train.parquet'
        self.mz = np.load(args.work + '/spec_mz.npy', mmap_mode='r'); self.it = np.load(args.work + '/spec_it.npy', mmap_mode='r')
        # pool exactly as casmi_engine.build_pool (COCONUT, new ChEBI / LIPID MAPS rows, training structures not in them; mass-sorted)
        import pickle
        cm = pickle.load(open(args.ds + '/coco/coco_meta.pkl', 'rb')); bm = pickle.load(open(args.ds + '/bio/bio_meta.pkl', 'rb'))     # prvsiyan's public files
        co_keys = np.asarray(cm['keys'], dtype=object); bkeys = np.asarray(bm['keys'], dtype=object); bnew = ~pd.Series(bkeys).isin(set(co_keys)).values
        z = np.load(args.train_fp, allow_pickle=False); ok = z['ok']
        tr_ik = pd.Series(M.ik.dropna().drop_duplicates().values); tr_ik = tr_ik[~tr_ik.isin(set(co_keys) | set(bkeys[bnew]))].values.astype(str)
        assert len(tr_ik) == len(z['ik']) and (tr_ik == z['ik']).all(), 'train_fp.npz does not match train.parquet (rebuild with scratch/mixed/eng2_local.py)'
        fp = np.vstack([np.load(args.ds + '/coco/coco_fp.npy'), np.load(args.ds + '/bio/bio_fp.npy')[bnew], z['fp'][ok]])
        mass = np.concatenate([np.load(args.ds + '/coco/coco_mass.npy'), np.load(args.ds + '/bio/bio_mass.npy')[bnew], z['mass'][ok]])
        keys = np.concatenate([co_keys, bkeys[bnew], z['ik'][ok].astype(object)])
        good = np.isfinite(mass); o = np.argsort(np.where(good, mass, 1e18), kind='mergesort')[: good.sum()]
        self.pfp = np.ascontiguousarray(fp[o]); self.pmass = mass[o]; pkeys = keys[o]
        st = pd.read_parquet(args.work + '/train_structs.parquet', columns=['inchikey14', 'key'])
        ik2key = dict(zip(st.inchikey14, st.key))                                    # metric key of the training structures (tautomer twins share it)
        mkey = np.array([ik2key.get(k, k) for k in pkeys], dtype=object)
        self.key2gid = {}; self.pgid = np.array([self.key2gid.setdefault(k, len(self.key2gid)) for k in mkey], np.int64)
        k2i = pd.Series(np.arange(len(pkeys)), index=pkeys); k2i = k2i[~k2i.index.duplicated()]
        self.pid = k2i.reindex(M.ik.values).fillna(-1).values.astype(np.int64)        # pool row of every spectrum's answer
        self.gid = np.where(self.pid >= 0, self.pgid[np.clip(self.pid, 0, None)], -1)
        self.M = M; self.lib = M.lib.values; self.ce = M.ce.values; self.prec = M.prec.values
        self.adduct = M.adduct.values.astype(np.int64); self.instr = M.instr.values.astype(np.int64); self.mode = M['mode'].values.astype(np.float32)
        self.usable = (self.pid >= 0) & np.isfinite(self.ce) & (np.diff(self.off) > 0)
        log(f'pool {len(self.pmass):,} rows; spectra {len(M):,}, usable (answer in the pool, collision energy, peaks) {int(self.usable.sum()):,}')
        # folds of the natural-product standards, by structure
        npg = np.unique(self.gid[(self.lib == NP_LIB) & (self.gid >= 0)]); rng = np.random.default_rng(args.fold_seed)
        self.np_gids = npg; self.np_fold = dict(zip(rng.permutation(npg).tolist(), (np.arange(len(npg)) % args.np_folds).tolist()))
        held = {g for g, f in self.np_fold.items() if f == args.hold_np_fold}
        self.val_np = np.array(sorted(held), np.int64); self.hold_reg = np.zeros(0, np.int64)
        if args.hold_keys:
            hk = pd.read_parquet(args.hold_keys, columns=['key']).key
            self.hold_reg = np.array(sorted({self.key2gid[k] for k in hk if k in self.key2gid}), np.int64)
        self.train_rows = np.flatnonzero(self.usable & ~np.isin(self.gid, np.concatenate([self.val_np, self.hold_reg])))
        log(f'np standards {len(npg)} structures, fold {args.hold_np_fold} holds {len(self.val_np)}; replay structures held out {len(self.hold_reg)}; training rows {len(self.train_rows):,}')

    def item(self, i):
        a, b = self.off[i], self.off[i + 1]
        return pv_fp.prep_peaks(self.mz[a:b], self.it[a:b], float(self.prec[i]))

    def collate(self, rows, dev):
        P = [self.item(int(i)) for i in rows]; B = len(P); N = max(1, max(len(a) for a, _ in P))
        mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
        for i, (a, b) in enumerate(P):
            mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False
        T = lambda x: torch.as_tensor(x, device=dev)
        return (T(mz), T(it), T(pad), T(self.prec[rows].astype(np.float32)), T(self.adduct[rows]), T(self.instr[rows]), T(self.ce[rows].astype(np.float32)), T(self.mode[rows]))

    def window(self, pid):
        m = self.pmass[pid]; tol = m * 10e-6
        return int(np.searchsorted(self.pmass, m - tol, 'left')), int(np.searchsorted(self.pmass, m + tol, 'right'))

    def negatives(self, pids, k, rng):
        idx = np.zeros((len(pids), k), np.int64); ok = np.zeros((len(pids), k), bool)
        for r, p in enumerate(pids):
            lo, hi = self.window(p); c = np.arange(lo, hi); c = c[self.pgid[c] != self.pgid[p]]
            if len(c) > k:
                c = rng.choice(c, k, replace=False)
            idx[r, :len(c)] = c; ok[r, :len(c)] = True
        return idx, ok

    def groups(self, gids, lib=None, n=None, rng=None, need_usable=True):
        """One query per structure (and library): up to 8 usable rows -> [(gid, rows)]."""
        sel = np.isin(self.gid, gids) & (self.usable if need_usable else (self.pid >= 0))
        if lib is not None:
            sel &= self.lib == lib
        rows = np.flatnonzero(sel); out = {}
        for i in rows:
            out.setdefault((int(self.gid[i]), self.lib[i]), []).append(int(i))
        keys = sorted(out)
        if n and len(keys) > n:
            keys = [keys[i] for i in sorted(rng.choice(len(keys), n, replace=False))]
        return [(k[0], out[k][:8]) for k in keys]


@torch.no_grad()
def mol_logits(model, D, ent, dev, bs=96, share=None):
    """Mean logits over the spectra of each query (the single-spectrum view of pv_fp.molecule_logits)."""
    model.eval(); flat = [i for _, rows in ent for i in rows]; Z = []
    for s in range(0, len(flat), bs):
        ts = time.time()
        Z.append(model(*D.collate(np.array(flat[s:s + bs]), dev)).float())
        if share is not None:
            if dev == 'cuda':
                torch.cuda.synchronize()
            share.tick(time.time() - ts)
    Z = torch.cat(Z) if Z else torch.zeros(0, NBITS, device=dev); out = []; p = 0
    for _, rows in ent:
        out.append(Z[p:p + len(rows)].mean(0)); p += len(rows)
    return torch.stack(out) if out else Z


@torch.no_grad()
def rank_eval(D, PFP, ent, Z):
    ranks = []
    for (g, rows), z in zip(ent, Z):
        pid = int(D.pid[rows[0]]); lo, hi = D.window(pid)
        sc = unpack(PFP[lo:hi]) @ z
        same = torch.as_tensor(D.pgid[lo:hi] == D.pgid[pid], device=sc.device)
        ranks.append(int(((sc > sc[pid - lo]) & ~same).sum().item()) + 1)
    r = np.array(ranks)
    if not len(r):
        return dict(n=0), r
    return dict(n=len(r), mrr25=round(float(np.where(r <= 25, 1.0 / r, 0).mean()), 4), top1=round(float((r == 1).mean()), 4), top5=round(float((r <= 5).mean()), 4)), r


def eval_sets(D):
    rng = np.random.default_rng(777)
    others = np.setdiff1d(np.unique(D.gid[D.usable]), np.concatenate([D.np_gids, D.hold_reg]))
    S = {}
    if len(D.val_np):
        S['np val (held-out fold, np-examples)'] = D.groups(D.val_np, NP_LIB)
    S['np train (np-examples, trained on)'] = D.groups(np.setdiff1d(D.np_gids, D.val_np), NP_LIB)
    if len(D.hold_reg):
        S['replay structures (held out of the fine-tune)'] = D.groups(D.hold_reg, None, args.eval_n, rng)
    S['enveda-180 sample'] = D.groups(others, 'enveda-180', args.eval_n, rng)
    for lib in ('massbank', 'mona', 'riken', 'pluskal_ms2'):
        S[f'{lib} sample'] = D.groups(others, lib, args.eval_n // 2, rng)
    return S


def run_eval(model, D, PFP, sets, dev, share=None, mix=None, detail=None):
    res = {}
    for name, ent in sets.items():
        Z = mol_logits(model, D, ent, dev, share=share)
        res[name], r = rank_eval(D, PFP, ent, Z)
        if detail is not None:
            detail += [dict(set=name, arm='model', gid=g, rank=int(x)) for (g, _), x in zip(ent, r)]
        if mix is not None:
            Zm = 0.5 * (Z + mol_logits(mix, D, ent, dev, share=share))
            res[name + ' | 1/2 (model + mix)'], r2 = rank_eval(D, PFP, ent, Zm)
            if detail is not None:
                detail += [dict(set=name, arm='mix', gid=g, rank=int(x)) for (g, _), x in zip(ent, r2)]
    model.train()
    return res


def sampler(D):
    """Cumulative sampling weights over the training rows: 1 per row, times the boost of its library / instrument."""
    w = np.ones(len(D.train_rows)); boosted = np.zeros(len(D.train_rows), bool)
    for spec in [s for s in args.boost.split(',') if s]:
        what, f = spec.split('='); kind, val = what.split(':')
        m = (D.lib[D.train_rows] == val) if kind == 'lib' else D.M.tims.values[D.train_rows] if (kind, val) == ('instr', 'timsTOF') else None
        assert m is not None, f'unknown boost {spec}'
        w[m] *= float(f); boosted |= m
        log(f'boost {spec}: {int(m.sum()):,} rows, share of samples {w[m].sum() / w.sum():.4f} (was {m.mean():.4f})')
    return np.cumsum(w) / w.sum(), boosted


def main():
    if args.soup:
        return soup(args.soup.split(','), args.out, args.soup_base, args.soup_alpha)
    dev = 'cpu' if args.cpu else 'cuda'
    share = gpushare.Share(args.duty, args.game_duty, args.yield_gb, log=log) if dev == 'cuda' else None
    D = Data(); PFP = torch.as_tensor(D.pfp, device=dev)
    sets = eval_sets(D); log({k: len(v) for k, v in sets.items()})
    mix = load_model(args.mix, dev).eval() if args.mix else None
    if args.eval_only:
        model = load_model(args.eval_only, dev); detail = [] if args.ranks_out else None
        res = run_eval(model, D, PFP, sets, dev, share, mix, detail)
        for k, v in res.items():
            log('EVAL', k, json.dumps(v))
        json.dump(res, open(os.path.join(args.run, f'{args.tag}_eval.json'), 'w'), indent=1)
        if detail is not None:
            pd.DataFrame(detail).to_parquet(os.path.join(args.run, f'{args.tag}_ranks.parquet'), index=False)
        return
    cdf, boosted = sampler(D)
    torch.manual_seed(args.seed)
    model = load_model(args.init, dev).train() if args.init else pv_fp.FPNet(NBITS).to(dev).train()
    use_amp = dev == 'cuda'
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]; no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=no_decay, weight_decay=0.0)], lr=args.lr, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    step, hist, active, temp = 0, [], 0.0, None
    rfile = os.path.join(args.run, f'{args.tag}_resume.pt')
    if args.resume and os.path.exists(rfile):
        ck = torch.load(rfile, map_location=dev, weights_only=False)                  # our own file
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        step, hist, active, temp = ck['step'], ck['hist'], ck['active'], ck['temp']
        log('resumed at step', step, 'temperature', temp)

    def save_resume():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, hist=hist, active=active, temp=temp), rfile + '.tmp')
        os.replace(rfile + '.tmp', rfile)

    def batch(rng):
        loc = np.minimum(np.searchsorted(cdf, rng.random(args.bs)), len(cdf) - 1)
        rows = D.train_rows[loc]; pos = D.pid[rows]; neg, ok = D.negatives(pos, args.k, rng)
        return D.collate(rows, 'cpu'), pos, neg, ok, boosted[loc]

    def forward(b):
        arrs, pos, neg, ok, bo = b
        inp = tuple(a.to(dev, non_blocking=True) for a in arrs)
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            z = model(*inp)
        z = z.float()
        fp = unpack(PFP[torch.as_tensor(np.concatenate([pos[:, None], neg], 1), device=dev)])
        sc = torch.einsum('bkn,bn->bk', fp, z)
        mask = torch.as_tensor(np.concatenate([np.ones((len(pos), 1), bool), ok], 1), device=dev)
        return z, fp, sc, mask, torch.as_tensor(bo, device=dev)

    if temp is None:                                       # the base checkpoint's own scale, before any step
        rng0 = np.random.default_rng([args.seed, 999]); model.eval(); SC, MK, bces, zs, d32 = [], [], [], [], []
        with torch.no_grad():
            for _ in range(8):
                b = batch(rng0); z, fp, sc, mask, _ = forward(b)
                SC.append(sc); MK.append(mask); bces.append(F.binary_cross_entropy_with_logits(z, fp[:, 0]).item()); zs.append(z)
                if use_amp and len(d32) < 2:
                    d32.append(float((model(*(a.to(dev) for a in b[0])).float() - z).abs().max()))
        model.train(); SC = torch.cat(SC); MK = torch.cat(MK); zs = torch.cat(zs); tgt = torch.zeros(len(SC), dtype=torch.long, device=dev)
        ce = lambda t: F.cross_entropy((SC / t).masked_fill(~MK, -1e4), tgt).item()
        grid = [2.0 ** (e / 2) for e in range(-6, 17)]; ces = [ce(t) for t in grid]; best = grid[int(np.argmin(ces))]
        temp = best if args.temp == 'auto' else float(args.temp)
        info = dict(logit_std_over_bits=round(float(zs.std(1).mean()), 3), logit_abs_mean=round(float(zs.abs().mean()), 3),
                    fz_std_in_list=round(float(torch.stack([s[m].std() for s, m in zip(SC, MK) if m.sum() > 2]).mean()), 3),
                    rank_loss_T1=round(ce(1.0), 4), best_T=best, rank_loss_best_T=round(min(ces), 4), bce=round(float(np.mean(bces)), 4),
                    top1=round(float(((SC.masked_fill(~MK, -1e9)).argmax(1) == 0).float().mean()), 4), amp_max_abs_diff=d32, temperature_used=temp)
        log('BASE', json.dumps(info)); hist.append(dict(step=0, base=info))
    q = queue.Queue(maxsize=24); stop = threading.Event()

    def worker(wid):
        rng = np.random.default_rng([args.seed, wid, step])
        while not stop.is_set():
            q.put(batch(rng))

    for w in range(args.threads):
        threading.Thread(target=worker, args=(w,), daemon=True).start()
    dfile = os.path.join(args.run, f'{args.tag}_ranks.parquet')             # per-structure ranks of the base (step 0) and of the result: paired comparisons
    if step == 0:
        det = []; ev = run_eval(model, D, PFP, sets, dev, share, detail=det); hist.append(dict(step=0, eval=ev)); log('EVAL', json.dumps(hist[-1]))
        pd.DataFrame(det).assign(when='base').to_parquet(dfile, index=False)
    t0 = time.time(); step0 = step; ema = {}
    while step < args.max_steps:
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * step / args.max_steps)))
        for g in opt.param_groups:
            g['lr'] = lr
        ts = time.time()
        if share is not None:
            share.poll()
            if share.must_yield():
                active += time.time() - t0 - share.slept; save_resume()
                log(f'YIELD: other processes hold {share.other:.1f} GB of GPU memory -> saved {rfile}, exiting'); stop.set(); sys.stdout.flush(); os._exit(YIELD_CODE)
        z, fp, sc, mask, bo = forward(q.get())
        wgt = torch.where(bo, args.boost_lambda, 1.0); wgt = wgt / wgt.mean()
        l_rank = (F.cross_entropy((sc / temp).masked_fill(~mask, -1e4), torch.zeros(len(sc), dtype=torch.long, device=dev), reduction='none') * wgt).mean()
        l_bce = (F.binary_cross_entropy_with_logits(z, fp[:, 0], reduction='none').mean(1) * wgt).mean()
        loss = l_rank + args.w_bce * l_bce
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); step += 1
        if share is not None:
            torch.cuda.synchronize(); share.tick(time.time() - ts)
        vals = dict(loss=loss.item(), rank=l_rank.item(), bce=l_bce.item(), acc=(sc.masked_fill(~mask, -1e9).argmax(1) == 0).float().mean().item(), zstd=z.std(1).mean().item())
        ema = vals if not ema else {k: 0.98 * ema[k] + 0.02 * v for k, v in vals.items()}
        if not math.isfinite(vals['loss']):
            log('non-finite loss at step', step); sys.stdout.flush(); os._exit(2)
        if step % 50 == 0 or step == step0 + 1:
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == 'cuda' else 0.0
            log(f"step {step}/{args.max_steps} lr {lr:.2e} " + ' '.join(f'{k} {v:.4f}' for k, v in ema.items()) +
                f" {(step - step0) / (time.time() - t0):.2f} it/s duty {share.duty if share else 1:.2f} gpu-mem {mem:.1f} GB")
        if step % 500 == 0:
            save_resume()
        if step % args.eval_every == 0 and step < args.max_steps:
            ev = run_eval(model, D, PFP, sets, dev, share); hist.append(dict(step=step, eval=ev)); log('EVAL', json.dumps(hist[-1]))
    active += time.time() - t0 - (share.slept if share else 0.0)
    det = []; ev = run_eval(model, D, PFP, sets, dev, share, mix, det); hist.append(dict(step=step, hours=active / 3600, final=True, eval=ev)); log('EVAL', json.dumps(hist[-1]))
    old = [pd.read_parquet(dfile)] if os.path.exists(dfile) else []
    pd.concat(old + [pd.DataFrame(det).assign(when='final')], ignore_index=True).to_parquet(dfile, index=False)
    save_model(os.path.join(args.run, f'{args.tag}_final.pt'), model, step, base=os.path.basename(args.init or ''), temp=float(temp),
               recipe=json.dumps({k: v for k, v in vars(args).items() if k in ('boost', 'boost_lambda', 'lr', 'bs', 'max_steps', 'k', 'w_bce', 'hold_np_fold', 'np_folds', 'fold_seed', 'seed', 'hold_keys')}))
    json.dump(hist, open(os.path.join(args.run, f'{args.tag}_history.json'), 'w'), indent=1)
    save_resume(); stop.set(); log('ALL_DONE', f'{active / 3600:.3f} h of training'); sys.stdout.flush(); os._exit(0)


if __name__ == '__main__':
    main()
