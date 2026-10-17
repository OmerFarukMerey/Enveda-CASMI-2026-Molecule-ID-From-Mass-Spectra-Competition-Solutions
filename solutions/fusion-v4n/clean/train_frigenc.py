"""Train our spectrum encoder with a FRIGID-fingerprint head (clean/frigenc.py) from scratch, with an exclusion list of training structures.

Licence and provenance. The loop (rank loss over the +-10 ppm pool window, BCE on the pipeline bits, element loss, schedule,
optimiser groups, GPU sharing, resume) is the one of clean/train_specenc.py, itself derived from the public fold-safe training loop
(dmitriigluzdov/casmi26-fold-safe-fpnet, train_fpnet.py, Apache-2.0; see THIRD_PARTY_NOTICES.md). Added here: the FRIGID head and
its BCE term on FRIGID's exact bits (the decoder's conditioning fingerprint, see clean/frigenc.py) of the spectrum's own structure,
and `--excl`: training structures (structure ids of train_structs.parquet) removed from the training positives on top of the
held-out folds, e.g. every structure that shares a generic Murcko scaffold with an evaluation molecule.
Data: the competition's train.parquet only (through the spectrum cache of clean/specdata.py). Leak line printed and stored.

usage: python clean/train_frigenc.py --work <dir> --out <run dir> --frig <structs_frig.npy> --excl <excl_sids.npy> --tag fe_s0 --seed 0
                                     --pair --multi --multi_peaks 160 --buckets 8 --max_steps 16000 --resume
Exit code 75 = yielded the GPU (train/guard_run.py restarts it with --resume).
"""
import argparse, hashlib, json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'train')); sys.path.insert(0, HERE)
import train_fpnet as T  # noqa: E402  (log, leave, eval_sets, YIELD_CODE)
import specenc  # noqa: E402

log = T.log


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True); ap.add_argument('--pool', default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--frig', required=True, help='(n_structures, 512) uint8: packed FRIGID fingerprints, row = structure id')
    ap.add_argument('--excl', default=None, help='.npy of structure ids removed from the training positives')
    ap.add_argument('--tag', default='fe'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pair', action='store_true'); ap.add_argument('--multi', action='store_true'); ap.add_argument('--multi_peaks', type=int, default=160)
    ap.add_argument('--max_steps', type=int, default=16000); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--warmup', type=int, default=2000); ap.add_argument('--merge_p', type=float, default=0.3)
    ap.add_argument('--w_bce', type=float, default=1.0); ap.add_argument('--w_el', type=float, default=0.1); ap.add_argument('--w_frig', type=float, default=1.0)
    ap.add_argument('--layers', type=int, default=8); ap.add_argument('--d', type=int, default=512)
    ap.add_argument('--workers', type=int, default=3); ap.add_argument('--buckets', type=int, default=8)
    ap.add_argument('--eval_groups', type=int, default=3000); ap.add_argument('--no_eval', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--yield_gb', type=float, default=9.0); ap.add_argument('--duty', type=float, default=0.8)
    ap.add_argument('--game_duty', type=float, default=0.25); ap.add_argument('--max_gb', type=float, default=8.0)
    return ap.parse_args(argv)


def restrict(D, excl):
    """Remove the excluded structures from the training spectra of D (positives and merge partners)."""
    if excl is None or not len(excl):
        return 0
    keep = ~np.isin(D.sid[D.tr], excl)
    D.restrict(D.tr[keep])
    return int((~keep).sum())


class FrigStream(specenc.EncStream):
    """specenc.EncStream with the exclusion applied inside every worker (module level: picklable for the spawned workers)."""

    def __init__(self, *a, excl_path=None, **k):
        super().__init__(*a, **k); self.excl_path = excl_path

    def stream(self, worker_id=0):
        import specdata
        D = specenc.EncData(log=lambda *a: None, need_fp=False, **self.data_kw)
        restrict(D, np.load(self.excl_path) if self.excl_path else None)
        rng = np.random.default_rng([self.seed, worker_id, self.start_step]); parts = specdata.parse_parts(D, self.parts_text)
        while True:
            yield D.enc_batch(rng, self.bs, self.cfg, self.merge_p, self.aug, parts)


def main(argv=None):
    args = get_args(argv)
    import gpushare
    gpushare.below_normal()
    import torch
    import torch.nn.functional as F
    import fpnet, specdata, frigenc
    import train_specenc as TS
    os.makedirs(args.out, exist_ok=True)
    P = lambda name: os.path.join(args.out, name)
    t_start = time.time()

    def ledger(event, **kw):
        with open(P('ledger.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), tag=args.tag, pid=os.getpid(), event=event, wall_s=round(time.time() - t_start, 1), **kw)) + '\n')

    if os.path.exists(P(f'{args.tag}_final.pt')):
        log('already finished:', P(f'{args.tag}_final.pt')); return 0
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    share = gpushare.Share(args.duty, args.game_duty, args.yield_gb, log=log)
    data_kw = dict(work=args.work, pool=args.pool, holdout=('hold', 'np'), ann_dir=None)
    D = specenc.EncData(log=log, need_adducts=False, **data_kw)
    excl = np.load(args.excl) if args.excl else np.zeros(0, np.int64)
    n_tr0 = len(D.tr); n_rm = restrict(D, excl)
    pos_sids = np.unique(D.sid[D.tr]); bad = int(np.isin(pos_sids, excl).sum())
    excl_msg = (f'EXCLUSION: {len(excl):,} structures listed, {n_rm:,} of {n_tr0:,} training spectra removed; training structures now {len(pos_sids):,}, '
                f'{len(D.tr):,} spectra; listed structures among the training positives: {bad}')
    log(excl_msg); assert bad == 0
    FRIG = np.load(args.frig); assert FRIG.shape == (len(D.fold), 512), FRIG.shape
    frig_sha = hashlib.sha1(FRIG.tobytes()).hexdigest(); FRIG_T = torch.from_numpy(FRIG).cuda()
    PFP = torch.from_numpy(np.load(os.path.join(args.pool or args.work, 'pool_fp.npy'))).cuda()
    bits_sha = hashlib.sha1(np.load(os.path.join(args.pool or args.work, 'fp_bits.npy')).tobytes()).hexdigest()
    cfg = specenc.make_cfg(tok='mz', pair=args.pair, multi=args.multi, d=args.d, layers=args.layers, multi_peaks=args.multi_peaks)
    aug = specdata.AUG_STD
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = frigenc.FrigEnc(D.nbits, cfg).cuda().train()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    log('token configuration', json.dumps(cfg)); log('parameters', sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW([dict(params=[p for _, p in params if p.ndim >= 2], weight_decay=args.wd),
                             dict(params=[p for _, p in params if p.ndim < 2], weight_decay=0.0)], lr=args.lr, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler('cuda')
    step, hist, active, wall0 = 0, [], 0.0, 0.0
    rfile = P(f'{args.tag}_resume.pt')
    if args.resume and os.path.exists(rfile):
        ck = torch.load(rfile, map_location='cuda', weights_only=False)             # our own file
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        step, hist, active, wall0 = ck['step'], ck['hist'], ck['active'], ck.get('wall', 0.0)
        log('resumed at step', step)
    ledger('resume' if step else 'start', step=step, args=vars(args), enc=cfg, leak=D.leak_msg, excl=excl_msg, torch=torch.__version__, bits_sha1=bits_sha, frig_sha1=frig_sha)

    def save_resume():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, hist=hist,
                        active=active + time.time() - t0 - share.slept, wall=wall0 + time.time() - t_start), rfile + '.tmp')
        os.replace(rfile + '.tmp', rfile)

    stream = FrigStream(data_kw, cfg, args.seed, step, args.bs, args.merge_p, aug, None, excl_path=args.excl)
    if args.workers:
        dl = iter(torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers, prefetch_factor=6, persistent_workers=True))
    else:
        dl = stream.stream(0)
    ELEM = torch.from_numpy(D.elem).cuda(); SID = torch.from_numpy(D.sid).cuda()
    t0 = time.time(); step0 = step; wait = 0.0; ema = {}
    target = torch.zeros(args.bs, dtype=torch.long, device='cuda')
    extra = lambda: dict(args=vars(args), seed=args.seed, bits_sha1=bits_sha, frig_sha1=frig_sha, leak_check=D.leak_msg, exclusion=excl_msg,
                         data='competition train.parquet only', frigid_fp='RDKit AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096), bits, no chirality',
                         licence='ours (weights); training code: clean/train_frigenc.py (Apache-2.0 derived loop), clean/frigenc.py, clean/specenc.py, clean/specdata.py (ours)')
    while step < args.max_steps:
        prog = step / args.max_steps
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g['lr'] = lr
        share.poll()
        if share.must_yield():
            save_resume(); ledger('yield', step=step, other_gb=share.other)
            log(f'YIELD: other processes hold {share.other:.1f} GB of GPU memory -> saved {rfile}, exiting'); T.leave(T.YIELD_CODE)
        tw = time.time(); b, pos, negs, ids = next(dl); wait += time.time() - tw
        ts = time.time()
        b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
        pos = np.asarray(pos); cand = torch.as_tensor(np.concatenate([pos[:, None], np.asarray(negs)], 1), device='cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            z, el, zf = frigenc.forward_bucketed(model, b, args.buckets)
        z = z.float(); el = el.float(); zf = zf.float()
        # window scores in chunks of 64 queries: the same arithmetic as one (B, 64, 10226) block, but no single 640 MB allocation
        # (seed 1's first attempt died of allocator fragmentation at the 8 GB cap on that allocation)
        scs, fp0 = [], []
        for c in range(0, len(pos), 64):
            fc = fpnet.unpack_bits(PFP[cand[c:c + 64]], D.nbits)
            scs.append(torch.einsum('bkn,bn->bk', fc, z[c:c + 64])); fp0.append(fc[:, 0])
        sc = torch.cat(scs); fp0 = torch.cat(fp0)
        yf = fpnet.unpack_bits(FRIG_T[SID[torch.as_tensor(np.asarray(ids), device='cuda')]], frigenc.FRIG_BITS)
        l_rank = F.cross_entropy(sc, target[:len(pos)])
        l_bce = F.binary_cross_entropy_with_logits(z, fp0)
        l_el = F.smooth_l1_loss(el, ELEM[cand[:, 0]])
        l_frig = F.binary_cross_entropy_with_logits(zf, yf)
        loss = l_rank + args.w_bce * l_bce + args.w_el * l_el + args.w_frig * l_frig
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        with torch.no_grad():
            pb = zf > 0; tb = yf > 0.5
            tan = ((pb & tb).sum(1).float() / (pb | tb).sum(1).clamp(min=1).float()).mean()
        vals = dict(loss=loss.item(), rank=l_rank.item(), bce=l_bce.item(), el=l_el.item(), frig=l_frig.item(), acc=(sc.argmax(1) == 0).float().mean().item(), ftan=tan.item())
        share.tick(time.time() - ts)
        for k, v in vals.items():
            ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v
        if not math.isfinite(vals['loss']):
            log('non-finite loss at step', step); ledger('nan', step=step); T.leave(1)
        if step % 200 == 0:
            el_t = time.time() - t0
            log(f"step {step}/{args.max_steps} lr {lr:.2e} " + ' '.join(f'{k} {v:.4f}' for k, v in ema.items()) +
                f" {(step - step0) / el_t:.2f} it/s data-wait {wait / el_t:.2f} duty {share.duty:.2f} gpu-mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if step % 2000 == 0 and step < args.max_steps:
            save_resume()
    active += time.time() - t0 - share.slept
    frigenc.save_ckpt(P(f'{args.tag}_final.pt'), model, step, extra())
    r = {}
    if not args.no_eval:                              # the hold-out window table of the trunk (comparison with the specenc runs)
        ts = time.time(); s0 = share.slept
        sets = T.eval_sets(D, args.eval_groups, np.random.default_rng(12345))
        sets = {k: v for k, v in sets.items() if k in ('hold all', 'hold NP-catalogue', 'np panel (timsTOF)', 'hold timsTOF')}
        r = TS.Evaluator(D, PFP, sets, cfg, args.out, None, None, share).run(model, modes=('true',), ranks_path=P(f'{args.tag}_ranks.parquet'))
        active += time.time() - ts - (share.slept - s0)
    hours = (wall0 + time.time() - t_start) / 3600
    hist.append(dict(step=step, active_hours=active / 3600, wall_hours=hours, final=True, eval=r, train=dict(ema))); log('EVAL', json.dumps(hist[-1]))
    json.dump(dict(eval=r, hist=hist), open(P(f'{args.tag}_eval.json'), 'w'), indent=1)
    ledger('final', step=step, active_hours=round(active / 3600, 3), wall_hours=round(hours, 3), eval=r, train=dict(ema))
    if os.path.exists(rfile):
        os.remove(rfile)
    log('ALL_DONE', f'{active / 3600:.2f} h active, {hours:.2f} h wall')
    T.leave(0)


if __name__ == '__main__':
    try:
        rc = main()
    except BaseException:                         # never leave DataLoader workers behind (T.leave kills the child processes)
        import traceback; traceback.print_exc(); T.leave(1)
    sys.exit(rc)
