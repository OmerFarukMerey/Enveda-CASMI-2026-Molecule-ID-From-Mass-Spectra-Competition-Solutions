"""Train or evaluate our own fingerprint network (clean/fpnet.py) on train.parquet with the published identity-disjoint split.

Licence and provenance. The loop (loss, schedule, optimiser groups, the hold-out window evaluation and its query groups)
is derived from the public fold-safe training loop: dmitriigluzdov/casmi26-fold-safe-fpnet, train_fpnet.py, Apache-2.0
(see THIRD_PARTY_NOTICES.md), through our train/train_fpnet.py. Everything it imports is ours: clean/fpnet.py (model),
clean/specdata.py (data), train/gpushare.py (GPU sharing). No file of the `casmi` package is imported or was opened.
Changes against the Apache-2.0 loop: our model and data modules, worker processes instead of fork, a step-based
schedule, resume, GPU sharing, a run ledger, per-group rank files, sampling parts / augmentation / freezing options.

Loss: cross-entropy of the true structure against K = 63 pool rows of its +-10 ppm window on f.z, + w_bce x BCE of the
bit logits against the true fingerprint, + w_el x smooth-L1 of the element head against log1p element counts.
Leak safety: specdata.SpecData asserts and prints that no held-out identity is among the training positives.

usage:
  python clean/train_fpnet.py --work <dir> --out <run dir> --tag own_s0 --seed 0 --bs 256 --max_steps 82855 --resume
  python clean/train_fpnet.py --work <dir> --out <run dir> --tag x_ref --eval_only <ckpt.pt>
Exit code 75 = yielded the GPU (train/guard_run.py restarts it with --resume).
"""
import argparse, hashlib, json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'train'))
YIELD_CODE = 75


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True, help='spectrum cache + split.parquet (+ pool tables unless --pool)')
    ap.add_argument('--pool', default=None, help='directory of the pool tables (default: --work)')
    ap.add_argument('--out', required=True, help='run directory (checkpoints, logs, ledger.jsonl)')
    ap.add_argument('--tag', default='own'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_steps', type=int, default=82855); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--warmup', type=int, default=2000); ap.add_argument('--merge_p', type=float, default=0.3)
    ap.add_argument('--w_bce', type=float, default=1.0); ap.add_argument('--w_el', type=float, default=0.1)
    ap.add_argument('--layers', type=int, default=8); ap.add_argument('--d', type=int, default=512)
    ap.add_argument('--workers', type=int, default=3, help='data worker processes')
    ap.add_argument('--buckets', type=int, default=4, help='forward the batch in this many chunks of similar peak count')
    ap.add_argument('--eval_every', type=int, default=20000); ap.add_argument('--eval_groups', type=int, default=3000)
    ap.add_argument('--holdout', default='hold,np')
    ap.add_argument('--aug', default='std', help="'std', 'none' or a JSON dict of specdata augmentation settings")
    ap.add_argument('--parts', default=None, help="sampling parts of the spectrum-uniform half, e.g. 'tims=0.5', 'instr_bal'")
    ap.add_argument('--only', default=None, help="train on one subset only, e.g. 'enveda' (fine-tuning)")
    ap.add_argument('--instr', action='store_true', help='instrument-class input (6 classes); export folds timsTOF into the reference layout')
    ap.add_argument('--init', default=None, help='warm start from a checkpoint'); ap.add_argument('--freeze_below', type=int, default=0,
                    help='with --init: freeze the embeddings and the first N blocks')
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--eval_only', default=None)
    ap.add_argument('--yield_gb', type=float, default=9.0); ap.add_argument('--duty', type=float, default=0.8)
    ap.add_argument('--game_duty', type=float, default=0.25); ap.add_argument('--max_gb', type=float, default=8.0)
    return ap.parse_args(argv)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def leave(code):
    """End the process and its data workers. os._exit alone leaves the DataLoader's worker processes alive on Windows:
    on 2026-10-10 24 such orphans (1.8 GB of committed memory each) grew the page file until drive C: was full."""
    try:
        import psutil
        for c in psutil.Process().children(recursive=True):
            c.kill()
    except Exception:
        pass
    sys.stdout.flush(); os._exit(code)


def groups_of(D, sel, n, rng):
    """(structure, polarity) query groups (<= 8 spectra) built from the selected spectra only; at most n groups."""
    g = {}
    for i in np.where(sel)[0]:
        g.setdefault((int(D.sid[i]), int(D.mode[i])), []).append(int(i))
    keys = sorted(g)
    if len(keys) > n:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), n, replace=False))]
    return [g[k][:8] for k in keys]


def eval_sets(D, n, rng):
    """Hold-out query groups: the 11 subsets of the published table (same draws), then our own Enveda / instrument splits."""
    s = np.where(D.sid >= 0, D.sid, 0)
    in_pool = (D.sid >= 0) & (D.sid2pid[s] >= 0)
    npcat = np.zeros(len(D.sid), bool); npcat[in_pool] = D.psrc[D.sid2pid[D.sid[in_pool]]] >= 2
    f = D.fold[s]; hold = (f == 'hold') & in_pool; npx = (f == 'np') & in_pool
    S = {'hold all': hold, 'hold timsTOF': hold & (D.instr == 0), 'hold other instruments': hold & (D.instr != 0),
         'hold NP-catalogue': hold & npcat, 'hold NP-catalogue positive': hold & npcat & (D.mode > 0),
         'hold NP-catalogue negative': hold & npcat & (D.mode < 0), 'hold not NP-catalogue': hold & ~npcat,
         'np panel (timsTOF)': npx & (D.lib == 9), 'np panel positive': npx & (D.lib == 9) & (D.mode > 0),
         'np panel negative': npx & (D.lib == 9) & (D.mode < 0), 'np structures, other libraries': npx & (D.lib != 9),
         'hold timsTOF NP-catalogue': hold & npcat & (D.instr == 0), 'hold other instruments NP-catalogue': hold & npcat & (D.instr != 0),
         'hold timsTOF not NP-catalogue': hold & ~npcat & (D.instr == 0)}
    return {k: groups_of(D, v, n, rng) for k, v in S.items()}


def window_eval(model, D, PFP, entries, share=None, bs=128, instr=False, yield_exit=False):
    """Rank of the true structure by f.z among the pool rows of its +-10 ppm window, one query group per entry."""
    import torch, fpnet
    model.eval(); ranks, nwin = [], []
    with torch.no_grad():
        for s in range(0, len(entries), bs):
            chunk = entries[s:s + bs]; ts = time.time()
            inp = fpnet.collate([D.query(e) for e in chunk], 'cuda')
            ins = torch.as_tensor(D.instr[[e[0] for e in chunk]], device='cuda') if instr else None
            with torch.autocast('cuda', dtype=torch.float16):
                z = model(*inp, instr=ins)[0].float()
            for r, e in enumerate(chunk):
                pid = D.sid2pid[D.sid[e[0]]]; lo, hi = D.window(pid)
                sc = fpnet.unpack_bits(PFP[lo:hi], D.nbits) @ z[r]
                ranks.append(int((sc > sc[pid - lo]).sum().item()) + 1); nwin.append(hi - lo)
            if share is not None:
                torch.cuda.synchronize(); share.tick(time.time() - ts)
                if yield_exit and share.must_yield():
                    log(f'YIELD: other processes hold {share.other:.1f} GB -> evaluation stopped'); leave(YIELD_CODE)
    model.train()
    ranks = np.array(ranks); nwin = np.array(nwin)
    if not len(ranks):
        return dict(n=0), ranks, nwin
    return dict(n=len(ranks), mrr25=round(float(np.where(ranks <= 25, 1.0 / ranks, 0).mean()), 4), top1=round(float((ranks == 1).mean()), 4),
                top5=round(float((ranks <= 5).mean()), 4), top25=round(float((ranks <= 25).mean()), 4)), ranks, nwin


def forward_bucketed(model, inp, ins, nb):
    """The same forward pass, run in nb chunks of similar peak count, each padded to its own longest spectrum (the output
    of an item does not depend on padding; this only saves the attention over padding)."""
    import torch
    if nb <= 1:
        return model(*inp, instr=ins)
    n = (~inp[2]).sum(1); order = torch.argsort(n); zs, es = [], []
    for c in order.chunk(nb):
        L = max(1, int(n[c].max()))
        a = [x[c] for x in inp]; a[0], a[1], a[2] = a[0][:, :L], a[1][:, :L], a[2][:, :L]
        z, e = model(*a, instr=None if ins is None else ins[c]); zs.append(z); es.append(e)
    inv = torch.empty_like(order); inv[order] = torch.arange(len(order), device=order.device)
    return torch.cat(zs)[inv], torch.cat(es)[inv]


def eval_all(model, D, PFP, sets, share, instr, ranks_path=None, yield_exit=False):
    import pandas as pd
    res, rows = {}, []
    for k, ent in sets.items():
        res[k], rk, nw = window_eval(model, D, PFP, ent, share, instr=instr, yield_exit=yield_exit)
        if len(ent) and ranks_path:
            rows.append(pd.DataFrame(dict(set=k, sid=[int(D.sid[e[0]]) for e in ent], mode=[int(D.mode[e[0]]) for e in ent],
                                          i0=[e[0] for e in ent], n_spec=[len(e) for e in ent], rank=rk, window=nw)))
    if ranks_path:
        pd.concat(rows, ignore_index=True).to_parquet(ranks_path, index=False)
    return res


def main(argv=None):
    args = get_args(argv)
    import gpushare
    gpushare.below_normal()
    import torch
    import torch.nn.functional as F
    import fpnet, specdata
    os.makedirs(args.out, exist_ok=True)
    P = lambda name: os.path.join(args.out, name)
    t_start = time.time()

    def ledger(event, **kw):
        with open(P('ledger.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), tag=args.tag, event=event, wall_s=round(time.time() - t_start, 1), **kw)) + '\n')

    if not args.eval_only and os.path.exists(P(f'{args.tag}_final.pt')):
        log('already finished:', P(f'{args.tag}_final.pt')); return 0
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    share = gpushare.Share(args.duty, args.game_duty, args.yield_gb, log=log)
    holdout = tuple(args.holdout.split(','))
    D = specdata.SpecData(args.work, args.pool, holdout, log=log)
    PFP = torch.from_numpy(np.load(os.path.join(args.pool or args.work, 'pool_fp.npy'))).cuda()
    bits_sha = hashlib.sha1(np.load(os.path.join(args.pool or args.work, 'fp_bits.npy')).tobytes()).hexdigest()
    sets = eval_sets(D, args.eval_groups, np.random.default_rng(12345))
    log({k: len(v) for k, v in sets.items()})
    if args.eval_only:
        model = fpnet.load_ckpt(args.eval_only, 'cuda')
        res = eval_all(model, D, PFP, sets, share, bool(model.n_instr), P(f'{args.tag}_ranks.parquet'), yield_exit=True)
        for k, v in res.items():
            log('EVAL', k, json.dumps(v))
        json.dump(res, open(P(f'{args.tag}_eval.json'), 'w'), indent=1)
        ledger('eval_only', ckpt=args.eval_only, leak=D.leak_msg)
        return 0
    aug = specdata.AUG_STD if args.aug == 'std' else specdata.AUG_NONE if args.aug == 'none' else json.loads(args.aug)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.init:
        ck0 = torch.load(args.init, map_location='cpu', weights_only=True)
        model = fpnet.FPNet(ck0['nbits'], d=ck0['d'], layers=ck0['layers'], heads=ck0.get('heads', 8), n_instr=6 if args.instr else 0)
        log('init', args.init, model.load_state_dict(ck0['model'], strict=False))
        if args.freeze_below:
            for n, p in model.named_parameters():
                blk = int(n.split('.')[1]) if n.startswith('blocks.') else (-1 if n.split('.')[0] in ('int_emb', 'pk', 'ad', 'gl', 'ins') else 10 ** 6)
                p.requires_grad = blk >= args.freeze_below
    else:
        model = fpnet.FPNet(D.nbits, d=args.d, layers=args.layers, n_instr=6 if args.instr else 0)
    assert model.nbits == D.nbits
    model = model.cuda().train()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    log('parameters', sum(p.numel() for p in model.parameters()), 'trainable', sum(p.numel() for _, p in params))
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
    ledger('resume' if step else 'start', step=step, args=vars(args), leak=D.leak_msg, torch=torch.__version__, bits_sha1=bits_sha)

    def save_resume():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, hist=hist,
                        active=active + time.time() - t0 - share.slept, wall=wall0 + time.time() - t_start), rfile + '.tmp')
        os.replace(rfile + '.tmp', rfile)

    stream = specdata.BatchStream(args.work, args.pool, holdout, args.seed, step, args.bs, args.merge_p, aug, args.parts, only=args.only)
    if args.workers:
        dl = iter(torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers, prefetch_factor=6, persistent_workers=True))
    else:
        dl = stream.stream(0)
    ELEM = torch.from_numpy(D.elem).cuda()
    t0 = time.time(); step0 = step; wait = 0.0
    ema = {}
    target = torch.zeros(args.bs, dtype=torch.long, device='cuda')
    while step < args.max_steps:
        prog = step / args.max_steps
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g['lr'] = lr
        share.poll()
        if share.must_yield():
            save_resume(); ledger('yield', step=step, other_gb=share.other)
            log(f'YIELD: other processes hold {share.other:.1f} GB of GPU memory -> saved {rfile}, exiting'); leave(YIELD_CODE)
        tw = time.time(); arrs, pos, negs, ids = next(dl); wait += time.time() - tw
        ts = time.time()
        inp = tuple(a.cuda(non_blocking=True) for a in arrs)
        pos = np.asarray(pos); cand = torch.as_tensor(np.concatenate([pos[:, None], np.asarray(negs)], 1), device='cuda')
        ins = torch.as_tensor(D.instr[np.asarray(ids)], device='cuda') if args.instr else None
        with torch.autocast('cuda', dtype=torch.float16):
            z, el = forward_bucketed(model, inp, ins, args.buckets)
        z = z.float(); el = el.float()
        fp = fpnet.unpack_bits(PFP[cand], D.nbits)
        sc = torch.einsum('bkn,bn->bk', fp, z)
        l_rank = F.cross_entropy(sc, target[:len(pos)])
        l_bce = F.binary_cross_entropy_with_logits(z, fp[:, 0])
        l_el = F.smooth_l1_loss(el, ELEM[cand[:, 0]])
        loss = l_rank + args.w_bce * l_bce + args.w_el * l_el
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        vals = dict(loss=loss.item(), rank=l_rank.item(), bce=l_bce.item(), el=l_el.item(), acc=(sc.argmax(1) == 0).float().mean().item())
        share.tick(time.time() - ts)                    # .item() synchronised; leave the rest of the GPU time to the desktop / the game
        for k, v in vals.items():
            ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v
        if not math.isfinite(vals['loss']):
            log('non-finite loss at step', step); ledger('nan', step=step); leave(1)
        if step % 200 == 0:
            el_t = time.time() - t0
            log(f"step {step}/{args.max_steps} lr {lr:.2e} " + ' '.join(f'{k} {v:.4f}' for k, v in ema.items()) +
                f" {(step - step0) / el_t:.2f} it/s data-wait {wait / el_t:.2f} duty {share.duty:.2f} gpu-mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if step % 2000 == 0 and step < args.max_steps:      # the final checkpoint follows at the last step
            save_resume()
        if step % args.eval_every == 0 and step < args.max_steps:
            ev = eval_all(model, D, PFP, sets, share, args.instr)
            hist.append(dict(step=step, eval=ev)); log('EVAL', json.dumps(hist[-1])); ledger('eval', step=step, top1={k: v.get('top1') for k, v in ev.items()})
    active += time.time() - t0 - share.slept
    ev = eval_all(model, D, PFP, sets, share, args.instr, P(f'{args.tag}_ranks.parquet'))
    hours = (wall0 + time.time() - t_start) / 3600
    hist.append(dict(step=step, active_hours=active / 3600, wall_hours=hours, final=True, eval=ev)); log('EVAL', json.dumps(hist[-1]))
    extra = dict(args=vars(args), seed=args.seed, bits_sha1=bits_sha, leak_check=D.leak_msg, data='competition train.parquet only',
                 licence='ours (weights); training code: clean/train_fpnet.py (Apache-2.0 derived loop), clean/fpnet.py, clean/specdata.py (ours)')
    fpnet.save_ckpt(P(f'{args.tag}_final.pt'), model, step, extra)
    if args.instr:                                      # reference-layout export for timsTOF queries
        mm = fpnet.FPNet(model.nbits, d=model.d, layers=len(model.blocks), heads=model.blocks[0].h)
        mm.load_state_dict(fpnet.fold_instrument(model.state_dict(), 0)); fpnet.save_ckpt(P(f'{args.tag}_final_tims.pt'), mm, step, extra)
    json.dump(dict(eval=ev, hist=hist), open(P(f'{args.tag}_eval.json'), 'w'), indent=1)
    ledger('final', step=step, active_hours=round(active / 3600, 3), wall_hours=round(hours, 3), top1={k: v.get('top1') for k, v in ev.items()})
    if os.path.exists(rfile):
        os.remove(rfile)                                # the final checkpoint is kept, the resume state is not
    log('ALL_DONE', f'{active / 3600:.2f} h active, {hours:.2f} h wall')
    leave(0)


if __name__ == '__main__':
    sys.exit(main())
