"""Train or evaluate the configurable spectrum encoder (clean/specenc.py): token-set ablations of our fingerprint network, with kept checkpoints and true / pipeline formula evaluation.

Licence and provenance. The loop (loss, schedule, optimiser groups, hold-out window evaluation and its query groups) is
clean/train_fpnet.py, which is derived from the public fold-safe training loop (dmitriigluzdov/casmi26-fold-safe-fpnet,
train_fpnet.py, Apache-2.0; see THIRD_PARTY_NOTICES.md). Everything it imports is ours: clean/specenc.py, clean/subform.py,
clean/specdata.py, clean/fpnet.py, train/gpushare.py. Added here: token-set options, checkpoints kept at several step
counts with the hold-out table at each of them (learning curves), evaluation under the true and under a
pipeline-assigned formula, the rank among same-formula rows, formula dropout / decoy formulas in training.

Leak safety: specdata.SpecData asserts and prints that no held-out identity is among the training positives.

usage:
  python clean/train_specenc.py --work <dir> --ann <annotation cache> --out <run dir> --tag a_form --tok form --max_steps 16000 --keep 4000,8000,12000 --resume
  python clean/train_specenc.py --work <dir> --ann <dir> --out <run dir> --tag x_name --eval_only <ckpt.pt> [--pipe pipe.parquet] [--dump_pipe pipe.parquet]
Exit code 75 = yielded the GPU (train/guard_run.py restarts it with --resume).
"""
import argparse, hashlib, json, math, os, pickle, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'train')); sys.path.insert(0, HERE)   # clean/ first: train/ has a train_fpnet.py too
import train_fpnet as T  # noqa: E402  (log, leave, eval_sets, YIELD_CODE)


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True); ap.add_argument('--pool', default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--ann', default=None, help='annotation cache of clean/subform.py (needed for --tok form / both)')
    ap.add_argument('--tag', default='enc'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tok', default='mz', choices=['mz', 'form', 'both']); ap.add_argument('--pair', action='store_true')
    ap.add_argument('--multi', action='store_true'); ap.add_argument('--multi_peaks', type=int, default=256, help='peak budget of a --multi query'); ap.add_argument('--rank', action='store_true')
    ap.add_argument('--dreams', default=None, help='(n_spectra, k) float16 .npy of frozen spectrum embeddings -> one extra token')
    ap.add_argument('--form_drop', type=float, default=0.0, help='probability of hiding every formula input of a training item')
    ap.add_argument('--decoy', default=None, help='second annotation cache built with a wrong window formula per spectrum')
    ap.add_argument('--decoy_p', type=float, default=0.0)
    ap.add_argument('--max_steps', type=int, default=16000); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--warmup', type=int, default=2000); ap.add_argument('--merge_p', type=float, default=0.3)
    ap.add_argument('--w_bce', type=float, default=1.0); ap.add_argument('--w_el', type=float, default=0.1)
    ap.add_argument('--layers', type=int, default=8); ap.add_argument('--d', type=int, default=512)
    ap.add_argument('--workers', type=int, default=3); ap.add_argument('--buckets', type=int, default=4)
    ap.add_argument('--keep', default='', help='comma-separated step counts at which a float16 checkpoint is kept and the hold-out table computed')
    ap.add_argument('--eval_groups', type=int, default=3000); ap.add_argument('--holdout', default='hold,np')
    ap.add_argument('--aug', default='std'); ap.add_argument('--parts', default=None)
    ap.add_argument('--pipe', default=None, help='parquet (set, i0, formula_pipe): pipeline-assigned formula per query group')
    ap.add_argument('--dump_pipe', default=None, help='with --eval_only: write that parquet from this network\'s first window row')
    ap.add_argument('--hide_form', action='store_true', help='with --eval_only: evaluate with every formula input hidden')
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--eval_only', default=None)
    ap.add_argument('--yield_gb', type=float, default=9.0); ap.add_argument('--duty', type=float, default=0.8)
    ap.add_argument('--game_duty', type=float, default=0.25); ap.add_argument('--max_gb', type=float, default=8.0)
    return ap.parse_args(argv)


log = T.log


def window_eval(model, D, PFP, FID, entries, cfg, ann, share=None, bs=128, hide=False, demb=None, yield_exit=False):
    """Per query group: rank of the true structure by f.z in its +-10 ppm window, its rank among the rows that share its
    formula, the window size and the pool row ranked first."""
    import torch, fpnet, specenc
    ranks, iso, nwin, top = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(entries), bs):
            chunk = entries[s:s + bs]; ts = time.time()
            items = [specenc.build_item(D, e, cfg, ann, hide_form=hide, demb=None if demb is None else np.asarray(demb[list(e)], np.float32).mean(0)) for e in chunk]
            b = specenc.collate(items, cfg, 'cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                z = model(b)[0].float()
            for r, e in enumerate(chunk):
                pid = D.sid2pid[D.sid[e[0]]]; lo, hi = D.window(pid)
                sc = fpnet.unpack_bits(PFP[lo:hi], D.nbits) @ z[r]; better = sc > sc[pid - lo]
                ranks.append(int(better.sum().item()) + 1); iso.append(int((better & (FID[lo:hi] == FID[pid])).sum().item()) + 1)
                nwin.append(hi - lo); top.append(lo + int(sc.argmax().item()))
            if share is not None:
                torch.cuda.synchronize(); share.tick(time.time() - ts)
                if yield_exit and share.must_yield():
                    log(f'YIELD: other processes hold {share.other:.1f} GB -> evaluation stopped'); T.leave(T.YIELD_CODE)
    ranks = np.array(ranks); iso = np.array(iso)
    if not len(ranks):
        return dict(n=0), ranks, iso, np.array(nwin), np.array(top)
    res = dict(n=len(ranks), mrr25=round(float(np.where(ranks <= 25, 1.0 / ranks, 0).mean()), 4), top1=round(float((ranks == 1).mean()), 4),
               top5=round(float((ranks <= 5).mean()), 4), top25=round(float((ranks <= 25).mean()), 4), iso_top1=round(float((iso == 1).mean()), 4))
    return res, ranks, iso, np.array(nwin), np.array(top)


class Evaluator:
    """Hold-out tables of a network under the true formula and, when a pipeline table is given, under the pipeline formula."""

    def __init__(self, D, PFP, sets, cfg, out_dir, pipe_path=None, demb=None, share=None):
        import torch, pandas as pd, specenc
        self.D, self.PFP, self.sets, self.cfg, self.share, self.demb = D, PFP, sets, cfg, share, demb
        self.FID = torch.as_tensor(pd.factorize(D.pformula)[0], device='cuda')
        self.uses_form = cfg['tok'] != 'mz'; self.pipe = None; self.memo_path = os.path.join(out_dir, 'pipe_memo.pkl'); self.memo = {}
        if pipe_path and self.uses_form:
            p = pd.read_parquet(pipe_path); self.pipe = {(s, int(i)): f for s, i, f in zip(p['set'], p.i0, p.formula_pipe)}
            if os.path.exists(self.memo_path):
                self.memo = pickle.load(open(self.memo_path, 'rb'))
        self.memo_n = len(self.memo)

    def run(self, model, modes=('true', 'pipe'), ranks_path=None, hide=False, yield_exit=False):
        import pandas as pd, specenc
        model.eval(); res, frames = {}, {}
        for mode in modes:
            if mode == 'pipe' and self.pipe is None:
                continue
            res[mode] = {}
            for k, ent in self.sets.items():
                if mode == 'true':
                    ann = self.D.ann
                else:
                    fo = {int(j): self.pipe[(k, e[0])] for e in ent for j in e}
                    ann = specenc.AnnFly(self.D, self.D.adducts, fo); ann.memo = self.memo
                res[mode][k], rk, iso, nw, top = window_eval(model, self.D, self.PFP, self.FID, ent, self.cfg, ann, self.share, hide=hide, demb=self.demb, yield_exit=yield_exit)
                if len(ent) and ranks_path:
                    f = frames.setdefault(k, pd.DataFrame(dict(set=k, sid=[int(self.D.sid[e[0]]) for e in ent], mode=[int(self.D.mode[e[0]]) for e in ent],
                                                               i0=[e[0] for e in ent], n_spec=[len(e) for e in ent], window=nw)))
                    f[f'rank_{mode}'] = rk; f[f'iso_{mode}'] = iso; f[f'top_{mode}'] = top
        if ranks_path and frames:
            d = pd.concat(list(frames.values()), ignore_index=True)
            if 'rank_true' in d:
                d['rank'] = d['rank_true']                       # the column name of the earlier rank files
            d.to_parquet(ranks_path, index=False)
        if self.pipe is not None and len(self.memo) > self.memo_n:
            pickle.dump(self.memo, open(self.memo_path + '.tmp', 'wb'), protocol=4); os.replace(self.memo_path + '.tmp', self.memo_path); self.memo_n = len(self.memo)
        model.train()
        return res


def main(argv=None):
    args = get_args(argv)
    import gpushare
    gpushare.below_normal()
    import torch
    import torch.nn.functional as F
    import fpnet, specdata, specenc
    os.makedirs(args.out, exist_ok=True)
    P = lambda name: os.path.join(args.out, name)
    t_start = time.time()

    def ledger(event, **kw):
        with open(P('ledger.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), tag=args.tag, pid=os.getpid(), event=event, wall_s=round(time.time() - t_start, 1), **kw)) + '\n')

    if not args.eval_only and os.path.exists(P(f'{args.tag}_final.pt')):
        log('already finished:', P(f'{args.tag}_final.pt')); return 0
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    share = gpushare.Share(args.duty, args.game_duty, args.yield_gb, log=log)
    holdout = tuple(args.holdout.split(','))
    data_kw = dict(work=args.work, pool=args.pool, holdout=holdout, ann_dir=args.ann)
    D = specenc.EncData(log=log, need_adducts=True, **data_kw)
    PFP = torch.from_numpy(np.load(os.path.join(args.pool or args.work, 'pool_fp.npy'))).cuda()
    bits_sha = hashlib.sha1(np.load(os.path.join(args.pool or args.work, 'fp_bits.npy')).tobytes()).hexdigest()
    sets = T.eval_sets(D, args.eval_groups, np.random.default_rng(12345))
    log({k: len(v) for k, v in sets.items()})
    demb = np.load(args.dreams, mmap_mode='r') if args.dreams else None

    if args.eval_only:
        model = specenc.load_ckpt(args.eval_only, 'cuda'); cfg = model.cfg
        assert cfg['tok'] == 'mz' or D.ann is not None, '--ann is needed for a formula token set'
        ev = Evaluator(D, PFP, sets, cfg, args.out, args.pipe, demb, share)
        res = ev.run(model, ranks_path=P(f'{args.tag}_ranks.parquet'), hide=args.hide_form, yield_exit=True)
        for mode, r in res.items():
            for k, v in r.items():
                log('EVAL', mode, k, json.dumps(v))
        json.dump(res, open(P(f'{args.tag}_eval.json'), 'w'), indent=1)
        if args.dump_pipe:
            import pandas as pd
            d = pd.read_parquet(P(f'{args.tag}_ranks.parquet'))
            d['formula_pipe'] = D.pformula[d.top_true.values]; d['formula_true'] = D.pformula[D.sid2pid[d.sid.values]]
            d[['set', 'i0', 'sid', 'n_spec', 'formula_pipe', 'formula_true']].to_parquet(args.dump_pipe, index=False)
            log('pipeline formulas written:', args.dump_pipe, 'correct', round(float((d.formula_pipe == d.formula_true).mean()), 4))
        ledger('eval_only', ckpt=args.eval_only, leak=D.leak_msg, enc=cfg, hide_form=args.hide_form)
        return 0

    cfg = specenc.make_cfg(tok=args.tok, pair=args.pair, multi=args.multi, rank=args.rank, d=args.d, layers=args.layers, multi_peaks=args.multi_peaks,
                           dreams=int(demb.shape[1]) if demb is not None else 0)
    assert cfg['tok'] == 'mz' or D.ann is not None, '--ann is needed for a formula token set'
    aug = specdata.AUG_STD if args.aug == 'std' else specdata.AUG_NONE if args.aug == 'none' else json.loads(args.aug)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = specenc.SpecEnc(D.nbits, cfg).cuda().train()
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
    ledger('resume' if step else 'start', step=step, args=vars(args), enc=cfg, leak=D.leak_msg, torch=torch.__version__, bits_sha1=bits_sha)
    keep = sorted(int(x) for x in args.keep.split(',') if x)
    ev = Evaluator(D, PFP, sets, cfg, args.out, args.pipe, demb, share)

    def save_resume():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, hist=hist,
                        active=active + time.time() - t0 - share.slept, wall=wall0 + time.time() - t_start), rfile + '.tmp')
        os.replace(rfile + '.tmp', rfile)

    stream = specenc.EncStream(data_kw, cfg, args.seed, step, args.bs, args.merge_p, aug, args.parts, args.form_drop, args.decoy, args.decoy_p, args.dreams)
    if args.workers:
        dl = iter(torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers, prefetch_factor=6, persistent_workers=True))
    else:
        dl = stream.stream(0)
    ELEM = torch.from_numpy(D.elem).cuda()
    t0 = time.time(); step0 = step; wait = 0.0; ema = {}
    target = torch.zeros(args.bs, dtype=torch.long, device='cuda')
    extra = lambda: dict(args=vars(args), seed=args.seed, bits_sha1=bits_sha, leak_check=D.leak_msg, data='competition train.parquet only',
                         licence='ours (weights); training code: clean/train_specenc.py (Apache-2.0 derived loop), clean/specenc.py, clean/subform.py, clean/specdata.py (ours)')
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
            z, el = specenc.forward_bucketed(model, b, args.buckets)
        z = z.float(); el = el.float()
        try:
            fp = fpnet.unpack_bits(PFP[cand], D.nbits)
        except torch.OutOfMemoryError:                  # fragmentation at the 8 GB cap: release the cached blocks and try once more
            torch.cuda.empty_cache(); fp = fpnet.unpack_bits(PFP[cand], D.nbits)
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
        share.tick(time.time() - ts)
        for k, v in vals.items():
            ema[k] = v if k not in ema else 0.98 * ema[k] + 0.02 * v
        if not math.isfinite(vals['loss']):
            log('non-finite loss at step', step); ledger('nan', step=step); T.leave(1)
        if step % 200 == 0:
            el_t = time.time() - t0
            log(f"step {step}/{args.max_steps} lr {lr:.2e} " + ' '.join(f'{k} {v:.4f}' for k, v in ema.items()) +
                f" {(step - step0) / el_t:.2f} it/s data-wait {wait / el_t:.2f} duty {share.duty:.2f} gpu-mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        if step in keep and step < args.max_steps:
            specenc.save_ckpt(P(f'{args.tag}_s{step}.pt'), model, step, extra(), half=True)
            r = ev.run(model, modes=('true',))
            hist.append(dict(step=step, eval=r, train=dict(ema))); log('EVAL', json.dumps(hist[-1]))
            ledger('eval', step=step, top1={k: v.get('top1') for k, v in r['true'].items()}, train=dict(ema))
            save_resume()
        elif step % 2000 == 0 and step < args.max_steps:
            save_resume()
    active += time.time() - t0 - share.slept
    r = ev.run(model, ranks_path=P(f'{args.tag}_ranks.parquet'))
    hours = (wall0 + time.time() - t_start) / 3600
    hist.append(dict(step=step, active_hours=active / 3600, wall_hours=hours, final=True, eval=r, train=dict(ema))); log('EVAL', json.dumps(hist[-1]))
    specenc.save_ckpt(P(f'{args.tag}_final.pt'), model, step, extra())
    json.dump(dict(eval=r, hist=hist), open(P(f'{args.tag}_eval.json'), 'w'), indent=1)
    ledger('final', step=step, active_hours=round(active / 3600, 3), wall_hours=round(hours, 3),
           top1={m: {k: v.get('top1') for k, v in rr.items()} for m, rr in r.items()})
    if os.path.exists(rfile):
        os.remove(rfile)
    log('ALL_DONE', f'{active / 3600:.2f} h active, {hours:.2f} h wall')
    T.leave(0)


if __name__ == '__main__':
    sys.exit(main())
