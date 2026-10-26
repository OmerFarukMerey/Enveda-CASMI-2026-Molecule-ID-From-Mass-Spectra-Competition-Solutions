"""Trainer of the ownneural model (clean/onnet.py): listwise + in-batch InfoNCE on train.parquet structures outside the evaluation scaffolds.

A run is defined by GPU-busy minutes (--comp_minutes; cosine schedule over that budget). Every --eval_every steps the model scores the fixed
Enveda validation lists (ondata.val_lists: scaffold-held-out structures, their Enveda timsTOF spectra only, 1 true + 24 hard candidates);
best.pt keeps the best validation MRR (the pre-registered model-selection rule), last.pt allows --resume. --chunk_minutes ends the process
after that many busy minutes of this process (exit 0, not done) so the runner can release the GPU lock between chunks. Exit 75 = the GPU was
yielded to another process (train/guard_run.py restarts with --resume). Shared GPU: train/gpushare.py duty cycle and memory cap.
Every process appends its wall / busy seconds to $CASMI_DATA/ownneural/gpu_ledger.jsonl.

usage: python train_on.py --tag on_s0 --seed 0 --comp_minutes 75 [--resume] [--chunk_minutes 60]
       python train_on.py --tag bench --bench 30        (throughput and memory only, no checkpoint)
"""
import argparse, json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import onpaths as P  # noqa: E402
import ondata as OD  # noqa: E402
import onnet  # noqa: E402  (clean/, put on the path by ondata)


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--bs', type=int, default=192); ap.add_argument('--k', type=int, default=31); ap.add_argument('--max_spec', type=int, default=8)
    ap.add_argument('--max_peaks', type=int, default=96); ap.add_argument('--workers', type=int, default=3); ap.add_argument('--prefetch', type=int, default=2)
    ap.add_argument('--w_env', type=float, default=1.5); ap.add_argument('--p_env', type=float, default=0.7); ap.add_argument('--w_envview', type=float, default=1.5)
    ap.add_argument('--d', type=int, default=256); ap.add_argument('--layers', type=int, default=6); ap.add_argument('--hidden', default='1024,512')
    ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--pool', default='attn'); ap.add_argument('--temp', type=float, default=0.05)
    ap.add_argument('--w_list', type=float, default=1.0); ap.add_argument('--w_nce', type=float, default=0.5); ap.add_argument('--w_m2s', type=float, default=0.25)
    ap.add_argument('--w_aux', type=float, default=0.0)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--comp_minutes', type=float, default=75.0); ap.add_argument('--chunk_minutes', type=float, default=0.0)
    ap.add_argument('--eval_every', type=int, default=1000); ap.add_argument('--save_every', type=int, default=500); ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--duty', type=float, default=0.85); ap.add_argument('--game_duty', type=float, default=0.25); ap.add_argument('--yield_gb', type=float, default=7.0)
    ap.add_argument('--max_gb', type=float, default=8.0); ap.add_argument('--bench', type=int, default=0)
    return ap.parse_args()


def ledger(tag, mode, wall, busy, extra=None):
    os.makedirs(P.ON, exist_ok=True)
    with open(P.ON + '/gpu_ledger.jsonl', 'a') as f:
        f.write(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), tag=tag, mode=mode, gpu_wall_s=round(wall, 1), busy_s=round(busy, 1), pid=os.getpid(), **(extra or {}))) + '\n')


def model_from(a):
    meta = json.load(open(P.JIDX + '/meta.json')); nb = int(np.load(P.WORK + '/fp_bits.npy').shape[0])
    return onnet.ONNet(nbits=nb, d=a.d, layers=a.layers, n_adduct=max(64, meta['adducts']), n_instr=max(8, meta['instr']), hidden=tuple(int(x) for x in a.hidden.split(',')),
                       dropout=a.dropout, pool=a.pool, temp=a.temp, w_list=a.w_list, w_nce=a.w_nce, w_m2s=a.w_m2s, w_aux=a.w_aux)


def winit(_):
    import torch
    torch.set_num_threads(1)
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def validate(m, D, V, dev, max_peaks):
    import torch
    m.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        arr = D.arrays(V['rows'], V['owner'], max_peaks); zs, _ = onnet.embed_molecules(m, arr, len(V['sids']), dev)
        ze = onnet.embed_structs(m, D.J.fp(V['src'], V['idx']), dev)
        sc = (zs[torch.as_tensor(V['cowner'], device=dev)] * ze).sum(1).float().cpu().numpy()
    m.train(); mrr, rr = OD.list_mrr(sc, V['cowner'])
    return dict(val_mrr=mrr, val_top1=float((rr == 1.0).mean()))


def main():
    a = args()
    import torch
    sys.path.insert(0, os.path.dirname(HERE)); import gpushare
    gpushare.below_normal(); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dev = 'cuda'; tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30; torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gb / tot))
    share = gpushare.Share(duty=a.duty, game_duty=a.game_duty, yield_gb=a.yield_gb, log=lambda *x: print(*x, flush=True))
    out = f'{P.RUNS}/{a.tag}'; os.makedirs(out, exist_ok=True); t_gpu = time.time()
    torch.manual_seed(a.seed); m = model_from(a).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd); scaler = torch.amp.GradScaler('cuda')
    step, busy, hist, best, done = 0, 0.0, [], -1.0, False
    if a.resume and os.path.exists(out + '/last.pt'):
        ck = torch.load(out + '/last.pt', map_location=dev, weights_only=False); m.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        step, busy, hist, best, done = ck['step'], ck['busy_s'], ck['hist'], ck['best'], ck.get('done', False)
        print(f'resumed at step {step}, busy {busy:.0f}s, best val {best:.4f}', flush=True)
        if done:
            print('run already complete', flush=True); return 0
    budget = a.comp_minutes * 60.0
    sched = lambda s, b: a.lr * min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, b / budget)))
    n_par = sum(p.numel() for p in m.parameters()); print(json.dumps(dict(params=n_par, args=vars(a))), flush=True)
    JD = OD.JD; JD.warm()
    D = OD.OnData(); V = OD.val_lists(D); print(f'train structures {len(D.tr)}, val lists {len(V["sids"])} ({len(V["idx"])} candidates)', flush=True)
    ds = OD.BatchDataset(bs=a.bs, k=a.k, max_spec=a.max_spec, max_peaks=a.max_peaks, steps=10 ** 7, seed=a.seed, offset=step, w_env=a.w_env, p_env=a.p_env, w_envview=a.w_envview)

    it = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers, prefetch_factor=a.prefetch, persistent_workers=False, worker_init_fn=winit, timeout=300)

    def save(path_best=False):
        ck = dict(model=m.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, busy_s=busy, hist=hist, best=best, done=done, args=vars(a), cfg=m.cfg)
        torch.save(ck, out + '/last.pt.tmp'); os.replace(out + '/last.pt.tmp', out + '/last.pt')
    if a.bench:
        torch.cuda.reset_peak_memory_stats()
    m.train(); t_proc = 0.0; code = 0; lsum = asum = 0.0; nlog = 0; nsp = 0; t_log = time.time()
    src = iter(it)
    while True:
        try:
            b = next(src)
        except StopIteration:
            break
        except RuntimeError as e:                                             # a worker died or timed out (e.g. no commit memory left): yield
            print(f'data loader failed at step {step}: {str(e)[:200]} -> checkpoint and exit 75', flush=True); code = 75; break
        b = onnet.to_torch(b, dev); t0 = time.time()
        for g in opt.param_groups:
            g['lr'] = sched(step, busy)
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                loss, acc = m(b)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update()
            torch.cuda.synchronize(); lsum += float(loss); asum += float(acc); nlog += 1; nsp += int(b['mz'].shape[0])
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); del b; torch.cuda.empty_cache(); print(f'out of memory at step {step}: batch skipped', flush=True)
        dt = time.time() - t0; busy += dt; t_proc += dt; step += 1; share.tick(dt)
        if a.bench and step >= a.bench:
            r = dict(bench_steps=step, step_ms=1e3 * t_proc / step, gpu_gb=torch.cuda.max_memory_reserved() / 2 ** 30, spectra_per_batch=nsp / max(nlog, 1),
                     wall_s=time.time() - t_gpu, loss=lsum / max(nlog, 1))
            print(json.dumps(r), flush=True); ledger(a.tag, 'bench', time.time() - t_gpu, t_proc); return 0
        if step % a.log_every == 0 and nlog:
            h = dict(step=step, loss=lsum / nlog, acc=asum / nlog, scale=float(m.logit_scale.exp()), lr=sched(step, busy), busy_s=round(busy), spectra_per_batch=nsp / nlog,
                     steps_per_wall_s=nlog / (time.time() - t_log), gpu_gb=round(torch.cuda.max_memory_reserved() / 2 ** 30, 2), duty=share.duty)
            hist.append(h); print(json.dumps(h), flush=True); lsum = asum = 0.0; nlog = 0; nsp = 0; t_log = time.time()
        finished = busy >= budget
        if step % a.eval_every == 0 or finished:
            t0 = time.time(); v = validate(m, D, V, dev, a.max_peaks); v.update(step=step, busy_s=round(busy)); hist.append(v); print(json.dumps(v), flush=True)
            if v['val_mrr'] > best:
                best = v['val_mrr']; onnet.save_ckpt(out + '/best.pt', m, dict(step=step, val=v, args=vars(a)))
            busy += time.time() - t0
        if finished:
            done = True; break
        if step % a.save_every == 0:
            save()
        if share.must_yield():
            print('yielding the GPU', flush=True); code = 75; break
        if a.chunk_minutes > 0 and t_proc >= a.chunk_minutes * 60:
            print('chunk done: checkpoint and exit', flush=True); code = 0; break
        if os.path.exists(out + '/STOP') or os.path.exists(P.RUNS + '/STOP'):
            print('STOP file', flush=True); code = 76; break
    save(); json.dump(hist, open(out + '/history.json', 'w'), indent=1)
    if done:
        onnet.save_ckpt(out + '/final.pt', m, dict(step=step, args=vars(a)))
        open(out + '/DONE', 'w').write(json.dumps(dict(step=step, busy_s=busy, best_val=best)))
    ledger(a.tag, 'train', time.time() - t_gpu, t_proc, dict(step=step, done=done))
    print(f'stopped at step {step}, busy {busy:.0f}s, best val {best:.4f}, done {done}', flush=True); return code


if __name__ == '__main__':
    sys.exit(main())
