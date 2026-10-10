"""Train, time and score the reference joint model: the fast ablation loop (shared GPU, time budget, resume, benchmark scores).

Modes
  load   data-loader throughput only (CPU; no GPU): batches and spectra per second for a variant with --workers processes
  bench  GPU throughput of a variant at several batch sizes: training steps on a repeated batch (model only) and through the loader
         (end to end), peak GPU memory, and inference speed of the two encoders
  train  InfoNCE training for --minutes of wall time or --max_steps; checkpoints every --save_every steps; --resume continues; yields the
         GPU (exit code 75, for train/guard_run.py) when other processes need the memory
  score  score the benchmark's (molecule, candidate) pairs of --half with a checkpoint -> parquet (molecule_id, candidate_key, score) for
         bench.evaluate; the molecule embedding is the mean over all its query spectra
  mine   hard negatives of a checkpoint: for every training structure the --mine_m best-scored wrong candidates among its pool isomers
         (at most --mine_cap, random) and its cached PubChem rows -> npy [n_pool, mine_m] for --mined

Phase 1 (reviews/joint1-2026-10-10.md): --comp_minutes defines a run by GPU-busy time (the cosine schedule and the stop follow the summed
step time, so game mode and yields change the wall time only); the ablation flags below; --out_root moves the run folders; the
leak-safety counts are printed at the start of every training process.

Sharing: train/gpushare.py duty cycle (--duty, --game_duty while a game runs), below-normal priority, a per-process memory cap (--max_gb).
Every GPU run appends its wall seconds to $CASMI_DATA/joint/gpu_ledger.jsonl.

usage: python train/joint/train_joint.py --mode train --variant fp --bs 256 --k 15 --minutes 10 --tag smoke [--resume]
       python train/joint/train_joint.py --mode score --ckpt <...>/smoke/last.pt --half dev --out <scores.parquet>
"""
import argparse, json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import jpaths as P
import data as JD


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['load', 'bench', 'train', 'score', 'mine']); ap.add_argument('--variant', default='fp', choices=['fp', 'gnn'])
    ap.add_argument('--bs', type=int, default=256); ap.add_argument('--bss', default=''); ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--max_spec', type=int, default=4); ap.add_argument('--max_peaks', type=int, default=96); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--frac', type=float, default=1.0); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--warmup', type=int, default=300); ap.add_argument('--max_steps', type=int, default=100000)
    ap.add_argument('--minutes', type=float, default=10.0); ap.add_argument('--save_every', type=int, default=500); ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--steps', type=int, default=40, help='measured steps per configuration (load / bench)')
    ap.add_argument('--duty', type=float, default=0.8); ap.add_argument('--game_duty', type=float, default=0.25); ap.add_argument('--yield_gb', type=float, default=9.0)
    ap.add_argument('--max_gb', type=float, default=8.0); ap.add_argument('--no_amp', action='store_true'); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--tag', default='run'); ap.add_argument('--ckpt', default=''); ap.add_argument('--half', default='dev'); ap.add_argument('--out', default='')
    # phase 1
    ap.add_argument('--comp_minutes', type=float, default=0.0, help='run length in minutes of GPU-busy time (0 = use --minutes of wall time and --max_steps)')
    ap.add_argument('--out_root', default='', help='folder of the run folders (default $CASMI_DATA/joint/runs)')
    ap.add_argument('--neg_mode', default='mix', choices=['mix', 'none', 'pool', 'pc']); ap.add_argument('--mined', default=''); ap.add_argument('--fixed_spec', action='store_true')
    ap.add_argument('--agg', default='mean', choices=['mean', 'attn', 'max', 'merged']); ap.add_argument('--mz_embed', default='fourier', choices=['fourier', 'bin'])
    ap.add_argument('--no_nl', action='store_true'); ap.add_argument('--no_meta', action='store_true'); ap.add_argument('--int_tf', default='both', choices=['both', 'sqrt', 'raw'])
    ap.add_argument('--d', type=int, default=256); ap.add_argument('--layers', type=int, default=6); ap.add_argument('--hidden', default='1024,512')
    ap.add_argument('--loss', default='nce', choices=['nce', 'list', 'margin']); ap.add_argument('--temp', type=float, default=0.07); ap.add_argument('--fix_temp', action='store_true')
    ap.add_argument('--w_m2s', type=float, default=0.25); ap.add_argument('--margin', type=float, default=0.2); ap.add_argument('--w_aux', type=float, default=0.0)
    ap.add_argument('--fp_drop', type=float, default=0.0); ap.add_argument('--aug', type=float, default=0.0); ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--mine_m', type=int, default=32); ap.add_argument('--mine_cap', type=int, default=200)
    return ap.parse_args()


MODEL_KEYS = dict(variant='fp', agg='mean', mz_embed='fourier', no_nl=False, no_meta=False, int_tf='both', d=256, layers=6, hidden='1024,512', loss='nce', temp=0.07, fix_temp=False, w_m2s=0.25,
                  margin=0.2, w_aux=0.0, max_peaks=96)


def model_args_from(a, ck_args):
    """Copy the model-defining arguments of a checkpoint onto `a` (older checkpoints lack the phase-1 keys: reference values)."""
    for k, v in MODEL_KEYS.items():
        setattr(a, k, ck_args.get(k, v))
    return a


def run_root(a):
    return (a.out_root or P.JOINT + '/runs').replace('\\', '/')


def worker_init(_):
    import torch
    torch.set_num_threads(1)
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def loader(a, bs, steps, fixed_spec=False, offset=0):
    import torch
    ds = JD.BatchDataset(bs=bs, k=a.k, max_spec=a.max_spec, max_peaks=a.max_peaks, steps=steps, seed=a.seed, frac=a.frac, graphs=a.variant == 'gnn', fixed_spec=fixed_spec or a.fixed_spec, offset=offset,
                         neg_mode=a.neg_mode, mined=a.mined, merge=a.agg == 'merged')
    if a.workers == 0:
        return ds, (torch.utils.data.default_convert(ds[i]) for i in range(steps))
    return ds, torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers, prefetch_factor=3, persistent_workers=False, worker_init_fn=worker_init)


def to_dev(b, dev):
    import torch
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def ledger(tag, mode, seconds, extra=None):
    os.makedirs(P.JOINT, exist_ok=True)
    with open(P.JOINT + '/gpu_ledger.jsonl', 'a') as f:
        f.write(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), tag=tag, mode=mode, gpu_wall_s=round(seconds, 1), **(extra or {}))) + '\n')


def ledger_total():
    f = P.JOINT + '/gpu_ledger.jsonl'
    return sum(json.loads(l)['gpu_wall_s'] for l in open(f)) if os.path.exists(f) else 0.0


def setup_gpu(a):
    import torch, gpushare
    gpushare.below_normal(); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30; torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gb / tot))
    return gpushare.Share(duty=a.duty, game_duty=a.game_duty, yield_gb=a.yield_gb, log=lambda *x: print(*x, flush=True))


def build(a, dev):
    import model as JM
    meta = json.load(open(JD.IDX + '/meta.json')); nb = int(np.load(P.WORK + '/fp_bits.npy').shape[0])
    m = JM.Joint(a.variant, nbits=nb, n_adduct=max(64, meta['adducts']), n_instr=max(8, meta['instr']), d=a.d, layers=a.layers, hidden=tuple(int(x) for x in a.hidden.split(',')), loss=a.loss,
                 temp=a.temp, fix_temp=a.fix_temp, w_m2s=a.w_m2s, margin=a.margin, w_aux=a.w_aux, mz_embed=a.mz_embed, use_nl=not a.no_nl, use_meta=not a.no_meta, int_tf=a.int_tf,
                 pool='mean' if a.agg == 'merged' else a.agg, fp_drop=getattr(a, 'fp_drop', 0.0), aug=getattr(a, 'aug', 0.0), dropout=getattr(a, 'dropout', 0.1)).to(dev)
    return m, dict(total=JM.n_params(m), spec=JM.n_params(m.spec), struct=JM.n_params(m.struct))


def train_step(m, b, opt, scaler, amp):
    import torch
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
        loss, acc, n = m(b)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update()
    return loss, acc, n


# ---------------------------------------------------------------- modes
def mode_load(a):
    res = []
    for bs in [int(x) for x in (a.bss or str(a.bs)).split(',')]:
        ds, it = loader(a, bs, a.steps + 8); t = None; n = 0; sp = 0; b0 = None
        for i, b in enumerate(it):
            if i == 7:
                t = time.time()                                                  # skip worker start-up
            elif i > 7:
                n += 1; sp += int(b['mz'].shape[0])
            b0 = b
        dt = time.time() - t
        r = dict(variant=a.variant, bs=bs, k=a.k, workers=a.workers, batches_per_s=n / dt, molecules_per_s=n * bs / dt, spectra_per_s=sp / dt, spectra_per_batch=sp / n, peaks_width=int(b0['mz'].shape[1]),
                 structures_per_s=n * bs * (1 + a.k) / dt)
        if a.variant == 'gnn':
            r['atoms_per_batch'] = int(b0['g_atoms'].shape[0])
        print(json.dumps(r), flush=True); res.append(r)
    return res


def mode_bench(a):
    import torch
    share = setup_gpu(a); dev = 'cuda'; t_gpu = time.time(); res = []; amp = not a.no_amp
    for bs in [int(x) for x in (a.bss or str(a.bs)).split(',')]:
        m, npar = build(a, dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01); scaler = torch.amp.GradScaler('cuda', enabled=amp)
        torch.cuda.reset_peak_memory_stats(); ds, it = loader(a, bs, a.steps + 10); r = dict(variant=a.variant, bs=bs, k=a.k, params=npar, amp=amp)
        try:
            it = iter(it); t_wall = None; comp = []; nsp = []; b_keep = None
            for i in range(a.steps + 10):
                b = to_dev(next(it), dev); t0 = time.time()
                loss, acc, n = train_step(m, b, opt, scaler, amp); torch.cuda.synchronize(); dt = time.time() - t0
                if i == 9:
                    t_wall = time.time(); b_keep = b
                if i > 9:
                    comp.append(dt); nsp.append(int(b['mz'].shape[0]))
                share.tick(dt)
            wall = time.time() - t_wall - 0.0
            r.update(spectra_per_batch=float(np.mean(nsp)), step_ms=1e3 * float(np.median(comp)), model_spectra_per_s=float(np.sum(nsp) / np.sum(comp)), model_molecules_per_s=bs * len(comp) / float(np.sum(comp)),
                     loop_spectra_per_s_at_duty=float(np.sum(nsp) / wall), duty=share.duty, loader_wait_share=float(max(0.0, 1 - np.sum(comp) / share.duty / wall)),
                     gpu_gb_allocated=torch.cuda.max_memory_allocated() / 2 ** 30, gpu_gb_reserved=torch.cuda.max_memory_reserved() / 2 ** 30, loss=float(loss), candidates_per_row=int(n))
            del it
        except torch.OutOfMemoryError as e:
            r.update(error='out of memory under the cap of %.0f GB' % a.max_gb)
        print(json.dumps(r), flush=True); res.append(r); del m, opt; torch.cuda.empty_cache()
    # inference speed of the encoders (eval mode, autocast)
    m, npar = build(a, dev); m.eval(); inf = dict(variant=a.variant)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
        ds, it = loader(a, 512, 6, fixed_spec=True); bl = [to_dev(b, dev) for b in it]
        for b in bl[:2]:
            m.spec(b)
        torch.cuda.synchronize(); t0 = time.time(); n = 0
        for b in bl[2:]:
            m.spec(b); n += int(b['mz'].shape[0])
        torch.cuda.synchronize(); inf['spec_encoder_spectra_per_s'] = n / (time.time() - t0)
        for b in bl[:2]:
            m.encode_structs(b)
        torch.cuda.synchronize(); t0 = time.time(); n = 0
        for b in bl[2:]:
            m.encode_structs(b); n += 512 * (1 + a.k)
        torch.cuda.synchronize(); inf['struct_encoder_candidates_per_s'] = n / (time.time() - t0); inf['gpu_gb_reserved'] = torch.cuda.max_memory_reserved() / 2 ** 30
    print(json.dumps(inf), flush=True); sec = time.time() - t_gpu; ledger(a.tag, 'bench_' + a.variant, sec)
    return dict(train=res, inference=inf, gpu_wall_s=sec)


def mode_train(a):
    import torch
    import pandas as pd
    M = pd.read_parquet(P.BENCH + '/mols.parquet'); M.attrs['spec_rows'] = pd.read_parquet(P.BENCH + '/spectra.parquet').row.values
    print('leak-safety:', end=' ', flush=True); JD.assert_leak_safe(M, log=lambda x: print(x, flush=True))        # dies unless 0 identifiers are shared
    out = f'{run_root(a)}/{a.tag}'; os.makedirs(out, exist_ok=True); share = setup_gpu(a); dev = 'cuda'; amp = not a.no_amp; t_gpu = time.time()
    torch.manual_seed(a.seed); m, npar = build(a, dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd); scaler = torch.amp.GradScaler('cuda', enabled=amp)
    step = 0; spent = 0.0; hist = []; comp_tot = 0.0; n_oom = 0; budget = a.comp_minutes * 60.0
    if a.resume and os.path.exists(out + '/last.pt'):
        ck = torch.load(out + '/last.pt', map_location=dev, weights_only=False); m.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        step = ck['step']; spent = ck['spent_s']; hist = ck['hist']; comp_tot = ck.get('comp_s', 0.0); n_oom = ck.get('n_oom', 0); print(f'resumed at step {step}, {spent:.0f}s spent, {comp_tot:.0f}s busy', flush=True)
        if ck.get('done'):
            print('run already complete', flush=True); return 0
    prog = lambda s: min(1.0, comp_tot / budget) if budget > 0 else min(1.0, s / a.max_steps)
    sched = lambda s: a.lr * min(1.0, (s + 1) / a.warmup) * (0.5 * (1 + math.cos(math.pi * prog(s))))
    print(json.dumps(dict(params=npar, args=vars(a))), flush=True)

    def save():
        ck = dict(model=m.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, spent_s=spent + time.time() - t_gpu, hist=hist, args=vars(a), params=npar, comp_s=comp_tot, n_oom=n_oom,
                  done=done)
        torch.save(ck, out + '/last.pt.tmp'); os.replace(out + '/last.pt.tmp', out + '/last.pt')
    done = False
    ds, it = loader(a, a.bs, max(a.max_steps - step, 1), offset=step)                    # batch i of this process = batch (step + i) of the stream
    m.train(); nsp = 0; comp = 0.0; t_log = time.time(); code = 0; lsum = asum = 0.0; nlog = 0; n = 0
    for b in it:
        b = to_dev(b, dev); t0 = time.time()
        for g in opt.param_groups:
            g['lr'] = sched(step)
        try:
            loss, acc, n = train_step(m, b, opt, scaler, amp); torch.cuda.synchronize(); lsum += float(loss); asum += float(acc); nlog += 1
        except torch.OutOfMemoryError:                                            # the 8 GB cap: skip the batch, keep the run
            opt.zero_grad(set_to_none=True); del b; torch.cuda.empty_cache(); n_oom += 1; print(f'out of memory at step {step}: batch skipped ({n_oom} so far)', flush=True); b = None
        dt = time.time() - t0; comp_tot += dt
        step += 1; nsp += int(b['mz'].shape[0]) if b is not None else 0; comp += dt; share.tick(dt)
        if step % a.log_every == 0 and nlog:
            h = dict(step=step, loss=lsum / nlog, acc=asum / nlog, candidates=int(n), scale=float(m.logit_scale.exp()), model_spectra_per_s=nsp / comp, loop_spectra_per_s=nsp / (time.time() - t_log),
                     wall_s=spent + time.time() - t_gpu, gpu_gb=torch.cuda.max_memory_reserved() / 2 ** 30, duty=share.duty, busy_s=comp_tot, lr=sched(step))
            hist.append(h); print(json.dumps(h), flush=True); nsp = 0; comp = 0.0; t_log = time.time(); lsum = asum = 0.0; nlog = 0
        if (budget > 0 and comp_tot >= budget) or (budget <= 0 and spent + time.time() - t_gpu > a.minutes * 60) or step >= a.max_steps:
            done = True; break
        if step % a.save_every == 0:
            save()
        if share.must_yield():
            print('yielding the GPU', flush=True); code = 75; break
        if os.path.exists(out + '/STOP') or os.path.exists(os.path.dirname(out) + '/STOP'):
            print('STOP file: checkpoint and exit', flush=True); code = 76; break
    save(); json.dump(hist, open(out + '/history.json', 'w'), indent=1); sec = time.time() - t_gpu; ledger(a.tag, 'train_' + a.variant, sec, dict(step=step, busy_s=round(comp_tot, 1), done=done, n_oom=n_oom))
    print(f'stopped at step {step}, {sec:.0f}s of GPU wall time in this process', flush=True); return code


def embed_molecules(m, D, rows, own, n, a, dev):
    """L2-normalised embedding of n molecules from spectrum rows (owner = molecule index): the model's own aggregation over all given spectra."""
    import torch
    H = []; O = []
    if a.agg == 'merged':                                                         # one merged peak set per (molecule, adduct, polarity); the sets of a molecule are averaged
        groups = []; go = []
        for i in np.unique(own):
            g = JD.merge_groups(D, rows[own == i]); groups += g; go += [i] * len(g)
        go = np.asarray(go, np.int64)
        for s in range(0, len(groups), 256):
            tk = JD.merged_arrays(D, groups[s:s + 256], go[s:s + 256], a.max_peaks); b = {k: torch.as_tensor(v, device=dev) for k, v in tk.items()}
            H.append(m.spec.spectra(b).float()); O.append(b['owner'])
    else:
        for s in range(0, len(rows), 512):
            tk = D.tokens(rows[s:s + 512], own[s:s + 512], a.max_peaks); b = {k: torch.as_tensor(v, device=dev) for k, v in tk.items()}
            H.append(m.spec.spectra(b).float()); O.append(b['owner'])
    return m.spec.embed(m.spec.pool_spectra(torch.cat(H), torch.cat(O), n))


def mode_mine(a):
    """Top --mine_m wrong candidates of every training structure under a checkpoint (pool isomers, cached PubChem rows)."""
    import torch
    share = setup_gpu(a); dev = 'cuda'; t_gpu = time.time(); ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model_args_from(a, ck['args']); m, _ = build(a, dev); m.load_state_dict(ck['model']); m.eval(); D = JD.JointData(); rng = np.random.default_rng(0)
    tr = np.asarray(D.tr_sid); out = np.full((len(D.pool_mass), a.mine_m), -1, np.int32); n_c = 0; n_top1 = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=not a.no_amp):
        for s in range(0, len(tr), 256):
            sids = tr[s:s + 256]; rows, own, src, idx, co = [], [], [], [], []
            pid = np.asarray(D.sid_pid[np.sort(sids)], np.int64)[np.argsort(np.argsort(sids))]
            for b, sd in enumerate(sids):
                r = D.spectra_of(int(sd)); r = r if len(r) <= 4 else rng.choice(r, 4, replace=False); rows.append(np.sort(r)); own.append(np.full(len(r), b))
                p = int(pid[b]); f = int(D.pool_form[p]); iso = np.asarray(D.form_rows[D.form_ptr[f]:D.form_ptr[f + 1]]); iso = iso[iso != p]
                iso = iso if len(iso) <= a.mine_cap else rng.choice(iso, a.mine_cap, replace=False); j = int(D.pc_of_pid[p]) if D.pc else -1
                pc = np.zeros(0, np.int64)
                if j >= 0:
                    pc = np.concatenate([np.asarray(D.pc_hard[j]), np.asarray(D.pc_rand[j])]); pc = np.unique(pc[pc >= 0])
                src.append(np.r_[np.zeros(1 + len(iso), np.int8), np.ones(len(pc), np.int8)]); idx.append(np.r_[p, iso, pc].astype(np.int64)); co.append(np.full(1 + len(iso) + len(pc), b))
            rows = np.concatenate(rows); own = np.concatenate(own); a.agg = 'mean' if a.agg == 'merged' else a.agg
            zm = embed_molecules(m, D, rows, own, len(sids), a, dev); src = np.concatenate(src); idx = np.concatenate(idx); co = np.concatenate(co); sc = np.zeros(len(idx), np.float32)
            for q in range(0, len(idx), 16384):
                e = m.struct(torch.as_tensor(D.fp(src[q:q + 16384], idx[q:q + 16384]), device=dev)); sc[q:q + 16384] = (e * zm[torch.as_tensor(co[q:q + 16384], device=dev)]).sum(1).float().cpu().numpy()
            st = np.r_[0, np.flatnonzero(np.diff(co)) + 1, len(co)]
            for b in range(len(sids)):
                sl = slice(st[b], st[b + 1]); v = sc[sl]; o = np.argsort(-v[1:])[:a.mine_m] + 1; n_top1 += int(len(v) < 2 or v[0] >= v[1:].max())
                enc = np.where(src[sl][o] == 0, idx[sl][o], -(idx[sl][o] + 2)); out[pid[b], :len(o)] = enc
            n_c += len(idx); share.tick(0.0)
            if (s // 256) % 100 == 0:
                print(json.dumps(dict(done=int(s), of=int(len(tr)), candidates=int(n_c), train_top1=n_top1 / (s + len(sids)), s=time.time() - t_gpu)), flush=True)
    o = a.out or f'{os.path.dirname(a.ckpt)}/mined.npy'; np.save(o + '.tmp.npy', out); os.replace(o + '.tmp.npy', o); sec = time.time() - t_gpu
    r = dict(out=o, structures=int(len(tr)), candidates=int(n_c), train_top1=n_top1 / len(tr), mean_mined=float((out[np.unique(np.asarray(D.sid_pid)[tr])] != -1).sum(1).mean()), gpu_wall_s=sec)
    print(json.dumps(r), flush=True); ledger(a.tag, 'mine', sec); return r


def mode_score(a):
    import torch, pandas as pd
    import bench, cand_fp
    share = setup_gpu(a); dev = 'cuda'; t_gpu = time.time(); ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model_args_from(a, ck['args']); m, _ = build(a, dev); m.load_state_dict(ck['model']); m.eval()
    B = bench.Bench(); pairs = B.pairs(a.half); D = JD.JointData(graphs=False); S = B.spectra(); S = S[S.mid.isin(set(pairs.mid))]
    mids = np.array(sorted(pairs.mid.unique())); mix = {x: i for i, x in enumerate(mids)}; rows = S.row.values; own = S.mid.map(mix).values
    o = np.argsort(own, kind='stable'); rows, own = rows[o], own[o]; t0 = time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=not a.no_amp):
        zm = embed_molecules(m, D, rows, own, len(mids), a, dev)
        torch.cuda.synchronize(); t_spec = time.time() - t0
        cu = np.sort(pairs.cand.unique()); cpid = B.C.set_index('cand').pid.reindex(cu).values; t0 = time.time(); E = torch.zeros(len(cu), zm.shape[1], device=dev)
        if a.variant == 'fp':
            fp = np.zeros((len(cu), D.nbytes), np.uint8); ip = cpid >= 0; fp[ip] = D.pool_fp[np.sort(cpid[ip])][np.argsort(np.argsort(cpid[ip]))]
            cid, cfp, cok = cand_fp.load(a.half); j = np.searchsorted(cid, cu[~ip]); assert (cid[j] == cu[~ip]).all(); fp[~ip] = cfp[j]
            for s in range(0, len(cu), 8192):
                E[s:s + 8192] = m.struct(torch.as_tensor(fp[s:s + 8192], device=dev))
        else:
            import graphs as G
            smi = B.C.set_index('cand').smiles.reindex(cu).values
            for s in range(0, len(cu), 2048):
                g = G.collate([G.featurize(x) for x in smi[s:s + 2048]]); E[s:s + 2048] = m.struct({k: torch.as_tensor(v, device=dev) for k, v in g.items() if k != 'g_ok'})
        torch.cuda.synchronize(); t_cand = time.time() - t0
        ci = np.searchsorted(cu, pairs.cand.values); mi = pairs.mid.map(mix).values; sc = np.zeros(len(pairs), np.float32)
        for s in range(0, len(pairs), 1 << 18):
            sc[s:s + (1 << 18)] = (zm[torch.as_tensor(mi[s:s + (1 << 18)], device=dev)] * E[torch.as_tensor(ci[s:s + (1 << 18)], device=dev)]).sum(1).float().cpu().numpy()
    out = a.out or f'{os.path.dirname(a.ckpt)}/scores_{a.half}.parquet'
    pd.DataFrame(dict(molecule_id=pairs.mid.values, candidate_key=pairs.key.values, score=sc)).to_parquet(out, index=False)
    sec = time.time() - t_gpu; r = dict(out=out, molecules=len(mids), spectra=int(len(rows)), candidates=int(len(cu)), pairs=int(len(pairs)), spec_s=t_spec, spectra_per_s=len(rows) / t_spec,
                                        cand_s=t_cand, candidates_per_s=len(cu) / t_cand, step=ck['step'], gpu_wall_s=sec)
    print(json.dumps(r), flush=True); ledger(a.tag, 'score', sec); return r


if __name__ == '__main__':
    a = args()
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if a.mode != 'score':
        JD.warm()
    if a.mode != 'load':
        print(f'GPU wall seconds used so far (ledger): {ledger_total():.0f}', flush=True)
    r = dict(load=mode_load, bench=mode_bench, train=mode_train, score=mode_score, mine=mode_mine)[a.mode](a)
    if a.out and a.mode in ('load', 'bench'):
        json.dump(r, open(a.out, 'w'), indent=1)
    sys.exit(r if isinstance(r, int) else 0)
