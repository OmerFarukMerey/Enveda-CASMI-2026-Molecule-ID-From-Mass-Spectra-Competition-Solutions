"""Fine-tune GLACIER (official ms-pred JointModel and loss) on timsTOF + general [M+H]+ spectra; early stopping on validation isomer MRR.

Model and objective: ms-pred's JointModel loaded from the official MassSpecGym checkpoint, trained with JointModel._common_step(batch,
'train') on contrastive batches (IntenContrDataset.collate_fn layout): end-to-end intensity loss (cosine) + contr_weight x the Sinkhorn
rank-1 contrastive loss (entropy similarity) against decoys, the loss of ms-pred's joint_contr_finetune configs. Optimiser and LR schedule:
JointModel.configure_optimizers() (AdamW, warm-up then x lr_decay_rate every 5000 steps) with our lr / warm-up.
Changes against the official contrastive fine-tune (README): only the intensity head trains (token_mapper, inten_decoder, inten_encoder,
output_map); the Graphormer encoder and the fragment generator are frozen and kept in eval mode, so the fragment loss (frag_weight) is
off and no MAGMa labels are needed; PubChem decoys come from our local pool; plain PyTorch loop instead of Lightning's Trainer (one GPU,
Windows, shared machine).

Validation = the runner's own scoring (gl_runner.py functions: molecule_plan / smiles_ok / Predictor / score_molecule) of the V items
in process; metric = isomer MRR of the truth in its group (ties 0.5). Best checkpoint = highest V MRR; stop after `patience` validations
without a gain >= min_delta, at max_steps, or when the GPU-time budget is used.

Shared machine: below-normal priority, gpushare duty cycle, exit 75 (resume state saved) when others hold > yield_gb (guard_run.py
restarts with --resume); --chunk_min ends a run after that many minutes with the resume state saved (exit 0) so the GPU lock can be
released between chunks.
usage: python train_glacierft.py --out DIR [--resume] [options]
"""
import argparse, importlib.util, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import env  # noqa: E402

RUNNER = os.path.join(env.GLACIER_PKG, 'gl_runner.py')
HEAD = ('token_mapper', 'inten_decoder', 'inten_encoder', 'output_map')
YIELD = 75


def load_runner():
    spec = importlib.util.spec_from_file_location('gl_runner_ft', RUNNER)
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    return R


def isomer_rr(scores, cands, truth):
    import numpy as np
    v = np.array([scores.get(c) if scores.get(c) is not None else -1.0 for c in cands]); t = np.array(truth, bool)
    ts = v[t].max(); dec = v[~t]
    return 1.0 / (1 + (dec > ts).sum() + 0.5 * (dec == ts).sum())


def evaluate_items(model, items, R, dev, bs=16, log=print):
    """Runner scoring of items with the in-memory model -> ({mid: {cand: score}}, n_pred, seconds). Restores the training mode."""
    import torch
    was = {n: m.training for n, m in model.named_children()}
    model.eval(); model._predict_prepared = True                       # skip JointModel.freeze() in predict_inten_frag_batch
    pred = R.Predictor(model, dev, bs, lambda *a: None)
    t0 = time.time(); out = {}
    for mol in items:
        plan = R.molecule_plan(mol)
        cands = [c for c in dict.fromkeys(mol['cands']) if R.smiles_ok(c)[0]]
        if not plan or not cands:
            out[mol['mid']] = {}; continue
        pred.predict(R.molecule_jobs(plan, cands), float('inf'))
        out[mol['mid']] = R.score_molecule(plan, cands, pred.cache)
        pred.cache.clear(); pred.pcache.clear()
    if dev.startswith('cuda'):
        torch.cuda.synchronize()
    for n, m in model.named_children():
        m.train(was[n])
    model._predict_prepared = False
    return out, pred.n_pred, time.time() - t0


def mrr(items, scores):
    import numpy as np
    return float(np.mean([isomer_rr(scores.get(x['mid'], {}), x['cands'], x['truth']) for x in items]))


def set_modes(model):
    model.train()
    for n, m in model.named_children():
        if n not in HEAD:
            m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True); ap.add_argument('--data', default='C:/Users/hemre/casmi-work/glacierft/data')
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--num_decoys', type=int, default=5)
    ap.add_argument('--all_decoy_nums', type=int, default=20); ap.add_argument('--p_tims', type=float, default=0.67)
    ap.add_argument('--epoch_size', type=int, default=64000); ap.add_argument('--max_steps', type=int, default=40000)
    ap.add_argument('--val_every', type=int, default=1000); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--min_delta', type=float, default=0.002); ap.add_argument('--max_gpu_h', type=float, default=8.0)
    ap.add_argument('--chunk_min', type=float, default=0.0); ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--duty', type=float, default=0.8); ap.add_argument('--game_duty', type=float, default=0.25)
    ap.add_argument('--yield_gb', type=float, default=9.0); ap.add_argument('--max_heavy', type=int, default=64)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--contr_threshold', type=float, default=0.4)
    ap.add_argument('--smoke', type=int, default=0, help='run this many steps without validation, report timings, save nothing')
    a = ap.parse_args()
    import gpushare
    gpushare.below_normal()
    import numpy as np, torch, random
    common = env.setup()
    import data as D
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'train.log'), 'a', encoding='utf-8')

    def log(*x):
        s = time.strftime('%H:%M:%S ') + ' '.join(str(v) for v in x)
        print(s, flush=True); logf.write(s + '\n'); logf.flush()

    dev = a.device
    cuda = dev.startswith('cuda')
    mem = lambda: torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    model, hp = env.load_model(device=dev)
    for n, p in model.named_parameters():
        p.requires_grad = n.split('.')[0] in HEAD
    # contrastive-loss settings of ms-pred's configs/glacier/joint_contr_finetune_msg.yaml (train_contr_joint.py forces them after loading)
    model.sk_tau, model.contr_weight, model.inten_weight, model.contr_threshold = 0.05, 1.0, 1.0, a.contr_threshold
    model.frag_weight = 0.0
    model.frag_loss = lambda *x, **k: torch.zeros((), device=dev)      # fragment generator frozen: no fragment loss
    model.lr, model.warmup = a.lr, a.warmup
    opt_cfg = model.configure_optimizers(); opt = opt_cfg['optimizer']; sched = opt_cfg['lr_scheduler']['scheduler']
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_f = os.path.join(a.out, 'resume.pt'); best_f = os.path.join(a.out, 'best.pt')
    st = dict(step=0, epoch=0, pos=0, best=-1.0, best_step=-1, bad=0, gpu_s=0.0, hist=[], done=False, model_step=0)
    if a.resume and os.path.exists(state_f):
        ck = torch.load(state_f, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        st = ck['st']; random.setstate(ck['py_rng']); np.random.set_state(ck['np_rng']); torch.set_rng_state(ck['torch_rng'])
        log('resumed', {k: st[k] for k in ('step', 'epoch', 'pos', 'best', 'best_step', 'bad', 'gpu_s')})
    else:
        log('start', vars(a), f'trainable {n_train:,} of {sum(p.numel() for p in model.parameters()):,}')
    if st['done']:
        log('already finished'); return 0
    model.step = st['model_step']
    set_modes(model)
    R = load_runner()
    V = json.load(open(os.path.join(a.data, 'items_V.json')))
    S = D.Samples(a.data, a.num_decoys, a.all_decoy_nums, a.max_heavy)
    share = gpushare.Share(duty=a.duty, game_duty=a.game_duty, yield_gb=a.yield_gb, log=log)
    t_chunk = time.time()

    def save_state():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), st=st, py_rng=random.getstate(),
                        np_rng=np.random.get_state(), torch_rng=torch.get_rng_state()), state_f + '.tmp')
        os.replace(state_f + '.tmp', state_f)

    def validate():
        share.poll(force=True)
        sc, n, dt = evaluate_items(model, V, R, dev)
        st['gpu_s'] += dt
        m = mrr(V, sc)
        gain = m - st['best']
        st['hist'].append(dict(step=st['step'], v_mrr=m, sec=round(dt, 1), n_pred=n))
        if st['best'] < 0 or gain >= a.min_delta or (st['step'] == 0):
            st['best'], st['best_step'], st['bad'] = max(m, st['best']), st['step'], 0
            if st['step'] > 0:
                env.save_model(model, hp, best_f, meta=dict(step=st['step'], v_mrr=m, lr=a.lr, ms_pred=env.MSPRED_COMMIT))
        else:
            st['bad'] += 1
        log(f"VALID step {st['step']}: V isomer MRR {m:.4f} (best {st['best']:.4f} @ {st['best_step']}, bad {st['bad']}/{a.patience}) "
            f"{n} preds {dt:.0f}s, gpu {st['gpu_s'] / 3600:.2f} h")
        share.tick(dt)

    if st['step'] == 0 and not st['hist'] and not a.smoke:
        validate(); save_state()
    stop = None; t_steps = []; n_used = 0
    while stop is None:
        pairs = S.epoch(st['epoch'], a.epoch_size, a.p_tims, a.seed)
        loader = torch.utils.data.DataLoader(D.EpochData(S, pairs, st['pos']), batch_size=a.bs, shuffle=False, num_workers=a.workers,
                                             collate_fn=D.collate, persistent_workers=False, prefetch_factor=4 if a.workers else None,
                                             multiprocessing_context='spawn' if a.workers else None)
        for b in loader:
            st['pos'] += a.bs
            if b is None:
                continue
            batch, nb = b
            share.poll()
            if share.must_yield():
                save_state(); log('others hold the GPU: saved, exit 75'); sys.exit(YIELD)
            t0 = time.time()
            batch = {k: (R._batch_to_device(v, dev) if isinstance(v, dict) else v.to(dev)) for k, v in batch.items()}
            out = model._common_step(batch, 'train')
            loss = out['loss']
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_value_([p for p in model.parameters() if p.requires_grad], 5.0)   # Trainer gradient_clip_val=5, value
                opt.step()
            sched.step()
            if cuda:
                torch.cuda.synchronize()
            dt = time.time() - t0; st['gpu_s'] += dt; t_steps.append(dt); n_used += nb
            st['step'] += 1; st['model_step'] = model.step
            share.tick(dt)
            if st['step'] % 100 == 0:
                log(f"step {st['step']} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} gpu-step {np.mean(t_steps[-100:]):.3f}s "
                    f"samples/batch {n_used / max(1, len(t_steps)):.1f} gpu {st['gpu_s'] / 3600:.2f} h mem {mem():.2f} GB")
            if a.smoke and st['step'] >= a.smoke:
                wall = time.time() - t_chunk
                log(f'SMOKE {a.smoke} steps: wall {wall:.0f}s ({a.smoke / wall:.2f} steps/s), gpu-step mean {np.mean(t_steps):.3f}s, '
                    f'max mem {mem():.2f} GB'); return 0
            if st['step'] % a.val_every == 0:
                validate()
                if st['bad'] >= a.patience:
                    stop = 'patience'
            if st['step'] >= a.max_steps:
                stop = stop or 'max_steps'
            if st['gpu_s'] >= a.max_gpu_h * 3600:
                stop = stop or 'gpu budget'
            if stop:
                break
            if a.chunk_min and time.time() - t_chunk > a.chunk_min * 60 and st['step'] % 100 == 0:
                save_state(); log(f"chunk end at step {st['step']} (resume state saved)"); return 0
        else:
            st['epoch'] += 1; st['pos'] = 0
        save_state()
    st['done'] = True; st['stop'] = stop
    save_state()
    json.dump(dict(stop=stop, best=st['best'], best_step=st['best_step'], steps=st['step'], gpu_h=st['gpu_s'] / 3600, hist=st['hist'],
                   args=vars(a), trainable=n_train), open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    log('DONE', stop, f"best V MRR {st['best']:.4f} at step {st['best_step']}, gpu {st['gpu_s'] / 3600:.2f} h")
    return 0


if __name__ == '__main__':
    sys.exit(main())
