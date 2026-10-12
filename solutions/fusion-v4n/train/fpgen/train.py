"""Trainer of fpgen: teacher forcing on SMILES conditioned on a corrupted fingerprint and the formula, with GPU sharing, resume, a ledger and leak assertions in every run.

Data workers are fixed-size, below normal priority, and exit with the parent (no orphans). The run yields the GPU (exit code 75, resume
file written) when other processes hold more than --yield_gb; train/guard_run.py restarts it.
usage: python train.py --tag ref0 --max_steps 20000 --bs 96 [--arm pool] [--noise 1.0] [--resume]
"""
import argparse, json, os, signal, sys, time

import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpaths as P, tok as T, data as D, model as MD, sets as S
import gpushare

YIELD = 75


def _worker(q, seed, args, idx):
    P.low_priority(); P.watch_parent()
    import pandas as pd
    U = pd.read_parquet(args['univ'])
    V = T.Vocab.load(P.TOK + '/vocab.json')
    B = D.Batches(U, V, seed=seed, bs=args['bs'], aug=args['aug'], noise_scale=args['noise'])
    while True:
        q.put(B.batch())


class Stream:
    """Batches from `workers` processes (0 = in this process)."""

    def __init__(self, univ_path, seed, bs, aug, noise, workers=2):
        import multiprocessing as mp
        self.workers = workers; self.procs = []
        if workers:
            ctx = mp.get_context('spawn'); self.q = ctx.Queue(maxsize=8)
            for i in range(workers):
                p = ctx.Process(target=_worker, args=(self.q, seed * 1000 + i, dict(univ=univ_path, bs=bs, aug=aug, noise=noise), i), daemon=True)
                p.start(); self.procs.append(p)
        else:
            import pandas as pd
            self.B = D.Batches(pd.read_parquet(univ_path), T.Vocab.load(P.TOK + '/vocab.json'), seed=seed, bs=bs, aug=aug, noise_scale=noise)

    def next(self):
        return self.q.get() if self.workers else self.B.batch()

    def close(self):
        for p in self.procs:
            try:
                p.terminate(); p.join(timeout=5)
            except Exception:
                pass
        self.procs = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--arm', default='pool'); ap.add_argument('--pc', action='store_true'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_steps', type=int, default=20000); ap.add_argument('--bs', type=int, default=96); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup', type=int, default=1000); ap.add_argument('--d', type=int, default=512); ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8); ap.add_argument('--n_fp', type=int, default=4); ap.add_argument('--drop', type=float, default=0.1)
    ap.add_argument('--aug', type=float, default=1.0); ap.add_argument('--noise', type=float, default=1.0)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--duty', type=float, default=0.8); ap.add_argument('--game_duty', type=float, default=0.25)
    ap.add_argument('--yield_gb', type=float, default=9.0); ap.add_argument('--mem_gb', type=float, default=8.0)
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--log_every', type=int, default=100); ap.add_argument('--save_every', type=int, default=2000)
    a = ap.parse_args()
    P.mk(); P.low_priority(); torch.set_num_threads(2)
    run = P.RUNS + '/' + a.tag; os.makedirs(run, exist_ok=True)
    univ = run + '/universe.parquet'
    log_f = open(run + '/log.txt', 'a')

    def log(*x, **k):
        s = ' '.join(str(y) for y in x); print(s, flush=True); log_f.write(s + '\n'); log_f.flush()
    if not os.path.exists(univ):
        U = D.universe(a.arm, log=log); V = D.build_vocab(U, log=log)
        if a.pc:
            U = D.with_pubchem(U, log=log)
        U = D.encodable(U, V, log=log); U.to_parquet(univ, index=False)
    else:
        import pandas as pd
        U = pd.read_parquet(univ, columns=['pid']); V = T.Vocab.load(P.TOK + '/vocab.json')
        log(S.leak_check(U.pid.values, log=lambda *x, **k: None, extra=' (resumed run %s, arm %s)' % (a.tag, a.arm)))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda'
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)))
    m = MD.FPGen(len(V), d=a.d, layers=a.layers, heads=a.heads, drop=a.drop, n_fp=a.n_fp).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda'); step = 0
    ck = run + '/resume.pt'
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        m.load_state_dict(st['model']); opt.load_state_dict(st['opt']); scaler.load_state_dict(st['scaler']); step = st['step']
        log('resumed at step', step)
    log('model: %d parameters, vocab %d, d %d, layers %d, batch %d, arm %s, noise %.2f, seed %d'
        % (MD.n_params(m), len(V), a.d, a.layers, a.bs, a.arm, a.noise, a.seed))
    share = gpushare.Share(a.duty, a.game_duty, a.yield_gb, log=log)
    st = Stream(univ, a.seed + step, a.bs, a.aug, a.noise, a.workers)
    t0 = time.time(); t_start = time.time(); losses = []; tok_n = 0; yielded = False

    def save(path, final=False):
        torch.save(dict(model=m.state_dict(), opt=None if final else opt.state_dict(), scaler=None if final else scaler.state_dict(), step=step,
                        args=vars(a), vocab=V.itos[3:], nbits=m.nbits), path)
    try:
        while step < a.max_steps:
            fp, cnt, x = st.next()
            fp = torch.from_numpy(fp).to(dev, non_blocking=True); cnt = torch.from_numpy(cnt).to(dev); x = torch.from_numpy(x.astype(np.int64)).to(dev)
            lr = a.lr * min(1.0, (step + 1) / max(a.warmup, 1)) * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * min(1.0, step / a.max_steps))))
            for g in opt.param_groups:
                g['lr'] = lr
            t = time.time()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                z = m(fp, cnt, x)[:, :-1]                      # row i predicts x[i + 1]
                tgt = x[:, 1:]
                loss = torch.nn.functional.cross_entropy(z.reshape(-1, z.shape[-1]).float(), tgt.reshape(-1), ignore_index=T.PAD)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update()
            torch.cuda.synchronize(); share.tick(time.time() - t)
            losses.append(float(loss)); tok_n += int((tgt != T.PAD).sum()); step += 1
            if step % a.log_every == 0:
                log('step %d loss %.4f lr %.2e %.2f steps/s mem %.1f GB %s' % (step, float(np.mean(losses[-a.log_every:])), lr, step and a.log_every / max(time.time() - t0, 1e-9),
                                                                               torch.cuda.max_memory_reserved() / 2 ** 30, 'game' if share.games else ''))
                t0 = time.time()
            if step % a.save_every == 0:
                save(ck)
            if share.must_yield():
                save(ck); yielded = True; log('yielding the GPU at step', step); break
    finally:
        st.close()
    if not yielded:
        save(run + '/final.pt', final=True)
    with open(P.LEDGER, 'a') as f:
        f.write(json.dumps(dict(tag=a.tag, step=step, max_steps=a.max_steps, loss=float(np.mean(losses[-200:])) if losses else None,
                                arm=a.arm, noise=a.noise, bs=a.bs, seed=a.seed, params=MD.n_params(m), tokens=tok_n,
                                wall_s=round(time.time() - t_start), yielded=yielded, t=time.strftime('%Y-%m-%d %H:%M'))) + '\n')
    sys.exit(YIELD if yielded else 0)


if __name__ == '__main__':
    main()
