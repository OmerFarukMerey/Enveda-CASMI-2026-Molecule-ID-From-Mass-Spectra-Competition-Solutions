"""isoxe v2 trainer: listwise softmax over NP same-formula groups (positive first), equal group weights, best epoch by V isomer MRR.

Same loop as train/isoxe/train_isoxe.py (GPU shared through train/gpushare.py; exit 75 after a checkpoint when others need the
memory, train/guard_run.py restarts with --resume; checkpoint every epoch) with model2.XE2 / Store2. `--mode score`: per-candidate
scores (mean over the stored spectra, at most --max_spec) of a feature directory for one or more checkpoints.

    python train/isoxe2/train_isoxe2.py --mode train --tag s0 --seed 0 [--epochs 15] [--no_graph]
    python train/isoxe2/train_isoxe2.py --mode score --ckpt <run>/best.pt[,<run2>/best.pt] --feat <feat dir> --out scores.npz
Features: $ISOXE2_W (default D:/casmi-data/isoxe2)/feat_{train,val}; runs: $ISOXE2_RUNS (default C:/Users/hemre/casmi-work/isoxe2/runs).
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import model2 as M  # noqa: E402

FEAT = os.environ.get('ISOXE2_W', 'D:/casmi-data/isoxe2').replace('\\', '/')
RUNS = os.environ.get('ISOXE2_RUNS', 'C:/Users/hemre/casmi-work/isoxe2/runs').replace('\\', '/')


def log(path, **k):
    k['t'] = time.strftime('%H:%M:%S'); print(json.dumps(k), flush=True)
    open(path, 'a').write(json.dumps(k) + '\n')


@torch.no_grad()
def score_store(net, S, device, bs=32, max_spec=16):
    net.eval(); out = np.zeros(int(S.ncand.sum()), np.float32)
    for i in range(0, S.n, bs):
        g = np.arange(i, min(S.n, i + bs))
        D, own, cmax = S.gather(g, [np.arange(min(int(S.nspec[x]), max_spec)) for x in g], device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            lg = net(**D).float()
        sc = M.group_scores(lg, own, len(g), cmax).cpu().numpy()
        for b, x in enumerate(g):
            out[S.c0[x]:S.c0[x] + S.ncand[x]] = sc[b, :S.ncand[x]]
    return out


def iso_mrr(S, sc):
    return np.array([M.isomer_rr(sc[S.c0[g]:S.c0[g] + S.ncand[g]], S.lab[S.c0[g]:S.c0[g] + S.ncand[g]]) for g in range(S.n)])


def make_net(a):
    return M.XE2(d=a['d'], layers=a['layers'], heads=a['heads'], ff=a['ff'], dropout=a['dropout'], use_graph=not a['no_graph'], dg=a['dg'],
                 glayers=a['glayers'])


def train(a):
    import gpushare
    gpushare.below_normal()
    run = f'{RUNS}/{a.tag}'; os.makedirs(run, exist_ok=True); L = run + '/log.jsonl'
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    share = gpushare.Share(duty=a.duty, game_duty=a.game_duty, yield_gb=a.yield_gb, log=lambda *x: print(*x, flush=True))
    TR = M.Store2(f'{FEAT}/feat_train', a.limit or None); VA = M.Store2(f'{FEAT}/feat_val')
    net = make_net(vars(a)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    spe = math.ceil(TR.n / a.bs); total = spe * a.epochs
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    state = dict(epoch=0, step=0, best=-1.0, gpu_s=0.0, hist=[])
    if a.resume and os.path.exists(run + '/last.pt'):
        ck = torch.load(run + '/last.pt', map_location=device, weights_only=False)
        net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt']); state = ck['state']
        print('resumed', state['epoch'], state['step'], flush=True)
    else:
        log(L, event='start', n_train=TR.n, n_val=VA.n, params=sum(p.numel() for p in net.parameters()), steps=total, **vars(a))
    while state['epoch'] < a.epochs:
        ep = state['epoch']; rng = np.random.default_rng([a.seed, ep]); perm = rng.permutation(TR.n)
        picks = [rng.permutation(int(TR.nspec[g]))[:a.spec_per] for g in range(TR.n)]
        net.train(); t_ep = time.time(); losses = []
        i0 = (state['step'] - ep * spe) * a.bs
        for i in range(i0, TR.n, a.bs):
            g = perm[i:i + a.bs]
            t0 = time.time()
            D, own, cmax = TR.gather(g, [picks[x] for x in g], device)
            for pg in opt.param_groups:
                pg['lr'] = a.lr * sched(state['step'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = net(**D).float()
            sc = M.group_scores(lg, own, len(g), cmax)
            loss = -torch.log_softmax(sc, 1)[:, 0].mean()                 # the positive is the first candidate of every group
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            torch.cuda.synchronize(); dt = time.time() - t0; state['gpu_s'] += dt; share.tick(dt)
            state['step'] += 1; losses.append(float(loss.detach()))
            if state['step'] % 200 == 0:
                log(L, event='step', epoch=ep, step=state['step'], loss=float(np.mean(losses[-200:])), lr=a.lr * sched(state['step']),
                    gpu_s=round(state['gpu_s']), duty=share.duty)
            if share.must_yield():
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), state=state), run + '/last.pt')
                log(L, event='yield', step=state['step']); sys.exit(75)
        t0 = time.time(); sc = score_store(net, VA, device); state['gpu_s'] += time.time() - t0
        v = float(iso_mrr(VA, sc).mean())
        state['epoch'] += 1; state['hist'].append(dict(epoch=ep, loss=float(np.mean(losses)) if losses else None, val=v))
        if v > state['best']:
            state['best'] = v; torch.save(dict(net=net.state_dict(), args=vars(a), epoch=ep, val=v), run + '/best.pt')
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), state=state), run + '/last.pt')
        log(L, event='epoch', epoch=ep, loss=float(np.mean(losses)) if losses else None, val=v, best=state['best'], wall_s=round(time.time() - t_ep),
            gpu_s=round(state['gpu_s']))
    log(L, event='done', best=state['best'], gpu_s=round(state['gpu_s']))
    open(run + '/DONE', 'w').write(json.dumps(dict(best=state['best'], gpu_s=state['gpu_s'])))


def load_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = make_net(ck['args']).to(device); net.load_state_dict(ck['net']); net.eval()
    return net


def score(a):
    import gpushare
    gpushare.below_normal()
    device = torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    S = M.Store2(a.feat); t0 = time.time(); res = {}
    for i, c in enumerate(a.ckpt.split(',')):
        res[f's{i}'] = score_store(load_net(c, device), S, device, max_spec=a.max_spec)
    np.savez(a.out, smiles=S.smiles.astype(str), lab=S.lab, src=S.src, c0=S.c0, ncand=S.ncand, gid=S.gid, **res)
    print('SCORED', S.n, 'groups', f'{time.time() - t0:.1f}s', device, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='train', choices=['train', 'score'])
    ap.add_argument('--tag', default='s0'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--d', type=int, default=128); ap.add_argument('--layers', type=int, default=4); ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--ff', type=int, default=256); ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--no_graph', action='store_true'); ap.add_argument('--dg', type=int, default=96); ap.add_argument('--glayers', type=int, default=3)
    ap.add_argument('--spec_per', type=int, default=4)
    ap.add_argument('--duty', type=float, default=0.8); ap.add_argument('--game_duty', type=float, default=0.25); ap.add_argument('--yield_gb', type=float, default=9.0)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--ckpt', default=''); ap.add_argument('--feat', default=''); ap.add_argument('--out', default='')
    ap.add_argument('--max_spec', type=int, default=16); ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    train(a) if a.mode == 'train' else score(a)


if __name__ == '__main__':
    main()
