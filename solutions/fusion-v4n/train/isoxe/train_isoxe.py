"""isoxe trainer: listwise softmax over same-formula groups, timsTOF groups weighted, checkpoint per epoch, best by V isomer MRR.

GPU shared with the user's games through train/gpushare.py (duty cycle; exits with code 75 after a checkpoint when others need
the memory; train/guard_run.py restarts it with --resume). Also `--mode score`: per-candidate scores of a feature directory
(all stored spectra, mean over spectra) with a checkpoint.

    python train/isoxe/train_isoxe.py --mode train --tag s0 --seed 0 [--epochs 12] [--no_bt]
    python train/isoxe/train_isoxe.py --mode score --ckpt <run>/best.pt --feat <feat dir> --out scores.npz
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import model as M  # noqa: E402

OUT = os.environ.get('ISOXE_W', 'C:/Users/hemre/casmi-work/isoxe').replace('\\', '/')


def log(path, **k):
    k['t'] = time.strftime('%H:%M:%S'); print(json.dumps(k), flush=True)
    open(path, 'a').write(json.dumps(k) + '\n')


@torch.no_grad()
def score_store(net, S, device, bs=64, max_spec=16, blind=False):
    """Mean-over-spectra score per candidate (all stored spectra, at most max_spec). -> np.float32 [n candidates].
    blind: diagnostic, every peak shown as unexplained (annotation removed; spectrum-side and candidate-global inputs kept)."""
    net.eval(); out = np.zeros(int(S.ncand.sum()), np.float32)
    for i in range(0, S.n, bs):
        g = np.arange(i, min(S.n, i + bs))
        D, own, cmax = S.gather(g, [np.arange(min(int(S.nspec[x]), max_spec)) for x in g], device)
        if blind:
            D['ai'] = torch.zeros_like(D['ai']); D['ai'][..., 1] = 7; D['ai'][..., 2] = 3; D['af'] = torch.zeros_like(D['af'])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            lg = net(**D).float()
        sc = M.group_scores(lg, own, len(g), cmax).cpu().numpy()
        for b, x in enumerate(g):
            out[S.c0[x]:S.c0[x] + S.ncand[x]] = sc[b, :S.ncand[x]]
    return out


def iso_mrr(S, sc, mask=None):
    rr = np.array([M.isomer_rr(sc[S.c0[g]:S.c0[g] + S.ncand[g]], S.lab[S.c0[g]:S.c0[g] + S.ncand[g]]) for g in range(S.n)])
    return rr if mask is None else rr[mask]


def train(a):
    import gpushare
    gpushare.below_normal()
    run = f'{OUT}/runs/{a.tag}'; os.makedirs(run, exist_ok=True); L = run + '/log.jsonl'
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    share = gpushare.Share(duty=a.duty, game_duty=a.game_duty, yield_gb=a.yield_gb, log=lambda *x: print(*x, flush=True))
    TR = M.Store(f'{OUT}/feat_train', a.limit or None); VA = M.Store(f'{OUT}/feat_val')
    net = M.XE(d=a.d, layers=a.layers, heads=a.heads, ff=a.ff, dropout=a.dropout, use_bt=not a.no_bt).to(device)
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
    w_e = torch.tensor(np.where(TR.e180, a.w_tims, 1.0), dtype=torch.float32)
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
            lp = torch.log_softmax(sc, 1)[:, 0]                  # the positive is the first candidate of every group
            w = w_e[g].to(device)
            loss = -(w * lp).sum() / w.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            torch.cuda.synchronize(); dt = time.time() - t0; state['gpu_s'] += dt; share.tick(dt)
            state['step'] += 1; losses.append(float(loss))
            if state['step'] % 200 == 0:
                log(L, event='step', epoch=ep, step=state['step'], loss=float(np.mean(losses[-200:])), lr=a.lr * sched(state['step']), gpu_s=round(state['gpu_s']),
                    duty=share.duty)
            if share.must_yield():
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), state=state), run + '/last.pt')
                log(L, event='yield', step=state['step']); sys.exit(75)
        t0 = time.time(); sc = score_store(net, VA, device); state['gpu_s'] += time.time() - t0
        rr = iso_mrr(VA, sc); v = float(rr.mean()); ve = float(rr[VA.e180].mean()); vo = float(rr[~VA.e180].mean())
        state['epoch'] += 1; state['hist'].append(dict(epoch=ep, loss=float(np.mean(losses)) if losses else None, val=v, val_e180=ve, val_other=vo))
        if v > state['best']:
            state['best'] = v; torch.save(dict(net=net.state_dict(), args=vars(a), epoch=ep, val=v), run + '/best.pt')
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), state=state), run + '/last.pt')
        log(L, event='epoch', epoch=ep, loss=float(np.mean(losses)) if losses else None, val=v, val_e180=ve, val_other=vo, best=state['best'],
            wall_s=round(time.time() - t_ep), gpu_s=round(state['gpu_s']))
    log(L, event='done', best=state['best'], gpu_s=round(state['gpu_s']))


def load_net(path, device):
    ck = torch.load(path, map_location=device, weights_only=False); a = ck['args']
    net = M.XE(d=a['d'], layers=a['layers'], heads=a['heads'], ff=a['ff'], dropout=a['dropout'], use_bt=not a['no_bt']).to(device)
    net.load_state_dict(ck['net']); net.eval()
    return net


def score(a):
    import gpushare
    gpushare.below_normal()
    device = torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    S = M.Store(a.feat); t0 = time.time()
    res = {}
    for i, c in enumerate(a.ckpt.split(',')):
        res[f's{i}'] = score_store(load_net(c, device), S, device, max_spec=a.max_spec)
        if a.blind:
            res[f'blind{i}'] = score_store(load_net(c, device), S, device, max_spec=a.max_spec, blind=True)
    np.savez(a.out, smiles=S.smiles.astype(str), lab=S.lab, c0=S.c0, ncand=S.ncand, gid=S.gid, e180=S.e180, **res)
    print('SCORED', S.n, 'groups', f'{time.time() - t0:.1f}s', device, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='train', choices=['train', 'score'])
    ap.add_argument('--tag', default='s0'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=12); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--d', type=int, default=128); ap.add_argument('--layers', type=int, default=3); ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--ff', type=int, default=256); ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--no_bt', action='store_true')
    ap.add_argument('--spec_per', type=int, default=3); ap.add_argument('--w_tims', type=float, default=2.0)
    ap.add_argument('--duty', type=float, default=0.8); ap.add_argument('--game_duty', type=float, default=0.25); ap.add_argument('--yield_gb', type=float, default=9.0)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--ckpt', default=''); ap.add_argument('--feat', default=''); ap.add_argument('--out', default='')
    ap.add_argument('--max_spec', type=int, default=16); ap.add_argument('--cpu', action='store_true'); ap.add_argument('--blind', action='store_true')
    a = ap.parse_args()
    train(a) if a.mode == 'train' else score(a)


if __name__ == '__main__':
    main()
