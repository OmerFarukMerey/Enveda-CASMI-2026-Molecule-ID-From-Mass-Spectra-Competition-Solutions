"""Evaluation of an fpgen checkpoint on the pre-registered sets: sample under a conditioning arm, keep valid formula-matching structures, rank them, report the pre-registered metrics.

Arms: `--cond pred|true` (fingerprint predicted by the fold-safe network from the molecule's spectra, or the answer's true bits),
`--forms pipe|true` (the pipeline's first three base-list formulas with an equal sample budget each, or the true formula).
Hit test: a sample whose stereo-free canonical SMILES is in the answer's enumerated tautomer set; the metric key (RDKit 2026.03.3,
tautomer-canonical InChIKey first block) of the first 25 ranked rows is computed as a check.
out: $FPGEN_WORK/eval/<tag>__<cond>_<forms>_K<k>[_sub]/{samples.parquet, rows.parquet, summary.json}
usage: python evalgen.py --ckpt <run>/final.pt --tag ref0 --K 1000 [--cond pred] [--forms pipe] [--sub 100] [--workers 5]
"""
import argparse, json, os, sys, time

import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpaths as P, tok as T, noise as NZ, sets as S, model as MD, data as DT
import gpushare

RANK_CAP = 500            # unique valid structures per molecule that get a fingerprint for the ranking
_W = {}


def _init():
    P.low_priority(); P.watch_parent()
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    sys.path.insert(0, P.CODE)
    from casmi import chem
    _W['Chem'] = Chem; _W['chem'] = chem; _W['bits'] = np.load(P.WORK + '/fp_bits.npy')


def _valid(task):
    """(target counts, smiles list) -> list of (canonical SMILES, index) of the samples that parse, are one component and match the formula exactly."""
    if not _W:
        _init()
    Chem = _W['Chem']; cnt, smis = task; out = []
    for i, s in enumerate(smis):
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None or '.' in s:
            continue
        c = T.mol_counts(m); v, h, other = T.counts_vec(c)
        if other or h != cnt[T.NE] or not np.array_equal(v, np.asarray(cnt[:T.NE], np.int16)):
            continue
        out.append((Chem.MolToSmiles(m, isomericSmiles=False), i))
    return out


def _fps(smis):
    if not _W:
        _init()
    chem = _W['chem']; b = _W['bits']; out = []
    for s in smis:
        f = chem.raw_fingerprint(s)
        out.append(None if f is None else np.packbits(f[b]))
    return out


def _keys(smis):
    if not _W:
        _init()
    return [_W['chem'].score_key(s) for s in smis]


def conditioning(M, cond, log=print):
    """-> (n, nbits) float32 conditioning vectors for the evaluation molecules, and the raw predicted logits for the ranking."""
    NM = NZ.load()
    z = {}
    for f in ('z_dev.npz', 'z_fair.npz'):
        d = np.load(P.G2 + '/' + f); z.update(zip([str(x) for x in d['mid']], d['z']))
    Z = np.stack([z[m] for m in M.mid])
    if cond == 'pred':
        C = NZ.probs(Z, NM['w'], NM['c'])
    else:
        import multiprocessing as mp
        with mp.Pool(4, initializer=_init) as pool:
            fp = sum(pool.map(_fps, np.array_split(list(M.smiles), 32)), [])
        nb = len(NM['w']); C = np.stack([np.unpackbits(f)[:nb] if f is not None else np.zeros(nb, np.uint8) for f in fp]).astype(np.float32)
    log('conditioning %s: %d molecules, mean mass of the vector %.1f' % (cond, len(C), float(C.sum(1).mean())), flush=True)
    return C.astype(np.float32), Z


def targets(M, forms):
    """-> list per molecule of formula strings (the pipeline's first three, or the true one)."""
    out = []
    for r in M.itertuples():
        fs = [f for f in (r.forms3.split('|') if r.forms3 else []) if f] if forms == 'pipe' else [r.formula]
        out.append(fs or [r.formula])
    return out


def run_sampling(m, V, M, C, tg, K, bs=2048, temp=1.0, top_p=0.95, hard=True, device='cuda', share=None, log=print, seed=0):
    """-> DataFrame(mid, smiles, n, ll) of unique valid formula-matching structures per molecule."""
    E = torch.from_numpy(V.elem.astype('float32'))
    rows = []
    for i, fs in enumerate(tg):
        per = [K // len(fs)] * len(fs)
        per[0] += K - sum(per)
        for f, k in zip(fs, per):
            rows.append((i, f, k))
    cnts = DT.formula_counts([f for _, f, _ in rows])
    flat_mol = np.repeat([i for i, _, _ in rows], [k for _, _, k in rows])
    flat_cnt = np.repeat(cnts, [k for _, _, k in rows], axis=0)
    g = torch.Generator(device=device); g.manual_seed(seed)
    smis = []; lls = np.zeros(len(flat_mol), np.float32); lens = np.zeros(len(flat_mol), np.int32); t0 = time.time()
    for a in range(0, len(flat_mol), bs):
        b = slice(a, min(a + bs, len(flat_mol)))
        fp = torch.from_numpy(C[flat_mol[b]]).to(device); cnt = torch.from_numpy(flat_cnt[b]).to(device)
        t = time.time()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            ids, ll = m.sample(fp, cnt, E, max_len=170, temperature=temp, top_p=top_p, hard=hard, generator=g)
        if share is not None:
            torch.cuda.synchronize(); share.tick(time.time() - t)
            if share.must_yield():                               # others hold more than yield_gb: give the memory back, the queue restarts the job
                log('yielding the GPU after %d sequences (others hold %.1f GB)' % (a, share.other), flush=True)
                with open(P.GPU_LEDGER, 'a') as f:               # the time spent counts against the budget although nothing was produced
                    f.write(json.dumps(dict(job='evalgen', name='yielded', wall_s=round(time.time() - t0), t=time.strftime('%Y-%m-%d %H:%M'))) + chr(10))
                sys.exit(75)
        n = (ids != T.PAD).sum(1); lens[b] = n
        smis += [V.decode(list(r)) for r in ids]
        lls[b] = ll / np.maximum(n, 1)
        if a % (bs * 20) == 0:
            log('sampling %d/%d %.0fs' % (a, len(flat_mol), time.time() - t0), flush=True)
    import multiprocessing as mp
    tasks = []; idx = []
    for a in range(0, len(flat_mol), 4096):
        b = slice(a, min(a + 4096, len(flat_mol)))
        tasks.append((flat_cnt[a], smis[b]))                     # all rows of a chunk share a formula only inside one molecule: split per row block
        idx.append(a)
    # chunk strictly inside one (molecule, formula) block so that the target counts are constant
    tasks = []; idx = []
    starts = np.r_[0, np.cumsum([k for _, _, k in rows])]
    for j, (_, _, k) in enumerate(rows):
        s0, s1 = starts[j], starts[j + 1]
        for a in range(s0, s1, 4096):
            b = min(a + 4096, s1); tasks.append((cnts[j], smis[a:b])); idx.append(a)
    with mp.Pool(5, initializer=_init) as pool:
        res = pool.map(_valid, tasks, chunksize=4)
    rec = {}
    for off, r in zip(idx, res):
        for can, i in r:
            k = (int(flat_mol[off + i]), can); e = rec.get(k)
            if e is None:
                rec[k] = [1, float(lls[off + i])]
            else:
                e[0] += 1; e[1] = max(e[1], float(lls[off + i]))
    D = pd.DataFrame([(M.mid.values[k[0]], k[1], v[0], v[1]) for k, v in rec.items()], columns=['mid', 'smiles', 'n', 'll'])
    log('sampling done: %d sequences, %d valid formula-matching unique structures (%.1f per molecule), %.0f s, %.0f seq/s'
        % (len(flat_mol), len(D), len(D) / max(len(M), 1), time.time() - t0, len(flat_mol) / max(time.time() - t0, 1e-9)), flush=True)
    return D, dict(sequences=int(len(flat_mol)), unique_valid=int(len(D)), valid_rate=float(sum(len(r) for r in res) / max(len(flat_mol), 1)),
                   mean_len=float(lens.mean()), seq_per_s=float(len(flat_mol) / max(time.time() - t0, 1e-9)))


def rank_and_score(M, D, Z, budgets=(100, 1000, 10000), K=1000, workers=5, log=print):
    """Rank each molecule's structures by fingerprint agreement, likelihood, both and frequency; -> (rows, summary)."""
    import multiprocessing as mp
    taut = {r.mid: set(r.tsmi.split('|')) for r in M.itertuples()}
    zz = {m: Z[i] for i, m in enumerate(M.mid)}
    D = D.sort_values(['mid', 'n'], ascending=[True, False]).groupby('mid', sort=False).head(RANK_CAP).reset_index(drop=True)
    with mp.Pool(workers, initializer=_init) as pool:
        fps = sum(pool.map(_fps, np.array_split(list(D.smiles), max(1, len(D) // 2000 + 1))), [])
    nb = Z.shape[1]
    fz = np.zeros(len(D), np.float32)
    for i, f in enumerate(fps):
        if f is not None:
            fz[i] = float(np.unpackbits(f)[:nb].astype(np.float32) @ zz[D.mid.values[i]]) / 100.0
    D['fz'] = fz; D['hit'] = [s in taut[m] for m, s in zip(D.mid.values, D.smiles.values)]
    out = []
    for mid, g in D.groupby('mid', sort=False):
        g = g.reset_index(drop=True)
        zf = (g.fz - g.fz.mean()) / (g.fz.std() + 1e-6); zl = (g.ll - g.ll.mean()) / (g.ll.std() + 1e-6)
        orders = dict(fz=(-g.fz).values, ll=(-g.ll).values, both=(-(zf + zl)).values, freq=(-g.n.values + 0.0))
        rec = dict(mid=mid, n_unique=len(g), generated=bool(g.hit.any()))
        for name, key in orders.items():
            o = np.argsort(key, kind='stable'); r = int(np.flatnonzero(g.hit.values[o])[0] + 1) if g.hit.any() else 0
            rec['rank_' + name] = r
        out.append(rec)
    R = pd.DataFrame(out)
    R = M[['mid', 'eset', 'key']].merge(R, on='mid', how='left')
    R['n_unique'] = R.n_unique.fillna(0); R['generated'] = R.generated.fillna(False)
    for c in [c for c in R.columns if c.startswith('rank_')]:
        R[c] = R[c].fillna(0).astype(int)
    f = lambda g, c, k: float((g[c].between(1, k)).mean())
    summ = {}
    for es, g in R.groupby('eset'):
        summ[es] = dict(n=int(len(g)), generated=float(g.generated.mean()), unique_mean=float(g.n_unique.mean()),
                        **{'%s_top%d' % (c[5:], k): f(g, c, k) for c in R.columns if c.startswith('rank_') for k in (1, 5, 25)})
    log(json.dumps(summ, indent=1), flush=True)
    return R, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--K', type=int, default=1000)
    ap.add_argument('--cond', default='pred', choices=['pred', 'true']); ap.add_argument('--forms', default='pipe', choices=['pipe', 'true'])
    ap.add_argument('--temp', type=float, default=1.0); ap.add_argument('--top_p', type=float, default=0.95); ap.add_argument('--soft', action='store_true')
    ap.add_argument('--sub', type=int, default=0); ap.add_argument('--sets', default='N,F,SH'); ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--workers', type=int, default=5); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--note', default='')
    ap.add_argument('--resume', action='store_true', help='accepted for the queue runner; an evaluation restarts from the beginning')
    a = ap.parse_args()
    P.mk(); P.low_priority(); torch.set_num_threads(2)
    name = '%s__%s_%s_K%d%s%s' % (a.tag, a.cond, a.forms, a.K, '_sub%d' % a.sub if a.sub else '', '_soft' if a.soft else '')
    out = P.EVAL + '/' + name; os.makedirs(out, exist_ok=True)
    log_f = open(out + '/log.txt', 'a')

    def log(*x, **k):
        s = ' '.join(str(y) for y in x); print(s, flush=True); log_f.write(s + '\n'); log_f.flush()
    M, _ = S.load(); M = M[M.eset.isin(a.sets.split(','))].reset_index(drop=True)
    if a.sub:
        import hashlib
        M = M.iloc[np.argsort([hashlib.md5(('sub' + m).encode()).hexdigest() for m in M.mid])].groupby('eset').head(a.sub).reset_index(drop=True)
    log(S.leak_check(np.zeros(0, int), log=lambda *x, **k: None, extra=' (evaluation %s)' % name))
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False); V = T.Vocab(ck['vocab'])
    ar = ck['args']; m = MD.FPGen(len(V), nbits=ck['nbits'], d=ar['d'], layers=ar['layers'], heads=ar['heads'], drop=0.0, n_fp=ar['n_fp']).cuda().eval()
    m.load_state_dict(ck['model'])
    log('checkpoint %s: step %d, %d parameters, arm %s, noise %.2f, seed %d; evaluating %d molecules (%s), K %d, cond %s, formulas %s, temp %.2f top_p %.2f %s'
        % (a.ckpt, ck['step'], MD.n_params(m), ar['arm'], ar['noise'], ar['seed'], len(M), M.eset.value_counts().to_dict(), a.K, a.cond, a.forms, a.temp, a.top_p, 'soft' if a.soft else 'hard'))
    C, Z = conditioning(M, a.cond, log=log)
    share = gpushare.Share(0.8, 0.25, 9.0, log=log)
    t0 = time.time()
    D, st = run_sampling(m, V, M, C, targets(M, a.forms), a.K, bs=a.bs, temp=a.temp, top_p=a.top_p, hard=not a.soft, share=share, log=log, seed=a.seed)
    D.to_parquet(out + '/samples.parquet', index=False)
    R, summ = rank_and_score(M, D, Z, K=a.K, workers=a.workers, log=log)
    R.to_parquet(out + '/rows.parquet', index=False)
    info = dict(name=name, ckpt=a.ckpt, step=int(ck['step']), args=vars(a), train_args=ar, sampling=st, summary=summ,
                wall_s=round(time.time() - t0), t=time.strftime('%Y-%m-%d %H:%M'), note=a.note)
    json.dump(info, open(out + '/summary.json', 'w'), indent=1)
    with open(P.GPU_LEDGER, 'a') as f:
        f.write(json.dumps(dict(job='evalgen', name=name, wall_s=round(time.time() - t0), t=time.strftime('%Y-%m-%d %H:%M'))) + '\n')
    log('done', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
