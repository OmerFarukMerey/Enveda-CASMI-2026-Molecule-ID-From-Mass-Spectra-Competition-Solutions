"""Evaluation of the ownneural model on s1's REAL final lists (the s1val and s1b dumps) against our previous scorers (scratch/ownneural/PREREG.md).

Steps: (1) score every row of every final list with the ownneural checkpoints (clean/onnet_infer.py) and with joint1 (train/joint model,
same spectrum adapter) -> <out>/scores_<run>.pkl; (2) per list: the answer's base position, its reciprocal rank among the movable rows
(ranks 2..n) under each scorer (ties averaged; NaN rows get the list median), its reciprocal rank over the full list, and the top-1-locked
RRF re-rank (k 1, w 0.25; scratch/top1lock rule); (3) primary metric on P (answer at base rank 2-25), paired molecule-bootstrap of
ON - best baseline, the unseen subset U3, the verdict; -> <out>/eval.json, eval.md, rows.parquet.

usage: python eval_lists.py --on <ckpt>[,<ckpt>...] --out <dir> [--j1 <ckpt>,<ckpt>] [--device cuda] [--score_only]
"""
import argparse, json, os, pickle, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import onpaths as P  # noqa: E402

sys.path.insert(0, P.CLEAN)
J1_DEFAULT = 'C:/Users/hemre/Desktop/Enveda-CASMI-2026-Molecule-ID-From-Mass-Spectra-Competition-Solutions/scratch/joint1/runs/j1_long_s0/last.pt,' \
             'C:/Users/hemre/Desktop/Enveda-CASMI-2026-Molecule-ID-From-Mass-Spectra-Competition-Solutions/scratch/joint1/runs/j1_long_s1/last.pt'
OLD = {'s1val': P.CW + '/top1lock', 's1b': P.CW + '/rerank25/sc_b'}
FEAT = {'s1val': P.CW + '/rerank25/feat_s1val.parquet', 's1b': P.CW + '/rerank25/feat_s1b.parquet'}
BASELINES = ('XE', 'FPZ', 'J1')
SHOWN = ('GL', 'IC', 'MF')


def load_run(run):
    d = P.DUMPS[run]; sub = pd.read_csv(d + '/submission.csv'); lists = {r.molecule_id: r.smiles.split(';') for r in sub.itertuples()}
    f = pd.read_parquet(FEAT[run], columns=['molecule_id', 'pos', 'y', 'ik14', 'lib'])
    ans = f[f.y == 1].set_index('molecule_id'); ik = f.groupby('molecule_id').ik14.first(); lib = f.groupby('molecule_id').lib.first()
    vt = pd.read_parquet(d + '/val_test.parquet'); spec = {m: g for m, g in vt.groupby('molecule_id', sort=False)}
    old = {}
    for k in ('XE', 'FPZ') + SHOWN:
        x = pickle.load(open(f'{OLD[run]}/{k}.pkl', 'rb'))['scores']; old[k] = {m: np.asarray(v, np.float64) for m, v in x.items()}
    seen = json.load(open(f'{OLD[run]}/seen.json'))['U']
    return dict(lists=lists, ans_pos=ans.pos.to_dict(), ik=ik.to_dict(), lib=lib.to_dict(), spec=spec, old=old, U={k: set(v) for k, v in seen.items()})


def score_on(run, R, ckpts, dev):
    import onnet_infer as OI
    S = OI.OnScorer(ckpts, P.WORK + '/fp_bits.npy', device=dev)
    return S.score_lists(R['spec'], R['lists'])


def score_j1(run, R, ckpts, dev):
    """joint1 checkpoints (train/joint/model.py Joint, fp variant): mean cosine over the checkpoints, same raw spectrum adapter."""
    import torch
    import onnet_infer as OI
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'joint')); sys.path.insert(0, os.path.dirname(HERE))
    import model as JM
    cb = OI.CandBits(np.load(P.WORK + '/fp_bits.npy')); mids = [m for m in R['lists'] if m in R['spec']]
    arr = OI.arrays_from_rows([R['spec'][m] for m in mids]); own = np.concatenate([np.full(len(R['lists'][m]), i) for i, m in enumerate(mids)])
    fp, ok = cb([x for m in mids for x in R['lists'][m]]); sc = np.zeros(len(own))
    meta = json.load(open(P.JIDX + '/meta.json'))
    for c in ckpts:
        ck = torch.load(c, map_location=dev, weights_only=False); a = ck['args']
        m = JM.Joint('fp', nbits=len(cb.bits), n_adduct=max(64, meta['adducts']), n_instr=max(8, meta['instr']), d=a.get('d', 256), layers=a.get('layers', 6),
                     hidden=tuple(int(x) for x in str(a.get('hidden', '1024,512')).split(',')), pool='mean' if a.get('agg', 'mean') == 'merged' else a.get('agg', 'mean')).to(dev)
        m.load_state_dict(ck['model']); m.eval()
        with torch.no_grad():
            b = {k: torch.as_tensor(v, device=dev) for k, v in arr.items()}
            zs = m.spec.embed(m.spec.pool_spectra(m.spec.spectra(b), b['owner'], len(mids)))
            ze = torch.cat([m.struct(torch.as_tensor(fp[s:s + 8192], device=dev)) for s in range(0, len(fp), 8192)])
            sc += (zs[torch.as_tensor(own, device=dev)] * ze).sum(1).double().cpu().numpy()
    sc /= len(ckpts); sc[~ok] = np.nan; out = {}; p = 0
    for m in mids:
        n = len(R['lists'][m]); out[m] = sc[p:p + n].astype(np.float32); p += n
    return out


def j1_unseen(R):
    """Molecule ids whose answer structure is not among joint1's training positives (joint index tr_sid)."""
    st = pd.read_parquet(P.WORK + '/train_structs.parquet', columns=['inchikey14', 'key']); tr = np.load(P.JIDX + '/tr_sid.npy')
    seen_ids = set(st.inchikey14.values[tr]) | set(st.key.values[tr])
    return {m for m, k in R['ik'].items() if k not in seen_ids}


def ranks_desc(v):
    """Rank (1 = best) by descending value, ties averaged; NaN -> median of the finite values."""
    v = np.asarray(v, np.float64).copy(); f = np.isfinite(v)
    v[~f] = np.median(v[f]) if f.any() else 0.0
    return pd.Series(-v).rank(method='average').values


def rr_among(v, j):
    """Reciprocal rank of element j in v (descending; ties averaged; NaN -> median)."""
    v = np.asarray(v, np.float64).copy(); f = np.isfinite(v); v[~f] = np.median(v[f]) if f.any() else 0.0
    return 1.0 / (1.0 + (np.delete(v, j) > v[j]).sum() + 0.5 * (np.delete(v, j) == v[j]).sum())


def locked_pos(v, apos, k=1.0, w=0.25):
    """Answer's new position (1-based) after the top-1-locked RRF re-rank of rows 2..n by scores v (full list order); apos = base position."""
    n = len(v); b = np.arange(2, n + 1); r = ranks_desc(np.asarray(v)[1:]); fused = 1.0 / (k + b) + w / (k + r)
    order = np.lexsort((b, -fused)); newpos = np.empty(n - 1, int); newpos[order] = np.arange(2, n + 1)
    return 1 if apos == 1 else int(newpos[apos - 2])


def boot(diff, clusters, n=10000, seed=0):
    d = pd.DataFrame(dict(d=diff, c=clusters)).groupby('c').d.agg(['sum', 'size']); s, c = d['sum'].values, d['size'].values
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(s), (n, len(s))); m = s[idx].sum(1) / c[idx].sum(1)
    return float(np.mean(diff)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def table(runs_data, scorers):
    rows = []
    for run, (R, S) in runs_data.items():
        for m, lst in R['lists'].items():
            apos = int(R['ans_pos'].get(m, 0)); n = len(lst)
            row = dict(run=run, molecule_id=m, ik14=R['ik'].get(m), lib=R['lib'].get(m), n=n, apos=apos, base_rr=1.0 / apos if apos else 0.0,
                       base_mov=(1.0 / (apos - 1)) if apos >= 2 else np.nan)
            for k in scorers:
                v = S[k].get(m)
                if v is None or len(v) != n:
                    row[f'{k}_mov'] = row[f'{k}_full'] = row[f'{k}_lock'] = np.nan; continue
                row[f'{k}_full'] = rr_among(v, apos - 1) if apos else 0.0
                row[f'{k}_mov'] = rr_among(np.asarray(v)[1:], apos - 2) if apos >= 2 else np.nan
                row[f'{k}_lock'] = (1.0 / locked_pos(v, apos)) if apos else 0.0
            row['U3'] = all(m in R['Useen'][k] for k in BASELINES)
            for k in BASELINES:
                row[f'U_{k}'] = m in R['Useen'][k]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--on', required=True); ap.add_argument('--out', required=True); ap.add_argument('--j1', default=J1_DEFAULT)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    import torch
    torch.set_num_threads(a.threads)
    runs_data = {}
    for run in P.DUMPS:
        R = load_run(run); f = f'{a.out}/scores_{run}.pkl'
        if os.path.exists(f):
            S = pickle.load(open(f, 'rb'))
        else:
            S = dict(ON=score_on(run, R, a.on.split(','), a.device), J1=score_j1(run, R, a.j1.split(','), a.device)); pickle.dump(S, open(f, 'wb'))
        S.update(R['old']); R['Useen'] = dict(XE=R['U']['XE'], FPZ=R['U']['FP'], J1=j1_unseen(R))
        runs_data[run] = (R, S); print(run, 'scored', f'{time.time() - t0:.0f}s', flush=True)
    sc = ('ON',) + BASELINES + SHOWN; T = table(runs_data, sc); T.to_parquet(a.out + '/rows.parquet', index=False)
    Pm = T[T.apos >= 2]; res = dict(n_lists=len(T), n_P=len(Pm), n_P_by_run=Pm.run.value_counts().to_dict(), n_P_U3=int(Pm.U3.sum()))
    res['primary'] = {k: float(Pm[f'{k}_mov'].mean()) for k in sc}; res['primary']['BASE'] = float(Pm.base_mov.mean())
    res['primary_U3'] = {k: float(Pm[Pm.U3][f'{k}_mov'].mean()) for k in sc}; res['primary_U3']['BASE'] = float(Pm[Pm.U3].base_mov.mean())
    res['primary_by_run'] = {r: {k: float(g[f'{k}_mov'].mean()) for k in sc} for r, g in Pm.groupby('run')}
    res['primary_by_lib'] = {l: {k: float(g[f'{k}_mov'].mean()) for k in sc} | dict(n=len(g)) for l, g in Pm.groupby('lib')}
    best = max(BASELINES, key=lambda k: res['primary'][k]); res['best_baseline'] = best
    res['delta_vs_best'] = boot((Pm.ON_mov - Pm[f'{best}_mov']).values, Pm.ik14.values)
    res['delta_vs_each'] = {k: boot((Pm.ON_mov - Pm[f'{k}_mov']).values, Pm.ik14.values) for k in BASELINES + SHOWN}
    PU = Pm[Pm.U3]; res['delta_vs_best_U3'] = boot((PU.ON_mov - PU[f'{best}_mov']).values, PU.ik14.values) if len(PU) else None
    res['delta_vs_each_unseen'] = {k: dict(n=int(Pm[f'U_{k}'].sum()), d=boot((Pm[Pm[f'U_{k}']].ON_mov - Pm[Pm[f'U_{k}']][f'{k}_mov']).values, Pm[Pm[f'U_{k}']].ik14.values))
                                   for k in BASELINES if Pm[f'U_{k}'].sum() > 1}
    res['full_list_mrr'] = {k: float(T[f'{k}_full'].mean()) for k in sc}
    res['locked'] = {r: {k: boot((g[f'{k}_lock'] - g.base_rr).values, g.ik14.values) for k in sc} for r, g in T.groupby('run')}
    res['locked_wl'] = {r: {k: [int(((g[f'{k}_lock'] - g.base_rr) > 1e-12).sum()), int(((g[f'{k}_lock'] - g.base_rr) < -1e-12).sum())] for k in sc} for r, g in T.groupby('run')}
    d, lo, hi = res['delta_vs_best']; du = res['delta_vs_best_U3']
    if res['primary']['ON'] <= res['primary'][best]:
        verdict = 'KILL'
    elif lo > 0 and len(PU) >= 30 and du[0] >= 0:
        verdict = 'PASS'
    else:
        verdict = 'WEAK'
    res['verdict'] = verdict
    bu = max(BASELINES, key=lambda k: res['primary_U3'][k]); res['best_baseline_U3'] = bu   # PREREG addendum 2 (fair rule)
    fu = boot((PU.ON_mov - PU[f'{bu}_mov']).values, PU.ik14.values) if len(PU) else (0.0, 0.0, 0.0); res['delta_vs_best_on_U3'] = fu
    if res['primary_U3']['ON'] <= res['primary_U3'][bu]:
        fair = 'FAIR-KILL'
    elif fu[1] > 0 and all(v['d'][0] >= 0 for v in res['delta_vs_each_unseen'].values()):
        fair = 'FAIR-PASS'
    else:
        fair = 'FAIR-WEAK'
    res['fair_verdict'] = fair; res['seconds'] = round(time.time() - t0)
    json.dump(res, open(a.out + '/eval.json', 'w'), indent=1, default=float)
    L = [f'# ownneural evaluation on s1 final lists ({time.strftime("%Y-%m-%d %H:%M")})', '', f'Verdict (section 4): **{verdict}** (best baseline on P {best}); fair verdict (addendum 2): **{fair}** (best baseline on P n U3 {bu}: '
         f'ON - {bu} {fu[0]:+.4f} [{fu[1]:+.4f}, {fu[2]:+.4f}])', '',
         f'P = {len(Pm)} lists (answer at base rank 2-25; {res["n_P_by_run"]}), of which U3 (unseen by XE, FPZ, J1) = {len(PU)}.', '',
         '| scorer | primary MRR (movable rows), P | P n U3 | s1val P | s1b P | full-list MRR, all 800 | locked RRF Delta s1val | locked RRF Delta s1b |', '|---|---|---|---|---|---|---|---|']
    for k in sc:
        lk = [res['locked'][r][k] for r in ('s1val', 's1b')]
        L.append(f'| {k} | {res["primary"][k]:.3f} | {res["primary_U3"][k]:.3f} | {res["primary_by_run"]["s1val"][k]:.3f} | {res["primary_by_run"]["s1b"][k]:.3f} | '
                 f'{res["full_list_mrr"][k]:.3f} | {lk[0][0]:+.4f} [{lk[0][1]:+.4f}, {lk[0][2]:+.4f}] | {lk[1][0]:+.4f} [{lk[1][1]:+.4f}, {lk[1][2]:+.4f}] |')
    L.append(f'| base order (leaked) | {res["primary"]["BASE"]:.3f} | {res["primary_U3"]["BASE"]:.3f} | | | | | |')
    L += ['', f'ON - {best} on P: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]; on P n U3: ' + (f'{du[0]:+.4f} [{du[1]:+.4f}, {du[2]:+.4f}]' if du else 'n/a'), '',
          'ON - each scorer on P: ' + '; '.join(f'{k} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]' for k, v in res['delta_vs_each'].items()), '',
          'ON - baseline on that baseline\'s unseen P lists: ' + '; '.join(f'{k} (n {v["n"]}) {v["d"][0]:+.4f} [{v["d"][1]:+.4f}, {v["d"][2]:+.4f}]' for k, v in res['delta_vs_each_unseen'].items()),
          '', 'By library (P): ' + '; '.join(f'{l} (n {v["n"]}): ' + ', '.join(f'{k} {v[k]:.3f}' for k in sc) for l, v in res['primary_by_lib'].items())]
    open(a.out + '/eval.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
