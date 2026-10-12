"""Fold-safe fingerprint-network logits of held-out molecules for fpgen's noise calibration, aggregated as the v1 engine does (GPU, shared).

Molecules: structures of the held-out folds ('hold', 'np') that are NOT benchmark molecules and not in set F; one library per structure
(seeded choice), at most 16 spectra; logits = 0.5 x (mean over spectra + mean over the per-polarity merged spectra), as scratch/gen2/g1_logits.py.
`--parity` recomputes stored DEV logits (scratch/gen2/cache/z_dev.npz) with this code path and prints the agreement.
out: $FPGEN_WORK/cal/z_cal.npy (float16), cal_mols.parquet.   usage: python fplogits.py [--n 20000] [--parity 60]
"""
import argparse, json, os, sys, time

import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpaths as P
import fpnet as FN
from specdata import merge_spectra
import gpushare


class Spectra:
    def __init__(self):
        self.meta = pd.read_parquet(P.WORK + '/spec_meta.parquet', columns=['adduct', 'adduct_ix', 'precursor_mz', 'ce_mean', 'ce_n', 'mode', 'sid', 'lib'])
        self.off = np.load(P.WORK + '/spec_off.npy'); self.mz = np.load(P.WORK + '/spec_mz.npy', mmap_mode='r'); self.it = np.load(P.WORK + '/spec_it.npy', mmap_mode='r')
        self.cols = {c: self.meta[c].values for c in self.meta.columns}

    def items(self, rows):
        """Rows of the spectrum cache (one molecule) -> (single-spectrum items, merged items per polarity)."""
        c = self.cols; sp = []
        for r in rows[:16]:
            a, b = self.off[r], self.off[r + 1]
            sp.append(dict(mz=np.asarray(self.mz[a:b], np.float64), it=np.asarray(self.it[a:b], np.float64), mode=int(c['mode'][r]), adduct=c['adduct'][r], ax=int(c['adduct_ix'][r]),
                           prec=float(c['precursor_mz'][r]), ce=float(c['ce_mean'][r]), ce_n=int(c['ce_n'][r]) if c['ce_n'][r] > 0 else 1))
        s_items = [FN.make_item(s['mz'], s['it'], s['prec'], s['ax'], s['ce'], s['ce_n'], s['mode']) for s in sp]
        m_items = []
        for md in (1, -1):
            g = [s for s in sp if s['mode'] == md]
            if not g:
                continue
            mm, ii = merge_spectra([(s['mz'], s['it']) for s in g]); prec = float(np.median([s['prec'] for s in g]))
            ces = [s['ce'] for s in g if not np.isnan(s['ce'])]; ads = [s['adduct'] for s in g]; ad = max(sorted(set(ads)), key=ads.count)
            ax = next(s['ax'] for s in g if s['adduct'] == ad)
            m_items.append(FN.make_item(mm, ii, prec, ax, float(np.mean(ces)) if ces else float('nan'), sum(max(1, s['ce_n']) for s in g), md))
        return s_items, m_items


@torch.no_grad()
def logits(model, S, groups, device, share=None, bs=256, log=print):
    """groups: list of row arrays (one per molecule) -> (n, nbits) float32 aggregated logits."""
    out = np.zeros((len(groups), model.nbits), np.float32); buf = []; t0 = time.time()

    def flush():
        if not buf:
            return
        items = [x for _, s, m in buf for x in s + m]; z = []
        o = np.argsort([len(x['mz']) for x in items])
        for i in range(0, len(items), bs):
            t = time.time(); idx = o[i:i + bs]
            zz = model(*FN.collate([items[j] for j in idx], device))[0].float().cpu().numpy(); z.append((idx, zz))
            if share is not None:
                if device != 'cpu':
                    torch.cuda.synchronize()
                share.tick(time.time() - t)
        Z = np.zeros((len(items), model.nbits), np.float32)
        for idx, zz in z:
            Z[idx] = zz
        k = 0
        for gi, s, m in buf:
            zs = Z[k:k + len(s)].mean(0); k += len(s); zm = Z[k:k + len(m)].mean(0); k += len(m); out[gi] = 0.5 * (zs + zm)
        buf.clear()
    for gi, rows in enumerate(groups):
        s, m = S.items(rows); buf.append((gi, s, m))
        if len(buf) >= 128:
            flush()
            if gi % 2048 < 128:
                log(f'logits {gi}/{len(groups)} {time.time() - t0:.0f}s', flush=True)
    flush(); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=20000); ap.add_argument('--parity', type=int, default=0); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda'); a = ap.parse_args()
    P.mk(); P.low_priority(); torch.set_num_threads(3); t0 = time.time()
    S = Spectra(); model = FN.load_ckpt(P.FOLD0, a.device); share = gpushare.Share(0.8, 0.25, 9.0) if a.device != 'cpu' else None
    if a.parity:
        z = np.load(P.G2 + '/z_dev.npz'); B = pd.read_parquet(P.BENCH + '/spectra.parquet'); rng = np.random.default_rng(0)
        pick = rng.choice(len(z['mid']), a.parity, replace=False); g = B.groupby('mid', sort=False).row.apply(lambda s: s.values)
        Z = logits(model, S, [g[m] for m in z['mid'][pick]], a.device, share)
        cor = [float(np.corrcoef(Z[i], z['z'][j])[0, 1]) for i, j in enumerate(pick)]; mad = float(np.abs(Z - z['z'][pick]).max())
        print(json.dumps(dict(parity_n=a.parity, corr_min=min(cor), corr_median=float(np.median(cor)), max_abs_diff=mad, sign_agree=float(((Z > 0) == (z['z'][pick] > 0)).mean()))), flush=True)
        return
    out = P.CAL + '/z_cal.npy'
    if os.path.exists(out):
        sys.exit('exists: ' + out)
    sp = pd.read_parquet(P.WORK + '/split.parquet'); B = pd.read_parquet(P.BENCH + '/mols.parquet', columns=['sid', 'key']); F = pd.read_parquet(P.G2 + '/eval_fair.parquet')
    assert set(B.sid) <= set(sp.sid)
    cand = sp[~sp.sid.isin(set(B.sid)) & ~sp.key.isin(set(B.key) | set(F.key))].sort_values('sid')
    rng = np.random.default_rng(a.seed); cand = cand.iloc[np.sort(rng.choice(len(cand), min(a.n, len(cand)), replace=False))].reset_index(drop=True)
    m = S.meta[S.meta.sid.isin(set(cand.sid))]; rows_of = {}
    for sid, g in m.groupby('sid', sort=False):
        libs = np.unique(g.lib.values); lib = libs[rng.integers(len(libs))]; rows_of[sid] = g.index.values[g.lib.values == lib]
    cand = cand[cand.sid.isin(rows_of)].reset_index(drop=True); cand['n_spec'] = [min(16, len(rows_of[s])) for s in cand.sid]; cand['row0'] = [int(rows_of[s][0]) for s in cand.sid]
    print(f'calibration molecules: {len(cand)} held-out structures outside the benchmark and F (benchmark sids shared: {int(cand.sid.isin(set(B.sid)).sum())})', flush=True)
    assert not cand.sid.isin(set(B.sid)).any() and not cand.key.isin(set(F.key)).any()
    Z = logits(model, S, [rows_of[s] for s in cand.sid], a.device, share)
    np.save(out + '.tmp.npy', Z.astype(np.float16)); os.replace(out + '.tmp.npy', out); cand.to_parquet(P.CAL + '/cal_mols.parquet', index=False)
    dt = time.time() - t0
    with open(P.GPU_LEDGER, 'a') as f:
        f.write(json.dumps(dict(job='fplogits', n=len(cand), wall_s=round(dt), t=time.strftime('%Y-%m-%d %H:%M'))) + '\n')
    print('done', len(cand), round(dt), 's', flush=True)


if __name__ == '__main__':
    main()
