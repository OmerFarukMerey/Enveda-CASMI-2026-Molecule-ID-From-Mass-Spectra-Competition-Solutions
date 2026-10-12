"""Noise model of fpgen: per-bit calibration of the fold-safe fingerprint network on held-out molecules, the conditioning vector, and the corruption of a true fingerprint into a predicted-looking one.

Calibration (`python noise.py --fit`), on the stored logits of $FPGEN_WORK/cal (held-out structures the network never saw) against the
structures' true pool bits: per bit j the two class-conditional means of the raw logit (mu1 = true bit on, mu0 = off) with their pooled
standard deviation sd, and the bit's frequency. From them
  * conditioning vector  p_j = sigmoid(w_j (z_j - m_j)),  w_j = (mu1 - mu0) / sd^2,  m_j = (mu1 + mu0) / 2 - log(freq / (1 - freq)) / w_j
    (the posterior of the Gaussian model: a calibrated probability the generator can read at training and at test time alike);
  * corruption  z_j ~ N(mu_{b_j}, sd_j) for the true bit b_j, then the same p (`scale` widens the overlap: the standard deviation is
    multiplied by `scale`, 0 = no noise, 1 = the measured model).
This is the heart of the method: at training time the generator sees fingerprints that look like predictions, not like truth.
out: $FPGEN_WORK/cal/noise.npz, noise.json.   usage: python noise.py --fit
"""
import argparse, json, os, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpaths as P

EPS = 1e-6


def true_bits(pids):
    """Packed pool fingerprints -> unpacked 0/1 uint8 (n, nbits) in the pool's selected-bit order."""
    fp = np.load(P.WORK + '/pool_fp.npy', mmap_mode='r'); nb = len(np.load(P.WORK + '/fp_bits.npy'))
    return np.unpackbits(np.asarray(fp[np.asarray(pids)]), axis=1)[:, :nb]


def cal_data():
    """(logits, true bits) of the calibration molecules that have a pool row."""
    Z = np.load(P.CAL + '/z_cal.npy'); C = pd.read_parquet(P.CAL + '/cal_mols.parquet')
    PM = pd.read_parquet(P.WORK + '/pool_meta.parquet', columns=['train_sid'])
    s = pd.Series(np.arange(len(PM)), index=PM.train_sid.values); s = s[s.index >= 0]; s = s[~s.index.duplicated()]
    keep = C.sid.isin(s.index).values
    return Z[keep].astype(np.float32), true_bits(s.loc[C.sid.values[keep]].values)


def params(mu1, mu0, sd, freq):
    """Posterior parameters: p_j = sigmoid(w_j z_j + c_j)."""
    w = (mu1 - mu0) / np.maximum(sd ** 2, EPS)
    c = -(mu1 ** 2 - mu0 ** 2) / (2 * np.maximum(sd ** 2, EPS)) + np.log(np.clip(freq, 1e-5, 1 - 1e-5) / (1 - np.clip(freq, 1e-5, 1 - 1e-5)))
    return w.astype(np.float32), c.astype(np.float32)


def probs(z, w, c):
    """Raw logits of the fingerprint network -> the calibrated conditioning vector in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, np.float32) * w + c, -30, 30)))


def load():
    z = np.load(P.CAL + '/noise.npz')
    return {k: z[k] for k in z.files}


def corrupt(B, M, rng, scale=1.0):
    """True bits (n, nbits) uint8 -> conditioning vectors that look like predictions. scale 0 = the truth (p = 0 / 1), 1 = the measured model."""
    B = np.asarray(B, np.float32)
    if scale <= 0:
        return B
    mu = np.where(B > 0, M['mu1'], M['mu0']); sd = M['sd'] * float(scale)
    z = mu + sd * rng.standard_normal(B.shape, dtype=np.float32)
    return probs(z, M['w'], M['c'])


def fit(log=print):
    t0 = time.time(); Z, B = cal_data(); n, nb = Z.shape
    n1 = B.sum(0).astype(np.float64); freq = n1 / n
    s1 = (Z * B).sum(0); s0 = Z.sum(0) - s1
    q1 = ((Z ** 2) * B).sum(0); q0 = (Z ** 2).sum(0) - q1
    mu1 = s1 / np.maximum(n1, 1); mu0 = s0 / np.maximum(n - n1, 1)
    v = (q1 - n1 * mu1 ** 2 + q0 - (n - n1) * mu0 ** 2) / max(n - 2, 1)
    sd = np.sqrt(np.maximum(v, 1e-4)); w, c = params(mu1, mu0, sd, freq)
    M = dict(mu1=mu1.astype(np.float32), mu0=mu0.astype(np.float32), sd=sd.astype(np.float32), freq=freq.astype(np.float32), w=w, c=c)
    rng = np.random.default_rng(0)
    half = slice(0, n // 2); test = slice(n // 2, n)                      # the numbers below are reported on the half not used for the fit
    Pr = probs(Z[test], w, c); S = corrupt(B[test], M, rng); Bt = B[test].astype(np.float32)
    tan = lambda A, C: float(np.mean(((A > 0.5) & (C > 0.5)).sum(1) / np.maximum(((A > 0.5) | (C > 0.5)).sum(1), 1)))
    bins = np.clip((Pr * 10).astype(int), 0, 9)
    rel = [dict(bin=float(b / 10 + 0.05), predicted=float(Pr[bins == b].mean()), true=float(Bt[bins == b].mean()),
                simulated_rate=float((S[np.clip((S * 10).astype(int), 0, 9) == b] > -1).mean() * 0 + (Bt[np.clip((S * 10).astype(int), 0, 9) == b]).mean()),
                share=float((bins == b).mean())) for b in range(10) if (bins == b).sum() > 1000]
    gi = np.argsort(-freq)
    info = dict(molecules=int(n), bits=int(nb), on_bits_true=float(Bt.sum(1).mean()),
                p_sum_predicted=float(Pr.sum(1).mean()), p_sum_simulated=float(S.sum(1).mean()),
                binary_on_predicted=float((Pr > 0.5).sum(1).mean()), binary_on_simulated=float((S > 0.5).sum(1).mean()),
                tanimoto_pred_true=tan(Pr, Bt), tanimoto_sim_true=tan(S, Bt),
                brier_predicted=float(np.mean((Pr - Bt) ** 2)), brier_simulated_against_truth=float(np.mean((S - Bt) ** 2)),
                mean_abs_logodds_w=float(np.abs(w).mean()), sd_median=float(np.median(sd)),
                separation_median=float(np.median((mu1 - mu0) / sd)),
                reliability=rel,
                by_frequency=[dict(group=k, bits=int(len(ix)), freq=float(freq[ix].mean()), separation=float(((mu1 - mu0) / sd)[ix].mean()),
                                   p_pred=float(Pr[:, ix].mean()), p_sim=float(S[:, ix].mean()), true=float(Bt[:, ix].mean()))
                              for k, ix in enumerate(np.array_split(gi, 5))],
                wall_s=round(time.time() - t0))
    np.savez(P.CAL + '/noise.npz', **M); json.dump(info, open(P.CAL + '/noise.json', 'w'), indent=1)
    log(json.dumps({k: v for k, v in info.items() if k not in ('reliability', 'by_frequency')}, indent=1), flush=True)
    log('reliability ' + json.dumps(rel), flush=True); log('by_frequency ' + json.dumps(info['by_frequency']), flush=True)
    return info


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--fit', action='store_true'); a = ap.parse_args()
    P.mk(); P.low_priority()
    if a.fit:
        fit()
