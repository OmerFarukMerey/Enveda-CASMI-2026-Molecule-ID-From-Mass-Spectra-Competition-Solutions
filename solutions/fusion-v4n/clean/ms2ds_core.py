"""MS2DeepScore 2.0 triangulation term: pure-torch encoder (weights_only load), sparse binning, reference search, per-candidate score, row rule.

Model: MS2DeepScore 2.0 dual-mode (Zenodo 17826815, CC BY 4.0; code Apache-2.0, matchms/ms2deepscore). Encoder = Linear(9902, 10000) + ReLU
-> Linear(10000, 500) + tanh; input = [ionmode bit (positive 1), precursor_mz / 1000, 9900 bins of 0.1 Da over m/z [10, 1000), max of
intensity ** 0.5 per bin] (ms2deepscore 2.7.2 tensorize_spectra / SpectralEncoder; intensities max-normalised as in train.parquet). The
first layer is applied as a sparse product (a spectrum has <= 200 peaks), which is exact up to float summation order. The checkpoint is
read with torch.load(weights_only=True): no arbitrary unpickling. The ms2deepscore package itself is not needed at inference; the
package's own embeddings of a fixed probe set are stored as a golden file and reproduced by `golden_check`.

Triangulation (docs/landscape/allowed-models-2026-10-24.md section 2b): the cosine of two embeddings predicts the RDKit daylight
(RDKFingerprint, 4096 bits) Tanimoto of the two structures. For a query molecule, pred(r) = predicted Tanimoto to every reference
identity r (mean query embedding per polarity, aggregated over r's spectra), neighbours = top K identities; a candidate c scores by the
agreement between pred and the exact Tanimoto(c, r) over those neighbours (`tri_score`). `tri_row` re-ranks one final list by
reciprocal-rank fusion with the term's ranks, only when lib_max < gate.
"""
import json
import math

import numpy as np

MIN_MZ, MAX_MZ, BIN_W, NBINS, INT_POW = 10.0, 1000.0, 0.1, 9900, 0.5
FP_BITS = 4096


# ---------------------------------------------------------------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------------------------------------------------------------
def load_encoder(path, device='cpu', dtype=None):
    """Checkpoint -> dict(W1m (10000, 2), W1sT (9900, 10000), b1, W2 (500, 10000), b2, settings) on device. weights_only=True."""
    import torch
    ck = torch.load(str(path), map_location='cpu', weights_only=True)
    s = json.loads(ck['settings_json'])
    assert s['base_dims'] == [10000] and s['embedding_dim'] == 500 and not s['train_binning_layer'], s
    assert s['activation_function'] == 'relu' and float(s['min_mz']) == MIN_MZ and float(s['max_mz']) == MAX_MZ
    assert float(s['mz_bin_width']) == BIN_W and float(s['intensity_scaling']) == INT_POW
    md = s['additional_metadata']
    assert md[0][0] == 'CategoricalToBinary' and md[0][1]['metadata_field'] == 'ionmode' and md[0][1]['entries_becoming_one'] == 'positive'
    assert md[1][0] == 'StandardScaler' and md[1][1]['metadata_field'] == 'precursor_mz' and float(md[1][1]['mean']) == 0.0
    assert float(md[1][1]['standard_deviation']) == 1000.0 and len(md) == 2
    sd = ck['state_dict']
    W1, b1 = sd['encoder.dense_layers.0.0.weight'], sd['encoder.dense_layers.0.0.bias']
    W2, b2 = sd['encoder.embedding_layer.0.weight'], sd['encoder.embedding_layer.0.bias']
    assert tuple(W1.shape) == (10000, NBINS + 2) and tuple(W2.shape) == (500, 10000) and len(sd) == 4
    dt = dtype or torch.float32
    return dict(W1m=W1[:, :2].contiguous().to(device, dt), W1sT=W1[:, 2:].t().contiguous().to(device, dt), b1=b1.to(device, dt),
                W2=W2.to(device, dt), b2=b2.to(device, dt), device=device, dtype=dt, settings=s)


def bin_csr(mzs, ints):
    """Lists of peak arrays -> (indptr, bins, vals) CSR of the binned spectra (max of intensity**0.5 per bin), float32 values."""
    lens = np.fromiter((len(m) for m in mzs), np.int64, len(mzs))
    if lens.sum() == 0:
        return np.zeros(len(mzs) + 1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    mz = np.concatenate([np.asarray(m, np.float64) for m in mzs])
    it = np.concatenate([np.asarray(i, np.float64) for i in ints])
    row = np.repeat(np.arange(len(mzs), dtype=np.int64), lens)
    ok = (mz >= MIN_MZ) & (mz < MAX_MZ) & np.isfinite(it)
    mz, it, row = mz[ok], it[ok], row[ok]
    b = ((mz - MIN_MZ) / BIN_W).astype(np.int64)                          # = int() of a non-negative float (numba vectorize_spectrum)
    b = np.minimum(b, NBINS - 1)
    v = np.power(np.maximum(it, 0.0), INT_POW)
    key = row * NBINS + b
    o = np.lexsort((-v, key))                                             # per (row, bin): largest value first
    key, v = key[o], v[o]
    first = np.ones(len(key), bool)
    first[1:] = key[1:] != key[:-1]
    key, v = key[first], v[first]
    r, bb = key // NBINS, key % NBINS
    indptr = np.zeros(len(mzs) + 1, np.int64)
    np.add.at(indptr, r + 1, 1)
    return np.cumsum(indptr), bb, v.astype(np.float32)


def embed(enc, mzs, ints, prec, positive, batch=4096):
    """Embeddings (n, 500) float32 numpy for spectra given as peak lists, precursor m/z and a positive-mode flag per spectrum."""
    import torch
    n = len(mzs)
    out = np.zeros((n, 500), np.float32)
    prec = np.asarray(prec, np.float64)
    pos = np.asarray(positive, bool)
    with torch.no_grad():
        for a in range(0, n, batch):
            b = min(n, a + batch)
            ip, bins, vals = bin_csr(mzs[a:b], ints[a:b])
            sp = torch.sparse_csr_tensor(torch.from_numpy(ip), torch.from_numpy(bins), torch.from_numpy(vals).to(enc['dtype']),
                                         size=(b - a, NBINS)).to(enc['device'])
            meta = torch.from_numpy(np.stack([pos[a:b].astype(np.float32), (prec[a:b] / 1000.0).astype(np.float32)], 1)).to(enc['device'], enc['dtype'])
            h = torch.sparse.mm(sp, enc['W1sT']) if len(vals) else torch.zeros((b - a, 10000), device=enc['device'], dtype=enc['dtype'])
            h = torch.relu(h + meta @ enc['W1m'].t() + enc['b1'])
            e = torch.tanh(h @ enc['W2'].t() + enc['b2'])
            out[a:b] = e.float().cpu().numpy()
    return out


def golden_check(enc, golden, tol=2e-4):
    """Reproduce the ms2deepscore package's embeddings of the probe spectra stored in `golden` (dict from make_golden). Returns max |diff|."""
    g = golden
    e = embed(enc, [np.asarray(x) for x in g['mzs']], [np.asarray(x) for x in g['ints']], g['prec'], g['positive'])
    d = float(np.abs(e - np.asarray(g['emb'], np.float32)).max())
    assert d < tol, ('MS2DeepScore golden check failed', d)
    return d


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------------------------------------------------------------
# reference search
# ---------------------------------------------------------------------------------------------------------------------------------
def predicted_by_key(qemb, qpos, R, rpos, rkey, nkeys, agg='max', exclude=None, chunk=400000):
    """Predicted Tanimoto of the query molecule to every reference identity.

    qemb (q, 500) query embeddings, qpos (q,) positive flags; R (n, 500) unit reference embeddings (float16 / float32), rpos (n,) flags,
    rkey (n,) int identity codes in [0, nkeys); agg 'max' or 'mean' over an identity's spectra (per polarity group, then the groups
    averaged with the query-spectrum counts as weights); exclude: identity codes set to NaN. Returns float32 (nkeys,) with NaN = no data."""
    qe = unit(qemb)
    qpos = np.asarray(qpos, bool)
    num = np.zeros(nkeys, np.float64); den = np.zeros(nkeys, np.float64)
    for flag in (True, False):
        sel = qpos == flag
        if not sel.any():
            continue
        qbar = qe[sel].mean(0).astype(np.float32)
        acc = np.full(nkeys, -np.inf if agg == 'max' else 0.0, np.float64)
        cnt = np.zeros(nkeys, np.float64)
        for a in range(0, len(R), chunk):
            m = rpos[a:a + chunk] == flag
            if not m.any():
                continue
            s = (R[a:a + chunk][m].astype(np.float32) @ qbar).astype(np.float64)
            k = rkey[a:a + chunk][m]
            if agg == 'max':
                np.maximum.at(acc, k, s)
            else:
                acc += np.bincount(k, weights=s, minlength=nkeys)
            cnt += np.bincount(k, minlength=nkeys)
        have = cnt > 0
        val = acc[have] if agg == 'max' else acc[have] / cnt[have]
        w = float(sel.sum())
        num[have] += w * val; den[have] += w
    out = np.full(nkeys, np.nan, np.float32)
    ok = den > 0
    out[ok] = (num[ok] / den[ok]).astype(np.float32)
    if exclude is not None:
        ex = np.asarray(list(exclude), np.int64)
        if len(ex):
            out[ex] = np.nan
    return out


def predicted_batch(qbar, qw, R, rpos, rkey, nkeys, agg='max', device='cpu', chunk=262144):
    """Batched predicted_by_key for M molecules at once (torch; CPU or GPU).

    qbar: dict {True: (M, 500), False: (M, 500)} mean unit query embedding per polarity (rows of molecules without that polarity are
    ignored through qw); qw: dict {True: (M,), False: (M,)} query-spectrum counts per polarity (0 = none). R: (n, 500) unit reference
    embeddings (numpy float16/32 or torch), rpos (n,) bool, rkey (n,) int64 codes. Returns torch float32 (M, nkeys) on device, NaN = no data."""
    import torch
    M = len(next(iter(qbar.values())))
    num = torch.zeros((M, nkeys), dtype=torch.float32, device=device)
    den = torch.zeros((M, nkeys), dtype=torch.float32, device=device)
    rpos_t = torch.as_tensor(np.asarray(rpos, bool))
    rkey_t = torch.as_tensor(np.asarray(rkey, np.int64))
    for flag in (True, False):
        w = torch.as_tensor(np.asarray(qw[flag], np.float32), device=device)
        if float(w.sum()) <= 0:
            continue
        Q = torch.as_tensor(np.asarray(qbar[flag], np.float32), device=device)
        acc = torch.full((M, nkeys), -float('inf') if agg == 'max' else 0.0, dtype=torch.float32, device=device)
        cnt = torch.zeros(nkeys, dtype=torch.float32, device=device)
        for a in range(0, len(rkey_t), chunk):
            m = rpos_t[a:a + chunk] == flag
            if not bool(m.any()):
                continue
            Rc = torch.as_tensor(R[a:a + chunk]) if not isinstance(R, torch.Tensor) else R[a:a + chunk]
            Rc = Rc[m].to(device=device, dtype=torch.float32)
            k = rkey_t[a:a + chunk][m].to(device)
            S = Q @ Rc.t()
            if agg == 'max':
                acc.scatter_reduce_(1, k.unsqueeze(0).expand(M, -1), S, reduce='amax', include_self=True)
            else:
                acc.index_add_(1, k, S)
            cnt.index_add_(0, k, torch.ones_like(k, dtype=torch.float32))
        have = cnt > 0
        val = acc if agg == 'max' else acc / cnt.clamp_min(1.0)
        wm = w.unsqueeze(1) * have.unsqueeze(0).float()
        num += torch.where(wm > 0, wm * val, torch.zeros_like(val))
        den += wm
        del acc, val
    out = torch.where(den > 0, num / den.clamp_min(1e-12), torch.full_like(num, float('nan')))
    return out


def neighbours(pred, K, anchors=0, seed=0):
    """Indices of the top-K identities by predicted Tanimoto, plus `anchors` random other identities with data (seeded)."""
    ok = np.where(np.isfinite(pred))[0]
    if len(ok) == 0:
        return np.zeros(0, np.int64)
    k = min(K, len(ok))
    top = ok[np.argpartition(-pred[ok], k - 1)[:k]]
    top = top[np.argsort(-pred[top], kind='stable')]
    if anchors:
        rest = np.setdiff1d(ok, top)
        if len(rest):
            rng = np.random.default_rng(seed)
            top = np.concatenate([top, rng.choice(rest, min(anchors, len(rest)), replace=False)])
    return top


# ---------------------------------------------------------------------------------------------------------------------------------
# fingerprints and the per-candidate score
# ---------------------------------------------------------------------------------------------------------------------------------
def fp_of(smiles):
    """RDKit daylight fingerprint as MS2DeepScore was trained on it (Chem.RDKFingerprint(mol, fpSize=4096)); None if unparsable."""
    from rdkit import Chem
    try:
        m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        return Chem.RDKFingerprint(m, fpSize=FP_BITS) if m is not None else None
    except Exception:
        return None


def tanimoto_many(fp, fps):
    from rdkit import DataStructs
    return np.asarray(DataStructs.BulkTanimotoSimilarity(fp, list(fps)), np.float64)


def tri_score(pred, T, form):
    """Agreement of predicted (pred) and exact (T) Tanimotos over the neighbours; w = max(pred, 0)**2. Higher = better."""
    p = np.asarray(pred, np.float64); t = np.asarray(T, np.float64)
    w = np.maximum(p, 0.0) ** 2
    if w.sum() <= 0 or len(p) < 2:
        return None
    w = w / w.sum()
    if form == 'mse':
        return float(-(w * (p - t) ** 2).sum())
    if form == 'wdot':
        return float((w * t).sum())
    if form == 'wcorr':
        mp, mt = (w * p).sum(), (w * t).sum()
        cv = (w * (p - mp) * (t - mt)).sum()
        vp, vt = (w * (p - mp) ** 2).sum(), (w * (t - mt) ** 2).sum()
        if vp <= 1e-12 or vt <= 1e-12:
            return 0.0
        return float(cv / math.sqrt(vp * vt))
    raise ValueError(form)


def term_ranks(scores):
    """Scores (None allowed) -> 1-based ranks (best = 1, ties share the better rank order by position); None scores get the median rank."""
    idx = [i for i, s in enumerate(scores) if s is not None and s == s]
    r = [None] * len(scores)
    order = sorted(idx, key=lambda i: (-scores[i], i))
    for k, i in enumerate(order):
        r[i] = k + 1
    med = (len(idx) + 1) / 2.0 if idx else 1.0
    return [x if x is not None else med for x in r]


def tri_row(F, scores, lib_max, sw, key, placeholder='CCO'):
    """One final list F and the term's score per row -> (new list, tag). Pure.

    sw: dict(on, gate, w, k). Applies only when lib_max < gate and at least 2 rows have a score; fused = 1/(k + base rank) +
    w / (k + term rank), stable (ties keep base order). Rows are never added or dropped. tag in off, gated, skipped, few, same, moved."""
    F = list(F)
    if not sw.get('on'):
        return F, 'off'
    if not F or not isinstance(F[0], str) or F[0].strip() in ('', placeholder):
        return F, 'skipped'
    try:
        lm = float(lib_max)
    except (TypeError, ValueError):
        lm = float('nan')
    if not (lm == lm) or lm >= float(sw['gate']):
        return F, 'gated'
    sc = list(scores)
    assert len(sc) == len(F)
    if sum(s is not None and s == s for s in sc) < 2:
        return F, 'few'
    tr = term_ranks(sc)
    k, w = float(sw['k']), float(sw['w'])
    fused = [1.0 / (k + i + 1) + w / (k + tr[i]) for i in range(len(F))]
    o = sorted(range(len(F)), key=lambda i: (-fused[i], i))
    new = [F[i] for i in o]
    return new, ('same' if new == F else 'moved')


def tri_selftest():
    on = dict(on=True, gate=0.7, w=0.25, k=60.0)
    key = lambda s: s
    F = [chr(65 + i) for i in range(25)]
    assert tri_row(F, [None] * 25, 0.1, dict(on, on=False), key) == (F, 'off')
    assert tri_row(F, list(range(25)), 0.75, on, key) == (F, 'gated')
    assert tri_row(F, list(range(25)), float('nan'), on, key) == (F, 'gated')
    assert tri_row(['CCO'], [1.0], 0.1, on, key) == (['CCO'], 'skipped')
    assert tri_row(F, [None] * 24 + [1.0], 0.1, on, key) == (F, 'few')
    up = [0.0] * 25; up[1] = 1.0; up[0] = -1.0                           # row 2 best by the term, row 1 worst
    new, tag = tri_row(F, up, 0.1, on, key)
    assert tag == 'moved' and new[0] == 'B' and new.index('A') == 3 and sorted(new) == sorted(F), new
    up = [0.0] * 25; up[6] = 1.0                                         # row 7 cannot pass row 1 at w 0.25, k 60 (bounded move)
    new, _ = tri_row(F, up, 0.1, on, key)
    assert new[0] == 'A' and new.index('G') < 6
    new, tag = tri_row(F, [float(-i) for i in range(25)], 0.1, on, key)  # term agrees with the base order
    assert (new, tag) == (F, 'same')
    assert term_ranks([0.5, None, 0.9, 0.5]) == [2, 2.0, 1, 3]
    p = np.array([0.9, 0.5, 0.2]); assert tri_score(p, p, 'mse') == 0.0 and abs(tri_score(p, p, 'wcorr') - 1.0) < 1e-12
    assert tri_score(p, np.array([0.2, 0.5, 0.9]), 'wcorr') < 0 and tri_score(np.array([-0.1, -0.2]), p[:2], 'mse') is None
    ip, b, v = bin_csr([np.array([10.0, 10.05, 999.99, 1000.0, 9.99, 50.0]), np.array([])], [np.array([0.25, 1.0, 0.04, 1.0, 1.0, 0.0])] * 1 + [np.array([])])
    assert list(ip) == [0, 3, 3] and list(b) == [0, 400, 9899] and np.allclose(v, [1.0, 0.0, 0.2])
    return 'ok'
