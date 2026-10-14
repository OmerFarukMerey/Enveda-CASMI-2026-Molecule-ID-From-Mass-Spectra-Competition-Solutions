"""The multi-spectrum adapter (clean/msadapt.py) on synthetic spectra, without competition data.

Pinned: raw rows -> the same peaks and metadata as the training cache (specdata.build_cache run on a synthetic train.parquet), the
engine-dict input == the row input, items == specenc.build_item, polarity grouping and chunking, batch invariance of the molecule
logits and their aggregation, the f.z / fp_cos scores, unpack == np.packbits order, a reference-layout (non-multi) checkpoint.
usage: python tests/test_msadapt.py   (or pytest)
"""
import os, sys, tempfile

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'clean'))
import numpy as np
import torch
import msadapt, specenc

ADDS = ['[M+H]+', '[M+Na]+', '[M-H2O+H]+', '[M-H]-', '[M+Cl]-', '[M+CH2O2-H]-', '[M+FOO]+']


def _rows(n, rng, prec=400.0):
    """Synthetic raw spectra as test.parquet-like rows (both polarities, CE lists with NaN / empty / several values, peaks above the
    precursor, tiny peaks, non-positive m/z)."""
    import pandas as pd
    rows = []
    for k in range(n):
        ad = ADDS[k % len(ADDS)]; neg = ad.endswith('-'); p = prec + rng.normal(0, 0.002)
        m = int(rng.integers(0, 300))
        mz = np.sort(rng.uniform(-1.0, p + 10.0, m)); it = rng.uniform(0, 1, m) ** 4 * 1000.0
        ce = [None, [20.0], [10.0, 20.0, 40.0], [np.nan], [np.nan, 30.0], []][k % 6]
        rows.append(dict(ingest_lib='massbank', inchikey14='AAAAAAAAAAAAAA', ionization_mode='negative' if neg else 'positive', instrument_type='Q-TOF',
                         adduct=ad, precursor_mz=p, precursor_error_ppm=0.0, ms2_mzs=mz, ms2_normalized_intensities=it, num_peaks=m,
                         base_peak_intensity=1.0, collision_energy_ev=None if ce is None else np.asarray(ce, np.float64),
                         collision_energy_orig_units='eV', molecule_id='m0'))
    return pd.DataFrame(rows)


def _tiny(td, **cfg):
    torch.manual_seed(0); net = specenc.SpecEnc(96, dict(d=64, layers=2, heads=4, **cfg)).eval()
    p = os.path.join(td, 'n.pt'); specenc.save_ckpt(p, net, 1); return p


def _same_items(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert set(x) == set(y), (set(x) ^ set(y))
        for k in x:
            assert np.array_equal(np.asarray(x[k]), np.asarray(y[k])), k


def test_rows_match_training_cache():
    """Raw rows through the adapter == the spectra and metadata that specdata.build_cache writes for training."""
    import pandas as pd, specdata
    rng = np.random.default_rng(0); df = _rows(14, rng)
    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, 'train.parquet'); df.drop(columns=['molecule_id']).to_parquet(tp)
        sp = os.path.join(td, 'structs.parquet'); pd.DataFrame(dict(inchikey14=['AAAAAAAAAAAAAA'], sid=[0])).to_parquet(sp)
        specdata.build_cache(tp, sp, td, log=lambda *a: None)
        off = np.load(os.path.join(td, 'spec_off.npy')); mz = np.load(os.path.join(td, 'spec_mz.npy')); it = np.load(os.path.join(td, 'spec_it.npy'))
        meta = pd.read_parquet(os.path.join(td, 'spec_meta.parquet'))
    A = msadapt.MolSpectra(msadapt.from_rows(df))
    for i in range(len(df)):
        a, b = A.peaks(i), (mz[off[i]:off[i + 1]], it[off[i]:off[i + 1]])
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[0].dtype == b[0].dtype == np.float32
    assert np.array_equal(A.prec, meta.precursor_mz.values) and np.array_equal(A.adduct, meta.adduct_ix.values)
    assert np.array_equal(A.mode, meta['mode'].values) and np.array_equal(A.ce_n, meta.ce_n.values)
    assert np.array_equal(np.isnan(A.ce), np.isnan(meta.ce_mean.values)) and np.allclose(np.nan_to_num(A.ce), np.nan_to_num(meta.ce_mean.values))
    assert A.adduct[6] == specdata.ADDUCT_OTHER


def test_engine_dicts_equal_rows():
    """The v1 engine's dict (nb/06_base_lists.py: ce = mean, ce_n = count) gives the same items as the raw rows whenever every CE is finite."""
    rng = np.random.default_rng(1); df = _rows(12, rng); df = df[np.array([c is not None and len(c) > 0 and bool(np.isfinite(c).all()) for c in df.collision_energy_ev])]
    eng = [dict(mz=r.ms2_mzs, it=r.ms2_normalized_intensities, mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct, prec=float(r.precursor_mz),
                ce=float(np.mean(r.collision_energy_ev)), ce_n=len(r.collision_energy_ev)) for r in df.itertuples()]
    with tempfile.TemporaryDirectory() as td:
        F = msadapt.SpecFP(_tiny(td, pair=True, multi=True, multi_peaks=160))
    a, ma = F.items(msadapt.MolSpectra(eng)); b, mb = F.items(msadapt.MolSpectra(msadapt.from_rows(df)))
    _same_items(a, b); assert ma == mb


def test_items_are_build_item():
    rng = np.random.default_rng(2); M = msadapt.MolSpectra(msadapt.from_rows(_rows(11, rng)))
    with tempfile.TemporaryDirectory() as td:
        F = msadapt.SpecFP(_tiny(td, pair=True, multi=True, multi_peaks=160))
    items, meta = F.items(M); want = [specenc.build_item(M, idx, F.cfg) for _, idx in msadapt.groups(M.mode)]
    _same_items(items, want); assert sum(n for _, n in meta) == len(M)
    for x in items:                                       # the per-spectrum budget of training: max(8, 160 // n) peaks per spectrum
        n = int(x['slot'].max()) + 1 if len(x['slot']) else 1
        assert np.bincount(x['slot'], minlength=n).max() <= max(8, 160 // n)


def test_groups():
    g = msadapt.groups([1] * 10 + [-1] * 3)
    assert [(m, len(i)) for m, i in g] == [(1, 5), (1, 5), (-1, 3)] and g[0][1] == [0, 1, 2, 3, 4]
    assert [len(i) for _, i in msadapt.groups([-1] * 17)] == [6, 6, 5] and msadapt.groups([]) == []
    assert [len(i) for _, i in msadapt.groups([1] * 8)] == [8]


def test_logits_batch_invariance_and_aggregation():
    rng = np.random.default_rng(3); mols = [msadapt.from_rows(_rows(int(n), rng, prec=300.0 + 50 * k)) for k, n in enumerate([1, 3, 9, 14])]
    with tempfile.TemporaryDirectory() as td:
        F = msadapt.SpecFP(_tiny(td, pair=True, multi=True, multi_peaks=160))
    Z, G, own = F.logits(mols, bs=3, return_groups=True)
    for k, m in enumerate(mols):
        assert np.allclose(F.logits([m], bs=64)[0], Z[k], atol=1e-4)
        pol = {}
        for r, (kk, md, n) in enumerate(own):
            if kk == k:
                pol.setdefault(md, []).append((n, G[r]))
        want = np.mean([sum(n * z for n, z in v) / sum(n for n, _ in v) for v in pol.values()], axis=0)
        assert np.allclose(Z[k], want, atol=1e-5)
    for k, m in enumerate(mols):                          # one item per (polarity, chunk of at most 8)
        assert len([o for o in own if o[0] == k]) == len(msadapt.groups(msadapt.MolSpectra(m).mode))
    assert [(md, n) for kk, md, n in own if kk == 2] == [(1, 6), (-1, 3)]          # 9 spectra: 6 positive, 3 negative
    assert F.logits([[]]).shape == (1, 96) and not F.logits([[]]).any()


def test_scores_and_unpack():
    rng = np.random.default_rng(4); bits = (rng.random((5, 21)) < 0.3).astype(np.float32); z = rng.normal(0, 3, 21)
    s = msadapt.scores(z, bits); assert np.allclose(s['fz'], bits @ z, atol=1e-4)
    pz = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
    assert np.allclose(s['cos'], (bits @ pz) / (np.sqrt(bits.sum(1)) * np.linalg.norm(pz) + 1e-9), atol=1e-5)
    assert np.array_equal(msadapt.unpack(np.packbits(bits.astype(np.uint8), axis=1), 21), bits)


def test_reference_layout():
    """A single-query network (base_s0-like): one merged item per chunk, and it loads from the reference layout."""
    import fpnet
    rng = np.random.default_rng(5); M = msadapt.MolSpectra(msadapt.from_rows(_rows(4, rng)))
    with tempfile.TemporaryDirectory() as td:
        ref = fpnet.FPNet(64, d=64, layers=2, heads=4); p = os.path.join(td, 'r.pt'); fpnet.save_ckpt(p, ref, 1); F = msadapt.SpecFP(p)
    items, meta = F.items(M); assert 'slot' not in items[0] and len(items) == 2 and F.logits([M]).shape == (1, 64)


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
