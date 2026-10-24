"""isoxe inference: score one molecule's candidate SMILES against its MS/MS spectra with the isomer cross-encoder (seed mean).

Ours (2026-10-24, for the s1x frag-gap switch; model and training: reviews/isoxe-2026-10-18.md, train/isoxe/). Pure functions,
offline, CPU or GPU; needs only NumPy, RDKit, torch (and optionally numba, through isoxe_feat). The network class is a copy of
train/isoxe/model.py's XE (same parameter names, so its state dicts load unchanged; a test checks equality of outputs) so that this
file and isoxe_feat.py are the whole delivery (no `model` module name that could collide inside a notebook).

Spectra are prepared exactly as the training features were: the library cache's cleaning (chemcore.clean_peaks: m/z <= precursor
+ 2, intensity >= 0.001 x max, top 512, float32), precursor as float32, collision energy = mean of the finite `collision_energy_ev`
values (float32), instrument class by clean/specdata.py's substring rules, polarity -1 if `ionization_mode` starts with 'neg' or
the adduct ends with '-'; then isoxe_feat.prep_spectrum (32 most intense peaks). Spectrum-side and annotation floats pass through
float16 as in the stored training chunks. A candidate's score = mean logit over the molecule's spectra (first `max_spec` rows,
empty ones dropped), then the mean over the seeds. Higher = better. Inference runs in float32 (no autocast).

    nets = load_models(['isoxe_s0.npz', 'isoxe_s1.npz'], device)
    spectra = prep_spectra(rows)                    # rows: dicts mz, it, prec, adduct, mode / ionization_mode, ce, instrument
    sc = score_candidates(nets, spectra, smiles, device, cache)    # np.float64 [n], NaN = candidate did not parse
    d = score_dict(nets, rows, smiles, device, cache)              # {smiles: score} for finite scores (FRAG_GAP format)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import isoxe_feat as X  # noqa: E402

VERSION = 1
N_ADD, N_INSTR = len(X.ADDUCTS) + 1, 8
MAX_SPEC = 16


# ------------------------------------------------------------------------------------------------ spectra -----------
def instrument_class(name) -> int:
    """= clean/specdata.instrument_class: 0 timsTOF, 1 Orbitrap / FT, 2 other TOF, 3 ion trap, 4 triple quadrupole, 5 other."""
    s = str(name).lower()
    if 'timstof' in s or 'tims' in s:
        return 0
    if any(k in s for k in ('orbitrap', 'exactive', 'ft', 'lumos', 'fusion', 'velos', 'elite', 'exploris', 'astral')):
        return 1
    if 'tof' in s:
        return 2
    if any(k in s for k in ('ion trap', 'iontrap', '-it', 'it-', 'trap', 'lcq', 'ltq')):
        return 3
    if any(k in s for k in ('qqq', 'triple', 'tq', 'qq')):
        return 4
    return 5


def clean_peaks(mz, it, prec_mz, floor=0.001, topk=512, over=2.0):
    """= clean/chemcore.clean_peaks (the library cache's cleaning) -> (m/z float32 ascending, intensity float32, base peak 1)."""
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    if len(mz) == 0 or len(it) != len(mz):
        return np.empty(0, np.float32), np.empty(0, np.float32)
    inr = mz <= prec_mz + over
    if not inr.any() or it[inr].max() <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = it[inr].max()
    idx = np.flatnonzero(inr & (it >= floor * mx) & (mz > 0))
    if len(idx) > topk:
        idx = idx[np.argsort(it[idx], kind='stable')[len(idx) - topk:]]
    o = np.argsort(mz[idx], kind='stable')
    return mz[idx][o].astype(np.float32), (it[idx] / mx).astype(np.float32)[o]


def _ce_mean(ce):
    if ce is None:
        return np.nan
    try:
        a = np.asarray(ce, np.float64).ravel()
    except (TypeError, ValueError):
        return np.nan
    a = a[np.isfinite(a)]
    return float(np.float32(a.mean())) if len(a) else np.nan


def _mode(row):
    m = row.get('mode')
    if m is not None and not isinstance(m, str):
        return 1 if float(m) > 0 else -1
    ion = str(row.get('ionization_mode', m) or '').lower()
    return -1 if ion.startswith('neg') or str(row.get('adduct', '')).endswith('-') else 1


def prep_row(row):
    """One spectrum (dict: mz, it, prec, adduct, mode or ionization_mode, ce = list of eV or None, instrument = str or int class)
    -> isoxe_feat prepared spectrum, or None if nothing is left after cleaning."""
    mz, it = clean_peaks(row['mz'], row['it'], float(row['prec']))      # the cache cleaned with the float64 precursor
    prec = float(np.float32(row['prec']))                              # ... and stored it as float32 (joint index spec_prec)
    if len(mz) == 0:
        return None
    ins = row.get('instrument', 0)
    ins = int(ins) if isinstance(ins, (int, np.integer)) else instrument_class(ins)
    ce = _ce_mean(row.get('ce'))
    return X.prep_spectrum(np.asarray(mz, np.float64), np.asarray(it, np.float64), prec, row['adduct'], _mode(row),
                           [ce] if np.isfinite(ce) else None, ins)


def prep_spectra(rows, max_spec=MAX_SPEC):
    """The first `max_spec` rows of a molecule (input order), prepared; empty spectra dropped."""
    out = []
    for r in list(rows)[:max_spec]:
        try:
            sp = prep_row(r)
        except Exception:
            sp = None
        if sp is not None:
            out.append(sp)
    return out


def rows_from_frame(g):
    """Rows of one molecule from a test-format DataFrame (ms2_mzs, ms2_normalized_intensities, precursor_mz, adduct,
    ionization_mode, instrument_type, collision_energy_ev)."""
    cols = set(g.columns)
    out = []
    for r in g.itertuples(index=False):
        out.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64), prec=float(r.precursor_mz),
                        adduct=r.adduct, ionization_mode=r.ionization_mode,
                        instrument=getattr(r, 'instrument_type') if 'instrument_type' in cols else 5,
                        ce=getattr(r, 'collision_energy_ev') if 'collision_energy_ev' in cols else None))
    return out


# ------------------------------------------------------------------------------------------------ network -----------
def _xe_class():
    import torch
    import torch.nn as nn

    class XE(nn.Module):
        """= train/isoxe/model.py XE (parameter names and forward identical)."""

        def __init__(self, d=128, layers=3, heads=4, ff=256, dropout=0.1, use_bt=True):
            super().__init__()
            self.use_bt = use_bt
            self.pk = nn.Linear(X.N_SPF + X.N_FLT, d)
            self.ek = nn.Embedding(5, d); self.edh = nn.Embedding(8, d); self.ecar = nn.Embedding(4, d)
            self.ebt = nn.Embedding(X.NBT, d, padding_idx=0)
            self.cls_in = nn.Linear(X.N_CAND + 2, d); self.cls0 = nn.Parameter(torch.zeros(d))
            self.eadd = nn.Embedding(N_ADD, d); self.emode = nn.Embedding(2, d); self.einstr = nn.Embedding(N_INSTR, d)
            enc = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation='gelu')
            self.tr = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        def forward(self, spf, mask, ai, af, meta, ce, cg):
            ai = ai.long(); meta = meta.long()
            x = self.pk(torch.cat([spf, af], -1)) + self.ek(ai[..., 0]) + self.edh(ai[..., 1]) + self.ecar(ai[..., 2])
            if self.use_bt:
                x = x + self.ebt(ai[..., 3:6]).sum(-2)
            m = self.eadd(meta[:, 0].clamp(0, N_ADD - 1)) + self.emode(meta[:, 1].clamp(0, 1)) + self.einstr(meta[:, 2].clamp(0, N_INSTR - 1))
            c = self.cls0 + self.cls_in(torch.cat([cg, ce[:, None], meta[:, 3:4].float()], -1)) + m
            x = x + m[:, None, :]
            seq = torch.cat([c[:, None, :], x], 1)
            pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
            h = self.tr(seq, src_key_padding_mask=pad)
            return self.head(h[:, 0]).squeeze(-1)
    return XE


CFG_KEYS = ('d', 'layers', 'heads', 'ff', 'dropout', 'no_bt')


def export_npz(ckpt_path, out_path):
    """Training checkpoint (train_isoxe.py best.pt) -> plain npz (float32 arrays + '__config__' JSON): no pickle at load time."""
    import torch
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = {k: ck['args'][k] for k in CFG_KEYS}
    cfg.update(seed=ck['args'].get('seed'), epoch=int(ck.get('epoch', -1)), val=float(ck.get('val', float('nan'))), version=VERSION)
    arrs = {k: v.detach().cpu().numpy() for k, v in ck['net'].items()}
    np.savez(out_path, __config__=np.array(json.dumps(cfg)), **arrs)
    return cfg


def load_model(npz_path, device='cpu'):
    import torch
    with np.load(npz_path, allow_pickle=False) as z:
        cfg = json.loads(str(z['__config__']))
        sd = {k: torch.from_numpy(np.array(z[k])) for k in z.files if k != '__config__'}
    XE = _xe_class()
    net = XE(d=cfg['d'], layers=cfg['layers'], heads=cfg['heads'], ff=cfg['ff'], dropout=cfg['dropout'], use_bt=not cfg['no_bt'])
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def load_models(paths, device='cpu'):
    return [load_model(p, device) for p in paths]


def pick_device(prefer='cuda'):
    import torch
    return torch.device('cuda' if prefer == 'cuda' and torch.cuda.is_available() else 'cpu')


# ------------------------------------------------------------------------------------------------ scoring -----------
def featurize(spectra, smiles, cache=None):
    """Arrays for every parsable candidate: (index list, cg [C, N_CAND], ai [C, S, P, N_INT], af [C, S, P, N_FLT])."""
    idx, cgs, ais, afs = [], [], [], []
    for i, s in enumerate(smiles):
        try:
            r = X.featurize_candidate(s, spectra, cache)
        except Exception:
            r = None
        if r is None:
            continue
        idx.append(i); cgs.append(r[0]); ais.append(r[1]); afs.append(r[2])
    if not idx:
        return idx, None, None, None
    return idx, np.stack(cgs), np.stack(ais), np.stack(afs)


def score_candidates(nets, spectra, smiles, device='cpu', cache=None, chunk=2048):
    """np.float64 [len(smiles)]: seed-mean of the mean-over-spectra logit; NaN where the candidate does not parse (or no spectra)."""
    import torch
    smiles = list(smiles)
    out = np.full(len(smiles), np.nan)
    if not spectra or not smiles or not nets:
        return out
    idx, cg, ai, af = featurize(spectra, smiles, cache)
    if not idx:
        return out
    f, mask, meta, ce = X.spectra_block(spectra)
    f = f.astype(np.float16).astype(np.float32)                  # training chunks store spec_f / ann_f as float16
    af = af.astype(np.float16).astype(np.float32)
    C, S = len(idx), len(spectra)
    ci = np.repeat(np.arange(C), S); si = np.tile(np.arange(S), C)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    tot = np.zeros(C)
    with torch.no_grad():
        for net in nets:
            lg = np.empty(C * S, np.float64)
            for a in range(0, C * S, chunk):
                b = min(C * S, a + chunk); c_, s_ = ci[a:b], si[a:b]
                D = dict(spf=T(f[s_]), mask=T(mask[s_]), ai=T(ai[c_, s_].astype(np.int64)), af=T(af[c_, s_]), meta=T(meta[s_].astype(np.int64)),
                         ce=T(ce[s_].astype(np.float32)), cg=T(cg[c_].astype(np.float32)))
                lg[a:b] = net(**D).float().cpu().numpy()
            tot += lg.reshape(C, S).mean(1)
    out[np.asarray(idx)] = tot / len(nets)
    return out


def score_dict(nets, rows, smiles, device='cpu', cache=None, max_spec=MAX_SPEC):
    """{smiles: score} with finite scores only (the FRAG_GAP column format)."""
    sp = prep_spectra(rows, max_spec)
    sc = score_candidates(nets, sp, smiles, device, cache)
    return {s: float(v) for s, v in zip(smiles, sc) if np.isfinite(v)}


# ------------------------------------------------------------------------------------------------ self-test ---------
SELFTEST_ROWS = [
    dict(mz=[77.0386, 105.0335, 123.0441, 133.0648, 151.0754], it=[0.15, 1.0, 0.2, 0.05, 0.4], prec=151.0754, adduct='[M+H]+',
         ionization_mode='positive', instrument='timsTOF Pro', ce=[20.0]),
    dict(mz=[92.0262, 136.0161, 137.0244], it=[0.3, 0.6, 1.0], prec=137.0244, adduct='[M-H]-', ionization_mode='negative',
         instrument='Orbitrap', ce=None),
    dict(mz=[105.0335, 173.0573], it=[1.0, 0.7], prec=173.0573, adduct='[M+Na]+', ionization_mode='positive', instrument='qTOF', ce=[10.0, 30.0]),
]
SELFTEST_SMILES = ['COC(=O)c1ccccc1O', 'COc1ccccc1C(=O)O', 'COc1ccc(C(=O)O)cc1', 'CC(=O)Oc1ccccc1', 'not a smiles']


def selftest_scores(nets, device='cpu'):
    return score_candidates(nets, prep_spectra(SELFTEST_ROWS), SELFTEST_SMILES, device, {})


def selftest(nets, expected, device='cpu', atol=2e-3):
    """Raises unless the fixed toy molecule's scores match `expected` (list, None = NaN) within atol and keep the same order."""
    got = selftest_scores(nets, device)
    exp = np.array([np.nan if e is None else float(e) for e in expected])
    assert np.array_equal(np.isnan(got), np.isnan(exp)), (got, exp)
    ok = ~np.isnan(exp)
    assert np.allclose(got[ok], exp[ok], atol=atol, rtol=0), (got, exp)
    assert list(np.argsort(-got[ok], kind='stable')) == list(np.argsort(-exp[ok], kind='stable')), (got, exp)
    return 'ok'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
