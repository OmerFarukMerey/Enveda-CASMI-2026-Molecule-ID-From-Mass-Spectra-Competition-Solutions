"""isoxe v2 inference: score one molecule's candidate SMILES against its MS/MS spectra with the graph-channel isomer cross-encoder.

Ours (2026-10-28; model and training: reviews/isoxe2-2026-10-28.md, train/isoxe2/). Pure functions, offline, CPU or GPU; needs NumPy,
RDKit, PyTorch (plain torch only: the graph layers use index_add_ / scatter_reduce, no torch_geometric / torch_scatter) and
optionally numba (through isoxe_feat). Ships with isoxe_feat.py (featuriser with the v2 graph outputs) and isoxe_infer.py (v1; its
spectrum preparation helpers are reused). The network classes are copies of train/isoxe2/model2.py's XE2 / GNN / CatEmb (same
parameter names, so the training state dicts load unchanged; tests/test_isoxe2_infer.py checks equality of outputs with the training
path).

Spectra: prepared as v1 (library-cache cleaning, precursor float32, mean collision energy, instrument class, polarity), but with
PEAKS = 40 most intense peaks. Spectrum-side and annotation floats pass through float16 as in the stored training chunks. Candidate
score = mean logit over the molecule's spectra (first `max_spec`, empty ones dropped), then the mean over seeds. Higher = better.

    nets = load_models(['isoxe2_s0.npz', 'isoxe2_s1.npz'], device)
    d = score_dict(nets, rows, smiles, device, cache)        # rows: dicts mz, it, prec, adduct, mode / ionization_mode, ce, instrument
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import isoxe_feat as X  # noqa: E402
import isoxe_infer as XI  # noqa: E402  (v1: clean_peaks, instrument_class, _ce_mean, _mode, rows_from_frame, sha256, pick_device)

VERSION = 2
PEAKS = 40
MAX_SPEC = 16
N_ADD, N_INSTR = len(X.ADDUCTS) + 1, 8
rows_from_frame, sha256, pick_device = XI.rows_from_frame, XI.sha256, XI.pick_device


def prep_row(row, peaks=PEAKS):
    """= isoxe_infer.prep_row with `peaks` most intense peaks (training: build_groups2 / Index.prepared with peaks=40)."""
    mz, it = XI.clean_peaks(row['mz'], row['it'], float(row['prec']))
    if len(mz) == 0:
        return None
    ins = row.get('instrument', 0)
    ins = int(ins) if isinstance(ins, (int, np.integer)) else XI.instrument_class(ins)
    ce = XI._ce_mean(row.get('ce'))
    return X.prep_spectrum(np.asarray(mz, np.float64), np.asarray(it, np.float64), float(np.float32(row['prec'])), row['adduct'], XI._mode(row),
                           [ce] if np.isfinite(ce) else None, ins, peaks=peaks)


def prep_spectra(rows, max_spec=MAX_SPEC):
    out = []
    for r in list(rows)[:max_spec]:
        try:
            sp = prep_row(r)
        except Exception:
            sp = None
        if sp is not None:
            out.append(sp)
    return out


# ------------------------------------------------------------------------------------------------ network -----------
def _classes():
    import torch
    import torch.nn as nn

    class CatEmb(nn.Module):
        def __init__(self, sizes, d):
            super().__init__()
            self.e = nn.ModuleList([nn.Embedding(s, d) for s in sizes])

        def forward(self, x):
            x = x.long()
            return sum(e(x[:, i].clamp(0, e.num_embeddings - 1)) for i, e in enumerate(self.e))

    class GNN(nn.Module):
        def __init__(self, dg=96, layers=3):
            super().__init__()
            self.ea, self.eb = CatEmb(X.AF_SIZES, dg), CatEmb(X.BF_SIZES, dg)
            self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(dg, 2 * dg), nn.GELU(), nn.Linear(2 * dg, dg)) for _ in range(layers)])
            self.norm = nn.ModuleList([nn.LayerNorm(dg) for _ in range(layers)])

        def forward(self, atom_f, bond_f, bond_e):
            h = self.ea(atom_f); e = self.eb(bond_f)
            a, b = bond_e[:, 0].long(), bond_e[:, 1].long()
            for mlp, ln in zip(self.mlp, self.norm):
                m = torch.zeros_like(h)
                m.index_add_(0, b, torch.relu(h[a] + e)); m.index_add_(0, a, torch.relu(h[b] + e))
                h = h + mlp(ln(h + m))
            return h, e

    class XE2(nn.Module):
        """= train/isoxe2/model2.py XE2 (parameter names and forward identical)."""

        def __init__(self, d=128, layers=4, heads=4, ff=256, dropout=0.1, use_bt=True, use_graph=True, dg=96, glayers=3):
            super().__init__()
            self.use_bt, self.use_graph = use_bt, use_graph
            self.pk = nn.Linear(X.N_SPF + X.N_FLT, d)
            self.ek = nn.Embedding(5, d); self.edh = nn.Embedding(8, d); self.ecar = nn.Embedding(4, d)
            self.ebt = nn.Embedding(X.NBT, d, padding_idx=0)
            self.cls_in = nn.Linear(X.N_CAND + 2, d); self.cls0 = nn.Parameter(torch.zeros(d))
            self.eadd = nn.Embedding(N_ADD, d); self.emode = nn.Embedding(2, d); self.einstr = nn.Embedding(N_INSTR, d)
            if use_graph:
                self.gnn = GNN(dg, glayers)
                self.gread = nn.Sequential(nn.Linear(2 * dg, d), nn.GELU(), nn.Linear(d, d))
                self.gbond = nn.Sequential(nn.Linear(dg, d), nn.GELU(), nn.Linear(d, d))
            enc = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation='gelu')
            self.tr = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        def graph(self, atom_f, bond_f, bond_e, atom_cand, n_cand):
            h, e = self.gnn(atom_f, bond_f, bond_e)
            dg = h.shape[1]
            s = torch.zeros(n_cand, dg, device=h.device, dtype=h.dtype).index_add_(0, atom_cand, h)
            n = torch.zeros(n_cand, device=h.device, dtype=h.dtype).index_add_(0, atom_cand, torch.ones_like(h[:, 0])).clamp(min=1)
            mx = torch.full((n_cand, dg), -1e4, device=h.device, dtype=h.dtype).scatter_reduce(0, atom_cand[:, None].expand(-1, dg), h, 'amax')
            mx = torch.where(mx < -1e3, torch.zeros_like(mx), mx)
            r = self.gread(torch.cat([s / n[:, None], mx], 1))
            a, b = bond_e[:, 0].long(), bond_e[:, 1].long()
            zb = self.gbond(h[a] + h[b] + e)
            zb = torch.cat([zb, torch.zeros_like(zb[:1])], 0)
            return r, zb

        def forward(self, spf, mask, ai, af, meta, ce, cg, seq_cand=None, ann_b=None, atom_f=None, bond_f=None, bond_e=None, atom_cand=None,
                    n_cand=None):
            ai = ai.long(); meta = meta.long()
            x = self.pk(torch.cat([spf, af], -1)) + self.ek(ai[..., 0]) + self.edh(ai[..., 1]) + self.ecar(ai[..., 2])
            if self.use_bt:
                x = x + self.ebt(ai[..., 3:6]).sum(-2)
            m = self.eadd(meta[:, 0].clamp(0, N_ADD - 1)) + self.emode(meta[:, 1].clamp(0, 1)) + self.einstr(meta[:, 2].clamp(0, N_INSTR - 1))
            c = self.cls0 + self.cls_in(torch.cat([cg, ce[:, None], meta[:, 3:4].float()], -1)) + m
            if self.use_graph:
                r, zb = self.graph(atom_f, bond_f, bond_e, atom_cand, n_cand)
                c = c + r[seq_cand].to(c.dtype)
                nb = zb.shape[0] - 1
                bi = ann_b.long(); bi = torch.where(bi < 0, torch.full_like(bi, nb), bi)
                x = x + zb[bi].sum(-2).to(x.dtype)
            x = x + m[:, None, :]
            seq = torch.cat([c[:, None, :], x], 1)
            pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
            h = self.tr(seq, src_key_padding_mask=pad)
            return self.head(h[:, 0]).squeeze(-1)

    return XE2


CFG_KEYS = ('d', 'layers', 'heads', 'ff', 'dropout', 'no_graph', 'dg', 'glayers')


def export_npz(ckpt_path, out_path):
    """Training checkpoint (train_isoxe2.py best.pt) -> plain npz (float32 arrays + '__config__' JSON): no pickle at load time."""
    import torch
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = {k: ck['args'][k] for k in CFG_KEYS}
    cfg.update(seed=ck['args'].get('seed'), epoch=int(ck.get('epoch', -1)), val=float(ck.get('val', float('nan'))), version=VERSION, peaks=PEAKS)
    arrs = {k: v.detach().cpu().numpy() for k, v in ck['net'].items()}
    np.savez(out_path, __config__=np.array(json.dumps(cfg)), **arrs)
    return cfg


def load_model(npz_path, device='cpu'):
    import torch
    with np.load(npz_path, allow_pickle=False) as z:
        cfg = json.loads(str(z['__config__']))
        sd = {k: torch.from_numpy(np.array(z[k])) for k in z.files if k != '__config__'}
    net = _classes()(d=cfg['d'], layers=cfg['layers'], heads=cfg['heads'], ff=cfg['ff'], dropout=cfg['dropout'], use_graph=not cfg['no_graph'],
                     dg=cfg['dg'], glayers=cfg['glayers'])
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def load_models(paths, device='cpu'):
    return [load_model(p, device) for p in paths]


# ------------------------------------------------------------------------------------------------ scoring -----------
def featurize(spectra, smiles, cache=None):
    """Parsable candidates -> (index list, list of (cg, ai, af, ab, graph))."""
    idx, feats = [], []
    for i, s in enumerate(smiles):
        try:
            r = X.featurize_candidate(s, spectra, cache, peaks=PEAKS, bonds=True)
        except Exception:
            r = None
        if r is None:
            continue
        idx.append(i); feats.append(r)
    return idx, feats


def score_candidates(nets, spectra, smiles, device='cpu', cache=None, cand_chunk=48):
    """np.float64 [len(smiles)]: seed-mean of the mean-over-spectra logit; NaN where the candidate does not parse (or no spectra)."""
    import torch
    smiles = list(smiles)
    out = np.full(len(smiles), np.nan)
    if not spectra or not smiles or not nets:
        return out
    idx, feats = featurize(spectra, smiles, cache)
    if not idx:
        return out
    f, mask, meta, ce = X.spectra_block(spectra, peaks=PEAKS)
    f = f.astype(np.float16).astype(np.float32)                  # training chunks store spec_f / ann_f as float16
    S = len(spectra)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    tot = np.zeros(len(idx))
    with torch.no_grad():
        for c0 in range(0, len(idx), cand_chunk):
            fe = feats[c0:c0 + cand_chunk]; C = len(fe)
            na = np.array([g['n'] for *_, g in fe]); nb = np.array([len(g['ea']) for *_, g in fe])
            aoff = np.r_[0, np.cumsum(na)[:-1]]; boff = np.r_[0, np.cumsum(nb)[:-1]]
            atom_f = np.concatenate([g['afeat'] for *_, g in fe]); bond_f = np.concatenate([g['bfeat'].reshape(-1, X.N_BF) for *_, g in fe])
            bond_e = np.concatenate([np.stack([g['ea'], g['eb']], 1).reshape(-1, 2) + aoff[k] for k, (*_, g) in enumerate(fe)]).astype(np.int64)
            atom_cand = np.repeat(np.arange(C), na)
            ci = np.repeat(np.arange(C), S); si = np.tile(np.arange(S), C)
            ai = np.concatenate([r[1] for r in fe]).astype(np.int64)                    # [C*S, P, N_INT] (candidate-major)
            af = np.concatenate([r[2] for r in fe]).astype(np.float16).astype(np.float32)
            ab = np.concatenate([r[3] for r in fe]).astype(np.int64)
            ab = np.where(ab >= 0, ab + boff[ci][:, None, None], -1)
            cg = np.stack([r[0] for r in fe]).astype(np.float32)
            D = dict(spf=T(f[si]), mask=T(mask[si]), ai=T(ai), af=T(af), meta=T(meta[si].astype(np.int64)), ce=T(ce[si].astype(np.float32)),
                     cg=T(cg[ci]), seq_cand=T(ci), ann_b=T(ab), atom_f=T(atom_f), bond_f=T(bond_f), bond_e=T(bond_e), atom_cand=T(atom_cand),
                     n_cand=C)
            acc = np.zeros(C)
            for net in nets:
                acc += net(**D).float().cpu().numpy().astype(np.float64).reshape(C, S).mean(1)
            tot[c0:c0 + C] = acc / len(nets)
    out[np.asarray(idx)] = tot
    return out


def score_dict(nets, rows, smiles, device='cpu', cache=None, max_spec=MAX_SPEC):
    """{smiles: score} with finite scores only (the format fz_core expects)."""
    sp = prep_spectra(rows, max_spec)
    sc = score_candidates(nets, sp, smiles, device, cache)
    return {s: float(v) for s, v in zip(smiles, sc) if np.isfinite(v)}


# ------------------------------------------------------------------------------------------------ self-test ---------
SELFTEST_ROWS = XI.SELFTEST_ROWS
SELFTEST_SMILES = XI.SELFTEST_SMILES


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
