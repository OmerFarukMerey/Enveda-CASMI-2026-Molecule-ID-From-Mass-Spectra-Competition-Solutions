"""Our own spectrum <-> structure compatibility network (ownneural): multi-spectrum peak-set transformer x fingerprint MLP, trained listwise.

Licence: ours. Imports NumPy and PyTorch only. Written for the ownneural study (scratch/ownneural/PREREG.md); the encoders follow our joint
study's reference model (train/joint/model.py, ours) so the two stay comparable, with three changes for the ranks-2-25 slot:
  * attention pooling over ALL spectra of a molecule (any adduct, both polarities), not a sample of four;
  * a listwise loss over the molecule's own candidate list (the true structure against same-formula isomers, PubChem window rows and
    near-mass analogues of another formula) next to the in-batch InfoNCE, with a per-molecule weight (Enveda timsTOF views weigh more);
  * an optional fingerprint-prediction head (w_aux) whose bit logits give a second, f.z-like score.

Inputs of a spectrum batch (dict of tensors; `spectra_arrays` builds it from (m/z, intensity) arrays):
  mz, it [T, P] (cleaned peaks, base peak 1, P most intense in m/z order), n_peaks [T], prec [T] (precursor m/z), add [T] (adduct index of
  clean/specdata.ADDUCT_IX), mode [T] (1 positive, 0 negative), ce [T] (mean collision energy, 0 when unknown), ce_known [T], instr [T]
  (clean/specdata.instrument_class), owner [T] (molecule index).
Structures: packed fingerprint bits (np.packbits order) of the pool's selected bits.
Score of (molecule, candidate) = cosine of the two embeddings (the trained quantity; higher = more compatible).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier(x, n_freq, lo, hi):
    """Sinusoidal features of a scalar, wavelengths log-spaced from lo to hi: [...] -> [..., 2 n_freq]."""
    w = 2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), n_freq, device=x.device, dtype=torch.float32)
    a = x.float().unsqueeze(-1) * w
    return torch.cat([a.sin(), a.cos()], -1)


class SpecEncoder(nn.Module):
    """One token per peak (Fourier features of m/z and of the neutral loss precursor - m/z, sqrt and log intensity) after a metadata token
    (precursor, collision energy, adduct, polarity, instrument class); pre-norm transformer; the metadata token of every spectrum is pooled
    over the molecule's spectra (attention or mean)."""

    def __init__(self, d=256, layers=6, heads=8, n_freq=32, n_adduct=64, n_instr=8, dropout=0.1, pool='attn'):
        super().__init__()
        self.n_freq = n_freq; self.pool = pool
        self.peak = nn.Sequential(nn.Linear(4 * n_freq + 2, d), nn.GELU(), nn.Linear(d, d))
        self.adduct = nn.Embedding(n_adduct, d); self.mode = nn.Embedding(2, d); self.instr = nn.Embedding(n_instr, d)
        self.meta = nn.Sequential(nn.Linear(2 * n_freq + 2 * 8 + 1, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
        if pool == 'attn':
            self.att = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))

    def spectra(self, b):
        mz, it, prec = b['mz'], b['it'], b['prec']; T, Pk = mz.shape
        pad = torch.arange(Pk, device=mz.device)[None, :] >= b['n_peaks'][:, None]
        nl = (prec[:, None] - mz).clamp(min=0)
        x = torch.cat([fourier(mz, self.n_freq, 0.01, 2000.0), fourier(nl, self.n_freq, 0.01, 2000.0), it.sqrt()[..., None],
                       (torch.log10(it.clamp(min=1e-4)) / 4)[..., None]], -1)
        tok = self.peak(x.to(self.peak[0].weight.dtype))
        ck = b['ce_known']
        m = torch.cat([fourier(prec, self.n_freq, 0.01, 2000.0), fourier(b['ce'], 8, 1.0, 400.0) * ck[:, None], ck[:, None]], -1)
        cls = self.meta(m.to(tok.dtype)) + self.adduct(b['add']) + self.mode(b['mode']) + self.instr(b['instr'])
        h = self.enc(torch.cat([cls[:, None], tok.to(cls.dtype)], 1),
                     src_key_padding_mask=torch.cat([torch.zeros(T, 1, dtype=torch.bool, device=mz.device), pad], 1))
        return self.norm(h[:, 0])

    def pool_spectra(self, h, own, n):
        h = h.float()
        if self.pool == 'attn':
            a = self.att(h.to(self.att[0].weight.dtype)).squeeze(-1).float()
            mx = torch.full((n,), -1e9, device=h.device).index_reduce_(0, own, a.detach(), 'amax', include_self=False)
            w = (a - mx[own]).exp()
        else:
            w = torch.ones(len(h), device=h.device)
        s = torch.zeros(n, h.shape[1], device=h.device).index_add_(0, own, h * w[:, None])
        c = torch.zeros(n, device=h.device).index_add_(0, own, w).clamp(min=1e-6)
        return s / c[:, None]


def unpack_bits(packed, nbits):
    """uint8 [n, nbytes] -> {0,1} [n, nbits] (np.packbits order)."""
    sh = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    return ((packed[:, :, None] >> sh) & 1).reshape(packed.shape[0], -1)[:, :nbits]


class FpEncoder(nn.Module):
    def __init__(self, nbits=10226, hidden=(1024, 512), out=256, dropout=0.1):
        super().__init__()
        self.nbits = nbits; dims = (nbits,) + tuple(hidden); layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], out))

    def forward(self, packed):
        return F.normalize(self.net(unpack_bits(packed, self.nbits).to(self.net[0].weight.dtype)).float(), dim=-1)


class ONNet(nn.Module):
    def __init__(self, nbits=10226, d=256, layers=6, out=256, n_adduct=64, n_instr=8, hidden=(1024, 512), dropout=0.1, pool='attn',
                 temp=0.05, w_list=1.0, w_nce=0.5, w_m2s=0.25, w_aux=0.0):
        super().__init__()
        self.cfg = dict(nbits=nbits, d=d, layers=layers, out=out, n_adduct=n_adduct, n_instr=n_instr, hidden=tuple(hidden), dropout=dropout, pool=pool,
                        temp=temp, w_list=w_list, w_nce=w_nce, w_m2s=w_m2s, w_aux=w_aux)
        self.spec = SpecEncoder(d=d, layers=layers, n_adduct=n_adduct, n_instr=n_instr, dropout=dropout, pool=pool)
        self.sproj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, out))
        self.struct = FpEncoder(nbits, hidden, out, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp)))
        self.w_aux = w_aux; self.nbits = nbits
        if w_aux > 0:
            self.aux = nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, nbits))

    def pooled(self, b, n):
        return self.spec.pool_spectra(self.spec.spectra(b), b['owner'], n)

    def embed_spec(self, hp):
        return F.normalize(self.sproj(hp.to(self.sproj[0].weight.dtype)).float(), dim=-1)

    def forward(self, b):
        """Training loss of a batch (spectra + pos_fp [B, nbytes] + neg_fp [B, K, nbytes] + neg_ok [B, K] + pid [B] + w [B])."""
        B = b['pos_fp'].shape[0]; K = b['neg_fp'].shape[1]
        hp = self.pooled(b, B); zs = self.embed_spec(hp)
        ze = self.struct(torch.cat([b['pos_fp'], b['neg_fp'].reshape(B * K, -1)], 0))
        loss, acc = list_nce_loss(zs, ze[:B], ze[B:].reshape(B, K, -1), b['pid'], b['neg_ok'], b['w'], self.logit_scale.exp().clamp(max=100.0),
                                  self.cfg['w_list'], self.cfg['w_nce'], self.cfg['w_m2s'])
        if self.w_aux > 0:
            y = unpack_bits(b['pos_fp'], self.nbits).float(); p = self.aux(hp.to(self.aux[0].weight.dtype)).float()
            loss = loss + self.w_aux * F.binary_cross_entropy_with_logits(p, y, reduction='none').mean(1).mul(b['w']).sum() / b['w'].sum()
        return loss, acc


def list_nce_loss(zs, zp, zn, pid, neg_ok, w, scale, w_list=1.0, w_nce=0.5, w_m2s=0.25):
    """zs [B, d] molecule embeddings, zp [B, d] true structures, zn [B, K, d] the molecule's own negatives, neg_ok [B, K] bool, w [B] weights.
    list: softmax over (true, own negatives); nce: over the batch's true structures (in-batch duplicates of a structure masked) and the own
    negatives, plus the structure -> molecule direction over the batch. Returns (loss, accuracy of the list term)."""
    B = zs.shape[0]; eye = torch.eye(B, dtype=torch.bool, device=zs.device); same = (pid[:, None] == pid[None, :]) & ~eye
    w = w.float(); wn = w / w.sum()
    cpos = (zs * zp).sum(-1); cown = torch.einsum('bd,bkd->bk', zs, zn)
    own = (cown * scale).masked_fill(~neg_ok, float('-inf'))
    ll = torch.cat([(cpos * scale)[:, None], own], 1)
    l_list = (F.cross_entropy(ll, torch.zeros(B, dtype=torch.long, device=zs.device), reduction='none') * wn).sum()
    acc = (ll.argmax(1) == 0).float().mean()
    loss = w_list * l_list
    if w_nce > 0:
        y = torch.arange(B, device=zs.device)
        inb = ((zs @ zp.T) * scale).masked_fill(same, float('-inf'))
        l_s2m = (F.cross_entropy(torch.cat([inb, own], 1), y, reduction='none') * wn).sum()
        l_m2s = (F.cross_entropy(inb.T, y, reduction='none') * wn).sum()
        loss = loss + w_nce * ((1 - w_m2s) * l_s2m + w_m2s * l_m2s)
    return loss, acc


# ---------------------------------------------------------------- inputs
def spectra_arrays(peaks, prec, add, mode, ce, instr, owner, max_peaks=96):
    """Pad (m/z, intensity) arrays to [T, P], keeping the max_peaks most intense peaks of each spectrum in m/z order. mode: >0 positive."""
    T = len(peaks); n = np.array([min(len(m), max_peaks) for m, _ in peaks], np.int64); Pm = max(int(n.max()) if T else 1, 1)
    mz = np.zeros((T, Pm), np.float32); it = np.zeros((T, Pm), np.float32)
    for t, (m, v) in enumerate(peaks):
        m = np.asarray(m); v = np.asarray(v)
        if len(m) > max_peaks:
            keep = np.sort(np.argpartition(-v, max_peaks - 1)[:max_peaks]); m = m[keep]; v = v[keep]
        mz[t, :len(m)] = m; it[t, :len(m)] = v
    ce = np.asarray(ce, np.float32); known = np.isfinite(ce)
    return dict(mz=mz, it=it, n_peaks=n, prec=np.asarray(prec, np.float32), add=np.asarray(add, np.int64), mode=(np.asarray(mode) > 0).astype(np.int64),
                ce=np.where(known, ce, 0.0).astype(np.float32), ce_known=known.astype(np.float32), instr=np.asarray(instr, np.int64), owner=np.asarray(owner, np.int64))


def to_torch(b, dev):
    return {k: (torch.as_tensor(v, device=dev) if isinstance(v, (np.ndarray, torch.Tensor)) else v) for k, v in b.items()}


def save_ckpt(path, model, extra=None):
    torch.save(dict(cfg=model.cfg, model={k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, **(extra or {})), path)


def load_ckpt(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck['cfg']); m = ONNet(**cfg).to(device)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck['model'].items()}); m.eval(); m.meta = {k: v for k, v in ck.items() if k not in ('model', 'opt')}
    return m


@torch.no_grad()
def embed_molecules(m, arrays, n_mol, dev, chunk=512):
    """Spectrum arrays (owner = molecule index 0..n_mol-1) -> [n_mol, out] embeddings (pooled over all the molecule's spectra), and the pooled
    hidden vectors. Spectra are encoded in chunks; pooling happens once over all of them."""
    T = len(arrays['owner']); H = []
    for s in range(0, max(T, 1), chunk):
        sub = {k: v[s:s + chunk] for k, v in arrays.items()}
        H.append(m.spec.spectra(to_torch(sub, dev)).float())
    hp = m.spec.pool_spectra(torch.cat(H), torch.as_tensor(arrays['owner'], device=dev), n_mol)
    return m.embed_spec(hp), hp


@torch.no_grad()
def embed_structs(m, packed, dev, chunk=8192):
    out = []
    for s in range(0, len(packed), chunk):
        out.append(m.struct(torch.as_tensor(np.ascontiguousarray(packed[s:s + chunk]), device=dev)))
    return torch.cat(out) if out else torch.zeros(0, m.cfg['out'], device=dev)
