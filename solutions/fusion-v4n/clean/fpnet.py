"""Our own fingerprint network (spectrum -> fingerprint-bit logits): model class, peak preparation, collate, checkpoint I/O.

Licence: ours. Written from clean/SPEC.md section 6 and from OBSERVED BEHAVIOUR of the reference network: the parameter
names and shapes of the public checkpoint `fpnet_fold0.pt` (a data file) and the inputs / outputs of the reference
module's submodules recorded with forward hooks (scratch/ownfp/probe_ref.py, probe_fit.py). The reference source file
(`casmi/fpnet.py`, no code licence) was not opened for this file. Clean-room limit: the specification was written by
an author who had read that source, and the layout below is pinned to the reference by black-box probing so that the
public checkpoint loads into this class and our checkpoints load into the current engine. See clean/README.md.

Layout (pinned by the checkpoint format; verified by tests/test_own_fpnet.py and scratch/ownfp/parity.py):
  peak token   = Linear(1536 -> d)([sin/cos(m/z), sin/cos(max(precursor - m/z, 0)), Linear(2 -> d)([s, s^2])]), s = sqrt intensity
  global token = Linear(644 -> d)([sin/cos(precursor), sin/cos(ce / 10 + 1), ce_known, n_merged / 4, mode,
                 log1p(precursor) / 10]) + Embedding(adduct)
  encoder      = pre-norm transformer blocks (fused qkv, GELU feed-forward 4d, residual dropout), final LayerNorm
  heads        = MLP([global token, mean of the valid peak tokens]) -> nbits logits; a second MLP -> 10 element values
Decided by us where neither the specification nor the checkpoint says anything: PyTorch default initialisation, no
dropout inside the attention, padding to the longest spectrum of the batch (the output does not depend on padding).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_PEAKS = 160
N_ADDUCT = 60
N_ELEM = 10
FORMAT = 'fusion-v4n/clean/fpnet v1'


class SinEmb(nn.Module):
    """x -> [sin(x * inv), cos(x * inv)] with wavelengths 2 pi / inv spaced geometrically between lam_min and lam_max."""

    def __init__(self, dim: int, lam_min: float, lam_max: float):
        super().__init__()
        lam = torch.logspace(math.log10(lam_min), math.log10(lam_max), dim // 2, dtype=torch.float64)
        self.register_buffer('inv', (2 * math.pi / lam).float())

    def forward(self, x):
        a = x.float().unsqueeze(-1) * self.inv
        return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, drop: float):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=(~pad)[:, None, None, :])
        x = x + self.drop(self.o(y.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.n2(x)))


class FPNet(nn.Module):
    def __init__(self, nbits: int, d: int = 512, layers: int = 8, heads: int = 8, drop: float = 0.1, n_elem: int = N_ELEM, n_instr: int = 0):
        super().__init__()
        self.nbits, self.d, self.n_instr = nbits, d, n_instr
        if n_instr:                                           # optional instrument-class input (not in the reference layout)
            self.ins = nn.Embedding(n_instr, d); nn.init.zeros_(self.ins.weight)
        self.mz_emb = SinEmb(d, 0.01, 10 ** 3.3); self.nl_emb = SinEmb(d, 0.01, 10 ** 3.3)
        self.int_emb = nn.Linear(2, d); self.pk = nn.Linear(3 * d, d)
        self.prec_emb = SinEmb(d, 0.01, 10 ** 3.3); self.ad = nn.Embedding(N_ADDUCT, d)
        self.ce_emb = SinEmb(d // 4, 1.0, 10 ** 2.2); self.gl = nn.Linear(d + d // 4 + 4, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, nbits))
        self.elem_head = nn.Sequential(nn.Linear(2 * d, d // 2), nn.GELU(), nn.Linear(d // 2, n_elem))

    def forward(self, mz, it, pad, prec, ad, ce, ce_known, merged, mode, return_emb: bool = False, instr=None):
        """mz, it (B, P) float; pad (B, P) bool, True = padding; the rest (B,). -> (bit logits (B, nbits), elements (B, 10))."""
        prec = prec.float()
        pk = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb((prec[:, None] - mz).clamp(min=0)), self.int_emb(torch.stack([it, it * it], -1))], -1))
        g = torch.cat([self.prec_emb(prec), self.ce_emb(ce / 10.0 + 1.0),
                       torch.stack([ce_known, merged, mode, torch.log1p(prec) / 10.0], -1)], -1)
        g = self.gl(g) + self.ad(ad)
        if self.n_instr and instr is not None:
            g = g + self.ins(instr)
        x = torch.cat([g[:, None].to(pk.dtype), pk], 1)
        m = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        for b in self.blocks:
            x = b(x, m)
        x = self.norm(x)
        v = (~pad).unsqueeze(-1).to(x.dtype)
        emb = torch.cat([x[:, 0], (x[:, 1:] * v).sum(1) / v.sum(1).clamp(min=1.0)], -1)
        if return_emb:
            return emb
        return self.head(emb), self.elem_head(emb)


def prep_peaks(mz, it, prec_mz, max_peaks: int = MAX_PEAKS, floor: float = 0.001):
    """Peaks of one (cleaned or merged) spectrum as the network sees them: 0 < m/z <= precursor + 2, intensity >= floor x
    the largest in that range, the max_peaks most intense, ascending m/z, sqrt of the base-peak-normalised intensity."""
    mz = np.asarray(mz, np.float32); it = np.asarray(it, np.float32)
    k = (mz > 0) & (mz <= prec_mz + 2.0)
    mz, it = mz[k], it[k]
    if not len(mz) or it.max() <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mx = it.max(); k = it >= floor * mx
    mz, it = mz[k], it[k]
    if len(mz) > max_peaks:
        o = np.argsort(-it)[:max_peaks]; mz, it = mz[o], it[o]       # NumPy's default sort decides ties at the cut (as the engine's preparation does)
    o = np.argsort(mz, kind='stable')
    return mz[o], np.sqrt(it[o] / mx)


def make_item(mz, it, prec, adduct_ix, ce, n_merged, mode):
    """One network input. ce = collision energy in eV or NaN (unknown); n_merged = number of acquisitions behind the peaks."""
    known = not (ce is None or np.isnan(ce))
    pm, pi = prep_peaks(mz, it, prec)
    return dict(mz=pm, it=pi, prec=float(prec), adduct_ix=int(adduct_ix), ce=float(ce) if known else 0.0,
                ce_known=1.0 if known else 0.0, n_merged=int(min(max(n_merged, 1), 8)), mode=float(mode))


def collate(items: Sequence[dict], device='cpu', pad_to: int | None = None):
    """Items -> the nine tensors of FPNet.forward, padded to the longest spectrum of the batch (or to pad_to)."""
    B = len(items); P = pad_to or max(1, max(len(x['mz']) for x in items))
    mz = np.zeros((B, P), np.float32); it = np.zeros((B, P), np.float32); pad = np.ones((B, P), bool)
    for i, x in enumerate(items):
        n = len(x['mz']); mz[i, :n] = x['mz']; it[i, :n] = x['it']; pad[i, :n] = False
        if n == 0:
            pad[i, 0] = False                                # an empty spectrum is one zero peak (observed interface behaviour)
    f = lambda k, dt=np.float32: torch.from_numpy(np.array([x[k] for x in items], dt))
    out = (torch.from_numpy(mz), torch.from_numpy(it), torch.from_numpy(pad), f('prec'), f('adduct_ix', np.int64), f('ce'),
           f('ce_known'), f('n_merged') / 4.0, f('mode'))
    return tuple(a.to(device) for a in out)


_BITW = None


def unpack_bits(packed: torch.Tensor, nbits: int) -> torch.Tensor:
    """(..., ceil(nbits / 8)) uint8 as written by numpy.packbits (most significant bit first) -> (..., nbits) float32."""
    global _BITW
    if _BITW is None or _BITW.device != packed.device:
        _BITW = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=packed.device)
    b = (packed.unsqueeze(-1) & _BITW) != 0
    return b.flatten(-2)[..., :nbits].float()


def save_ckpt(path, model: FPNet, step: int, extra: dict | None = None):
    """Checkpoint in the field layout the engine's loader reads (step, nbits, d, layers, heads, model) plus our own fields."""
    b = model.blocks[0]
    ck = dict(step=int(step), nbits=model.nbits, d=model.d, layers=len(model.blocks), heads=b.h, format=FORMAT,
              model={k: v.detach().cpu() for k, v in model.state_dict().items()})
    ck.update(extra or {})
    torch.save(ck, path)


def fold_instrument(state: dict, instr: int = 0) -> dict:
    """State dict of an instrument-conditioned network -> reference layout for ONE instrument class (0 = timsTOF): its
    embedding row is added to the bias of the global-token layer, which gives the same outputs for that class."""
    s = dict(state); w = s.pop('ins.weight')
    s['gl.bias'] = s['gl.bias'] + w[instr].to(s['gl.bias'].dtype)
    return s


def load_ckpt(path, device='cpu') -> FPNet:
    """Our checkpoints and the public reference format (same field names) -> FPNet in eval mode. weights_only: no pickle code."""
    ck = torch.load(path, map_location='cpu', weights_only=True)
    n_instr = ck['model']['ins.weight'].shape[0] if 'ins.weight' in ck['model'] else 0
    m = FPNet(ck['nbits'], d=ck['d'], layers=ck['layers'], heads=ck.get('heads', 8), n_instr=n_instr)
    m.load_state_dict(ck['model'])
    return m.to(device).eval()
