"""FRIGID-fingerprint head on our spectrum encoder: FRIGID's exact fingerprint, the model wrapper, checkpoints, prediction for query items.

Licence: ours. The trunk is clean/specenc.SpecEnc (ours) unchanged, with its own heads (the 10,226 pipeline bits and the element
counts); this file adds one more head of FRIGID's fingerprint size, trained with binary cross-entropy on FRIGID's exact bits so
that the frozen FRIGID decoder (coleygroup/FRIGID, Zenodo 19685145; not imported here) can be conditioned on our prediction.

FRIGID's fingerprint, read from the decoder checkpoint's config (model.fingerprint_bits 4096, model.fingerprint_radius 2) and from
dlm/utils/utils_data.Collator._safe_to_fingerprint: RDKit AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=4096) of the
parsed SMILES, a bit vector (no counts), no chirality, no feature invariants. The decoder's FingerprintSequenceEncoder embeds the
SET of bits > 0 (activation_threshold 0.0, at most 256 bits), so a prediction has to be binarised before it is passed on.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import specenc  # noqa: E402

FORMAT = 'fusion-v4n/clean/frigenc v1'
FRIG_BITS = 4096
FRIG_RADIUS = 2


def frigid_fp(smiles: str) -> np.ndarray | None:
    """FRIGID's conditioning fingerprint of a SMILES string: (4096,) uint8 of 0/1, or None when RDKit cannot parse it."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    a = np.zeros(FRIG_BITS, np.uint8)
    a[list(AllChem.GetMorganFingerprintAsBitVect(mol, FRIG_RADIUS, nBits=FRIG_BITS).GetOnBits())] = 1
    return a


def tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Tanimoto of two (n, bits) 0/1 arrays (0 when both are empty)."""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    inter = (a & b).sum(-1); union = (a | b).sum(-1)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


class FrigEnc(nn.Module):
    """specenc.SpecEnc plus a FRIGID head: forward(b) -> (pipeline-bit logits, element values, FRIGID-bit logits)."""

    def __init__(self, nbits: int, cfg: dict | None = None, drop: float = 0.1, base_rate: float = 46.0 / FRIG_BITS):
        super().__init__()
        self.net = specenc.SpecEnc(nbits, cfg, drop); d = self.net.d
        self.cfg, self.nbits, self.d = self.net.cfg, nbits, d
        self.fhead = nn.Sequential(nn.Linear(2 * d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, FRIG_BITS))
        with torch.no_grad():                                   # start at the mean bit density (median 46 bits set of 4,096)
            self.fhead[-1].bias.fill_(float(np.log(base_rate / (1 - base_rate))))

    def forward(self, b: dict):
        emb = self.net(b, return_emb=True)
        return self.net.head(emb), self.net.elem_head(emb), self.fhead(emb)


def forward_bucketed(model: FrigEnc, b: dict, nb: int):
    """specenc.forward_bucketed for the three outputs (outputs do not depend on padding)."""
    if nb <= 1:
        return model(b)
    n = (~b['pad']).sum(1); order = torch.argsort(n); outs = [[], [], []]
    for c in order.chunk(nb):
        for o, x in zip(outs, model(specenc.take(b, c, max(1, int(n[c].max()))))):
            o.append(x)
    inv = torch.empty_like(order); inv[order] = torch.arange(len(order), device=order.device)
    return tuple(torch.cat(o)[inv] for o in outs)


def save_ckpt(path, model: FrigEnc, step: int, extra: dict | None = None):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ck = dict(step=int(step), nbits=model.nbits, frig_bits=FRIG_BITS, frig_radius=FRIG_RADIUS, format=FORMAT, enc=dict(model.cfg), model=sd)
    ck.update(extra or {})
    torch.save(ck, path + '.tmp'); os.replace(path + '.tmp', path)


def load_ckpt(path, device='cpu') -> FrigEnc:
    ck = torch.load(path, map_location='cpu', weights_only=True)
    m = FrigEnc(ck['nbits'], ck['enc'])
    m.load_state_dict(ck['model'])
    return m.to(device).eval()


@torch.no_grad()
def predict(models, D, groups, bs: int = 64, device='cuda'):
    """FRIGID-bit probabilities (n, 4096) float32 for query groups (lists of spectrum indices of the cache D), averaged over
    `models` (one or more FrigEnc of the same token configuration)."""
    models = models if isinstance(models, (list, tuple)) else [models]
    cfg = models[0].cfg; out = []
    for s in range(0, len(groups), bs):
        b = specenc.collate([specenc.build_item(D, g, cfg) for g in groups[s:s + bs]], cfg, device)
        p = 0.0
        for m in models:
            with torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
                p = p + torch.sigmoid(m(b)[2].float())
        out.append((p / len(models)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, FRIG_BITS), np.float32)
