"""isoxe model: isomer cross-encoder over (candidate, spectrum) sequences of peak tokens annotated with the candidate's fragments.

One sequence = CLS (candidate-global features + spectrum metadata) + up to PEAKS peak tokens (spectrum-side features + the
candidate-specific annotation of isoxe_feat: cleavage count, H shift, carrier, bond-type embeddings, error / fragment fractions).
A small pre-norm Transformer encoder reads it; the CLS state gives one logit per (candidate, spectrum); a candidate's score is
the mean over the molecule's spectra. Trained listwise over same-formula groups (train_isoxe.py).
Also: `Store` (all feature chunks of a split in RAM, with group / candidate / spectrum offsets) and `gather` (one batch).
"""
import glob, os, sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'clean'))
import isoxe_feat as X  # noqa: E402

N_ADD, N_INSTR = len(X.ADDUCTS) + 1, 8


class XE(nn.Module):
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
        """spf [N, P, N_SPF], mask [N, P] bool, ai [N, P, N_INT] long, af [N, P, N_FLT], meta [N, 4] long, ce [N], cg [N, N_CAND] -> [N]."""
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


class Store:
    """All chunks of a feature directory in RAM (torch CPU tensors) with offsets."""

    def __init__(self, d, limit=None):
        files = sorted(glob.glob(os.path.join(d, 'chunk_*.npz')))
        if limit:
            files = files[:limit]
        parts = [dict(np.load(f, allow_pickle=False)) for f in files]
        cat = lambda k: np.concatenate([p[k] for p in parts])
        self.ncand = cat('g_ncand').astype(np.int64); self.nspec = cat('g_nspec').astype(np.int64)
        self.e180 = cat('g_e180').astype(bool); self.gid = cat('g_gid')
        self.chunk = np.concatenate([np.full(len(p['g_ncand']), i) for i, p in enumerate(parts)])
        self.c0 = np.r_[0, np.cumsum(self.ncand)[:-1]]; self.s0 = np.r_[0, np.cumsum(self.nspec)[:-1]]
        self.a0 = np.r_[0, np.cumsum(self.ncand * self.nspec)[:-1]]
        self.lab = cat('cand_lab').astype(np.int64); self.src = cat('cand_src'); self.smiles = cat('cand_smiles')
        T = lambda k, dt=None: torch.from_numpy(cat(k) if dt is None else cat(k).astype(dt))
        self.cg = T('cand_g', np.float32); self.spf = T('spec_f'); self.mask = T('spec_mask'); self.meta = T('spec_meta')
        self.ce = T('spec_ce', np.float32); self.ai = T('ann_i'); self.af = T('ann_f')
        self.n = len(self.ncand)
        assert self.ai.shape[0] == int((self.ncand * self.nspec).sum()) and self.cg.shape[0] == int(self.ncand.sum())

    def seq_index(self, groups, spec_pick):
        """Rows for every (candidate, picked spectrum) of the groups. spec_pick: list of arrays of spectrum positions per group.
        Returns (cand rows, spec rows, ann rows, owner = position of the candidate within the batch's padded [B, Cmax] grid)."""
        cr, sr, ar, own = [], [], [], []
        cmax = int(self.ncand[groups].max())
        for b, (g, sp) in enumerate(zip(groups, spec_pick)):
            nc, ns = int(self.ncand[g]), int(self.nspec[g])
            c = np.repeat(np.arange(nc), len(sp)); s = np.tile(sp, nc)
            cr.append(self.c0[g] + c); sr.append(self.s0[g] + s); ar.append(self.a0[g] + c * ns + s); own.append(b * cmax + c)
        return np.concatenate(cr), np.concatenate(sr), np.concatenate(ar), np.concatenate(own), cmax

    def gather(self, groups, spec_pick, device):
        cr, sr, ar, own, cmax = self.seq_index(groups, spec_pick)
        t = lambda x: torch.from_numpy(x)
        cr, sr, ar = t(cr), t(sr), t(ar)
        D = dict(spf=self.spf[sr].float(), mask=self.mask[sr], ai=self.ai[ar], af=self.af[ar].float(), meta=self.meta[sr], ce=self.ce[sr], cg=self.cg[cr])
        D = {k: v.to(device, non_blocking=True) for k, v in D.items()}
        return D, torch.from_numpy(own).to(device), cmax


def group_scores(logits, own, B, cmax):
    """Mean logit per candidate over its sequences -> [B, cmax] with -inf padding."""
    s = torch.zeros(B * cmax, device=logits.device, dtype=logits.dtype).index_add_(0, own, logits)
    n = torch.zeros(B * cmax, device=logits.device, dtype=logits.dtype).index_add_(0, own, torch.ones_like(logits))
    out = torch.where(n > 0, s / n.clamp(min=1), torch.full_like(s, float('-inf')))
    return out.view(B, cmax)


def isomer_rr(scores, lab):
    """1 / (1 + #negatives above + 0.5 #ties) of the best-scored positive (numpy)."""
    pos = scores[lab == 1].max(); neg = scores[lab != 1]
    return 1.0 / (1.0 + (neg > pos).sum() + 0.5 * (neg == pos).sum())
