"""isoxe v2 model: v1's isomer cross-encoder (peak tokens annotated with the candidate's fragments) plus a graph channel.

v1 (train/isoxe/model.py): one sequence per (candidate, spectrum) = CLS + peak tokens; pre-norm Transformer; candidate score =
mean logit over the molecule's spectra. v2 adds a small message-passing network over the candidate's heavy-atom graph
(isoxe_feat atom / bond categorical features, GINE-style residual layers, pure torch, no PyG):
  - its readout (mean + max over atoms) is added to the CLS token of every sequence of that candidate;
  - every annotated peak adds, for each bond its fragment broke (isoxe_feat bonds=True, <= 3), an embedding computed from the
    bond's two end-atom states and its own features, so the transformer sees WHERE in the candidate a peak's cleavage happens.
`use_graph=False` gives v1's architecture on v2 features (ablation). No list rank or pipeline score is an input.
Also `Store2` (feature chunks of build_groups2.py in RAM with candidate / spectrum / atom / bond offsets) and its `gather`.
"""
import glob, os, sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'clean'))
import isoxe_feat as X  # noqa: E402

N_ADD, N_INSTR = len(X.ADDUCTS) + 1, 8


class CatEmb(nn.Module):
    """Sum of one embedding per categorical column."""

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
        """-> per-candidate readout [n_cand, d], per-bond embedding [n_bonds + 1, d] (last row = zeros for 'no bond')."""
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
        """v1 inputs per sequence + graph batch: seq_cand [N] (candidate of each sequence), ann_b [N, P, 3] (bond row in the batch,
        -1 none), atom_f [A, N_AF], bond_f [B, N_BF], bond_e [B, 2] (atom rows in the batch), atom_cand [A] -> logits [N]."""
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


def ranges(starts, lens):
    """Concatenation of arange(s, s + l) for each (s, l) (int64)."""
    starts = np.asarray(starts, np.int64); lens = np.asarray(lens, np.int64)
    if lens.sum() == 0:
        return np.zeros(0, np.int64)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    return np.repeat(starts - off, lens) + np.arange(lens.sum())


class Store2:
    """All chunks of a build_groups2 feature directory in RAM (torch CPU tensors) with offsets."""

    def __init__(self, d, limit=None, files=None):
        files = files if files is not None else sorted(glob.glob(os.path.join(d, 'chunk_*.npz')))
        if limit:
            files = files[:limit]
        parts = [dict(np.load(f, allow_pickle=False)) for f in files]
        cat = lambda k: np.concatenate([p[k] for p in parts])
        self.ncand = cat('g_ncand').astype(np.int64); self.nspec = cat('g_nspec').astype(np.int64); self.gid = cat('g_gid')
        self.n = len(self.ncand); self.e180 = np.zeros(self.n, bool)
        self.c0 = np.r_[0, np.cumsum(self.ncand)[:-1]]; self.s0 = np.r_[0, np.cumsum(self.nspec)[:-1]]
        self.a0 = np.r_[0, np.cumsum(self.ncand * self.nspec)[:-1]]
        self.lab = cat('cand_lab').astype(np.int64); self.src = cat('cand_src'); self.smiles = cat('cand_smiles')
        self.na = cat('cand_na').astype(np.int64); self.nb = cat('cand_nb').astype(np.int64)
        self.ao = np.r_[0, np.cumsum(self.na)[:-1]]; self.bo = np.r_[0, np.cumsum(self.nb)[:-1]]
        T = lambda k, dt=None: torch.from_numpy(cat(k) if dt is None else cat(k).astype(dt))
        self.cg = T('cand_g', np.float32); self.spf = T('spec_f'); self.mask = T('spec_mask'); self.meta = T('spec_meta')
        self.ce = T('spec_ce', np.float32); self.ai = T('ann_i'); self.af = T('ann_f'); self.ab = T('ann_b')
        self.atom_f = T('atom_f'); self.bond_f = T('bond_f'); self.bond_e = torch.from_numpy(cat('bond_e').astype(np.int64))
        assert self.ai.shape[0] == int((self.ncand * self.nspec).sum()) and self.cg.shape[0] == int(self.ncand.sum())
        assert self.atom_f.shape[0] == int(self.na.sum()) and self.bond_e.shape[0] == int(self.nb.sum())

    def seq_index(self, groups, spec_pick):
        cr, sr, ar, own = [], [], [], []
        cmax = int(self.ncand[groups].max())
        for b, (g, sp) in enumerate(zip(groups, spec_pick)):
            nc, ns = int(self.ncand[g]), int(self.nspec[g])
            c = np.repeat(np.arange(nc), len(sp)); s = np.tile(sp, nc)
            cr.append(self.c0[g] + c); sr.append(self.s0[g] + s); ar.append(self.a0[g] + c * ns + s); own.append(b * cmax + c)
        return np.concatenate(cr), np.concatenate(sr), np.concatenate(ar), np.concatenate(own), cmax

    def gather(self, groups, spec_pick, device):
        cr, sr, ar, own, cmax = self.seq_index(groups, spec_pick)
        ucand = ranges(self.c0[groups], self.ncand[groups])                    # every candidate of the batch, batch order
        pos = np.full(int(ucand.max()) + 1, -1, np.int64); pos[ucand] = np.arange(len(ucand))
        na, nb = self.na[ucand], self.nb[ucand]
        arow = ranges(self.ao[ucand], na); brow = ranges(self.bo[ucand], nb)
        aoff = np.r_[0, np.cumsum(na)[:-1]]; boff = np.r_[0, np.cumsum(nb)[:-1]]
        atom_cand = np.repeat(np.arange(len(ucand)), na)
        bond_e = self.bond_e[brow] + torch.from_numpy(np.repeat(aoff, nb))[:, None]
        seq_cand = pos[cr]
        ab = self.ab[ar].long()
        ab = torch.where(ab >= 0, ab + torch.from_numpy(boff[seq_cand])[:, None, None], torch.full_like(ab, -1))
        t = torch.from_numpy
        D = dict(spf=self.spf[sr].float(), mask=self.mask[sr], ai=self.ai[ar], af=self.af[ar].float(), meta=self.meta[sr], ce=self.ce[sr],
                 cg=self.cg[cr], seq_cand=t(seq_cand), ann_b=ab, atom_f=self.atom_f[arow], bond_f=self.bond_f[brow], bond_e=bond_e,
                 atom_cand=t(atom_cand))
        D = {k: v.to(device, non_blocking=True) for k, v in D.items()}
        D['n_cand'] = len(ucand)
        return D, torch.from_numpy(own).to(device), cmax


def group_scores(logits, own, B, cmax):
    s = torch.zeros(B * cmax, device=logits.device, dtype=logits.dtype).index_add_(0, own, logits)
    n = torch.zeros(B * cmax, device=logits.device, dtype=logits.dtype).index_add_(0, own, torch.ones_like(logits))
    out = torch.where(n > 0, s / n.clamp(min=1), torch.full_like(s, float('-inf')))
    return out.view(B, cmax)


def isomer_rr(scores, lab):
    pos = scores[lab == 1].max(); neg = scores[lab != 1]
    return 1.0 / (1.0 + (neg > pos).sum() + 0.5 * (neg == pos).sum())
