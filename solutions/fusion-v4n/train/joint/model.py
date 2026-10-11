"""Reference joint model: a peak-set transformer for spectra, a fingerprint MLP or a small GNN for structures, InfoNCE with isomer negatives.

SpecEncoder  one token per peak (Fourier features of m/z and of the neutral loss precursor - m/z, the intensity), a leading token from the
             metadata (precursor m/z, adduct, polarity, collision energy with a known flag, instrument family); transformer encoder; the
             leading token of every spectrum is averaged over the spectra of a molecule, then projected and L2-normalised.
FpEncoder    packed fingerprint bits (the pool's 10,226 selected bits) -> MLP -> L2-normalised embedding.
GraphEncoder message passing over the molecular graph with index_add_ (graphs.py; no torch-geometric) -> mean and sum readout -> embedding.
info_nce     every molecule against the positives of the batch and its own K negatives (masking in-batch duplicates of its structure),
             plus the structure-to-spectrum direction over the batch; a learnable temperature.
"""
import math

import torch, torch.nn as nn, torch.nn.functional as F

import graphs as G


def fourier(x, n_freq, lo, hi):
    """Sinusoidal features of a scalar: wavelengths log-spaced from `lo` to `hi`. x [...] -> [..., 2 * n_freq]."""
    w = 2 * math.pi / torch.logspace(math.log10(lo), math.log10(hi), n_freq, device=x.device, dtype=torch.float32)
    a = x.float().unsqueeze(-1) * w
    return torch.cat([a.sin(), a.cos()], -1)


class SpecEncoder(nn.Module):
    """Options (defaults = the phase-0 reference): mz_embed 'fourier' | 'bin' (learned 0.1 Da bins), use_nl (neutral-loss features),
    int_tf 'both' | 'sqrt' | 'raw', use_meta (adduct / polarity / collision energy / instrument; the precursor always stays),
    pool 'mean' | 'attn' | 'max' over the spectra of a molecule."""

    def __init__(self, d=256, layers=6, heads=8, ff=None, out=256, n_freq=32, n_adduct=64, n_instr=8, dropout=0.1, mz_embed='fourier', use_nl=True, int_tf='both', use_meta=True, pool='mean', aug=0.0):
        super().__init__()
        self.aug = aug; self.n_freq = n_freq; self.mz_embed = mz_embed; self.use_nl = use_nl; self.int_tf = int_tf; self.use_meta = use_meta; self.pool = pool; ff = ff or 4 * d
        if mz_embed == 'bin':
            self.mzb = nn.Embedding(20001, d); self.nlb = nn.Embedding(20001, d); self.peak = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        else:
            self.peak = nn.Sequential(nn.Linear(4 * n_freq + 2, d), nn.GELU(), nn.Linear(d, d))
        self.adduct = nn.Embedding(n_adduct, d); self.mode = nn.Embedding(2, d); self.instr = nn.Embedding(n_instr, d)
        self.meta = nn.Sequential(nn.Linear(2 * n_freq + 2 * 8 + 1, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, out))
        if pool == 'attn':
            self.att = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))

    def spectra(self, b):
        """Per-spectrum vectors [T, d] (before pooling over the spectra of a molecule)."""
        mz, it = b['mz'], b['it']; T, Pk = mz.shape; prec = b['prec']
        pad = torch.arange(Pk, device=mz.device)[None, :] >= b['n_peaks'][:, None]
        if self.training and self.aug > 0:                                        # augmentation: drop peaks (never the first), jitter intensities
            drop = torch.rand_like(mz) < self.aug; drop[:, 0] = False; pad = pad | drop; it = (it * torch.exp(0.3 * torch.randn_like(it))).clamp(max=1.0)
        nl = (prec[:, None] - mz).clamp(min=0)
        iv = {'both': (it.sqrt(), torch.log10(it.clamp(min=1e-4)) / 4), 'sqrt': (it.sqrt(), torch.zeros_like(it)), 'raw': (it, torch.zeros_like(it))}[self.int_tf]
        if self.mz_embed == 'bin':
            tok = self.mzb((mz * 10).round().long().clamp(0, 20000)) + self.peak(torch.stack(iv, -1).to(self.peak[0].weight.dtype))
            if self.use_nl:
                tok = tok + self.nlb((nl * 10).round().long().clamp(0, 20000))
        else:
            fn = fourier(nl, self.n_freq, 0.01, 2000.0)
            x = torch.cat([fourier(mz, self.n_freq, 0.01, 2000.0), fn if self.use_nl else torch.zeros_like(fn), iv[0][..., None], iv[1][..., None]], -1)
            tok = self.peak(x.to(self.peak[0].weight.dtype))
        ck = b['ce_known'] if self.use_meta else torch.zeros_like(b['ce_known'])
        m = torch.cat([fourier(prec, self.n_freq, 0.01, 2000.0), fourier(b['ce'], 8, 1.0, 400.0) * ck[:, None], ck[:, None]], -1)
        cls = self.meta(m.to(tok.dtype))
        if self.use_meta:
            cls = cls + self.adduct(b['add']) + self.mode(b['mode']) + self.instr(b['instr'])
        h = self.enc(torch.cat([cls[:, None], tok.to(cls.dtype)], 1), src_key_padding_mask=torch.cat([torch.zeros(T, 1, dtype=torch.bool, device=mz.device), pad], 1))
        return self.norm(h[:, 0])

    def pool_spectra(self, h, own, n):
        """Per-spectrum vectors [T, d] -> one vector per molecule [n, d] (mean, attention or max over its spectra)."""
        h = h.float()
        if self.pool == 'max':
            return torch.zeros(n, h.shape[1], device=h.device).index_reduce_(0, own, h, 'amax', include_self=False)
        if self.pool == 'attn':
            a = self.att(h).squeeze(-1).float(); mx = torch.full((n,), -1e9, device=h.device).index_reduce_(0, own, a.detach(), 'amax', include_self=False)
            w = (a - mx[own]).exp()
        else:
            w = torch.ones(len(h), device=h.device)
        s = torch.zeros(n, h.shape[1], device=h.device).index_add_(0, own, h * w[:, None]); c = torch.zeros(n, device=h.device).index_add_(0, own, w).clamp(min=1e-6)
        return s / c[:, None]

    def pooled(self, b, n_mol=None):
        return self.pool_spectra(self.spectra(b), b['owner'], int(n_mol if n_mol is not None else b['n_mol']))

    def embed(self, pooled):
        return F.normalize(self.proj(pooled.to(self.proj[0].weight.dtype)).float(), dim=-1)

    def forward(self, b, n_mol=None):
        """One L2-normalised embedding per molecule."""
        return self.embed(self.pooled(b, n_mol))


def unpack_bits(packed, nbits):
    """uint8 [n, nbytes] -> float [n, nbits] (numpy.packbits order: most significant bit first)."""
    sh = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    return ((packed[:, :, None] >> sh) & 1).reshape(packed.shape[0], -1)[:, :nbits]


class FpEncoder(nn.Module):
    def __init__(self, nbits=10226, hidden=(1024, 512), out=256, dropout=0.1, in_drop=0.0):
        super().__init__()
        self.in_drop = in_drop; self.nbits = nbits; dims = (nbits,) + tuple(hidden); layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], out))

    def forward(self, packed):
        x = unpack_bits(packed, self.nbits).to(self.net[0].weight.dtype)
        if self.training and self.in_drop > 0:
            x = F.dropout(x, self.in_drop)
        return F.normalize(self.net(x).float(), dim=-1)


class GraphEncoder(nn.Module):
    def __init__(self, d=256, layers=5, out=256, dropout=0.1):
        super().__init__()
        self.atom = nn.ModuleList([nn.Embedding(n, d) for n in G.A_SIZES]); self.bond = nn.ModuleList([nn.ModuleList([nn.Embedding(n, d) for n in G.B_SIZES]) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)]); self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, out))

    def forward(self, b):
        a = b['g_atoms']; h = sum(e(a[:, i]) for i, e in enumerate(self.atom)); src, dst = b['g_edges']; bo = b['g_bonds']
        for l in range(len(self.mlp)):
            msg = F.relu(h[src] + sum(e(bo[:, i]) for i, e in enumerate(self.bond[l])))
            agg = torch.zeros_like(h).index_add_(0, dst, msg.to(h.dtype))
            h = h + self.drop(self.mlp[l](self.norm[l](h + agg)))
        n = int(b['g_n']); gid = b['g_id']
        s = torch.zeros(n, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, gid, h)
        c = torch.zeros(n, device=h.device, dtype=h.dtype).index_add_(0, gid, torch.ones_like(gid, dtype=h.dtype)).clamp(min=1)
        return F.normalize(self.out(torch.cat([s / c[:, None], s / 16.0], -1)).float(), dim=-1)


class Joint(nn.Module):
    """loss 'nce' (in-batch positives + own negatives, plus the reverse direction with weight w_m2s) | 'list' (softmax over the molecule's own
    candidate list only) | 'margin' (batch-all triplet hinge on cosines). w_aux > 0 adds a head predicting the fingerprint bits from the
    pooled spectrum vector (BCE + 1 - cosine), the public recipe's objective, as a multi-task term."""

    def __init__(self, variant='fp', nbits=10226, d=256, layers=6, out=256, n_adduct=64, n_instr=8, temp=0.07, hidden=(1024, 512), loss='nce', w_m2s=0.25, margin=0.2, fix_temp=False,
                 w_aux=0.0, fp_drop=0.0, **spec_kw):
        super().__init__()
        self.variant = variant; self.spec = SpecEncoder(d=d, layers=layers, out=out, n_adduct=n_adduct, n_instr=n_instr, **spec_kw)
        self.struct = FpEncoder(nbits, hidden=tuple(hidden), out=out, in_drop=fp_drop, dropout=spec_kw.get('dropout', 0.1)) if variant == 'fp' else GraphEncoder(out=out)
        self.loss = loss; self.w_m2s = w_m2s; self.margin = margin; self.w_aux = w_aux; self.nbits = nbits
        ls = torch.tensor(math.log(1.0 / temp))
        if fix_temp:
            self.register_buffer('logit_scale', ls)
        else:
            self.logit_scale = nn.Parameter(ls)
        if w_aux > 0:
            self.aux = nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, nbits))

    def encode_structs(self, b):
        """[B * (1 + K), out]: the positives first (B rows), then the negatives molecule by molecule."""
        if self.variant == 'fp':
            return self.struct(torch.cat([b['pos_fp'], b['neg_fp'].reshape(-1, b['neg_fp'].shape[-1])], 0))
        return self.struct(b)

    def forward(self, b):
        hp = self.spec.pooled(b); zs = self.spec.embed(hp); B = zs.shape[0]; ze = self.encode_structs(b); K = b['neg_kind'].shape[1]
        loss, acc, n = joint_loss(zs, ze[:B], ze[B:].reshape(B, K, -1), b['pid'], b['neg_kind'] >= 0, self.logit_scale.exp().clamp(max=100.0), self.loss, self.w_m2s, self.margin)
        if self.w_aux > 0:
            y = unpack_bits(b['pos_fp'], self.nbits).float(); p = self.aux(hp.to(self.aux[0].weight.dtype)).float()
            loss = loss + self.w_aux * (10.0 * F.binary_cross_entropy_with_logits(p, y) + (1 - F.cosine_similarity(p.sigmoid(), y, dim=-1)).mean())
        return loss, acc, n


def joint_loss(zs, zp, zn, pid, neg_ok, scale, kind='nce', w_m2s=0.25, margin=0.2):
    """zs spectra [B, d], zp positives [B, d], zn negatives [B, K, d], pid structure id per molecule, neg_ok [B, K].
    Returns (loss, top-1 accuracy of the spectrum-to-structure direction, number of candidates per row)."""
    B = zs.shape[0]; eye = torch.eye(B, dtype=torch.bool, device=zs.device); same = (pid[:, None] == pid[None, :]) & ~eye
    cin = zs @ zp.T; cown = torch.einsum('bd,bkd->bk', zs, zn); y = torch.arange(B, device=zs.device)
    if kind == 'margin':
        neg = torch.cat([cin, cown], 1); ok = torch.cat([~(same | eye), neg_ok], 1); v = F.relu(margin - cin.diag()[:, None] + neg) * ok
        loss = v.sum() / (v > 0).sum().clamp(min=1); acc = (neg.masked_fill(~ok, -2.0).max(1).values < cin.diag()).float().mean()
        return loss, acc, neg.shape[1]
    inb = (cin * scale).masked_fill(same, float('-inf')); own = (cown * scale).masked_fill(~neg_ok, float('-inf'))
    if kind == 'list':
        logits = torch.cat([inb.diag()[:, None], own], 1); t = torch.zeros(B, dtype=torch.long, device=zs.device)
        return F.cross_entropy(logits, t), (logits.argmax(1) == 0).float().mean(), logits.shape[1]
    logits = torch.cat([inb, own], 1)
    l_s2m = F.cross_entropy(logits, y); l_m2s = F.cross_entropy(inb.T, y)
    acc = (logits.argmax(1) == y).float().mean()
    return (1 - w_m2s) * l_s2m + w_m2s * l_m2s, acc, logits.shape[1]


def info_nce(zs, zp, zn, pid, neg_ok, scale):
    """The phase-0 reference loss (kept for the tests)."""
    return joint_loss(zs, zp, zn, pid, neg_ok, scale, 'nce', 0.25)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
