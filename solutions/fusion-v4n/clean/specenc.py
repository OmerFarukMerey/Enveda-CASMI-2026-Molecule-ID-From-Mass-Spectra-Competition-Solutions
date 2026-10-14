"""Spectrum encoder with configurable peak tokens (m/z, subformula, pairwise mass differences, multi-spectrum, intensity rank): model, items, collate, data access.

Licence: ours. A generalisation of clean/fpnet.py (ours): with the token set 'mz' and nothing else switched on, the
parameter names, shapes and outputs are those of fpnet.FPNet, so a reference checkpoint loads and gives the same logits
(tests/test_specenc.py). Subformula annotation comes from clean/subform.py (ours). Nothing third-party is imported
except NumPy and PyTorch.

Token sets (cfg['tok']):  'mz'   [sin/cos(m/z), sin/cos(precursor - m/z), intensity]                (the reference)
                          'form' [fragment formula, neutral-loss formula, intensity]                (no mass in the token)
                          'both' [sin/cos(m/z), sin/cos(precursor - m/z), fragment formula, neutral-loss formula, intensity]
Switches: cfg['pair']  adds, per peak, which of LOSSES links it to a peak above / below it or to the precursor;
          cfg['multi'] keeps the spectra of a query apart (own peaks, collision energy, adduct, precursor and slot per peak);
          cfg['rank']  feeds the intensity rank instead of the square-root intensity;
          cfg['dreams'] adds one token made from a frozen external spectrum embedding.
With a formula token set the global token also gets the neutral formula; has_form = 0 hides every formula input.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fpnet  # noqa: E402
import specdata  # noqa: E402
import subform  # noqa: E402

FORMAT = 'fusion-v4n/clean/specenc v1'
NE = subform.NE
# largest count with its own embedding row, per element of subform.ELEMS (C H N O P S F Cl Br I Na K); larger counts share the last row
CMAX = (80, 160, 20, 40, 8, 8, 20, 10, 8, 8, 3, 3)
# mass differences (Da) with a chemical meaning, fixed before any run (scratch/specenc/PREREG.md, run c_pair)
LOSSES = {'H2': 2.015650, 'CH2': 14.015650, 'CH3': 15.023475, 'O': 15.994915, 'CH4': 16.031300, 'NH3': 17.026549, 'H2O': 18.010565,
          'HF': 20.006228, 'C2H2': 26.015650, 'HCN': 27.010899, 'CO': 27.994915, 'C2H4': 28.031300, 'CH2O': 30.010565,
          'CH4O': 32.026215, 'H2S': 33.987721, 'Cl': 34.968853, 'HCl': 35.976678, '2H2O': 36.021130, 'C2H2O': 42.010565,
          'C3H6': 42.046950, 'CHNO': 43.005814, 'CO2': 43.989829, 'C2H4O': 44.026215, 'CH2O2': 46.005479, 'C2H6O': 46.041865,
          'C4H8': 56.062600, 'C3H6O': 58.041865, 'C2H4O2': 60.021129, 'SO2': 63.961900, 'C5H8': 68.062600, 'Br': 78.918338,
          'HBr': 79.926160, 'SO3': 79.956815, 'HPO3': 79.966331, 'C3H2O3': 86.000394, 'H3PO4': 97.976896, 'I': 126.904473,
          'C5H8O4': 132.042259, 'C9H6O2': 146.036779, 'C6H10O4': 146.057909, 'C7H4O4': 152.010959, 'C9H6O3': 162.031694,
          'C6H10O5': 162.052824, 'C6H8O6': 176.032088, 'C10H8O3': 176.047344, 'C6H12O6': 180.063388, 'C8H13NO5': 203.079373}
DEFAULT = dict(tok='mz', pair=False, multi=False, rank=False, dreams=0, d=512, layers=8, heads=8, max_peaks=160, multi_peaks=256)


def make_cfg(**kw):
    c = dict(DEFAULT); c.update({k: v for k, v in kw.items() if v is not None})
    assert c['tok'] in ('mz', 'form', 'both')
    return c


class FormEmb(nn.Module):
    """Element counts (..., 12) -> d: one small embedding per (element, count), concatenated, then a linear layer."""

    def __init__(self, d, e=32):
        super().__init__()
        self.register_buffer('cmax', torch.tensor(CMAX)); self.register_buffer('base', torch.tensor(np.r_[0, np.cumsum(np.array(CMAX) + 1)[:-1]]))
        self.emb = nn.Embedding(int(sum(CMAX) + len(CMAX)), e); self.lin = nn.Linear(NE * e, d)

    def forward(self, c):
        c = torch.minimum(c.long().clamp(min=0), self.cmax) + self.base
        return self.lin(self.emb(c).flatten(-2))


class SpecEnc(nn.Module):
    def __init__(self, nbits: int, cfg: dict | None = None, drop: float = 0.1, n_elem: int = fpnet.N_ELEM):
        super().__init__()
        cfg = self.cfg = make_cfg(**(cfg or {})); d = self.d = cfg['d']; self.nbits = nbits; self.n_instr = 0
        self.use_mz = cfg['tok'] in ('mz', 'both'); self.use_form = cfg['tok'] in ('form', 'both'); n = 1
        if self.use_mz:
            self.mz_emb = fpnet.SinEmb(d, 0.01, 10 ** 3.3); self.nl_emb = fpnet.SinEmb(d, 0.01, 10 ** 3.3); n += 2
        if self.use_form:
            self.frag = FormEmb(d); self.loss = FormEmb(d); self.amb = nn.Embedding(6, d); self.mform = FormEmb(d)
            self.noform = nn.Parameter(torch.zeros(d)); n += 2
        if cfg['pair']:
            self.register_buffer('losses', torch.tensor(list(LOSSES.values()), dtype=torch.float32))
            self.pair = nn.Linear(3 * len(LOSSES), d); n += 1
        if cfg['multi']:
            self.slot = nn.Embedding(8, d); self.pad_ad = nn.Embedding(fpnet.N_ADDUCT, d); self.pce = nn.Linear(d // 4 + 1, d)
            for m in (self.slot, self.pad_ad):
                nn.init.normal_(m.weight, std=0.02)
        if cfg['dreams']:
            self.dre = nn.Sequential(nn.Linear(cfg['dreams'], d), nn.GELU(), nn.Linear(d, d))
        self.int_emb = nn.Linear(2, d); self.pk = nn.Linear(n * d, d)
        self.prec_emb = fpnet.SinEmb(d, 0.01, 10 ** 3.3); self.ad = nn.Embedding(fpnet.N_ADDUCT, d)
        self.ce_emb = fpnet.SinEmb(d // 4, 1.0, 10 ** 2.2); self.gl = nn.Linear(d + d // 4 + 4, d)
        self.blocks = nn.ModuleList([fpnet.Block(d, cfg['heads'], drop) for _ in range(cfg['layers'])])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, nbits))
        self.elem_head = nn.Sequential(nn.Linear(2 * d, d // 2), nn.GELU(), nn.Linear(d // 2, n_elem))

    @torch.no_grad()
    def pair_feats(self, mz, it, pad, pprec, slot=None, tol=0.004, tol_prec=0.006):
        """(B, P, 3K): strongest partner peak at m/z + L and at m/z - L, and whether precursor - m/z = L, for L in LOSSES."""
        mz = mz.float(); w = it.float() * (~pad)
        diff = mz[:, None, :] - mz[:, :, None]                             # [b, i, j] = m/z_j - m/z_i
        ok = (~pad)[:, None, :] & (~pad)[:, :, None]
        if slot is not None:
            ok = ok & (slot[:, None, :] == slot[:, :, None])
        up, dn = [], []
        for L in self.losses.tolist():
            m = ((diff - L).abs() < tol) & ok
            up.append((m * w[:, None, :]).amax(2)); dn.append((m.transpose(1, 2) * w[:, None, :]).amax(2))
        pl = (((pprec.float() - mz)[:, :, None] - self.losses).abs() < tol_prec) & (~pad)[:, :, None]
        return torch.cat([torch.stack(up, -1), torch.stack(dn, -1), pl.float()], -1)

    def forward(self, b: dict, return_emb: bool = False):
        """b: the dict of `collate`. -> (bit logits (B, nbits), element values (B, 10))."""
        mz, it, pad = b['mz'], b['it'], b['pad']; prec = b['prec'].float()
        pprec = b['pprec'] if 'pprec' in b else prec[:, None]
        ch = []
        if self.use_mz:
            ch += [self.mz_emb(mz), self.nl_emb((pprec - mz).clamp(min=0))]
        if self.use_form:
            ch += [self.frag(b['fcnt']) + self.amb(b['nc'].long().clamp(max=5)), self.loss(b['lcnt'])]
        if self.cfg['pair']:
            ch.append(self.pair(self.pair_feats(mz, it, pad, pprec, b.get('slot')).to(self.pk.weight.dtype)))
        ch.append(self.int_emb(torch.stack([it, it * it], -1)))
        pk = self.pk(torch.cat([c.to(ch[-1].dtype) for c in ch], -1))
        if self.cfg['multi']:
            pk = pk + self.slot(b['slot']) + self.pad_ad(b['pad_ix']) + self.pce(torch.cat([self.ce_emb(b['pce'] / 10.0 + 1.0), b['pce_known'][..., None]], -1).to(pk.dtype))
        g = torch.cat([self.prec_emb(prec), self.ce_emb(b['ce'] / 10.0 + 1.0),
                       torch.stack([b['ce_known'], b['merged'], b['mode'], torch.log1p(prec) / 10.0], -1)], -1)
        g = self.gl(g) + self.ad(b['ad'])
        if self.use_form:
            hf = b['has_form'][:, None].to(g.dtype)
            g = g + hf * self.mform(b['mcnt']).to(g.dtype) + (1 - hf) * self.noform.to(g.dtype)
        toks = [g[:, None].to(pk.dtype)]; lead = [torch.zeros_like(pad[:, :1])]
        if self.cfg['dreams']:
            toks.append(self.dre(b['demb'].to(pk.dtype))[:, None]); lead.append(torch.zeros_like(pad[:, :1]))
        k = len(toks)
        x = torch.cat(toks + [pk], 1); m = torch.cat(lead + [pad], 1)
        for blk in self.blocks:
            x = blk(x, m)
        x = self.norm(x)
        v = (~pad).unsqueeze(-1).to(x.dtype)
        emb = torch.cat([x[:, 0], (x[:, k:] * v).sum(1) / v.sum(1).clamp(min=1.0)], -1)
        if return_emb:
            return emb
        return self.head(emb), self.elem_head(emb)


class RefSignature(nn.Module):
    """A formula-free, single-sequence SpecEnc ('mz' tokens, optionally pair features) behind the positional interface of
    fpnet.FPNet.forward, so that code written for the reference network (the engine, the notebook) can call it unchanged."""

    def __init__(self, net: SpecEnc):
        super().__init__()
        assert net.cfg['tok'] == 'mz' and not net.cfg['multi'] and not net.cfg['dreams'], 'needs inputs the reference interface does not carry'
        self.net = net; self.nbits, self.d, self.n_instr = net.nbits, net.d, 0

    def forward(self, mz, it, pad, prec, ad, ce, ce_known, merged, mode, return_emb: bool = False, instr=None):
        return self.net(dict(mz=mz, it=it, pad=pad, prec=prec, ad=ad, ce=ce, ce_known=ce_known, merged=merged, mode=mode), return_emb)


# ---------------------------------------------------------------------------------------------------------- items

def merge_idx(mzs, its, tol: float = 0.005):
    """specdata.merge_spectra with the indices of the kept peaks: (m/z, intensity, index into the concatenated inputs)."""
    mz = np.concatenate([np.asarray(m, np.float32) for m in mzs])
    it = np.concatenate([np.asarray(i, np.float32) / max(float(np.max(i)), 1e-12) if len(i) else np.zeros(0, np.float32) for i in its])
    if not len(mz):
        return mz, it, np.zeros(0, np.int64)
    o = np.argsort(mz, kind='stable'); mz, it = mz[o], it[o]
    near = np.diff(mz) <= tol
    if near.any():
        keep = np.ones(len(mz), bool); lm, li = mz.tolist(), it.tolist(); cur = 0
        for j in np.flatnonzero(np.r_[False, near] | np.r_[near, False]).tolist():
            if j == 0 or not near[j - 1]:
                cur = j; continue
            if lm[j] - lm[cur] <= tol:
                if li[j] >= li[cur]:
                    keep[cur] = False; cur = j
                else:
                    keep[j] = False
            else:
                cur = j
        mz, it, o = mz[keep], it[keep], o[keep]
    return mz, it, o


def prep_idx(mz, it, prec_mz, max_peaks: int = fpnet.MAX_PEAKS, floor: float = 0.001):
    """fpnet.prep_peaks with the indices of the kept peaks: (m/z, sqrt intensity, index into the inputs)."""
    mz = np.asarray(mz, np.float32); it = np.asarray(it, np.float32)
    ix = np.flatnonzero((mz > 0) & (mz <= prec_mz + 2.0))
    if not len(ix) or it[ix].max() <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    mx = it[ix].max(); ix = ix[it[ix] >= floor * mx]
    if len(ix) > max_peaks:
        ix = ix[np.argsort(-it[ix])[:max_peaks]]
    ix = ix[np.argsort(mz[ix], kind='stable')]
    return mz[ix], np.sqrt(it[ix] / mx), ix


def rank_feature(s):
    """Square-root intensities -> intensity rank in [0, 1] (1 = the most intense peak of the item)."""
    n = len(s)
    if n <= 1:
        return np.ones(n, np.float32)
    r = np.empty(n, np.float32); r[np.argsort(-s, kind='stable')] = np.arange(n, dtype=np.float32)
    return 1.0 - r / (n - 1)


class AnnFull:
    """Annotation cache aligned with the spectrum cache (subform.build_cache over all spectra): the true formulas."""

    def __init__(self, ann_dir, D):
        self.cnt = np.load(os.path.join(ann_dir, 'ann_cnt.npy'), mmap_mode='r'); self.nc = np.load(os.path.join(ann_dir, 'ann_nc.npy'), mmap_mode='r')
        self.ion = np.load(os.path.join(ann_dir, 'ann_ion.npy')); self.off = D.off
        assert len(self.cnt) == int(D.off[-1]) and len(self.ion) == len(D.off) - 1, 'annotation cache does not match the spectrum cache'
        self._D = D; self._m = None

    def mcounts(self, i):
        if self._m is None:                                       # neutral formula counts per pool row (zeros: foreign element)
            self._m = np.stack([c if c is not None else np.zeros(NE, np.int64) for c in map(subform.parse_counts, self._D.pformula)]).astype(np.int16)
        return self._m[self._D.sid2pid[self._D.sid[i]]]

    def get(self, i):
        a, b = int(self.off[i]), int(self.off[i + 1])
        return self.cnt[a:b], self.nc[a:b], self.ion[i]


class AnnFly:
    """Annotation computed on demand for a formula given per spectrum (pipeline-assigned or decoy formulas), memoised."""

    def __init__(self, D, adducts, formula_of):
        self.D, self.adducts, self.formula_of = D, adducts, formula_of      # formula_of: dict spectrum index -> formula
        self.A = subform.Annotator(); self.memo = {}

    def mcounts(self, i):
        c = subform.parse_counts(self.formula_of.get(int(i), ''))
        return np.zeros(NE, np.int16) if c is None else np.minimum(c, 32767).astype(np.int16)

    def get(self, i):
        i = int(i); f = self.formula_of.get(i, ''); r = self.memo.get((i, f))
        if r is None:
            mz = self.D.peaks(i)[0]; ic = subform.ion_counts(f, self.adducts[i]); ion = np.zeros(NE + 1, np.int16)
            if ic is None:
                r = (np.zeros((len(mz), NE), np.uint8), np.zeros(len(mz), np.uint8), ion)
            else:
                c, n, _, ok = self.A.annotate(mz, ic[0], ic[1]); ion[:NE] = np.minimum(ic[0], 32767); ion[NE] = int(ok)
                r = (c, np.minimum(n, 255).astype(np.uint8), ion)
            self.memo[(i, f)] = r
        return r


def build_item(D, idxs, cfg, ann=None, rng=None, aug=None, hide_form=False, demb=None):
    """Network item of a query (1 to 8 spectra of one structure and polarity). ann: AnnFull / AnnFly (formula token sets).
    rng + aug: training augmentation (specdata.AUG_STD keys jitter, drop, drop_p, ce_hide). demb: external embedding."""
    idxs = [int(i) for i in idxs]; i0 = idxs[0]; aug = aug or {}; use_form = cfg['tok'] != 'mz'
    c = D.ce[idxs]; ck = c[~np.isnan(c)]; ce = ck.mean() if len(ck) else np.nan
    nm = int(sum(max(1, D.ce_n[j]) for j in idxs))
    if rng is not None and aug.get('ce_hide', 0) > 0 and rng.random() < aug['ce_hide']:
        ce = np.nan; c = np.full(len(idxs), np.nan)
    known = not np.isnan(ce)
    it0 = dict(prec=float(D.prec[i0]), adduct_ix=int(D.adduct[i0]), ce=float(ce) if known else 0.0, ce_known=1.0 if known else 0.0,
               n_merged=int(min(max(nm, 1), 8)), mode=float(D.mode[i0]))
    parts = [D.peaks(j) for j in idxs]
    lens = np.array([len(p[0]) for p in parts]); spec_of = np.repeat(np.arange(len(idxs)), lens)

    def augment(mz, it, ix):
        if rng is None:
            return mz, it, ix
        it = np.array(it, np.float32)
        if aug.get('jitter', 0) > 0:
            it = it * np.exp(rng.normal(0.0, aug['jitter'], len(it))).astype(np.float32)
        if aug.get('drop', 0) > 0 and len(it) > 1 and rng.random() < aug.get('drop_p', 0.5):
            keep = rng.random(len(it)) >= rng.uniform(0.0, aug['drop'])
            if keep.any():
                mz, it, ix = mz[keep], it[keep], ix[keep]
        return mz, it, ix

    if not cfg['multi']:
        if len(idxs) == 1:
            mz, it = np.asarray(parts[0][0]), np.asarray(parts[0][1]); src = np.arange(len(mz))
        else:
            mz, it, src = merge_idx([p[0] for p in parts], [p[1] for p in parts])
        mz, it, src = augment(mz, it, src)
        mz, s, k = prep_idx(mz, it, it0['prec'], cfg['max_peaks']); src = src[k]
    else:
        per = max(8, cfg['multi_peaks'] // len(idxs)); ms, ss, srcs, start = [], [], [], 0     # the budget holds for any number of spectra
        for q, (pm, pi) in enumerate(parts):
            m1, i1, x1 = augment(np.asarray(pm), np.asarray(pi), np.arange(len(pm)))
            m1, s1, k = prep_idx(m1, i1, float(D.prec[idxs[q]]), per)
            ms.append(m1); ss.append(s1); srcs.append(start + x1[k]); start += len(pm)
        mz, s, src = np.concatenate(ms), np.concatenate(ss), np.concatenate(srcs)
        sp = spec_of[src] if len(src) else np.zeros(0, np.int64); ia = np.asarray(idxs)
        pc = c[sp] if len(sp) else np.zeros(0)
        it0.update(pprec=D.prec[ia[sp]].astype(np.float32), pce=np.nan_to_num(pc, nan=0.0).astype(np.float32),
                   pce_known=(~np.isnan(pc)).astype(np.float32), pad_ix=D.adduct[ia[sp]].astype(np.int64), slot=np.minimum(sp, 7).astype(np.int64))
    it0.update(mz=mz, it=rank_feature(s) if cfg['rank'] else s)
    if use_form:
        P = len(mz); has = not hide_form
        if has and P:
            got = [ann.get(j) for j in idxs]
            cnt = np.concatenate([g[0] for g in got])[src]; nc = np.concatenate([g[1] for g in got])[src]
            ion = np.stack([g[2] for g in got])[spec_of[src]]
            loss = np.clip(ion[:, :NE].astype(np.int32) - cnt, 0, 255) * (nc > 0)[:, None]
            it0.update(fcnt=np.asarray(cnt, np.uint8), lcnt=loss.astype(np.uint8), nc=np.minimum(nc, 5).astype(np.uint8))
        else:
            it0.update(fcnt=np.zeros((P, NE), np.uint8), lcnt=np.zeros((P, NE), np.uint8), nc=np.zeros(P, np.uint8))
        it0.update(mcnt=ann.mcounts(i0) if has else np.zeros(NE, np.int16), has_form=1.0 if has else 0.0)
    if demb is not None:
        it0['demb'] = demb
    return it0


def collate(items, cfg, device='cpu'):
    """Items -> dict of tensors, padded to the longest item of the batch. An empty spectrum is one zero peak (as fpnet.collate)."""
    B = len(items); P = max(1, max(len(x['mz']) for x in items))
    out = dict(mz=np.zeros((B, P), np.float32), it=np.zeros((B, P), np.float32), pad=np.ones((B, P), bool))
    per_peak = {'pprec': np.float32, 'pce': np.float32, 'pce_known': np.float32, 'pad_ix': np.int64, 'slot': np.int64, 'nc': np.uint8}
    for k, dt in per_peak.items():
        if k in items[0]:
            out[k] = np.zeros((B, P), dt)
    for k in ('fcnt', 'lcnt'):
        if k in items[0]:
            out[k] = np.zeros((B, P, NE), np.uint8)
    for i, x in enumerate(items):
        n = len(x['mz']); out['mz'][i, :n] = x['mz']; out['it'][i, :n] = x['it']; out['pad'][i, :n] = False
        if n == 0:
            out['pad'][i, 0] = False
        for k in out:
            if k not in ('mz', 'it', 'pad') and n:
                out[k][i, :n] = x[k]
    f = lambda k, dt=np.float32: np.array([x[k] for x in items], dt)
    out.update(prec=f('prec'), ad=f('adduct_ix', np.int64), ce=f('ce'), ce_known=f('ce_known'), merged=f('n_merged') / 4.0, mode=f('mode'))
    if 'mcnt' in items[0]:
        out.update(mcnt=np.stack([x['mcnt'] for x in items]).astype(np.int16), has_form=f('has_form'))
    if 'demb' in items[0]:
        out['demb'] = np.stack([x['demb'] for x in items]).astype(np.float32)
    return {k: torch.from_numpy(v).to(device) for k, v in out.items()}


def take(b: dict, rows, L=None):
    """Rows of a batch dict, peak axis cut to L."""
    o = {}
    for k, v in b.items():
        v = v[rows]
        o[k] = v[:, :L] if (L is not None and v.ndim >= 2 and k not in ('mcnt', 'demb')) else v
    return o


def forward_bucketed(model, b: dict, nb: int):
    """The forward pass in nb chunks of similar peak count, each cut to its own longest item (outputs do not depend on padding)."""
    if nb <= 1:
        return model(b)
    n = (~b['pad']).sum(1); order = torch.argsort(n); zs, es = [], []
    for c in order.chunk(nb):
        z, e = model(take(b, c, max(1, int(n[c].max())))); zs.append(z); es.append(e)
    inv = torch.empty_like(order); inv[order] = torch.arange(len(order), device=order.device)
    return torch.cat(zs)[inv], torch.cat(es)[inv]


def save_ckpt(path, model: SpecEnc, step: int, extra: dict | None = None, half: bool = False):
    """Checkpoint with the reference field names plus `enc` (the token configuration). half: float16 weights (kept curve points)."""
    par = {k for k, _ in model.named_parameters()}                # buffers (the sinusoid frequencies) stay float32
    sd = {k: (v.detach().cpu().half() if half and k in par else v.detach().cpu()) for k, v in model.state_dict().items()}
    ck = dict(step=int(step), nbits=model.nbits, d=model.d, layers=len(model.blocks), heads=model.blocks[0].h, format=FORMAT, enc=dict(model.cfg), model=sd)
    ck.update(extra or {})
    torch.save(ck, path + '.tmp'); os.replace(path + '.tmp', path)


def load_ckpt(path, device='cpu') -> SpecEnc:
    """Our checkpoints, and reference-layout checkpoints (fpnet.FPNet / the public file) as the token set 'mz'."""
    ck = torch.load(path, map_location='cpu', weights_only=True)
    cfg = ck.get('enc') or dict(tok='mz', d=ck['d'], layers=ck['layers'], heads=ck.get('heads', 8))
    m = SpecEnc(ck['nbits'], cfg)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck['model'].items()})
    return m.to(device).eval()


class EncData(specdata.SpecData):
    """SpecData plus the annotation cache and the adduct strings (needed to annotate under another formula)."""

    def __init__(self, work, pool=None, holdout=('hold', 'np'), ann_dir=None, K=63, ppm=10.0, log=print, need_fp=True, need_adducts=False):
        super().__init__(work, pool, holdout, K, ppm, log, need_fp)
        self.ann = AnnFull(ann_dir, self) if ann_dir else None
        self.adducts = None
        if need_adducts:
            import pandas as pd
            self.adducts = pd.read_parquet(os.path.join(work, 'spec_meta.parquet'), columns=['adduct']).adduct.values

    def group_of(self, i, rng, merge_p):
        """Training query of spectrum i: itself, or with probability merge_p 2 to 8 training spectra of its group."""
        g = self.g_of[i]; members = self.grp_order[self.gs[g]:self.ge[g]]
        if len(members) > 1 and rng.random() < merge_p:
            members = members[self.in_tr[members]]
            if len(members) > 1:
                k = int(rng.integers(2, min(8, len(members)) + 1)); others = members[members != i]
                return [i] + rng.choice(others, size=min(k - 1, len(others)), replace=False).tolist()
        return [i]

    def enc_batch(self, rng, bs, cfg, merge_p=0.3, aug=None, parts=None, form_drop=0.0, decoy=None, decoy_p=0.0, demb=None):
        """decoy: AnnFull of a second cache built with a WRONG window formula per spectrum; used with probability decoy_p."""
        ids = self.sample_ids(rng, bs, parts); items = []
        for i in ids:
            idxs = self.group_of(int(i), rng, merge_p)
            ann = decoy if (decoy is not None and rng.random() < decoy_p) else self.ann
            hide = form_drop > 0 and rng.random() < form_drop
            e = None if demb is None else demb[idxs].mean(0)
            items.append(build_item(self, idxs, cfg, ann, rng, specdata.AUG_STD if aug is None else aug, hide, e))
        pos = self.sid2pid[self.sid[ids]]
        return collate(items, cfg), pos, self.negatives(pos, rng), ids


class EncStream(specdata._Iterable):
    """Endless stream of training batches for DataLoader worker processes (each opens the tables itself, memory-mapped)."""

    def __init__(self, data_kw, cfg, seed, start_step, bs, merge_p, aug, parts_text, form_drop=0.0, decoy_dir=None, decoy_p=0.0, demb_path=None):
        self.data_kw, self.cfg, self.seed, self.start_step, self.bs = data_kw, cfg, seed, start_step, bs
        self.merge_p, self.aug, self.parts_text, self.form_drop = merge_p, aug, parts_text, form_drop
        self.decoy_dir, self.decoy_p, self.demb_path = decoy_dir, decoy_p, demb_path

    def __iter__(self):
        from torch.utils.data import get_worker_info
        w = get_worker_info()
        return self.stream(w.id if w is not None else 0)

    def stream(self, worker_id=0):
        D = EncData(log=lambda *a: None, need_fp=False, **self.data_kw)
        decoy = None
        if self.decoy_dir:
            decoy = AnnFull(self.decoy_dir, D)
            decoy._m = np.load(os.path.join(self.decoy_dir, 'decoy_mcnt.npy')); decoy.mcounts = lambda i, d=decoy, D=D: d._m[i]
        demb = np.load(self.demb_path, mmap_mode='r') if self.demb_path else None
        rng = np.random.default_rng([self.seed, worker_id, self.start_step]); parts = specdata.parse_parts(D, self.parts_text)
        while True:
            yield D.enc_batch(rng, self.bs, self.cfg, self.merge_p, self.aug, parts, self.form_drop, decoy, self.decoy_p, demb)
