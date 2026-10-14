"""Adapter: a molecule's spectra (v1-engine in-memory dicts or test.parquet rows) -> fingerprint logits of a clean/specenc.py network (multi-spectrum or reference) and per-candidate scores.

Licence: ours. Standalone: imports only clean/specenc.py, clean/specdata.py, clean/chemcore.py (ours), NumPy and PyTorch; it does not
touch the engine or the notebook. Written for the msint study (reviews/msint-2026-10-14.md).

Exactness with training (clean/train_specenc.py). The network item is built by `specenc.build_item` itself, on a stand-in object that
offers the five fields and the one method build_item reads from `specdata.SpecData` (ce, ce_n, prec, adduct, mode; peaks(i)). The fields
are filled the way `specdata.build_cache` fills the training cache:
  peaks    chemcore.clean_peaks(raw m/z, raw intensity, precursor m/z)  (0 < m/z <= precursor + 2, floor 0.001, top 512, base peak 1)
  ce       mean of the finite collision energies (eV), NaN when there is none;  ce_n = number of finite values
  adduct   specdata.ADDUCT_IX (59 = any other adduct);  mode = -1 for negative ionisation or an adduct ending in '-', else +1
Everything after that (per-spectrum peak budget, sqrt intensities, pairwise-difference features, slots, collision energy and adduct per
peak) is the trained model's own code. No augmentation at inference (rng None), as in the study's hold-out evaluation.

Grouping (a choice; the network was trained on query groups of ONE structure and ONE polarity, 1 to 8 spectra, first spectrum = global
token): spectra are split by polarity; a polarity with more than 8 spectra is cut into ceil(n / 8) near-equal chunks in input order.
Molecule logits = mean over the polarities present of (spectrum-count-weighted mean over that polarity's chunks): the convention of the
engine's merged view (casmi/engine.py:347-362: one merged item per polarity, logits averaged).
Scores: f.z = bits . z (the training score; the engine's fz uses the same product, engine.py:364-366) and cos = the engine's fp_cos
(engine.py:367-370: cosine of sigmoid(z) with the bits).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import chemcore  # noqa: E402
import specdata  # noqa: E402

MAX_GROUP = 8                        # the largest query group the network saw in training (specdata.group_of / slot embedding of 8)


def _finite_ce(ce):
    """Collision energy input (scalar, list, array or None) -> (mean of the finite values or NaN, count of finite values)."""
    if ce is None:
        return np.nan, 0
    a = np.atleast_1d(np.asarray(ce, dtype=np.float64)); a = a[np.isfinite(a)]
    return (float(a.mean()), int(len(a))) if len(a) else (np.nan, 0)


def _mode(mode, adduct):
    if isinstance(mode, str):
        neg = mode.lower().startswith('neg')
    else:
        neg = mode is not None and float(mode) < 0
    return -1 if (neg or str(adduct).endswith('-')) else 1


class MolSpectra:
    """The spectra of one molecule in the shape `specenc.build_item` reads (a stand-in for specdata.SpecData)."""

    def __init__(self, spectra, clean: bool = True):
        """spectra: dicts with mz, it, prec, adduct, mode (+1 / -1 or 'positive' / 'negative'), and the collision energy as either
        `ce_list` (raw values, preferred: the training cache's rule) or `ce` (+ optional `ce_n`, the engine's dict).
        clean=False: the peaks are already cleaned cache spectra (base peak 1)."""
        mz, it, self.prec, self.adduct, self.mode, self.ce, self.ce_n = [], [], [], [], [], [], []
        for s in spectra:
            p = float(s['prec'])
            m, i = (chemcore.clean_peaks(s['mz'], s['it'], p) if clean else (np.asarray(s['mz'], np.float32), np.asarray(s['it'], np.float32)))
            mz.append(m); it.append(i); self.prec.append(p)
            self.adduct.append(specdata.ADDUCT_IX.get(s['adduct'], specdata.ADDUCT_OTHER)); self.mode.append(_mode(s.get('mode'), s['adduct']))
            if 'ce_list' in s:
                c, n = _finite_ce(s['ce_list'])
            else:
                c, n = _finite_ce(s.get('ce')); n = int(s['ce_n']) if (n and s.get('ce_n') is not None) else n
            self.ce.append(c); self.ce_n.append(n)
        self._mz, self._it = mz, it
        self.prec = np.asarray(self.prec, np.float64); self.adduct = np.asarray(self.adduct, np.int64); self.mode = np.asarray(self.mode, np.int64)
        self.ce = np.asarray(self.ce, np.float64); self.ce_n = np.asarray(self.ce_n, np.int64)

    def __len__(self):
        return len(self._mz)

    def peaks(self, i):
        return self._mz[i], self._it[i]


def groups(modes, max_group: int = MAX_GROUP):
    """[(polarity, [spectrum indices])]: per polarity (positive first) in input order, cut into ceil(n / max_group) near-equal chunks."""
    modes = np.asarray(modes); out = []
    for md in (1, -1):
        ix = np.flatnonzero(modes == md)
        if len(ix):
            out += [(md, c.tolist()) for c in np.array_split(ix, math.ceil(len(ix) / max_group))]
    return out


def from_rows(sub):
    """test.parquet / train.parquet rows of one molecule (a DataFrame) -> spectrum dicts with the raw collision-energy list."""
    return [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64), prec=float(r.precursor_mz),
                 adduct=r.adduct, mode=r.ionization_mode, ce_list=r.collision_energy_ev) for r in sub.itertuples()]


class SpecFP:
    """A clean/specenc.py checkpoint behind a per-molecule interface."""

    def __init__(self, ckpt: str, device: str = 'cpu', threads: int | None = None, amp: bool = False):
        import torch
        import specenc
        if threads:
            torch.set_num_threads(int(threads))
        self.torch, self.specenc = torch, specenc
        self.model = specenc.load_ckpt(ckpt, device); self.cfg = self.model.cfg; self.device = device; self.nbits = self.model.nbits
        self.amp = bool(amp and device.startswith('cuda'))
        assert self.cfg['tok'] == 'mz' and not self.cfg['dreams'], 'formula token sets and DreaMS tokens need inputs this adapter does not build'

    def items(self, mol: MolSpectra):
        """Network items of one molecule and their (polarity, number of spectra)."""
        out, meta = [], []
        for md, idx in groups(mol.mode):
            out.append(self.specenc.build_item(mol, idx, self.cfg)); meta.append((md, len(idx)))
        return out, meta

    def logits(self, mols, bs: int = 64, return_groups: bool = False):
        """mols: list of MolSpectra (or of spectrum-dict lists) -> (n_mol, nbits) float32 molecule logits (zeros for a molecule without
        spectra). return_groups: also the per-item logits with (molecule, polarity, n spectra)."""
        torch = self.torch
        mols = [m if isinstance(m, MolSpectra) else MolSpectra(m) for m in mols]
        items, owner = [], []
        for k, m in enumerate(mols):
            it, meta = self.items(m); items += it; owner += [(k, md, n) for md, n in meta]
        Z = np.zeros((len(items), self.nbits), np.float32)
        with torch.no_grad():
            for s in range(0, len(items), bs):
                b = self.specenc.collate(items[s:s + bs], self.cfg, self.device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
                    z = self.model(b)[0]
                Z[s:s + bs] = z.float().cpu().numpy()
        out = np.zeros((len(mols), self.nbits), np.float32)
        for k in range(len(mols)):
            pol = {}
            for r, (kk, md, n) in enumerate(owner):
                if kk == k:
                    a = pol.setdefault(md, [np.zeros(self.nbits, np.float64), 0]); a[0] += n * Z[r]; a[1] += n
            if pol:
                out[k] = np.mean([v[0] / v[1] for v in pol.values()], axis=0)
        return (out, Z, owner) if return_groups else out


def unpack(packed, nbits: int) -> np.ndarray:
    """Packed fingerprints (n, ceil(nbits / 8)) uint8 -> (n, nbits) float32 0/1 (np.packbits order, as the pool tables)."""
    return np.unpackbits(np.asarray(packed, np.uint8), axis=1, count=nbits).astype(np.float32)


def scores(z, bits) -> dict:
    """Per-candidate scores of one molecule: fz = bits . z and the engine's fp_cos. bits: (n, nbits) 0/1 (dense) array."""
    bits = np.asarray(bits, np.float32); z = np.asarray(z, np.float64)
    pz = np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), 1e-6, 1 - 1e-6)
    cb = bits.sum(1)
    return dict(fz=(bits @ z).astype(np.float32), cos=((bits @ pz) / (np.sqrt(cb) * np.linalg.norm(pz) + 1e-9)).astype(np.float32))
