"""Popularity of PubChem-tier rows for the grafts JOIN_POP_MU and PC_PROMOTE (reviews/graft-2026-10-10.md); used only when one of them is on.

The table is the one the public line reads (dmitriigluzdov/casmi26-pubchem-popularity-prior: pc_lsid.npy = log1p of the PubChem substance count,
pc_lpmid.npy = log1p of the PubMed count, row-aligned with the PubChem tier of casmi26-pubchem-tier); popularity = their sum, the quantity POP_MU
uses for pool rows (pool_lsid + pool_lpmid). A joined row and a row of the PubChem channel carry the tier's SMILES text unchanged, so a row is found
again by its neutral-mass window and its text. No network, no model.
"""
import os

import numpy as np


class TierPop:
    def __init__(self, pc_dir, pop_dir):
        self.mass = np.load(os.path.join(pc_dir, 'pc_mass.npy'), mmap_mode='r')
        self.off = np.load(os.path.join(pc_dir, 'pc_off.npy'), mmap_mode='r')
        self.buf = np.load(os.path.join(pc_dir, 'pc_smiles.npy'), mmap_mode='r')
        self.ls = np.load(os.path.join(pop_dir, 'pc_lsid.npy'), mmap_mode='r')
        self.lp = np.load(os.path.join(pop_dir, 'pc_lpmid.npy'), mmap_mode='r')
        assert len(self.ls) == len(self.mass) == len(self.lp), 'popularity arrays are not row-aligned with the PubChem tier'

    def pops(self, smiles, target, ppm=12.0):
        """{smiles: popularity} for the rows of the +-ppm window around `target` whose SMILES text is in `smiles` (max over duplicates)."""
        out = {}
        if target is None or not np.isfinite(target) or not smiles:
            return out
        tol = float(target) * ppm * 1e-6
        lo = int(np.searchsorted(self.mass, target - tol, 'left')); hi = int(np.searchsorted(self.mass, target + tol, 'right'))
        if hi <= lo:
            return out
        off = np.asarray(self.off[lo:hi + 1]).astype(np.int64); raw = bytes(self.buf[off[0]:off[-1]]); b0 = int(off[0])
        want = {s.encode('ascii', 'ignore') for s in smiles}
        for i in range(hi - lo):
            x = raw[off[i] - b0:off[i + 1] - b0]
            if x in want:
                s = x.decode('ascii'); p = float(self.ls[lo + i]) + float(self.lp[lo + i])
                if p > out.get(s, -1.0):
                    out[s] = p
        return out

    def pops_wide(self, smiles, target):
        """pops with a second, wider look (40 ppm) for rows the 12 ppm window does not hold (mass-offset targets)."""
        out = self.pops(smiles, target)
        miss = [s for s in smiles if s not in out]
        if miss:
            out.update(self.pops(miss, target, ppm=40.0))
        return out


def join_pop_table(base, targets, tier):
    """({molecule: {smiles: popularity}}, stats) for the joined PubChem rows (pool id -2) of every base list."""
    out = {}; st = dict(molecules=0, rows=0, found=0)
    for m, v in base.items():
        if len(v) < 6:
            continue
        js = [s for s, p in zip(v[0], v[5]) if p == -2]
        if not js:
            continue
        d = tier.pops_wide(js, targets.get(m))
        out[m] = d; st['molecules'] += 1; st['rows'] += len(js); st['found'] += len(d)
    return out, st


def prep_pc_pop(pc_pop, tier):
    """The popularity-ordered channel's lists with the two quantities of the public rule: S = the first listed score (z of f.z over the scored
    shortlist + 0.25 x popularity, as the worker returns it) and top_pop = the popularity of that first row. Returns (lists, stats)."""
    out = {}; st = dict(lists=0, first_found=0, s_gt6_pop5=0)
    for m, p in (pc_pop or {}).items():
        if not p or not p.get('pc'):
            continue
        q = dict(p); q['S'] = float(p['pc_fz'][0])
        d = tier.pops_wide([p['pc'][0]], p.get('target'))
        q['top_pop'] = float(d.get(p['pc'][0], 0.0))
        out[m] = q; st['lists'] += 1; st['first_found'] += int(p['pc'][0] in d); st['s_gt6_pop5'] += int(q['S'] > 6.0 and q['top_pop'] >= 5.0)
    return out, st
