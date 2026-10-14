"""Subformula annotation of MS/MS peaks: every peak gets the best chemical subformula of the precursor ION formula within a mass tolerance.

Licence: ours. Written from the published idea (peaks as subformulae of the precursor formula, e.g. MIST, Goldman et al.
2023) without looking at any implementation; no third-party code except NumPy and clean/chemcore.py (ours).

Convention. The ion formula is n x M plus / minus the adduct's groups ('[2M+Na]+' of C6H12O6 -> C12H24O12Na). A singly
charged fragment is a subformula F of the ion formula with m/z = mass(F) - sign x m_e; the hydrogen count is free, so
radical ions and hydrogen rearrangements need no special case. The neutral loss is ion formula - F. A candidate is kept
when it has a heavy atom and neither it nor its loss carries more hydrogens than 2C + 3N + 3P + 4; among the candidates
within max(ppm, da) the smallest absolute mass error wins. Spectra whose formula holds an element outside ELEMS, whose
adduct does not parse, or whose heavy-atom grid exceeds `cap` combinations are not annotated (ok = False).

Cache (build_cache): for the spectrum cache of clean/specdata.py, aligned with spec_mz.npy,
  ann_cnt.npy (n_peaks, 12) uint8   element counts of the best subformula (0 everywhere = not annotated)
  ann_nc.npy  (n_peaks,)    uint8   number of candidates within the tolerance (0 = not annotated; clipped at 255)
  ann_err.npy (n_peaks,)    int8    mass error of the best candidate in quarter ppm (clipped at +-127)
  ann_ion.npy (n_spec, 13)  int16   ion formula counts and ok flag (last column)
usage: python clean/subform.py --work <spectrum cache> --out <dir> [--limit N] [--workers 6]
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chemcore  # noqa: E402

ELEMS = ('C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Na', 'K')
NE = len(ELEMS)
_IX = {e: i for i, e in enumerate(ELEMS)}
_H = 1
_HEAVY = [i for i in range(NE) if i != _H]
MASS = np.array([chemcore.mono(e) for e in ELEMS])
PPM, DA, CAP = 15.0, 0.004, 3_000_000
_TOK = re.compile(r'([A-Z][a-z]?)(\d*)')
_ADD = re.compile(r'^\[(\d*)M((?:[+-]\d*[A-Za-z][A-Za-z0-9]*)*)\](\d*)([+-])$')
_GRP = re.compile(r'([+-])(\d*)([A-Za-z][A-Za-z0-9]*)')


def parse_counts(formula):
    """'C6H12O6' (a trailing charge such as '+' or '-2' is ignored) -> int64 (12,) counts, or None for a foreign element."""
    c = np.zeros(NE, np.int64)
    f = re.sub(r'[+-]\d*$', '', formula or '')
    if not f:
        return None
    for el, k in _TOK.findall(f):
        if el not in _IX:
            return None
        c[_IX[el]] += int(k) if k else 1
    return c


def ion_counts(formula, adduct):
    """(neutral formula, adduct string) -> (ion formula counts (12,), sign) or None. Negative counts are clipped at 0."""
    m = _ADD.match(adduct or ''); c = parse_counts(formula)
    if m is None or c is None:
        return None
    c = c * (int(m.group(1)) if m.group(1) else 1)
    for sg, k, f in _GRP.findall(m.group(2)):
        g = parse_counts(f)
        if g is None:
            return None
        c = c + (1 if sg == '+' else -1) * (int(k) if k else 1) * g
    return np.maximum(c, 0), (1 if m.group(4) == '+' else -1)


class Annotator:
    """Annotates spectra; keeps the sorted heavy-atom mass grids of the last few ion formulas."""

    def __init__(self, ppm=PPM, da=DA, cap=CAP, keep=6):
        self.ppm, self.da, self.cap, self.keep, self._grids = ppm, da, cap, keep, {}

    def _grid(self, c):
        key = tuple(int(c[i]) for i in _HEAVY)
        g = self._grids.get(key)
        if g is None:
            rad = np.array(key, np.int64) + 1
            if float(np.prod(rad.astype(np.float64))) > self.cap:
                g = None
            else:
                hm = np.zeros(1)
                for r, i in zip(rad, _HEAVY):                 # mixed radix, the last element least significant
                    if r > 1:
                        hm = (hm[:, None] + np.arange(r) * MASS[i]).ravel()
                order = np.argsort(hm, kind='stable')
                g = (hm[order], order, rad)
            if len(self._grids) >= self.keep:
                self._grids.pop(next(iter(self._grids)))
            self._grids[key] = (g,)
            return g
        return g[0]

    def annotate(self, mz, c, sign):
        """m/z (P,), ion counts (12,), sign -> (counts (P, 12) uint8, n candidates (P,) int64, error ppm (P,) float32, ok)."""
        mz = np.asarray(mz, np.float64); P = len(mz)
        cnt = np.zeros((P, NE), np.uint8); nc = np.zeros(P, np.int64); err = np.zeros(P, np.float32)
        g = self._grid(c)
        if g is None:
            return cnt, nc, err, False
        if P == 0:
            return cnt, nc, err, True
        hs, order, rad = g; cH = int(c[_H])
        hgrid = np.arange(cH + 1)
        t = (mz + sign * chemcore.ELECTRON)[:, None] - hgrid[None, :] * MASS[_H]          # heavy-atom mass wanted
        tol = np.maximum(mz * self.ppm * 1e-6, self.da)[:, None]
        lo = np.searchsorted(hs, t - tol, 'left'); n = np.searchsorted(hs, t + tol, 'right') - lo
        pp, hh = np.nonzero(n > 0)
        if not len(pp):
            return cnt, nc, err, True
        k = n[pp, hh]; tot = int(k.sum())
        rp = np.repeat(pp, k); rh = np.repeat(hh, k)
        j = np.repeat(lo[pp, hh], k) + (np.arange(tot) - np.repeat(np.cumsum(k) - k, k))
        e = (hs[j] - t[rp, rh]) / mz[rp] * 1e6
        code = order[j]; hv = np.empty((tot, len(_HEAVY)), np.int64)
        for q in range(len(_HEAVY) - 1, -1, -1):
            code, hv[:, q] = np.divmod(code, rad[q])
        full = np.empty((tot, NE), np.int64); full[:, _HEAVY] = hv; full[:, _H] = rh
        loss = c[None, :] - full
        hmax = lambda x: 2 * x[:, 0] + 3 * x[:, 2] + 3 * x[:, 4] + 4
        ok = (hv.sum(1) > 0) & (full[:, _H] <= hmax(full)) & (loss[:, _H] <= hmax(loss))
        rp, e, full = rp[ok], e[ok], full[ok]
        if not len(rp):
            return cnt, nc, err, True
        nc = np.bincount(rp, minlength=P)
        o = np.lexsort((np.abs(e), rp)); first = o[np.r_[True, np.diff(rp[o]) > 0]]
        cnt[rp[first]] = np.minimum(full[first], 255).astype(np.uint8); err[rp[first]] = e[first]
        return cnt, nc, err, True


def _work(args):
    """Worker: annotate the spectra `idx` (sorted so that equal ion formulas are neighbours) -> arrays in the order of idx."""
    work, idx, forms, adducts, ppm, da = args
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    off = np.load(os.path.join(work, 'spec_off.npy')); MZ = np.load(os.path.join(work, 'spec_mz.npy'), mmap_mode='r')
    A = Annotator(ppm, da); cs, ns, es, ions = [], [], [], np.zeros((len(idx), NE + 1), np.int16)
    for r, i in enumerate(idx):
        mz = MZ[off[i]:off[i + 1]]; ic = ion_counts(forms[r], adducts[r])
        if ic is None:
            cs.append(np.zeros((len(mz), NE), np.uint8)); ns.append(np.zeros(len(mz), np.uint8)); es.append(np.zeros(len(mz), np.int8)); continue
        c, n, e, ok = A.annotate(mz, ic[0], ic[1])
        ions[r, :NE] = np.minimum(ic[0], 32767); ions[r, NE] = int(ok)
        cs.append(c); ns.append(np.minimum(n, 255).astype(np.uint8)); es.append(np.clip(np.rint(e * 4), -127, 127).astype(np.int8))
    return idx, np.concatenate(cs), np.concatenate(ns), np.concatenate(es), ions


def build_cache(work, out, formulas, adducts, idx=None, workers=6, chunk=4000, ppm=PPM, da=DA, log=print):
    """Annotate the spectra `idx` (default: all) of the spectrum cache `work` with `formulas[i]` / `adducts[i]` (arrays over
    ALL spectra) and write the four tables into `out`. Fixed-size pool of below-normal-priority workers."""
    from multiprocessing import Pool
    os.makedirs(out, exist_ok=True)
    off = np.load(os.path.join(work, 'spec_off.npy')); nspec = len(off) - 1
    idx = np.arange(nspec) if idx is None else np.asarray(idx)
    full = len(idx) == nspec
    shape = (int(off[-1]), NE) if full else None
    key = np.array([f'{formulas[i]}|{adducts[i]}' for i in idx], dtype=object)
    idx = idx[np.argsort(key, kind='stable')]                    # equal ion formulas next to each other: the grid is reused
    tasks = [(work, idx[s:s + chunk], [formulas[i] for i in idx[s:s + chunk]], [adducts[i] for i in idx[s:s + chunk]], ppm, da)
             for s in range(0, len(idx), chunk)]
    if full:
        CNT = np.lib.format.open_memmap(os.path.join(out, 'ann_cnt.npy'), 'w+', np.uint8, shape)
        NC = np.lib.format.open_memmap(os.path.join(out, 'ann_nc.npy'), 'w+', np.uint8, (shape[0],))
        ER = np.lib.format.open_memmap(os.path.join(out, 'ann_err.npy'), 'w+', np.int8, (shape[0],))
        ION = np.zeros((nspec, NE + 1), np.int16); pos = None
    else:                                                        # a subset: compact tables with their own offsets
        lens = (off[idx + 1] - off[idx]).astype(np.int64); sub_off = np.r_[0, np.cumsum(lens)]
        CNT = np.zeros((int(sub_off[-1]), NE), np.uint8); NC = np.zeros(int(sub_off[-1]), np.uint8); ER = np.zeros(int(sub_off[-1]), np.int8)
        ION = np.zeros((len(idx), NE + 1), np.int16); pos = {int(i): r for r, i in enumerate(idx)}
    t0 = time.time(); done = 0
    with Pool(workers) as pool:
        for ii, c, n, e, ions in pool.imap_unordered(_work, tasks):
            p = 0
            for r, i in enumerate(ii):
                L = int(off[i + 1] - off[i])
                if full:
                    a = int(off[i]); ION[i] = ions[r]
                else:
                    q = pos[int(i)]; a = int(sub_off[q]); ION[q] = ions[r]
                CNT[a:a + L] = c[p:p + L]; NC[a:a + L] = n[p:p + L]; ER[a:a + L] = e[p:p + L]; p += L
            done += len(ii)
            if (done // chunk) % 25 == 0:
                log(f'subform: {done:,} / {len(idx):,} spectra, {time.time() - t0:.0f} s')
    if full:
        CNT.flush(); NC.flush(); ER.flush(); np.save(os.path.join(out, 'ann_ion.npy'), ION)
    else:
        np.save(os.path.join(out, 'ann_cnt.npy'), CNT); np.save(os.path.join(out, 'ann_nc.npy'), NC); np.save(os.path.join(out, 'ann_err.npy'), ER)
        np.save(os.path.join(out, 'ann_ion.npy'), ION); np.save(os.path.join(out, 'ann_idx.npy'), idx); np.save(os.path.join(out, 'ann_off.npy'), sub_off)
    log(f'subform: done, {len(idx):,} spectra in {time.time() - t0:.0f} s with {workers} workers')
    return time.time() - t0


def true_formulas(work):
    """(formula per spectrum, adduct per spectrum) of the spectrum cache: the structure's formula ('' without a structure)."""
    import pandas as pd
    m = pd.read_parquet(os.path.join(work, 'spec_meta.parquet'), columns=['adduct', 'sid'])
    st = pd.read_parquet(os.path.join(work, 'train_structs.parquet'), columns=['sid', 'formula'])
    f = np.array([''] + list(st.formula.values), dtype=object)[np.where(m.sid.values >= 0, m.sid.values + 1, 0)]
    return f, m.adduct.values


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='annotate a random sample of this many spectra (a compact cache)')
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--ppm', type=float, default=PPM); ap.add_argument('--da', type=float, default=DA)
    a = ap.parse_args()
    f, ad = true_formulas(a.work)
    idx = None
    if a.limit:
        idx = np.sort(np.random.default_rng(0).choice(len(f), a.limit, replace=False))
    build_cache(a.work, a.out, f, ad, idx, a.workers, ppm=a.ppm, da=a.da, log=lambda *x: print(time.strftime('%H:%M:%S'), *x, flush=True))
