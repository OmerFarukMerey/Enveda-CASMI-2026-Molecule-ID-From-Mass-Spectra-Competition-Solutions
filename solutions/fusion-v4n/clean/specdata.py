"""Our own spectrum data access for the fingerprint network: cache builder, loader, leak check, merging, sampling, batches.

Licence: ours. Written from clean/SPEC.md (sections 1, 3, 6), from the file formats, and from the Apache-2.0 fold-safe
training loop (dmitriigluzdov/casmi26-fold-safe-fpnet, train_fpnet.py: batch composition "half spectrum-uniform, half
structure-and-polarity-uniform", K = 63 negatives of the +-10 ppm pool window, query groups of at most 8 spectra). The
unlicensed `casmi` package files (traindata.py, spectra.py, build.py) were not opened for this file. Observed, not read:
the behaviour of the reference merge on toy inputs (0.005 Da, the more intense peak's m/z, maximum intensity), the
element table (log1p of the counts of C H N O P S F Cl Br I) and the adduct -> index table of the existing cache; these
three are interface facts the current engine relies on at inference, so our network is trained with the same ones.

Decided by us (the specification is silent): the augmentation (intensity jitter, random peak dropout, hiding the
collision energy), how a merge draws its spectra (2 to 8 of the structure-and-polarity group, uniformly), how a window
with fewer than K other rows is filled (nearest pool rows by mass outside the window), the instrument substring rules,
and that hold-out structures stay in the pool as negatives (as catalogue-only rows do: the hidden test's answers are
negatives-only rows for every model as well).

Tables read (a directory `work`): spec_off.npy, spec_mz.npy, spec_it.npy, spec_meta.parquet (SPEC 3; `build_cache`
writes them from train.parquet), pool_meta.parquet, pool_fp.npy, fp_bits.npy, train_structs.parquet (SPEC 1;
clean/pool_build.py writes them), split.parquet (sid, fold: the published identity-disjoint split).
"""
from __future__ import annotations

import os
import re

import numpy as np

try:
    from torch.utils.data import IterableDataset as _Iterable
except Exception:                                             # the data-free parts work without torch
    _Iterable = object

LIBS = ('enveda-180', 'pluskal_ms2', 'riken', 'gnps', 'massbank', 'mona', 'spectraverse', 'msdial', 'drug_plus',
        'enveda-np-examples', 'masaryk')
ENVEDA_LIBS = (0, 9)
ELEMENTS = ('C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I')
# adduct -> input index of the network (index 59 = any other adduct). The first ten are the test's adducts (SPEC 2); the
# others are the indices found in the existing spectrum cache (an interface table: the engine feeds these indices).
ADDUCT_IX = {'[M+H]+': 0, '[M+NH4]+': 1, '[M-H2O+H]+': 2, '[M-2H2O+H]+': 3, '[M+Na]+': 4, '[M+K]+': 5, '[M-H]-': 6,
             '[M-H2O-H]-': 7, '[M+CH2O2-H]-': 8, '[M+Cl]-': 9, '[M+Li]+': 10, '[M-H2O]+': 11, '[M]+': 12, '[M+2H]2+': 13,
             '[M+3H]3+': 14, '[M+HNa]2+': 16, '[M+2Na-H]+': 18, '[M+Ca]2+': 19, '[M+C2H4N]+': 23, '[M+CH5O]+': 24,
             '[M-H2N]+': 25, '[M-H5O3]+': 27, '[M-2H]-': 30, '[M-H2]2-': 32, '[M]-': 33, '[M+C2H4O2-H]-': 35, '[M+Br]-': 37,
             '[M+Na-H2]-': 40, '[M+K-H2]-': 41, '[M-CH3]-': 42, '[M-C3H6NO2]-': 43, '[M+C2F3O2]-': 44, '[2M+H]+': 46,
             '[2M+Na]+': 47, '[2M+K]+': 48, '[2M+NH4]+': 49, '[2M+Ca]2+': 50, '[2M-H]-': 51, '[2M+CH2O2-H]-': 52,
             '[2M+C2H4O2-H]-': 53, '[2M+Na-2H]-': 54, '[3M+Na]+': 57, '[3M-H]-': 58}
ADDUCT_OTHER = 59
_FTOK = re.compile(r'([A-Z][a-z]?)(\d*)')


def instrument_class(name) -> int:
    """0 timsTOF, 1 Orbitrap / FT, 2 other TOF, 3 ion trap, 4 triple quadrupole, 5 other (substring rules, in this order)."""
    s = str(name).lower()
    if 'timstof' in s or 'tims' in s:
        return 0
    if any(k in s for k in ('orbitrap', 'exactive', 'ft', 'lumos', 'fusion', 'velos', 'elite', 'exploris', 'astral')):
        return 1
    if 'tof' in s:
        return 2
    if any(k in s for k in ('ion trap', 'iontrap', '-it', 'it-', 'trap', 'lcq', 'ltq')):
        return 3
    if any(k in s for k in ('qqq', 'triple', 'tq', 'qq')):
        return 4
    return 5


def element_targets(formulas) -> np.ndarray:
    """(n, 10) float32: log1p of the counts of ELEMENTS in each formula string."""
    ix = {e: i for i, e in enumerate(ELEMENTS)}
    out = np.zeros((len(formulas), len(ELEMENTS)), np.float32)
    for r, f in enumerate(formulas):
        for el, k in _FTOK.findall(f or ''):
            if el in ix:
                out[r, ix[el]] += int(k) if k else 1
    return np.log1p(out)


def merge_spectra(pairs, tol: float = 0.005, max_peaks: int | None = None):
    """Join spectra of one structure and polarity: each scaled to base peak 1, peaks pooled and sorted by m/z, then one
    pass in ascending m/z: a peak within `tol` Da of the current merged peak's m/z joins it (the larger intensity is kept,
    and the m/z moves to the joining peak when that one is at least as intense), otherwise it starts a new merged peak.
    -> (m/z, intensity) float32, ascending m/z."""
    mz = np.concatenate([np.asarray(p[0], np.float32) for p in pairs])
    it = np.concatenate([np.asarray(p[1], np.float32) / max(float(np.max(p[1])), 1e-12) if len(p[1]) else np.zeros(0, np.float32) for p in pairs])
    if not len(mz):
        return mz, it
    o = np.argsort(mz, kind='stable'); mz, it = mz[o], it[o]
    near = np.diff(mz) <= tol                                  # a larger gap to the previous peak always starts a new one
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
        mz, it = mz[keep], it[keep]
    if max_peaks is not None and len(mz) > max_peaks:
        k = np.sort(np.argsort(-it, kind='stable')[:max_peaks]); mz, it = mz[k], it[k]
    return mz, it


def build_cache(train_parquet, structs_parquet, out_dir, row_groups=None, log=print):
    """Spectrum cache of train.parquet (SPEC 3) with our own cleaning (clean/chemcore.py). row_groups: subset, for checks."""
    import pandas as pd
    import pyarrow.parquet as pq
    import chemcore
    st = pd.read_parquet(structs_parquet, columns=['inchikey14', 'sid'])
    sid_of = dict(zip(st.inchikey14.values, st.sid.values)); lib_of = {n: i for i, n in enumerate(LIBS)}
    cols = ['ingest_lib', 'inchikey14', 'ionization_mode', 'instrument_type', 'adduct', 'precursor_mz', 'precursor_error_ppm',
            'ms2_mzs', 'ms2_normalized_intensities', 'num_peaks', 'base_peak_intensity', 'collision_energy_ev', 'collision_energy_orig_units']
    f = pq.ParquetFile(train_parquet); mzs, its, lens, metas = [], [], [], []
    for g in (range(f.metadata.num_row_groups) if row_groups is None else row_groups):
        t = f.read_row_group(g, columns=cols).to_pandas(); rows = []
        for r in t.itertuples(index=False):
            mz, it = chemcore.clean_peaks(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
            mzs.append(mz); its.append(it); lens.append(len(mz))
            ce = np.asarray(r.collision_energy_ev if r.collision_energy_ev is not None else [], np.float64); ce = ce[~np.isnan(ce)]
            neg = str(r.ionization_mode).lower().startswith('neg') or str(r.adduct).endswith('-')
            rows.append((r.adduct, r.precursor_mz, r.precursor_error_ppm, r.collision_energy_orig_units, r.base_peak_intensity, r.num_peaks,
                         ce.mean() if len(ce) else np.nan, ce.min() if len(ce) else np.nan, ce.max() if len(ce) else np.nan, len(ce),
                         sid_of.get(r.inchikey14, -1), lib_of.get(r.ingest_lib, -1), instrument_class(r.instrument_type),
                         ADDUCT_IX.get(r.adduct, ADDUCT_OTHER), -1 if neg else 1, chemcore.neutral_mass(r.precursor_mz, r.adduct), len(mz)))
        metas.append(pd.DataFrame(rows, columns=['adduct', 'precursor_mz', 'precursor_error_ppm', 'collision_energy_orig_units', 'base_peak_intensity',
                                                 'num_peaks', 'ce_mean', 'ce_min', 'ce_max', 'ce_n', 'sid', 'lib', 'instr', 'adduct_ix', 'mode', 'nm', 'n_clean']))
        log(f'build_cache: row group {g} done, {sum(lens):,} peaks')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'spec_off.npy'), np.r_[0, np.cumsum(lens)].astype(np.int64))
    np.save(os.path.join(out_dir, 'spec_mz.npy'), np.concatenate(mzs)); np.save(os.path.join(out_dir, 'spec_it.npy'), np.concatenate(its))
    pd.concat(metas, ignore_index=True).to_parquet(os.path.join(out_dir, 'spec_meta.parquet'), index=False)


class SpecData:
    """Spectrum cache + pool tables + split in memory (spectra memory-mapped), with the samplers of the trainer."""

    def __init__(self, work, pool=None, holdout=('hold', 'np'), K=63, ppm=10.0, log=print, need_fp=True):
        import pandas as pd
        pool = pool or work
        self.K, self.ppm, self.holdout = K, ppm, tuple(holdout)
        self.off = np.load(os.path.join(work, 'spec_off.npy')); self.mz = np.load(os.path.join(work, 'spec_mz.npy'), mmap_mode='r')
        self.it = np.load(os.path.join(work, 'spec_it.npy'), mmap_mode='r')
        m = pd.read_parquet(os.path.join(work, 'spec_meta.parquet'), columns=['precursor_mz', 'ce_mean', 'ce_n', 'sid', 'lib', 'instr', 'adduct_ix', 'mode', 'n_clean'])
        self.sid = m.sid.values.astype(np.int64); self.adduct = m.adduct_ix.values.astype(np.int64); self.mode = m['mode'].values.astype(np.int64)
        self.prec = m.precursor_mz.values.astype(np.float64); self.ce = m.ce_mean.values.astype(np.float64); self.ce_n = m.ce_n.values.astype(np.int64)
        self.lib = m.lib.values.astype(np.int64); self.instr = m.instr.values.astype(np.int64); self.n_clean = m.n_clean.values.astype(np.int64)
        pm = pd.read_parquet(os.path.join(pool, 'pool_meta.parquet'), columns=['key', 'formula', 'mass', 'src', 'train_sid'])
        self.pmass = pm.mass.values.astype(np.float64); self.psrc = pm.src.values.astype(np.int64); self.pkey = pm.key.values
        self.pformula = pm.formula.values
        assert np.all(np.diff(self.pmass) >= 0), 'pool_meta must be sorted by mass'
        st = pd.read_parquet(os.path.join(pool, 'train_structs.parquet'), columns=['sid', 'inchikey14', 'key', 'rawkey'])
        assert np.array_equal(st.sid.values, np.arange(len(st)))
        key2pid = dict(zip(self.pkey, np.arange(len(pm))))           # by metric key: tautomer twins share one pool row
        self.sid2pid = np.array([key2pid.get(k, -1) for k in st.key.values], np.int64)
        sp = pd.read_parquet(os.path.join(work, 'split.parquet'), columns=['sid', 'fold'])
        self.fold = np.array(['train'] * len(st), dtype=object); self.fold[sp.sid.values] = sp.fold.values
        self.nbits = int(len(np.load(os.path.join(pool, 'fp_bits.npy'))))
        self.pfp = np.load(os.path.join(pool, 'pool_fp.npy'), mmap_mode='r') if need_fp else None
        # ---- training spectra and the leak check: no hold-out identity among the training positives
        held = pd.Series(self.fold).isin(self.holdout).values
        ids = lambda mask: set(st.key.values[mask]) | set(st.rawkey.values[mask]) | set(st.inchikey14.values[mask])
        hold_ids = ids(held)
        twin = ~held & (st.key.isin(hold_ids) | st.rawkey.isin(hold_ids) | st.inchikey14.isin(hold_ids)).values
        self.n_twin = int(twin.sum())
        ok_sid = ~held & ~twin & (self.sid2pid >= 0)
        s = np.where(self.sid >= 0, self.sid, 0)
        self.tr = np.flatnonzero((self.sid >= 0) & ok_sid[s] & (self.n_clean > 0))
        pos_sids = np.unique(self.sid[self.tr])
        pos_mask = np.zeros(len(st), bool); pos_mask[pos_sids] = True
        self.leak = len(ids(pos_mask) & hold_ids)
        self.leak_pid = len(set(self.sid2pid[pos_sids].tolist()) & set(self.sid2pid[held & (self.sid2pid >= 0)].tolist()))
        self.leak_msg = (f'LEAK CHECK: {int(held.sum()):,} held-out structures {self.holdout}, {len(pos_sids):,} training structures, '
                         f'{len(self.tr):,} training spectra; identifiers (metric key, raw first block, inchikey14) shared between the two: '
                         f'{self.leak}; shared pool rows: {self.leak_pid}; training structures dropped as identity twins of a held-out one: {self.n_twin}')
        log(self.leak_msg)
        assert self.leak == 0 and self.leak_pid == 0, 'a held-out identity is among the training positives'
        # ---- (structure, polarity) groups over all spectra with a structure
        key = np.where(self.sid >= 0, self.sid * 2 + (self.mode > 0), -1)
        order = np.argsort(key, kind='stable'); order = order[key[order] >= 0]; ks = key[order]
        b = np.flatnonzero(np.diff(ks)) + 1
        self.grp_order = order; self.gs = np.r_[0, b]; self.ge = np.r_[b, len(ks)]; self.gk = ks[self.gs]
        self.g_of = np.full(len(self.sid), -1, np.int64); self.g_of[order] = np.repeat(np.arange(len(self.gs)), self.ge - self.gs)
        self.tr_groups = np.unique(self.g_of[self.tr])
        in_tr = np.zeros(len(self.sid), bool); in_tr[self.tr] = True; self.in_tr = in_tr
        self._elem = None

    @property
    def elem(self):
        if self._elem is None:
            self._elem = element_targets(self.pformula)
        return self._elem

    def peaks(self, i):
        return self.mz[self.off[i]:self.off[i + 1]], self.it[self.off[i]:self.off[i + 1]]

    def query(self, idxs):
        """Network item of a query group (1 to 8 spectra, merged when more than one); metadata of the first spectrum."""
        import fpnet
        i = idxs[0]
        if len(idxs) == 1:
            mz, it = self.peaks(i); ce = self.ce[i]; nm = max(1, self.ce_n[i])
        else:
            mz, it = merge_spectra([self.peaks(j) for j in idxs])
            c = self.ce[list(idxs)]; c = c[~np.isnan(c)]; ce = c.mean() if len(c) else np.nan
            nm = int(sum(max(1, self.ce_n[j]) for j in idxs))
        return fpnet.make_item(mz, it, self.prec[i], self.adduct[i], ce, nm, self.mode[i])

    def train_item(self, i, rng, merge_p=0.3, aug=None):
        """Training item of spectrum i: with probability merge_p a merge of 2 to 8 training spectra of its group, then the
        augmentation `aug` (dict; see AUG_STD and the trainer's --aug)."""
        import fpnet
        aug = AUG_STD if aug is None else aug
        g = self.g_of[i]; members = self.grp_order[self.gs[g]:self.ge[g]]
        idxs = [i]
        if len(members) > 1 and rng.random() < merge_p:
            members = members[self.in_tr[members]]
            if len(members) > 1:
                k = int(rng.integers(2, min(8, len(members)) + 1))
                others = members[members != i]
                idxs = [i] + rng.choice(others, size=min(k - 1, len(others)), replace=False).tolist()
        if len(idxs) == 1:
            mz, it = self.peaks(i); ce = self.ce[i]; nm = max(1, self.ce_n[i])
        else:
            mz, it = merge_spectra([self.peaks(j) for j in idxs])
            c = self.ce[idxs]; c = c[~np.isnan(c)]; ce = c.mean() if len(c) else np.nan
            nm = int(sum(max(1, self.ce_n[j]) for j in idxs))
        mz = np.asarray(mz); it = np.array(it, np.float32)
        if aug.get('jitter', 0) > 0:
            it = it * np.exp(rng.normal(0.0, aug['jitter'], len(it))).astype(np.float32)
        if aug.get('warp', 0) > 0 and rng.random() < aug.get('warp_p', 0.5):          # intensity warping: it ** gamma
            it = np.power(np.maximum(it, 1e-9), np.exp(rng.normal(0.0, aug['warp']))).astype(np.float32)
        if aug.get('drop', 0) > 0 and len(it) > 1 and rng.random() < aug.get('drop_p', 0.5):
            keep = rng.random(len(it)) >= rng.uniform(0.0, aug['drop'])
            if keep.any():
                mz, it = mz[keep], it[keep]
        if aug.get('topn') is not None and rng.random() < aug.get('topn_p', 0.5) and len(it) > 1:   # thin to a target peak-count profile
            n = int(rng.choice(aug['topn']))
            if n < len(it):
                k = np.sort(np.argsort(-it, kind='stable')[:max(n, 1)]); mz, it = mz[k], it[k]
        if aug.get('ce_hide', 0) > 0 and rng.random() < aug['ce_hide']:
            ce = np.nan
        return fpnet.make_item(mz, it, self.prec[i], self.adduct[i], ce, nm, self.mode[i])

    def restrict(self, idx):
        """Train on a subset of the training spectra only (fine-tuning on one source)."""
        self.tr = np.asarray(idx); self.in_tr = np.zeros(len(self.sid), bool); self.in_tr[self.tr] = True
        self.tr_groups = np.unique(self.g_of[self.tr])

    def window(self, pid):
        m = self.pmass[pid]; t = m * self.ppm * 1e-6
        return int(np.searchsorted(self.pmass, m - t, 'left')), int(np.searchsorted(self.pmass, m + t, 'right'))

    def negatives(self, pos, rng):
        """(len(pos), K) pool rows: random other rows of each positive's +-ppm window, filled with the nearest rows by mass."""
        out = np.empty((len(pos), self.K), np.int64); n = len(self.pmass)
        for r, pid in enumerate(pos):
            lo, hi = self.window(pid)
            c = np.r_[np.arange(lo, pid), np.arange(pid + 1, hi)]
            if len(c) >= self.K:
                out[r] = rng.choice(c, self.K, replace=False)
            else:
                need = self.K - len(c)
                e = np.r_[np.arange(max(0, lo - need), lo), np.arange(hi, min(n, hi + need))]
                e = e[np.argsort(np.abs(self.pmass[e] - self.pmass[pid]), kind='stable')[:need]]
                c = np.r_[c, e]
                out[r] = c if len(c) == self.K else rng.choice(c, self.K, replace=True)
        return out

    def sample_ids(self, rng, bs, parts=None):
        """Half of the batch uniform over training spectra, half uniform over (structure, polarity) groups (one random
        spectrum each). parts: [(share of the spectrum-uniform half, index array)] draws that share from a subset instead."""
        h = bs // 2; out = []; used = 0
        acc = 0.0
        for share, idx in (parts or []):
            acc += h * share; n = min(int(round(acc)) - used, h - used)          # cumulative rounding: the shares add up exactly
            if n > 0:
                out.append(rng.choice(idx, size=n, replace=len(idx) < n)); used += n
        if h > used:
            out.append(rng.choice(self.tr, size=h - used, replace=False))
        gs = rng.choice(self.tr_groups, size=bs - h, replace=False)
        b = np.empty(len(gs), np.int64)
        for j, g in enumerate(gs):
            mem = self.grp_order[self.gs[g]:self.ge[g]]; mem = mem[self.in_tr[mem]]
            b[j] = mem[rng.integers(len(mem))]
        out.append(b)
        return np.concatenate(out)

    def batch(self, rng, bs, merge_p=0.3, aug=None, parts=None, ids=None):
        import fpnet
        ids = self.sample_ids(rng, bs, parts) if ids is None else ids
        items = [self.train_item(int(i), rng, merge_p, aug) for i in ids]
        pos = self.sid2pid[self.sid[ids]]
        return fpnet.collate(items, 'cpu'), pos, self.negatives(pos, rng), ids


AUG_NONE = dict()
AUG_STD = dict(jitter=0.2, drop=0.25, drop_p=0.5, ce_hide=0.1)


def subset(D: SpecData, spec: str):
    """Index array of training spectra for a sampling part: 'tims', 'enveda', 'np' (natural-product-catalogue structures),
    'instr:<k>', 'lib:<k>'."""
    t = D.tr
    if spec == 'tims':
        return t[D.instr[t] == 0]
    if spec == 'enveda':
        return t[np.isin(D.lib[t], ENVEDA_LIBS)]
    if spec == 'np':
        return t[D.psrc[D.sid2pid[D.sid[t]]] >= 2]
    kind, k = spec.split(':')
    return t[(D.instr[t] if kind == 'instr' else D.lib[t]) == int(k)]


def parse_parts(D: SpecData, text):
    """'tims=0.2,instr:1=0.3' -> [(share, index array)]; 'instr_bal' -> every instrument class an equal share of the
    spectrum-uniform half; 'lib_bal' the same over the libraries."""
    if not text:
        return None
    if text in ('instr_bal', 'lib_bal'):
        v = D.instr[D.tr] if text == 'instr_bal' else D.lib[D.tr]; ks = np.unique(v)
        return [(1.0 / len(ks), D.tr[v == k]) for k in ks]
    return [(float(s), subset(D, n)) for n, s in (x.split('=') for x in text.split(','))]


class BatchStream(_Iterable):
    """Endless stream of training batches for torch.utils.data.DataLoader worker processes. Each worker opens the tables
    itself (memory-mapped) and seeds its generator with (seed, worker id, start step): a run is reproducible given its
    seed and its resume points (up to the non-determinism of GPU arithmetic)."""

    def __init__(self, work, pool, holdout, seed, start_step, bs, merge_p, aug, parts_text, K=63, only=None):
        self.cfg = dict(work=work, pool=pool, holdout=holdout, K=K); self.seed, self.start_step = seed, start_step
        self.bs, self.merge_p, self.aug, self.parts_text, self.only = bs, merge_p, aug, parts_text, only

    def __iter__(self):
        from torch.utils.data import get_worker_info
        w = get_worker_info()
        return self.stream(w.id if w is not None else 0)

    def stream(self, worker_id=0):
        D = SpecData(log=lambda *a: None, need_fp=False, **self.cfg)
        if self.only:
            D.restrict(subset(D, self.only))
        rng = np.random.default_rng([self.seed, worker_id, self.start_step]); parts = parse_parts(D, self.parts_text)
        while True:
            arrs, pos, neg, ids = D.batch(rng, self.bs, self.merge_p, self.aug, parts)
            yield arrs, pos, neg, ids
