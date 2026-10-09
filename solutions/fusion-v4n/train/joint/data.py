"""Leak-safe training data of the joint spectrum-structure model: compact index, molecule batches with isomer negatives, statistics.

Positives: structures of train.parquet that are NOT in the held-out folds of the identity-disjoint split (`hold` = fold 0, `np` = the
natural-product panel) and share none of their three identifiers (InChIKey first block, tautomer-canonical metric key, raw key) with a
held-out structure. `build_index` writes small memory-mappable arrays (no copy of the spectrum cache); `assert_leak_safe` proves the
disjointness against the benchmark's answers and prints the counts.

A batch (built inside a DataLoader worker, `BatchDataset[i]` = batch i of an epoch, deterministic in (seed, epoch, i)):
  spectra   up to `max_spec` spectra of each of B molecules, flattened: mz / inten [T, P] (top `max_peaks` peaks by intensity, m/z order),
            n_peaks, precursor m/z, adduct index, polarity, collision energy (+ known flag), instrument family, owner (molecule index)
  positive  pool index of the true structure and its packed fingerprint (the pool's 10,226 selected bits)
  negatives K per molecule, in this order of preference: same-formula isomers in the pool; cached PubChem window negatives of
            train/mine_pc_negatives.py (half of the rest hard = most similar, half random; mostly same formula); other pool rows within +-10 ppm; random pool rows.
            Held-out structures may be NEGATIVES (default): on the hidden test the answer of a class-2 molecule is a pool row and is drawn
            as a negative of other spectra too, so excluding them would favour the benchmark's answers. `neg_hold=False` excludes them.
Negative kinds (neg_kind): 0 pool isomer, 1 PubChem hard, 2 PubChem random, 3 pool near-mass, 4 pool random, -1 none.

usage: python train/joint/data.py --build          (index under $CASMI_DATA/joint/index, resumable per file)
       python train/joint/data.py --stats          (leak-safety counts and negative statistics as JSON)
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jpaths as P

HOLD = ('hold', 'np')
IDX = P.JOINT + '/index'
PPM = 10.0


def _save(name, arr):
    np.save(f'{IDX}/{name}.tmp.npy', arr); os.replace(f'{IDX}/{name}.tmp.npy', f'{IDX}/{name}.npy')


def csr(codes, n):
    """Rows grouped by code: (ptr [n + 1], rows) with rows[ptr[c]:ptr[c + 1]] = positions whose code is c."""
    order = np.argsort(codes, kind='stable').astype(np.int32); ptr = np.searchsorted(codes[order], np.arange(n + 1)).astype(np.int64); return ptr, order


def train_mask(structs, split, hold=HOLD):
    """Boolean per structure: may be a training positive. Removes the held-out folds and every structure sharing an identifier with them."""
    held = structs.sid.isin(set(split.sid[split.fold.isin(hold)])).values
    banned = set()
    for c in ('inchikey14', 'key', 'rawkey'):
        if c in structs:
            banned |= set(structs[c].values[held])
    for c in ('inchikey14', 'key'):
        if c in split:
            banned |= set(split[c][split.fold.isin(hold)])
    shared = np.zeros(len(structs), bool)
    for c in ('inchikey14', 'key', 'rawkey'):
        if c in structs:
            shared |= structs[c].isin(banned).values
    return ~(held | shared), int((shared & ~held).sum())


def build_index(log=print):
    import pandas as pd
    os.makedirs(IDX, exist_ok=True)
    if os.path.exists(IDX + '/meta.json'):
        log('index exists'); return
    st = pd.read_parquet(P.WORK + '/train_structs.parquet'); sp = pd.read_parquet(P.WORK + '/split.parquet')
    pool = pd.read_parquet(P.WORK + '/pool_meta.parquet', columns=['key', 'formula', 'mass', 'smiles'])
    sm = pd.read_parquet(P.WORK + '/spec_meta.parquet', columns=['sid', 'adduct_ix', 'mode', 'precursor_mz', 'ce_mean', 'instr', 'n_clean'])
    assert (st.sid.values == np.arange(len(st))).all() and pool.key.is_unique
    ok, n_shared = train_mask(st, sp); k2p = dict(zip(pool.key.values, np.arange(len(pool))))
    sid_pid = np.array([k2p.get(k, -1) for k in st.key.values], np.int32)
    held_pid = np.zeros(len(pool), bool); held_pid[sid_pid[~ok & (sid_pid >= 0)]] = True
    sid = sm.sid.values.astype(np.int32); ptr, rows = csr(sid, len(st)); nspec = np.diff(ptr)
    tr = np.flatnonzero(ok & (sid_pid >= 0) & (nspec > 0)).astype(np.int32)
    fcode = pd.factorize(pool.formula)[0].astype(np.int32); fptr, frows = csr(fcode, int(fcode.max()) + 1)
    for name, arr in dict(spec_sid=sid, spec_prec=sm.precursor_mz.values.astype(np.float32), spec_add=sm.adduct_ix.values.astype(np.int16), spec_mode=sm['mode'].values.astype(np.int8),
                          spec_ce=sm.ce_mean.values.astype(np.float32), spec_instr=sm.instr.values.astype(np.int8), sid_ptr=ptr, sid_rows=rows, sid_pid=sid_pid, tr_sid=tr,
                          pool_mass=pool.mass.values.astype(np.float64), pool_form=fcode, form_ptr=fptr, form_rows=frows, pool_held=held_pid).items():
        _save(name, arr)
    b = '\n'.join(pool.smiles.values).encode('ascii'); off = np.r_[0, np.cumsum([len(s) + 1 for s in pool.smiles.values])].astype(np.int64)
    _save('pool_smiles', np.frombuffer(b, np.uint8)); _save('pool_smiles_off', off)
    meta = dict(structures=len(st), held_out=int((~ok).sum()), shared_identifier_removed=n_shared, train_structures=int(len(tr)), train_spectra=int(nspec[tr].sum()),
                train_pool_rows=int(len(np.unique(sid_pid[tr]))), pool=len(pool), spectra=len(sm), adducts=int(sm.adduct_ix.max()) + 1, instr=int(sm.instr.max()) + 1)
    if os.path.exists(P.PCNEG + '/pid.npy'):                                     # PubChem window negatives (train/mine_pc_negatives.py)
        pid = np.load(P.PCNEG + '/pid.npy'); m = np.full(len(pool), -1, np.int32); m[pid] = np.arange(len(pid)); _save('pc_of_pid', m)
        rows = np.load(P.PCNEG + '/rows.npy'); _save('pc_fp_mass', np.asarray(np.load(P.TIER + '/pc_mass.npy', mmap_mode='r')[rows], np.float64))
        meta['pc_structures'] = int(len(pid)); meta['pc_fp_rows'] = int(len(rows))
    json.dump(meta, open(IDX + '/meta.json', 'w'), indent=1); log(json.dumps(meta))


class JointData:
    """Memory-mapped view of the index, the spectrum cache and the fingerprint tables. Cheap to open in every worker."""

    def __init__(self, neg_hold=True, pc_neg=True, graphs=False):
        L = lambda n: np.load(f'{IDX}/{n}.npy', mmap_mode='r')
        self.meta = json.load(open(IDX + '/meta.json'))
        self.off = np.load(P.WORK + '/spec_off.npy', mmap_mode='r'); self.mz = np.load(P.WORK + '/spec_mz.npy', mmap_mode='r'); self.it = np.load(P.WORK + '/spec_it.npy', mmap_mode='r')
        for n in ('spec_sid', 'spec_prec', 'spec_add', 'spec_mode', 'spec_ce', 'spec_instr', 'sid_ptr', 'sid_rows', 'sid_pid', 'tr_sid', 'pool_mass', 'pool_form', 'form_ptr', 'form_rows', 'pool_held'):
            setattr(self, n, L(n))
        self.pool_fp = np.load(P.WORK + '/pool_fp.npy', mmap_mode='r'); self.nbits = int(np.load(P.WORK + '/fp_bits.npy').shape[0]); self.nbytes = self.pool_fp.shape[1]
        self.neg_hold = neg_hold; self.pc = pc_neg and os.path.exists(IDX + '/pc_of_pid.npy') and os.path.exists(P.PCNEG + '/fp.npy')
        if self.pc:
            self.pc_of_pid = L('pc_of_pid'); self.pc_hard = np.load(P.PCNEG + '/hard.npy', mmap_mode='r'); self.pc_rand = np.load(P.PCNEG + '/rand.npy', mmap_mode='r')
            self.pc_fp = np.load(P.PCNEG + '/fp.npy', mmap_mode='r'); self.pc_fp_mass = L('pc_fp_mass')
        self.graphs = graphs
        if graphs:
            self.pool_smiles = L('pool_smiles'); self.pool_smiles_off = L('pool_smiles_off')
            if self.pc:
                self.pc_rows = np.load(P.PCNEG + '/rows.npy', mmap_mode='r'); self.t_off = np.load(P.TIER + '/pc_off.npy', mmap_mode='r'); self.t_smi = np.load(P.TIER + '/pc_smiles.npy', mmap_mode='r')

    # ---- structures
    def smiles(self, src, i):
        """SMILES of a structure reference (src 0 = pool row, 1 = row of the PubChem negative table)."""
        if src == 0:
            return bytes(self.pool_smiles[self.pool_smiles_off[i]:self.pool_smiles_off[i + 1] - 1]).decode('ascii')
        r = int(self.pc_rows[i]); return bytes(self.t_smi[int(self.t_off[r]):int(self.t_off[r + 1])]).decode('ascii')

    def fp(self, src, idx):
        """Packed fingerprints [n, nbytes] of structure references (arrays)."""
        out = np.zeros((len(idx), self.nbytes), np.uint8); p = src == 0
        if p.any():
            o = np.argsort(idx[p]); out[np.flatnonzero(p)[o]] = self.pool_fp[idx[p][o]]
        if (~p).any():
            o = np.argsort(idx[~p]); out[np.flatnonzero(~p)[o]] = self.pc_fp[idx[~p][o]]
        return out

    def negatives(self, pid, k, rng, mode='mix', mined=None):
        """k negatives of one positive pool row: (src, idx, kind) arrays; kind -1 = none left.
        mode 'mix' (reference: half pool isomers, then PubChem hard / random) | 'none' (no own negatives) | 'pool' (pool rows only) |
        'pc' (cached PubChem rows first, no pool isomers). `mined` [n_pool, M] int32 (>= 0 pool row, <= -2 PubChem row -(v + 2), -1 none):
        half of k is drawn from it first (kind 5)."""
        src = np.zeros(k, np.int8); idx = np.zeros(k, np.int64); kind = np.full(k, -1, np.int8); n = 0
        if mode == 'none':
            return src, idx, kind

        def take(cand, s, kd, limit):
            nonlocal n
            if n >= k or len(cand) == 0:
                return
            m = min(limit, k - n, len(cand)); c = cand if len(cand) == m else rng.choice(cand, m, replace=False)
            src[n:n + m] = s; idx[n:n + m] = c; kind[n:n + m] = kd; n += m
        f = int(self.pool_form[pid]); iso = np.asarray(self.form_rows[self.form_ptr[f]:self.form_ptr[f + 1]]); iso = iso[iso != pid]
        if not self.neg_hold:
            iso = iso[~self.pool_held[iso]]
        j = int(self.pc_of_pid[pid]) if (self.pc and mode != 'pool') else -1
        hard = rnd = np.zeros(0, np.int64)
        if j >= 0:
            hard = np.asarray(self.pc_hard[j]); hard = hard[hard >= 0]; rnd = np.asarray(self.pc_rand[j]); rnd = rnd[rnd >= 0]
        if mined is not None:
            mv = np.asarray(mined[pid]); mv = mv[mv != -1]
            if len(mv):
                mv = mv if len(mv) <= k // 2 else rng.choice(mv, k // 2, replace=False); m = len(mv); p = mv >= 0
                src[:m] = np.where(p, 0, 1); idx[:m] = np.where(p, mv, -(mv + 2)); kind[:m] = 5; n = m
                iso = iso[~np.isin(iso, mv[p])]; pcm = -(mv[~p] + 2); hard = hard[~np.isin(hard, pcm)]; rnd = rnd[~np.isin(rnd, pcm)]
        k0 = k - n
        if mode == 'pc' and len(hard) + len(rnd):
            iso = iso[:0]
        half = (k0 + 1) // 2 if len(hard) + len(rnd) else k0
        take(iso, 0, 0, half); n_hard = (k - n + 1) // 2; take(hard, 1, 1, n_hard); take(rnd, 1, 2, k)
        if n < k and len(hard) > n_hard:                                        # more hard ones when the random ones ran out
            take(np.setdiff1d(hard, idx[:n][kind[:n] == 1]), 1, 1, k)
        if n < k and len(iso) > half:                                           # more pool isomers when PubChem had too few
            take(np.setdiff1d(iso, idx[:n][src[:n] == 0]), 0, 0, k)
        if n < k:
            m = float(self.pool_mass[pid]); lo = np.searchsorted(self.pool_mass, m * (1 - PPM * 1e-6)); hi = np.searchsorted(self.pool_mass, m * (1 + PPM * 1e-6))
            near = np.arange(lo, hi); near = near[(near != pid) & (self.pool_form[lo:hi] != f)]
            if not self.neg_hold:
                near = near[~self.pool_held[near]]
            take(near, 0, 3, k)
        if n < k:
            r = rng.integers(0, len(self.pool_mass), k - n); r[r == pid] = (pid + 1) % len(self.pool_mass); take(r, 0, 4, k)
        return src, idx, kind

    # ---- spectra
    def spectra_of(self, sid):
        return np.asarray(self.sid_rows[self.sid_ptr[sid]:self.sid_ptr[sid + 1]])

    def tokens(self, rows, owner, max_peaks):
        """Flattened spectra -> padded arrays (the spectrum part of a batch)."""
        return spectra_arrays([(self.mz[self.off[r]:self.off[r + 1]], self.it[self.off[r]:self.off[r + 1]]) for r in rows], np.asarray(self.spec_prec[rows] if len(rows) else [], np.float32),
                              np.asarray(self.spec_add[rows], np.int64), np.asarray(self.spec_mode[rows], np.int64), np.asarray(self.spec_ce[rows], np.float32),
                              np.asarray(self.spec_instr[rows], np.int64), owner, max_peaks)


def merge_groups(D, rows):
    """Group spectrum rows by (adduct, polarity): a merged peak set only makes sense within one precursor type. Returns a list of row arrays."""
    rows = np.asarray(rows); a = np.asarray(D.spec_add[np.sort(rows)])[np.argsort(np.argsort(rows))].astype(np.int64) * 2 + (np.asarray(D.spec_mode[np.sort(rows)])[np.argsort(np.argsort(rows))] > 0)
    return [rows[a == v] for v in np.unique(a)]


def merged_arrays(D, groups, owner, max_peaks):
    """One pseudo-spectrum per group of rows: the union of their peaks (each spectrum scaled to a base peak of 1), most intense `max_peaks`
    kept; precursor, adduct, polarity, instrument of the first row, mean of the known collision energies."""
    peaks, prec, add, mode, ce, instr = [], [], [], [], [], []
    for r in groups:
        mz = np.concatenate([np.asarray(D.mz[D.off[x]:D.off[x + 1]]) for x in r]); it = np.concatenate([np.asarray(D.it[D.off[x]:D.off[x + 1]]) / max(float(np.max(D.it[D.off[x]:D.off[x + 1]], initial=0.0)), 1e-9) for x in r])
        o = np.argsort(mz, kind='stable'); peaks.append((mz[o], it[o])); r0 = int(r[0]); c = np.asarray([D.spec_ce[int(x)] for x in r], np.float32); c = c[np.isfinite(c)]
        prec.append(D.spec_prec[r0]); add.append(D.spec_add[r0]); mode.append(D.spec_mode[r0]); instr.append(D.spec_instr[r0]); ce.append(c.mean() if len(c) else np.nan)
    return spectra_arrays(peaks, np.asarray(prec, np.float32), np.asarray(add, np.int64), np.asarray(mode, np.int64), np.asarray(ce, np.float32), np.asarray(instr, np.int64), owner, max_peaks)


def spectra_arrays(peaks, prec, add, mode, ce, instr, owner, max_peaks):
    """Pad a list of (mz, intensity) arrays to [T, P]; keeps the `max_peaks` most intense peaks of a spectrum in m/z order."""
    T = len(peaks); n = np.array([min(len(m), max_peaks) for m, _ in peaks], np.int64); Pm = max(int(n.max()) if T else 1, 1)
    mz = np.zeros((T, Pm), np.float32); it = np.zeros((T, Pm), np.float32)
    for t, (m, v) in enumerate(peaks):
        if len(m) > max_peaks:
            keep = np.sort(np.argpartition(-np.asarray(v), max_peaks - 1)[:max_peaks]); m = np.asarray(m)[keep]; v = np.asarray(v)[keep]
        mz[t, :len(m)] = m; it[t, :len(m)] = v
    ce = np.asarray(ce, np.float32); known = np.isfinite(ce)
    return dict(mz=mz, it=it, n_peaks=n, prec=np.asarray(prec, np.float32), add=np.asarray(add, np.int64), mode=(np.asarray(mode) > 0).astype(np.int64),
                ce=np.where(known, ce, 0.0).astype(np.float32), ce_known=known.astype(np.float32), instr=np.asarray(instr, np.int64), owner=np.asarray(owner, np.int64))


def warm(pc_neg=True, log=print):
    """Read the two fingerprint tables once so the OS file cache holds them (0.85 + 4.0 GB of standby memory, shared by all workers).
    Cold, a random row costs a page fault from disk (measured 0.3 ms: 1.2 s per batch of 256 x 16 structures); warm, 2.5 microseconds."""
    t0 = time.time(); n = 0
    for f in [P.WORK + '/pool_fp.npy'] + ([P.PCNEG + '/fp.npy'] if pc_neg and os.path.exists(P.PCNEG + '/fp.npy') else []):
        with open(f, 'rb', buffering=0) as fh:
            while True:
                b = fh.read(64 << 20)
                if not b:
                    break
                n += len(b)
    log(f'warmed {n / 2 ** 30:.1f} GB of fingerprint tables in {time.time() - t0:.1f}s'); return time.time() - t0


class BatchDataset:
    """Map-style dataset whose item i is a whole batch (use DataLoader(batch_size=None)). Opens the memory maps lazily in each worker."""

    def __init__(self, bs=256, k=15, max_spec=4, max_peaks=96, steps=1000, seed=0, epoch=0, frac=1.0, graphs=False, neg_hold=True, pc_neg=True, fixed_spec=False, offset=0, neg_mode='mix', mined='', merge=False):
        self.a = dict(bs=bs, k=k, max_spec=max_spec, max_peaks=max_peaks, steps=steps, seed=seed, epoch=epoch, frac=frac, graphs=graphs, neg_hold=neg_hold, pc_neg=pc_neg, fixed_spec=fixed_spec, offset=offset,
                      neg_mode=neg_mode, mined=mined, merge=merge)
        self.D = None; self.mined = None

    def __len__(self):
        return self.a['steps']

    def _open(self):
        if self.D is None:
            a = self.a; self.D = JointData(a['neg_hold'], a['pc_neg'], a['graphs']); tr = np.asarray(self.D.tr_sid)
            if a['frac'] < 1.0:                                                   # a fixed subset of the training molecules (ablation runs)
                tr = np.sort(np.random.default_rng(12345).permutation(tr)[:max(int(len(tr) * a['frac']), a['bs'])])
            self.tr = tr; self.mined = np.load(a['mined'], mmap_mode='r') if a['mined'] else None
        return self.D

    def __getitem__(self, i):
        a = self.a; D = self._open(); i = i + a['offset']; rng = np.random.default_rng([a['seed'], a['epoch'], i])        # offset: resume inside the stream
        sids = rng.choice(self.tr, a['bs'], replace=False); rows, owner = [], []
        for b, s in enumerate(sids):
            r = D.spectra_of(int(s))
            if a['merge']:                                                        # spectra of one precursor type only, merged below
                g = merge_groups(D, r); r = g[int(rng.integers(0, len(g)))]
            m = min(len(r), a['max_spec']); m = m if a['fixed_spec'] else int(rng.integers(1, m + 1))
            r = r if len(r) == m else rng.choice(r, m, replace=False); rows.append(np.sort(r)); owner.append(np.full(m, b))
        if a['merge']:
            out = merged_arrays(D, rows, np.arange(a['bs']), a['max_peaks'])
        else:
            rows = np.concatenate(rows); out = D.tokens(rows, np.concatenate(owner), a['max_peaks'])
        pid = np.asarray(D.sid_pid[np.sort(sids)], np.int64)[np.argsort(np.argsort(sids))]; K = a['k']
        ns = np.zeros((a['bs'], K), np.int8); ni = np.zeros((a['bs'], K), np.int64); nk = np.zeros((a['bs'], K), np.int8)
        for b in range(a['bs']):
            ns[b], ni[b], nk[b] = D.negatives(int(pid[b]), K, rng, a['neg_mode'], self.mined)
        out.update(pid=pid, neg_src=ns, neg_idx=ni, neg_kind=nk, n_mol=np.int64(a['bs']))
        out['pos_fp'] = D.fp(np.zeros(a['bs'], np.int8), pid); out['neg_fp'] = D.fp(ns.ravel(), ni.ravel()).reshape(a['bs'], K, -1)
        if a['graphs']:
            import graphs as G
            smi = [D.smiles(0, int(p)) for p in pid] + [D.smiles(int(s), int(j)) for s, j in zip(ns.ravel(), ni.ravel())]
            out.update(G.collate([G.featurize(s) for s in smi]))
        return out


def assert_leak_safe(bench_mols=None, log=print):
    """No training positive shares an InChIKey first block (or metric key) with a held-out structure or a benchmark answer. Returns the counts."""
    import pandas as pd
    st = pd.read_parquet(P.WORK + '/train_structs.parquet', columns=['sid', 'inchikey14', 'key', 'rawkey']); sp = pd.read_parquet(P.WORK + '/split.parquet')
    tr = np.load(IDX + '/tr_sid.npy'); T = st.iloc[tr]; held = st[st.sid.isin(set(sp.sid))]
    ids_tr = set(T.inchikey14) | set(T.key) | set(T.rawkey); ids_held = set(held.inchikey14) | set(held.key) | set(held.rawkey) | set(sp.inchikey14) | set(sp.key)
    c = dict(train_positive_structures=int(len(T)), train_first_blocks=int(T.inchikey14.nunique()), train_metric_keys=int(T.key.nunique()), held_out_structures=int(len(held)),
             shared_with_held_out=len(ids_tr & ids_held))
    assert c['shared_with_held_out'] == 0, 'a training positive shares an identifier with a held-out structure'
    sid = np.load(IDX + '/spec_sid.npy', mmap_mode='r'); pos_sids = set(tr.tolist())
    if bench_mols is not None:
        ans = set(bench_mols.key); ik = st.inchikey14[st.key.isin(ans)]; ans_all = ans | set(ik)
        c.update(benchmark_answers=int(len(bench_mols)), benchmark_answer_first_blocks=int(ik.nunique()), shared_first_block_or_key_with_answers=len(ids_tr & ans_all),
                 benchmark_answers_in_held_out_folds=int(bench_mols.key.isin(set(sp.key)).sum()))
        assert c['shared_first_block_or_key_with_answers'] == 0 and c['benchmark_answers_in_held_out_folds'] == len(bench_mols)
        if 'spec_rows' in bench_mols.attrs:
            q = np.asarray(bench_mols.attrs['spec_rows']); c['benchmark_spectra'] = int(len(q)); c['benchmark_spectra_of_training_structures'] = int(np.isin(np.asarray(sid)[q], tr).sum())
            assert c['benchmark_spectra_of_training_structures'] == 0
    log(json.dumps(c)); return c


def stats(k_values=(15, 31), log=print):
    """Counts of the training set and of the negatives available per training structure."""
    import pandas as pd
    D = JointData(); tr = np.asarray(D.tr_sid); pid = np.unique(np.asarray(D.sid_pid)[tr]); q = lambda v: {str(p): float(np.percentile(v, p)) for p in (0, 5, 25, 50, 75, 95, 100)}
    f = np.asarray(D.pool_form)[pid]; n_iso = (np.diff(np.asarray(D.form_ptr))[f] - 1).astype(np.int64)
    held = np.asarray(D.pool_held); cnt_h = np.bincount(np.asarray(D.pool_form)[held], minlength=len(D.form_ptr) - 1)[f]
    pm = np.asarray(D.pool_mass); m = pm[pid]; near = np.searchsorted(pm, m * (1 + PPM * 1e-6)) - np.searchsorted(pm, m * (1 - PPM * 1e-6)) - 1 - n_iso; near = np.maximum(near, 0)
    nspec = np.diff(np.asarray(D.sid_ptr))[tr]; npk = np.diff(np.asarray(D.off))
    out = dict(index=D.meta, train_structures=int(len(tr)), train_pool_rows=int(len(pid)), train_spectra=int(nspec.sum()), spectra_per_structure=q(nspec),
               peaks_per_spectrum=q(npk[::7]), share_spectra_over_96_peaks=float((npk > 96).mean()), share_spectra_over_128_peaks=float((npk > 128).mean()),
               pool_isomers=dict(with_at_least_1=int((n_iso >= 1).sum()), share=float((n_iso >= 1).mean()), dist=q(n_iso), of_which_held_out_mean=float(cnt_h.mean()),
                                 share_with_a_held_out_isomer=float((cnt_h > 0).mean())),
               pool_near_mass_other_formula=dict(with_at_least_1=int((near >= 1).sum()), dist=q(near)))
    tot = n_iso.copy(); iso_any = n_iso >= 1
    if D.pc:
        j = np.asarray(D.pc_of_pid)[pid]; has = j >= 0; S = pd.read_parquet(P.PCNEG + '/stats.parquet').set_index('pid').reindex(pid)
        nh = S.n_hard.fillna(0).values.astype(int); nr = S.n_rand.fillna(0).values.astype(int); same = (S.hard_same.fillna(0) + S.rand_same.fillna(0)).values.astype(int)
        out['pubchem_cached'] = dict(structures_with_cache=int(has.sum()), share=float(has.mean()), hard=q(nh), random=q(nr), same_formula_among_cached=q(same),
                                     with_at_least_1_same_formula=int((same >= 1).sum()), window_rows=q(S.n_window.fillna(0).values), mean_tanimoto_of_hard=float(S.tan_mean.mean()),
                                     fp_rows=int(D.pc_fp.shape[0]))
        tot = tot + nh + nr; iso_any = iso_any | (same >= 1)
    else:
        out['pubchem_cached'] = 'missing: ' + P.PCNEG
    out['isomer_negatives'] = dict(structures_with_at_least_1=int(iso_any.sum()), share=float(iso_any.mean()), without_any=int((~iso_any).sum()))
    out['negatives_available_per_positive'] = dict(dist=q(tot), **{f'share_with_at_least_{k}': float((tot >= k).mean()) for k in k_values})
    ds = BatchDataset(bs=512, k=15, steps=8); kinds = np.concatenate([ds[i]['neg_kind'].ravel() for i in range(4)])
    out['drawn_kinds_k15'] = {n: float((kinds == c).mean()) for c, n in enumerate(['pool isomer', 'PubChem hard', 'PubChem random', 'pool near-mass', 'pool random'])}
    log(json.dumps(out, indent=1)); return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--build', action='store_true'); ap.add_argument('--stats', action='store_true'); ap.add_argument('--out', default=''); a = ap.parse_args()
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if a.build:
        build_index()
    if a.stats:
        import pandas as pd
        r = {}
        if os.path.exists(P.BENCH + '/mols.parquet'):
            M = pd.read_parquet(P.BENCH + '/mols.parquet'); M.attrs['spec_rows'] = pd.read_parquet(P.BENCH + '/spectra.parquet').row.values; r['leak'] = assert_leak_safe(M)
        else:
            r['leak'] = assert_leak_safe()
        r['stats'] = stats()
        if a.out:
            json.dump(r, open(a.out, 'w'), indent=1)
