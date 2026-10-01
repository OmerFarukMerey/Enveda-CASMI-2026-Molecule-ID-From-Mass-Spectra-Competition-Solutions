"""Build an EXTERNAL validation set from public-library spectra (GNPS + MassBank), leak-free with respect to train.parquet.

Every public artefact of the pipeline (FPNets, rankers, v2 pool) was trained on all of train.parquet, so any holdout drawn
from train is leaked (EXPERIMENTS.md, val1). This tool draws two labelled parts from libraries newer than train:

  part A 'novel' (molecule_id x_*): structures absent from train (neither the plain InChIKey14 nor the tautomer-canonical
      metric key), i.e. class-2/3-like molecules no artefact has seen. ext v1 showed that GNPS re-deposits identical spectra
      under different positional-isomer annotations (404/500 v1 molecules had a train spectrum of another structure with the
      truth's formula at cosine >= 0.99), so an A spectrum is dropped when its best cosine against train spectra of the same
      ion mode within PPM of its precursor m/z is >= --ambig-cos (all train structures differ from an A truth).
  part B 'known' (molecule_id k_*): structures present in train (metric key) whose external spectrum is not a copy of a train
      spectrum: best cosine to the same structure's train spectra (same ion mode, any precursor) < --copy-cos. Up to --n-known
      structures, seeded, preferring structures with a TOF spectrum with [M+H]+/[M-H]-, then [M+H]+/[M-H]-, then TOF:
      class-1-like molecules the library already knows.

Both parts drop spectra whose rounded peak list (m/z at 0.01 Da of peaks with intensity > 0 and m/z <= precursor + 2) equals
that of any train spectrum, and exact duplicates within a structure (GNPS re-imports MassBank); at most --max-spec spectra per
structure, preferring TOF and [M+H]+/[M-H]-. Cosine: sqrt intensities, unit norm, peaks <= precursor + 2, one-to-one matching
at 0.01 Da (each library peak to its nearest query peak, each query peak keeps its best partner). Differences from the hidden
test: mostly Orbitrap / qTOF spectra (relabelled timsTOF, as in build_validation), GNPS has no collision energy (imputed,
flagged), and the structures are library compounds, not NP extracts.

Inputs (downloaded by hand, see docs): GNPS ALL_GNPS_NO_PROPOGATED.mgf, MassBank 2026.03 MassBank_NISTformat.msp,
and train.parquet. Adducts are assigned by precursor m/z (<= PPM of neutral mass + adduct), not by the record's label,
and only the test's 10 adducts are kept. Output dir: test.parquet (competition columns), sample_submission.csv,
val_labels.csv (molecule_id, smiles, inchikey14, source, n_spectra, ce_imputed, part), manifest.json, audit.parquet (every
candidate spectrum with its cosines and fate) and kaggle/{ext_test.parquet, ext_val_labels.csv, ext_manifest.json}.

usage: python tools/build_ext_val.py --gnps <mgf> --massbank <msp> --train <train.parquet> --out <dir>
"""
import argparse, hashlib, json, os, re, resource, sys, time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROTON = 1.007276467
ADDUCTS = {  # name -> (ionization mode, m/z shift added to the neutral mass M)
    '[M+H]+': ('positive', PROTON), '[M+NH4]+': ('positive', 18.033825566), '[M-H2O+H]+': ('positive', PROTON - 18.010564684),
    '[M-2H2O+H]+': ('positive', PROTON - 2 * 18.010564684), '[M+Na]+': ('positive', 22.989221), '[M+K]+': ('positive', 38.963158),
    '[M-H]-': ('negative', -PROTON), '[M-H2O-H]-': ('negative', -PROTON - 18.010564684),
    '[M+CH2O2-H]-': ('negative', 44.998201), '[M+Cl]-': ('negative', 34.969402),
}
PPM = 10.0
TOL = 0.01                         # Da, peak matching tolerance of the cosine and rounding step of the peak-list hash
THRESHOLDS = (0.7, 0.85, 0.95)     # duplicate-threshold table
PREF_ADDUCTS = ('[M+H]+', '[M-H]-')
CE_IMPUTED = [20.0, 40.0, 60.0]   # most common CE list of the visible test; used where the library gives none
TEST_COLS = ['molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct',
             'ionization_mode', 'instrument_type', 'precursor_mz', 'collision_energy_orig', 'collision_energy_ev',
             'collision_energy_orig_units']
U64 = np.uint64


def describe(smi):
    """(plain IK14, metric IK14, canonical SMILES, exact mass) or None (unparsable / multi-component / charged)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    global _TAUT
    if '_TAUT' not in globals():
        _TAUT = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None or len(Chem.GetMolFrags(m)) != 1 or Chem.GetFormalCharge(m) != 0:
            return smi, None
        return smi, (Chem.MolToInchiKey(m)[:14], Chem.MolToInchiKey(_TAUT.Canonicalize(m))[:14],
                     Chem.MolToSmiles(m), rdMolDescriptors.CalcExactMolWt(m))
    except Exception:
        return smi, None


def read_gnps(path):
    rec, mz, it = {}, [], []
    for line in open(path, encoding='utf-8', errors='replace'):
        if line.startswith('BEGIN IONS'):
            rec, mz, it = {}, [], []
        elif line.startswith('END IONS'):
            if rec.get('MSLEVEL', '2') == '2' and rec.get('SMILES', 'N/A') not in ('', 'N/A', ' '):
                yield dict(source='gnps', record=rec.get('SPECTRUMID', ''), smiles=rec['SMILES'].strip(),
                           mode=rec.get('IONMODE', '').lower(), prec=float(rec.get('PEPMASS', '0').split()[0] or 0),
                           instrument=rec.get('SOURCE_INSTRUMENT', ''), ce='', mz=np.asarray(mz), it=np.asarray(it))
        elif line[:1].isdigit():
            p = line.split()
            if len(p) >= 2:
                mz.append(float(p[0])); it.append(float(p[1]))
        elif '=' in line:
            k, v = line.rstrip('\n').split('=', 1); rec[k] = v


def read_massbank(path):
    rec, mz, it = {}, [], []
    def flush():
        if rec and rec.get('spectrum_type', '').upper() == 'MS2' and rec.get('smiles'):
            yield dict(source='massbank', record=rec.get('db#', ''), smiles=rec['smiles'], mode=rec.get('ion_mode', '').lower(),
                       prec=float(re.sub(r'[^0-9.].*', '', rec.get('precursormz', '0')) or 0),
                       instrument=rec.get('instrument_type', ''), ce=rec.get('collision_energy', ''),
                       mz=np.asarray(mz), it=np.asarray(it))
    for line in open(path, encoding='utf-8', errors='replace'):
        if not line.strip():
            yield from flush(); rec, mz, it = {}, [], []
        elif line[:1].isdigit():
            p = line.split()
            if len(p) >= 2:
                mz.append(float(p[0])); it.append(float(p[1]))
        elif ':' in line:
            k, v = line.split(':', 1); rec.setdefault(k.strip().lower(), v.strip())
    yield from flush()


def assign_adduct(mass, prec, mode):
    best = None
    for name, (m, shift) in ADDUCTS.items():
        if m != mode:
            continue
        err = abs(prec - (mass + shift)) / (mass + shift) * 1e6
        if err <= PPM and (best is None or err < best[1]):
            best = (name, err)
    return best


def ce_values(text):
    nums = [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)', text or '')]
    if nums and ('ev' in text.lower() or re.fullmatch(r'[\d.,\s-]+', text.strip())):
        return text.strip(), nums, 'eV', False
    return ','.join(f'{x:g}' for x in CE_IMPUTED), list(CE_IMPUTED), 'eV', True


def splitmix64(x):
    x = x.astype(U64) + U64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> U64(30))) * U64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> U64(27))) * U64(0x94D049BB133111EB)
    return x ^ (x >> U64(31))


def peak_hashes(row, mz, n_rows):
    """Order-free 64-bit hash of each row's SET of TOL-rounded m/z (row = spectrum index per peak); 0 for rows without peaks."""
    key = np.unique((row.astype(np.int64) << 32) | np.rint(np.asarray(mz) / TOL).astype(np.int64))
    out = np.zeros(n_rows, U64)
    if len(key):
        r = key >> 32
        start = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        cnt = np.diff(np.r_[start, len(key)]).astype(U64)
        out[r[start]] = splitmix64(np.add.reduceat(splitmix64(key & 0xFFFFFFFF), start) + cnt * U64(0x2545F4914F6CDD1D))
    return out


def unit_sqrt(it):
    w = np.sqrt(it)
    return w / np.sqrt((w * w).sum())


def matched_cosine(qmz, qw, lmz, lw, lsid, n):
    """Cosine of one query (sorted m/z, unit weights) against n library spectra given as flat peaks (lsid = spectrum index)."""
    if n == 0 or len(lmz) == 0:
        return np.zeros(n)
    j = np.searchsorted(qmz, lmz)
    jl, jr = np.maximum(j - 1, 0), np.minimum(j, len(qmz) - 1)
    dl, dr = np.abs(lmz - qmz[jl]), np.abs(qmz[jr] - lmz)
    jn = np.where(dl <= dr, jl, jr)
    ok = np.minimum(dl, dr) <= TOL + 1e-9
    sid, jn = lsid[ok], jn[ok]
    if not len(sid):
        return np.zeros(n)
    prod = lw[ok] * qw[jn]
    pair = sid.astype(np.int64) * len(qmz) + jn
    o = np.lexsort((-prod, pair))        # one-to-one: each (spectrum, query peak) keeps its best library peak
    pair, prod, sid = pair[o], prod[o], sid[o]
    first = np.r_[True, pair[1:] != pair[:-1]]
    return np.minimum(np.bincount(sid[first], weights=prod[first], minlength=n), 1.0)


class Store:
    """Unit-sqrt peaks of a sorted subset of train rows (global row ids)."""
    def __init__(self, rows, off, mz, w):
        self.rows, self.off, self.mz, self.w = rows, off, mz, w

    def gather(self, rows):
        loc = np.searchsorted(self.rows, rows)
        assert len(rows) == 0 or np.array_equal(self.rows[loc], rows)
        s = self.off[loc]; ln = self.off[loc + 1] - s
        sid = np.repeat(np.arange(len(rows)), ln)
        pos = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln) + np.repeat(s, ln)
        return self.mz[pos], self.w[pos], sid


def scan_train(path, prec, need, with_hash, batch_size=65536):
    """Stream train peaks once: peak-list hash of every row (if with_hash) and unit sqrt peaks of the rows in `need`."""
    pf = pq.ParquetFile(path)
    hashes = np.zeros(pf.metadata.num_rows, U64) if with_hash else None
    offs, mzs, ws, base, total = [np.zeros(1, np.int64)], [], [], 0, 0
    for b in pf.iter_batches(batch_size=batch_size, columns=['ms2_mzs', 'ms2_normalized_intensities']):
        n = b.num_rows
        ln = pc.fill_null(pc.list_value_length(b.column(0)), 0).to_numpy().astype(np.int64)
        mz = b.column(0).flatten().to_numpy(); it = b.column(1).flatten().to_numpy()
        assert len(mz) == len(it) == ln.sum()
        row = np.repeat(np.arange(n), ln)
        k = (it > 0) & (mz <= prec[base + row] + 2.0)
        if with_hash:
            hashes[base:base + n] = peak_hashes(row[k], mz[k], n)
        lo, hi = np.searchsorted(need, [base, base + n])
        sel = need[lo:hi] - base
        if len(sel):
            rm = np.zeros(n, bool); rm[sel] = True
            take = k & rm[row]
            r, m, w = row[take], mz[take], np.sqrt(it[take])
            o = np.lexsort((m, r)); r, m, w = r[o], m[o], w[o]
            w = w / np.sqrt(np.bincount(r, weights=w * w, minlength=n))[r]
            cnt = np.bincount(r, minlength=n)[sel]
            offs.append(total + np.cumsum(cnt)); total += int(cnt.sum())
            mzs.append(m); ws.append(w.astype(np.float32))
        base += n
    return hashes, Store(need, np.concatenate(offs), np.concatenate(mzs) if mzs else np.zeros(0),
                         np.concatenate(ws) if ws else np.zeros(0, np.float32))


def spec_rank(instrument, adduct):
    """0 TOF + [M+H]+/[M-H]-, 1 [M+H]+/[M-H]-, 2 TOF, 3 other: preference order towards the hidden test (timsTOF, mostly H)."""
    tof, h = 'tof' in instrument.lower(), adduct in PREF_ADDUCTS
    return 0 if tof and h else 1 if h else 2 if tof else 3


def dist(x):
    x = np.asarray(x, float)
    if not len(x):
        return {}
    hist = np.histogram(x, bins=np.round(np.arange(0, 1.0001, 0.1), 1))[0]
    return dict(n=len(x), quantiles={str(q): round(float(np.quantile(x, q)), 4) for q in (0, .1, .25, .5, .75, .9, 1)},
                hist_0_to_1_step_0_1=hist.tolist(), **{f'ge_{t}': int((x >= t).sum()) for t in THRESHOLDS})


def peak_rss_mb():
    scale = 1 if sys.platform == 'darwin' else 1024   # ru_maxrss: bytes on macOS, KiB on Linux
    return {w: round(resource.getrusage(getattr(resource, f'RUSAGE_{w.upper()}')).ru_maxrss * scale / 2**20, 1)
            for w in ('self', 'children')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gnps', required=True); ap.add_argument('--massbank', required=True)
    ap.add_argument('--train', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n-known', type=int, default=150); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-spec', type=int, default=9); ap.add_argument('--min-peaks', type=int, default=3)
    ap.add_argument('--ambig-cos', type=float, default=0.85); ap.add_argument('--copy-cos', type=float, default=0.95)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    t_start = time.time()
    log = lambda msg: print(f'[{time.time() - t_start:7.1f}s] {msg}', flush=True)

    # ---- train metadata (peaks are streamed later) ----
    tr = pq.read_table(a.train, columns=['normalized_smiles', 'inchikey14', 'ionization_mode', 'precursor_mz'])
    tcode = pc.dictionary_encode(tr.column('normalized_smiles').combine_chunks())
    tsmiles = tcode.dictionary.to_pylist(); tcode = tcode.indices.to_numpy()
    tprec = tr.column('precursor_mz').to_numpy()
    tpos = pc.equal(tr.column('ionization_mode'), 'positive').to_numpy()
    train_plain = set(pc.unique(tr.column('inchikey14')).to_pylist())
    del tr
    log(f'train: {len(tprec)} spectra, {len(tsmiles)} SMILES')

    # ---- external records ----
    recs = [r for src in (read_gnps(a.gnps), read_massbank(a.massbank)) for r in src if len(r['mz']) >= a.min_peaks]
    log(f'external records with >= {a.min_peaks} peaks: {len(recs)}')
    smiles = sorted({r['smiles'] for r in recs} | set(tsmiles))
    with Pool(a.workers) as p:
        desc = dict(p.imap_unordered(describe, smiles, chunksize=2000))
    log(f'described {len(smiles)} SMILES')
    keys = sorted({d[1] for d in desc.values() if d})
    kid = {k: i for i, k in enumerate(keys)}
    tkey = np.array([kid[desc[s][1]] if desc.get(s) else -1 for s in tsmiles], np.int64)[tcode]
    train_taut = {keys[i] for i in np.unique(tkey[tkey >= 0])}

    # ---- common filters, split into part A (novel) / B (known) candidates ----
    counts = defaultdict(int)
    rows, peaks = [], []
    for r in recs:
        d = desc.get(r['smiles'])
        if d is None:
            counts['unparsable_or_multi_or_charged'] += 1; continue
        plain, taut, can, mass = d
        if not 100 <= mass <= 1300:
            counts['mass_range'] += 1; continue
        hit = assign_adduct(mass, r['prec'], r['mode'])
        if hit is None:
            counts['no_test_adduct'] += 1; continue
        mz, it = r['mz'], r['it']
        keep = (it > 0) & (mz <= r['prec'] + 2.0)
        if keep.sum() < a.min_peaks:
            counts['too_few_peaks'] += 1; continue
        mz, it = mz[keep], it[keep]
        o = np.argsort(mz, kind='stable'); mz, it = mz[o], it[o]
        known = taut in train_taut
        if not known and plain in train_plain:
            counts['plain_key_in_train_metric_key_not'] += 1; continue
        ce_orig, ce_ev, ce_units, imputed = ce_values(r['ce'])
        rows.append(dict(part='known' if known else 'novel', key=taut, plain=plain, smiles=can, source=r['source'], record=r['record'],
                         instrument=r['instrument'], ce_imputed=imputed, adduct=hit[0], ppm=hit[1], ionization_mode=r['mode'],
                         precursor_mz=r['prec'], n_peaks=len(mz), base_peak_intensity=float(it.max()),
                         collision_energy_orig=ce_orig, collision_energy_ev=ce_ev, collision_energy_orig_units=ce_units))
        peaks.append((mz, it))
    del recs
    df = pd.DataFrame(rows); del rows
    df['rank'] = [spec_rank(i, ad) for i, ad in zip(df.instrument, df.adduct)]
    df['hash'] = peak_hashes(np.repeat(np.arange(len(peaks)), df.n_peaks.to_numpy()),
                             np.concatenate([m for m, _ in peaks]), len(peaks))
    for c in ('best_cos_diff', 'best_cos_same'):
        df[c] = np.nan
    df['best_diff_row'] = -1
    df['n_window'] = 0
    df['n_same'] = 0
    counts['candidate_spectra'] = len(df)
    log(f'candidates: {df.part.value_counts().to_dict()}')

    # ---- train precursor index per ion mode ----
    index = {}
    for mode, mask in (('positive', tpos), ('negative', ~tpos)):
        r = np.flatnonzero(mask); r = r[np.argsort(tprec[r], kind='stable')]
        index[mode] = (r, tprec[r])

    def window(mode, p):
        r, pr = index[mode]
        lo = np.searchsorted(pr, p * (1 - PPM * 1e-6), 'left'); hi = np.searchsorted(pr, p * (1 + PPM * 1e-6), 'right')
        return np.sort(r[lo:hi])

    A = df.index[df.part == 'novel'].to_numpy()
    B = df[df.part == 'known']
    bstruct = B.groupby('key')['rank'].min()
    perm = pd.Series(np.random.RandomState(a.seed).permutation(len(bstruct)), index=bstruct.index)  # keys sorted by groupby
    border = pd.DataFrame({'tier': bstruct, 'perm': perm}).sort_values(['tier', 'perm']).index.tolist()
    bspec = {k: g.index.to_numpy() for k, g in B.groupby('key')}
    struct_rows = {}   # B key -> train rows of that structure (filled per examined batch)

    def same_rows(key, mode):
        r = struct_rows[key]
        return r[tpos[r] == (mode == 'positive')]

    def cosine_rows(i, rows_):
        mz, it = peaks[i]
        lmz, lw, lsid = store.gather(rows_)
        return matched_cosine(mz, unit_sqrt(it), lmz, lw.astype(float), lsid, len(rows_))

    def best_other(i, w):
        c = cosine_rows(i, w)
        df.at[i, 'n_window'] = len(w)
        df.at[i, 'best_cos_diff'] = c.max() if len(w) else 0.0
        if len(w):
            df.at[i, 'best_diff_row'] = w[int(np.argmax(c))]

    # first scan: hash every train spectrum, load the precursor windows of all A spectra
    a_need = np.unique(np.concatenate([window(df.ionization_mode[i], df.precursor_mz[i]) for i in A] + [np.zeros(0, np.int64)]))
    thashes, store = scan_train(a.train, tprec, a_need, with_hash=True)
    thashes = np.unique(thashes)
    df['exact_dup_train'] = np.isin(df.hash.to_numpy(), thashes)
    log(f'train scan: hashed {len(tprec)} spectra, loaded {len(a_need)} rows')
    for i in A:   # part A: every train spectrum differs from the truth (key absent from train)
        best_other(i, window(df.ionization_mode[i], df.precursor_mz[i]))
    log(f'part A cosines: {len(A)} spectra')

    # part B: structures in preference order; those whose spectra are all exact train duplicates can never be accepted.
    # Batches of growing size, each loading the same-structure train rows and the windows of its non-duplicate spectra.
    exact = df.exact_dup_train.to_numpy()
    bspec = {k: v[~exact[v]] for k, v in bspec.items()}
    border = [k for k in border if len(bspec[k])]
    start, size, accepted = 0, 4 * a.n_known, []
    while len(accepted) < a.n_known and start < len(border):
        bkeys = border[start:start + size]
        tkey_rows = np.flatnonzero(np.isin(tkey, [kid[k] for k in bkeys]))
        struct_rows.clear()
        struct_rows.update({k: tkey_rows[tkey[tkey_rows] == kid[k]] for k in bkeys})
        need = np.unique(np.concatenate([tkey_rows] + [window(df.ionization_mode[i], df.precursor_mz[i])
                                                       for k in bkeys for i in bspec[k]]))
        del store
        _, store = scan_train(a.train, tprec, need, with_hash=False)
        log(f'train scan for known structures {start}..{start + len(bkeys)}: loaded {len(need)} rows')
        for k in bkeys:
            ok = 0
            for i in bspec[k]:
                mode = df.ionization_mode[i]
                s = same_rows(k, mode)
                df.at[i, 'n_same'] = len(s)
                df.at[i, 'best_cos_same'] = cosine_rows(i, s).max() if len(s) else 0.0
                w = window(mode, df.precursor_mz[i])
                best_other(i, w[tkey[w] != kid[k]])
                ok += df.best_cos_same[i] < a.copy_cos
            start += 1
            if ok:
                accepted.append(k)
                if len(accepted) == a.n_known:
                    break
        size *= 2
    examined = start
    log(f'part B: examined {examined} structures with a non-duplicate spectrum, accepted {len(accepted)}')
    del store

    # ---- fates ----
    novel, known = df.part == 'novel', df.part == 'known'
    df['examined'] = novel | df.key.isin(set(border[:examined]))
    df['fate'] = ''
    df.loc[df.exact_dup_train, 'fate'] = 'exact_dup_train'
    df.loc[(df.fate == '') & ~df.examined, 'fate'] = 'not_examined'
    df.loc[novel & (df.fate == '') & (df.best_cos_diff >= a.ambig_cos), 'fate'] = 'ambiguous_dup_train'
    df.loc[known & (df.fate == '') & (df.best_cos_same >= a.copy_cos), 'fate'] = 'copy_same_structure'
    df.loc[known & (df.fate == '') & ~df.key.isin(set(accepted)), 'fate'] = 'not_sampled'
    df = df.sort_values(['part', 'key', 'rank', 'source', 'record'], kind='stable')
    live = df.fate == ''
    idup = live & df[live].duplicated(['key', 'hash']).reindex(df.index, fill_value=False)   # earlier preferred copy wins
    df.loc[idup, 'fate'] = 'internal_dup'
    live = df.fate == ''
    cap = live & (df[live].groupby('key').cumcount() >= a.max_spec).reindex(df.index, fill_value=False)
    df.loc[cap, 'fate'] = 'max_spec'
    df.loc[df.fate == '', 'fate'] = 'kept'

    # ---- outputs ----
    q = df[df.fate == 'kept'].copy()
    q['molecule_id'] = [('x_' if p == 'novel' else 'k_') + hashlib.md5(k.encode()).hexdigest()[:8] for p, k in zip(q.part, q.key)]
    q = q.sort_values(['part', 'molecule_id', 'rank', 'source', 'record'], ascending=[False, True, True, True, True], kind='stable')
    n_a = int((q.part == 'novel').sum())
    q['spectrum_id'] = [f'xs_{i:06d}' for i in range(n_a)] + [f'ks_{i:06d}' for i in range(len(q) - n_a)]
    q['ms2_mzs'] = [peaks[i][0].tolist() for i in q.index]
    q['ms2_normalized_intensities'] = [(peaks[i][1] / peaks[i][1].max()).tolist() for i in q.index]
    q['instrument_type'] = 'timsTOF'
    os.makedirs(os.path.join(a.out, 'kaggle'), exist_ok=True)
    test = q[TEST_COLS].reset_index(drop=True)
    test.to_parquet(os.path.join(a.out, 'test.parquet'), index=False)
    lab = q.groupby('molecule_id', sort=False).agg(smiles=('smiles', 'first'), inchikey14=('key', 'first'),
                                                   source=('source', 'first'), n_spectra=('spectrum_id', 'size'),
                                                   ce_imputed=('ce_imputed', 'all'), part=('part', 'first')).reset_index()
    lab.to_csv(os.path.join(a.out, 'val_labels.csv'), index=False)
    pd.DataFrame({'molecule_id': lab.molecule_id, 'smiles': 'CCO'}).to_csv(os.path.join(a.out, 'sample_submission.csv'), index=False)
    audit = df.drop(columns=['hash']).assign(hash=[f'{int(h):016x}' for h in df.hash])
    audit['best_diff_train_smiles'] = [tsmiles[tcode[r]] if r >= 0 else None for r in audit.best_diff_row]
    audit.to_parquet(os.path.join(a.out, 'audit.parquet'), index=False)

    # ---- duplicate-threshold table: what each cosine threshold drops among the spectra that are not exact train duplicates
    #      (structures_lost = structures whose every non-duplicate spectrum is >= t, so the structure leaves the set) ----
    def thr_table(sub, col):
        nd = sub[~sub.exact_dup_train]
        out = dict(exact_dup_spectra=int(sub.exact_dup_train.sum()), structures=int(sub.key.nunique()),
                   structures_only_exact_dups=int(sub.exact_dup_train.groupby(sub.key).all().sum()),
                   nondup_spectra=len(nd), nondup_structures=int(nd.key.nunique()))
        for t in THRESHOLDS:
            hit = nd[col] >= t
            g = hit.groupby(nd.key)
            out[str(t)] = dict(spectra=int(hit.sum()), structures_any=int(g.any().sum()), structures_lost=int(g.all().sum()),
                               spectra_by_source={s: int(hit[nd.source == s].sum()) for s in sorted(nd.source.unique())})
        return out
    ex = df[df.examined]
    table = {'novel_vs_other_structures (all part-A candidates)': thr_table(ex[ex.part == 'novel'], 'best_cos_diff'),
             'known_vs_same_structure (examined part-B structures)': thr_table(ex[ex.part == 'known'], 'best_cos_same'),
             'known_vs_other_structures (examined part-B structures)': thr_table(ex[ex.part == 'known'], 'best_cos_diff')}

    # ---- sanity checks, recomputed from the written test.parquet ----
    lens = test.ms2_mzs.map(len).to_numpy()
    out_hash = peak_hashes(np.repeat(np.arange(len(test)), lens), np.concatenate(test.ms2_mzs.to_numpy()), len(test))
    lab_a, lab_b = lab[lab.part == 'novel'], lab[lab.part == 'known']
    sanity = dict(novel_metric_keys_in_train=int(lab_a.inchikey14.isin(train_taut).sum()),
                  novel_plain_keys_in_train=int(q.plain[q.part == 'novel'].isin(train_plain).sum()),
                  known_metric_keys_not_in_train=int((~lab_b.inchikey14.isin(train_taut)).sum()),
                  spectra_exact_dup_of_train=int(np.isin(out_hash, thashes).sum()),
                  duplicate_molecule_ids=int(lab.molecule_id.duplicated().sum()),
                  best_cos=dict(novel_vs_other_structures=dist(q.best_cos_diff[q.part == 'novel']),
                                known_vs_same_structure=dist(q.best_cos_same[q.part == 'known']),
                                known_vs_other_structures=dist(q.best_cos_diff[q.part == 'known'])))

    def part_summary(p):
        s, l = q[q.part == p], lab[lab.part == p]
        return dict(molecules=len(l), spectra=len(s), by_source=l.source.value_counts().to_dict(),
                    spectra_by_source=s.source.value_counts().to_dict(), adducts=s.adduct.value_counts().to_dict(),
                    ionization_modes=s.ionization_mode.value_counts().to_dict(),
                    instruments=s.instrument.value_counts().head(15).to_dict(),
                    tof_spectra=int(s.instrument.str.lower().str.contains('tof').sum()),
                    ce_imputed_spectra=int(s.ce_imputed.sum()), spectra_per_molecule=l.n_spectra.describe().round(2).to_dict())

    def part_counts(p):
        s = df[df.part == p]
        c = dict(candidate_spectra=len(s), candidate_structures=int(s.key.nunique()))
        if p == 'known':
            c.update(structures_only_exact_dups=int(s.exact_dup_train.groupby(s.key).all().sum()),
                     examined_structures=examined, examined_spectra=int((s.examined & ~s.exact_dup_train).sum()))
        c.update({f'fate_{f}': int(n) for f, n in s.fate.value_counts().items()})
        c.update(kept_structures=int(s.key[s.fate == 'kept'].nunique()))
        return c

    manifest = dict(counts=dict(common=dict(counts), novel=part_counts('novel'), known=part_counts('known')),
                    molecules=len(lab), spectra=len(test), parts=dict(novel=part_summary('novel'), known=part_summary('known')),
                    dup_threshold_table=table, sanity=sanity,
                    params=dict(seed=a.seed, n_known=a.n_known, max_spec=a.max_spec, min_peaks=a.min_peaks, ppm=PPM, tol_da=TOL,
                                ambig_cos=a.ambig_cos, copy_cos=a.copy_cos, thresholds=list(THRESHOLDS)),
                    inputs={k: os.path.basename(v) for k, v in dict(gnps=a.gnps, massbank=a.massbank, train=a.train).items()},
                    runtime_s=round(time.time() - t_start, 1), peak_rss_mb=peak_rss_mb())
    with open(os.path.join(a.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    test.to_parquet(os.path.join(a.out, 'kaggle', 'ext_test.parquet'), index=False)
    lab.to_csv(os.path.join(a.out, 'kaggle', 'ext_val_labels.csv'), index=False)
    with open(os.path.join(a.out, 'kaggle', 'ext_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
