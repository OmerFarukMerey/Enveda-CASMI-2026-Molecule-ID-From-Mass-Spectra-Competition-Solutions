"""Spectral analog library from train.parquet: [M+H]+, [M-H]- and [M+Na]+ spectra, top 32 peaks at or below the
precursor + 1 Da, sqrt(intensity / max), up to 3 spectra per structure and adduct (the ones with the most peaks).
usage: build_lib.py TRAIN_PARQUET OUT_DIR
Writes lib_peaks.npz (pk_mz, pk_it, pk_n, prec, grp, sidx, src) and lib_struct.parquet (key, smiles, formula)."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

GROUPS = ['[M+H]+', '[M-H]-', '[M+Na]+']
NPK, PER = 32, 3
src_path, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
f = pq.ParquetFile(src_path)
cols = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'molecular_formula', 'adduct', 'precursor_mz', 'ms2_mzs',
        'ms2_normalized_intensities']
LIBS = {}
keys, smiles, formulas = {}, [], []
recs = []          # (sidx, grp, npk, prec, src, mz[32], it[32])
for rg in range(f.num_row_groups):
    t = f.read_row_group(rg, columns=cols)
    ad = t.column('adduct').to_pylist()
    keep = np.array([a in GROUPS for a in ad])
    if not keep.any():
        continue
    t = t.filter(keep)
    ad = t.column('adduct').to_pylist(); ik = t.column('inchikey14').to_pylist(); sm = t.column('normalized_smiles').to_pylist()
    fo = t.column('molecular_formula').to_pylist(); lb = t.column('ingest_lib').to_pylist()
    pr = t.column('precursor_mz').to_numpy()
    mzc, itc = t.column('ms2_mzs').combine_chunks(), t.column('ms2_normalized_intensities').combine_chunks()
    mzo, ito = mzc.offsets.to_numpy(), itc.offsets.to_numpy()
    mzv, itv = mzc.values.to_numpy(zero_copy_only=False), itc.values.to_numpy(zero_copy_only=False)
    for i in range(len(ad)):
        m = mzv[mzo[i]:mzo[i + 1]]; it = itv[ito[i]:ito[i + 1]]
        if len(m) != len(it) or not len(m) or not np.isfinite(pr[i]):
            continue
        ok = np.isfinite(m) & np.isfinite(it) & (it > 0) & (m <= pr[i] + 1.0)
        m, it = m[ok], it[ok]
        if len(m) < 3:
            continue
        o = np.argsort(-it, kind='stable')[:NPK]; m, it = m[o], np.sqrt(it[o] / it[o].max())
        k = ik[i]
        if k not in keys:
            keys[k] = len(smiles); smiles.append(sm[i]); formulas.append(fo[i])
        s = LIBS.setdefault(lb[i], len(LIBS))
        mzp = np.zeros(NPK, np.float64); itp = np.zeros(NPK, np.float32); mzp[:len(m)] = m; itp[:len(m)] = it
        recs.append((keys[k], GROUPS.index(ad[i]), len(m), float(pr[i]), s, mzp, itp))
    print(f'row group {rg}: {len(recs):,} spectra, {len(smiles):,} structures, {time.time() - t0:.0f}s', flush=True)
    del t, mzc, itc, mzv, itv

sidx = np.array([r[0] for r in recs], np.int32); grp = np.array([r[1] for r in recs], np.int8)
npk = np.array([r[2] for r in recs], np.int16)
# up to PER spectra per (structure, group), most peaks first
order = np.lexsort((-npk, grp, sidx))
pair = sidx[order].astype(np.int64) * 4 + grp[order]
first = np.r_[0, np.flatnonzero(np.diff(pair)) + 1]
rank = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
sel = np.sort(order[rank < PER])
np.savez(out / 'lib_peaks.npz', pk_mz=np.stack([recs[i][5] for i in sel]), pk_it=np.stack([recs[i][6] for i in sel]),
         pk_n=npk[sel], prec=np.array([recs[i][3] for i in sel]), grp=grp[sel], sidx=sidx[sel],
         src=np.array([recs[i][4] for i in sel], np.int8))
pd.DataFrame(dict(key=list(keys), smiles=smiles, formula=formulas)).to_parquet(out / 'lib_struct.parquet', index=False)
pd.Series(LIBS).to_json(out / 'lib_sources.json')
print('kept', len(sel), 'of', len(recs), 'spectra;', len(smiles), 'structures;', f'{time.time() - t0:.0f}s')
