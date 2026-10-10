"""Write small golden fixtures from the stored intermediate outputs of the current chain, so that a rewrite can be verified.

Reads only DATA (tables and caches the current pipeline produced: the spectrum cache, the pool tables, the fragment table)
and train.parquet; no third-party code is imported. Output: $CASMI_DATA/clean/golden/ (about 15 MB, not in git: the
fixtures contain competition-derived rows). Used by tests/test_clean_contract.py, which skips when they are absent.

usage: python clean/golden.py [--work DIR] [--frag DIR] [--train F]      (resumable: existing fixtures are kept)
"""
import argparse, json, os

import numpy as np
import pandas as pd

DATA = os.environ.get('CASMI_DATA', 'data')
OUT = os.path.join(DATA, 'clean', 'golden')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.path.join(DATA, 'directions', 'run', 'work'))       # spectrum cache + pool tables
    ap.add_argument('--frag', default=os.path.join(DATA, 'ranker_study', 'ds', 'v2pool'))    # pool_frag_off / pool_frag_mass
    ap.add_argument('--train', default=os.path.join(DATA, 'input', 'competitions', 'enveda-CASMI26-molecule-id-mass-spectra', 'train.parquet'))
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); rng = np.random.default_rng(0)
    f = os.path.join(OUT, 'adducts.json')
    if not os.path.exists(f):
        sm = pd.read_parquet(os.path.join(a.work, 'spec_meta.parquet'), columns=['adduct', 'precursor_mz', 'nm'])
        g = sm.groupby('adduct', sort=True).head(3)
        json.dump([dict(adduct=x, mz=float(m), nm=(None if np.isnan(n) else float(n))) for x, m, n in zip(g.adduct, g.precursor_mz, g.nm)],
                  open(f, 'w'), indent=0)
    f = os.path.join(OUT, 'spectra.npz')
    if not os.path.exists(f):
        import pyarrow.parquet as pq
        t = pq.ParquetFile(a.train).read_row_group(0, columns=['precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']).to_pandas()
        off = np.load(os.path.join(a.work, 'spec_off.npy'), mmap_mode='r'); cm = np.load(os.path.join(a.work, 'spec_mz.npy'), mmap_mode='r')
        ci = np.load(os.path.join(a.work, 'spec_it.npy'), mmap_mode='r')
        rows = np.sort(rng.choice(len(t), 400, replace=False)); d = dict(rows=rows, prec=t.precursor_mz.values[rows])
        for j, r in enumerate(rows):
            d[f'mz{j}'] = np.asarray(t.ms2_mzs.values[r], np.float64); d[f'it{j}'] = np.asarray(t.ms2_normalized_intensities.values[r], np.float64)
            d[f'cmz{j}'] = np.array(cm[off[r]:off[r + 1]]); d[f'cit{j}'] = np.array(ci[off[r]:off[r + 1]])
        np.savez_compressed(f, **d)
    f = os.path.join(OUT, 'pool_sample.parquet')
    if not os.path.exists(f):
        pm = pd.read_parquet(os.path.join(a.work, 'pool_meta.parquet')); ix = np.sort(rng.choice(len(pm), 3000, replace=False))
        pm.iloc[ix].to_parquet(f, index=False)
        np.save(os.path.join(OUT, 'pool_sample_fp.npy'), np.load(os.path.join(a.work, 'pool_fp.npy'), mmap_mode='r')[ix])
        np.save(os.path.join(OUT, 'fp_bits_ref.npy'), np.load(os.path.join(a.work, 'fp_bits.npy')))
    f = os.path.join(OUT, 'frags.npz')
    if not os.path.exists(f) and os.path.exists(os.path.join(a.frag, 'pool_frag_off.npy')):
        pm = pd.read_parquet(os.path.join(a.work, 'pool_meta.parquet'), columns=['smiles', 'n_heavy'])
        fo = np.load(os.path.join(a.frag, 'pool_frag_off.npy'), mmap_mode='r'); fm = np.load(os.path.join(a.frag, 'pool_frag_mass.npy'), mmap_mode='r')
        ix = np.sort(rng.choice(np.flatnonzero(pm.n_heavy.values <= 45), 400, replace=False))
        d = dict(pid=ix, smiles=pm.smiles.values[ix].astype(str))
        for j, p in enumerate(ix):
            d[f'f{j}'] = np.array(fm[fo[p]:fo[p + 1]])
        np.savez_compressed(f, **d)
    f = os.path.join(OUT, 'train_structs_head.parquet')
    if not os.path.exists(f):
        pd.read_parquet(os.path.join(a.work, 'train_structs.parquet')).head(2000).to_parquet(f, index=False)
    print('golden fixtures in', OUT, sorted(os.listdir(OUT)))


if __name__ == '__main__':
    main()
