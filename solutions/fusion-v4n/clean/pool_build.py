"""Build our own candidate-pool tables (the schema the v1 engine reads) from train.parquet and an upstream COCONUT release only.

Written by us against the DATA CONTRACT in clean/SPEC.md (section "Pool tables"); no code of the `casmi` package is
imported or copied. Inputs (see clean/README.md for licences):
  train.parquet                         competition data (columns inchikey14, normalized_smiles, molecular_formula,
                                        ingest_lib, instrument_type)
  coconut_csv-<mm>-<yyyy>.zip           COCONUT release from https://coconut.naturalproducts.net/download (CC0)
Stages (every stage resumes from its files; nothing is recomputed that is on disk):
  sources   -> <out>/src_train.parquet, <out>/src_coconut.parquet           (unique input structures, 1 to 2 min)
  structs   -> <out>/chunks/<src>_<i>.parquet + .npy                         (standardise, key, raw fingerprint; CPU hours)
  assemble  -> <out>/tables/{pool_meta.parquet, pool_fp.npy, fp_bits.npy, train_structs.parquet, train_fp_sel.npy,
               stats.json}
  verify    compares <out>/tables with an existing pool directory (keys, masses, formulas, fingerprints) -> verify.json

usage: python clean/pool_build.py <stage> [--out DIR] [--train F] [--coconut ZIP] [--workers 4] [--chunk 4000]
                                           [--ref DIR] [--bits own|ref]
Workers run at below-normal priority in a fixed-size pool. RDKit must be 2026.03.3 (the metric's version).
"""
import argparse, glob, json, os, sys, time, zipfile

import numpy as np
import pandas as pd

RDKIT_PIN = '2026.03.3'
LIBS = ['enveda-180', 'pluskal_ms2', 'riken', 'gnps', 'massbank', 'mona', 'spectraverse', 'msdial', 'drug_plus',
        'enveda-np-examples', 'masaryk']                       # bit i of libmask = LIBS[i] (order of the data page)
# raw fingerprint layout: (name, bits); concatenated in this order -> 16,551 bits
FP_BLOCKS = [('ecfp4', 4096), ('ecfp6', 4096), ('fcfp4', 2048), ('rdk6', 2048), ('ap', 2048), ('tt', 2048), ('maccs', 167)]
FP_RAW = sum(n for _, n in FP_BLOCKS)
BIT_LO, BIT_HI = 0.005, 0.995                                 # a bit is kept if its frequency over the pool is inside
DEF_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                       'scratch', 'clean', 'pool')
DATA = os.environ.get('CASMI_DATA', 'data')
DEF_TRAIN = os.path.join(DATA, 'input', 'competitions', 'enveda-CASMI26-molecule-id-mass-spectra', 'train.parquet')
DEF_COCO = os.path.join(DATA, 'extra_db', 'raw', 'coconut_csv-10-2026.zip')

_R = {}


def _low_priority():
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    else:
        os.nice(10)


def _rd():
    if not _R:
        import rdkit
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator as G, rdMolDescriptors
        from rdkit.Chem.MolStandardize import rdMolStandardize as S
        assert rdkit.__version__ == RDKIT_PIN, f'RDKit {rdkit.__version__}, need {RDKIT_PIN}'
        RDLogger.DisableLog('rdApp.*')
        _R.update(Chem=Chem, D=Descriptors, MD=rdMolDescriptors, MACCS=MACCSkeys, taut=S.TautomerEnumerator(),
                  lfc=S.LargestFragmentChooser(), unc=S.Uncharger(),
                  gens=[G.GetMorganGenerator(radius=2, fpSize=4096), G.GetMorganGenerator(radius=3, fpSize=4096),
                        G.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen()),
                        G.GetRDKitFPGenerator(fpSize=2048, maxPath=6), G.GetAtomPairGenerator(fpSize=2048),
                        G.GetTopologicalTorsionGenerator(fpSize=2048)])
    return _R


def standardise(smiles):
    """(stereo-free canonical SMILES of the largest fragment after neutralisation, net formal charge) or None."""
    r = _rd(); Chem = r['Chem']
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = r['unc'].uncharge(r['lfc'].choose(mol))
    q = Chem.GetFormalCharge(mol)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol), int(q)


def record(smiles):
    """Fields of one standardised structure: formula, exact mass, heavy atoms, metric key, plain key, packed raw fingerprint."""
    r = _rd(); Chem = r['Chem']
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = np.concatenate([g.GetFingerprintAsNumPy(mol).astype(np.uint8) for g in r['gens']]
                        + [np.array(list(r['MACCS'].GenMACCSKeys(mol)), np.uint8)])
    assert len(fp) == FP_RAW
    raw = Chem.MolToInchiKey(mol)
    key = Chem.MolToInchiKey(r['taut'].Canonicalize(mol))      # the metric's key: tautomer-canonical InChIKey, first block
    if not raw or not key:
        return None
    return dict(smiles=smiles, formula=r['MD'].CalcMolFormula(mol), mass=float(r['D'].ExactMolWt(mol)),
                n_heavy=int(mol.GetNumHeavyAtoms()), key=key[:14], rawkey=raw[:14]), np.packbits(fp)


def _chunk_job(job):
    out, src, i, rows = job
    f = os.path.join(out, 'chunks', f'{src}_{i:05d}')
    if os.path.exists(f + '.parquet'):
        return src, i, 0, 0.0
    t0 = time.time(); recs, fps = [], []
    for idx, smi in rows:
        try:
            st = standardise(smi)
            rc = record(st[0]) if st else None
        except Exception:
            st = rc = None
        if rc is None:
            recs.append(dict(idx=idx, ok=False)); continue
        d, fp = rc; d.update(idx=idx, ok=True, charge=st[1]); recs.append(d); fps.append(fp)
    np.save(f + '.npy.tmp.npy', np.stack(fps) if fps else np.zeros((0, (FP_RAW + 7) // 8), np.uint8))
    os.replace(f + '.npy.tmp.npy', f + '.npy')
    pd.DataFrame(recs).to_parquet(f + '.tmp'); os.replace(f + '.tmp', f + '.parquet')
    return src, i, len(rows), time.time() - t0


def stage_sources(a):
    import pyarrow.parquet as pq
    os.makedirs(a.out, exist_ok=True)
    ft, fc = os.path.join(a.out, 'src_train.parquet'), os.path.join(a.out, 'src_coconut.parquet')
    if not os.path.exists(ft):
        pf = pq.ParquetFile(a.train); first, agg = {}, {}; n = 0
        for rg in range(pf.metadata.num_row_groups):
            d = pf.read_row_group(rg, columns=['inchikey14', 'normalized_smiles', 'molecular_formula', 'ingest_lib',
                                               'instrument_type']).to_pandas()
            tims = d.instrument_type.astype(str).str.lower().str.contains('timstof').values
            for k, s, f, lib, t in zip(d.inchikey14.values, d.normalized_smiles.values, d.molecular_formula.values,
                                       d.ingest_lib.values, tims):
                if k not in first:
                    first[k] = (n, s, f); agg[k] = [0, 0, 0]
                g = agg[k]; g[0] += 1; g[1] += int(t); g[2] |= 1 << LIBS.index(lib); n += 1
        ks = sorted(first, key=lambda k: first[k][0])
        pd.DataFrame(dict(inchikey14=ks, smiles_in=[first[k][1] for k in ks], formula_given=[first[k][2] for k in ks],
                          n_spec=[agg[k][0] for k in ks], n_tims=[agg[k][1] for k in ks],
                          libmask=[agg[k][2] for k in ks])).to_parquet(ft + '.tmp')
        os.replace(ft + '.tmp', ft); print('train structures', len(ks), 'spectra', n, flush=True)
    if not os.path.exists(fc):
        z = zipfile.ZipFile(a.coconut)
        with z.open(z.infolist()[0]) as fh:
            d = pd.read_csv(fh, usecols=['identifier', 'canonical_smiles', 'standard_inchi_key', 'np_likeness'])
        n_all = len(d)
        d = d.rename(columns=dict(canonical_smiles='smiles_in')).dropna(subset=['smiles_in'])
        # stereoisomers share the first InChIKey block and become one stereo-free row anyway: keep the first of each
        d['ik14'] = d.standard_inchi_key.astype(str).str[:14]
        d = d.drop_duplicates('ik14').drop(columns=['standard_inchi_key']).reset_index(drop=True)
        print('coconut release rows', n_all, flush=True)
        d.to_parquet(fc + '.tmp'); os.replace(fc + '.tmp', fc); print('coconut rows', len(d), flush=True)


def stage_structs(a):
    from multiprocessing import Pool
    os.makedirs(os.path.join(a.out, 'chunks'), exist_ok=True)
    jobs = []
    for src in ('train', 'coconut'):
        s = pd.read_parquet(os.path.join(a.out, f'src_{src}.parquet'), columns=['smiles_in']).smiles_in.values
        for i in range(0, len(s), a.chunk):
            if not os.path.exists(os.path.join(a.out, 'chunks', f'{src}_{i // a.chunk:05d}.parquet')):
                jobs.append((a.out, src, i // a.chunk, list(zip(range(i, min(i + a.chunk, len(s))), s[i:i + a.chunk]))))
    print('chunks to do', len(jobs), flush=True); t0 = time.time(); done = 0
    with Pool(a.workers, initializer=_low_priority) as pool:           # fixed-size pool
        for src, i, n, dt in pool.imap_unordered(_chunk_job, jobs):
            done += 1
            print(f'{src} {i} n {n} {dt:.0f}s | {done}/{len(jobs)} wall {time.time() - t0:.0f}s', flush=True)
    print('STRUCTS DONE', flush=True)


def _load(out, src):
    fs = sorted(glob.glob(os.path.join(out, 'chunks', f'{src}_*.parquet')))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    fp = np.concatenate([np.load(f[:-8] + '.npy') for f in fs])
    ok = d[d.ok].reset_index(drop=True); assert len(ok) == len(fp)
    return ok, fp, len(d)


def stage_assemble(a):
    T = os.path.join(a.out, 'tables'); os.makedirs(T, exist_ok=True)
    st = pd.read_parquet(os.path.join(a.out, 'src_train.parquet')); co = pd.read_parquet(os.path.join(a.out, 'src_coconut.parquet'))
    tr, tfp, ntr = _load(a.out, 'train'); cc, cfp, ncc = _load(a.out, 'coconut')
    stats = dict(train_in=ntr, train_ok=len(tr), coconut_in=ncc, coconut_ok=len(cc),
                 train_charged=int((tr.charge != 0).sum()), coconut_charged=int((cc.charge != 0).sum()))
    # training structures: one row per inchikey14 of train.parquet, sid = order of first appearance among the kept rows
    tr = tr.join(st.drop(columns=['smiles_in']), on='idx')
    tr['sid'] = np.arange(len(tr))
    tr['row'] = np.arange(len(tr))
    cc = cc[cc.charge == 0].copy(); cc['np_likeness'] = co.np_likeness.values[cc.idx.values]; cc['row'] = cc.index.values
    # pool: one row per metric key. A training structure wins over a COCONUT one; inside a source the first row wins.
    t1 = tr.drop_duplicates('key'); c1 = cc.drop_duplicates('key')
    both = set(t1.key) & set(c1.key)
    pt = pd.DataFrame(dict(key=t1.key, smiles=t1.smiles, formula=t1.formula, mass=t1.mass, n_heavy=t1.n_heavy.astype(np.int16),
                           src=np.where(t1.key.isin(both), 3, 1).astype(np.int8), train_sid=t1.sid.astype(np.int64),
                           np_likeness=t1.key.map(dict(zip(c1.key, c1.np_likeness))), _s=0, _r=t1.row))
    c2 = c1[~c1.key.isin(both)]
    pc = pd.DataFrame(dict(key=c2.key, smiles=c2.smiles, formula=c2.formula, mass=c2.mass, n_heavy=c2.n_heavy.astype(np.int16),
                           src=np.int8(2), train_sid=np.int64(-1), np_likeness=c2.np_likeness, _s=1, _r=c2.row))
    pool = pd.concat([pt, pc], ignore_index=True).sort_values(['mass', 'key'], kind='stable').reset_index(drop=True)
    raw = np.empty((len(pool), tfp.shape[1]), np.uint8)
    mt = pool._s.values == 0; raw[mt] = tfp[pool._r.values[mt]]; raw[~mt] = cfp[pool._r.values[~mt]]
    freq = np.zeros(FP_RAW)
    for s in range(0, len(raw), 20000):
        freq += np.unpackbits(raw[s:s + 20000], axis=1)[:, :FP_RAW].sum(0)
    freq /= len(raw)
    bits = np.load(os.path.join(a.ref, 'fp_bits.npy')).astype(np.int32) if a.bits == 'ref' else \
        np.where((freq >= BIT_LO) & (freq <= BIT_HI))[0].astype(np.int32)

    def sel(x):
        out = np.empty((len(x), (len(bits) + 7) // 8), np.uint8)
        for s in range(0, len(x), 20000):
            out[s:s + 20000] = np.packbits(np.unpackbits(x[s:s + 20000], axis=1)[:, bits], axis=1)
        return out
    pool['pid'] = np.arange(len(pool), dtype=np.int64)
    pool.drop(columns=['_s', '_r']).to_parquet(os.path.join(T, 'pool_meta.parquet'), index=False)
    np.save(os.path.join(T, 'pool_fp.npy'), sel(raw)); np.save(os.path.join(T, 'fp_bits.npy'), bits)
    np.save(os.path.join(T, 'fp_bit_freq.npy'), freq.astype(np.float32))
    tr[['inchikey14', 'smiles', 'formula_given', 'n_spec', 'n_tims', 'libmask', 'formula', 'mass', 'n_heavy', 'key', 'rawkey',
        'sid']].to_parquet(os.path.join(T, 'train_structs.parquet'), index=False)
    np.save(os.path.join(T, 'train_fp_sel.npy'), sel(tfp))
    stats.update(pool=len(pool), src={int(k): int(v) for k, v in pool.src.value_counts().items()}, nbits=int(len(bits)),
                 bits=a.bits, train_structs=len(tr), mass_min=float(pool.mass.min()), mass_max=float(pool.mass.max()))
    json.dump(stats, open(os.path.join(T, 'stats.json'), 'w'), indent=1); print(json.dumps(stats), flush=True)


def stage_verify(a):
    T = os.path.join(a.out, 'tables'); rep = {}
    new = pd.read_parquet(os.path.join(T, 'pool_meta.parquet')); ref = pd.read_parquet(os.path.join(a.ref, 'pool_meta.parquet'))
    kn, kr = set(new.key), set(ref.key); sh = kn & kr
    rep['pool'] = dict(new=len(new), ref=len(ref), shared=len(sh), only_new=len(kn - kr), only_ref=len(kr - kn),
                       ref_covered=len(sh) / len(kr))
    for s in (1, 2, 3):
        rk = set(ref.key[ref.src == s]); rep['pool'][f'ref_src{s}'] = dict(n=len(rk), covered=len(rk & kn))
    m = ref.merge(new, on='key', suffixes=('_r', '_n'))
    rep['shared_rows'] = dict(n=len(m), same_smiles=int((m.smiles_r == m.smiles_n).sum()), same_formula=int((m.formula_r == m.formula_n).sum()),
                              mass_max_abs=float((m.mass_r - m.mass_n).abs().max()), same_src=int((m.src_r == m.src_n).sum()),
                              same_n_heavy=int((m.n_heavy_r == m.n_heavy_n).sum()))
    # fingerprints of shared rows, on the REFERENCE's bit selection, sample of rows with the same SMILES and of all shared rows
    rb = np.load(os.path.join(a.ref, 'fp_bits.npy')); nb = np.load(os.path.join(T, 'fp_bits.npy'))
    rep['bits'] = dict(ref=len(rb), new=len(nb), shared=int(len(np.intersect1d(rb, nb))))
    rfp = np.load(os.path.join(a.ref, 'pool_fp.npy'), mmap_mode='r'); nfp = np.load(os.path.join(T, 'pool_fp.npy'), mmap_mode='r')
    com, ir, inn = np.intersect1d(rb, nb, return_indices=True)
    rng = np.random.default_rng(0)
    for nm, mm in (('same_smiles', m[m.smiles_r == m.smiles_n]), ('all_shared', m)):
        ix = rng.choice(len(mm), min(a.sample, len(mm)), replace=False)
        A = np.unpackbits(rfp[np.sort(mm.pid_r.values[ix])], axis=1)[:, :len(rb)][:, ir]
        o = np.argsort(mm.pid_r.values[ix]); B = np.unpackbits(nfp[mm.pid_n.values[ix][o]], axis=1)[:, :len(nb)][:, inn]
        rep[f'fp_{nm}'] = dict(rows=int(len(ix)), bits_compared=int(len(com)), identical_rows=int((A == B).all(1).sum()),
                               bit_mismatch=int((A != B).sum()))
    nt = pd.read_parquet(os.path.join(T, 'train_structs.parquet')); rt = pd.read_parquet(os.path.join(a.ref, 'train_structs.parquet'))
    mt = rt.merge(nt, on='inchikey14', suffixes=('_r', '_n'))
    rep['train_structs'] = dict(new=len(nt), ref=len(rt), shared=len(mt), same_sid=int((mt.sid_r == mt.sid_n).sum()),
                                same_smiles=int((mt.smiles_r == mt.smiles_n).sum()), same_key=int((mt.key_r == mt.key_n).sum()),
                                same_rawkey=int((mt.rawkey_r == mt.rawkey_n).sum()), same_n_spec=int((mt.n_spec_r == mt.n_spec_n).sum()),
                                same_n_tims=int((mt.n_tims_r == mt.n_tims_n).sum()), same_libmask=int((mt.libmask_r == mt.libmask_n).sum()))
    json.dump(rep, open(os.path.join(a.out, 'verify.json'), 'w'), indent=1); print(json.dumps(rep, indent=1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['sources', 'structs', 'assemble', 'verify'])
    ap.add_argument('--out', default=DEF_OUT); ap.add_argument('--train', default=DEF_TRAIN); ap.add_argument('--coconut', default=DEF_COCO)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--chunk', type=int, default=4000)
    ap.add_argument('--ref', default=os.path.join(DATA, 'directions', 'run', 'work')); ap.add_argument('--bits', default='own', choices=['own', 'ref'])
    ap.add_argument('--sample', type=int, default=20000)
    a = ap.parse_args()
    _low_priority()
    dict(sources=stage_sources, structs=stage_structs, assemble=stage_assemble, verify=stage_verify)[a.stage](a)
