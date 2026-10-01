# SMOKE switch: the commit run (visible test = train examples) scores SMOKE_N molecules unless FULL_ON_COMMIT;
# the scoring rerun (hidden test) always runs everything. VALIDATION (commit runs only) swaps in a held-out test.
import pandas as pd
_te = pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=['molecule_id', 'spectrum_id'])
_sig = hashlib.md5(','.join(sorted(_te.spectrum_id.astype(str))).encode()).hexdigest()
IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'
ICE_BUDGET = CFG['ICE_BUDGET']

TEST_ADDUCTS = ['[M+H]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+', '[M+Na]+', '[M+K]+',
                '[M-H]-', '[M-H2O-H]-', '[M+CH2O2-H]-', '[M+Cl]-']

def build_validation(comp, out, lib, n, seed, max_spec):
    """Hold out n structures of `lib`: their spectra (test adducts, base peak >= 1000 counts, <= max_spec per
    molecule) become test.parquet, and EVERY train row with those InChIKey14s is dropped from train.parquet, so the
    library cannot answer them (candidate pools still contain them: a class-2-like setting). Writes labels too."""
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    src = os.path.join(comp, 'train.parquet')
    meta = pq.read_table(src, columns=['ingest_lib', 'inchikey14', 'adduct']).to_pandas()
    keys = sorted(meta.loc[(meta.ingest_lib == lib) & meta.adduct.isin(TEST_ADDUCTS), 'inchikey14'].unique())
    keys = [keys[i] for i in np.random.RandomState(seed).permutation(len(keys))[:n]]
    hold = pa.array(keys)
    os.makedirs(out, exist_ok=True)
    pf, writer, parts = pq.ParquetFile(src), None, []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        held = pc.is_in(t['inchikey14'], value_set=hold)
        kept = t.filter(pc.invert(held))
        writer = writer or pq.ParquetWriter(os.path.join(out, 'train.parquet'), kept.schema)
        writer.write_table(kept)
        q = t.filter(pc.and_(pc.and_(held, pc.equal(t['ingest_lib'], lib)),
                             pc.is_in(t['adduct'], value_set=pa.array(TEST_ADDUCTS))))
        if q.num_rows:
            parts.append(q)
    writer.close()
    q = pa.concat_tables(parts).to_pandas()
    q = q[q.base_peak_intensity.isna() | (q.base_peak_intensity >= 1000)]
    q = q.groupby('inchikey14', group_keys=False).head(max_spec).reset_index(drop=True)
    q['molecule_id'] = ['v_' + hashlib.md5(k.encode()).hexdigest()[:8] for k in q.inchikey14]
    q['spectrum_id'] = [f'vs_{i:06d}' for i in range(len(q))]
    q['instrument_type'] = 'timsTOF'
    test_cols = ['molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct',
                 'ionization_mode', 'instrument_type', 'precursor_mz', 'collision_energy_ev', 'collision_energy_orig',
                 'collision_energy_orig_units']
    q[test_cols].to_parquet(os.path.join(out, 'test.parquet'), index=False)
    lab = q.groupby('molecule_id').agg(smiles=('normalized_smiles', 'first'), inchikey14=('inchikey14', 'first')).reset_index()
    pd.DataFrame({'molecule_id': lab.molecule_id, 'smiles': 'CCO'}).to_csv(os.path.join(out, 'sample_submission.csv'), index=False)
    os.makedirs(STAGE, exist_ok=True)
    lab.to_csv(os.path.join(STAGE, 'val_labels.csv'), index=False)
    return len(lab), len(q)

if CFG['VALIDATION'] and not IS_RERUN:
    _t0 = time.time()
    _nm, _ns = build_validation(COMP, os.path.join(WORK, 'val_comp'), CFG['VAL_LIB'], CFG['VAL_N'], CFG['VAL_SEED'],
                                CFG['VAL_MAX_SPEC'])
    COMP = os.path.join(WORK, 'val_comp')
    log(f'VALIDATION: {_nm} held-out {CFG["VAL_LIB"]} molecules / {_ns} spectra, library purged ({time.time()-_t0:.0f}s)')
elif not IS_RERUN and not CFG['FULL_ON_COMMIT']:
    SM = os.path.join(WORK, 'smoke_comp'); os.makedirs(SM, exist_ok=True)
    _full = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
    _keep = sorted(_full.molecule_id.unique())[:CFG['SMOKE_N']]
    _full[_full.molecule_id.isin(_keep)].to_parquet(os.path.join(SM, 'test.parquet'))
    for _f in ['train.parquet', 'sample_submission.csv']:
        if not os.path.exists(os.path.join(SM, _f)):
            os.symlink(os.path.join(COMP, _f), os.path.join(SM, _f))
    COMP = SM
    ICE_BUDGET = CFG['SMOKE_ICE_BUDGET']
log('RERUN', IS_RERUN, '| sig', _sig, '| COMP', COMP, '| ICE_BUDGET', ICE_BUDGET, '| FULL_ON_COMMIT', CFG['FULL_ON_COMMIT'],
    '| VALIDATION', CFG['VALIDATION'])
