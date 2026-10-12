# SMOKE switch: the commit run (visible test = train examples) scores SMOKE_N molecules unless FULL_ON_COMMIT;
# the scoring rerun (hidden test) always runs everything. VALIDATION (commit runs only) swaps in a held-out test.
import pandas as pd
_te = pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=['molecule_id', 'spectrum_id'])
_sig = hashlib.md5(','.join(sorted(_te.spectrum_id.astype(str))).encode()).hexdigest()
IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'
ICE_BUDGET = CFG['ICE_BUDGET']

TEST_ADDUCTS = ['[M+H]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+', '[M+Na]+', '[M+K]+',
                '[M-H]-', '[M-H2O-H]-', '[M+CH2O2-H]-', '[M+Cl]-']

def fold0_np_keys():
    """Structures that the fold-safe FPNet (dmitriigluzdov/casmi26-fold-safe-fpnet, fold 'hold') never saw and that
    are in a natural-product catalogue (pool_np > 0: COCONUT / LOTUS / NPAtlas). Ordinary natural products: median
    popularity 5.4 against 13.0 for the famous enveda-np-examples."""
    import numpy as np
    sp = pd.read_parquet(find('casmi26-fold-safe-fpnet/**/split.parquet'))
    pm = pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'), columns=['key', 'pid'])
    npf = np.load(find('casmi26-pubchem-popularity-prior/**/pool_np.npy'))
    nat = set(pm.key.values[npf[pm.pid.values] > 0])
    h = sp[(sp.fold == 'hold') & sp.key.isin(nat)]
    return set(h.inchikey14)


def build_validation_mix(comp, out, lib, n, seed, max_spec):
    """VAL_PURGE 'mix': n structures of `lib` are held out; the first max_spec eligible `lib` spectra of each (file order) become its test
    spectra. Structures at even positions of the draw lose EVERY training row of their metric key (class-2-like, as mode 'all'); those at odd
    positions lose only the test spectra themselves, so the library keeps their other spectra (library hits where there are any). Labels
    carry the class in `val_class` ('all' / 'lib')."""
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    src = os.path.join(comp, 'train.parquet')
    meta = pq.read_table(src, columns=['ingest_lib', 'inchikey14', 'adduct']).to_pandas()
    keys = sorted(meta.loc[meta.adduct.isin(TEST_ADDUCTS) & (meta.ingest_lib == lib), 'inchikey14'].unique())
    _st = pd.read_parquet(os.path.join(POOL_DIR, 'train_structs.parquet'), columns=['inchikey14', 'key'])
    ik2key = dict(zip(_st.inchikey14, _st.key))
    order = [keys[i] for i in np.random.RandomState(seed).permutation(len(keys))]
    keys, seen = [], set()
    for k in order:
        mk = ik2key.get(k, k)
        if mk not in seen:
            seen.add(mk); keys.append(k)
        if len(keys) >= n:
            break
    cls = {k: ('all' if i % 2 == 0 else 'lib') for i, k in enumerate(keys)}
    seen_all = {ik2key.get(k, k) for k in keys if cls[k] == 'all'}
    purge = {k for k in keys if cls[k] == 'all'} | set(_st.inchikey14[_st.key.isin(seen_all)])
    os.makedirs(out, exist_ok=True)
    pf, writer, parts, cnt = pq.ParquetFile(src), None, [], {}
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        d = t.select(['inchikey14', 'ingest_lib', 'adduct', 'base_peak_intensity']).to_pandas()
        elig = (d.inchikey14.isin(cls) & (d.ingest_lib == lib) & d.adduct.isin(TEST_ADDUCTS) &
                (d.base_peak_intensity.isna() | (d.base_peak_intensity >= 1000))).values
        take = np.zeros(len(d), bool)
        for i in np.where(elig)[0]:
            k = d.inchikey14.values[i]
            if cnt.get(k, 0) < max_spec:
                cnt[k] = cnt.get(k, 0) + 1; take[i] = True
        drop = d.inchikey14.isin(purge).values | take
        kept = t.filter(pa.array(~drop))
        writer = writer or pq.ParquetWriter(os.path.join(out, 'train.parquet'), kept.schema)
        writer.write_table(kept)
        if take.any():
            parts.append(t.filter(pa.array(take)))
    writer.close()
    q = pa.concat_tables(parts).to_pandas().reset_index(drop=True)
    q['molecule_id'] = ['v_' + hashlib.md5(k.encode()).hexdigest()[:8] for k in q.inchikey14]
    q['spectrum_id'] = [f'vs_{i:06d}' for i in range(len(q))]
    test_cols = ['molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct',
                 'ionization_mode', 'instrument_type', 'precursor_mz', 'collision_energy_ev', 'collision_energy_orig',
                 'collision_energy_orig_units']
    q[test_cols].to_parquet(os.path.join(out, 'test.parquet'), index=False)
    lab = q.groupby('molecule_id').agg(smiles=('normalized_smiles', 'first'), inchikey14=('inchikey14', 'first')).reset_index()
    lab['val_class'] = [cls[k] for k in lab.inchikey14]
    pd.DataFrame({'molecule_id': lab.molecule_id, 'smiles': 'CCO'}).to_csv(os.path.join(out, 'sample_submission.csv'), index=False)
    os.makedirs(STAGE, exist_ok=True)
    lab.to_csv(os.path.join(STAGE, 'val_labels.csv'), index=False)
    return len(lab), len(q)


def build_validation(comp, out, lib, n, seed, max_spec, mode='all', only=None):
    """Hold out n structures: their spectra (test adducts, base peak >= 1000 counts, <= max_spec per molecule) become
    test.parquet, and EVERY train row with those InChIKey14s is dropped from train.parquet, so the library cannot
    answer them (candidate pools still contain them: a class-2-like setting). mode='lib' drops only the held-out
    `lib` rows, so spectra of the same structures from OTHER libraries stay searchable: a class-1-like setting (a new
    timsTOF measurement of a compound with public reference spectra). lib=None with `only` (a set of InChIKey14s)
    draws the structures from that set and their test spectra from any library. Writes labels too."""
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    src = os.path.join(comp, 'train.parquet')
    meta = pq.read_table(src, columns=['ingest_lib', 'inchikey14', 'adduct']).to_pandas()
    ok = meta.adduct.isin(TEST_ADDUCTS)
    if lib is not None:
        ok &= meta.ingest_lib == lib
    if only is not None:
        ok &= meta.inchikey14.isin(only)
    keys = sorted(meta.loc[ok, 'inchikey14'].unique())
    # audit 2026-10-02: the metric compares tautomer-canonical keys, so sample one structure per metric key and purge
    # every training InChIKey14 that shares a held-out metric key (train_structs.parquet carries key per structure)
    _st = pd.read_parquet(os.path.join(POOL_DIR, 'train_structs.parquet'), columns=['inchikey14', 'key'])
    ik2key = dict(zip(_st.inchikey14, _st.key))
    order = [keys[i] for i in np.random.RandomState(seed).permutation(len(keys))]
    keys, seen = [], set()
    for k in order:
        mk = ik2key.get(k, k)
        if mk not in seen:
            seen.add(mk); keys.append(k)
        if len(keys) >= n:
            break
    hold = pa.array(keys)
    purge = pa.array(sorted(set(keys) | set(_st.inchikey14[_st.key.isin(seen)])))   # held-out keys and their twins
    os.makedirs(out, exist_ok=True)
    pf, writer, parts = pq.ParquetFile(src), None, []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        held = pc.is_in(t['inchikey14'], value_set=hold)
        twin = pc.is_in(t['inchikey14'], value_set=purge)
        in_lib = pc.equal(t['ingest_lib'], lib) if lib is not None else held
        drop = twin if mode == 'all' else pc.and_(held, in_lib)
        kept = t.filter(pc.invert(drop))
        writer = writer or pq.ParquetWriter(os.path.join(out, 'train.parquet'), kept.schema)
        writer.write_table(kept)
        q = t.filter(pc.and_(pc.and_(held, in_lib), pc.is_in(t['adduct'], value_set=pa.array(TEST_ADDUCTS))))
        if q.num_rows:
            parts.append(q)
    writer.close()
    q = pa.concat_tables(parts).to_pandas()
    q = q[q.base_peak_intensity.isna() | (q.base_peak_intensity >= 1000)]
    if lib is None:                               # many libraries per structure: a seeded random subset, not the first rows
        q = q.sample(frac=1.0, random_state=seed).sort_values('inchikey14', kind='stable')
    q = q.groupby('inchikey14', group_keys=False).head(max_spec).reset_index(drop=True)
    q['molecule_id'] = ['v_' + hashlib.md5(k.encode()).hexdigest()[:8] for k in q.inchikey14]
    q['spectrum_id'] = [f'vs_{i:06d}' for i in range(len(q))]
    if lib is not None:                           # enveda-np-examples really is timsTOF; other libraries keep their label
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
    if CFG['VAL_SET'] == 'fold0_np':
        _nm, _ns = build_validation(COMP, os.path.join(WORK, 'val_comp'), None, CFG['VAL_N'], CFG['VAL_SEED'],
                                    CFG['VAL_MAX_SPEC'], 'all', only=fold0_np_keys())
    elif CFG['VAL_PURGE'] == 'mix':
        _nm, _ns = build_validation_mix(COMP, os.path.join(WORK, 'val_comp'), CFG['VAL_LIB'], CFG['VAL_N'], CFG['VAL_SEED'], CFG['VAL_MAX_SPEC'])
    else:
        _nm, _ns = build_validation(COMP, os.path.join(WORK, 'val_comp'), CFG['VAL_LIB'], CFG['VAL_N'], CFG['VAL_SEED'],
                                    CFG['VAL_MAX_SPEC'], CFG['VAL_PURGE'])
    COMP = os.path.join(WORK, 'val_comp')
    log(f'VALIDATION {CFG["VAL_SET"]}: {_nm} held-out molecules / {_ns} spectra, purge={CFG["VAL_PURGE"]} ({time.time()-_t0:.0f}s)')
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
# SOFT: per-molecule failures are logged and counted instead of raised. True on the scoring rerun; VAL_AS_RERUN makes a validation run behave
# the same way, so that it rehearses the hidden run instead of stopping at the first failure.
SOFT = IS_RERUN or bool(CFG['VALIDATION'] and CFG['VAL_AS_RERUN'])
log('RERUN', IS_RERUN, '| sig', _sig, '| COMP', COMP, '| ICE_BUDGET', ICE_BUDGET, '| FULL_ON_COMMIT', CFG['FULL_ON_COMMIT'],
    '| VALIDATION', CFG['VALIDATION'], '| SOFT', SOFT)
