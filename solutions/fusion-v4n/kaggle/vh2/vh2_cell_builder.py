# ===== VH (Arsokan private harness, never submitted): held-out synthetic test + reduced train, then redirect =====
# v2: three groups from VH_SEL['groups'] - nphard (natural products with no external reference spectrum, identity fold 0,
# TOF-type query spectra), npref0 (the 35 enveda-np-examples without external reference), e180 (150 controls).
# Inserted right after the session cell. Test = the 250 enveda-np-examples structures (all their np-examples spectra) plus 150
# enveda-180 structures (identity fold 0, seed 20261123; at most VH_CTRL_MAX_SPECTRA spectra in the visible test's adducts).
# Every train row of a held-out inchikey14 (and of plain keys sharing its metric key) leaves the train file the pipeline sees.
# np spectra: those in the visible test's adducts (all of a molecule's spectra if it has none). Redirect: COMP and the run-mode globals of the session cell are recomputed on the synthetic dir; inside this kernel process
# glob / pathlib searches that hit the real competition files return the synthetic ones; the frozen pipeline (subprocesses)
# reads CASMI_VH_COMP (three patched lines). Answers go to /kaggle/working/vh/answers.csv.
import json as _vj, os as _vos, time as _vt
from pathlib import Path as _VP
VH_SEL = _vj.loads(r'''__SEL_JSON__''')
# engine 2's neutral-mass deltas (pv.ADDUCTS, n = z = 1) for the visible-test adducts
VH_ADDUCT_DELTA = {'[M+H]+': 1.00727645, '[M-H]-': -1.00727645, '[M+CH2O2-H]-': 44.99820285, '[M+Na]+': 22.9892207,
                   '[M+NH4]+': 18.03382555, '[M+K]+': 38.9631581, '[M+Cl]-': 34.96940126}
VH_CLEAN_MAIN = True              # main engine (frozen cell 37) loads fpnet_fold0.pt (never saw fold 0) instead of fpnet_full1
VH_COMP = _VP('/tmp/enveda-casmi26-vh2-mass-spectra')
VH_OUT = _VP('/kaggle/working/vh')
TEST_COLS = ['molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct', 'ionization_mode',
             'instrument_type', 'precursor_mz', 'collision_energy_orig', 'collision_energy_ev', 'collision_energy_orig_units']


def vh_build(src_dir, comp_dir, out_dir, sel):
    """Write comp_dir/{train,test}.parquet + sample_submission.csv and out_dir/{answers.csv, test.parquet, audit.json}."""
    import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    t0 = _vt.time()
    comp_dir, out_dir = _VP(comp_dir), _VP(out_dir)
    comp_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    groups = sel['groups']
    drop = set().union(*[set(g['keys']) for g in groups]) | set(sel['twin_keys'])
    pf = pq.ParquetFile(str(_VP(src_dir) / 'train.parquet'))
    assert pf.metadata.num_rows == sel['n_train_rows'], (pf.metadata.num_rows, sel['n_train_rows'])
    dset = pa.array(sorted(drop))
    w = pq.ParquetWriter(str(comp_dir / 'train.parquet'), pf.schema_arrow, compression='zstd')
    parts, removed, kept = [], 0, 0
    for g in range(pf.num_row_groups):
        tb = pf.read_row_group(g)
        m = pc.is_in(tb.column('inchikey14'), value_set=dset)
        q = tb.filter(m)
        if q.num_rows:
            parts.append(q.to_pandas())
        removed += q.num_rows
        rest = tb.filter(pc.invert(m))
        kept += rest.num_rows
        if rest.num_rows:
            w.write_table(rest)
        del tb, q, rest
    w.close()
    assert removed == sel['expect']['rows_removed'], (removed, sel['expect'])
    q = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(sel['seed'])
    import re as _re

    def cap(df, n):
        if n is None:
            return df
        return df.groupby('inchikey14', group_keys=False).apply(
            lambda x: x.iloc[np.sort(rng.permutation(len(x))[:n])] if len(x) > n else x)
    ce = q.collision_energy_ev.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan)
    _ex = sel['mass']                    # answers' exact masses, computed with RDKit at build time (no RDKit in the image yet)
    ppm = (q.precursor_mz - q.adduct.map(VH_ADDUCT_DELTA) - q.inchikey14.map(_ex)) / q.inchikey14.map(_ex) * 1e6
    parts_t, n_spec = [], {}
    for g in groups:
        x = q[q.inchikey14.isin(set(g['keys']))]
        if g['libs']:
            x = x[x.ingest_lib.isin(g['libs'])]
        m = x.adduct.isin(sel['vis_adducts']) & (x.num_peaks.fillna(0) >= g['min_peaks'])
        if g['inst_re']:
            m &= x.instrument_type.fillna('').str.contains(g['inst_re'], regex=True)
        if g['ce']:
            m &= ce[x.index].between(g['ce'][0], g['ce'][1])
        if g.get('max_ppm') is not None:
            m &= ppm[x.index].abs() <= g['max_ppm']
        x = x[m | (~x.inchikey14.isin(set(x.loc[m, 'inchikey14'])) & (g['name'] == 'npref0'))]
        x = cap(x.sort_values(['inchikey14', 'precursor_mz']).reset_index(drop=True), g['cap'])
        n_spec[g['name']] = int(len(x))
        parts_t.append(x.assign(_grp=g['name']))
    tq = pd.concat(parts_t, ignore_index=True)
    keys = sorted(set(tq.inchikey14))
    assert set(keys) == drop - set(sel['twin_keys']), (len(keys), len(drop))
    hexes = rng.choice(16 ** 6, size=len(keys), replace=False)
    mid = {k: 'm_%06x' % h for k, h in zip(rng.permutation(keys), hexes)}
    tq['molecule_id'] = tq.inchikey14.map(mid)
    sh = rng.choice(16 ** 8, size=len(tq), replace=False)
    tq['spectrum_id'] = ['s_%08x' % h for h in sh]
    tq = tq.sort_values(['molecule_id', 'spectrum_id']).reset_index(drop=True)
    _ls, _ld = pa.large_string(), pa.list_(pa.float64())
    vis_schema = pa.schema([(c, _ld if c in ('ms2_mzs', 'ms2_normalized_intensities', 'collision_energy_ev') else
                             pa.float64() if c in ('base_peak_intensity', 'precursor_mz') else _ls) for c in TEST_COLS])
    test = pa.Table.from_pandas(tq[TEST_COLS], schema=vis_schema, preserve_index=False)
    pq.write_table(test, str(comp_dir / 'test.parquet'))
    ids = sorted(set(tq.molecule_id))
    pd.DataFrame({'molecule_id': ids, 'smiles': ';'.join(['CCO'] * 25)}).to_csv(comp_dir / 'sample_submission.csv', index=False)
    ans = tq.drop_duplicates('molecule_id')[['molecule_id', 'inchikey14', '_grp']].rename(columns={'_grp': 'group'})
    smi = q.drop_duplicates('inchikey14').set_index('inchikey14').normalized_smiles
    ans.insert(2, 'smiles', ans.inchikey14.map(smi).values)
    ans.sort_values('molecule_id').to_csv(out_dir / 'answers.csv', index=False)
    left = set(pq.read_table(str(comp_dir / 'train.parquet'), columns=['inchikey14']).column(0).to_pylist())
    audit = dict(molecules=len(ids), groups=ans.group.value_counts().to_dict(), spectra=n_spec, test_spectra=int(len(tq)),
                 rows_removed=int(removed), rows_kept=int(kept), held_out_keys_left_in_train=len(drop & left), seconds=round(_vt.time() - t0, 1))
    assert audit['held_out_keys_left_in_train'] == 0 and all(audit['groups'].get(g['name']) == len(g['keys']) for g in groups), audit
    (out_dir / 'audit.json').write_text(_vj.dumps(audit, indent=1))
    pq.write_table(test, str(out_dir / 'test.parquet'))
    return audit


def vh_install_redirect(real_comp, vh_comp):
    """In-process: glob.glob / glob.iglob / Path.glob / Path.rglob hits on the real competition files -> the synthetic files."""
    import glob as _g, pathlib as _pl
    real = _vos.path.realpath(str(real_comp)); names = ('train.parquet', 'test.parquet', 'sample_submission.csv')

    def m(p):
        s = str(p)
        if _vos.path.basename(s) in names and _vos.path.realpath(_vos.path.dirname(s)) == real:
            return _vos.path.join(str(vh_comp), _vos.path.basename(s))
        return None

    def remap(items):
        out, seen = [], set()
        for p in items:
            r = m(p)
            q = (r if isinstance(p, str) else type(p)(r)) if r else p
            if str(q) not in seen:
                seen.add(str(q)); out.append(q)
        return out
    if not getattr(_g, '_vh', False):
        _glob, _iglob = _g.glob, _g.iglob
        _g.glob = lambda *a, **k: remap(_glob(*a, **k))
        _g.iglob = lambda *a, **k: iter(remap(_iglob(*a, **k)))
        _g._vh = True
    if not getattr(_pl.Path, '_vh', False):
        _pglob, _prglob = _pl.Path.glob, _pl.Path.rglob
        _pl.Path.glob = lambda self, *a, **k: iter(remap(_pglob(self, *a, **k)))
        _pl.Path.rglob = lambda self, *a, **k: iter(remap(_prglob(self, *a, **k)))
        _pl.Path._vh = True


def vh_snap(tag):
    """Copy of submission.csv at a stage boundary (missing file: nothing)."""
    import shutil
    src = _VP('/kaggle/working/submission.csv')
    VH_OUT.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, VH_OUT / f'sub_{tag}.csv')
    print('[vh] snapshot', tag, src.is_file(), flush=True)


if 'COMP' in globals() and 'SAMPLE_IDS' in globals():
    VH_REAL_COMP = COMP
    VH_AUDIT = vh_build(VH_REAL_COMP, VH_COMP, VH_OUT, VH_SEL)
    print('[vh] AUDIT', _vj.dumps(VH_AUDIT), flush=True)
    COMP = VH_COMP
    _vos.environ['CASMI_VH_COMP'] = str(VH_COMP)
    _vos.environ['VH_CLEAN_MAIN'] = '1' if VH_CLEAN_MAIN else '0'
    vh_install_redirect(VH_REAL_COMP, VH_COMP)
    _ids = pd.read_parquet(COMP / 'test.parquet', columns=['spectrum_id']).spectrum_id.astype(str)
    TEST_SIGNATURE = hashlib.md5(','.join(sorted(_ids)).encode()).hexdigest()
    del _ids
    IS_RERUN = bool(_vos.getenv('KAGGLE_IS_COMPETITION_RERUN')) or TEST_SIGNATURE != '654e030e4173780e945252b4a5adf70b'
    assert IS_RERUN, 'VH: synthetic test must run in full mode'
    RUN_PIPELINE = IS_RERUN or COMMIT_MODE != 'fast'
    SHOW = not IS_RERUN
    SAMPLE_IDS = pd.read_csv(COMP / 'sample_submission.csv', dtype=str).molecule_id.tolist()
    MODE = 'VH harness (full pipeline on the synthetic test)'
    import glob as _vg
    assert sorted(INPUT.rglob('train.parquet'), key=lambda p: ('mass-spectra' not in str(p).lower(), len(str(p))))[0] == VH_COMP / 'train.parquet'
    assert sorted(_vg.glob(str(INPUT) + '/**/test.parquet', recursive=True), key=lambda p: ('mass-spectra' not in p.lower(), len(p)))[0] == str(VH_COMP / 'test.parquet')
    print('[vh] COMP', COMP, '| IS_RERUN', IS_RERUN, '| molecules', len(SAMPLE_IDS), flush=True)
