# ===== VH pre-flight: abort in the first minutes if any synthetic molecule would be skipped by an engine =====
# No RDKit here (installed later by the pipeline): masses and metric keys come embedded in VH_SEL.
# Run 1 of this harness died after 3 h in frozen cell 48 ("Engine 2 did not cover all active molecules"): engine 2 (and engine 1)
# drop a molecule whose candidate window is empty, i.e. whose neutral-mass target (median over its spectra, engine 2's own
# adduct deltas) misses every pool structure by more than 30 ppm. Checked here, per molecule of the synthetic test:
#   ids of test.parquet == ids of sample_submission.csv; every spectrum's adduct has a neutral-mass rule; >= 1 peak per spectrum;
#   the target lies within 10 ppm of the answer's exact mass (answers.csv; selection used 5 ppm for nphard / e180); the answer is in the COCONUT part of the pool;
#   >= 1 v2-pool structure within 10 ppm of the target.
if 'VH_COMP' in globals() and 'VH_OUT' in globals():
    import numpy as _pf_np, pandas as _pf_pd
    _te = _pf_pd.read_parquet(VH_COMP / 'test.parquet'); _ss = _pf_pd.read_csv(VH_COMP / 'sample_submission.csv', dtype=str)
    _an = _pf_pd.read_csv(VH_OUT / 'answers.csv', dtype=str).set_index('molecule_id')
    _pm_path = sorted(INPUT.rglob('pool_meta.parquet'), key=lambda p: len(str(p)))[0]
    _pm = _pf_pd.read_parquet(_pm_path, columns=['key', 'mass', 'src'])
    _pmass = _pf_np.sort(_pm['mass'].to_numpy(dtype=float)); _coco = set(_pm.loc[_pm['src'] >= 2, 'key'])
    # engine 2's own pool masses (COCONUT / PubChemLite capped at 1,300 Da, ChEBI / LIPID MAPS): >= 1 within 30 ppm is required
    _e2 = []
    for _n in ('coco_mass.npy', 'bio_mass.npy', 'pubchemlite_mass.npy'):
        _h = sorted(INPUT.rglob(_n), key=lambda p: len(str(p)))
        assert _h, 'VH pre-flight: ' + _n + ' not found'
        _e2.append(_pf_np.load(_h[0]))
    _e2 = _pf_np.sort(_pf_np.concatenate(_e2)); _e2 = _e2[_pf_np.isfinite(_e2)]
    assert _te.precursor_mz.max() <= VH_SEL['max_precursor'], ('precursor above the passing envelope', _te.precursor_mz.max())
    _bad = []
    assert set(_te.molecule_id.astype(str)) == set(_ss.molecule_id) == set(_an.index), 'VH pre-flight: id sets differ'
    assert _te.adduct.isin(list(VH_ADDUCT_DELTA)).all(), sorted(set(_te.adduct) - set(VH_ADDUCT_DELTA))
    assert (_te.ms2_mzs.map(len) > 0).all(), 'VH pre-flight: empty spectrum'
    for _mid, _sub in _te.groupby('molecule_id'):
        _nm = (_sub.precursor_mz - _sub.adduct.map(VH_ADDUCT_DELTA)).to_numpy(dtype=float)
        _t = float(_pf_np.median(_nm)); _ik = _an.inchikey14[_mid]; _true = float(VH_SEL['mass'][_ik])
        _ppm = abs(_t - _true) / _true * 1e6
        _n10 = int(_pf_np.searchsorted(_pmass, _t * (1 + 1e-5)) - _pf_np.searchsorted(_pmass, _t * (1 - 1e-5)))
        _key = VH_SEL['mkey'].get(_ik)    # metric key, embedded at build time
        _w30 = int(_pf_np.searchsorted(_e2, _t * (1 + 3e-5)) - _pf_np.searchsorted(_e2, _t * (1 - 3e-5)))
        _np_need = _an.group[_mid] == 'nphard'
        if not (_ppm <= 10.0 and _n10 >= 1 and _w30 >= 1 and _t <= VH_SEL['max_target'] and (not _np_need or _key in _coco or _an.inchikey14[_mid] in _coco)):
            _bad.append(dict(mid=_mid, group=_an.group[_mid], ppm=round(_ppm, 1), pool10=_n10, eng2_w30=_w30, target=round(_t, 3), coconut=_key in _coco or _an.inchikey14[_mid] in _coco))
    print('[vh] pre-flight:', len(_te.molecule_id.unique()), 'molecules,', len(_te), 'spectra, failures', len(_bad), flush=True)
    assert not _bad, f'VH pre-flight failed: {_bad[:10]}'
