# ===== Arsokan NPX-TIER: real natural products missing from the candidate pool, as an extra row set of the virtual tier =====
# natural_products.parquet (COCONUT / LOTUS / NPAtlas structures) and casmi26-lotus-candidates (LOTUS, CC0) hold ~20,000 natural
# products that are not in the 710,701-structure pool, so no stage of the stack can return them. They are written here in the
# virtual tier's own row layout (mass-sorted <prefix>mass / off / smiles .npy) and handed to the tier cell as one extra row set
# (vt_core's cfg EXTRA), so they go through the tier's unchanged scoring, de-duplication (metric key; pool structures dropped),
# neural re-rank, merge rule, caps and gate. Route prior: these rows have none, so a fixed log-odds of -1.80 is used = the median
# learned prior (vt_lp_e1) of tier rows that are real PubChem compounds (vt_pc == 1); the median over all tier rows is -7.66.
# Duplicates with vt0 rows are left to the tier's own key de-duplication. Fails closed: on any error the tier runs as before.
ARSOKAN_NP_EXTRA = None
_NPE_LP = -1.80
try:
    import glob as _npe_glob, os as _npe_os
    import numpy as _npe_np, pandas as _npe_pd
    _npe_f = lambda pat: sorted(_npe_glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)[0]
    _npe_pool = set(_npe_pd.read_parquet(_npe_f('pool_meta.parquet'), columns=['key'])['key'])
    _npe_a = _npe_pd.read_parquet(_npe_f('natural_products.parquet'), columns=['ik14', 'scorer_key', 'smiles', 'mono_mass'])
    _npe_a = _npe_a[~_npe_a['ik14'].isin(_npe_pool) & ~_npe_a['scorer_key'].isin(_npe_pool)]
    _npe_a = _npe_a.rename(columns={'ik14': 'k', 'mono_mass': 'mass'})[['k', 'smiles', 'mass']]
    _npe_b = _npe_pd.read_parquet(_npe_f('candidates_lotus.parquet'), columns=['inchikey14', 'smiles', 'monoisotopic_mass'])
    _npe_b = _npe_b[~_npe_b['inchikey14'].isin(_npe_pool)].rename(columns={'inchikey14': 'k', 'monoisotopic_mass': 'mass'})
    _npe = _npe_pd.concat([_npe_a, _npe_b], ignore_index=True).dropna()
    _npe = _npe[_npe['smiles'].map(lambda s: isinstance(s, str) and s.isascii() and 0 < len(s) < 2000) & (_npe['mass'] > 50)]
    _npe = _npe.drop_duplicates('k').sort_values('mass', kind='mergesort').reset_index(drop=True)
    assert len(_npe) > 1000, len(_npe)
    _npe_dir = '/kaggle/working/arsokan_np_extra'; _npe_os.makedirs(_npe_dir, exist_ok=True)
    _npe_raw = [s.encode('ascii') for s in _npe['smiles']]
    _npe_off = _npe_np.zeros(len(_npe_raw) + 1, _npe_np.uint64); _npe_off[1:] = _npe_np.cumsum([len(x) for x in _npe_raw])
    _npe_np.save(f'{_npe_dir}/npx_mass.npy', _npe['mass'].to_numpy(_npe_np.float64))
    _npe_np.save(f'{_npe_dir}/npx_off.npy', _npe_off)
    _npe_np.save(f'{_npe_dir}/npx_smiles.npy', _npe_np.frombuffer(b''.join(_npe_raw), _npe_np.uint8))
    _npe_np.save(f'{_npe_dir}/npx_lp.npy', _npe_np.full(len(_npe_raw), _NPE_LP, _npe_np.float16))
    ARSOKAN_NP_EXTRA = dict(mass=f'{_npe_dir}/npx_mass.npy', lp=f'{_npe_dir}/npx_lp.npy', rows=len(_npe_raw))
    print(f'[np-extra] {len(_npe_raw):,} natural products outside the pool written as an extra tier row set (prior {_NPE_LP})', flush=True)
except Exception as _npe_e:
    ARSOKAN_NP_EXTRA = None
    print('[np-extra] skipped, the tier runs unchanged:', repr(_npe_e)[:300], flush=True)
