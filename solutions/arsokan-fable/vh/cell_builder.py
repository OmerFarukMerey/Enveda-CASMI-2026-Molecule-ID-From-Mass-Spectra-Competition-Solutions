"""Notebook cell (vh): build the held-out test and the reduced train from an embedded selection; set the CASMI_* redirect variables."""
# === VH - held-out test + reduced train (runs before the path cell) ===
# Query rows are real training spectra of the selected molecules; they (class 1) or every row of the molecule (class 2)
# leave the train file the pipeline sees. Selection was made offline and is embedded below.
import json, os, time
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
_t0 = time.time()
VH_SEL = json.loads(r'''__SEL_JSON__''')
VH = Path('/kaggle/working/vh'); VH.mkdir(parents=True, exist_ok=True)
VH_POOL = Path('/tmp/vh_pool'); VH_POOL.mkdir(parents=True, exist_ok=True)
_src_dir = next(p for p in (Path('/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra'), Path('/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra'),
                            Path('/kaggle/input/enveda-casmi26-molecule-id-mass-spectra')) if (p / 'train.parquet').is_file())
TEST_COLS = ['molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct', 'ionization_mode',
             'instrument_type', 'precursor_mz', 'collision_energy_orig', 'collision_energy_ev', 'collision_energy_orig_units']
_qrow = {}; _drop = []
for _m in VH_SEL['mols']:
    for _i, _r in enumerate(_m['q']): _qrow[int(_r)] = (_m['mid'], f"{_m['mid']}_s{_i}")
    _drop += [int(x) for x in _m['d']]
_want = np.array(sorted(_qrow), np.int64); _drop = np.array(sorted(set(_drop)), np.int64)
assert set(_want.tolist()) <= set(_drop.tolist()), 'every query row must leave the pool'
_pf = pq.ParquetFile(str(_src_dir / 'train.parquet')); assert _pf.metadata.num_rows == VH_SEL['n_train_rows'], _pf.metadata.num_rows
_w = pq.ParquetWriter(str(VH_POOL / 'train.parquet'), _pf.schema_arrow, compression='zstd'); _parts = []; _base = 0; _nd = 0; _keys_ok = True
for _g in range(_pf.num_row_groups):
    _t = _pf.read_row_group(_g); _idx = np.arange(_base, _base + _t.num_rows, dtype=np.int64); _base += _t.num_rows
    _mq = np.isin(_idx, _want); _md = np.isin(_idx, _drop); _nd += int(_md.sum())
    if _mq.any(): _parts.append(_t.filter(pa.array(_mq)).to_pandas().assign(_row=_idx[_mq]))
    if _md.any(): _t = _t.filter(pa.array(~_md))
    if _t.num_rows: _w.write_table(_t)
_w.close(); assert _nd == len(_drop), (_nd, len(_drop))
_q = pd.concat(_parts, ignore_index=True).sort_values('_row').reset_index(drop=True)
_key_of = {m['mid']: m['key'] for m in VH_SEL['mols']}
_q['molecule_id'] = [_qrow[int(r)][0] for r in _q._row]; _q['spectrum_id'] = [_qrow[int(r)][1] for r in _q._row]
assert all(_key_of[a] == b for a, b in zip(_q.molecule_id, _q.inchikey14.astype(str))), 'row index does not point at the selected molecule'
_q[TEST_COLS].to_parquet(VH_POOL / 'test.parquet', index=False); _q[TEST_COLS].to_parquet(VH / 'test.parquet', index=False)
_sub = pd.DataFrame({'molecule_id': [m['mid'] for m in VH_SEL['mols']], 'smiles': 'C'})
assert set(_sub.molecule_id) == set(_q.molecule_id)
_sub.to_csv(VH_POOL / 'sample_submission.csv', index=False)
pd.DataFrame(VH_SEL['mols']).drop(columns=['q', 'd']).rename(columns={'mid': 'molecule_id'}).to_parquet(VH / 'truth.parquet', index=False)
_left = pd.read_parquet(VH_POOL / 'train.parquet', columns=['inchikey14']).inchikey14.astype(str).value_counts()
_aud = dict(n_molecules=len(VH_SEL['mols']), n_query_rows=int(len(_q)), n_rows_removed=int(_nd), n_pool_rows=int(_base - _nd),
            classes=pd.Series([m['cls'] for m in VH_SEL['mols']]).value_counts().to_dict(),
            class2_keys_left_in_pool=int(sum(_left.get(m['key'], 0) > 0 for m in VH_SEL['mols'] if m['cls'] == 'C2')),
            class1_without_rows_left=int(sum(_left.get(m['key'], 0) == 0 for m in VH_SEL['mols'] if m['cls'] != 'C2')),
            seconds=round(time.time() - _t0))
assert _aud['class2_keys_left_in_pool'] == 0 and _aud['class1_without_rows_left'] == 0, _aud
(VH / 'audit.json').write_text(json.dumps(_aud)); print('[vh] AUDIT', json.dumps(_aud), flush=True)
for _k, _v in (('CASMI_DATA_DIR', VH_POOL), ('CASMI_COMP_DIR', VH_POOL), ('CASMI_TEST_PATH', VH_POOL / 'test.parquet'), ('CASMI_SAMPLE_PATH', VH_POOL / 'sample_submission.csv')):
    os.environ[_k] = str(_v)
