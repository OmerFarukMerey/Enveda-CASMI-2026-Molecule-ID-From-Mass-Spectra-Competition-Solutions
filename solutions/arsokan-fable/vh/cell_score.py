"""Notebook cell (vh): score every submission*.csv snapshot per class with the metric key and write vh/stage_scores.csv."""
# === VH - score every saved stage with the official key, per class ===
import json, glob, os
from pathlib import Path
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*'); _te = rdMolStandardize.TautomerEnumerator(); _kc = {}
def _key(s):
    if s not in _kc:
        try:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            _kc[s] = Chem.MolToInchiKey(_te.Canonicalize(m))[:14] if m is not None else None
        except Exception: _kc[s] = None
    return _kc[s]
_VH = Path('/kaggle/working/vh'); _tr = pd.read_parquet(_VH / 'truth.parquet'); _tr['mk'] = [_key(s) for s in _tr.smiles]
_rows = []
for _p in sorted(glob.glob('/kaggle/working/submission*.csv')):
    try:
        _s = pd.read_csv(_p, dtype=str, keep_default_na=False)
        if list(_s.columns) != ['molecule_id', 'smiles'] or set(_s.molecule_id) != set(_tr.molecule_id): continue
        _L = dict(zip(_s.molecule_id, _s.smiles)); _rr = []
        for _r in _tr.itertuples():
            _g = [x for x in _L[_r.molecule_id].split(';') if x][:25]; _h = next((i + 1 for i, x in enumerate(_g) if _key(x) == _r.mk), 0)
            _rr.append(1.0 / _h if _h else 0.0)
        _d = pd.DataFrame(dict(cls=_tr.cls.values, rr=_rr)); _row = dict(file=os.path.basename(_p), mrr=round(float(_d.rr.mean()), 4), top1=round(float((_d.rr == 1).mean()), 4), recall=round(float((_d.rr > 0).mean()), 4))
        _row.update({f'mrr_{k}': round(float(v), 4) for k, v in _d.groupby('cls').rr.mean().items()}); _rows.append(_row)
    except Exception as _e: print('[vh] score skipped', _p, type(_e).__name__, _e)
_sc = pd.DataFrame(_rows); _sc.to_csv(_VH / 'stage_scores.csv', index=False); print('[vh] STAGE_SCORES'); print(_sc.to_string(index=False), flush=True)
print('[vh] files', sorted(os.path.basename(p) for p in glob.glob('/kaggle/working/*.parquet') + glob.glob('/kaggle/working/*.json'))[:80], flush=True)
