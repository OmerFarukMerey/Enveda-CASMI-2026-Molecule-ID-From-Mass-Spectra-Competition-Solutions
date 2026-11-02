"""Concatenate set tables (later duplicates of a query dropped). usage: merge_sets.py OUT_DIR IN_DIR [IN_DIR ...]"""
import sys
from pathlib import Path
import pandas as pd
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
parts, seen = [], set()
for d in sys.argv[2:]:
    x = pd.read_parquet(Path(d) / 'sets.parquet')
    x = x[~x.qkey.isin(seen)]; seen |= set(x.qkey); parts.append(x)
df = pd.concat(parts, ignore_index=True); df.to_parquet(out / 'sets.parquet')
print('sets', df.qkey.nunique(), 'rows', len(df))
