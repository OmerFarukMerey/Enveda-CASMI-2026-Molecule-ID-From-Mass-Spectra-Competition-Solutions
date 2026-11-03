"""molecule_candidates (ae_core) must reproduce eval2's rows on held-out molecules."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ae_core as A
lib = Path(sys.argv[1]); ref = pd.read_parquet(sys.argv[2])
L = A.Library(lib / 'lib_peaks.npz', pd.read_parquet(lib / 'lib_struct.parquet')); keys = set(L.keys)
src = np.load(lib / 'lib_peaks.npz')['src']; code = json.loads((lib / 'lib_sources.json').read_text())['enveda-180']
rows_of = pd.Series(np.arange(len(L.sidx))).groupby(L.sidx).apply(np.array)
bad = 0; n = 0
for s in ref[ref.cand_rank >= 0].sidx.unique()[:60]:
    q = [(L.mz[r][:L.n[r]], L.it[r][:L.n[r]], L.prec[r], int(L.grp[r])) for r in rows_of[s] if src[r] == code and L.n[r] >= 3]
    new = pd.DataFrame(A.molecule_candidates(q, L, exclude=int(s), drop_keys=keys, keep_key=L.keys[s]))
    old = ref[ref.sidx == s].reset_index(drop=True)
    for c in A.FEATS:
        if c == 'is_add':
            continue
        if not np.allclose(new[c].astype(float).to_numpy(), old[c].astype(float).to_numpy()):
            bad += 1; print('differs', s, c); break
    n += 1
    if not (new.pick.to_numpy() == old.pick.to_numpy()).all():
        bad += 1; print('pick differs', s)
print('molecules compared', n, 'mismatches', bad)
