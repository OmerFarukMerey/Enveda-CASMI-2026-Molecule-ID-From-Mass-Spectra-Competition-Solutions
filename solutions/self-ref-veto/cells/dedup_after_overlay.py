# ===== D-1 (Arsokan): metric-key deduplication after the external-library overlay ===========================
# The overlay inserts the external reference's SMILES string at rank 1. When the same structure was already listed under
# another SMILES string (stereo marks, other canonical form), the list now carries a dead duplicate further down and has
# lost its 25th candidate. Only the first copy of a structure can ever score, so later copies are removed (everything
# below moves up one place) and the freed places are refilled from the pre-overlay list. This can only move candidates
# up; the first occurrence of every structure keeps its rank. Any failure leaves submission.csv unchanged.
def _dedup_after_overlay(work='/kaggle/working'):
    import os, shutil, time, pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    t0 = time.time(); sp = os.path.join(work, 'submission.csv'); pre = os.path.join(work, 'submission_pre_apex_extlib.csv')
    if not os.path.exists(sp): print('[d1] no submission.csv'); return
    sub = pd.read_csv(sp, dtype=str, keep_default_na=False)
    prev = {}
    if os.path.exists(pre):
        p = pd.read_csv(pre, dtype=str, keep_default_na=False)
        prev = {m: [x for x in s.split(';') if x] for m, s in zip(p.molecule_id.astype(str), p.smiles)}
    cache = {}
    def key(s):
        if s not in cache:
            try:
                mol = Chem.MolFromSmiles(s); cache[s] = Chem.MolToInchiKey(mol)[:14] if mol is not None else None
            except Exception:
                cache[s] = None
        return cache[s]
    out, nfix, nrem, nfill = [], 0, 0, 0
    for m, s in zip(sub.molecule_id.astype(str), sub.smiles):
        lst = [x for x in s.split(';') if x]; seen, new = set(), []
        for x in lst:
            k = key(x)
            if k is not None and k in seen: nrem += 1; continue
            if k is not None: seen.add(k)
            new.append(x)
        if len(new) < len(lst):
            nfix += 1
            for x in prev.get(m, []):
                if len(new) >= 25: break
                k = key(x)
                if x in new or (k is not None and k in seen): continue
                new.append(x); nfill += 1
                if k is not None: seen.add(k)
            out.append(';'.join(new[:25]))
        else:
            out.append(s)
    if nfix:
        shutil.copy(sp, os.path.join(work, 'submission_pre_d1.csv'))
        sub['smiles'] = out; tmp = sp + '.tmp_d1'; sub.to_csv(tmp, index=False); os.replace(tmp, sp)
    print(f'[d1] lists fixed {nfix}, duplicates removed {nrem}, refilled {nfill}, {time.time() - t0:.0f}s', flush=True)
try:
    _dedup_after_overlay()
except Exception:
    import traceback; traceback.print_exc(); print('[d1] FAILED -> submission unchanged', flush=True)
