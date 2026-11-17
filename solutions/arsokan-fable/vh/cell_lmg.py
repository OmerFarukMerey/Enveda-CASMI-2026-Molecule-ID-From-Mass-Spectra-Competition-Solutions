# -- LMG-1 - leader-move guard (place directly after the merge cell, before the v8 cells) -------------------------
# The merge cell lets the fragment check replace rank 1 of the V2 list. On 250 held-out natural products with
# cross-lab references that move was wrong 7 times out of 8. This cell undoes the move when the replaced leader
# was rank 1 in BOTH engines (main_rank == 1 and engine2_rank == 1 in frozen_candidate_evidence.parquet) AND the
# molecule's best library match is >= LMG_LIB. Only rank 1 is restored; the rest of the merged order is kept.
# Lists without a merge leader move are never touched. Any error keeps the merge cell's file.
LMG_ON = True
LMG_LIB = 0.70
LMG_REPORT = dict(status='off')


def _lmg_main():
    import os, shutil
    import pandas as pd
    rep = dict(status='off')
    if not LMG_ON:
        return rep
    if globals().get('MERGE_REPORT', {}).get('status') != 'applied':
        rep['status'] = 'off: merge cell did not apply'
        return rep
    sub_path = ROOT / 'submission.csv'
    sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    v2 = pd.read_csv(ROOT / 'variants/submission_V2.csv', dtype=str, keep_default_na=False)
    assert list(sub.columns) == list(v2.columns) == ['molecule_id', 'smiles']
    assert sub.molecule_id.tolist() == v2.molecule_id.tolist() == SAMPLE_IDS
    ev = pd.read_parquet(ROOT / 'frozen_candidate_evidence.parquet',
                         columns=['molecule_id', 'smiles', 'main_rank', 'engine2_rank', 'library_max'])
    ev['molecule_id'] = ev.molecule_id.astype(str)
    lib = {m: float(v) for m, v in ev.groupby('molecule_id', sort=False).library_max.max().items()}
    top = {(m, s) for m, s, a, b in zip(ev.molecule_id, ev.smiles.astype(str), ev.main_rank, ev.engine2_rank)
           if a == 1 and b == 1}
    rows, moved, guarded, restored, ids = [], 0, 0, 0, []
    for mid, ms, vs in zip(sub.molecule_id.astype(str), sub.smiles.astype(str), v2.smiles.astype(str)):
        m_list, old = ms.split(';'), vs.split(';')[0]
        new = m_list
        if m_list[0] != old:
            moved += 1
            if (mid, old) in top and lib.get(mid, 0.0) >= LMG_LIB:
                guarded += 1
                if old in m_list:
                    new = [old] + [s for s in m_list if s != old]
                    restored += 1
                    ids.append(mid)
        rows.append(';'.join(new))
    out = sub.copy()
    out['smiles'] = rows
    # checks: same members and length everywhere; byte-identical lists wherever the guard did not act
    _ids = set(ids)
    for mid, a, b in zip(sub.molecule_id.astype(str), sub.smiles, out.smiles):
        if mid in _ids:
            x, y = a.split(';'), b.split(';')
            assert len(x) == len(y) == len(set(y)) and set(x) == set(y) and y[1:] == [s for s in x if s != y[0]]
        else:
            assert a == b, 'unguarded list changed'
    moved_ids = globals().get('MERGE_REPORT', {}).get('moved_ids')
    shutil.copyfile(sub_path, ROOT / 'submission_before_lmg.csv')
    tmp = ROOT / 'submission.lmg.tmp'
    out.to_csv(tmp, index=False)
    os.replace(tmp, sub_path)
    rep = dict(status='applied', lists=len(sub), merge_leader_moves=moved, guarded=guarded, restored=restored,
               unguarded_identical=True, lib_gate=LMG_LIB, restored_ids=ids[:50],
               merge_report_leader_moves=(len(moved_ids) if isinstance(moved_ids, list) else None))
    return rep


try:
    if not PIPELINE_DONE and not IS_RERUN and COMMIT_MODE == 'fast':
        LMG_REPORT = dict(status='skipped_fast_commit')
    else:
        LMG_REPORT = _lmg_main()
except Exception as exc:
    import traceback
    traceback.print_exc()
    LMG_REPORT = dict(status='off after an error: merge output kept', exception=repr(exc)[:300])
    try:
        if (ROOT / 'submission_before_lmg.csv').is_file():
            shutil.copyfile(ROOT / 'submission_before_lmg.csv', ROOT / 'submission.csv')
    except Exception:
        traceback.print_exc()
try:
    (ROOT / 'lmg_report.json').write_text(json.dumps(LMG_REPORT, indent=2, default=str))
except Exception:
    pass
# dry-run block: these counts must be read in the commit log before any submission
print('LMG-1 DRY RUN | status', LMG_REPORT.get('status'), '| lists', LMG_REPORT.get('lists'),
      '| merge leader moves', LMG_REPORT.get('merge_leader_moves'), '(merge report:', LMG_REPORT.get('merge_report_leader_moves'), ')',
      '| guarded', LMG_REPORT.get('guarded'), '| restored', LMG_REPORT.get('restored'),
      '| unguarded lists identical', LMG_REPORT.get('unguarded_identical'), '| ids', LMG_REPORT.get('restored_ids'))
final = pd.read_csv(ROOT / 'submission.csv', dtype=str, keep_default_na=False)
assert list(final.columns) == ['molecule_id', 'smiles'] and final.molecule_id.tolist() == SAMPLE_IDS
assert final.smiles.str.split(';').map(len).between(1, 25).all()
assert final.smiles.str.split(';').map(lambda a: all(bool(s) for s in a) and len(set(a)) == len(a)).all()
