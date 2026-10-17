"""v72 probe cell (B1 + B2 forward-model arbitration after E2R2), appended verbatim to the stack1 notebook as its last cell.

POST-HOC LEADERBOARD PROBE, not a validated candidate (solutions/fusion-v4n/reviews/arb-2026-10-17.md, sections 4 and 6): the rule was assembled
after seeing that B1 and B2 each pass the pre-registered criterion in one popularity arm only, and ICEBERG / GLACIER have very probably seen the
local molecules it was measured on.

Acts on stack1's final file (after E2R2) per molecule, with x2 = engine 2's first usable row (E2R2's own choice), r1 = rank 1, F0 = the list
before E2R2 (E2R2's base bytes), y = F0's rank 2:
  B1  x2 sits at rank 2 with another identity than r1, ICEBERG and GLACIER scores exist for r1 and x2, and BOTH score x2 higher -> swap ranks 1, 2
  B2  (only if B1 did not fire) E2R2 changed the row, scores exist for x2 and y, and BOTH score x2 lower than y -> the row before E2R2 (F0)
Scores = ICE_SCORES / GL_SCORES the chain already computed (looked up by SMILES); nothing is re-scored. Every other row is left as it is.
Switch V72 (both False = this cell writes nothing: the submission is stack1's byte for byte). A self-test on synthetic rows runs first and
switches the patch off if it fails; any per-molecule error keeps stack1's row; any file-level error restores stack1's file. Log: V72_PATCH {...}.
The pure function v72_row is imported by solutions/fusion-v4n/tests/test_arb_v72.py (the runner below only runs as a notebook cell).
"""
V72 = dict(b1=True, b2=True)


def v72_row(F, F0, eng_list, ice, gl, key, sw, placeholder='CCO'):
    """One final list F (after E2R2) -> (new list, tag). Pure. tag in skipped, no_x2, b1_swap, b2_undo, none."""
    F = list(F)
    if not F or not isinstance(F[0], str) or F[0].strip() in ('', placeholder):
        return F, 'skipped'
    if isinstance(eng_list, str):
        eng_list = [eng_list]
    x2 = next((s.strip() for s in (eng_list or []) if isinstance(s, str) and s.strip() not in ('', placeholder)), None)
    kx = key(x2) if x2 is not None and not any(c in x2 for c in ';\r\n') else None
    if kx is None:
        return F, 'no_x2'

    def fin(d, s):
        try:
            v = d.get(s)
            v = None if v is None else float(v)
        except (TypeError, ValueError, AttributeError):
            return None
        return v if v is not None and v == v and v not in (float('inf'), float('-inf')) else None

    if sw.get('b1') and len(F) > 1 and key(F[0]) != kx and key(F[1]) == kx:
        a, b, c, d = fin(ice, F[0]), fin(ice, x2), fin(gl, F[0]), fin(gl, x2)
        if None not in (a, b, c, d) and b > a and d > c:
            return [F[1], F[0]] + F[2:], 'b1_swap'
    F0 = list(F0 or [])
    if sw.get('b2') and F0 != F and len(F0) > 1:
        y = F0[1]
        a, b, c, d = fin(ice, x2), fin(ice, y), fin(gl, x2), fin(gl, y)
        if None not in (a, b, c, d) and a < b and c < d:
            return F0, 'b2_undo'
    return F, 'none'


def v72_selftest():
    """Synthetic rows for every branch; raises on any mismatch."""
    key = lambda s: s.lower() if isinstance(s, str) and s.strip() and s != 'CCO' else None
    on = dict(b1=True, b2=True)
    F, F0 = ['A', 'X', 'B'], ['A', 'B', 'X']
    ice, gl = {'A': 0.2, 'X': 0.5, 'B': 0.4}, {'A': 0.1, 'X': 0.6, 'B': 0.3}
    assert v72_row(F, F0, ['X'], ice, gl, key, on) == (['X', 'A', 'B'], 'b1_swap')
    assert v72_row(F, F0, ['X'], ice, dict(gl, A=0.9), key, on) == (F, 'none')               # GLACIER disagrees, X > B on ICE only
    assert v72_row(F, F0, ['X'], dict(ice, A=0.9, B=0.9), dict(gl, A=0.9, B=0.9), key, on) == (F0, 'b2_undo')
    assert v72_row(F, F0, ['X'], {'A': 0.2}, gl, key, on) == (F, 'none')                       # missing score: untouched
    assert v72_row(F, F0, ['X'], {'A': 0.2, 'X': float('nan'), 'B': 0.1}, gl, key, on) == (F, 'none')
    assert v72_row(['A', 'x', 'B'], F0, ['X'], ice, gl, key, on) == (['x', 'A', 'B'], 'b1_swap')   # twin SMILES of x2 at rank 2: scores of x2 itself
    assert v72_row(F, F0, ['X'], ice, gl, key, dict(b1=False, b2=False)) == (F, 'none')
    assert v72_row(F, F, ['X'], dict(ice, A=0.9, B=0.9), dict(gl, A=0.9, B=0.9), key, on) == (F, 'none')   # E2R2 did not change it
    assert v72_row(['X', 'A'], ['X', 'A'], ['X'], ice, gl, key, on) == (['X', 'A'], 'none')   # engines agree
    assert v72_row(['CCO'], [], ['X'], ice, gl, key, on) == (['CCO'], 'skipped')
    assert v72_row(F, F0, ['', 'CCO', 'bad;smi'], ice, gl, key, on) == (F, 'no_x2')
    assert v72_row(F, F0, [], ice, gl, key, on) == (F, 'no_x2')


def _v72_main(scope, workdir='.'):
    import hashlib, json, os
    from collections import Counter
    from pathlib import Path
    import pandas as pd
    sw = dict(scope.get('V72') or {})
    stat = dict(sw, selftest='not run', status='off', rows=0, b1_swaps=0, b2_undos=0, fallback=0, changed=0, changed_top1=0)
    if not (sw.get('b1') or sw.get('b2')):
        print('V72_PATCH', json.dumps(stat), flush=True); return stat
    try:
        v72_selftest(); stat['selftest'] = 'ok'
    except Exception as exc:  # a failed self-test switches the patch off: the submission stays stack1's
        stat['selftest'] = 'FAILED ' + repr(exc)[:200]; stat['status'] = 'off (self-test)'
        print('V72_SELFTEST_FAILED patch switched off', repr(exc)[:300], flush=True)
        print('V72_PATCH', json.dumps(stat), flush=True); return stat
    root = Path(workdir); sub = root / 'submission.csv'; tmp = root / 'submission_v72.tmp.csv'
    base0 = scope.get('_e2r2_base_bytes'); e2 = scope.get('E2R2_STATS')
    if base0 is None or not isinstance(e2, dict):        # E2R2 did not apply: nothing to arbitrate, stack1's file stays
        stat['status'] = 'off (E2R2 not applied)'; print('V72_PATCH', json.dumps(stat), flush=True); return stat
    cur = sub.read_bytes()
    try:
        import io
        A = pd.read_csv(io.BytesIO(cur), dtype=str, keep_default_na=False)
        A0 = pd.read_csv(io.BytesIO(base0), dtype=str, keep_default_na=False)
        assert list(A.columns) == ['molecule_id', 'smiles'] and list(A.molecule_id) == list(A0.molecule_id)
        F0 = dict(zip(A0.molecule_id, A0.smiles))
        ident = scope['metric_identity']; ph = ident('CCO')

        def key(s):
            if not isinstance(s, str) or not s.strip() or s.strip() == 'CCO':
                return None
            k = ident(s)
            return None if k is None or k == ph else k
        eng = scope['ENG']
        assert isinstance(eng, dict), type(eng)
        ICE = scope.get('ICE_SCORES') or {}; GL = scope.get('GL_SCORES') or {}

        def raw(table, mid):
            if not isinstance(table, dict):
                return None
            v = table.get(mid)
            if v is None and mid.strip().lstrip('-').isdigit():
                v = table.get(int(mid))
            return v

        def per(table, mid):
            d = raw(table, mid)
            return d if isinstance(d, dict) else {}

        def engine_list(mid):
            e = raw(eng, mid)
            if isinstance(e, dict):
                return e.get('smiles') or []
            return list(e) if isinstance(e, (list, tuple)) else []
        tags, rows = Counter(), []
        for mid, s in zip(A.molecule_id, A.smiles):
            old = s.split(';')
            try:
                new, tag = v72_row(old, F0[mid].split(';'), engine_list(mid), per(ICE, mid), per(GL, mid), key, sw)
                if new != old:
                    ks = [key(t) for t in new]
                    assert 1 <= len(new) <= 25 and all(t.strip() and ';' not in t for t in new), 'v72 list invalid'
                    assert None not in ks[:2], 'v72 first rows unparseable'
                    if tag == 'b1_swap':
                        assert sorted(new) == sorted(old) and new[2:] == old[2:]
                    else:
                        assert tag == 'b2_undo' and new == F0[mid].split(';')
            except Exception as exc:
                stat['fallback'] += 1; print('V72_FALLBACK', mid, repr(exc)[:200], flush=True); new, tag = old, 'fallback'
            tags[tag] += 1
            stat['changed'] += int(new != old); stat['changed_top1'] += int(new[:1] != old[:1])
            rows.append((mid, ';'.join(new)))
        stat.update(rows=len(rows), b1_swaps=tags['b1_swap'], b2_undos=tags['b2_undo'], no_x2=tags['no_x2'], skipped=tags['skipped'],
                    base_sha256=hashlib.sha256(cur).hexdigest())
        if stat['changed']:
            out = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
            out.to_csv(tmp, index=False)
            back = pd.read_csv(tmp, dtype=str, keep_default_na=False)
            assert list(back.molecule_id) == list(A.molecule_id) and list(back.smiles) == list(out.smiles)
            assert back.smiles.map(lambda v: 1 <= len(v.split(';')) <= 25).all() and (back.smiles.str.strip() != '').all()
            os.replace(tmp, sub)
        stat['status'] = 'applied'; stat['submission_sha256'] = hashlib.sha256(sub.read_bytes()).hexdigest()
    except Exception as exc:  # file-level failure: stack1's file is the submission
        try:
            if sub.read_bytes() != cur:
                sub.write_bytes(cur)
            if tmp.exists():
                tmp.unlink()
        except Exception as exc2:
            print('V72_RESTORE_FAILED', repr(exc2)[:200], flush=True)
        stat['status'] = 'FAILED ' + repr(exc)[:200]
        print('V72_FAILED', repr(exc)[:300], '| stack1 file kept', flush=True)
    print('V72_PATCH', json.dumps(stat), flush=True)
    return stat


if __name__ == '__main__':
    V72_STATS = _v72_main(globals())
