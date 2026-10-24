"""s1m cell (engine-1 margin shield), appended verbatim as the last code cell of s1 (after the V43 receipt); its switch and capture live
in S1M_CAPTURE_CELL, inserted right after s1's base-list cell 40.

Rule (Fable, debate round 4): when engine 1's own score margin between its ranks 1 and 2 exceeds a fixed threshold tau, engine 1's
leading row goes back to rank 1 of the final list; the rank-2 slot keeps E2R2's choice (engine 2's leading structure, as cell 54 placed
it), every other row keeps its order (metric twins removed, list cut at 25). The result equals E2R2 re-applied after the shield.
Margin = cell 40's score of base rank 1 minus base rank 2 (score = z of the ranker over the <= 60 base rows + POOLPOP_MU * pool
popularity, the value engine 1 orders by), captured before cell 42 truncates BASE to (smiles, keys, lib_max). tau is fixed label-free
as the 80th percentile of that margin over the s1val dump's lists (scratch/s1var/PREREG.md) and written into S1M by the builder.
Switch S1M (shield False = nothing captured, this cell writes nothing: the submission is s1's byte for byte). Self-test first (fails ->
off); per-molecule error -> s1's row; file-level error -> s1's file restored. Log: S1M_PATCH {...}.
"""

S1M_CAPTURE_CELL = r'''"""s1m (ours): switch, and a read-only capture of engine 1's leading row and its score margin over rank 2 (cell 40's scores) before
the simulator cell truncates BASE. The shield itself is the last code cell (s1m_shield.py)."""
S1M = __S1M__
S1M_E1 = {}
if S1M.get('shield'):
    try:
        for _s1m_m, _s1m_v in BASE.items():
            _s1m_s, _s1m_sc = list(_s1m_v[0]), [float(x) for x in _s1m_v[3]]
            S1M_E1[str(_s1m_m)] = dict(top=_s1m_s[0] if _s1m_s else None,
                                       margin=(_s1m_sc[0] - _s1m_sc[1]) if len(_s1m_sc) >= 2 and len(_s1m_s) >= 2 else None)
    except Exception as _s1m_e:
        S1M_E1 = {}
        print('S1M_CAPTURE_FAILED', repr(_s1m_e)[:300], flush=True)
    print('S1M_CAPTURE', json.dumps(dict(S1M, molecules=len(S1M_E1))), flush=True)
'''


def s1m_margin(scores, smiles=None):
    """Engine 1's margin from its ordered score list (None if fewer than 2 rows or non-finite)."""
    try:
        if scores is None or len(scores) < 2 or (smiles is not None and len(smiles) < 2):
            return None
        m = float(scores[0]) - float(scores[1])
    except (TypeError, ValueError):
        return None
    return m if m == m and abs(m) != float('inf') else None


def s1m_tau(margins, q=80.0):
    """Label-free threshold: the q-th percentile (numpy linear interpolation) of the defined margins."""
    import numpy as np
    x = np.asarray([m for m in margins if m is not None and m == m], np.float64)
    assert len(x) >= 10, ('too few margins', len(x))
    return float(np.percentile(x, q))


def s1m_row(F, top, margin, tau, e2_top, key, sw, placeholder='CCO'):
    """One final list F (after E2R2) -> (new list, tag). Pure. tag in off, skipped, no_margin, below, no_top, kept, shield."""
    F = list(F)
    if not sw.get('shield'):
        return F, 'off'
    if not F or not isinstance(F[0], str) or F[0].strip() in ('', placeholder):
        return F, 'skipped'
    if margin is None or tau is None:
        return F, 'no_margin'
    try:
        mg, tu = float(margin), float(tau)
    except (TypeError, ValueError):
        return F, 'no_margin'
    if not (mg == mg and tu == tu):
        return F, 'no_margin'
    if not mg > tu:
        return F, 'below'
    b = top.strip() if isinstance(top, str) else None
    kb = key(b) if b and b != placeholder and not any(c in b for c in ';\r\n') else None
    if kb is None:
        return F, 'no_top'
    if key(F[0]) == kb:
        return F, 'kept'
    head = [next((t for t in F if key(t) == kb), b)]
    e = e2_top.strip() if isinstance(e2_top, str) else None
    ke = key(e) if e and e != placeholder and not any(c in e for c in ';\r\n') else None
    drop = {kb}
    if ke is not None and ke != kb:
        head.append(next((t for t in F if key(t) == ke), e))
        drop.add(ke)
    new = head + [t for t in F if key(t) not in drop]
    return new[:25], 'shield'


def s1m_selftest():
    key = lambda s: s.lower() if isinstance(s, str) and s.strip() and s != 'CCO' else None
    on = dict(shield=True)
    F = ['X', 'E', 'B', 'Y']                                                   # s1 final: E2R2 put E at rank 2
    assert s1m_row(F, 'B', 2.0, 1.0, 'E', key, on) == (['B', 'E', 'X', 'Y'], 'shield')
    assert s1m_row(['E', 'X', 'B'], 'B', 2.0, 1.0, 'E', key, on) == (['B', 'E', 'X'], 'shield')   # E2R2 already_top
    assert s1m_row(['X', 'B', 'Y'], 'b', 2.0, 1.0, 'B', key, on) == (['B', 'X', 'Y'], 'shield')   # engine 2 top = engine 1 top (twin)
    assert s1m_row(['X', 'E', 'Y'], 'Z', 2.0, 1.0, 'E', key, on) == (['Z', 'E', 'X', 'Y'], 'shield')   # absent: inserted
    assert s1m_row(['X', 'Y'], 'B', 2.0, 1.0, None, key, on) == (['B', 'X', 'Y'], 'shield')       # no engine-2 list
    assert s1m_row(['X', 'Y'], 'B', 2.0, 1.0, 'E', key, on) == (['B', 'E', 'X', 'Y'], 'shield')   # E2R2 had not run: E inserted
    assert s1m_row(F, 'x', 2.0, 1.0, 'E', key, on) == (F, 'kept')
    assert s1m_row(F, 'B', 1.0, 1.0, 'E', key, on) == (F, 'below')                 # strict >
    assert s1m_row(F, 'B', None, 1.0, 'E', key, on) == (F, 'no_margin')
    assert s1m_row(F, 'B', float('nan'), 1.0, 'E', key, on) == (F, 'no_margin')
    assert s1m_row(F, 'B', 2.0, None, 'E', key, on) == (F, 'no_margin')
    assert s1m_row(F, None, 2.0, 1.0, 'E', key, on) == (F, 'no_top')
    assert s1m_row(F, 'CCO', 2.0, 1.0, 'E', key, on) == (F, 'no_top')
    assert s1m_row(F, 'B', 2.0, 1.0, 'E', key, dict(shield=False)) == (F, 'off')
    assert s1m_row(['CCO'], 'B', 2.0, 1.0, 'E', key, on) == (['CCO'], 'skipped')
    L = [chr(65 + i) for i in range(25)]
    assert s1m_row(L, 'z', 5.0, 1.0, 'y', key, on) == (['z', 'Y'] + [t for t in L if t != 'Y'][:23], 'shield')
    assert s1m_margin([3.0, 1.5, 0.0]) == 1.5 and s1m_margin([3.0]) is None and s1m_margin([1.0, float('nan')]) is None
    assert abs(s1m_tau([float(i) for i in range(11)] + [None]) - 8.0) < 1e-12
    return 'ok'


def _s1m_main(scope, workdir='.'):
    import hashlib, io, json, os
    from collections import Counter
    from pathlib import Path
    import pandas as pd
    sw = dict(scope.get('S1M') or {})
    stat = dict(sw, selftest='not run', status='off', rows=0, shield=0, fallback=0, changed=0, changed_top1=0)
    root = Path(workdir); sub = root / 'submission.csv'; tmp = root / 'submission_s1m.tmp.csv'
    if not sw.get('shield'):
        print('S1M_PATCH', json.dumps(stat), flush=True); return stat
    try:
        s1m_selftest(); stat['selftest'] = 'ok'
        assert sw.get('tau') is not None and float(sw['tau']) == float(sw['tau']), 'S1M tau missing'
    except Exception as exc:
        stat['selftest'] = 'FAILED ' + repr(exc)[:200]; stat['status'] = 'off (self-test)'
        print('S1M_SELFTEST_FAILED patch switched off', repr(exc)[:300], flush=True)
        print('S1M_PATCH', json.dumps(stat), flush=True); return stat
    cur = sub.read_bytes()
    try:
        A = pd.read_csv(io.BytesIO(cur), dtype=str, keep_default_na=False)
        assert list(A.columns) == ['molecule_id', 'smiles']
        ident = scope['metric_identity']; ph = ident('CCO')

        def key(s):
            if not isinstance(s, str) or not s.strip() or s.strip() == 'CCO':
                return None
            k = ident(s)
            return None if k is None or k == ph else k
        E1 = scope.get('S1M_E1') or {}
        assert isinstance(E1, dict) and len(E1) > 0, 'S1M_E1 empty (capture cell)'
        ENG = scope['ENG'] if isinstance(scope.get('ENG'), dict) else {}
        tau = float(sw['tau'])
        tags, rows = Counter(), []
        for mid, s in zip(A.molecule_id, A.smiles):
            old = s.split(';')
            try:
                e1 = E1.get(str(mid)) or {}
                listed = [x for x in ((ENG.get(str(mid)) or {}).get('smiles') or []) if x and x != 'CCO']   # = cell 54's engine top
                new, tag = s1m_row(old, e1.get('top'), e1.get('margin'), tau, listed[0] if listed else None, key, sw)
                if new != old:
                    assert 1 <= len(new) <= 25 and all(t.strip() and ';' not in t for t in new), 's1m list invalid'
                    ids = [key(t) for t in new]
                    assert None not in ids and len(ids) == len(set(ids)), 's1m identities'
                    lost = set(old) - set(new)
                    assert lost <= set(old[23:]) | {t for t in old if key(t) in set(ids[:2])}, 's1m dropped rows'
            except Exception as exc:
                stat['fallback'] += 1; print('S1M_FALLBACK', mid, repr(exc)[:200], flush=True); new, tag = old, 'fallback'
            tags[tag] += 1
            stat['changed'] += int(new != old); stat['changed_top1'] += int(new[:1] != old[:1])
            rows.append((mid, ';'.join(new)))
        stat.update(rows=len(rows), tags=dict(tags), shield=tags['shield'], base_sha256=hashlib.sha256(cur).hexdigest(),
                    captured=len(E1), margins_defined=sum(v.get('margin') is not None for v in E1.values()),
                    above_tau=sum(v.get('margin') is not None and v['margin'] > tau for v in E1.values()))
        if stat['changed']:
            out = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
            out.to_csv(tmp, index=False)
            back = pd.read_csv(tmp, dtype=str, keep_default_na=False)
            assert list(back.molecule_id) == list(A.molecule_id) and list(back.smiles) == list(out.smiles)
            assert back.smiles.map(lambda v: 1 <= len(v.split(';')) <= 25).all() and (back.smiles.str.strip() != '').all()
            os.replace(tmp, sub)
        stat['status'] = 'applied'; stat['submission_sha256'] = hashlib.sha256(sub.read_bytes()).hexdigest()
    except Exception as exc:
        try:
            if sub.read_bytes() != cur:
                sub.write_bytes(cur)
            if tmp.exists():
                tmp.unlink()
        except Exception as exc2:
            print('S1M_RESTORE_FAILED', repr(exc2)[:200], flush=True)
        stat['status'] = 'FAILED ' + repr(exc)[:200]
        print('S1M_FAILED', repr(exc)[:300], '| s1 file kept', flush=True)
    print('S1M_PATCH', json.dumps(stat), flush=True)
    return stat


if __name__ == '__main__':
    S1M_STATS = _s1m_main(globals())
