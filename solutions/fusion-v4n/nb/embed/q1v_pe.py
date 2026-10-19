"""q1p cell (PE promotion after E2R2), appended verbatim as the last cell of casmi26-sub-q1 by scratch/q1v/build_q1v.py.

Rule (scratch/q1v/PREREG.md; Fable round 3 #1): for a molecule whose engine-1 library match is weak (BASE lib_max < LIB_TAU, the gate
under which q1's cell 24 merges the PubChem channel), if the PubChem channel's first row (PC[m]['pc'][0], after cell 22's ICE / GLACIER
re-rank) and engine 2's first row (ENG[m]['smiles'][0]) have the same metric identity (q1's metric_identity: stereo stripped,
tautomer-canonical InChIKey-14 + heavy-atom composition) and that identity is not at rank 1 of q1's final list, it moves to rank 1
(the list's own token, or engine 2's string when absent); other rows keep their order, twins are dropped, 25 rows kept.
Switch PE (promote False = this cell writes nothing: the submission is q1's byte for byte). Self-test on synthetic rows first
(fails -> off); per-molecule error -> q1's row; file-level error -> q1's file restored. Log: PE_PATCH {...}.
"""
PE = dict(promote=False)


def pe_first(lst, placeholder='CCO'):
    """First row of a channel list, or None (no list, empty, or a row that is not a usable SMILES token)."""
    if isinstance(lst, str):
        lst = [lst]
    if not lst:
        return None
    s = lst[0]
    if not isinstance(s, str) or s.strip() in ('', placeholder) or any(c in s for c in ';\r\n'):
        return None
    return s.strip()


def pe_row(F, pc_list, eng_list, lib_max, tau, key, sw, placeholder='CCO', limit=25):
    """One final list F (best first) -> (new list, tag, moved_from). Pure.
    tag in off, skipped, lib_strong, no_pc, no_eng, invalid, disagree, already_top, promoted."""
    F = list(F)
    if not sw.get('promote'):
        return F, 'off', None
    if not F or not isinstance(F[0], str) or F[0].strip() in ('', placeholder):
        return F, 'skipped', None
    try:
        lm = float(lib_max)
    except (TypeError, ValueError):
        return F, 'lib_strong', None
    if not (lm == lm and lm < tau):
        return F, 'lib_strong', None
    if not pc_list:
        return F, 'no_pc', None
    if not eng_list:
        return F, 'no_eng', None
    p, e = pe_first(pc_list, placeholder), pe_first(eng_list, placeholder)
    kp = key(p) if p is not None else None
    ke = key(e) if e is not None else None
    if kp is None or ke is None:
        return F, 'invalid', None
    if kp != ke:
        return F, 'disagree', None
    keys = [key(t) for t in F]
    if keys[0] == kp:
        return F, 'already_top', None
    hits = [i for i, k in enumerate(keys) if k == kp]
    top = F[hits[0]] if hits else e
    new = [top] + [t for t, k in zip(F, keys) if k != kp]
    return new[:limit], 'promoted', (str(hits[0] + 1) if hits else 'absent')


def pe_selftest(real_key=None):
    """Toy cases; with `real_key` (q1's metric identity as E2R2 keys it) also RDKit stereo / tautomer twins."""
    key = lambda s: s.lower() if isinstance(s, str) and s.strip() and s != 'CCO' else None
    on, tau = dict(promote=True), 0.9
    F = ['A', 'X', 'B', 'C']
    assert pe_row(F, ['X', 'Q'], ['x', 'A'], 0.5, tau, key, on) == (['X', 'A', 'B', 'C'], 'promoted', '2')
    assert pe_row(F, ['c'], ['C'], 0.5, tau, key, on) == (['C', 'A', 'X', 'B'], 'promoted', '4')      # list's own token moves
    assert pe_row(F, ['z'], ['Z'], 0.5, tau, key, on) == (['Z', 'A', 'X', 'B', 'C'], 'promoted', 'absent')   # engine 2's string
    assert pe_row(['A', 'X', 'b', 'B'], ['b'], ['B'], 0.5, tau, key, on) == (['b', 'A', 'X'], 'promoted', '3')   # twins dropped
    assert pe_row(F, ['a'], ['A'], 0.5, tau, key, on) == (F, 'already_top', None)
    assert pe_row(F, ['X'], ['B'], 0.5, tau, key, on) == (F, 'disagree', None)
    assert pe_row(F, ['X'], ['X'], 0.9, tau, key, on) == (F, 'lib_strong', None)                       # strict < LIB_TAU
    assert pe_row(F, ['X'], ['X'], float('nan'), tau, key, on) == (F, 'lib_strong', None)
    assert pe_row(F, ['X'], ['X'], None, tau, key, on) == (F, 'lib_strong', None)
    assert pe_row(F, [], ['X'], 0.5, tau, key, on) == (F, 'no_pc', None)
    assert pe_row(F, ['X'], [], 0.5, tau, key, on) == (F, 'no_eng', None)
    assert pe_row(F, ['CCO', 'X'], ['X'], 0.5, tau, key, on) == (F, 'invalid', None)                  # first row only, no fall-through
    assert pe_row(F, ['X;Y'], ['X'], 0.5, tau, key, on) == (F, 'invalid', None)
    assert pe_row(F, ['X'], ['X'], 0.5, tau, key, dict(promote=False)) == (F, 'off', None)
    assert pe_row(['CCO'], ['X'], ['X'], 0.5, tau, key, on) == (['CCO'], 'skipped', None)
    L = [chr(65 + i) for i in range(25)]
    assert pe_row(L, ['z'], ['z'], 0.1, tau, key, on) == (['z'] + L[:24], 'promoted', 'absent')
    assert pe_row(L, ['Y'], ['Y'], 0.1, tau, key, on) == (['Y'] + L[:24], 'promoted', '25')
    if real_key is not None:
        A, B = 'C=C(C)C1CCC2CCC(=O)C=C2C1', 'C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1'   # stereo twin (metric regression pair)
        K, T = 'CC(=O)CC(=O)C', 'CC(O)=CC(=O)C'                                   # tautomers
        G = ['Oc1ccccc1', B, 'NCC(=O)O']
        assert pe_row(G, [A], [B], 0.5, tau, real_key, on) == ([B, 'Oc1ccccc1', 'NCC(=O)O'], 'promoted', '2')
        assert pe_row(G + [K], [T], [K], 0.5, tau, real_key, on) == ([K] + G, 'promoted', '4')
        assert pe_row(G, [A], ['NCC(=O)O'], 0.5, tau, real_key, on)[1] == 'disagree'
        assert pe_row(G, ['CCO'], ['CCO'], 0.5, tau, real_key, on)[1] == 'invalid'
    return 'ok'


def _pe_main(scope, workdir='.'):
    import hashlib, io, json, os
    from collections import Counter
    from pathlib import Path
    import pandas as pd
    sw = dict(scope.get('PE') or {})
    stat = dict(sw, selftest='not run', status='off', rows=0, promoted=0, fallback=0, changed=0, changed_top1=0)
    root = Path(workdir); sub = root / 'submission.csv'; tmp = root / 'submission_pe.tmp.csv'
    if not sw.get('promote'):
        print('PE_PATCH', json.dumps(stat), flush=True); return stat
    try:
        ident0 = scope['metric_identity']; ph0 = ident0('CCO')
        stat['selftest'] = pe_selftest(lambda s: None if not isinstance(s, str) or not s.strip() or s.strip() == 'CCO'
                                       or ident0(s) in (None, ph0) else ident0(s))
    except Exception as exc:
        stat['selftest'] = 'FAILED ' + repr(exc)[:200]; stat['status'] = 'off (self-test)'
        print('PE_SELFTEST_FAILED patch switched off', repr(exc)[:300], flush=True)
        print('PE_PATCH', json.dumps(stat), flush=True); return stat
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
        BASE, PC, ENG = scope['BASE'], scope['PC'], scope['ENG']
        tau = float(scope.get('LIB_TAU', 0.9))
        assert isinstance(BASE, dict) and len(BASE) > 0 and isinstance(PC, dict) and isinstance(ENG, dict), 'pe inputs'
        assert abs(tau - 0.9) < 1e-9, tau
        stat['lib_tau'] = tau

        def get(d, mid):
            v = d.get(mid)
            if v is None and mid.strip().lstrip('-').isdigit():
                v = d.get(int(mid))
            return v

        def inputs(mid):
            b, p, e = get(BASE, mid), get(PC, mid), get(ENG, mid)
            lm = b[2] if b is not None else None
            pcl = list(p.get('pc') or []) if isinstance(p, dict) else []
            el = list(e.get('smiles') or []) if isinstance(e, dict) else (list(e) if isinstance(e, (list, tuple)) else [])
            return pcl, el, lm
        tags, moved, rows, examples = Counter(), Counter(), [], []
        for mid, s in zip(A.molecule_id, A.smiles):
            old = s.split(';')
            try:
                pcl, el, lm = inputs(mid)
                new, tag, mf = pe_row(old, pcl, el, lm, tau, key, sw)
                if new != old:
                    assert tag == 'promoted' and 1 <= len(new) <= 25 and all(t.strip() and ';' not in t for t in new), 'pe list invalid'
                    k = key(new[0]); assert k is not None and sum(key(t) == k for t in new) == 1, 'pe identity'
                    assert [t for t in new[1:]] == [t for t in old if key(t) != k][:len(new) - 1], 'pe order'
                    if mf is not None:
                        moved[mf] += 1
                    if len(examples) < 3:
                        examples.append(dict(molecule_id=mid, moved_from=mf, before=old[:3], after=new[:3]))
            except Exception as exc:
                stat['fallback'] += 1; print('PE_FALLBACK', mid, repr(exc)[:200], flush=True); new, tag = old, 'fallback'
            tags[tag] += 1
            stat['changed'] += int(new != old); stat['changed_top1'] += int(new[:1] != old[:1])
            rows.append((mid, ';'.join(new)))
        stat.update(rows=len(rows), promoted=tags['promoted'], moved_from=dict(moved), already_top=tags['already_top'],
                    disagree=tags['disagree'], lib_strong=tags['lib_strong'], no_pc=tags['no_pc'], no_eng=tags['no_eng'],
                    invalid=tags['invalid'], skipped=tags['skipped'], base_sha256=hashlib.sha256(cur).hexdigest())
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
            print('PE_RESTORE_FAILED', repr(exc2)[:200], flush=True)
        stat['status'] = 'FAILED ' + repr(exc)[:200]
        print('PE_FAILED', repr(exc)[:300], '| q1 file kept', flush=True)
    print('PE_PATCH', json.dumps(stat), flush=True)
    for ex in (examples if stat.get('status') == 'applied' else []):
        print('PE_EXAMPLE', json.dumps(ex), flush=True)
    return stat


if __name__ == '__main__':
    PE_STATS = _pe_main(globals())
