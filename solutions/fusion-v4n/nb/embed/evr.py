# ---- v72b / EVR (ours): evidence-row rescue in promoted PubChem lists ----
# Appended after the PCF5 cell (v72a); every earlier cell stays byte-identical. Cell 24 decides a promotion on the
# PubChem channel's ORIGINAL evidence (S and top_pop of the S-order row 1, and the raw-f.z margin; cell 22 l.165-167 pins
# pc_fz[0] so that 'gate decisions are unchanged'), but promote() (cell 24 l.28-34) puts the ICE/GL re-ordered pc[0] = P
# first. When P differs from the S-order top O (cell 15's pc_lists.json, never modified on disk), stack1's fixed slots
# push O to slot >= 8 or out of the list. EVR puts O at rank max(5, rank(P) + 1) when it sits lower.
# Only promoted lists with P != O can change, and only at ranks >= 5 (rank(P), E2R2's rank 2 and every rank above O's
# new place are kept). No gate decision, PubChem field or non-promoted list changes.
# Commit run (IS_RERUN False): any error raises. Scoring rerun: any error restores this cell's input bytes (the PCF5
# output) and prints EVR_FAILED.


def evr_segment(mid, PC, BASE, LIB_TAU, S_TAU, POP_TAU, rel_th=150.0):
    """Exact mirror of stack1 cell 24 lines 49-68 (same as PCF5's). Returns (segment, PubChem dict or None)."""
    import math
    smis, keys, lib_max = tuple(BASE.get(mid, ([], [], 0.0)))[:3]
    p = PC.get(mid)
    if p is None or not p.get('pc'):
        return 'no_pc', None
    if lib_max >= LIB_TAU:
        return 'gate_closed', p
    bp = p['best_pool_fz']
    fin = bp is not None and math.isfinite(float(bp))
    rel = ((p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]) - bp) if fin else 1e9
    S = p.get('S')
    pop_top = float(p.get('top_pop') or 0.0)
    if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) or (rel > rel_th):
        return 'promoted', p
    return 'open', p


def evr_row(toks, O, P, key, floor=5, limit=25, placeholder='CCO'):
    """Put O at rank k = max(floor, rank(P) + 1) of one final list unless O's metric identity already sits at a rank <= k.
    Pure. Returns (new_toks, tag, moved_from); tag in skipped | invalid_O | invalid_P | same_as_P | no_P |
    already_above | no_room | moved | inserted."""
    toks = list(toks)
    if not toks or not isinstance(toks[0], str) or toks[0].strip() in ('', placeholder):
        return toks, 'skipped', None
    if not isinstance(O, str) or not O.strip() or any(c in O for c in ';\r\n'):
        return toks, 'invalid_O', None
    o = O.strip()
    ko = key(o)
    if ko is None:
        return toks, 'invalid_O', None
    kp = key(P) if isinstance(P, str) else None
    if kp is None:
        return toks, 'invalid_P', None
    if ko == kp:
        return toks, 'same_as_P', None
    keys = [key(t) for t in toks]
    rp = next((i + 1 for i, k in enumerate(keys) if k == kp), None)
    if rp is None:
        return toks, 'no_P', None
    k = max(int(floor), rp + 1)
    ro = next((i + 1 for i, kk in enumerate(keys) if kk == ko), None)
    if ro is not None and ro <= k:
        return toks, 'already_above', str(ro)
    rest = [t for t, kk in zip(toks, keys) if kk != ko]
    at = min(k - 1, len(rest))
    new = (rest[:at] + [o] + rest[at:])[:limit]
    if new == toks or ko not in [key(t) for t in new]:
        return toks, 'no_room', None
    return new, ('moved' if ro else 'inserted'), (str(ro) if ro else 'absent')


def evr_rows(rows, active, PC, BASE, ORIG, PRE, key, LIB_TAU, S_TAU, POP_TAU, floor=5):
    """Apply EVR to [(molecule_id, 'a;b;c')]. Pure. Returns (new_rows, fired ids, tag counts, per-molecule audit)."""
    from collections import Counter
    tags, out, fired, audit = Counter(), [], [], []
    for mid, s in rows:
        if mid not in active:
            out.append((mid, s))
            continue
        seg, p = evr_segment(mid, PC, BASE, LIB_TAU, S_TAU, POP_TAU)
        if seg != 'promoted':
            out.append((mid, s))
            continue
        fired.append(mid)
        old = s.split(';')
        P = p['pc'][0]
        o_list = (ORIG.get(mid) or {}).get('pc') or []
        pre = PRE.get(mid) or []
        if not o_list:
            tag, new, moved_from, O = 'no_orig', old, None, None
        elif sorted(o_list) != sorted(p['pc']):           # cell 22 only permutes the list
            tag, new, moved_from, O = 'not_permutation', old, None, o_list[0]
        elif not pre or pre[0] != P:                      # cell 24 promote() put P first
            tag, new, moved_from, O = 'prefusion_mismatch', old, None, o_list[0]
        else:
            O = o_list[0]
            new, tag, moved_from = evr_row(old, O, P, key, floor=floor)
        tags[tag] += 1
        audit.append(dict(molecule_id=mid, tag=tag, moved_from=moved_from, P=P, O=O))
        out.append((mid, ';'.join(new)))
    return out, fired, tags, audit


def _evr_apply(scope, base_bytes, floor=5, workdir='.'):
    """Apply evr_rows to every row of the current submission.csv, run the same-run control and the checks, then replace
    the file (temp file, re-read, os.replace)."""
    import hashlib
    import json
    import os
    import time
    from collections import Counter
    from pathlib import Path
    import pandas as pd

    t_start = time.time()
    root = Path(workdir)
    sub_path, tmp_path = root / 'submission.csv', root / 'submission_evr.tmp.csv'
    (root / 'submission_base_evr.csv').write_bytes(base_bytes)
    base = pd.read_csv(root / 'submission_base_evr.csv', dtype=str, keep_default_na=False)
    sample = pd.read_csv(os.path.join(scope['COMP'], 'sample_submission.csv'), dtype=str, keep_default_na=False)
    key = scope['e2r2_make_key'](scope['metric_identity'])
    active = set(scope['te'].molecule_id.astype(str))
    strict = not bool(scope.get('IS_RERUN', True))
    PC, BASE, PRE = scope['PC'], scope['BASE'], scope['PREFUSION_LISTS']
    assert isinstance(PC, dict) and isinstance(BASE, dict) and isinstance(PRE, dict), (type(PC), type(BASE), type(PRE))
    LIB_TAU, S_TAU, POP_TAU = float(scope['LIB_TAU']), float(scope['S_TAU']), float(scope['POP_TAU'])
    orig_path = '/kaggle/working/pc_lists.json' if os.path.isfile('/kaggle/working/pc_lists.json') else str(root / 'pc_lists.json')
    with open(orig_path, encoding='utf-8') as f:            # written only by cell 15's pc_runner; cell 22 mutates PC in memory only
        ORIG = json.load(f)
    assert isinstance(ORIG, dict), type(ORIG)

    def check(frame):
        assert list(frame.columns) == ['molecule_id', 'smiles'], list(frame.columns)
        assert list(frame.molecule_id) == list(base.molecule_id)
        assert len(frame) == len(sample) and frame.molecule_id.is_unique
        assert set(frame.molecule_id) == set(sample.molecule_id)
        assert frame.smiles.notna().all() and (frame.smiles.str.strip() != '').all()
        assert frame.smiles.map(lambda v: 1 <= len(v.split(';')) <= 25 and all(t.strip() for t in v.split(';'))).all()

    def n_dup(ts):
        ks = [k for k in map(key, ts) if k is not None]
        return len(ks) - len(set(ks))

    check(base)
    rows = list(zip(base.molecule_id, base.smiles))
    new_rows, fired, tags, audit = evr_rows(rows, active, PC, BASE, ORIG, PRE, key, LIB_TAU, S_TAU, POP_TAU, floor=floor)

    # same-run control: the replica fires exactly on cell 24's promotions
    st = scope.get('stats')
    assert isinstance(st, dict) and 'promoted' in st, st
    assert len(fired) == int(st['promoted']), ('EVR promotion replica differs from cell 24 stats', len(fired), st)
    p5 = scope.get('PCF5_STATS')
    if isinstance(p5, dict) and isinstance(p5.get('segments'), dict):
        assert int(p5['segments'].get('promoted', 0)) == len(fired), ('EVR vs PCF5 promoted count', p5['segments'], len(fired))

    fired_set = set(fired)
    aud = {a['molecule_id']: a for a in audit}
    changed, moved_from, examples = 0, Counter(), []
    for (mid, s_old), (mid2, s_new) in zip(rows, new_rows):
        assert mid == mid2
        if s_old == s_new:
            continue
        old, new = s_old.split(';'), s_new.split(';')
        a = aud[mid]
        assert mid in fired_set and a['tag'] in ('inserted', 'moved'), (mid, a['tag'])
        ko = key(a['O'].strip())
        kp = key(a['P'])
        r_new = [key(t) for t in new].index(ko) + 1
        rp_old = [key(t) for t in old].index(kp) + 1
        k = max(int(floor), rp_old + 1)
        assert r_new == min(k, len(new)), (mid, r_new, k)                    # a short list gets O appended
        assert new[:r_new - 1] == old[:r_new - 1], mid                        # every rank above O's new place is kept
        assert sum(key(t) == ko for t in new) == 1, mid
        assert [t for t in new if key(t) != ko] == [t for t in old if key(t) != ko][:len(new) - 1], mid
        assert n_dup(new) <= n_dup(old) and 1 <= len(new) <= 25 and all(t.strip() for t in new), mid
        changed += 1
        moved_from[a['moved_from']] += 1
        if len(examples) < 3:
            examples.append(dict(molecule_id=mid, tag=a['tag'], moved_from=a['moved_from'], k=k, before=old[:6], after=new[:6]))
    result = pd.DataFrame(new_rows, columns=['molecule_id', 'smiles'])
    check(result)
    for (mid, s_old), (_, s_new) in zip(rows, new_rows):
        if mid not in fired_set:
            assert s_old == s_new, f'non-promoted row changed: {mid}'

    stats = dict(cell='EVR', version='v72b', floor=int(floor), rel_th=150.0, is_rerun=not strict, rows=len(base),
                 active=len(active), orig_lists=len(ORIG), fired=len(fired), promoted_cell24=int(st['promoted']),
                 p_ne_o=int(sum(tags.get(t, 0) for t in ('already_above', 'no_room', 'moved', 'inserted', 'no_P'))),
                 consistency_skips=int(sum(tags.get(t, 0) for t in ('no_orig', 'not_permutation', 'prefusion_mismatch'))),
                 tags=dict(sorted(tags.items())),
                 moved_from=dict(sorted(moved_from.items())), changed_rows=changed,
                 base_sha256=hashlib.sha256(base_bytes).hexdigest())
    new_bytes = base_bytes
    if changed:
        result.to_csv(tmp_path, index=False)
        back = pd.read_csv(tmp_path, dtype=str, keep_default_na=False)
        check(back)
        assert list(back.smiles) == list(result.smiles) and list(back.molecule_id) == list(result.molecule_id)
        new_bytes = tmp_path.read_bytes()
    stats['submission_sha256'] = hashlib.sha256(new_bytes).hexdigest()
    stats['seconds'] = round(time.time() - t_start, 2)
    (root / 'evr_stats.json').write_text(json.dumps(dict(stats, status='applied', examples=examples,
                                                         molecules=audit), indent=1) + '\n')
    if changed:
        os.replace(tmp_path, sub_path)
    print('EVR_APPLIED', json.dumps(stats), flush=True)
    for ex in examples:
        print('EVR_EXAMPLE', json.dumps(ex), flush=True)
    return stats


# ==== CELL BODY (executes) ====
import json as _ev_json
import os as _ev_os
import sys as _ev_sys
import time as _ev_time
import traceback as _ev_tb
EVR_FLOOR = 5                     # same rank floor as PCF5; chosen offline (diag displacement cost), never on the LB
EVR_DEADLINE_S = 8.5 * 3600
_EV_STRICT = not bool(globals().get('IS_RERUN', True))
_ev_base_bytes = None
EVR_STATS = None
try:
    if _ev_time.time() - float(globals().get('T0', _ev_time.time())) > EVR_DEADLINE_S:
        raise TimeoutError(f'EVR deadline: more than {EVR_DEADLINE_S:.0f} s since T0, cell skipped')
    with open('submission.csv', 'rb') as _ev_f:      # the PCF5 output (cwd is /kaggle/working)
        _ev_base_bytes = _ev_f.read()
    print('EVR reads', _ev_os.path.abspath('submission.csv'), len(_ev_base_bytes), 'bytes', flush=True)
    EVR_STATS = _evr_apply(globals(), _ev_base_bytes, floor=EVR_FLOOR)
except Exception as _ev_exc:
    if _EV_STRICT:
        raise
    _ev_state = 'not read (left as it was)'
    if _ev_base_bytes is not None:
        try:
            _ev_now = None
            if _ev_os.path.isfile('submission.csv'):
                with open('submission.csv', 'rb') as _ev_f:
                    _ev_now = _ev_f.read()
            if _ev_now != _ev_base_bytes:
                with open('submission.csv', 'wb') as _ev_f:
                    _ev_f.write(_ev_base_bytes)
            _ev_state = 'untouched' if _ev_now == _ev_base_bytes else 'restored from the input bytes'
        except Exception as _ev_exc2:
            _ev_state = f'RESTORE FAILED {_ev_exc2!r}'
    try:
        if _ev_os.path.exists('submission_evr.tmp.csv'):
            _ev_os.remove('submission_evr.tmp.csv')
        with open('evr_stats.json', 'w') as _ev_f:
            _ev_f.write(_ev_json.dumps(dict(cell='EVR', status='failed', submission=_ev_state,
                                            error=f'{type(_ev_exc).__name__}: {_ev_exc}'[:500])) + '\n')
    except Exception:
        pass
    try:
        print('!' * 100, flush=True)
        print(f'EVR_FAILED {type(_ev_exc).__name__}: {str(_ev_exc)[:300]} | submission.csv {_ev_state} '
              '| the input list (stack1 + E2R2 + PCF5) is the submission', flush=True)
        _ev_tb.print_exc(file=_ev_sys.stdout)
        print('!' * 100, flush=True)
    except Exception:
        print('EVR_FAILED (error report failed) | submission.csv', _ev_state, flush=True)
