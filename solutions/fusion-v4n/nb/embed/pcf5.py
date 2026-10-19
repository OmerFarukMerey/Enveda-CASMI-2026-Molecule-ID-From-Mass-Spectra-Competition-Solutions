# ---- v72a / PCF5 (ours): PubChem leader floor at rank 5 in every list that stack1 does not promote ----
# Appended after stack1 cell 36 (E2R2); stack1 cells 0-36 stay byte-identical. It reads the final E2R2 submission.csv and
# changes rows only at ranks >= 5:
#   gate_closed : cell 24 'untouched' (PubChem list non-empty, lib_max >= LIB_TAU). stack1 never lists a PubChem row
#                 there; the channel's S-order leader PC[mid]['pc'][0] (cell 22 never re-orders these lists) goes to
#                 rank 5.
#   open        : lib_max < LIB_TAU and not promoted by cell 24. The leader that cell 24 put in gentle slot 4 is lifted
#                 back to rank 5 when the engine fusion / GLACIER left it below rank 5.
#   promoted, no_pc, placeholders : unchanged.
# Ranks 1-4, every gate and promotion decision, every PubChem field and every upstream list stay as they are.
# Commit run (IS_RERUN False): any error raises. Scoring rerun: any error restores this cell's input bytes (stack1's
# final file) and prints PCF5_FAILED.


def pcf5_segment(mid, PC, BASE, LIB_TAU, S_TAU, POP_TAU, rel_th=150.0):
    """Exact mirror of stack1 cell 24 lines 49-68. Returns (segment, PubChem dict or None)."""
    import math
    smis, keys, lib_max = tuple(BASE.get(mid, ([], [], 0.0)))[:3]
    p = PC.get(mid)
    if p is None or not p.get('pc'):
        return 'no_pc', None
    if lib_max >= LIB_TAU:
        if p.get('pc_form') is not None:            # cell 22 l.97 sets pc_form only below the gate
            raise AssertionError(f'gate-closed PubChem list was re-ordered: {mid}')
        return 'gate_closed', p
    bp = p['best_pool_fz']
    fin = bp is not None and math.isfinite(float(bp))
    rel = ((p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]) - bp) if fin else 1e9
    S = p.get('S')
    pop_top = float(p.get('top_pop') or 0.0)
    if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) or (rel > rel_th):
        return 'promoted', p
    return 'open', p


def pcf5_row(toks, leader, key, pos=5, limit=25, placeholder='CCO'):
    """Put `leader` at 1-based rank `pos` of one final list unless its metric identity already sits at a rank <= pos.
    Pure. Returns (new_toks, tag, moved_from); tag in skipped | invalid | already_above | moved | inserted."""
    toks = list(toks)
    if not toks or not isinstance(toks[0], str) or toks[0].strip() in ('', placeholder):
        return toks, 'skipped', None
    if not isinstance(leader, str) or not leader.strip() or any(c in leader for c in ';\r\n'):
        return toks, 'invalid', None
    x = leader.strip()
    kx = key(x)
    if kx is None:
        return toks, 'invalid', None
    keys = [key(t) for t in toks]
    hits = [i for i, k in enumerate(keys) if k == kx]
    if hits and hits[0] < pos:
        return toks, 'already_above', str(hits[0] + 1)
    rest = [t for t, k in zip(toks, keys) if k != kx]
    at = min(pos - 1, len(rest))
    new = (rest[:at] + [x] + rest[at:])[:limit]
    return new, ('moved' if hits else 'inserted'), (str(hits[0] + 1) if hits else 'absent')


def pcf5_rows(rows, active, PC, BASE, key, LIB_TAU, S_TAU, POP_TAU, pos=5):
    """Apply PCF5 to [(molecule_id, 'a;b;c')]. Pure. Returns (new_rows, segment counts, tag counts, per-molecule audit)."""
    from collections import Counter
    seg_n, tags, out, audit = Counter(), Counter(), [], []
    for mid, s in rows:
        if mid not in active:
            out.append((mid, s))
            continue
        old = s.split(';')
        seg, p = pcf5_segment(mid, PC, BASE, LIB_TAU, S_TAU, POP_TAU)
        seg_n[seg] += 1
        new, tag, moved_from = old, 'untouched', None
        if seg in ('gate_closed', 'open'):
            new, tag, moved_from = pcf5_row(old, p['pc'][0], key, pos=pos)
        tags[f'{seg}:{tag}'] += 1
        audit.append(dict(molecule_id=mid, segment=seg, tag=tag, moved_from=moved_from))
        out.append((mid, ';'.join(new)))
    return out, seg_n, tags, audit


def _pcf5_apply(scope, base_bytes, pos=5, workdir='.'):
    """Apply pcf5_rows to every row of the final submission.csv, run the same-run control and the checks, then replace
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
    sub_path, tmp_path = root / 'submission.csv', root / 'submission_pcf5.tmp.csv'
    (root / 'submission_base_pcf5.csv').write_bytes(base_bytes)
    base = pd.read_csv(root / 'submission_base_pcf5.csv', dtype=str, keep_default_na=False)
    sample = pd.read_csv(os.path.join(scope['COMP'], 'sample_submission.csv'), dtype=str, keep_default_na=False)
    key = scope['e2r2_make_key'](scope['metric_identity'])
    active = set(scope['te'].molecule_id.astype(str))
    strict = not bool(scope.get('IS_RERUN', True))
    PC, BASE = scope['PC'], scope['BASE']
    assert isinstance(PC, dict) and isinstance(BASE, dict), (type(PC), type(BASE))
    LIB_TAU, S_TAU, POP_TAU = float(scope['LIB_TAU']), float(scope['S_TAU']), float(scope['POP_TAU'])

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
    new_rows, seg, tags, audit = pcf5_rows(rows, active, PC, BASE, key, LIB_TAU, S_TAU, POP_TAU, pos=pos)

    # same-run control: the segment replica must reproduce cell 24's own aggregate counts
    st = scope.get('stats')
    assert isinstance(st, dict) and all(k in st for k in ('untouched', 'gentle', 'aggressive', 'no_pc', 'promoted')), st
    expect = dict(gate_closed=st['untouched'], promoted=st['promoted'], no_pc=st['no_pc'],
                  open=st['gentle'] + st['aggressive'] - st['promoted'])
    got = {k: int(seg.get(k, 0)) for k in expect}
    assert got == expect, ('PCF5 segment replica differs from cell 24 stats', got, expect)
    assert sum(seg.values()) == sum(1 for m, _ in rows if m in active), (dict(seg), len(active))
    if strict:   # commit run: every processed molecule is a row of the file
        assert sum(seg.values()) == len(active), (dict(seg), len(active))

    # per-row checks: only segments gate_closed/open may change; ranks 1..pos-1 identical; the leader's identity exactly
    # once; the other tokens keep their order; no new duplicates; 1-25 non-blank tokens
    seg_of = {a['molecule_id']: a['segment'] for a in audit}
    changed, by_seg, moved_from, examples = 0, Counter(), Counter(), []
    for (mid, s_old), (mid2, s_new) in zip(rows, new_rows):
        assert mid == mid2
        if s_old == s_new:
            continue
        old, new = s_old.split(';'), s_new.split(';')
        assert mid in active and seg_of[mid] in ('gate_closed', 'open'), (mid, seg_of.get(mid))
        kx = key(new[pos - 1]) if len(new) >= pos else key(new[-1])   # a short list gets the leader appended
        assert kx is not None and sum(key(t) == kx for t in new) == 1, mid
        n_keep = min(pos - 1, len(old))
        assert new[:n_keep] == old[:n_keep], mid                        # ranks 1..pos-1 (or a short list) unchanged
        assert [t for t in new if key(t) != kx] == [t for t in old if key(t) != kx][:len(new) - 1], mid
        assert n_dup(new) <= n_dup(old) and 1 <= len(new) <= 25 and all(t.strip() for t in new), mid
        changed += 1
        by_seg[seg_of[mid]] += 1
        a = next(x for x in audit if x['molecule_id'] == mid)
        moved_from[a['moved_from']] += 1
        if len(examples) < 3:
            examples.append(dict(molecule_id=mid, segment=seg_of[mid], tag=a['tag'], moved_from=a['moved_from'],
                                 before=old[:6], after=new[:6]))
    result = pd.DataFrame(new_rows, columns=['molecule_id', 'smiles'])
    check(result)
    for (mid, s_old), (_, s_new) in zip(rows, new_rows):
        if mid not in active:
            assert s_old == s_new, f'placeholder changed: {mid}'

    stats = dict(cell='PCF5', version='v72a', pos=pos, rel_th=150.0, lib_tau=LIB_TAU, s_tau=S_TAU, pop_tau=POP_TAU,
                 is_rerun=not strict, rows=len(base), active=len(active), segments=dict(sorted(seg.items())),
                 stats_control=dict(expected=expect, got=got), tags=dict(sorted(tags.items())),
                 moved_from=dict(sorted(moved_from.items())), changed_rows=changed,
                 changed_by_segment=dict(sorted(by_seg.items())), base_sha256=hashlib.sha256(base_bytes).hexdigest())
    new_bytes = base_bytes
    if changed:
        result.to_csv(tmp_path, index=False)
        back = pd.read_csv(tmp_path, dtype=str, keep_default_na=False)
        check(back)
        assert list(back.smiles) == list(result.smiles) and list(back.molecule_id) == list(result.molecule_id)
        new_bytes = tmp_path.read_bytes()
    stats['submission_sha256'] = hashlib.sha256(new_bytes).hexdigest()
    stats['seconds'] = round(time.time() - t_start, 2)
    (root / 'pcf5_stats.json').write_text(json.dumps(dict(stats, status='applied', examples=examples,
                                                          molecules=audit), indent=1) + '\n')
    if changed:
        os.replace(tmp_path, sub_path)
    print('PCF5_APPLIED', json.dumps(stats), flush=True)
    for ex in examples:
        print('PCF5_EXAMPLE', json.dumps(ex), flush=True)
    return stats


# ==== CELL BODY (executes) ====
import json as _p5_json
import os as _p5_os
import sys as _p5_sys
import time as _p5_time
import traceback as _p5_tb
PCF5_POS = 5                      # rank floor; chosen offline from the diag displacement profile, never on the LB
PCF5_DEADLINE_S = 8.5 * 3600      # skip (keep stack1's file) if the run is this late; the cell itself takes seconds
_P5_STRICT = not bool(globals().get('IS_RERUN', True))   # commit run: fail loudly; scoring rerun: keep the input file
_p5_base_bytes = None
PCF5_STATS = None
try:
    if _p5_time.time() - float(globals().get('T0', _p5_time.time())) > PCF5_DEADLINE_S:
        raise TimeoutError(f'PCF5 deadline: more than {PCF5_DEADLINE_S:.0f} s since T0, cell skipped')
    with open('submission.csv', 'rb') as _p5_f:   # the final E2R2 file (cwd is /kaggle/working)
        _p5_base_bytes = _p5_f.read()
    print('PCF5 reads', _p5_os.path.abspath('submission.csv'), len(_p5_base_bytes), 'bytes', flush=True)
    PCF5_STATS = _pcf5_apply(globals(), _p5_base_bytes, pos=PCF5_POS)
except Exception as _p5_exc:
    if _P5_STRICT:
        raise
    _p5_state = 'not read (left as it was)'
    if _p5_base_bytes is not None:
        try:
            _p5_now = None
            if _p5_os.path.isfile('submission.csv'):
                with open('submission.csv', 'rb') as _p5_f:
                    _p5_now = _p5_f.read()
            if _p5_now != _p5_base_bytes:
                with open('submission.csv', 'wb') as _p5_f:
                    _p5_f.write(_p5_base_bytes)
            _p5_state = 'untouched' if _p5_now == _p5_base_bytes else 'restored from the input bytes'
        except Exception as _p5_exc2:
            _p5_state = f'RESTORE FAILED {_p5_exc2!r}'
    try:
        if _p5_os.path.exists('submission_pcf5.tmp.csv'):
            _p5_os.remove('submission_pcf5.tmp.csv')
        with open('pcf5_stats.json', 'w') as _p5_f:
            _p5_f.write(_p5_json.dumps(dict(cell='PCF5', status='failed', submission=_p5_state,
                                            error=f'{type(_p5_exc).__name__}: {_p5_exc}'[:500])) + '\n')
    except Exception:
        pass
    try:
        print('!' * 100, flush=True)
        print(f'PCF5_FAILED {type(_p5_exc).__name__}: {str(_p5_exc)[:300]} | submission.csv {_p5_state} '
              '| the input list (stack1 + E2R2) is the submission', flush=True)
        _p5_tb.print_exc(file=_p5_sys.stdout)
        print('!' * 100, flush=True)
    except Exception:
        print('PCF5_FAILED (error report failed) | submission.csv', _p5_state, flush=True)
