"""s1v cell (engine 2 votes on the PubChem channel's interleave rows), inserted verbatim before s1's engine-2 cell; the apply step is
S1V_APPLY_CELL, inserted right after s1's PubChem merge cell (44).

Mechanism (the PubChemLite double vote, generalised): a PubChem-channel row that is also in engine 2's list gets two votes in the
metric fusion (1/(3+r1) + 0.6/(3+r2)). PubChemLite gave that second vote only to the ~357k structures in its fixed list. s1v lets
engine 2 score the channel's own interleave rows (the <= k rows merge() places into the slots, molecules with an open channel:
PubChem list non-empty and lib_max < LIB_TAU) with its OWN fitted rankers (no new weights), jointly with the molecule's candidate
window (its features are window-relative), and inserts each row into engine 2's exported list where its blend score puts it,
the existing order kept. vote_only (default): a row can not become engine 2's first row (position >= 1), so the rank-2 overlay
(E2R2, cell 54) never places a channel row; the row only adds its fusion vote.

How: S1V_HOOK makes engine 2's runner (switch on) also pickle what scoring needs (its fitted rankers, every window's records and
spectra, the pool rows those windows use, its blend and its export) to S1V['state']; the runner's export is unchanged.
S1V_APPLY_CELL writes S1V_PASS2 next to the engine code and runs it in its own process after the channel has run (identity gate:
pass 2 first re-scores the stored windows and must reproduce the runner's blend to 1e-6, else nothing changes), then inserts the
rows into ENG before the engine fusion (cell 46) reads it. Any failure -> ENG unchanged (s1's lists). Switch S1V['on'] False -> the
hook returns the runner unchanged, nothing is pickled, the apply step returns at once: the submission is s1's byte for byte.
Log lines: S1V_SWITCH, S1V_STATE (runner), S1V_PASS2 (pass 2), S1V_PATCH (apply).
"""
import os, json

S1V = dict(on=True, k=5, vote_only=True, max_pos=40, workers=4, timeout=1800, score_closed=True, state='/tmp/s1v_state.pkl',
           out='/kaggle/working/s1v_eng2.json', keep_state=False, gate=None)
# gate None: every open molecule (s1v as measured). gate 'aggressive' (G1, scratch/s1v/PREREG.md section 6): the rows are inserted only
# on molecules whose channel takes cell 44's aggressive branch (channel-level rel > REL_TH); gate 'rel>X' (Ga = 'rel>300', PREREG section 7):
# the same with the fixed threshold X instead of REL_TH. Pass 2 still scores every open molecule.

# ---- engine-2 runner patch (pure text; anchors must occur exactly once) --------------------------------------------------------
S1V_RUNNER_ANCHORS = (
    ("    T0 = time.time()\n    subs, recs, S = E.main(",
     "    _S1V_CAP = {}\n"
     "    if os.environ.get('S1V_STATE'):                # s1v (ours): record the fitted rankers and each window's spectra\n"
     "        _s1v_cf, _s1v_sm = E.compute_frag, E.score_molecules\n"
     "        def _s1v_compute_frag(recs, workers):\n"
     "            for _r in recs:\n"
     "                if len(_r['cand']) and 'specs' in _r:\n"
     "                    _r['s1v_specs'] = [(np.array(_a, np.float32), np.array(_b, np.float32)) for _a, _b in _r['specs']]\n"
     "            return _s1v_cf(recs, workers)\n"
     "        def _s1v_score_molecules(recs, rankers, blocks, use_fp):\n"
     "            _S1V_CAP['fp' if use_fp else 'nofp'] = (rankers, [str(b) for b in blocks])\n"
     "            return _s1v_sm(recs, rankers, blocks, use_fp)\n"
     "        E.compute_frag, E.score_molecules = _s1v_compute_frag, _s1v_score_molecules\n"
     "    T0 = time.time()\n    subs, recs, S = E.main("),
    ("    bl = E.blend_scores(S['pv'], S['ours'], 0.88)\n", None),          # checked only: pass 2 uses the same 0.88
    ("    json.dump(res, open(out, 'w'))\n",
     "    json.dump(res, open(out, 'w'))\n"
     "    _s1v_path = os.environ.get('S1V_STATE')\n"
     "    if _s1v_path:                                  # s1v (ours): state for engine-2 pass 2 (the export above is unchanged)\n"
     "        try:\n"
     "            import pickle\n"
     "            _t1 = time.time()\n"
     "            _good = [r for r in recs if len(r['cand']) and r['mid'] in bl and str(r['mid']) in res]\n"
     "            _need = np.unique(np.concatenate([np.asarray(r['cand'], np.int64) for r in _good] +\n"
     "                                             [np.asarray([p for p, _ in list(r.get('ana') or []) + list(r.get('anapol') or [])],\n"
     "                                                         np.int64) for r in _good] + [np.zeros(0, np.int64)]))\n"
     "            _P = E.POOL\n"
     "            _pool = dict(fp=np.asarray(_P['fp'][_need]), mass=np.asarray(_P['mass'][_need]),\n"
     "                         smiles=np.asarray(_P['smiles'][_need], dtype=object), keys=np.asarray(_P['keys'][_need], dtype=object))\n"
     "            _rx = lambda a: np.searchsorted(_need, np.asarray(a, np.int64))\n"
     "            _recs, _reps, _bl = [], {}, {}\n"
     "            for r in _good:\n"
     "                d = {k: v for k, v in r.items() if k != 'specs'}\n"
     "                d['cand'] = _rx(r['cand'])\n"
     "                for _f in ('ana', 'anapol'):\n"
     "                    d[_f] = [(int(_rx([p])[0]), float(s)) for p, s in (r.get(_f) or [])]\n"
     "                _recs.append(d)\n"
     "                _o = np.argsort(-bl[r['mid']], kind='mergesort')[:80]\n"
     "                _sm = [str(x) for x in _P['smiles'][r['cand'][_o]]]\n"
     "                _reps[str(r['mid'])] = [int(_o[_sm.index(s)]) for s in res[str(r['mid'])]['smiles']]\n"
     "                _bl[str(r['mid'])] = np.asarray(bl[r['mid']], np.float64)\n"
     "            pickle.dump(dict(version=1, w_pv=0.88, recs=_recs, pool=_pool, rankers=_S1V_CAP, bl=_bl, reps=_reps, eng=res),\n"
     "                        open(_s1v_path, 'wb'), protocol=4)\n"
     "            print('S1V_STATE', json.dumps(dict(molecules=len(_recs), pool_rows=int(len(_need)), rankers=sorted(_S1V_CAP),\n"
     "                                               bytes=os.path.getsize(_s1v_path), secs=round(time.time() - _t1, 1))), flush=True)\n"
     "        except Exception as _se:\n"
     "            print('S1V_STATE_FAILED', repr(_se)[:300], flush=True)\n"
     "            try:\n"
     "                os.remove(_s1v_path)\n"
     "            except OSError:\n"
     "                pass\n"),
)


def s1v_patch_runner(src):
    """Engine-2 runner source -> the same runner that also writes the s1v state (pure; raises on a missing / repeated anchor)."""
    for old, _ in S1V_RUNNER_ANCHORS:
        assert src.count(old) == 1, ('s1v anchor', old[:50], src.count(old))
    for old, new in S1V_RUNNER_ANCHORS:
        if new is not None:
            src = src.replace(old, new)
    compile(src, 'eng_runner.py', 'exec')
    return src


# ---- pass 2: its own process, next to the engine code -----------------------------------------------------------------------
S1V_PASS2 = r'''"""s1v pass 2 (ours): engine 2 scores extra candidates (the PubChem channel's interleave rows) with its own fitted rankers, jointly
with each molecule's stored window (lib features 0: these structures have no library spectrum), and reports for each row the position
its blend score gives it in engine 2's exported list (number of exported rows whose blend is >= its own).
usage: eng_s1v.py <state.pkl> <rows.json {mid: [smiles]}> <out.json> <workers>"""
import os, sys, json, time, pickle
import numpy as np
if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    state_p, rows_p, out_p, workers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    T0 = time.time()
    rep = dict(status='started')

    def done(**kw):
        rep.update(kw, secs=round(time.time() - T0, 1))
        json.dump(rep, open(out_p, 'w'))
        print('S1V_PASS2', json.dumps({k: v for k, v in rep.items() if k != 'mols'}), flush=True)
    try:
        import casmi_engine as E
        _orig_find = E.find

        def _find(n):                                  # as the runner: prvsiyan's 6,930 bits next to coco_fp.npy
            if n == 'fp_bits.npy':
                return os.path.join(os.path.dirname(_orig_find('coco_fp.npy')), 'fp_bits.npy')
            return _orig_find(n)
        E.find = _find
        st = pickle.load(open(state_p, 'rb'))
        rows = json.load(open(rows_p))
        P = st['pool']
        (Ro, Bo), (Rp, Bp) = st['rankers']['nofp'], st['rankers']['fp']
        w = float(st['w_pv'])
        by = {str(r['mid']): r for r in st['recs']}
        sub = [by[m] for m in rows if m in by]
        E.POOL = dict(P)
        dev = 0.0
        if sub:                                        # identity gate: the stored state reproduces the runner's blend
            b0 = E.blend_scores(E.score_molecules(sub, Rp, Bp, True), E.score_molecules(sub, Ro, Bo, False), w)
            dev = max(float(np.max(np.abs(np.asarray(b0[r['mid']], np.float64) - st['bl'][str(r['mid'])]))) for r in sub)
        if not dev <= 1e-6:
            done(status='identity_failed', dev=dev)
        else:
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog('rdApp.*')

            def ik14(s):
                try:
                    m = Chem.MolFromSmiles(s)
                    return Chem.MolToInchiKey(m)[:14] if m is not None else None
                except Exception:
                    return None
            if 'bits' not in E._g:                     # fp_and_mass only initialises when _g is empty, and canon_key fills it
                E._fp_init()
            n0 = len(P['mass'])
            nfp, nmass, nsmi, nkey = [], [], [], []
            info = {}
            for m, smis in rows.items():
                r = by.get(m)
                if r is None or not len(r['cand']):
                    info[m] = [dict(smiles=s, reason='no_engine_window') for s in smis]
                    continue
                ek = list((st['eng'].get(m) or {}).get('keys') or [])
                wk = set(map(str, P['keys'][r['cand']])); ws = set(map(str, P['smiles'][r['cand']]))
                lst, seen = [], set()
                for s in smis:
                    k = E.canon_key(s)
                    e = dict(smiles=s, key=k)
                    if k is None:
                        e['reason'] = 'no_key'
                    elif k in seen:
                        e['reason'] = 'duplicate'
                    elif k in ek:
                        e['reason'] = 'listed'; e['eng_pos'] = ek.index(k)
                    elif s in ws or k in wk or ik14(s) in wk:
                        e['reason'] = 'in_window'
                    else:
                        fm = E.fp_and_mass(s)
                        if fm is None:
                            e['reason'] = 'no_fp'
                        else:
                            e['idx'] = n0 + len(nfp); nfp.append(fm[0]); nmass.append(fm[1]); nsmi.append(s); nkey.append(k)
                    if k is not None:
                        seen.add(k)
                    lst.append(e)
                info[m] = lst
            minis = []
            if nfp:
                E.POOL = dict(fp=np.vstack([P['fp'], np.stack(nfp).astype(P['fp'].dtype)]),
                              mass=np.concatenate([np.asarray(P['mass'], np.float64), np.asarray(nmass, np.float64)]),
                              smiles=np.concatenate([P['smiles'], np.asarray(nsmi, dtype=object)]),
                              keys=np.concatenate([P['keys'], np.asarray(nkey, dtype=object)]))
                for m, lst in info.items():
                    idx = [e['idx'] for e in lst if 'idx' in e]
                    if idx:
                        r = by[m]
                        minis.append(dict(mid=r['mid'], cand=np.asarray(idx, np.int64), mode=r['mode'], modes=r['modes'],
                                          adducts=r['adducts'], specs=[(np.array(a, np.float32), np.array(b, np.float32))
                                                                       for a, b in r['s1v_specs']]))
                E.compute_frag(minis, workers)         # MetFrag-lite features of the new rows, the engine's own code
            aug = []
            for mi in minis:
                r = by[str(mi['mid'])]; k = len(mi['cand'])
                a = dict(r); a['cand'] = np.concatenate([np.asarray(r['cand'], np.int64), mi['cand']])
                for f in ('lib', 'lib_mean', 'libsh', 'lib_cnt'):
                    a[f] = np.concatenate([np.asarray(r[f], np.float32), np.zeros(k, np.float32)])
                for f in ('frag', 'frag_ad', 'frag_mean', 'frag_pol'):
                    a[f] = np.concatenate([np.asarray(r[f], np.float32), np.asarray(mi[f], np.float32)])
                aug.append(a)
            if aug:
                b1 = E.blend_scores(E.score_molecules(aug, Rp, Bp, True), E.score_molecules(aug, Ro, Bo, False), w)
                for a, mi in zip(aug, minis):
                    m = str(a['mid']); n = len(by[m]['cand'])
                    bb = np.asarray(b1[a['mid']], np.float64)
                    rb = bb[np.asarray(st['reps'][m], np.int64)]
                    j = n
                    for e in info[m]:
                        if 'idx' in e:
                            e.update(blend=float(bb[j]), pos=int((rb >= bb[j]).sum()), rank_in_window=int((bb[:n] >= bb[j]).sum()),
                                     n_window=int(n), n_listed=int(len(rb)))
                            j += 1
            for lst in info.values():
                for e in lst:
                    e.pop('idx', None)
            cnt = {}
            for lst in info.values():
                for e in lst:
                    t = e.get('reason', 'scored'); cnt[t] = cnt.get(t, 0) + 1
            done(status='ok', dev=dev, molecules=len(info), counts=cnt, mols=info)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        done(status='failed', error=repr(exc)[:400])
'''


# ---- pure rules --------------------------------------------------------------------------------------------------------------
def s1v_rows(base, pc, lib_tau, k, score_closed=False):
    """-> (rows {str(mid): [smiles]}, opened {str(mid)}). Rows = the first k PubChem rows merge() would interleave (channel rows whose
    key is not in the 25-row base list), for molecules with a non-empty PubChem list and lib_max < lib_tau (open channel). score_closed:
    also score (never apply) the rows of molecules whose channel is closed (commit-run diagnostics)."""
    rows, opened = {}, set()
    for mid, v in base.items():
        p = (pc.get(mid) or pc.get(str(mid))) if isinstance(pc, dict) else None
        if not p or not p.get('pc'):
            continue
        is_open = float(v[2]) < lib_tau
        if not (is_open or score_closed):
            continue
        bk = set(v[1])
        r = [s for s, kk in zip(p['pc'], p['pc_keys']) if kk not in bk][:int(k)]
        if r:
            rows[str(mid)] = list(r)
            if is_open:
                opened.add(str(mid))
    return rows, opened


def s1v_gate_ok(gate, p, rel_th):
    """-> (allowed, rel). gate None: always. 'aggressive': cell 44's channel-level test, rel = (fz_top if present else pc_fz[0]) -
    best_pool_fz (1e9 when best_pool_fz is not finite), allowed iff rel > rel_th. 'rel>X': allowed iff rel > float(X).
    Unknown gate -> ValueError."""
    import math
    if gate is None:
        return True, None
    if gate == 'aggressive':
        th = float(rel_th)
    elif isinstance(gate, str) and gate.startswith('rel>'):
        th = float(gate[4:])
    else:
        raise ValueError('unknown s1v gate %r' % (gate,))
    if not p or not p.get('pc'):
        return False, None
    b = p.get('best_pool_fz')
    if b is None or not math.isfinite(float(b)):
        rel = 1e9
    else:
        rel = float(p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]) - float(b)
    return rel > th, rel


def s1v_insert(eng, entries, vote_only=True, max_pos=40):
    """One engine-2 list {smiles, keys} + pass-2 entries -> (new list or None if unchanged, tags). Pure. An entry is inserted before the
    exported row at index pos (after every row with a blend >= its own); vote_only: pos >= 1 (never engine 2's first row); keys already
    listed are skipped; the list is cut at max_pos (rows pushed past it drop)."""
    tags = {}

    def tag(t):
        tags[t] = tags.get(t, 0) + 1
    smis, keys = list((eng or {}).get('smiles') or []), list((eng or {}).get('keys') or [])
    if not smis:
        for _ in entries or []:
            tag('no_engine_list')
        return None, tags
    have = set(keys)
    ex, seen = [], set()
    for e in entries or []:
        if e.get('pos') is None:
            tag(e.get('reason') or 'unscored')
            continue
        k = e.get('key')
        if not k or k in have or k in seen:
            tag('listed_or_duplicate')
            continue
        p = int(e['pos'])
        if p < 0:
            tag('bad_pos'); continue
        if vote_only and p < 1:
            p = 1; tag('top1_blocked')
        if p >= max_pos:
            tag('beyond_cap'); continue
        seen.add(k)
        ex.append((p, -float(e.get('blend', 0.0)), e['smiles'], k))
    if not ex:
        return None, tags
    ex.sort(key=lambda t: (t[0], t[1]))
    out_s, out_k, j = [], [], 0
    for i in range(len(smis) + 1):
        while j < len(ex) and ex[j][0] <= i:
            out_s.append(ex[j][2]); out_k.append(ex[j][3]); j += 1
        if i < len(smis):
            out_s.append(smis[i]); out_k.append(keys[i])
    for t in ex[j:]:                                   # positions past a short list: after its last row
        out_s.append(t[2]); out_k.append(t[3])
    out_s, out_k = out_s[:max_pos], out_k[:max_pos]
    n_in = sum(1 for k in out_k if k in seen)
    for p, _, _, k in ex:
        if k in out_k:
            q = out_k.index(k)                         # 0-based index in the new list
            tag('inserted'); tag('at_0' if q == 0 else 'at_1' if q == 1 else 'at_2_5' if q <= 5 else 'at_6_10' if q <= 10
                                 else 'at_11_20' if q <= 20 else 'at_21_39')
        else:
            tag('pushed_out')
    if n_in == 0:
        return None, tags
    return dict(smiles=out_s, keys=out_k), tags


def s1v_selftest():
    b = {'m1': (['A', 'B', 'C'], ['ka', 'kb', 'kc'], 0.5), 'm2': (['A'], ['ka'], 0.95), 'm3': (['A'], ['ka'], 0.1)}
    pc = {'m1': dict(pc=['X', 'B', 'Y', 'Z'], pc_keys=['kx', 'kb', 'ky', 'kz']), 'm2': dict(pc=['X'], pc_keys=['kx']), 'm3': dict(pc=[], pc_keys=[])}
    r, o = s1v_rows(b, pc, 0.9, 2)
    assert r == {'m1': ['X', 'Y']} and o == {'m1'}, (r, o)
    r, o = s1v_rows(b, pc, 0.9, 2, score_closed=True)
    assert r == {'m1': ['X', 'Y'], 'm2': ['X']} and o == {'m1'}, (r, o)
    eng = dict(smiles=['E0', 'E1', 'E2'], keys=['e0', 'e1', 'e2'])
    ent = [dict(smiles='X', key='kx', pos=0, blend=0.9), dict(smiles='Y', key='ky', pos=2, blend=0.5), dict(smiles='W', key='e1', pos=1),
           dict(smiles='Q', key=None, reason='no_key')]
    new, t = s1v_insert(eng, ent, vote_only=True)
    assert new['smiles'] == ['E0', 'X', 'E1', 'Y', 'E2'] and new['keys'] == ['e0', 'kx', 'e1', 'ky', 'e2'], new
    assert t['top1_blocked'] == 1 and t['inserted'] == 2 and t['listed_or_duplicate'] == 1 and t['no_key'] == 1, t
    new, t = s1v_insert(eng, ent, vote_only=False)
    assert new['smiles'] == ['X', 'E0', 'E1', 'Y', 'E2'], new
    new, t = s1v_insert(eng, [dict(smiles='X', key='kx', pos=3, blend=0.1)], max_pos=3)
    assert new is None and t == {'beyond_cap': 1}, (new, t)
    new, t = s1v_insert(eng, [dict(smiles='X', key='kx', pos=2, blend=0.1)], max_pos=3)
    assert new['smiles'] == ['E0', 'E1', 'X'], new
    assert s1v_insert(dict(smiles=[], keys=[]), ent[:1])[0] is None
    pa = dict(pc=['X'], pc_fz=[900.0], fz_top=None, best_pool_fz=100.0)
    assert s1v_gate_ok(None, pa, 600.0) == (True, None) and s1v_gate_ok('aggressive', pa, 600.0) == (True, 800.0)
    assert s1v_gate_ok('aggressive', dict(pa, fz_top=650.0), 600.0) == (False, 550.0)
    assert s1v_gate_ok('aggressive', dict(pa, best_pool_fz=float('nan')), 600.0) == (True, 1e9)
    assert s1v_gate_ok('aggressive', dict(pc=[]), 600.0) == (False, None)
    assert s1v_gate_ok('rel>300', dict(pa, fz_top=650.0), 600.0) == (True, 550.0)
    assert s1v_gate_ok('rel>300', dict(pa, fz_top=350.0), 600.0) == (False, 250.0)
    assert s1v_patch_runner is not None
    return 'ok'


# ---- notebook side ----------------------------------------------------------------------------------------------------------
S1V_HOOK_STATE = {}


def S1V_HOOK(src):
    """The engine-2 cell writes eng_runner.py through this hook. Switch off, or any anchor problem -> src unchanged (and s1v off)."""
    st = dict(status='off')
    if S1V.get('on'):
        try:
            new = s1v_patch_runner(src)
            st = dict(status='armed', runner_bytes=[len(src), len(new)])
            src = new
        except Exception as exc:
            st = dict(status='off (' + repr(exc)[:200] + ')')
            S1V['on'] = False
            os.environ.pop('S1V_STATE', None)
    S1V_HOOK_STATE.update(st)
    print('S1V_HOOK', json.dumps(st), flush=True)
    return src


def s1v_apply(scope):
    """After the PubChem merge: pass 2 on the channel's interleave rows, then the rows go into ENG (in place, all at once)."""
    import sys, time, subprocess
    sw = scope.get('S1V') or {}
    rep = dict(status='off')
    if not sw.get('on'):
        print('S1V_PATCH', json.dumps(rep), flush=True)
        return rep
    t0 = time.time()
    try:
        ENG, PC, BASE = scope['ENG'], scope['PC'], scope['BASE']
        assert isinstance(ENG, dict) and ENG, 'no engine-2 lists'
        assert os.path.isfile(sw['state']), 'no engine-2 state (runner hook)'
        closed = bool(sw.get('score_closed')) and not scope.get('IS_RERUN', True)
        rows, opened = s1v_rows(BASE, PC, float(scope['LIB_TAU']), int(sw['k']), closed)
        rep.update(open=len(opened), scored_molecules=len(rows), rows=sum(len(v) for v in rows.values()))
        if not rows:
            rep['status'] = 'ok (no rows)'
        else:
            d = scope['ENG_DIR']
            script, rp = os.path.join(d, 'eng_s1v.py'), os.path.join(d, 's1v_rows.json')
            open(script, 'w').write(S1V_PASS2)
            json.dump(rows, open(rp, 'w'))
            if os.path.exists(sw['out']):
                os.remove(sw['out'])
            rr = subprocess.run([sys.executable, script, sw['state'], rp, sw['out'], str(int(sw['workers']))], capture_output=True,
                                text=True, timeout=float(sw['timeout']), cwd=d, env={**os.environ, 'PYTHONHASHSEED': '0'})
            print(rr.stdout[-2000:]); print(rr.stderr[-2000:])
            res = json.load(open(sw['out']))
            rep.update(pass2=res.get('status'), dev=res.get('dev'), pass2_counts=res.get('counts'), pass2_secs=res.get('secs'))
            assert res.get('status') == 'ok', ('pass 2', res.get('status'), res.get('error'))
            new, tags = {}, {}
            gate = sw.get('gate')
            gated_out = 0
            for m in sorted(opened):
                if gate is not None and not s1v_gate_ok(gate, PC.get(m) if isinstance(PC, dict) else None, float(scope['REL_TH']))[0]:
                    gated_out += 1
                    continue
                out, t = s1v_insert(ENG.get(m), (res.get('mols') or {}).get(m) or [], bool(sw['vote_only']), int(sw['max_pos']))
                for kk, vv in t.items():
                    tags[kk] = tags.get(kk, 0) + vv
                if out is not None:
                    new[m] = out
            pos = [e['pos'] for m, l in (res.get('mols') or {}).items() for e in l if e.get('pos') is not None]
            rep['pos_all_scored'] = dict(n=len(pos), top1=sum(p == 0 for p in pos), le5=sum(p <= 5 for p in pos),
                                         le20=sum(p <= 20 for p in pos), lt40=sum(p < 40 for p in pos))
            for m, v in new.items():                   # all at once, after every check
                ENG[m] = v
            rep.update(status='ok', changed_lists=len(new), tags=tags, gate=gate, gated_out=gated_out, gated_in=len(opened) - gated_out)
    except Exception as exc:
        rep.update(status='failed (ENG unchanged)', error=repr(exc)[:300])
    finally:
        if not sw.get('keep_state'):
            try:
                os.remove(sw['state'])
            except OSError:
                pass
    rep['secs'] = round(time.time() - t0, 1)
    print('S1V_PATCH', json.dumps(rep, default=str), flush=True)
    return rep


S1V_APPLY_CELL = '''"""s1v (ours): engine 2 scores the PubChem channel's interleave rows (pass 2) and they enter its list before the fusion (cell 46).
Code: the s1v cell before the engine-2 cell (switch S1V). Switch off -> nothing happens."""
S1V_REPORT = s1v_apply(globals())
'''

if S1V.get('on'):
    try:
        assert s1v_selftest() == 'ok'
        os.environ['S1V_STATE'] = S1V['state']
    except Exception as _s1v_e:
        S1V['on'] = False
        os.environ.pop('S1V_STATE', None)
        print('S1V_SELFTEST_FAILED -> off', repr(_s1v_e)[:300], flush=True)
print('S1V_SWITCH', json.dumps(S1V), flush=True)
