"""Tests of s1v (nb/embed/s1v_eng2vote.py: engine 2 votes on the PubChem channel's interleave rows; scratch/s1v/build.py); synthetic data.

Pure rules: s1v_rows (open / closed channel, base-key filter, k), s1v_insert (vote-only blocks position 0, order, ties, cap, listed keys),
the self-test, the hook (switch off -> runner unchanged; on -> anchors once, compiles, composes with s1val's hook).
End to end (needs s1's notebook for the real engine-2 code, numba, scikit-learn, RDKit; skipped otherwise): the real casmi_engine.py /
pv.py of s1's engine-2 cell with a synthetic main() (36-structure pool, 3 windows, small HistGradientBoosting rankers) run through s1's
real runner text, unhooked and hooked: (1) the export eng_lists.json is byte-identical, (2) the pickled state passes pass 2's identity
gate (dev 0), (3) every scored row's blend and position equal an independent computation in which the row is simply one more candidate
of the window (lib 0, MetFrag-lite by the engine), (4) s1v_apply changes ENG exactly as s1v_insert on pass 2's output, only for open
molecules, and leaves ENG untouched with the switch off; with gate 'aggressive' only aggressive-branch molecules change. Workers: 2.
usage: python solutions/fusion-v4n/tests/test_s1v.py   (or pytest)
"""
import ast, copy, json, os, pickle, shutil, subprocess, sys, tempfile

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(SOL))
EMBED = os.path.join(SOL, 'nb', 'embed', 's1v_eng2vote.py')
S1_NB = os.path.join(ROOT, 'scratch', 'land1024', 'sub', 'casmi26-sub-s1', 'casmi26-sub-s1.ipynb')
S1VAL_CELLS = os.path.join(ROOT, 'scratch', 's1val', 'cells_val.py')
HAVE_S1 = os.path.exists(S1_NB)


def load(on=True):
    t = open(EMBED, encoding='utf-8').read()
    if not on:
        t = t.replace('S1V = dict(on=True, ', 'S1V = dict(on=False, ')
    old = os.environ.get('S1V_STATE')
    g = {'__name__': 's1v_test'}
    exec(compile(t, 's1v_eng2vote.py', 'exec'), g)
    if old is None:
        os.environ.pop('S1V_STATE', None)
    else:
        os.environ['S1V_STATE'] = old
    return g


M = load(False)


def s1_cells():
    return [''.join(c['source']) for c in json.load(open(S1_NB, encoding='utf-8'))['cells']]


def eng_cell_parts():
    """(ENG_FILES dict, runner text) from s1's engine-2 cell."""
    c = [s for s in s1_cells() if s.startswith("'''Our engine (two-ranker")]
    assert len(c) == 1
    tree = ast.parse(c[0])
    files = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'ENG_FILES':
            files = ast.literal_eval(n.value)
    run = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'write' and n.args
           and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str) and n.args[0].value.startswith('"""Our engine')]
    assert files and len(run) == 1
    return files, run[0]


# ---- pure ---------------------------------------------------------------------------------------------------------------------
def test_selftest():
    assert M['s1v_selftest']() == 'ok'


def test_rows():
    R = M['s1v_rows']
    base = {'a': (['A', 'B'], ['ka', 'kb'], 0.2), 'b': (['A'], ['ka'], 0.9), 'c': (['A'], ['ka'], 0.0), 'd': (['A'], ['ka'], 0.1)}
    pc = {'a': dict(pc=['P1', 'B', 'P2', 'P3', 'P4', 'P5', 'P6'], pc_keys=['k1', 'kb', 'k2', 'k3', 'k4', 'k5', 'k6']),
          'b': dict(pc=['P1'], pc_keys=['k1']), 'c': dict(pc=[], pc_keys=[])}
    rows, op = R(base, pc, 0.9, 5)
    assert rows == {'a': ['P1', 'P2', 'P3', 'P4', 'P5']} and op == {'a'}          # lib_max 0.9 is closed (cell 44: >= LIB_TAU)
    rows, op = R(base, pc, 0.9, 2, score_closed=True)
    assert rows == {'a': ['P1', 'P2'], 'b': ['P1']} and op == {'a'}
    rows, op = R({1: (['A'], ['ka'], 0.1)}, {'1': dict(pc=['X'], pc_keys=['kx'])}, 0.9, 5)   # str / int molecule ids
    assert rows == {'1': ['X']} and op == {'1'}


def test_rows_match_merge():
    """The rows are exactly the PubChem rows s1's merge() interleaves (first k of them)."""
    base = (['A', 'B', 'C', 'D'], ['ka', 'kb', 'kc', 'kd'], 0.3)
    pc = dict(pc=['X', 'B', 'Y', 'Z', 'C', 'W', 'V', 'U'], pc_keys=['kx', 'kb', 'ky', 'kz', 'kc', 'kw', 'kv', 'ku'])
    rows, _ = M['s1v_rows']({'m': base}, {'m': pc}, 0.9, 5)
    bk = set(base[1]); pcs = [s for s, k in zip(pc['pc'], pc['pc_keys']) if k not in bk]      # cell 44's merge()
    assert rows['m'] == pcs[:5]


def test_insert():
    I = M['s1v_insert']
    eng = dict(smiles=['E%d' % i for i in range(40)], keys=['e%d' % i for i in range(40)])
    new, t = I(eng, [dict(smiles='X', key='kx', pos=0, blend=1.0)], vote_only=True)
    assert new['smiles'][:3] == ['E0', 'X', 'E1'] and len(new['smiles']) == 40 and new['smiles'][-1] == 'E38' and t['top1_blocked'] == 1
    new, _ = I(eng, [dict(smiles='X', key='kx', pos=0, blend=1.0)], vote_only=False)
    assert new['smiles'][:2] == ['X', 'E0']
    ent = [dict(smiles='X', key='kx', pos=3, blend=0.2), dict(smiles='Y', key='ky', pos=3, blend=0.7), dict(smiles='Z', key='kz', pos=39, blend=0.0)]
    new, t = I(eng, ent)
    assert new['smiles'][:6] == ['E0', 'E1', 'E2', 'Y', 'X', 'E3'], new['smiles'][:6]                # ties: higher blend first
    assert 'Z' not in new['smiles'] and t['pushed_out'] == 1 and t['inserted'] == 2                    # pushed past 40 by X, Y
    new, t = I(eng, [dict(smiles='Q', key='e5', pos=2, blend=0.0)])
    assert new is None and t == {'listed_or_duplicate': 1}
    short = dict(smiles=['E0', 'E1'], keys=['e0', 'e1'])
    new, _ = I(short, [dict(smiles='X', key='kx', pos=7, blend=0.0)])
    assert new == dict(smiles=['E0', 'E1', 'X'], keys=['e0', 'e1', 'kx'])
    assert I(None, [dict(smiles='X', key='kx', pos=1)])[0] is None
    a = copy.deepcopy(eng); I(a, ent)
    assert a == eng                                                                                     # pure


def test_apply_off_and_failure():
    g = load(False)
    eng = {'m': dict(smiles=['A'], keys=['ka'])}
    sc = dict(S1V=g['S1V'], ENG=eng, PC={}, BASE={}, LIB_TAU=0.9)
    assert g['s1v_apply'](sc)['status'] == 'off' and eng == {'m': dict(smiles=['A'], keys=['ka'])}
    g = load(True)
    sw = dict(g['S1V'], state=os.path.join(tempfile.gettempdir(), 's1v_missing_state.pkl'))
    rep = g['s1v_apply'](dict(S1V=sw, ENG=eng, PC={}, BASE={}, LIB_TAU=0.9))
    assert rep['status'].startswith('failed') and eng == {'m': dict(smiles=['A'], keys=['ka'])}


def test_hook():
    if not HAVE_S1:
        print('skip test_hook (no s1 notebook)'); return
    _, run = eng_cell_parts()
    g = load(False)
    assert g['S1V_HOOK'](run) == run
    g = load(True)
    new = g['S1V_HOOK'](run)
    assert new != run and g['S1V_HOOK_STATE']['status'] == 'armed'
    compile(new, 'r', 'exec')
    assert new.count("    json.dump(res, open(out, 'w'))\n") == 1 and new.count('E.main(') == 1
    if os.path.exists(S1VAL_CELLS):                                            # composes with s1val's runner hook (a later val run)
        sys.path.insert(0, os.path.dirname(S1VAL_CELLS))
        import cells_val
        ns = {}
        exec(compile(cells_val.CELL_VAL.split('import numpy as np, json, time, glob')[1].split("if S1VAL['validation']:")[0]
                     .replace('S1VAL = __S1VAL__', '').replace('S1VAL_SET = __SET__', ''), 'cv', 'exec'), ns)
        both = ns['s1val_patch_runner'](new)
        compile(both, 'r', 'exec')
    bad = run.replace("    json.dump(res, open(out, 'w'))\n", '')
    g = load(True)
    assert g['S1V_HOOK'](bad) == bad and g['S1V']['on'] is False


def test_builder():
    if not HAVE_S1:
        print('skip test_builder (no s1 notebook)'); return
    sys.path.insert(0, os.path.join(ROOT, 'scratch', 's1v'))
    import build as B
    s1 = s1_cells()
    for on in (True, False):
        nb = B.build(on)
        c = [''.join(x['source']) for x in nb['cells']]
        assert len(c) == len(s1) + 3 and c[2:34] == s1[1:33] and c[36:47] == s1[34:45] and c[48:] == s1[45:]
        assert ('S1V = dict(on=True, ' in c[34]) == on and ('S1V = dict(on=False, ' in c[34]) == (not on)
        assert 'S1V_HOOK(' in c[35] and c[47].strip().endswith('S1V_REPORT = s1v_apply(globals())')
    on_ = [''.join(x['source']) for x in B.build(True)['cells']]
    g1 = [''.join(x['source']) for x in B.build(True, 'aggressive')['cells']]
    diff = [i for i, (x, y) in enumerate(zip(on_, g1)) if x != y]
    assert len(on_) == len(g1) and diff == [1, 34], diff                       # notice + the one switch line
    dl = [(x, y) for x, y in zip(on_[34].splitlines(), g1[34].splitlines()) if x != y]
    assert dl == [(x, x.replace('gate=None)', "gate='aggressive')")) for x, _ in dl] and len(dl) == 1, dl
    g2 = [''.join(x['source']) for x in B.build(True, 'rel>300')['cells']]
    assert [i for i, (x, y) in enumerate(zip(on_, g2)) if x != y] == [1, 34] and "gate='rel>300')" in g2[34] and 'exceeds 300' in g2[1]


def test_gate_matches_replay():
    """s1v_gate_ok('aggressive') = the G1 flags replayed from the T3 / T2 dumps (scratch/s1v/replay_gates.py); local files only."""
    import gzip
    for k in ('t2', 't3'):
        gp, dp = f'C:/Users/hemre/casmi-work/s1v/{k}/gates.json', f'C:/Users/hemre/casmi-work/s1v/{k}/dump/s1val/s1val_dump.json.gz'
        if not (os.path.exists(gp) and os.path.exists(dp)):
            print('skip test_gate_matches_replay', k); continue
        G = json.load(open(gp))['mols']; D = json.load(gzip.open(dp, 'rt', encoding='utf-8'))
        rel_th = D['meta']['consts']['REL_TH']
        for m, r in G.items():
            if r['open']:
                assert M['s1v_gate_ok']('aggressive', D['mols'][m]['pc'], rel_th)[0] == r['aggressive'], (k, m)
        g2p = f'C:/Users/hemre/casmi-work/s1v/{k}/gates2.json'                     # Ga (rel > 300) = PREREG section 7's replayed flag
        if os.path.exists(g2p):
            for m, r in json.load(open(g2p))['mols'].items():
                if r['open'] and r['rel'] is not None:
                    assert M['s1v_gate_ok']('rel>300', D['mols'][m]['pc'], rel_th)[0] == (r['rel'] > 300.0), (k, m)
        _learned_matches_fit(k, D, rel_th)


def _learned_matches_fit(k, D, rel_th):
    """gate 'learned' = scratch/s1v/learned_gate.py's features and frozen fit (PREREG section 8) on every open molecule of a dump."""
    lp = 'C:/Users/hemre/casmi-work/s1v/learned_gate.json'
    if not os.path.exists(lp):
        print('skip learned gate check', k); return
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, 'scratch', 's1v'))
    import learned_gate as LG
    F = json.load(open(lp))['final']
    L = M['S1V_LEARNED']
    assert list(L['feats']) == F['feats'] == LG.FEATS and L['threshold'] == F['threshold'] and L['b'] == F['b']
    assert all(list(L[x]) == F[x] for x in ('mu', 'sd', 'coef'))
    n_open = n_in = 0
    for m, d in D['mols'].items():
        _, op = M['s1v_rows']({m: tuple(d['base25'])}, {m: d['pc']} if d.get('pc') else {}, D['meta']['consts']['LIB_TAU'], 5)
        if m not in op:
            continue
        ent = d.get('s1v') or []
        f = M['s1v_gate_features'](d['pc'], tuple(d['base25']), d['eng0'], ent)
        assert f == LG.features(d['pc'], d['base25'], d['eng0'], ent), (k, m)
        x = np.array([f[n] for n in F['feats']])
        want = 1.0 / (1.0 + np.exp(-(((x - np.array(F['mu'])) / np.array(F['sd'])) @ np.array(F['coef']) + F['b'])))
        ok, s = M['s1v_gate_ok']('learned', d['pc'], rel_th, tuple(d['base25']), d['eng0'], ent)
        assert abs(s - want) < 1e-9 and ok == (s >= F['threshold']), (k, m, s, want)
        n_open += 1; n_in += ok
    print(f'learned gate {k}: {n_in} of {n_open} open molecules pass (in-sample, frozen fit)')


# ---- end to end -----------------------------------------------------------------------------------------------------------------
POOL_SMILES = [
    'O=C1C=C(Oc2cc(O)cc(O)c12)c1ccc(O)cc1', 'O=C1C=C(Oc2cc(O)cc(O)c12)c1ccc(O)c(O)c1', 'O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1ccc(O)cc1',
    'O=C1C(O)=C(Oc2cc(O)cc(O)c12)c1ccc(O)c(O)c1', 'COc1cc(C=CC(=O)O)ccc1O', 'O=C(O)C=Cc1ccc(O)c(O)c1', 'O=C(O)C=Cc1ccc(O)cc1',
    'COc1cc(C=O)ccc1O', 'Oc1ccc(C=Cc2cc(O)cc(O)c2)cc1', 'CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C', 'OCC1OC(O)C(O)C(O)C1O',
    'CC(C)=CCCC(C)=CCO', 'CC1=CCC(CC1)C(C)=C', 'CC(C)C1CCC(C)CC1O', 'O=c1ccc2ccc(O)cc2o1', 'COc1cc2ccc(=O)oc2cc1O',
    'Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2', 'NCCc1ccc(O)c(O)c1', 'NCCc1c[nH]c2ccc(O)cc12', 'CC(N)Cc1ccccc1', 'OC(=O)CC(O)(CC(=O)O)C(=O)O',
    'CCCCCCCCCCCCCCCC(=O)O', 'O=C(O)c1ccccc1O', 'Oc1ccccc1', 'c1ccc2[nH]ccc2c1', 'NC(Cc1ccccc1)C(=O)O', 'NC(Cc1ccc(O)cc1)C(=O)O',
    'NC(Cc1c[nH]c2ccccc12)C(=O)O', 'CC(=O)NC1C(O)OC(CO)C(O)C1O', 'COC1=CC(=O)C=CC1=O', 'Cc1ccc(C(C)C)cc1O', 'CC(C)Cc1ccc(C(C)C(=O)O)cc1',
    'CC(C)c1ccc(C)cc1O', 'OC1CCCCC1', 'CCOC(=O)c1ccccc1']
EXTRA_NEW = ['O=C1C=C(Oc2cc(O)ccc12)c1ccc(O)cc1', 'COc1ccc(C=CC(=O)O)cc1', 'Oc1ccc(cc1)C1CC(=O)c2c(O)cc(O)cc2O1']

FAKE_MAIN = r'''

# ---------------- s1v test: synthetic main (36-structure pool, 3 windows, small rankers) ----------------
import json


def find(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def s1v_fake_build():
    global POOL
    smis = json.load(open(find('pool_smiles.json')))
    fm = [fp_and_mass(s) for s in smis]
    POOL = dict(fp=np.stack([f[0] for f in fm]), mass=np.array([f[1] for f in fm]), smiles=np.asarray(smis, dtype=object),
                keys=np.asarray([canon_key(s) for s in smis], dtype=object))
    rng = np.random.RandomState(0)
    recs = []
    for mid, cand in (('mA', range(0, 12)), ('mB', range(12, 24)), ('mC', range(24, 30))):
        cand = np.asarray(list(cand), np.int64); n = len(cand)
        specs = []
        for _ in range(2):
            mz = np.sort(rng.uniform(40, 300, 25)).astype(np.float32); it = rng.uniform(0.01, 1, 25).astype(np.float32)
            specs.append((mz, it / it.max()))
        lib = (rng.rand(n) * (rng.rand(n) > 0.5)).astype(np.float32)
        rec = dict(mid=mid, target=float(POOL['mass'][cand].mean()), cand=cand,
                   zlog=None if mid == 'mC' else rng.randn(6930).astype(np.float32), specs=specs, mode=1.0, modes=np.array([1.0, 1.0]),
                   adducts=['[M+H]+', '[M+H]+'], ana=[(int(p), float(s)) for p, s in zip(rng.choice(30, 5, replace=False), sorted(rng.rand(5))[::-1])],
                   anapol=[(int(p), float(s)) for p, s in zip(rng.choice(30, 4, replace=False), sorted(rng.rand(4))[::-1])])
        rec['lib'] = lib; rec['lib_mean'] = (lib * 0.7).astype(np.float32); rec['libsh'] = (lib * 0.5).astype(np.float32)
        rec['lib_cnt'] = (lib > 0).astype(np.float32) * 2
        recs.append(rec)
    return recs


def s1v_fake_rankers(recs, blocks, use_fp, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.vstack([np.hstack([BLOCKS[b](r if use_fp else dict(r, zlog=None), pool_fps(r['cand'])) for b in blocks]) for r in recs])
    rng = np.random.RandomState(seed)
    X = np.vstack([X] * 4) + rng.randn(len(X) * 4, X.shape[1]).astype(np.float32) * 0.1
    Y = (X[:, 0] + rng.randn(len(X)) > np.median(X[:, 0])).astype(int)
    return [HistGradientBoostingClassifier(max_iter=12, max_depth=3, min_samples_leaf=5, random_state=s).fit(X, Y) for s in (0, 1)], blocks


def main(test_path, train_path, sample_path, our_rank_path, pv_rank_path, fp_model_paths, workers=4, limit=0, w_pv=0.88):
    recs = s1v_fake_build()
    compute_frag(recs, 2)
    ours, blocks = s1v_fake_rankers(recs, ['base', 'mass', 'libx', 'anapol', 'fragx'], False, 1)
    s_ours = score_molecules(recs, ours, blocks, use_fp=False)
    pvr, _ = s1v_fake_rankers(recs, ['base'], True, 2)
    s_pv = score_molecules(recs, pvr, ['base'], use_fp=True)
    return {}, recs, {'pv': s_pv, 'ours': s_ours}
'''


def _have_e2e():
    if not HAVE_S1:
        return 'no s1 notebook'
    try:
        import numba, sklearn, rdkit  # noqa: F401
    except Exception as e:
        return repr(e)
    return None


def _setup(d):
    files, run = eng_cell_parts()
    for n, s in files.items():
        open(os.path.join(d, n), 'w', encoding='utf-8').write(s + (FAKE_MAIN if n == 'casmi_engine.py' else ''))
    import numpy as np
    np.save(os.path.join(d, 'fp_bits.npy'), np.sort(np.random.RandomState(0).choice(4096 * 2 + 2048 + 167, 6930, replace=False)))
    np.save(os.path.join(d, 'coco_fp.npy'), np.zeros(1))
    json.dump(POOL_SMILES, open(os.path.join(d, 'pool_smiles.json'), 'w'))
    return run


def _env(extra=None):
    e = {**os.environ, 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '2',
         'PYTHONIOENCODING': 'utf-8'}
    e.pop('S1V_STATE', None)
    e.update(extra or {})
    return e


def _run(args, cwd, env):
    r = subprocess.run([sys.executable] + args, cwd=cwd, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    return r.stdout


def test_end_to_end():
    why = _have_e2e()
    if why:
        print('skip test_end_to_end:', why); return
    import numpy as np
    d = tempfile.mkdtemp(prefix='s1v_e2e_')
    try:
        run = _setup(d)
        g = load(True)
        open(os.path.join(d, 'runner0.py'), 'w', encoding='utf-8').write(run)
        open(os.path.join(d, 'runner1.py'), 'w', encoding='utf-8').write(g['S1V_HOOK'](run))
        st = os.path.join(d, 'state.pkl')
        _run(['runner0.py', 't', 'tr', 's', os.path.join(d, 'eng0.json')], d, _env())
        out1 = _run(['runner1.py', 't', 'tr', 's', os.path.join(d, 'eng1.json')], d, _env({'S1V_STATE': st}))
        assert 'S1V_STATE {' in out1, out1[-2000:]
        b0, b1 = open(os.path.join(d, 'eng0.json'), 'rb').read(), open(os.path.join(d, 'eng1.json'), 'rb').read()
        assert b0 == b1, 'hooked runner changed the engine-2 export'
        ENG = json.loads(b0)
        S = pickle.load(open(st, 'rb'))
        assert sorted(S['rankers']) == ['fp', 'nofp'] and set(S['reps']) == {'mA', 'mB', 'mC'}
        for m, r in ENG.items():                       # reps point at the exported rows
            rec = [x for x in S['recs'] if x['mid'] == m][0]
            assert [str(S['pool']['smiles'][rec['cand'][i]]) for i in S['reps'][m]] == r['smiles']
        # rows: new structures, a pool structure outside every window, an exported (listed) one, a duplicate, a bad SMILES
        rows = {'mA': [EXTRA_NEW[0], POOL_SMILES[32], POOL_SMILES[3], EXTRA_NEW[0], 'not_a_smiles'],
                'mB': [EXTRA_NEW[1], EXTRA_NEW[2]], 'mC': [EXTRA_NEW[2], POOL_SMILES[33]], 'mZ': ['CCO']}
        json.dump(rows, open(os.path.join(d, 'rows.json'), 'w'))
        open(os.path.join(d, 'eng_s1v.py'), 'w', encoding='utf-8').write(g['S1V_PASS2'])
        _run(['eng_s1v.py', st, os.path.join(d, 'rows.json'), os.path.join(d, 'p2.json'), '2'], d, _env())
        P2 = json.load(open(os.path.join(d, 'p2.json')))
        assert P2['status'] == 'ok' and P2['dev'] <= 1e-9, (P2.get('status'), P2.get('dev'), P2.get('error'))
        rs = {m: [e.get('reason', 'scored') for e in l] for m, l in P2['mols'].items()}
        assert rs['mA'] == ['scored', 'scored', 'listed', 'duplicate', 'no_key'] and rs['mB'] == ['scored', 'scored'], rs
        assert rs['mC'] == ['scored', 'scored'] and rs['mZ'] == ['no_engine_window'], rs
        S2 = copy.deepcopy(S)                          # a window structure that is not exported -> in_window (never added twice)
        k3 = P2['mols']['mA'][2]['key']
        S2['eng']['mA'] = dict(smiles=[s for s, k in zip(ENG['mA']['smiles'], ENG['mA']['keys']) if k != k3],
                               keys=[k for k in ENG['mA']['keys'] if k != k3])
        pickle.dump(S2, open(os.path.join(d, 'state2.pkl'), 'wb'))
        json.dump({'mA': [POOL_SMILES[3]]}, open(os.path.join(d, 'rows2.json'), 'w'))
        _run(['eng_s1v.py', os.path.join(d, 'state2.pkl'), os.path.join(d, 'rows2.json'), os.path.join(d, 'p2w.json'), '2'], d, _env())
        assert json.load(open(os.path.join(d, 'p2w.json')))['mols']['mA'][0]['reason'] == 'in_window'
        # identity gate: a corrupted stored blend stops pass 2
        S2 = copy.deepcopy(S); S2['bl']['mA'] = S2['bl']['mA'] + 1e-3
        pickle.dump(S2, open(os.path.join(d, 'state3.pkl'), 'wb'))
        _run(['eng_s1v.py', os.path.join(d, 'state3.pkl'), os.path.join(d, 'rows.json'), os.path.join(d, 'p2x.json'), '2'], d, _env())
        assert json.load(open(os.path.join(d, 'p2x.json')))['status'] == 'identity_failed'
        # independent computation: the row as one more candidate of the window (lib 0), engine code in this process
        sys.path.insert(0, d)
        for k in [k for k in sys.modules if k in ('casmi_engine', 'pv', 'pv_fp')]:
            del sys.modules[k]
        import casmi_engine as E
        recs = E.s1v_fake_build()
        full = dict(E.POOL)
        for m in ('mA', 'mB', 'mC'):
            sc = [e for e in P2['mols'][m] if 'pos' in e]
            fm = [E.fp_and_mass(e['smiles']) for e in sc]
            E.POOL = dict(fp=np.vstack([full['fp']] + [f[0][None] for f in fm]), mass=np.concatenate([full['mass'], [f[1] for f in fm]]),
                          smiles=np.concatenate([full['smiles'], np.asarray([e['smiles'] for e in sc], dtype=object)]),
                          keys=np.concatenate([full['keys'], np.asarray([e['key'] for e in sc], dtype=object)]))
            r = copy.deepcopy([x for x in recs if x['mid'] == m][0]); n = len(r['cand'])
            r['cand'] = np.concatenate([r['cand'], len(full['mass']) + np.arange(len(sc))])
            for f in ('lib', 'lib_mean', 'libsh', 'lib_cnt'):
                r[f] = np.concatenate([r[f], np.zeros(len(sc), np.float32)])
            E.compute_frag([r], 1)
            (Ro, Bo), (Rp, Bp) = S['rankers']['nofp'], S['rankers']['fp']
            bl = E.blend_scores(E.score_molecules([r], Rp, Bp, True), E.score_molecules([r], Ro, Bo, False), 0.88)[m]
            reps = np.asarray(S['reps'][m])
            for j, e in enumerate(sc):
                assert abs(bl[n + j] - e['blend']) < 1e-9, (m, j, bl[n + j], e['blend'])
                assert e['pos'] == int((bl[reps] >= bl[n + j]).sum()) and e['n_window'] == n, (m, e)
        # notebook side: applied only to open molecules, exactly s1v_insert on pass 2's output
        base = {'mA': (['x'], ['kx'], 0.3), 'mB': (['x'], ['kx'], 0.95), 'mC': (['x'], ['kx'], 0.1)}
        pc = {m: dict(pc=rows[m], pc_keys=['p%d' % i for i in range(len(rows[m]))]) for m in ('mA', 'mB', 'mC')}
        eng = json.loads(b0)
        sw = dict(g['S1V'], state=st, out=os.path.join(d, 'p2b.json'), keep_state=True, workers=2)
        rep = g['s1v_apply'](dict(S1V=sw, ENG=eng, PC=pc, BASE=base, LIB_TAU=0.9, ENG_DIR=d, IS_RERUN=True))
        assert rep['status'] == 'ok' and rep['open'] == 2 and rep['scored_molecules'] == 2, rep
        P2b = json.load(open(sw['out']))
        for m in ('mA', 'mC'):
            want = g['s1v_insert'](ENG[m], P2b['mols'][m], True, 40)[0] or ENG[m]
            assert eng[m] == want, m
            assert eng[m]['smiles'][0] == ENG[m]['smiles'][0]                         # vote only: engine 2's first row kept
        assert eng['mB'] == ENG['mB']
        assert any(eng[m] != ENG[m] for m in ('mA', 'mC')), 'nothing inserted in the synthetic case'
        eng2 = json.loads(b0)
        rep = g['s1v_apply'](dict(S1V=dict(sw, on=False), ENG=eng2, PC=pc, BASE=base, LIB_TAU=0.9, ENG_DIR=d, IS_RERUN=True))
        assert rep['status'] == 'off' and eng2 == ENG
        sw2 = dict(sw, out=os.path.join(d, 'p2c.json'))                              # commit diagnostics: closed molecules scored, not applied
        eng3 = json.loads(b0)
        rep = g['s1v_apply'](dict(S1V=sw2, ENG=eng3, PC=pc, BASE=base, LIB_TAU=0.9, ENG_DIR=d, IS_RERUN=False))
        assert rep['scored_molecules'] == 3 and eng3['mB'] == ENG['mB'] and eng3['mA'] == eng['mA'], rep
        # gate 'aggressive' (G1): rows only for molecules whose channel takes cell 44's aggressive branch (rel > REL_TH)
        pcg = {m: dict(pc[m], pc_fz=[100.0] * len(rows[m]), fz_top=None, best_pool_fz=0.0) for m in pc}
        pcg['mA']['fz_top'] = 900.0                                                   # mA aggressive (rel 900), mC gentle (rel 100)
        eng4 = json.loads(b0)
        sw4 = dict(sw, out=os.path.join(d, 'p2d.json'), gate='aggressive')
        rep = g['s1v_apply'](dict(S1V=sw4, ENG=eng4, PC=pcg, BASE=base, LIB_TAU=0.9, REL_TH=600.0, ENG_DIR=d, IS_RERUN=True))
        assert rep['status'] == 'ok' and rep['gate'] == 'aggressive' and rep['gated_in'] == 1 and rep['gated_out'] == 1, rep
        assert eng4['mA'] == eng['mA'] and eng4['mC'] == ENG['mC'] and eng4['mB'] == ENG['mB'], 'gate not applied as specified'
        eng5 = json.loads(b0)
        rep = g['s1v_apply'](dict(S1V=dict(sw4, gate='bogus'), ENG=eng5, PC=pcg, BASE=base, LIB_TAU=0.9, REL_TH=600.0, ENG_DIR=d,
                                  IS_RERUN=True))
        assert rep['status'].startswith('failed') and eng5 == ENG, rep                 # unknown gate -> ENG unchanged
        pcg['mC']['fz_top'] = 400.0                                                   # mC rel 400: in for 'rel>300', out for 'aggressive'
        eng6 = json.loads(b0)
        rep = g['s1v_apply'](dict(S1V=dict(sw4, out=os.path.join(d, 'p2e.json'), gate='rel>300'), ENG=eng6, PC=pcg, BASE=base, LIB_TAU=0.9,
                                  REL_TH=600.0, ENG_DIR=d, IS_RERUN=True))
        assert rep['gate'] == 'rel>300' and rep['gated_in'] == 2 and eng6['mA'] == eng['mA'] and eng6['mC'] == eng['mC'], rep
        # gate 'learned' (PREREG section 8): exactly the molecules whose frozen score passes, from pass 2's own entries
        eng7, o7 = json.loads(b0), os.path.join(d, 'p2f.json')
        rep = g['s1v_apply'](dict(S1V=dict(sw4, out=o7, gate='learned'), ENG=eng7, PC=pcg, BASE=base, LIB_TAU=0.9, REL_TH=600.0,
                                  ENG_DIR=d, IS_RERUN=True))
        P7 = json.load(open(o7))['mols']
        want = {m: g['s1v_gate_ok']('learned', pcg[m], 600.0, base[m], ENG[m], P7[m])[0] for m in ('mA', 'mC')}
        assert rep['status'] == 'ok' and rep['gate'] == 'learned' and rep['gated_in'] == sum(want.values()), (rep, want)
        for m in ('mA', 'mC'):
            assert eng7[m] == (eng[m] if want[m] else ENG[m]), m
        assert eng7['mB'] == ENG['mB']
    finally:
        sys.path[:] = [p for p in sys.path if p != d]
        for k in [k for k in sys.modules if k in ('casmi_engine', 'pv', 'pv_fp')]:
            del sys.modules[k]
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_') and callable(f):
            f(); print(name, 'OK', flush=True)
    print('ALL OK')
