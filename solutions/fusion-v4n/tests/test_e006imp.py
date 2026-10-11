"""E006_BASE ("e006 improved": the public notebook's script on OUR base lists): off by default and off == v59a; the patches; the stage cell.

Checks, without competition data:
  * the six keys are off by default; a build without the option carries neither the two cells nor a cell of the public notebook and equals a
    build made before the option existed cell for cell (the cells of nb/[0-9][0-9]_*.py only); a build with it is the same notebook plus the
    stage cell after cell 06, the final cell at the end and the embedded cells, with the same datasets;
  * the improved script parses; undoing the four patches and removing our preamble / postamble gives the E006_FUSE script back; with no
    environment variable set the patched lines keep the notebook's own values (rel 250, GLACIER on [M+H]+ only, no final promotion);
  * patch 4 in isolation: the patched engine block takes our rows for a molecule in the hand-over file and calls the engine for another;
  * the hand-over dict (order and scores kept, formulas untagged, TOPN cut), the joined-row popularity term == fusion_core's own function;
  * the final promotion: the promoted row leads, a row with the same identity is moved instead of duplicated, 25 rows at most, the other
    rows keep their order, molecules without a popularity-branch promotion keep their row byte for byte;
  * the stage cell and the final cell executed with stubs: stage ok -> the script's table is the submission and cells 07 / 08 are told to
    skip the forward models; stage failed / late / bad table / a fallback inside the script -> CFG untouched and our own table stays;
  * the presets v67a / v67b / v67c are v59a + FWD_GL_ALONE + the option(s); FWD_GL_ALONE itself changes nothing while ICEBERG is healthy
    (tests/test_fwd_gl_alone.py holds that check; here: the two presets' fusion_core output on a healthy synthetic run is identical).
usage: python tests/test_e006imp.py   (or pytest)
"""
import glob, json, os, subprocess, sys, tempfile, types

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.join(SOL, 'tools'))
import e006_stage as ES
import fusion_core as FC
import build_notebook as BN

KEYS = dict(E006_BASE=None, E006_PROM_REL=None, E006_GL_ALL=False, E006_PROM_FINAL=False, E006_JOIN_POP_MU=0.0)
PRESETS = ('v67a-e006-ours-base.json', 'v67b-e006-ours-base-promfinal.json', 'v67c-e006-ours-base-promfinal-joinpop010.json')


def load_cfg():
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    return ns['CFG']


def snapshot_cells():
    nb = json.load(open(BN.E006_NB, encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def build(cfg_name, out):
    r = subprocess.run([sys.executable, os.path.join(SOL, 'tools', 'build_notebook.py'), '--user', 'u', '--slug', 's', '--config',
                        os.path.join(SOL, 'configs', cfg_name), '--out', out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    nb = json.load(open(os.path.join(out, 's.ipynb'), encoding='utf-8')); meta = json.load(open(os.path.join(out, 'kernel-metadata.json')))
    ns = {}
    exec(''.join(nb['cells'][0]['source']), ns)
    return [(c['id'], ''.join(c['source'])) for c in nb['cells']], ns['EMBED'], meta


def test_off_by_default():
    cfg = load_cfg()
    assert all(cfg[k] == v for k, v in KEYS.items()), {k: cfg[k] for k in KEYS}
    assert cfg['E006_BASE_DEADLINE_S'] + 75 * 60 <= 7.5 * 3600 + 300          # the stage's deadline leaves room for our own forward stage
    assert ES.improved_env(cfg) == {}
    assert ES.improved_env(dict(cfg, E006_PROM_REL=1e12, E006_GL_ALL=True, E006_PROM_FINAL=True)) == \
        dict(OURS_E006_PROM_REL=repr(1e12), OURS_E006_GL_ALL='1', OURS_E006_PROM_FINAL='1')


def test_builds():
    with tempfile.TemporaryDirectory() as t:
        c0, e0, m0 = build('v59a-v33b-ccby.json', os.path.join(t, 'a'))
        c1, e1, m1 = build('v67a-e006-ours-base.json', os.path.join(t, 'b'))
    files = sorted(glob.glob(os.path.join(SOL, 'nb', '[0-9][0-9]_*.py')))
    assert [i for i, _ in c0] == ['embed'] + [os.path.basename(p)[:-3].replace('_', '-') for p in files]      # no cell of the option without it
    assert not any(k.startswith('e006/') for k in e0)
    assert set(e1) - set(e0) == {'e006/cells.json', 'e006/NOTICE.md'} and all(e1[k] == e0[k] for k in e0)
    ids1 = [i for i, _ in c1]
    assert ids1.index('e006imp-stage') == ids1.index('06-base-lists') + 1 and ids1[-1] == 'e006imp-final' and ids1.index('07-rescore') == ids1.index('e006imp-stage') + 1
    d0, d1 = dict(c0), dict(c1)
    assert all(d0[k] == d1[k] for k in d0 if k not in ('embed', '00-config')) and [i for i in ids1 if i not in d0] == ['e006imp-stage', 'e006imp-final']
    assert d0['00-config'].split('\nCFG.update(')[0] == d1['00-config'].split('\nCFG.update(')[0]
    assert m0['dataset_sources'] == m1['dataset_sources'] and m0['docker_image'] == m1['docker_image'] and m1['is_private'] is True
    assert json.loads(e1['e006/cells.json']) == snapshot_cells()


def test_script_patches():
    cells = snapshot_cells(); work = '/kaggle/working/e006'
    s0 = ES.script_from_cells(cells, work); s1 = ES.script_improved(cells, work)
    compile(s1, 'e006imp', 'exec')
    assert s1.startswith(ES.PREAMBLE + ES.PREAMBLE_IMP) and s1.endswith(ES.POSTAMBLE_IMP)
    body = s1[len(ES.PREAMBLE + ES.PREAMBLE_IMP):-len(ES.POSTAMBLE_IMP)]
    for old, new in ((ES.BASE_CALL, ES.BASE_CALL_NEW), (ES.REL_TEXT, ES.REL_TEXT_NEW), (ES.GL_LINE, ES.GL_LINE_NEW)):
        assert body.count(new) == 1
        body = body.replace(new, old)
    assert ES.PREAMBLE + body == s0                                             # nothing else of the notebook is touched
    assert s1.count('/kaggle/working') == s1.count('/kaggle/working/e006')
    # defaults of the patched lines = the notebook's own values
    ns = {}
    env = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith('OURS_E006_')}
    try:
        exec(compile(ES.PREAMBLE + ES.PREAMBLE_IMP, 'pre', 'exec'), ns)
    finally:
        os.environ.update(env)
    assert ns['_OURS_PROM_REL'] == 250.0 and ns['_OURS_BASE'] == {} and ns['_OURS_PROM'] == {}
    assert "if os.environ.get('OURS_E006_GL_ALL'):" in s1 and "if os.environ.get('OURS_E006_PROM_FINAL'):" in s1
    return ns


def test_patch4_block():
    """The patched engine block, cut out of the script and run with stubs: our rows for a handed-over molecule, the engine for another."""
    s1 = ES.script_improved(snapshot_cells(), '/kaggle/working/e006')
    a = s1.index('        try:\n' + ES.BASE_CALL_NEW); b = s1.index("        except Exception as e:\n            print('ERROR', mid, repr(e)); n_err += 1")
    block = s1[a:b] + "        except Exception as e:\n            raise\n"
    src = 'def run(mid, spectra, target, V):\n    smis, keys, lib_max, scs, forms, pids_ = [], [], 0.0, [], [], []\n    if True:\n' + block + \
        '    return smis, keys, lib_max, scs, forms, pids_\n'
    calls = []

    class V:
        @staticmethod
        def run(spectra, target):
            calls.append(target)
            return types.SimpleNamespace(pid=[]), None, None, [], dict(lib_max=0.33)
    rows = [['A', 'B', 'C'], ['ka', 'kb', 'kc'], 0.42, [3.0, 2.0, 1.0], ['C2H6|gen', 'C2H6', 'CH4'], [-2, 5, 7]]
    ns = dict(types=types, TOPN=2, _OURS_BASE={'m1': rows}, _ours_base_rows=test_script_patches()['_ours_base_rows'])
    exec(compile(src, 'blk', 'exec'), ns)
    assert ns['run']('m1', [], 100.0, V) == (['A', 'B'], ['ka', 'kb'], 0.42, [3.0, 2.0], ['C2H6', 'C2H6'], [-2, 5]) and calls == []
    assert ns['run']('m2', [], 100.0, V) == ([], [], 0.33, [], [], []) and calls == [100.0]


def test_handover_and_join_pop():
    base = {'m1': [['A', 'J1', 'B', 'J2'], ['ka', 'kj1', 'kb', 'kj2'], 0.3, [4.0, 3.0, 2.0, 1.0], ['F', 'F', 'G', 'G'], [5, -2, 7, -2]],
            'm2': [[], [], 0.0, [], [], []], 'm3': [['X'], ['kx'], 0.95, [1.0], ['H'], [1]]}
    out, n = ES.base_for_handover(base)
    assert n == 0 and set(out) == {'m1', 'm3'} and out['m1'] == [base['m1'][0], base['m1'][1], 0.3, base['m1'][3], ['F', 'F', 'G', 'G'], [5, -2, 7, -2]]
    jp = {'m1': {'J1': 1.0, 'J2': 60.0}}
    out2, n2 = ES.base_for_handover(base, jp, 0.10, FC.popularity_reorder_join)
    ref = FC.popularity_reorder_join(base['m1'][0], base['m1'][1], base['m1'][3], base['m1'][4], base['m1'][5], None, 0.0, [None, 1.0, None, 60.0], 0.10, False)
    assert n2 == 1 and out2['m1'][0] == ref[0] == ['A', 'J2', 'B', 'J1'] and out2['m1'][3] == ref[2] and out2['m1'][5] == ref[4]
    assert out2['m1'][4] == ['F', 'G', 'G', 'F'] and out2['m3'] == out['m3']                # formulas untagged, pool rows keep their slots
    assert ES.base_for_handover(base, jp, 0.0, FC.popularity_reorder_join)[0] == out            # weight 0 == off


def test_final_promotion():
    ns = test_script_patches()
    ident = lambda s: None if s.startswith('bad') else s.split('~')[0]
    pr = ns['_ours_promote_rows']
    L = ['a', 'b', 'c~1', 'd']
    assert pr(L, 'c', ident) == ['c~1', 'a', 'b', 'd']                           # same identity: the list's own row moves
    assert pr(L, 'z', ident) == ['z', 'a', 'b', 'c~1', 'd'] and pr(L, 'a', ident) == L
    assert pr(['r%d' % i for i in range(25)], 'z', ident) == ['z'] + ['r%d' % i for i in range(24)]
    assert pr(['a', 'bad1', 'b'], 'bad1', ident) == ['bad1', 'a', 'b']
    import pandas as pd
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, 's.csv')
        pd.DataFrame(dict(molecule_id=['m1', 'm2', 'm3'], smiles=['a;b;c', 'a;b;c', 'CCO'])).to_csv(p, index=False)
        ns['_OURS_PROM'].update(m1='c', m3='q')
        ns['_ours_promote_final'](p, ident)
        got = pd.read_csv(p, dtype=str)
        assert list(got.smiles) == ['c;a;b', 'a;b;c', 'CCO']                     # no promotion / placeholder: unchanged


def run_cells(tmp, cfg_over, script_result, table=None, log_text=None):
    """nb/e006imp_stage.py and nb/e006imp_final.py with stubs. script_result = (returncode, status) of the stubbed script run."""
    import pandas as pd
    cfg = dict(load_cfg(), E006_BASE='ours', FWD_GL_ALONE=True, **cfg_over)
    work = os.path.join(tmp, 'work'); comp = os.path.join(tmp, 'comp'); ours = os.path.join(tmp, 'ours')
    for d in (work, comp, os.path.join(ours, 'e006')):
        os.makedirs(d, exist_ok=True)
    json.dump(snapshot_cells(), open(os.path.join(ours, 'e006', 'cells.json'), 'w'))
    ids = ['m1', 'm2', 'm3']
    pd.DataFrame(dict(molecule_id=ids, smiles='CCO')).to_csv(os.path.join(comp, 'sample_submission.csv'), index=False)
    base = {'m1': [['A', 'B'], ['ka', 'kb'], 0.3, [2.0, 1.0], ['F', 'F'], [5, -2]], 'm2': [['C'], ['kc'], 0.95, [1.0], ['G'], [3]]}
    seen = {}

    def fake_run(script, w, env, timeout_s, log_path):
        seen.update(env={k: v for k, v in env.items() if k.startswith('OURS_E006_')}, timeout=timeout_s, base=json.load(open(env['OURS_E006_BASE'])))
        compile(open(script).read(), 'script', 'exec')
        open(log_path, 'w').write(log_text if log_text is not None else 'x\nE006_STAGE base lists used: 2 of 2 | promotions by the popularity branch: 0\n')
        if table is not None:
            pd.DataFrame(dict(molecule_id=ids, smiles=table)).to_csv(os.path.join(w, 'submission.csv'), index=False)
        return script_result[0], 5.0, script_result[1]
    real = ES.run_script; ES.run_script = fake_run
    real_si = ES.script_improved; ES.script_improved = lambda cells, w: real_si(cells, '/kaggle/working/e006')   # the work folder must be Kaggle's
    logs = []
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
    import time
    ns = dict(CFG=cfg, BASE=base, PC={}, ENG={'m1': 1}, FAILS=[], WORK=work, COMP=comp, OURS=ours, T0=time.time(), IS_RERUN=True, os=os, json=json, time=time,
              pd=pd, torch=torch, fusion_core=FC, log=lambda *a: logs.append(a), dump=lambda *a: None, find=None, PC_DIR=None)
    try:
        exec(compile(open(os.path.join(SOL, 'nb', 'e006imp_stage.py'), encoding='utf-8').read(), 'stage', 'exec'), ns)
        cfg_after_stage = dict(ns['CFG'])
        # cell 08's stand-in / own table and manifest
        ns['samp'] = pd.read_csv(os.path.join(comp, 'sample_submission.csv')); ns['sub'] = pd.DataFrame(dict(molecule_id=ids, smiles=['o1;o2', 'o3', 'CCO']))
        ns['sub'].to_csv(os.path.join(work, 'submission.csv'), index=False); json.dump({}, open(os.path.join(work, 'run_manifest.json'), 'w'))
        exec(compile(open(os.path.join(SOL, 'nb', 'e006imp_final.py'), encoding='utf-8').read(), 'final', 'exec'), ns)
    finally:
        ES.run_script = real; ES.script_improved = real_si
    return ns, cfg_after_stage, seen, list(pd.read_csv(os.path.join(work, 'submission.csv'), dtype=str).smiles)


def test_stage_cells():
    cfg0 = load_cfg()
    with tempfile.TemporaryDirectory() as t:                                     # healthy stage
        ns, c, seen, final = run_cells(t, dict(E006_PROM_FINAL=True), (0, 'ok'), table=['e1;e2', 'e3', 'CCO'])
    assert ns['E6I']['status'] == 'ok' and final == ['e1;e2', 'e3', 'CCO'] and not ns['FAILS']
    assert (c['USE_ICE'], c['USE_GL'], c['FRAG_LAM']) == (False, False, 0.0) and c['FWD_GL_ALONE'] is True
    assert seen['env'].get('OURS_E006_PROM_FINAL') == '1' and 'OURS_E006_GL_ALL' not in seen['env'] and 'OURS_E006_PROM_REL' not in seen['env']
    assert 'OURS_E006_ENG' in seen['env'] and 'OURS_E006_FULL' not in seen['env'] and set(seen['base']) == {'m1', 'm2'}
    assert 0 < seen['timeout'] <= cfg0['E006_BASE_DEADLINE_S']
    bad = [((1, 'failed'), ['e1', 'e3', 'CCO'], None), ((None, 'timeout'), None, None),                    # script failed / ran out of time
           ((0, 'ok'), ['CCO', 'e3', 'CCO'], None),                                                        # a molecule with a base list has no row
           ((0, 'ok'), ['e1', 'e3', 'CCO'], 'x\nICE FAILED -> ranker order kept\nE006_STAGE base lists used: 2 of 2 |\n'),   # fallback inside
           ((0, 'ok'), ['e1', 'e3', 'CCO'], 'x\nE006_STAGE base lists used: 0 of 2 |\n'),                    # the hand-over did not happen
           ((0, 'ok'), [';'.join('r%d' % i for i in range(26)), 'e3', 'CCO'], None)]                         # 26 rows
    for res, table, lg in bad:
        with tempfile.TemporaryDirectory() as t:
            ns, c, seen, final = run_cells(t, {}, res, table=table, log_text=lg)
        assert ns['E6I']['status'].startswith('failed') and final == ['o1;o2', 'o3', 'CCO'], (res, table, ns['E6I'])
        assert (c['USE_ICE'], c['USE_GL'], c['FRAG_LAM']) == (cfg0['USE_ICE'], cfg0['USE_GL'], cfg0['FRAG_LAM']) and len(ns['FAILS']) == 1
    # option off: the cells define E6I and touch nothing else
    ns = dict(CFG=dict(cfg0))
    exec(compile(open(os.path.join(SOL, 'nb', 'e006imp_stage.py'), encoding='utf-8').read(), 'stage', 'exec'), ns)
    assert set(ns) - {'__builtins__'} == {'CFG', 'E6I'} and ns['E6I'] == dict(status='off') and ns['CFG'] == cfg0


def test_presets():
    cfg = load_cfg(); ref = json.load(open(os.path.join(SOL, 'configs', 'v59a-v33b-ccby.json')))
    found = 0
    for name in PRESETS:
        f = os.path.join(SOL, 'configs', name)
        if not os.path.exists(f):
            continue
        found += 1; p = json.load(open(f))
        extra = {k: v for k, v in p.items() if k not in ref}
        assert all(p[k] == v for k, v in ref.items() if k != 'VERSION') and set(p) <= set(cfg)
        assert extra.pop('FWD_GL_ALONE') is True and extra.pop('E006_BASE') == 'ours' and set(extra) <= set(KEYS), extra
    assert found == 3
    # FWD_GL_ALONE on a healthy run (every molecule has ICEBERG scores): the fallback table is v59a's, bit for bit
    import random
    rng = random.Random(7); base, eng, ice, gl, pc = {}, {}, {}, {}, {}
    for i in range(40):
        m = 'm%d' % i; n = rng.randint(3, 30); smis = ['%s_s%d' % (m, j) for j in range(n)]
        base[m] = [smis, ['k' + s for s in smis], rng.random(), sorted((rng.gauss(0, 1) for _ in smis), reverse=True), ['F%d' % rng.randint(0, 3) for _ in smis],
                   [rng.choice([-2, -1, 1, 2, 3]) for _ in smis]]
        eng[m] = dict(smiles=smis[::-1][:10] + ['%s_e%d' % (m, j) for j in range(5)], keys=['k' + s for s in smis[::-1][:10]] + ['k%s_e%d' % (m, j) for j in range(5)])
        ice[m] = {s: rng.random() for s in smis + eng[m]['smiles']}; gl[m] = {s: (rng.random() if rng.random() < 0.9 else None) for s in ice[m]}
        pc[m] = dict(pc=['%s_p%d' % (m, j) for j in range(6)], pc_keys=['k%s_p%d' % (m, j) for j in range(6)], pc_fz=sorted((rng.gauss(0, 1) for _ in range(6)), reverse=True),
                     best_pool_fz=0.0, target=100.0)
        ice[m].update({s: rng.random() for s in pc[m]['pc']})
    import importlib.util

    class Fz:                                   # the packages' group re-rank is not shipped with the tests: fusion_core's own arithmetic stands in
        @staticmethod
        def rerank(smis, keys, scs, forms, d, lam=1.0, top_n=60):
            return FC.rerank_joined(smis, keys, scs, forms, [d], [lam], top_n, [False] * len(smis), 1.0)

        @staticmethod
        def rerank_multi(smis, keys, scs, forms, ds, lams, top_n=60):
            return FC.rerank_joined(smis, keys, scs, forms, ds, lams, top_n, [False] * len(smis), 1.0)
    out = []
    for flag in (False, True):
        c = dict(ref, FWD_GL_ALONE=flag); mids = sorted(base)
        sub, _ = FC.build_submission(base, pc, eng, ice, gl, c, mids, mids, ice_fuse=Fz, gl_fuse=Fz, score_key=lambda s: 'k' + s, formula=lambda s: 'F', log=lambda *a: None,
                                     pool_pop=__import__('numpy').ones(10))
        out.append(list(sub.smiles))
    assert out[0] == out[1]


if __name__ == '__main__':
    test_off_by_default(); test_builds(); test_script_patches(); test_patch4_block(); test_handover_and_join_pop(); test_final_promotion()
    test_stage_cells(); test_presets()
    print('E006_BASE OK: off by default, builds, script patches, base hand-over, joined-row popularity, final promotion, stage cells with stubs, presets, '
          'FWD_GL_ALONE inert on a healthy run')
