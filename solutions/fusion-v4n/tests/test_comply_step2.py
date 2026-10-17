"""Tests of the stack1 compliance step-2 builder (scratch/comply/step2/build_step2.py) and of the training / inference consistency of
the engine-1 ranker rows (train/comply/e1_rows.py); no competition data, no network, no Kaggle."""
import contextlib, json, os, sys, types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'scratch', 'comply', 'step2'))
sys.path.insert(0, os.path.join(ROOT, 'solutions', 'fusion-v4n', 'train', 'comply'))
import build_step2 as B  # noqa: E402
import e1_rows as ER  # noqa: E402

SHA = {n: chr(97 + i) * 64 for i, n in enumerate(B.FILES)}
CFG_ON = dict(e1=True, sha256=SHA, net_sha256='n' * 64, n_features=3)
CFG_OFF = dict(CFG_ON, e1=False)


@contextlib.contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f'{exc.__name__} not raised')


def _code(text):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=text.splitlines(keepends=True))


def _toy_nb():
    """The anchor cells of s1m (step-1 switch, engine cell 18, base-list cell 20) with their exact anchor lines, plus untouched cells."""
    c18 = '\n'.join([B.ANCHOR_18, 'import v1engine'] + [o for o, _ in B.SUBS_18[:3]] + ['def rank_score(X, names):', '    pos = {}',
                                                                                        B.SUBS_18[3][0], '    return Xs']) + '\n'
    c20 = B.ANCHOR_20 + '\nn_err, n_pp = 0, 0\n' + B.SUBS_20[0][0] + '\n'
    return dict(nbformat=4, nbformat_minor=4, metadata={}, cells=[
        _code("print('cell 8')\n"), _code(B.ANCHOR_S1 + '\n"""\nCOMPLY_S1 = {}\n'), _code(c18), _code(c20), _code("print('untouched')\n")])


def _srcs(nb):
    return [''.join(c['source']) for c in nb['cells']]


def test_patch_changes_only_the_anchors():
    nb = _toy_nb(); p = B.patch(nb, CFG_ON)
    a, b = _srcs(nb), _srcs(p)
    assert len(b) == len(a) + 1 and b[:2] == a[:2] and b[-1] == a[-1]
    assert b[2].startswith('"""Compliance step 2')
    want18 = a[2]
    for old, new in B.SUBS_18:
        want18 = want18.replace(old, new)
    assert b[3] == want18 and b[4] == a[3].replace(B.SUBS_20[0][0], B.SUBS_20[0][1])
    for s in b:
        compile(s, 'cell', 'exec')
    assert _srcs(nb) == a                                    # input not mutated


def test_patch_refuses_ambiguous_or_misplaced_anchor():
    nb = _toy_nb(); nb['cells'].append(_code("x = '''" + B.SUBS_18[2][0] + "'''\n"))
    with raises(AssertionError):
        B.patch(nb, CFG_ON)
    nb = _toy_nb(); nb['cells'][2]['source'] += ['\n', B.SUBS_18[0][0] + '\n']
    with raises(AssertionError):
        B.patch(nb, CFG_ON)


def _ns(cfg, files=None):
    """Namespace of cell 8 stand-ins (find / sha256 / MAN) after executing the inserted switch cell."""
    files = files or {}
    man = dict(families=['derivation', 'frag2', 'analog_struct', 'model_views', 'fragnet'])
    root = '/in'

    def find(pat):
        if not files:
            raise FileNotFoundError(pat)
        return root + ('/casmi26-ours-pcnets/fpnet_own_s0.pt' if 'pcnets' in pat else '/casmi26-ours-e1/e1_features.json')
    feats = ['a', 'b', 'c']
    real_open = open

    def fake_open(p, *a, **k):
        import io
        p = str(p).replace('\\', '/')
        if p.endswith('e1_features.json'):
            return io.StringIO(json.dumps(feats))
        if '/casmi26-ours-e1/' in p:
            return io.StringIO('tree')
        return real_open(p, *a, **k)
    ns = dict(os=os, json=json, MAN=man, find=find, open=fake_open,
              sha256=lambda p: files[os.path.basename(str(p).replace('\\', '/'))])
    exec(B.switch_cell(cfg), ns)
    return ns


def test_switch_off_is_s1m():
    ns = _ns(CFG_OFF)
    assert ns['COMPLY2_BANK'] == [] and ns['COMPLY2_FE_PATHS'] is None and ns['COMPLY2_RANKER'] is None
    assert 'derivation' in ns['MAN']['families']
    X = np.arange(12.0).reshape(3, 4)
    assert ns['COMPLY2_PREP'](X, ['a', 'b', 'c', 'd']) is X                           # identity, same object
    assert ns['COMPLY2_ENGINE_HOOK']() is None and ns['COMPLY2_ENGINE_CHECK'](None) is None
    # the substituted expressions fall back to the original ones
    assert (ns['COMPLY2_BANK'] or ['glob']) == ['glob'] and (ns['COMPLY2_FE_PATHS'] or ['orig']) == ['orig'] and (ns['COMPLY2_RANKER'] or 'pkl') == 'pkl'


def test_switch_on_pins_and_drops_derivation():
    files = dict(SHA); files['fpnet_own_s0.pt'] = 'n' * 64
    ns = _ns(CFG_ON, files)
    assert ns['COMPLY2_BANK'] == ['/in/casmi26-ours-pcnets/fpnet_own_s0.pt'] and ns['COMPLY2_FE_PATHS'] == ns['COMPLY2_BANK'] * 3
    assert ns['COMPLY2_RANKER']['features'] == ['a', 'b', 'c'] and len(ns['COMPLY2_RANKER']['boosters']) == 4
    assert ns['MAN']['families'] == ['frag2', 'analog_struct', 'model_views', 'fragnet']
    bad = dict(files); bad['e1_ranker_s2.txt'] = 'x' * 64
    with raises(AssertionError):
        _ns(CFG_ON, bad)                                     # a changed file stops the run
    bad = dict(files); bad['fpnet_own_s0.pt'] = 'x' * 64
    with raises(AssertionError):
        _ns(CFG_ON, bad)
    with raises(FileNotFoundError):
        _ns(CFG_ON, {})                                      # no dataset: no silent fallback


def test_prep_fallback_per_list():
    files = dict(SHA); files['fpnet_own_s0.pt'] = 'n' * 64
    ns = _ns(CFG_ON, files)
    X = np.zeros((3, 2))
    out = ns['COMPLY2_PREP'](X, ['fz', 'x'])                 # source columns missing -> the engine's own values, counted
    assert out is X and ns['_C2']['fallback'] == 1 and ns['_C2']['lists'] == 0


def test_inference_prep_equals_training_rows():
    """The notebook's COMPLY2_PREP on one list == e1_rows.rerank_average on the same list (training / inference consistency)."""
    files = dict(SHA); files['fpnet_own_s0.pt'] = 'n' * 64
    ns = _ns(CFG_ON, files)
    rng = np.random.RandomState(0)
    names = ['fz', 'lib_max', 'ap', 'el_sum', 'fr_int', 'sf_int', 'mass_ppm', 'other'] + list(ER.RANK_SOURCES)
    for n in (1, 2, 7, 50):
        X = rng.randint(-3, 4, size=(n, len(names))).astype(np.float64) / 3.0          # many ties, negative values
        a = ns['COMPLY2_PREP'](X, names); b = ER.rerank_average(X.astype(np.float32), names, np.array([0, n]))
        assert a.dtype == np.float32 and np.array_equal(a, b), n
    assert ER.RANK_SOURCES == ns['_C2_RANK']


def test_feature_names_drop_derivation():
    man = dict(family_features=['derivation__x', 'frag2__a', 'fragnet__fn_score', 'derivation__y', 'model_views__fzD_z'])
    assert ER.feature_names(['fz', 'fz_rank'], man) == ['fz', 'fz_rank', 'frag2__a', 'fragnet__fn_score', 'model_views__fzD_z']


def test_strip_injected():
    R = types.SimpleNamespace(names_x=['fz'], names_f=[], mids=['a', 'b'], empty=[], start=np.array([0, 3, 4]), lib_max=np.array([0.1, 0.2]),
                              X=np.zeros((4, 1), np.float32), F=np.zeros((4, 0), np.float32), pid=np.array([1, -2, 3, -2]),
                              y=np.array([0, 1, 0, 0], bool), inj=np.array([0, 1, 0, 1], bool), smiles=np.array(list('wxyz'), object),
                              formula=np.array(list('ffff'), object), key=np.array(list('kkkk'), object))
    S = ER.strip_injected(R)
    assert S.mids == ['a'] and S.empty == ['b'] and list(S.start) == [0, 2] and list(S.smiles) == ['w', 'y'] and not S.y.any()


def test_real_notebook_if_present():
    if not os.path.exists(B.BASE_NB):
        print('  (s1m notebook not present: skipped)'); return
    raw = open(B.BASE_NB, 'rb').read(); nb = json.loads(raw.decode('utf-8'))
    p = B.patch(nb, CFG_OFF)
    a, b = _srcs(nb), _srcs(p)
    assert len(b) == len(a) + 1
    changed = [i for i in range(len(a)) if a[i] != b[i + (1 if i > next(j for j, s in enumerate(a) if s.startswith(B.ANCHOR_S1)) else 0)]]
    assert len(changed) == 2, changed                        # cells 18 and 20 only
    for c in p['cells']:
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), 'cell', 'exec')


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            f(); print('ok', k)
    print('ALL OK')
