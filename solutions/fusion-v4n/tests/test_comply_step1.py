"""Tests of the stack1 compliance step-1 builder (scratch/comply/step1/build_step1.py): off = base bytes, anchors, switch semantics; no competition data."""
import ast, contextlib, json, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'scratch', 'comply', 'step1'))
import build_step1 as B  # noqa: E402


@contextlib.contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    raise AssertionError(f'{exc.__name__} not raised')

SHA = {'fpnet_own_s0.pt': 'a' * 64, 'fpnet_own_s1.pt': 'b' * 64}


def _code(text):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=text.splitlines(keepends=True))


def _toy_nb():
    """Three cells carrying the exact anchor lines of stack1's cells 11, 15 and 24, plus an untouched cell."""
    return dict(nbformat=4, nbformat_minor=4, metadata={}, cells=[
        _code(B.ANCHOR_SWITCH + "\nIS_RERUN = True\n"),
        _code("PC = {}\ntry:\n    rr = subprocess.run(" + B.OLD_15 + " os.path.join(COMP, 'test.parquet')], timeout=1)\nexcept Exception:\n    pass\n"),
        _code("if (S is not None and float(S) > S_TAU and pop_top >= POP_TAU) " + B.OLD_24 + "\n    x = 1\n"),
        _code("print('untouched')\n")])


def _srcs(nb):
    return [''.join(c['source']) for c in nb['cells']]


def test_patch_changes_only_the_anchors():
    nb = _toy_nb(); p = B.patch(nb, dict(pc_nets=True, k=10.0, sha256=SHA))
    a, b = _srcs(nb), _srcs(p)
    assert len(b) == len(a) + 1 and b[0] == a[0] and b[-1] == a[-1]
    assert b[1].startswith('"""Compliance step 1') and b[2] == a[1].replace(B.OLD_15, B.NEW_15) and b[3] == a[2].replace(B.OLD_24, B.NEW_24)
    for s in b:
        compile(s, 'cell', 'exec')
    assert _srcs(nb) == a                                    # input not mutated


def test_patch_refuses_ambiguous_anchor():
    nb = _toy_nb(); nb['cells'].append(_code("x = '" + B.OLD_24 + "'\n"))
    with raises(AssertionError):
        B.patch(nb, dict(pc_nets=True, k=10.0, sha256=SHA))


def _run_switch(cfg, files=None, rel_th=600.0):
    """Execute the inserted cell with stand-ins for cell 8's find / sha256 / glob."""
    files = files or {}
    ns = dict(os=os, json=json, V3MOD='/in/casmi26-v3-models', REL_TH=rel_th,
              find=lambda pat: (_ for _ in ()).throw(FileNotFoundError(pat)) if not files else '/in/casmi26-ours-pcnets/fpnet_own_s0.pt',
              sha256=lambda p: files[os.path.basename(p)],
              glob=type('G', (), {'glob': staticmethod(lambda pat: sorted('/in/casmi26-ours-pcnets/' + n for n in files if n.startswith('fpnet_')))}))
    exec(B.switch_cell(cfg), ns)
    return ns


def test_switch_off_is_stack1():
    ns = _run_switch(dict(pc_nets=False, k=12.5, sha256=SHA))
    assert ns['COMPLY_PC_MODELS'] == '/in/casmi26-v3-models' and ns['COMPLY_REL_PROMO'] == 150.0 and ns['REL_TH'] == 600.0
    assert type(ns['COMPLY_REL_PROMO']) is float


def test_switch_on_rescales_and_pins():
    ns = _run_switch(dict(pc_nets=True, k=12.5, sha256=SHA), files=dict(SHA))
    assert ns['COMPLY_PC_MODELS'].replace('\\', '/') == '/in/casmi26-ours-pcnets'
    assert abs(ns['REL_TH'] - 48.0) < 1e-12 and abs(ns['COMPLY_REL_PROMO'] - 12.0) < 1e-12


def test_switch_on_matched_thresholds():
    ns = _run_switch(dict(pc_nets=True, k=12.5, sha256=SHA, thresholds=[20.0, 57.0]), files=dict(SHA))
    assert ns['COMPLY_REL_PROMO'] == 20.0 and ns['REL_TH'] == 57.0
    ns = _run_switch(dict(pc_nets=False, k=12.5, sha256=SHA, thresholds=[20.0, 57.0]))
    assert ns['COMPLY_REL_PROMO'] == 150.0 and ns['REL_TH'] == 600.0     # thresholds only act with the switch on


def test_switch_on_never_falls_back():
    with raises(FileNotFoundError):                    # dataset missing: the run stops, A+B are not used silently
        _run_switch(dict(pc_nets=True, k=12.5, sha256=SHA))
    bad = dict(SHA); bad['fpnet_own_s1.pt'] = 'c' * 64
    with raises(AssertionError):                       # hash mismatch
        _run_switch(dict(pc_nets=True, k=12.5, sha256=SHA), files=bad)
    extra = dict(SHA); extra['fpnet_0.pt'] = 'd' * 64
    with raises(AssertionError):                       # a third fpnet_*.pt would enter the runner's ModelBank
        _run_switch(dict(pc_nets=True, k=12.5, sha256=SHA), files=extra)


def test_gate_literal_equivalence():
    """With the switch off the substituted comparison is the literal one for every rel (cell 24)."""
    for rel in (-1e9, 0.0, 149.999, 150.0, 150.0000001, 1e9, float('nan')):
        assert (rel > 150.0) == (rel > _run_switch(dict(pc_nets=False, k=3.0, sha256=SHA))['COMPLY_REL_PROMO'])


def test_real_notebook_off_bytes_and_on_diff():
    if not os.path.exists(B.BASE_NB):                       # stack1 notebook not pulled (scratch/nb1016/omer): nothing to check
        print('skip: base notebook missing'); return
    with tempfile.TemporaryDirectory() as d:
        w = B.build(d, 12.5, sha=SHA, variants=('off', 'patched-off', 'on', 'on-matched'), matched=(20.0, 57.0))
        assert open(w['off'], 'rb').read() == open(B.BASE_NB, 'rb').read()
        base = _srcs(json.load(open(B.BASE_NB, encoding='utf-8')))
        for v in ('patched-off', 'on', 'on-matched'):
            s = _srcs(json.load(open(w[v], encoding='utf-8')))
            assert len(s) == len(base) + 1
            changed = [i for i, x in enumerate(base) if x not in s]
            assert len(changed) == 2, changed                  # cells 15 and 24 only
            code = [x for x, c in zip(s, json.load(open(w[v], encoding='utf-8'))['cells']) if c['cell_type'] == 'code']
            for x in code:
                ast.parse(x)
            cell = [x for x in s if x.startswith('"""Compliance step 1')][0]
            cfg_line = [ln for ln in cell.splitlines() if ln.startswith('COMPLY_S1 = ')][0]
            assert ("'pc_nets': True" in cfg_line) == v.startswith('on') and ("'thresholds'" in cfg_line) == (v == 'on-matched')
        meta = json.load(open(os.path.join(os.path.dirname(w['on']), 'kernel-metadata.json')))
        assert B.DATASET in meta['dataset_sources'] and meta['is_private'] and meta['id'].endswith('casmi26-comply-s1')


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            f(); print('ok', k)
