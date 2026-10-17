"""Tests of the v72 probe cell (nb/embed/arb_v72.py: B1 + B2 after E2R2) and its notebook builder; no competition data."""
import json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); SOL = os.path.dirname(HERE); ROOT = os.path.dirname(os.path.dirname(SOL))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed'))
import arb_v72 as P  # noqa: E402

KEY = lambda s: s.lower() if isinstance(s, str) and s.strip() and s != 'CCO' else None
ON = dict(b1=True, b2=True)


def test_selftest_passes():
    P.v72_selftest()


def test_default_switch_is_on_and_off_is_identity():
    assert P.V72 == ON
    F, F0 = ['A', 'X', 'B'], ['A', 'B', 'X']
    ice, gl = {'A': 0.1, 'X': 0.9, 'B': 0.95}, {'A': 0.1, 'X': 0.9, 'B': 0.95}
    assert P.v72_row(F, F0, ['X'], ice, gl, KEY, dict(b1=False, b2=False)) == (F, 'none')


def test_b1_needs_both_models_and_x2_at_rank2():
    F, F0 = ['A', 'X', 'B'], ['A', 'B', 'X']
    ice, gl = {'A': 0.2, 'X': 0.5, 'B': 0.4}, {'A': 0.1, 'X': 0.6, 'B': 0.3}
    assert P.v72_row(F, F0, ['X'], ice, gl, KEY, ON) == (['X', 'A', 'B'], 'b1_swap')
    assert P.v72_row(F, F0, ['X'], ice, gl, KEY, dict(b1=True, b2=False)) == (['X', 'A', 'B'], 'b1_swap')
    assert P.v72_row(F, F0, ['X'], {'A': 0.2, 'X': 0.5}, {'A': 0.7, 'X': 0.6}, KEY, ON)[1] == 'none'
    assert P.v72_row(['A', 'B', 'X'], F0, ['X'], ice, gl, KEY, ON)[1] == 'none'           # x2 not at rank 2 (E2R2 did not run)


def test_b2_restores_the_row_before_e2r2():
    F, F0 = ['A', 'X', 'B', 'C'], ['A', 'B', 'X', 'C']
    ice, gl = {'A': 0.9, 'X': 0.1, 'B': 0.4}, {'A': 0.9, 'X': 0.1, 'B': 0.3}
    assert P.v72_row(F, F0, ['X'], ice, gl, KEY, ON) == (F0, 'b2_undo')
    assert P.v72_row(F, F0, ['X'], ice, gl, KEY, dict(b1=True, b2=False)) == (F, 'none')


def _scope(tmp, rows, rows0, eng, ice, gl, sw, e2=True):
    import pandas as pd
    pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(os.path.join(tmp, 'submission.csv'), index=False)
    b0 = pd.DataFrame(rows0, columns=['molecule_id', 'smiles']).to_csv(index=False).encode()
    return dict(V72=sw, _e2r2_base_bytes=b0, E2R2_STATS=dict(status='applied') if e2 else None, metric_identity=lambda s: s.lower(),
                ENG={k: dict(smiles=v) for k, v in eng.items()}, ICE_SCORES=ice, GL_SCORES=gl)


def _case(tmp, sw, e2=True):
    rows = [('m1', 'A;X;B'), ('m2', 'C;Y;D'), ('m3', 'E;F')]
    rows0 = [('m1', 'A;B;X'), ('m2', 'C;D;Y'), ('m3', 'E;F')]
    eng = {'m1': ['X'], 'm2': ['Y'], 'm3': ['E']}
    ice = {'m1': {'A': 0.1, 'X': 0.5}, 'm2': {'Y': 0.1, 'D': 0.5}}
    gl = {'m1': {'A': 0.1, 'X': 0.5}, 'm2': {'Y': 0.1, 'D': 0.5}}
    return _scope(tmp, rows, rows0, eng, ice, gl, sw, e2)


def test_main_switch_off_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        sc = _case(tmp, dict(b1=False, b2=False)); f = os.path.join(tmp, 'submission.csv'); before = open(f, 'rb').read()
        st = P._v72_main(sc, workdir=tmp)
        assert st['status'] == 'off' and open(f, 'rb').read() == before and os.listdir(tmp) == ['submission.csv']


def test_main_applies_b1_and_b2():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        st = P._v72_main(_case(tmp, ON), workdir=tmp)
        out = pd.read_csv(os.path.join(tmp, 'submission.csv'), dtype=str, keep_default_na=False)
        assert dict(zip(out.molecule_id, out.smiles)) == {'m1': 'X;A;B', 'm2': 'C;D;Y', 'm3': 'E;F'}
        assert (st['status'], st['b1_swaps'], st['b2_undos'], st['changed'], st['changed_top1'], st['fallback']) == ('applied', 1, 1, 2, 1, 0)


def test_main_off_without_e2r2_and_no_rewrite_without_change():
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, 'submission.csv')
        sc = _case(tmp, ON, e2=False); before = open(f, 'rb').read()
        assert P._v72_main(sc, workdir=tmp)['status'] == 'off (E2R2 not applied)' and open(f, 'rb').read() == before
        sc = _case(tmp, ON); sc['ICE_SCORES'] = {}; before = open(f, 'rb').read()
        st = P._v72_main(sc, workdir=tmp)
        assert st['status'] == 'applied' and st['changed'] == 0 and open(f, 'rb').read() == before


def test_main_per_molecule_fallback_and_file_level_restore():
    with tempfile.TemporaryDirectory() as tmp:
        sc = _case(tmp, ON)
        def ident(s):
            if s == 'Y':
                raise RuntimeError('boom')
            return s.lower()
        sc['metric_identity'] = ident
        st = P._v72_main(sc, workdir=tmp)
        assert st['fallback'] == 1 and st['b1_swaps'] == 1 and st['status'] == 'applied'
        f = os.path.join(tmp, 'submission.csv'); before = open(f, 'rb').read()
        sc = _case(tmp, ON); before = open(f, 'rb').read(); sc['ENG'] = None
        st = P._v72_main(sc, workdir=tmp)
        assert st['status'].startswith('FAILED') and open(f, 'rb').read() == before


def test_selftest_failure_switches_off(monkeypatch=None):
    orig = P.v72_selftest
    with tempfile.TemporaryDirectory() as tmp:
        sc = _case(tmp, ON); f = os.path.join(tmp, 'submission.csv'); before = open(f, 'rb').read()
        try:
            P.v72_selftest = lambda: (_ for _ in ()).throw(AssertionError('x'))
            st = P._v72_main(sc, workdir=tmp)
        finally:
            P.v72_selftest = orig
        assert st['status'] == 'off (self-test)' and open(f, 'rb').read() == before


def test_builder_keeps_every_stack1_cell():
    b = os.path.join(ROOT, 'scratch', 'arb', 'build_v72.py')
    if not os.path.exists(b) or not os.path.exists(os.path.join(ROOT, 'scratch', 'nb1016', 'omer', 'casmi26-om-stack1.ipynb')):
        return  # scratch/ is not in git: the builder check runs where the base notebook was pulled
    sys.path.insert(0, os.path.dirname(b))
    import build_v72 as B
    nb0, nb = B.build_nb()
    assert len(nb['cells']) == len(nb0['cells']) + 1 and nb['cells'][:-1] == nb0['cells']
    src = ''.join(nb['cells'][-1]['source'])
    assert src == open(B.CELL, encoding='utf-8').read() and "if __name__ == '__main__':" in src
    off = ''.join(B.build_nb("V72 = dict(b1=False, b2=False)\n")[1]['cells'][-1]['source'])
    assert off.count('V72 = dict(b1=False, b2=False)') == 1
