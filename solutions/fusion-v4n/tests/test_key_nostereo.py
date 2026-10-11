"""KEY_NOSTEREO: final-list de-duplication by the metric key of 2026-10-07 (stereo stripped before the tautomer step); default off.

Checks without competition data: the flag is off in the notebook config and in fusion_core; with the flag off build_submission returns the
same frame whether or not a key function is passed (random cases, with and without engine 2); with the flag on a row that shares the
stereo-free key with an earlier row is dropped before the cut to 25 and the next fused row takes the slot; make_nostereo_key reproduces
the metric notebook's own regression pair (needs RDKit; skipped without it) and leaves SMILES without stereo marks to score_key.
usage: python tests/test_key_nostereo.py   (or pytest)
"""
import os, random, sys

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed'))
import fusion_core

KEY = lambda s: s
NS = lambda s: s.replace('@', '')          # toy stereo-free key: 'A@' and 'A' are one structure


def load_cfg():
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    return ns['CFG']


def case(rnd, n_mol=6, with_eng=True):
    base, eng, order = {}, {}, []
    for i in range(n_mol):
        mid = f'm{i}'; order.append(mid)
        names = [f'x{i}_{j}' + ('@' if rnd.random() < 0.3 else '') for j in range(rnd.randint(3, 45))]
        if rnd.random() < 0.7 and len(names) > 4:
            names.insert(rnd.randint(1, len(names) - 1), names[0].rstrip('@') + ('' if names[0].endswith('@') else '@'))   # a stereo twin of the first row
        sc = sorted((rnd.random() for _ in names), reverse=True)
        base[mid] = [names, list(names), rnd.random(), sc, ['C6H6'] * len(names)]
        if with_eng:
            e = [f'e{i}_{j}' for j in range(rnd.randint(0, 40))] + rnd.sample(names, min(5, len(names)))
            rnd.shuffle(e); eng[mid] = dict(smiles=e, keys=list(e))
    return base, eng, order


def run(base, eng, order, cfg, **kw):
    sub, stats = fusion_core.build_submission(base, {}, eng, {}, {}, cfg, order, order, score_key=KEY, formula=lambda s: 'C6H6', log=lambda *a: None, **kw)
    return sub, stats


def test_default_off():
    assert load_cfg()['KEY_NOSTEREO'] is False and fusion_core.DEFAULTS['KEY_NOSTEREO'] is False


def test_flag_off_is_identical():
    rnd = random.Random(0)
    for t in range(40):
        base, eng, order = case(rnd, with_eng=t % 4 != 0)
        for extra in ({}, dict(FILL_25=True)):
            cfg = dict(fusion_core.DEFAULTS, **extra)
            a, sa = run(base, eng, order, cfg)
            b, sb = run(base, eng, order, cfg, nostereo_key=NS)                       # key passed, flag off
            assert a.equals(b) and sa == sb and 'nostereo' not in sa


def test_flag_on_drops_twins_before_the_cut():
    rnd = random.Random(1); n_removed = 0
    for t in range(40):
        base, eng, order = case(rnd, with_eng=t % 4 != 0)
        cfg = dict(fusion_core.DEFAULTS, KEY_NOSTEREO=True)
        off, _ = run(base, eng, order, dict(fusion_core.DEFAULTS)); on, st = run(base, eng, order, cfg, nostereo_key=NS)
        for (m0, s0), (m1, s1) in zip(off.values, on.values):
            l0, l1 = s0.split(';'), s1.split(';')
            assert m0 == m1 and len(l1) <= 25 and len({NS(x) for x in l1}) == len(l1), 'no two rows of one stereo-free key'
            assert l1[:len(fusion_core.dedup_by(l0, NS))] == fusion_core.dedup_by(l0, NS), 'the survivors keep their order; freed slots are filled from below'
            assert l1[0] == l0[0]
        n_removed += st.get('nostereo', {}).get('removed_top25', 0)
        on2, _ = run(base, eng, order, cfg)                                           # flag on without a key function: nothing happens
        assert on2.equals(off)
    assert n_removed > 0, 'the cases must exercise the de-duplication'


def test_refill_from_the_fused_tail():
    v = [f'v{j}' for j in range(30)]; v[3] = 'v0@'
    base = {'m': [v, list(v), 0.0, [1.0 - j / 100 for j in range(30)], ['C6H6'] * 30]}; eng = {'m': dict(smiles=list(v), keys=list(v))}
    off, _ = run(base, eng, ['m'], dict(fusion_core.DEFAULTS)); on, st = run(base, eng, ['m'], dict(fusion_core.DEFAULTS, KEY_NOSTEREO=True), nostereo_key=NS)
    l0, l1 = off.smiles[0].split(';'), on.smiles[0].split(';')
    assert 'v0@' in l0 and 'v0@' not in l1 and len(l0) == 25 and len(l1) == 25
    assert set(l1) - set(l0) == {'v25'} and st['nostereo'] == dict(molecules=1, removed=1, removed_top25=1)


def test_metric_regression_pair():
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        print('RDKit not installed: skipped'); return
    te = rdMolStandardize.TautomerEnumerator()

    def old(s):
        return Chem.MolToInchiKey(te.Canonicalize(Chem.MolFromSmiles(s)))[:14]
    calls = []
    key = fusion_core.make_nostereo_key(lambda s: (calls.append(s), old(s))[1])
    a, b = 'C=C(C)C1CCC2CCC(=O)C=C2C1', 'C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1'                # the pair in the metric notebook's doctest
    assert key(a) == key(b) == old(a), 'stereo twin lands on the stereo-free key'
    assert calls == [a], 'a SMILES without stereo marks goes to score_key, one with marks does not'
    assert key('not a molecule @') is None and key('C[C@H](N)C(=O)O') == old('CC(N)C(=O)O')


if __name__ == '__main__':
    for n, f in sorted(globals().items()):
        if n.startswith('test_') and callable(f):
            f(); print('ok', n)
