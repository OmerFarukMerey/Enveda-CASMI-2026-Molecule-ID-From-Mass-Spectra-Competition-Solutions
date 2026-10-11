"""FUSE_KEY_VOTE: the engine fusion keyed by the stereo-free metric key, one vote per key and engine; default off.

Checks without competition data: the flag is off in the notebook config and in fusion_core; with the flag off build_submission returns the
same frame whether or not a key function is passed; with the flag on and no stereo twin anywhere the result equals the flag-off result
(both modes); a twin inside one list votes once, at its first rank ('best') or at its rank among the distinct keys ('compact'); the two
engines' votes for one metric key are summed although their strings differ; an engine row without a stereo mark keeps its delivered key;
the fused list holds every metric key once.
usage: python tests/test_fuse_key_vote.py   (or pytest)
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


def case(rnd, n_mol=6, twins=True):
    base, eng, order = {}, {}, []
    for i in range(n_mol):
        mid = f'm{i}'; order.append(mid)
        names = [f'x{i}_{j}' + ('@' if (twins and rnd.random() < 0.3) else '') for j in range(rnd.randint(3, 45))]
        if twins and rnd.random() < 0.7 and len(names) > 4:
            names.insert(rnd.randint(1, len(names) - 1), names[0].rstrip('@') + ('' if names[0].endswith('@') else '@'))
        sc = sorted((rnd.random() for _ in names), reverse=True)
        base[mid] = [names, list(names), rnd.random(), sc, ['C6H6'] * len(names)]
        e = [f'e{i}_{j}' for j in range(rnd.randint(0, 40))] + rnd.sample(names, min(5, len(names)))
        if twins and rnd.random() < 0.5:
            e.append(names[1].rstrip('@') + ('' if names[1].endswith('@') else '@'))     # engine 2 names row 2 with the other string
        rnd.shuffle(e); eng[mid] = dict(smiles=e, keys=list(e))
    return base, eng, order


def run(base, eng, order, cfg, **kw):
    return fusion_core.build_submission(base, {}, eng, {}, {}, cfg, order, order, score_key=KEY, formula=lambda s: 'C6H6', log=lambda *a: None, **kw)


def test_default_off():
    assert load_cfg()['FUSE_KEY_VOTE'] is None and fusion_core.DEFAULTS['FUSE_KEY_VOTE'] is None


def test_flag_off_is_identical():
    rnd = random.Random(0)
    for t in range(40):
        base, eng, order = case(rnd)
        a, sa = run(base, eng, order, dict(fusion_core.DEFAULTS)); b, sb = run(base, eng, order, dict(fusion_core.DEFAULTS), nostereo_key=NS)
        assert a.equals(b) and sa == sb


def test_no_twins_no_change():
    rnd = random.Random(1)
    for t in range(40):
        base, eng, order = case(rnd, twins=False)
        off, _ = run(base, eng, order, dict(fusion_core.DEFAULTS))
        for mode in ('best', 'compact'):
            on, _ = run(base, eng, order, dict(fusion_core.DEFAULTS, FUSE_KEY_VOTE=mode), nostereo_key=NS)
            assert on.equals(off), mode


def test_votes():
    a, k = 0.6, 3.0
    v = ['A@', 'A', 'B']; e = ['C', 'A']; ek = ['C', 'A']
    s0, k0, c0 = fusion_core.fuse_rrf(v, e, ek, KEY, a, k, 40)                        # off: 'A@' and 'A' are two candidates
    assert dict(zip(k0, c0)) == {'A@': 1 / 4, 'A': 1 / 5 + a / 5, 'B': 1 / 6, 'C': a / 4} and s0[0] == 'A'
    s1, k1, c1 = fusion_core.fuse_rrf(v, e, ek, KEY, a, k, 40, None, 0.0, NS, False)  # best: one vote per engine, first rank
    assert dict(zip(k1, c1)) == {'A': 1 / 4 + a / 5, 'B': 1 / 6, 'C': a / 4} and s1 == ['A@', 'B', 'C']
    s2, k2, c2 = fusion_core.fuse_rrf(v, e, ek, KEY, a, k, 40, None, 0.0, NS, True)   # compact: the skipped twin frees its rank
    assert dict(zip(k2, c2)) == {'A': 1 / 4 + a / 5, 'B': 1 / 5, 'C': a / 4} and s2 == ['A@', 'B', 'C']
    # an engine row without a stereo mark keeps its delivered key; one with a mark gets the key function's
    s3, k3, c3 = fusion_core.fuse_rrf(['A'], ['x', 'A@'], ['K_x', 'ignored'], KEY, a, k, 40, None, 0.0, NS, False)
    assert dict(zip(k3, c3)) == {'A': 1 / 4 + a / 5, 'K_x': a / 4}
    # the credit of a joined row is kept
    s4, k4, c4 = fusion_core.fuse_rrf(['A@', 'B'], ['B'], ['B'], KEY, a, k, 40, {'A@'}, 1.0, NS, False)
    assert dict(zip(k4, c4)) == {'A': (1 + a) / 4, 'B': 1 / 5 + a / 4}


def test_fused_list_has_every_key_once():
    rnd = random.Random(2); changed = 0
    for t in range(40):
        base, eng, order = case(rnd)
        off, _ = run(base, eng, order, dict(fusion_core.DEFAULTS))
        for mode in ('best', 'compact'):
            on, _ = run(base, eng, order, dict(fusion_core.DEFAULTS, FUSE_KEY_VOTE=mode), nostereo_key=NS)
            for s in on.smiles:
                l = s.split(';'); assert len(l) <= 25 and len({NS(x) for x in l}) == len(l)
            changed += int(not on.equals(off))
    assert changed > 0, 'the cases must exercise the flag'


def test_default_key_function():
    base, eng, order = case(random.Random(3))
    try:
        run(base, eng, order, dict(fusion_core.DEFAULTS, FUSE_KEY_VOTE='other'), nostereo_key=NS)
    except AssertionError:
        pass
    else:
        raise AssertionError('an unknown mode must be refused')
    src = open(os.path.join(SOL, 'nb', '08_submit.py'), encoding='utf-8').read()
    assert "if (CFG['KEY_NOSTEREO'] or CFG['FUSE_KEY_VOTE']) else None" in src, 'cell 08 hands the key function to the fusion when the flag is set'


if __name__ == '__main__':
    for n, f in sorted(globals().items()):
        if n.startswith('test_') and callable(f):
            f(); print('ok', n)
