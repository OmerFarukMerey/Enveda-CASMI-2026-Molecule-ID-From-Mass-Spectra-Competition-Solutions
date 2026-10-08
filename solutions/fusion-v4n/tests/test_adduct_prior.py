"""fusion_core ADD_JOIN_MU and nb/embed/adduct_prior.py: the adduct-evidence term of stage A (reviews/newinfo-2026-10-08.md).

Checks, on random lists with small stand-ins for the re-rank modules and for the LightGBM models (no competition data, no
model files): off (key absent, 0, no scorer, or a scorer that returns None) gives the reference submission bit for bit,
also with a PubChem list and an engine-2 list and at any POP_MU; with a scorer the flag equals the NP_JOIN_MU arithmetic on
the same values, so joined rows move only among the joined rows' slots; lists with fewer than two joined rows take the
standard path; a failing scorer falls back; the scorer module maps adduct sets to presence vectors, ranks a candidate
whose predicted adducts match the observed ones first, and returns None for a molecule without an adduct record.
usage: python tests/test_adduct_prior.py   (or pytest)
"""
import os, random, sys

import numpy as np

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adduct_prior
import fusion_core
from test_join_fwd_a import Fwd
from test_pop_group import case, extras, POP

CFG = dict(ICE_LAM=0.75, GL_LAM=0.75, POP_MU=0.25)


def vals(v, seed=1):
    r = random.Random(seed); return {s: r.gauss(0, 1) for s in v[0]}


def run(v, ice, gl, cfg, pc=None, eng=None, **kw):
    sub, st = fusion_core.build_submission({'m': v}, {'m': pc} if pc else {}, {'m': eng} if eng else {}, {'m': ice}, {'m': gl}, dict(CFG, **cfg), ['m'], ['m'],
                                           ice_fuse=Fwd, gl_fuse=Fwd, formula=lambda s: 'F', score_key=lambda s: s, log=lambda *a: None, pool_pop=POP, **kw)
    return sub.smiles[0].split(';'), st


def test_off_is_the_reference_bit_for_bit():
    rng = random.Random(0); differ = 0
    for _ in range(400):
        v, ice, gl = case(rng, p_gen=0.4); pc, eng = extras(rng, v); d = vals(v); sc = lambda m, ss: [d[x] for x in ss]
        for kw in (dict(), dict(pc=pc), dict(pc=pc, eng=eng)):
            for mu in (0.0, 0.25, 0.35):
                ref = run(v, ice, gl, dict(POP_MU=mu), **kw)[0]
                assert run(v, ice, gl, dict(POP_MU=mu, ADD_JOIN_MU=0.0), add_score=sc, **kw)[0] == ref              # weight 0, scorer present
                assert run(v, ice, gl, dict(POP_MU=mu), add_score=sc, **kw)[0] == ref                               # key absent
                assert run(v, ice, gl, dict(POP_MU=mu, ADD_JOIN_MU=0.5), add_score=None, **kw)[0] == ref            # no scorer
                assert run(v, ice, gl, dict(POP_MU=mu, ADD_JOIN_MU=0.5), add_score=lambda m, ss: None, **kw)[0] == ref   # no adduct record
                differ += int(run(v, ice, gl, dict(POP_MU=mu, ADD_JOIN_MU=0.5), add_score=sc, **kw)[0] != ref)
    assert differ > 300, differ                                  # the term does something when it is set
    assert fusion_core.DEFAULTS['ADD_JOIN_MU'] == 0.0


def test_equals_the_joined_row_arithmetic_and_needs_two_joined_rows():
    rng = random.Random(1); n_two = n_less = 0
    for _ in range(400):
        v, ice, gl = case(rng, p_gen=0.4); d = vals(v); nj = sum(1 for p in v[5] if p == -2)
        on, st = run(v, ice, gl, dict(ADD_JOIN_MU=0.5), add_score=lambda m, ss: [d[x] for x in ss])
        if nj > 1:
            assert on == run(v, ice, gl, dict(NP_JOIN_MU=0.5), np_score=d)[0] and st['adduct']['molecules'] == 1; n_two += 1
        else:
            assert on == run(v, ice, gl, {})[0] and 'adduct' not in st; n_less += 1
    assert n_two > 100 and n_less > 20, (n_two, n_less)


def test_joined_rows_stay_in_their_slots():
    rng = random.Random(2)
    for _ in range(300):
        v, _, _ = case(rng, p_gen=0.5); s = [rng.gauss(0, 1) for _ in v[0]]
        ref = fusion_core.popularity_reorder(v[0], v[1], v[3], v[4], v[5], POP, 0.25)
        a = fusion_core.np_reorder(v[0], v[1], v[3], v[4], v[5], POP, 0.25, s, 0.0, 0.5)          # what the flag calls
        assert [x for x, p in zip(a[0], a[4]) if p != -2] == [x for x, p in zip(ref[0], ref[4]) if p != -2]
        assert [i for i, p in enumerate(a[4]) if p == -2] == [i for i, p in enumerate(v[5]) if p == -2]


def test_failing_scorer_falls_back():
    rng = random.Random(3)
    for _ in range(100):
        v, ice, gl = case(rng, p_gen=0.4)

        def boom(m, ss):
            raise RuntimeError('no model')
        assert run(v, ice, gl, dict(ADD_JOIN_MU=0.5), add_score=boom)[0] == run(v, ice, gl, {})[0]
        nan = run(v, ice, gl, dict(ADD_JOIN_MU=0.5), add_score=lambda m, ss: [None] * len(ss))[0]
        assert nan == run(v, ice, gl, dict(ADD_JOIN_MU=0.5), add_score=lambda m, ss: [0.0] * len(ss))[0]


class Stub:
    """Stands in for a LightGBM booster: probability from one feature column (the aliphatic-OH / acid counts are enough here)."""
    def __init__(self, col, hi=0.9, lo=0.1):
        self.col, self.hi, self.lo = col, hi, lo

    def predict(self, X):
        return np.where(np.nan_to_num(X[:, self.col]) > 0, self.hi, self.lo)


def test_scorer_module():
    assert list(adduct_prior.observed(['[M+H]+', '[M+Na]+'])) == [1, 0, 1, 0, 0, 0, 0]
    assert list(adduct_prior.observed(['[M-H]-', '[M+CH2O2-H]-', '[2M+Na]+'])) == [0, 1, 1, 0, 1, 0, 0]
    P = np.array([[0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5], [0.1, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5]])
    ll = adduct_prior.log_likelihood(P, adduct_prior.observed(['[M+H]+']))
    assert ll[0] > ll[1] and abs(float(adduct_prior.standardise(ll).mean())) < 1e-9
    assert list(adduct_prior.standardise([1.0, 1.0, 1.0])) == [0.0, 0.0, 0.0]                   # a constant list carries nothing
    X = adduct_prior.featurize(['CC(=O)O', 'CCN', 'not a smiles', 'OCC1OC(O)C(O)C(O)C1O'])
    assert X.shape == (4, adduct_prior.N_DESC + adduct_prior.N_FP) and np.isnan(X[2, 0]) and X[2, adduct_prior.N_DESC:].sum() == 0
    names = list(adduct_prior.SMARTS)
    assert X[0, names.index('cooh')] == 1 and X[1, names.index('amine1')] == 1 and X[3, names.index('oh_ali')] == 5 and X[3, -1 + adduct_prior.N_DESC] == 1
    assert X[0, adduct_prior.N_DESC:].sum() > 0
    # an acid and an amine, query seen in negative mode only: the acid must come first; in positive mode only: the amine
    models = {t: Stub(names.index('amine1'), 0.5, 0.5) for t in adduct_prior.TARGETS}
    models['mH'] = Stub(names.index('cooh')); models['H'] = Stub(names.index('amine1'))
    sc = adduct_prior.Scorer(models, {'neg': ['[M-H]-'], 'pos': ['[M+H]+']})
    a = sc('neg', ['CC(=O)O', 'CCN', 'CCCC(=O)O']); b = sc('pos', ['CC(=O)O', 'CCN', 'CCCC(=O)O'])
    assert a[0] > a[1] and a[2] > a[1] and b[1] > b[0] and len(a) == 3
    assert sc('unknown', ['CC(=O)O', 'CCN']) is None and sc('neg', ['CC(=O)O']) is None
    assert len(sc.cache) == 3 and sc.n == 2                                                    # predictions are cached per structure


if __name__ == '__main__':
    test_off_is_the_reference_bit_for_bit(); test_equals_the_joined_row_arithmetic_and_needs_two_joined_rows(); test_joined_rows_stay_in_their_slots()
    test_failing_scorer_falls_back(); test_scorer_module()
    print('adduct-evidence term OK: off == reference bit for bit, joined rows only, failures fall back, scorer module consistent')
