"""Tests of prototypes/isomer_rank.py (no competition data): default knobs equal the z-sum rule, slots are kept, knobs act as documented.

usage: python solutions/fusion-v4n/prototypes/test_isomer_rank.py   (or pytest)
If $CASMI_DATA holds the casmi26-glacier dataset, the default is also compared with the real gl_fuse.rerank_multi.
"""
import copy, glob, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isomer_rank import rerank_conf, adduct_weights, RECOMMENDED


def ref_zsum(smis, scores, formulas, score_dicts, lams, top_n=60, min_covered=2, tie_eps=1e-9):
    """Independent re-statement of the rule in the gl_fuse.rerank_multi docstring (written for this test)."""
    def zcol(v):
        xs = [x for x in v if x is not None]
        if len(xs) < 2:
            return [0.0] * len(v)
        mu = sum(xs) / len(xs); sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))
        return [0.0 if (x is None or not sd > 0) else (x - mu) / sd for x in v]
    order = list(range(len(smis))); m = min(top_n, len(smis)); groups = {}
    for i in range(m):
        groups.setdefault(formulas[i], []).append(i)
    for idx in groups.values():
        cols = [([d.get(smis[i]) for i in idx], l) for d, l in zip(score_dicts, lams) if d]
        if len(idx) < 2 or not any(sum(x is not None for x in v) >= max(2, min_covered) for v, _ in cols):
            continue
        fused = zcol([scores[i] for i in idx]); n_act = 0
        for v, l in cols:
            if sum(x is not None for x in v) >= 2:
                fused = [a + l * b for a, b in zip(fused, zcol(v))]; n_act += 1
        if n_act >= 2:
            fused = [round(x / tie_eps) for x in fused]
        new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
        for slot, src in zip(idx, new):
            order[slot] = src
    return order


def random_case(rng, n_scorers=2):
    n = rng.randint(0, 70)
    smis = [f's{i}' for i in range(n)]; keys = [f'k{i}' for i in range(n)]
    scores = sorted((rng.gauss(0, 2) for _ in range(n)), reverse=True)
    forms = [rng.choice(['C10H12O2', 'C9H8O4', 'C15H10O5', 'C10H12O2|gen']) for _ in range(n)]
    dicts = []
    for _ in range(n_scorers):
        p = rng.choice([0.0, 0.1, 0.5, 1.0])
        d = {s: (None if rng.random() < p else round(rng.random(), rng.choice([1, 6]))) for s in smis if rng.random() < 0.9}
        dicts.append(d if rng.random() < 0.9 else {})
    return smis, keys, scores, forms, dicts


def test_default_is_the_zsum_rule():
    rng = random.Random(0)
    for _ in range(600):
        k = rng.choice([1, 2, 3])
        smis, keys, scores, forms, dicts = random_case(rng, k)
        lams = [rng.choice([0.0, 0.5, 1.0, 2.0]) for _ in range(k)]; top_n = rng.choice([25, 40, 60])
        assert rerank_conf(smis, keys, scores, forms, dicts, lams, top_n=top_n) == ref_zsum(smis, scores, forms, dicts, lams, top_n)


def test_default_equals_real_gl_fuse_if_present():
    hits = glob.glob(os.path.join(os.environ.get('CASMI_DATA', ''), 'input', '**', 'gl_fuse.py'), recursive=True) if os.environ.get('CASMI_DATA') else []
    if not hits:
        print('  (gl_fuse.py not found: skipped)'); return
    sys.path.append(os.path.dirname(sorted(hits, key=len)[0]))
    import gl_fuse
    rng = random.Random(1)
    for _ in range(600):
        k = rng.choice([1, 2, 3])
        smis, keys, scores, forms, dicts = random_case(rng, k)
        lams = [rng.choice([0.0, 0.5, 1.0, 1.5]) for _ in range(k)]
        assert rerank_conf(smis, keys, scores, forms, dicts, lams, top_n=60) == gl_fuse.rerank_multi(smis, keys, scores, forms, dicts, lams, top_n=60)


def test_permutation_slots_and_no_mutation():
    rng = random.Random(2)
    for _ in range(300):
        smis, keys, scores, forms, dicts = random_case(rng, 2)
        before = copy.deepcopy((smis, keys, scores, forms, dicts))
        knobs = dict(w_disagree=rng.choice([0.0, 0.5, 1.0]), margin_tau=rng.choice([None, 1.0]), agree_bonus=rng.choice([0.0, 1.0]),
                     sd_floor=rng.choice([0.0, 0.05]), prior_w=rng.choice([1.0, 1.5]))
        top_n = rng.choice([10, 60])
        o = rerank_conf(smis, keys, scores, forms, dicts, [1.0, 1.0], top_n=top_n, **knobs)
        assert sorted(o) == list(range(len(smis)))
        assert all(forms[o[i]] == forms[i] for i in range(len(smis)))            # a slot keeps its formula group
        assert o[top_n:] == list(range(top_n, len(smis)))                        # beyond top_n nothing moves
        assert (smis, keys, scores, forms, dicts) == before


def _group(n=6):
    smis = [f's{i}' for i in range(n)]
    return smis, [f'k{i}' for i in range(n)], [float(n - i) for i in range(n)], ['F'] * n


def test_confidence_weight():
    smis, keys, scores, forms = _group()
    ice = dict(zip(smis, [0.1, 0.2, 0.9, 0.2, 0.1, 0.2])); gl_same = dict(zip(smis, [0.2, 0.1, 0.8, 0.1, 0.2, 0.1]))
    gl_other = dict(zip(smis, [0.2, 0.1, 0.1, 0.1, 0.9, 0.1]))
    full = rerank_conf(smis, keys, scores, forms, [ice, gl_same], [1, 1])
    assert full[0] == 2                                                           # both models pick s2: it goes first
    assert rerank_conf(smis, keys, scores, forms, [ice, gl_same], [1, 1], w_disagree=0.0) == full   # agreement: untouched
    assert rerank_conf(smis, keys, scores, forms, [ice, gl_other], [1, 1], w_disagree=0.0) == list(range(6))   # disagreement: prior order
    half = rerank_conf(smis, keys, scores, forms, [ice, gl_other], [1, 1], w_disagree=0.5)
    assert half == ref_zsum(smis, scores, forms, [ice, gl_other], [0.5, 0.5])     # = the rule at half weight
    # a large joint margin counts as confident when margin_tau is given
    assert rerank_conf(smis, keys, scores, forms, [ice, gl_other], [1, 1], w_disagree=0.0, margin_tau=0.01) == \
        rerank_conf(smis, keys, scores, forms, [ice, gl_other], [1, 1])
    # weight-0 re-scorers (the library gate) never vote: both lams 0 -> order by the prior, as rerank_multi does
    assert rerank_conf(smis, keys, scores, forms, [ice, gl_other], [0.0, 0.0], w_disagree=0.5, agree_bonus=3.0) == list(range(6))
    # one contributing re-scorer: no agreement to check, weight unchanged
    assert rerank_conf(smis, keys, scores, forms, [ice, {}], [1, 1], w_disagree=0.0) == rerank_conf(smis, keys, scores, forms, [ice], [1])


def test_agree_bonus_and_sd_floor():
    smis, keys, scores, forms = _group()
    ice = dict(zip(smis, [0.50, 0.50, 0.50, 0.50, 0.50, 0.52])); gl = dict(zip(smis, [0.40, 0.40, 0.40, 0.40, 0.40, 0.43]))
    o = rerank_conf(smis, keys, scores, forms, [ice, gl], [1, 1])
    assert o[0] == 5                                   # a 0.02 difference, z-scored, overturns the prior
    assert rerank_conf(smis, keys, scores, forms, [ice, gl], [1, 1], sd_floor=0.05)[0] == 0      # with a floor it does not
    weak_i = dict(zip(smis, [0.30, 0.20, 0.25, 0.10, 0.20, 0.31])); weak_g = dict(zip(smis, [0.30, 0.25, 0.20, 0.10, 0.20, 0.31]))
    assert rerank_conf(smis, keys, scores, forms, [weak_i, weak_g], [1, 1])[0] == 0
    assert rerank_conf(smis, keys, scores, forms, [weak_i, weak_g], [1, 1], agree_bonus=5.0)[0] == 5


def test_per_spectrum_scores():
    rng = random.Random(3)
    smis, keys, scores, forms = _group(8)
    spec = {s: [rng.random(), rng.random(), None if i == 3 else rng.random()] for i, s in enumerate(smis)}
    mean = {s: sum(x for x in v if x is not None) / sum(x is not None for x in v) for s, v in spec.items()}
    gl = {s: rng.random() for s in smis}
    assert rerank_conf(smis, keys, scores, forms, [spec, gl], [1, 1]) == rerank_conf(smis, keys, scores, forms, [mean, gl], [1, 1])
    adducts = ['[M+H]+', '[M+Na]+', '[M+H]+']
    w = adduct_weights(adducts)
    assert w == [1.0, 0.0, 1.0]
    mh = {s: sum(x for x, a in zip(v, adducts) if x is not None and a == '[M+H]+') / sum(x is not None and a == '[M+H]+' for x, a in zip(v, adducts))
          for s, v in spec.items()}
    assert rerank_conf(smis, keys, scores, forms, [spec, gl], [1, 1], spec_weights=[w, None]) == rerank_conf(smis, keys, scores, forms, [mh, gl], [1, 1])
    # a molecule with [M+Na]+ spectra only: the re-scorer drops out, the prior order stays
    assert adduct_weights(['[M+Na]+', '[M+Na]+']) == [0.0, 0.0]
    assert rerank_conf(smis, keys, scores, forms, [spec], [1], spec_weights=[[0.0, 0.0, 0.0]]) == list(range(8))
    # zmean: a spectrum with a 10x larger spread must not dominate
    big = {s: [v[0], 10 * v[1]] for s, v in spec.items()}; small = {s: [v[0], v[1]] for s, v in spec.items()}
    assert rerank_conf(smis, keys, scores, forms, [big], [1], spec_agg='zmean') == rerank_conf(smis, keys, scores, forms, [small], [1], spec_agg='zmean')
    try:
        rerank_conf(smis, keys, scores, forms, [spec], [1], spec_agg='nope'); raise AssertionError
    except ValueError:
        pass


def test_recommended_knobs():
    smis, keys, scores, forms = _group()
    ice = dict(zip(smis, [0.1, 0.2, 0.9, 0.2, 0.1, 0.2])); gl_same = dict(zip(smis, [0.2, 0.1, 0.8, 0.1, 0.2, 0.1]))
    gl_other = dict(zip(smis, [0.2, 0.1, 0.1, 0.1, 0.9, 0.1]))
    assert RECOMMENDED == dict(w_disagree=0.5, agree_bonus=1.0)
    assert rerank_conf(smis, keys, scores, forms, [ice, gl_same], [1, 1], **RECOMMENDED)[0] == 2          # agreed pick first
    assert rerank_conf(smis, keys, scores, forms, [ice, gl_other], [1, 1], **RECOMMENDED) ==         ref_zsum(smis, scores, forms, [ice, gl_other], [0.5, 0.5])                                         # no agreement: half weight, no bonus
    assert rerank_conf(smis, keys, scores, forms, [ice, gl_same], [0.0, 0.0], **RECOMMENDED) == list(range(6))   # library gate (weights 0): off


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
