"""The merged-adduct tie rule (ADDUCT_TIE, nb/embed/determ.py) and HASHSEED: pure helpers, a stub engine under real hash seeds, cell nb/06.

No competition data and no public package. A stub module holds the public engine's expression `max(set(adducts), key=adducts.count)`
and is run in fresh interpreters with different PYTHONHASHSEED values. What is pinned down:
  * the public expression depends on the hash seed for a tied molecule (the reason the flag exists) and not for an untied one;
  * with a rule installed the tied molecule's label is the same under every seed and equals the rule's definition;
  * a molecule without a tie gets exactly the label of the unpatched engine under every rule and seed (the patch hands the call to the
    builtin), and every other `max` call of the module is untouched;
  * 'mean': the wrapped bank averages the merged item's logits over the tied adducts and passes every other call through;
  * install refuses a module that does not hold the expression; rule None restores the module;
  * cell nb/06 with ADDUCT_TIE / HASHSEED set gives the v24a base lists bit for bit on stub molecules without a tie (GOLDEN of
    tests/test_fs_pass.py), and the preset v36a differs from v33b in these two keys only.
usage: python tests/test_determ.py   (or pytest)
"""
import importlib.util, json, os, subprocess, sys, tempfile, types

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import determ

STUB = '''
def label(spectra):
    out = []
    for md in (1, -1):
        grp = [s for s in spectra if s['mode'] == md]
        if not grp:
            continue
        adducts = [s['adduct'] for s in grp]
        ad = max(set(adducts), key=adducts.count)
        out.append([ad, max(1, len(grp)), max(len(a) for a in adducts), max(set(adducts))])
    return out
'''
TRAIN = {'[M+H]+': 100, '[M+Na]+': 50, '[M+NH4]+': 20, '[M-H]-': 80, '[M+Cl]-': 10, '[M+CH2O2-H]-': 30}
SP = lambda *ad: [dict(adduct=a, mode=1 if a.endswith('+') else -1) for a in ad]
TIED = SP('[M+Na]+', '[M+NH4]+', '[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+H]+', '[M+Cl]-', '[M-H]-')       # three-way and two-way tie
UNTIED = SP('[M+Na]+', '[M+H]+', '[M+H]+', '[M-H]-', '[M+Cl]-', '[M-H]-')
ONE = SP('[M+H]+')


def merged(adducts):
    """== fe_v4/v1engine.py::merged_adduct with a small training-count table."""
    cnt = {}
    for a in adducts:
        cnt[a] = cnt.get(a, 0) + 1
    best = max(cnt.values())
    return min((a for a in cnt if cnt[a] == best), key=lambda a: (-TRAIN.get(a, 0), a))


def stub_module(src=STUB, name='stub_engine'):
    d = tempfile.mkdtemp(); p = os.path.join(d, name + '.py'); open(p, 'w').write(src)
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, p


CHILD = '''
import importlib.util, json, sys
sys.path.insert(0, {embed!r}); sys.path.insert(0, {tests!r})
import determ, test_determ as T
spec = importlib.util.spec_from_file_location('stub_engine', {path!r}); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
out = {{}}
for rule in (None, 'train', 'first', 'mean'):
    determ.install(m, None, rule, merged=T.merged)
    out[str(rule)] = [m.label(T.TIED), m.label(T.UNTIED), m.label(T.ONE)]
print(json.dumps(out))
'''


def under_seeds(seeds=range(12)):
    _, path = stub_module()
    code = CHILD.format(embed=os.path.join(SOL, 'nb', 'embed'), tests=os.path.dirname(os.path.abspath(__file__)), path=path)
    res = {}
    for s in seeds:
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED=str(s)))
        assert r.returncode == 0, r.stderr[-2000:]
        res[s] = json.loads(r.stdout)
    return res


def test_helpers():
    assert determ.tied(['a', 'b', 'a', 'b', 'c']) == ['a', 'b'] and determ.tied(['x']) == ['x'] and determ.tied([]) == []
    ad = [s['adduct'] for s in TIED if s['mode'] == 1]
    assert determ.pick(ad, 'first') == '[M+Na]+' and determ.pick(ad, 'mean') == '[M+Na]+' and determ.pick(ad, 'train', merged) == '[M+H]+'
    assert determ.pick(ad, lambda a, t: t[2]) == '[M+H]+' and determ.pick(['[M+H]+', '[M+Na]+', '[M+H]+'], 'first') == '[M+H]+'
    assert determ.groups(TIED) == [['[M+Na]+', '[M+NH4]+', '[M+H]+'], ['[M+Cl]-', '[M-H]-']] and determ.groups(UNTIED) == [['[M+H]+'], ['[M-H]-']]
    assert determ.groups(TIED, max_query=3) == [['[M+Na]+', '[M+NH4]+', '[M+H]+']]                     # only the engine's first spectra count
    for bad in ('nope', 3):
        try:
            determ.pick(ad, bad); raise SystemExit('an unknown rule must raise')
        except (ValueError, TypeError):
            pass
    f = determ.hash_facts(); assert set(f) == {'PYTHONHASHSEED', 'hash_randomization', 'probe'} and f['probe'] in ('[M+H]+', '[M+Na]+')


def test_set_order_depends_on_the_seed_and_the_rules_do_not():
    res = under_seeds()
    pub = {json.dumps(r['None'][0]) for r in res.values()}
    assert len(pub) > 1, 'the public expression gave one label under 12 hash seeds: the test molecule is not tied?'
    for rule, want in (('train', ['[M+H]+', '[M-H]-']), ('first', ['[M+Na]+', '[M+Cl]-']), ('mean', ['[M+Na]+', '[M+Cl]-'])):
        got = {json.dumps(r[rule][0]) for r in res.values()}
        assert len(got) == 1, (rule, got)
        assert [g[0] for g in json.loads(got.pop())] == want, rule
    for r in res.values():                                   # untied molecules: the unpatched label under every rule and every seed
        for rule in ('train', 'first', 'mean'):
            assert r[rule][1] == r['None'][1] == res[0]['None'][1] and r[rule][2] == r['None'][2] == res[0]['None'][2], rule
        for rule in ('None', 'train', 'first', 'mean'):      # the other max calls of the module (ints, a set without key) are the builtin's
            assert [g[1:] for g in r[rule][0]] == [[6, 8, '[M+Na]+'], [2, 7, '[M-H]-']], (rule, r[rule][0])


def test_install_guards_and_restore():
    m, _ = stub_module()
    determ.install(m, None, 'first')
    assert m.label(TIED)[0][0] == '[M+Na]+' and determ.counts()['ties'] >= 2
    determ.install(m, None, None)
    assert 'max' not in m.__dict__ and determ._STATE['rule'] is None
    bad, _ = stub_module('def label(x):\n    return max(x)\n', 'no_expr')
    try:
        determ.install(bad, None, 'train', merged=merged); raise SystemExit('a module without the expression must be refused')
    except AssertionError as e:
        assert 'exactly once' in str(e)
    try:
        determ.install(m, None, 'nope'); raise SystemExit('an unknown rule must be refused')
    except AssertionError:
        pass
    determ.install(m, None, None)


def test_mean_averages_the_merged_item_over_the_tied_adducts():
    IX = {'[M+H]+': 0, '[M+Na]+': 1, '[M+NH4]+': 2, '[M-H]-': 3, '[M+Cl]-': 4}
    calls = []

    class Bank:
        def logits_el(self, items, bs=64):
            calls.append([it['adduct_ix'] for it in items])
            z = np.array([[10.0 * it['adduct_ix'] + it['x'], 1.0] for it in items], np.float32)
            return z, z[:, :1] * 2
    m, _ = stub_module()
    E = types.SimpleNamespace(cfg=types.SimpleNamespace(max_query_spectra=16), bank=Bank())

    def run(spectra, target):
        lab = m.label(spectra)                               # the engine builds the merged items, then asks the bank twice
        zs, _ = E.bank.logits_el([dict(adduct_ix=IX[s['adduct']], x=0.0) for s in spectra])
        zm, em = E.bank.logits_el([dict(adduct_ix=IX[l[0]], x=0.5) for l in lab])
        return zs, zm, em
    E.run = run
    plain = E.run(TIED, 0.0)
    determ.install(m, E, 'mean'); determ.wrap_bank(E.bank, IX.get); determ.wrap_bank(E.bank, IX.get)       # idempotent
    calls.clear(); zs, zm, em = E.run(TIED, 0.0)
    assert np.array_equal(zs, plain[0]) and len(calls) == 2 and calls[1] == [1, 2, 0, 4, 3]                 # per-spectrum call untouched; 3 + 2 evaluations
    assert np.allclose(zm[:, 0], [10.5, 35.5]) and np.allclose(em[:, 0], [21.0, 71.0]) and zm.shape == (2, 2)
    calls.clear(); zs2, zm2, _ = E.run(UNTIED, 0.0)                                                         # no tie: one evaluation per item
    assert calls[1] == [0, 3] and np.allclose(zm2[:, 0], [0.5, 30.5])
    determ.install(m, E, 'first'); calls.clear(); _, zm3, _ = E.run(TIED, 0.0)                              # another rule: the wrapper is inert
    assert calls[1] == [1, 4] and np.allclose(zm3[:, 0], [10.5, 40.5])
    determ.install(m, E, None)


def test_cell_06_is_unchanged_for_molecules_without_a_tie_and_preset():
    import test_fs_pass as TF
    ref = TF.run_cell(TF.cell(), TF.load_cfg())[0]
    assert TF.digest(ref) == TF.GOLDEN
    for rule in determ.RULES:
        base, fails, V, E, logs = TF.run_cell(TF.cell(), TF.load_cfg(ADDUCT_TIE=rule, HASHSEED=0))
        assert TF.digest(base) == TF.GOLDEN and not fails, rule
        assert any('ADDUCT_TIE' in l for l in logs)
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    assert ns['CFG']['ADDUCT_TIE'] is None and ns['CFG']['HASHSEED'] is None                                # default: the public behaviour
    a = json.load(open(os.path.join(SOL, 'configs', 'v33b-v24a-fsranker-gate07-rankavg.json'))); b = json.load(open(os.path.join(SOL, 'configs', 'v36a-v33b-determ.json')))
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {'VERSION', 'ADDUCT_TIE', 'HASHSEED'}
    assert b['ADDUCT_TIE'] in determ.RULES and isinstance(b['HASHSEED'], int) and set(b) <= set(ns['CFG'])


def test_cells_wire_the_flags():
    s1 = open(os.path.join(SOL, 'nb', '01_setup.py'), encoding='utf-8').read(); s5 = open(os.path.join(SOL, 'nb', '05_v1_engine.py'), encoding='utf-8').read()
    assert s1.index("os.environ['PYTHONHASHSEED']") < s1.index('subprocess.run('), 'HASHSEED must be exported before the first subprocess'
    assert 'determ.hash_facts()' in s1
    assert s5.index('determ.install(') > s5.index('V = v1engine.V1FE(') and s5.index('determ.install(') > s5.index('FS_BANK = FS_VIEWS = None')
    assert "if CFG['ADDUCT_TIE'] is not None:" in s5


if __name__ == '__main__':
    test_helpers(); test_set_order_depends_on_the_seed_and_the_rules_do_not(); test_install_guards_and_restore()
    test_mean_averages_the_merged_item_over_the_tied_adducts(); test_cell_06_is_unchanged_for_molecules_without_a_tie_and_preset(); test_cells_wire_the_flags()
    print('determ OK: set order depends on the hash seed, the rules do not; untied molecules and the default config are untouched; preset v36a = v33b + 2 keys')
