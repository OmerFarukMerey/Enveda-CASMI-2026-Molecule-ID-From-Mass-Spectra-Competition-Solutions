"""GEN_ANALOG: the analog generator channel (nb/embed/gen_analog.py, reviews/gen3-2026-10-10.md).

Checks, without competition data (a nine-structure pool and random logits):
  * flag off is today's notebook: the config default is None; in the cells every statement that touches the channel sits under
    `if CFG['GEN_ANALOG']` (the only other statement is the creation of two empty dicts), so with the flag off no line of the channel
    runs; removing those blocks leaves cells that still parse; when git still holds the cells without the channel, the stripped
    cells are compared with them statement for statement; the built notebook of the v59a preset and of a v59a + GEN_ANALOG build
    differ in the CFG override line only;
  * the placement is the identity without generated rows, with a closed gate and without a fitted model; 'slots' never moves the
    rows above the slot and keeps every other row in order; 'gate' never reorders database rows among themselves;
  * the generator returns only structures with a target formula that are neither pool rows nor excluded keys, at most ROW_CAP, keys at
    most KEY_CAP products, deterministically, and treats a removed answer as new (replays);
  * the rule table equals train/gen/rules.py.
usage: python tests/test_gen_analog.py   (or pytest)
"""
import ast, json, os, subprocess, sys, tempfile

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.join(SOL, 'train', 'gen'))
import numpy as np
import gen_analog as GA

CELLS = ['05_v1_engine.py', '06_base_lists.py', '07_rescore.py', '08_submit.py']
NAMES = {'gen_analog', '_gen_merge', 'GEN_STATS', 'GEN_Z', 'GEN_S1', '_gp', '_gk', '_gl', '_gs'}
INIT = ast.unparse(ast.parse('GEN_Z, GEN_S1 = {}, {}'))


def _cfg():
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    return ns['CFG']


def _is_guard(node):
    return isinstance(node, ast.If) and "CFG['GEN_ANALOG']" in ast.unparse(node.test)


def _strip(src):
    """The cell without the channel: guarded blocks and the creation of the two dicts removed (statement level)."""
    class T(ast.NodeTransformer):
        def generic_visit(self, node):
            super().generic_visit(node)
            for f in ('body', 'orelse', 'finalbody'):
                b = getattr(node, f, None)
                if isinstance(b, list) and b and isinstance(b[0], ast.stmt):
                    kept = [s for s in b if not (_is_guard(s) and not s.orelse) and ast.unparse(s) != INIT]
                    setattr(node, f, kept or ([ast.Pass()] if f == 'body' else []))
            return node
    return ast.unparse(T().visit(ast.parse(src)))


def test_default_off_and_keys():
    cfg = _cfg()
    assert cfg['GEN_ANALOG'] is None and cfg['GEN_ANALOG_POS'] == 6 and cfg['GEN_ANALOG_K'] == 5
    assert GA.MODES == ('slots', 'gate') and (GA.KF, GA.M_PARENTS, GA.KEY_CAP, GA.ROW_CAP) == (3, 3, 60, 25)


def test_cells_touch_the_channel_only_behind_the_flag():
    seen = 0
    for c in CELLS:
        src = open(os.path.join(SOL, 'nb', c), encoding='utf-8').read(); tree = ast.parse(src)
        parent = {ch: p for p in ast.walk(tree) for ch in ast.iter_child_nodes(p)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id in NAMES:
                seen += 1; x = n; ok = False; stmt = None
                while x in parent:
                    if isinstance(x, ast.stmt) and stmt is None:
                        stmt = x
                    x = parent[x]
                    if _is_guard(x):
                        ok = True; break
                assert ok or ast.unparse(stmt) == INIT, f'{c}: `{ast.unparse(stmt)[:80]}` uses the channel outside `if CFG[\'GEN_ANALOG\']`'
        stripped = _strip(src)
        assert not (set(n.id for n in ast.walk(ast.parse(stripped)) if isinstance(n, ast.Name)) & NAMES), c
    assert seen > 10                                                  # the channel is wired (cells 06 and 08)


def test_stripped_cells_equal_the_cells_before_the_channel():
    """Bit for bit at statement level against git, while git still holds the cells without the channel (skipped afterwards)."""
    root = os.path.dirname(os.path.dirname(SOL)); done = 0
    for c in CELLS + ['00_config.py']:
        rel = f'solutions/fusion-v4n/nb/{c}'
        try:
            old = subprocess.run(['git', 'show', f'HEAD:{rel}'], cwd=root, capture_output=True, timeout=30)
        except Exception:
            return
        if old.returncode != 0 or b'GEN_ANALOG' in old.stdout:
            continue
        new = open(os.path.join(SOL, 'nb', c), encoding='utf-8').read(); old = old.stdout.decode('utf-8')
        if c == '00_config.py':
            a, b = {}, {}
            exec(old, a); exec(new, b)
            extra = {k: v for k, v in b['CFG'].items() if k not in a['CFG']}
            assert extra == dict(GEN_ANALOG=None, GEN_ANALOG_POS=6, GEN_ANALOG_K=5) and all(b['CFG'][k] == v for k, v in a['CFG'].items())
        else:
            assert _strip(new) == ast.unparse(ast.parse(old)), f'{c}: the cell differs from HEAD outside the GEN_ANALOG blocks'
        done += 1
    print(f'  compared with git HEAD: {done} cells')


def test_built_notebooks_differ_in_the_override_only():
    cfgd = os.path.join(SOL, 'configs'); base = os.path.join(cfgd, 'v59a-v33b-ccby.json')
    with tempfile.TemporaryDirectory() as d:
        nbs = []
        for tag, extra in (('a', []), ('b', ['--set', 'GEN_ANALOG="slots"'])):
            o = os.path.join(d, tag)
            r = subprocess.run([sys.executable, os.path.join(SOL, 'tools', 'build_notebook.py'), '--user', 'u', '--slug', 's', '--config', base, '--out', o] + extra, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            nbs.append(json.load(open(os.path.join(o, 's.ipynb'), encoding='utf-8')))
            meta = json.load(open(os.path.join(o, 'kernel-metadata.json')))
            nbs.append(meta['dataset_sources'])
        assert nbs[1] == nbs[3]                                       # the channel needs no dataset
        diff = [i for i, (x, y) in enumerate(zip(nbs[0]['cells'], nbs[2]['cells'])) if x['source'] != y['source']]
        assert len(nbs[0]['cells']) == len(nbs[2]['cells']) and len(diff) == 1
        x, y = nbs[0]['cells'][diff[0]]['source'], nbs[2]['cells'][diff[0]]['source']
        assert x[:-1] == y[:-1] and 'CFG.update' in x[-1] and "'GEN_ANALOG': 'slots'" in y[-1] and 'GEN_ANALOG' not in x[-1]
    for name in os.listdir(cfgd):                                     # presets of the channel = v59a + the channel's keys
        if 'gen-analog' in name:
            p = json.load(open(os.path.join(cfgd, name))); b = json.load(open(base))
            assert {k: v for k, v in p.items() if not k.startswith('GEN_ANALOG') and k != 'VERSION'} == {k: v for k, v in b.items() if k != 'VERSION'}, name
            assert p['GEN_ANALOG'] in GA.MODES and (p['GEN_ANALOG'] != 'gate' or GA.MODEL is not None)


def test_rule_table_is_the_study_table():
    import rules as R
    assert GA.RULES == R.RULES and GA.MOVES == R.MOVES and len(GA.RULES) == 66 and len(GA.MOVES) == 61


# ------------------------------------------------------------------------------------------------ placement
def test_placement_identity_and_slots():
    fin = [f'd{i}' for i in range(25)]
    assert GA.place_slots(fin, [], 6, 5) == fin and GA.place(fin, [], 'slots') == fin
    for pos in (3, 6, 11):
        for g in (['g1'], ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7']):
            out = GA.place_slots(fin, g, pos, 5); k = min(5, len(g))
            assert len(out) == 25 and out[:pos - 1] == fin[:pos - 1] and out[pos - 1:pos - 1 + k] == g[:k] and out[pos - 1 + k:] == fin[pos - 1:25 - k]
    assert GA.place_slots(['a', 'b'], ['g1', 'g2'], 6, 5) == ['a', 'b', 'g1', 'g2'] and GA.place_slots([], ['g'], 6, 5) == ['g']
    assert GA.place_slots(fin, ['g1'], 6, 0) == fin


MODEL = dict(max_lib=0.7, s1_max=1.0, db=dict(mean=[0, 0, 0, 0], std=[1, 1, 1, 1], coef=[-1.0, 0.5, 0.6, 0.2], intercept=-1.0),
             gen=dict(mean=[0] * len(GA.GEN_FEATURES), std=[1] * len(GA.GEN_FEATURES), coef=[-0.8, -0.5, -0.3, 0.6, 0.2, 1.0, 0.5, 0.3, -0.4, 0.0, -0.7, -1.0, 0, 0, 0, 0], intercept=-1.5))


def _rows(n, seed=0):
    r = np.random.RandomState(seed)
    return [dict(smiles=f'g{i}', key=f'K{i}', ord=i + 1, prank=int(r.randint(1, 4)), step='1' if r.rand() < 0.7 else 'move', nsup=int(r.randint(1, 5)), nrule=int(r.randint(1, 3)), tan=float(r.rand()),
                 fz=float(r.randn()), pfz=float(r.randn()), fzn=float(r.randn()), fidx=int(r.randint(1, 4)), rule='methyl_O') for i in range(n)]


def test_gate_closed_is_identity_and_merge_keeps_database_order():
    fin = [f'd{i}' for i in range(25)]; rows = _rows(12)
    assert GA.place(fin, rows, 'gate', s1=0.0, lib_max=0.2, model=None) == fin or GA.MODEL is not None      # no fitted model: nothing happens
    for s1, lib in ((1.0, 0.2), (5.0, 0.0), (-3.0, 0.7), (-3.0, 0.95), (None, 0.1), (float('nan'), 0.1)):
        assert not GA.gate_open(lib, s1, MODEL) and GA.place(fin, rows, 'gate', s1=s1, lib_max=lib, model=MODEL) == fin
    assert GA.gate_open(0.69, 0.99, MODEL) and GA.gate_open(0.1, 5.0, MODEL, use_s1=False)
    moved = 0
    for seed in range(40):
        s1 = float(np.random.RandomState(seed).uniform(-6, 0.99)); rows = _rows(1 + seed % 25, seed)
        out = GA.place(fin, rows, 'gate', s1=s1, lib_max=0.3, model=MODEL)
        db = [x for x in out if x.startswith('d')]; gen = [x for x in out if x.startswith('g')]
        assert len(out) == 25 and len(set(out)) == 25 and db == fin[:len(db)] and set(gen) <= {r['smiles'] for r in rows}
        assert GA.place_merge(fin, [], s1, 0.3, MODEL) == fin
        moved += out != fin
    assert moved > 0
    X = GA.gen_features(_rows(7), 0.3, 0.2); assert X.shape == (7, len(GA.GEN_FEATURES)) and np.isfinite(X).all()
    assert GA.db_features(25, 99.0)[0].tolist() == [0.0, 1.0, 8.0, 8.0]


# ------------------------------------------------------------------------------------------------ generator
SMI = ['CCCCCCCC(=O)O', 'CCCCCCC(=O)O', 'CCCCCCCCC(=O)O', 'CCCCCC(O)CC(=O)O', 'CCCCCCC(C)C(=O)O', 'CCCCCCCC(=O)OC', 'OC(=O)CCCCCC=C', 'Oc1ccccc1', 'COc1ccccc1O']


def _pool(seed=0):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    f = [rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s)) for s in SMI]; smi = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in SMI]
    fp = np.packbits(np.random.RandomState(seed).rand(len(SMI), 64) > 0.5, axis=1)
    return GA.PoolIndex(smi, [GA.metric_key(s) for s in smi], f, fp), f, smi, np.random.RandomState(seed + 1).randn(64).astype(np.float32)


def test_generator_products_are_new_have_the_formula_and_respect_the_caps():
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    pool, forms, smi, z = _pool(); calls = []

    def key_fn(s):
        calls.append(s); return GA.metric_key(s)
    st = {}
    rows = GA.generate(pool, z, ['C9H18O2', None, 'C9H18O2', 'C8H16O2', 'C7H8O2', 'C6H6O'], set(), key_fn=key_fn, stats=st)
    assert rows and len(rows) <= GA.ROW_CAP and len(calls) <= GA.KEY_CAP and st['n_formulas'] == 3 and st['n_rows'] == len(rows)
    pk = set(pool.key); keys = [r['key'] for r in rows]
    assert len(set(keys)) == len(keys) and not (set(keys) & pk) and [r['ord'] for r in rows] == list(range(1, len(rows) + 1))
    for r in rows:
        assert rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(r['smiles'])) == r['formula'] and r['formula'] in ('C9H18O2', 'C8H16O2', 'C7H8O2')
        assert r['fidx'] in (1, 2, 3) and r['prank'] >= 1 and r['nsup'] >= 1 and r['nrule'] >= 1 and r['step'] in ('1', 'move') and 0.0 <= r['tan'] <= 1.0
        assert r['psmi'] in smi and '.' not in r['smiles'] and '@' not in r['smiles']
    assert GA.generate(pool, z, ['C9H18O2', 'C8H16O2', 'C7H8O2'], set()) == GA.generate(pool, z, ['C9H18O2', 'C8H16O2', 'C7H8O2'], set())     # deterministic
    ex = {rows[0]['key'], rows[1]['key']}
    rows2 = GA.generate(pool, z, ['C9H18O2', 'C8H16O2', 'C7H8O2'], ex)
    assert not ({r['key'] for r in rows2} & ex)
    fp = lambda s: np.ones(64, np.float32)
    rows3 = GA.generate(pool, z, ['C9H18O2'], set(), fp_fn=fp)
    assert all(abs(r['fz'] - float(z.sum()) / 100.0) < 1e-6 for r in rows3) and all(r['fidx'] == 1 for r in rows3)
    assert GA.generate(pool, z, [], set()) == [] and GA.generate(pool, z, ['C99H200'], set()) == []


def test_removed_answer_is_new_and_banned_as_a_parent():
    pool, forms, smi, z = _pool()
    ans = smi[4]; k = pool.key[4]                                     # 2-methyloctanoic acid: one methylation from octanoic acid
    rows = GA.generate(pool, z, ['C9H18O2'], set())
    assert k not in {r['key'] for r in rows}                           # a pool row is never generated
    rows = GA.generate(pool, z, ['C9H18O2'], set(), banned={4}, removed_keys={k})
    hit = [r for r in rows if r['key'] == k]
    assert hit and hit[0]['pid'] != 4 and all(r['pid'] != 4 for r in rows)


def test_interleave_and_formulas():
    a = [dict(smiles=f'a{i}') for i in range(4)]; b = [dict(smiles=f'b{i}') for i in range(3)]; c = [dict(smiles='c0')]
    out = GA.interleave([a, b, c])
    assert [d['smiles'] for d in out] == ['a0', 'a1', 'b0', 'a2', 'c0', 'a3', 'b1', 'b2'] and [d['fidx'] for d in out] == [1, 1, 2, 1, 3, 1, 2, 2]
    assert GA.target_formulas(['A', None, 'A', 'B', 'C', 'D'], 3) == ['A', 'B', 'C'] and GA.target_formulas(['A', 'B'], 1) == ['A']
    assert GA.fstr(GA.minus(GA.parse('C9H18O2'), dict(C=1, H=2))) == 'C8H16O2' and GA.fstr(GA.minus(GA.parse('CH4'), dict(C=2))) is None


def test_run_channel():
    pool, forms, smi, z = _pool()
    fin = {'m1': [f'CCCC{"C" * i}N' for i in range(25)], 'm2': ['CCO'], 'm3': ['CCN', 'CCCN'], 'm4': ['CCN']}
    base = {m: [[smi[2]], [pool.key[2]], lib, [1.0], ['C9H18O2'], [2]] for m, lib in (('m1', 0.2), ('m2', 0.95), ('m3', 0.2))}
    zs = {'m1': z, 'm2': z, 'm4': z}; log = []
    out, st = GA.run_channel(list(fin), fin, base, zs, {}, pool, 'slots', log=lambda *a: log.append(a))
    assert out['m3'] == fin['m3'] and out['m4'] == fin['m4'] and st['no_context'] == 2 and st['generated_for'] == 2 and st['failed'] == 0
    assert out['m1'][:5] == fin['m1'][:5] and out['m1'][10:] == fin['m1'][5:20] and len(out['m1']) == 25 and out['m2'][0] == 'CCO' and len(out['m2']) == 6
    new = out['m1'][5:10]; assert len({GA.metric_key(s) for s in new}) == 5 and not ({GA.metric_key(s) for s in new} & set(pool.key))
    o2, s2 = GA.run_channel(list(fin), fin, base, zs, {}, pool, 'gate', model=MODEL)             # no fold-safe score: the gate stays closed
    assert o2 == fin and s2['gate_open'] == 0 and s2['generated_for'] == 0
    o3, s3 = GA.run_channel(list(fin), fin, base, zs, {'m1': -2.0, 'm2': -2.0}, pool, 'gate', model=MODEL)
    assert s3['gate_open'] == 1 and o3['m2'] == fin['m2'] and [x for x in o3['m1'] if x in fin['m1']] == fin['m1'][:len([x for x in o3['m1'] if x in fin['m1']])]
    o4, s4 = GA.run_channel(list(fin), fin, base, zs, {}, pool, 'slots', budget_s=-1.0)
    assert o4 == fin and s4['skipped_budget'] == 2
    boom = lambda s: (_ for _ in ()).throw(RuntimeError('x'))
    o5, s5 = GA.run_channel(list(fin), fin, base, zs, {}, pool, 'slots', key_fn=boom, log=lambda *a: log.append(a))
    assert o5 == fin and s5['failed'] == 2
    try:
        GA.run_channel(list(fin), fin, base, zs, {}, pool, 'slots', key_fn=boom, strict=True); raise AssertionError('strict must raise')
    except RuntimeError:
        pass


if __name__ == '__main__':
    for n, f in sorted(globals().items()):
        if n.startswith('test_'):
            f(); print('ok', n)
    print('GEN_ANALOG OK: off by default and unreachable with the flag off; placement and generator invariants hold')
