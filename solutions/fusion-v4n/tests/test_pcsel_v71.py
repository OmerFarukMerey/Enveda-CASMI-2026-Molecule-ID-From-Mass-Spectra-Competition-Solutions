"""Tests of the v71 notebook builder (scratch/pcsel/build_v71.py): runner patch anchors, cell syntax, base cells kept, switches; no competition data."""
import ast, json, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'scratch', 'pcsel'))
import build_v71 as B  # noqa: E402


def _cells():
    return [''.join(c['source']) for c in json.load(open(B.BASE_NB, encoding='utf-8'))['cells']]


def _runner():
    src = next(s for s in _cells() if s.startswith('V17_PATCH = '))
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'RUNNER':
            return ast.literal_eval(n.value)


def _patch_fn():
    ns = {}
    exec(B.CELL_C.split('V71_PC_OLD = ')[0], ns)
    return ns['_v71_patch_runner']


def test_runner_patch_compiles_and_only_adds():
    r = _runner(); p = _patch_fn()(r)
    compile(p, 'runner', 'exec')
    assert "os.environ['V71_Z']" in p and "q['best_pool_fz'] * q['v71_r']" in p
    removed = [l for l in r.splitlines() if l not in p.splitlines()]
    assert removed == ["            best = float((F @ z.astype(np.float64)).max())", "            best = float('nan')",
                       "        extra[mid] = dict(best_pool_fz=best, target=target)"], removed


def test_patch_refuses_unknown_runner():
    try:
        _patch_fn()('print(1)\n')
    except AssertionError:
        return
    raise AssertionError('patch accepted a runner without its anchors')


def test_new_cells_compile():
    for s in (B.CELL_A.replace('__V71__', '{}').replace('__VAL__', '{}'), B.CELL_B.replace('__HELPER__', repr(B.HELPER)), B.CELL_D, B.HELPER):
        compile(s, 'cell', 'exec')


def test_build_keeps_base_cells():
    base = _cells(); old = B.HERE
    with tempfile.TemporaryDirectory() as td:
        B.HERE = td
        try:
            B.build()
        finally:
            B.HERE = old
        for slug, sw in B.VARIANTS.items():
            nb = json.load(open(os.path.join(td, 'nb', slug, f'{slug}.ipynb'), encoding='utf-8')); src = [''.join(c['source']) for c in nb['cells']]
            pc = next(b for b in base if b.startswith('V17_PATCH = '))
            assert sum(s in base for s in src) == len(base) - 1 and sum(s.startswith(pc.rstrip('\n')) and s != pc for s in src) == 1 and len(src) == len(base) + 3
            assert repr(sw) in next(s for s in src if s.startswith('"""v71 (ours'))
            meta = json.load(open(os.path.join(td, 'nb', slug, 'kernel-metadata.json')))
            assert meta['is_private'] and meta['id'].endswith(slug) and B.DATASET in meta['dataset_sources']


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
