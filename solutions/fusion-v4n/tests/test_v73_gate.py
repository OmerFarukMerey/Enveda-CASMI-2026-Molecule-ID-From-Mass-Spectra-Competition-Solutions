"""Tests of the v73 notebook builder (scratch/v71diag/build_v73.py: stack1 + our network as the PubChem order, stack1's gate decisions); no competition data.

Covers: the merge function (gate 'old' reproduces the original rel on every branch of the merge cell's expression), cell syntax, base cells kept
verbatim, the switch-off path (PC stays the original object), the per-molecule fallbacks and the report's decision check, run on a fake channel.
"""
import ast, json, os, subprocess, sys, tempfile, types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'scratch', 'v71diag'))
import build_v73 as B  # noqa: E402


def _tail_ns(**extra):
    ns = dict(np=np, json=json, os=os, sys=sys, time=__import__('time'), subprocess=subprocess)
    ns.update(extra)
    exec(compile(B.cell_c(), 'cell_c', 'exec'), ns)
    return ns


def _base_cells():
    return [''.join(c['source']) if isinstance(c['source'], list) else c['source'] for c in json.load(open(B.BASE_NB, encoding='utf-8'))['cells']]


def test_merge_selftest_and_rel_identity():
    ns = _tail_ns(V73=dict(pc_net=False, gate='old'), V71_STAT=dict(active=False), PC={})
    assert ns['_v73_selftest']() == 'ok'
    rel, entry = ns['_v73_rel'], ns['_v73_entry']
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = dict(pc=['a'], pc_fz=[float(rng.normal())], pc_keys=['k'], fz_top=None if rng.random() < 0.3 else float(rng.normal() * 300),
                 best_pool_fz=float('nan') if rng.random() < 0.1 else float(rng.normal() * 300), S=1.0, top_pop=0.0)
        q = dict(pc=['b'], pc_fz=[2.0], pc_keys=['j'], fz_top=float(rng.normal() * 300), best_pool_fz=float(rng.normal() * 300), S=8.0, top_pop=6.0, v71_r=30.0)
        e = entry(q, p, 'old')
        assert rel(e) == rel(p) and e['pc'] == ['b'] and e['S'] == 8.0 and e['top_pop'] == 6.0
        assert rel(entry(q, p, 'rescaled')) == rel(q)


def test_switch_off_keeps_pc_object():
    pc = {'m1': dict(pc=['a'], pc_fz=[1.0], pc_keys=['k'], fz_top=1.0, best_pool_fz=0.0)}
    ns = _tail_ns(V73=dict(pc_net=False, gate='old'), V71_STAT=dict(active=False), PC=pc)
    assert ns['PC'] is pc and ns['V71_PC_OLD'] is pc and pc == {'m1': dict(pc=['a'], pc_fz=[1.0], pc_keys=['k'], fz_top=1.0, best_pool_fz=0.0)}


def test_fake_channel_fallbacks_and_report():
    old = {'m1': dict(pc=['A', 'B'], pc_fz=[7.0, 6.0], pc_keys=['ka', 'kb'], fz_top=400.0, best_pool_fz=300.0, S=7.0, top_pop=6.0),   # rel 100
           'm2': dict(pc=['C'], pc_fz=[3.0], pc_keys=['kc'], fz_top=None, best_pool_fz=float('nan'), S=3.0, top_pop=0.0),            # rel 1e9
           'm3': dict(pc=[], pc_fz=[], pc_keys=[], fz_top=None, best_pool_fz=1.0, S=None, top_pop=0.0),                               # original empty
           'm4': dict(pc=['D'], pc_fz=[1.0], pc_keys=['kd'], fz_top=10.0, best_pool_fz=0.0, S=1.0, top_pop=0.0)}                     # not listed
    new = {'m1': dict(pc=['E', 'A'], pc_fz=[9.0, 1.0], pc_keys=['ke', 'ka'], fz_top=20.0, best_pool_fz=1.0, S=9.0, top_pop=7.0, v71_new=True, v71_r=30.0),  # rescaled rel 570
           'm2': dict(pc=['F'], pc_fz=[2.0], pc_keys=['kf'], fz_top=5.0, best_pool_fz=float('nan'), S=2.0, top_pop=0.0, v71_new=True, v71_r=30.0),
           'm3': dict(pc=['G'], pc_fz=[2.0], pc_keys=['kg'], fz_top=5.0, best_pool_fz=0.0, S=2.0, top_pop=0.0, v71_new=True, v71_r=30.0)}
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'pc_lists_v71.json'); json.dump(new, open(out, 'w'))
        real_open = open

        def fake_open(f, *a, **k):
            if isinstance(f, str) and f.startswith('/kaggle/working/'):
                f = os.path.join(td, os.path.basename(f))
            return real_open(f, *a, **k)

        def fake_run(cmd, **k):
            return types.SimpleNamespace(returncode=0, stdout='V71_RUNNER {}', stderr='')
        fake_sub = types.SimpleNamespace(run=fake_run)
        stat = dict(selftest='ok', active=True, n_mols=0, n_new=0, n_fallback=0, fallback_ids=[], channel=None, secs={})
        logs = []
        ns = dict(np=np, json=json, os=os, sys=sys, time=__import__('time'), subprocess=fake_sub, open=fake_open,
                  V73=dict(pc_net=True, gate='old'), V71_STAT=stat, PC=old, RUNNER='', V3CODE='', V3MOD='', POOL_DIR='', PC_DIR='', COMP=td, V71_Z='',
                  v71_log=lambda tag, **kw: logs.append((tag, kw)))
        src = B.cell_c().replace("open('/kaggle/working/pc_runner_v71.py', 'w').write(_v71_patch_runner(RUNNER))", 'pass')
        exec(compile(src, 'cell_c', 'exec'), ns)
        assert stat['channel'] == 'ok', stat
        PC = ns['PC']
        assert PC['m1']['pc'] == ['E', 'A'] and ns['_v73_rel'](PC['m1']) == 100.0 and PC['m1']['S'] == 9.0
        assert PC['m2']['pc'] == ['F'] and ns['_v73_rel'](PC['m2']) == 1e9
        assert PC['m3'] is ns['V71_PC_OLD']['m3'] and PC['m4']['pc'] == ['D'] and stat['n_fallback'] == 2 and stat['n_new'] == 2
        assert {kw.get('reason') for t, kw in logs if t == 'V73_FALLBACK'} == {'original list empty', 'not listed by the patched run'}
        # report cell: decisions of the used channel equal the original's (rel-clause, slots); S-clause from the new rows
        rep = dict(ns, BASE={m: (['x'], ['k'], 0.0) for m in old}, LIB_TAU=0.9, REL_TH=600.0, S_TAU=6.0, POP_TAU=5.0, IS_RERUN=False, T0=0.0)
        exec(compile(B.CELL_D.replace("open('/kaggle/working/v73_report.json', 'w')", "open(os.path.join(COMP, 'r.json'), 'w')"), 'cell_d', 'exec'), rep)
        r = [kw for t, kw in logs if t == 'V73_PATCH' and kw.get('stage') == 'report'][0]
        assert r['rel_or_slot_differ'] == 0 and r['rel_clause_used'] == r['rel_clause_old'] == 1 and r['s_clause_old'] == 1 and r['s_clause_used'] == 1, r
        assert [kw for t, kw in logs if t == 'V73_CHECK'][0]['problems'] == []


def test_build_keeps_base_cells_and_parses():
    base = _base_cells()
    with tempfile.TemporaryDirectory() as td:
        out = B.build(td)
        for slug, sw in B.VARIANTS.items():
            nb = json.load(open(out[slug]['path'], encoding='utf-8'))
            src = [''.join(c['source']) for c in nb['cells']]
            pc = next(b for b in base if b.startswith('V17_PATCH = '))
            assert sum(s in base for s in src) == len(base) - 1 and len(src) == len(base) + 3
            assert sum(s.startswith(pc.rstrip('\n')) and s != pc for s in src) == 1
            for c, s in zip(nb['cells'], src):
                if c['cell_type'] == 'code':
                    ast.parse(s)
            assert repr(sw) in next(s for s in src if s.startswith('"""v73 (ours'))
            assert src[-1].startswith('"""v73 report') and 'def e2r2_row(' in src[-2]
            meta = json.load(open(os.path.join(os.path.dirname(out[slug]['path']), 'kernel-metadata.json')))
            assert meta['is_private'] and meta['id'] == f'huseyinemreaksoy/{slug}' and B.DATASET in meta['dataset_sources'] and 'id_no' not in meta


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
