"""FUSE_KRR (the constant of the engine-2 rank fusion) and the preset v35a-v33b-krr1 (reviews/regime-audit-2026-10-07.md).

Checks without competition data: the default stays 3.0 in the notebook config and in fusion_core; the preset is v33b's config
plus FUSE_KRR 1.0 and nothing else; fuse_rrf at 3.0 gives the reference scores 1/(3+r) + 0.6/(3+r); with 1.0 a first row that
engine 2 cannot list (a joined PubChem row) keeps the first place against a second row that engine 2 lists third, which it
loses at 3.0; the order of a list that engine 2 does not touch never depends on the constant; build_submission passes the key
through; the build tool writes it into the notebook.
usage: python tests/test_fuse_krr.py   (or pytest)
"""
import json, os, random, subprocess, sys, tempfile

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed'))
import fusion_core

KEY = lambda s: s


def load_cfg():
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    return ns['CFG']


def test_default_unchanged():
    assert load_cfg()['FUSE_KRR'] == 3.0 and fusion_core.DEFAULTS['FUSE_KRR'] == 3.0
    assert load_cfg()['FUSE_ALPHA'] == 0.6 and fusion_core.DEFAULTS['FUSE_ALPHA'] == 0.6


def test_preset_is_v33b_plus_one_key():
    a = json.load(open(os.path.join(SOL, 'configs', 'v33b-v24a-fsranker-gate07-rankavg.json')))
    b = json.load(open(os.path.join(SOL, 'configs', 'v35a-v33b-krr1.json')))
    a.pop('VERSION'); assert b.pop('VERSION') == 'v35a-v33b-krr1'
    assert b.pop('FUSE_KRR') == 1.0 and a == b
    assert set(b) <= set(load_cfg()), 'every key of the preset is a config key'


def test_reference_scores_at_default():
    v = ['A', 'B', 'C']; e = ['B', 'D', 'A']
    smis, keys, sc = fusion_core.fuse_rrf(v, e, e, KEY, 0.6, 3.0, 40)
    ref = {'A': 1 / 4 + 0.6 / 6, 'B': 1 / 5 + 0.6 / 4, 'C': 1 / 6, 'D': 0.6 / 5}
    assert keys == sorted(ref, key=lambda k: -ref[k]) and all(abs(s - ref[k]) < 1e-12 for k, s in zip(keys, sc))


def test_first_row_engine2_cannot_list():
    v = ['JOINED', 'POOL', 'X']; e = ['Y', 'Z', 'POOL']          # engine 2 lists the second row third, the first row not at all
    assert fusion_core.fuse_rrf(v, e, e, KEY, 0.6, 3.0, 40)[1][0] == 'POOL'      # 1/4 < 1/5 + 0.6/6
    assert fusion_core.fuse_rrf(v, e, e, KEY, 0.6, 1.0, 40)[1][0] == 'JOINED'    # 1/2 > 1/3 + 0.6/4
    e = ['POOL', 'Y', 'Z']                                       # engine 2 puts the second row first: it wins at both constants
    assert fusion_core.fuse_rrf(v, e, e, KEY, 0.6, 3.0, 40)[1][0] == 'POOL' and fusion_core.fuse_rrf(v, e, e, KEY, 0.6, 1.0, 40)[1][0] == 'POOL'


def test_untouched_list_keeps_its_order():
    rnd = random.Random(0)
    for _ in range(50):
        v = ['m%d' % i for i in range(rnd.randint(1, 30))]
        for k in (0.5, 1.0, 3.0, 10.0):
            assert fusion_core.fuse_rrf(v, [], [], KEY, 0.6, k, 40)[0] == v
            assert fusion_core.fuse_rrf(v, v, v, KEY, 0.6, k, 40)[0] == v, 'two engines in agreement'


def test_build_submission_passes_the_key():
    base = {'m': [['JOINED', 'POOL', 'X'], ['JOINED', 'POOL', 'X'], 0.3, [3.0, 2.0, 1.0], ['f1', 'f2', 'f3'], [-2, 5, 6]]}
    eng = {'m': dict(smiles=['Y', 'Z', 'POOL'], keys=['Y', 'Z', 'POOL'])}
    run = lambda cfg: fusion_core.build_submission(base, {}, eng, {}, {}, cfg, ['m'], ['m'], score_key=KEY, formula=lambda s: s, log=lambda *a: None)[0].smiles[0].split(';')
    assert run({})[0] == 'POOL' and run(dict(FUSE_KRR=3.0))[0] == 'POOL' and run(dict(FUSE_KRR=1.0))[0] == 'JOINED'
    assert sorted(run({})) == sorted(run(dict(FUSE_KRR=1.0))), 'the constant re-orders, it never adds or drops a candidate here'


def test_build_tool_writes_the_key():
    out = tempfile.mkdtemp()
    subprocess.run([sys.executable, os.path.join(SOL, 'tools', 'build_notebook.py'), '--user', 'someone', '--out', out, '--config',
                    os.path.join(SOL, 'configs', 'v35a-v33b-krr1.json')], check=True, capture_output=True)
    nb = json.load(open(os.path.join(out, [f for f in os.listdir(out) if f.endswith('.ipynb')][0]), encoding='utf-8'))
    src = ''.join(''.join(c['source']) for c in nb['cells'])
    assert "'FUSE_KRR': 1.0" in src and "'RANKER': 'fs_avg'" in src
    meta = json.load(open(os.path.join(out, 'kernel-metadata.json')))
    ref = tempfile.mkdtemp()
    subprocess.run([sys.executable, os.path.join(SOL, 'tools', 'build_notebook.py'), '--user', 'someone', '--out', ref, '--config',
                    os.path.join(SOL, 'configs', 'v33b-v24a-fsranker-gate07-rankavg.json')], check=True, capture_output=True)
    assert meta['dataset_sources'] == json.load(open(os.path.join(ref, 'kernel-metadata.json')))['dataset_sources'], 'no new dataset'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); print('ok', name)
