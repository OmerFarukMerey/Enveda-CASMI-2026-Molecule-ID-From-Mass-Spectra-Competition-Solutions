"""RANKER_FS_NET: the fold-safe network of the second pass is a config key; None == v59a bit for bit.

Pinned without data: (1) the default is None and v59a does not set the key; (2) the only place that locates the network (cell 05)
resolves None to the public file's glob exactly as the cell of v59a did, and the v63a preset to our dataset; (3) apart from that one
expression cell 05 is identical to the v59a cell (the line is the only difference: replacing it back gives a source without the key);
(4) the notebook builder attaches our dataset only when the key is set; (5) v63a differs from v59a in VERSION and the two keys only.
usage: python tests/test_fs_net.py   (or pytest)
"""
import json, os, re, subprocess, sys, tempfile

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = "    _fs = find('casmi26-fold-safe-fpnet/**/fpnet_fold0.pt')\n"
NEW = "    _fs = find(CFG['RANKER_FS_NET'] or 'casmi26-fold-safe-fpnet/**/fpnet_fold0.pt')\n"


def cfg(preset=None):
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    c = ns['CFG']
    if preset:
        c.update(json.load(open(os.path.join(SOL, 'configs', preset), encoding='utf-8')))
    return c


def test_default_off_and_presets():
    assert cfg()['RANKER_FS_NET'] is None and cfg('v59a-v33b-ccby.json')['RANKER_FS_NET'] is None
    a = json.load(open(os.path.join(SOL, 'configs', 'v59a-v33b-ccby.json'))); b = json.load(open(os.path.join(SOL, 'configs', 'v63a-v59a-own-fsnet.json')))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {'VERSION', 'RANKER_FS_NET', 'RANKER_FS_DS'}, diff
    assert b['RANKER_FS_NET'] == 'casmi26-ours-fpnet-fs/**/fpnet_own_fs.pt' and b['RANKER_FS_DS'] == 'casmi26-ours-ranker-fs3'


def test_cell_resolves_the_glob():
    src = open(os.path.join(SOL, 'nb', '05_v1_engine.py'), encoding='utf-8').read()
    assert src.count(NEW) == 1 and OLD not in src and src.count('RANKER_FS_NET') == 1
    assert 'RANKER_FS_NET' not in src.replace(NEW, OLD)                 # that line is the only use of the key in the cell
    expr = re.search(r"_fs = find\((.*)\)\n", NEW).group(1)
    for preset, want in ((None, 'casmi26-fold-safe-fpnet/**/fpnet_fold0.pt'), ('v59a-v33b-ccby.json', 'casmi26-fold-safe-fpnet/**/fpnet_fold0.pt'),
                         ('v63a-v59a-own-fsnet.json', 'casmi26-ours-fpnet-fs/**/fpnet_own_fs.pt')):
        assert eval(expr, dict(CFG=cfg(preset))) == want, (preset, eval(expr, dict(CFG=cfg(preset))))
    other = [p for p in os.listdir(os.path.join(SOL, 'nb')) if p.endswith('.py') and p[:2] not in ('00', '05')]
    assert not [p for p in other if 'RANKER_FS_NET' in open(os.path.join(SOL, 'nb', p), encoding='utf-8').read()]


def test_builder_datasets():
    with tempfile.TemporaryDirectory() as t:
        got = {}
        for name in ('v59a-v33b-ccby.json', 'v63a-v59a-own-fsnet.json'):
            out = os.path.join(t, name[:4])
            subprocess.run([sys.executable, os.path.join(SOL, 'tools', 'build_notebook.py'), '--user', 'someone', '--slug', 'x', '--config',
                            os.path.join(SOL, 'configs', name), '--out', out], check=True, capture_output=True)
            got[name[:4]] = json.load(open(os.path.join(out, 'kernel-metadata.json')))['dataset_sources']
            nb = json.load(open(os.path.join(out, 'x.ipynb'), encoding='utf-8'))
            assert sum(NEW in ''.join(c['source']) for c in nb['cells']) == 1
        assert 'someone/casmi26-ours-fpnet-fs' not in got['v59a'] and 'someone/casmi26-ours-ranker-fs' in got['v59a']
        assert set(got['v63a']) - set(got['v59a']) == {'someone/casmi26-ours-fpnet-fs', 'someone/casmi26-ours-ranker-fs3'}
        assert set(got['v59a']) - set(got['v63a']) == {'someone/casmi26-ours-ranker-fs'}


if __name__ == '__main__':
    test_default_off_and_presets(); test_cell_resolves_the_glob(); test_builder_datasets()
    print('fs net OK: RANKER_FS_NET None == the public fold-safe file (v59a), v63a points to our network and ranker, builder attaches them')
