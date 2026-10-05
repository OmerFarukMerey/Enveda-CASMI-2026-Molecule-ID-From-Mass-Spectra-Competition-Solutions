"""ENG_EXTRA_FPNET: an extra FPNet averaged into engine-2's fingerprint logits (default off).

Checks without competition data or weights: the flag exists and is off; with it off the runner selects exactly the
reference's two models; with it on the extra checkpoint is appended and lands in the per-spectrum group; the preset
and the build tool attach the dataset only when the flag is set; averaging a third network changes the logits by the
group mean (small random networks of the engine's own FPNet class, skipped when torch is not installed).
usage: python tests/test_eng_extra_fpnet.py   (or pytest)
"""
import json, os, subprocess, sys, tempfile

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed', 'eng'))
import eng_runner

REF = ['/kaggle/input/casmi26-fp-models-v2/fp_single_s2.pt', '/kaggle/input/casmi26-fp-models-v2/fp_merged_m1.pt']
OTHER = ['/kaggle/input/casmi26-simulated-ranker-rows/ours_fpnet_single_16k.pt', '/kaggle/input/x/fp_other.pt']
NP = '/kaggle/input/datasets/goooman/casmi26-fp-npx4-soup/fp_single_npx4soup.pt'


def load_cfg():
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    return ns['CFG']


def test_flag_default_off():
    assert load_cfg()['ENG_EXTRA_FPNET'] is None


def test_selection():
    assert eng_runner.select_fp_models(REF + OTHER + [NP]) == sorted(REF), 'off: only the reference models'
    got = eng_runner.select_fp_models(REF + OTHER + [NP], [NP])
    assert got == sorted(REF) + [NP]
    assert eng_runner.select_fp_models(REF, [REF[0]]) == sorted(REF), 'a reference model is never loaded twice'
    group = lambda p: 'merged' if 'merged' in p.split('/')[-1] else 'single'      # the rule of pv_fp.load_fp_models
    assert [group(p) for p in got] == ['merged', 'single', 'single']


def test_cell_passes_paths_and_fails_commit_without_dataset():
    src = open(os.path.join(SOL, 'nb', '03_engine2.py'), encoding='utf-8').read()
    assert 'EXTRA_FP=_extra' in src and "CFG['ENG_EXTRA_FPNET']" in src
    assert src.index('assert IS_RERUN') < src.index('    try:'), 'the dataset check must sit outside the fallback try'


def _build(args):
    out = tempfile.mkdtemp()
    subprocess.run([sys.executable, os.path.join(SOL, 'tools', 'build_notebook.py'), '--user', 'someone', '--out', out] + args,
                   check=True, capture_output=True)
    meta = json.load(open(os.path.join(out, 'kernel-metadata.json')))
    nb = json.load(open(os.path.join(out, 'casmi26-ours.ipynb'), encoding='utf-8'))
    return meta['dataset_sources'], ''.join(''.join(c['source']) for c in nb['cells'][1:3])


def test_build_attaches_dataset_only_with_flag():
    ds, _ = _build(['--config', os.path.join(SOL, 'configs', 'v13a-v12a-n100.json')])
    assert not any('npx4' in d for d in ds)
    ds, cfg = _build(['--config', os.path.join(SOL, 'configs', 'v25a-v13a-np-eng2.json')])
    assert 'goooman/casmi26-fp-npx4-soup' in ds and "'ENG_EXTRA_FPNET': 'goooman/casmi26-fp-npx4-soup'" in cfg
    a = json.load(open(os.path.join(SOL, 'configs', 'v13a-v12a-n100.json'))); b = json.load(open(os.path.join(SOL, 'configs', 'v25a-v13a-np-eng2.json')))
    a.pop('VERSION'); b.pop('VERSION')
    assert b.pop('ENG_EXTRA_FPNET') and a == b, 'v25a = v13a + the one flag'


def test_np_ours_value_selects_our_dataset():
    ds, cfg = _build(['--config', os.path.join(SOL, 'configs', 'v25b-v13a-np-ours.json')])
    assert 'someone/casmi26-fp-np-ours' in ds and not any('goooman' in d for d in ds) and "'ENG_EXTRA_FPNET': 'np-ours'" in cfg
    a = json.load(open(os.path.join(SOL, 'configs', 'v13a-v12a-n100.json'))); b = json.load(open(os.path.join(SOL, 'configs', 'v25b-v13a-np-ours.json')))
    a.pop('VERSION'); b.pop('VERSION')
    assert b.pop('ENG_EXTRA_FPNET') == 'np-ours' and a == b, 'v25b = v13a + the one flag'
    src = open(os.path.join(SOL, 'nb', '03_engine2.py'), encoding='utf-8').read()
    assert "{'np-ours': 'casmi26-fp-np-ours'}" in src, 'the cell looks for the folder of our dataset'
    ours = '/kaggle/input/datasets/someone/casmi26-fp-np-ours/fp_single_np_ours.pt'
    assert eng_runner.select_fp_models(REF + [ours], [ours]) == sorted(REF) + [ours] and 'merged' not in ours.split('/')[-1]


def test_soup_is_the_weight_average():
    try:
        import torch
    except ImportError:
        print('torch not installed: soup check skipped'); return
    import pv_fp
    d = tempfile.mkdtemp(); torch.manual_seed(0); nets = [pv_fp.FPNet(32, d=16, layers=1, heads=2) for _ in range(3)]
    for i, n in enumerate(nets):
        torch.save(dict(step=1, nbits=32, d=16, layers=1, model=n.state_dict()), os.path.join(d, f'm{i}.pt'))
    tr = os.path.join(SOL, 'train', 'train_eng_fpnet.py')
    subprocess.run([sys.executable, tr, '--run', d, '--soup', ','.join(os.path.join(d, f'm{i}.pt') for i in (1, 2)), '--out', os.path.join(d, 'fp_single_soup.pt')],
                   check=True, capture_output=True)
    ck = torch.load(os.path.join(d, 'fp_single_soup.pt'), weights_only=True)
    assert (ck['nbits'], ck['d'], ck['layers']) == (32, 16, 1) and ck['soup_of'] == ['m1.pt', 'm2.pt']
    k = 'head.3.weight'; assert torch.allclose(ck['model'][k], (nets[1].state_dict()[k] + nets[2].state_dict()[k]) / 2, atol=1e-6)
    subprocess.run([sys.executable, tr, '--run', d, '--soup', os.path.join(d, 'm1.pt'), '--out', os.path.join(d, 'half.pt'), '--soup_base', os.path.join(d, 'm0.pt'),
                    '--soup_alpha', '0.5'], check=True, capture_output=True)
    h = torch.load(os.path.join(d, 'half.pt'), weights_only=True)['model'][k]
    assert torch.allclose(h, (nets[0].state_dict()[k] + nets[1].state_dict()[k]) / 2, atol=1e-6), 'base + alpha * (soup - base)'
    pv_fp.FPNet(32, d=16, layers=1, heads=2).load_state_dict(ck['model'])       # loads into the engine's class


def test_average_of_the_single_group():
    try:
        import torch
    except ImportError:
        print('torch not installed: averaging check skipped'); return
    import numpy as np, pv_fp
    torch.manual_seed(0)
    nets = [pv_fp.FPNet(32, d=16, layers=1, heads=2).eval() for _ in range(3)]
    specs = [(np.array([50.0, 80.0, 120.0], np.float32), np.array([1.0, 0.4, 0.2], np.float32))] * 2
    args = (specs, np.array([150.0, 150.0], np.float32), ['[M+H]+'] * 2, ['timsTOF'] * 2, np.array([20.0, 40.0], np.float32), np.array([1.0, 1.0]))
    z = lambda single, merged: pv_fp.molecule_logits((single, merged, 'cpu'), *args)
    a, b, m = z([nets[0]], []), z([nets[1]], []), z([], [nets[2]])
    assert np.allclose(z([nets[0], nets[1]], []), (a + b) / 2, atol=1e-5), 'the extra model is averaged with the public per-spectrum model'
    assert np.allclose(z([nets[0], nets[1]], [nets[2]]), ((a + b) / 2 + m) / 2, atol=1e-5), 'then averaged with the merged view'


if __name__ == '__main__':
    test_flag_default_off(); test_selection(); test_cell_passes_paths_and_fails_commit_without_dataset()
    test_build_attaches_dataset_only_with_flag(); test_np_ours_value_selects_our_dataset(); test_soup_is_the_weight_average(); test_average_of_the_single_group()
    print('ENG_EXTRA_FPNET OK: default off, selection, commit check, dataset attached only with the flag, group average')
