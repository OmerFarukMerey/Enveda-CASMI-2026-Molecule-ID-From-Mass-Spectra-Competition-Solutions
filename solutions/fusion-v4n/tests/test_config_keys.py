"""Every CFG key a notebook cell reads, and every fusion_core knob, must exist in nb/00_config.py.

Regression: FRAG_LAM was only in fusion_core.DEFAULTS; cell 07 read CFG['FRAG_LAM'] and the val2 / vis runs crashed
with KeyError after ~3 h (only runs built with --set FRAG_LAM=... worked).
usage: python tests/test_config_keys.py   (or pytest)
"""
import glob, os, re, sys

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed'))
import fusion_core


def load_cfg():
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    return ns['CFG']


def test_cell_keys_defined():
    cfg = load_cfg()
    used = set()
    for p in glob.glob(os.path.join(SOL, 'nb', '[0-9][0-9]_*.py')):
        used |= set(re.findall(r"CFG\[['\"]([A-Za-z0-9_]+)['\"]\]", open(p, encoding='utf-8').read()))
    missing = sorted(used - set(cfg))
    assert not missing, f'cells read CFG keys missing from 00_config.py: {missing}'


def test_fusion_defaults_in_config_with_same_values():
    cfg = load_cfg()
    missing = sorted(k for k in fusion_core.DEFAULTS if k not in cfg and k != 'FALLBACK')
    assert not missing, f'fusion_core knobs missing from 00_config.py: {missing}'
    differ = {k: (cfg[k], v) for k, v in fusion_core.DEFAULTS.items() if k in cfg and cfg[k] != v}
    assert not differ, f'00_config.py defaults differ from fusion_core.DEFAULTS: {differ}'


if __name__ == '__main__':
    test_cell_keys_defined(); test_fusion_defaults_in_config_with_same_values()
    print('config keys OK: every CFG key used by the cells and every fusion_core knob is defined with the same default')
