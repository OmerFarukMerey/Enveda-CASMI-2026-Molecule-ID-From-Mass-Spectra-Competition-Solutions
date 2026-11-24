"""casmi26-sub-gnxv2 = gnxv (0.509 notebook + E180X 1.1) + real natural products missing from the pool as an extra row set of
the virtual tier. Changes against gnxv:
  + cell_np_extra.py inserted right before the tier cell (builds the row set from attached tables; fails closed)
  ~ tier cell, three marked edits: VT_CFG['EXTRA'] set when the row set exists; EXTRA paths that are absolute files are used as
    they are (the cell's _vt_find only searches /kaggle/input); if the overlay fails WITH the extra rows, it is re-run once
    without them (so a failure of the block falls back to gnxv's tier exactly)
  + an audit cell after the tier cell (extra rows in each window, entries in the final lists and their ranks; read-only)
  + dataset qwertykeypadapple/casmi26-lotus-candidates (CC0)
usage: python -I scratch/land1124/build_gnxv2.py"""
import copy, hashlib, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'casmi26-sub-gnxv2'
src = lambda c: c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
codes = lambda nb: [c for c in nb['cells'] if c['cell_type'] == 'code']
lines = lambda s: s.splitlines(keepends=True)
mk = lambda s: dict(cell_type='code', metadata={}, source=lines(s), outputs=[], execution_count=None)
base = json.load(open(os.path.join(HERE, 'sub', 'casmi26-sub-gnxv', 'casmi26-sub-gnxv.ipynb'), encoding='utf-8'))
bmeta = json.load(open(os.path.join(HERE, 'sub', 'casmi26-sub-gnxv', 'kernel-metadata.json'), encoding='utf-8'))


def rep(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:90])
    return s.replace(old, new)


nb = copy.deepcopy(base); C = codes(nb)
tier = C[-1]; t = src(tier); assert t.startswith('# ===== VIRTUAL NATURAL-PRODUCT TIER')
vtcfg_line = [l for l in t.splitlines() if l.startswith('VT_CFG = ')][0] + '\n'
t = rep(t, vtcfg_line, vtcfg_line + "if globals().get('ARSOKAN_NP_EXTRA'):   # Arsokan: real natural products missing from the pool (cell above), one extra row set\n"
                      "    VT_CFG['EXTRA'] = [[ARSOKAN_NP_EXTRA['mass'], ARSOKAN_NP_EXTRA['lp']]]\n")
t = rep(t, "        cfg['EXTRA'] = [[_vt_find(pat), _vt_find(lp) if lp else None] for pat, lp in cfg['EXTRA']]",
        "        _af = lambda p: p if os.path.isabs(p) and os.path.isfile(p) else _vt_find(p)   # Arsokan: absolute files as they are\n"
        "        cfg['EXTRA'] = [[_af(pat), _af(lp) if lp else None] for pat, lp in cfg['EXTRA']]")
t = rep(t, "try:\n    _vt_overlay(_vt_rerun)\nexcept Exception:\n",
        "try:\n    try:\n        _vt_overlay(_vt_rerun)\n    except Exception:\n"
        "        if not VT_CFG.get('EXTRA'):\n            raise\n"
        "        traceback.print_exc(); print('[np-extra] the tier failed WITH the extra rows: re-run without them', flush=True)   # Arsokan\n"
        "        VT_CFG.pop('EXTRA', None); _vt_overlay(_vt_rerun)\nexcept Exception:\n")
tier['source'] = lines(t)
audit = '''# ===== Arsokan NPX-TIER audit (read-only): extra rows per precursor window and their entries in the final lists =====
try:
    import json as _au_j, numpy as _au_np, pandas as _au_pd, os as _au_os
    if ARSOKAN_NP_EXTRA and _au_os.path.isfile('/kaggle/working/vt_lists.json'):
        _m = _au_np.load(ARSOKAN_NP_EXTRA['mass']); _o = _au_np.load(_au_os.path.dirname(ARSOKAN_NP_EXTRA['mass']) + '/npx_off.npy')
        _sb = _au_np.load(_au_os.path.dirname(ARSOKAN_NP_EXTRA['mass']) + '/npx_smiles.npy')
        _R = _au_j.load(open('/kaggle/working/vt_lists.json'))
        _n = []
        for _d in _R.values():
            _t = float(_d.get('target') or 0)
            _n.append(int(_au_np.searchsorted(_m, _t / (1 - 3e-6), 'right') - _au_np.searchsorted(_m, _t / (1 + 4.6e-6), 'left')) if _t else 0)
        _ext = set(bytes(_sb).decode('ascii')[int(_o[i]):int(_o[i + 1])] for i in range(len(_m)))
        _sub = _au_pd.read_csv('/kaggle/working/submission.csv', dtype=str)
        _pos = [i + 1 for x in _sub.smiles for i, s in enumerate(x.split(';')) if s in _ext]
        print(f'[np-extra] audit: extra rows {ARSOKAN_NP_EXTRA["rows"]:,}; per window median {_au_np.median(_n):.0f}, p90 {_au_np.percentile(_n, 90):.0f}, max {max(_n)}; '
              f'entries in the final lists {len(_pos)} at ranks {sorted(_pos)[:40]}', flush=True)
except Exception as _au_e:
    print('[np-extra] audit failed (no effect on the submission):', repr(_au_e)[:200], flush=True)
'''
extra = open(os.path.join(HERE, 'cell_np_extra.py'), encoding='utf-8').read().replace('\r\n', '\n')
i = nb['cells'].index(tier)
nb['cells'][i:i + 1] = [mk(extra), tier, mk(audit)]
for c in codes(nb):
    compile(src(c), NAME, 'exec'); c['outputs'] = []; c['execution_count'] = None
meta = dict(bmeta, id='huseyinemreaksoy/' + NAME, title=NAME, code_file=NAME + '.ipynb', is_private=True)
meta['dataset_sources'] = list(meta['dataset_sources']) + ['qwertykeypadapple/casmi26-lotus-candidates']
d = os.path.join(HERE, 'sub', NAME); os.makedirs(d, exist_ok=True); out = os.path.join(d, NAME + '.ipynb')
def dump():
    json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), ensure_ascii=False, separators=(',', ':'))
    return os.path.getsize(out)
size = dump()
if size >= 1_048_576:      # room: keep only the first markdown cell
    md = [c for c in nb['cells'] if c['cell_type'] == 'markdown'][0]
    nb['cells'] = [c for c in nb['cells'] if c['cell_type'] == 'code' or c is md]; size = dump()
assert size < 1_048_576, size
json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8', newline='\n'), indent=1)
back = codes(json.load(open(out, encoding='utf-8')))
for c in back:
    compile(src(c), NAME, 'exec')
B = codes(base)
print(NAME, 'bytes', size, '| code cells', len(back), '(gnxv', len(B), ') | unchanged from gnxv:', sum(src(c) in {src(x) for x in B} for c in back),
      '| sha', hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16])
