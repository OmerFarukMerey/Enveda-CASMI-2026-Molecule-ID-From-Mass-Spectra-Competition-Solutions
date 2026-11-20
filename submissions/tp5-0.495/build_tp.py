"""tp5: copy of taeyangg4/casmi26-lb-0-495-network-fr-p5-tail (public 0.495) with the 117 GNPS-NIST14-MATCHES rows
dropped from the external-library overlay (team rule: no NIST-derived data) and a notice cell.
usage: python -I scratch/land1120/build_tp.py"""
import copy, hashlib, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); NAME = 'casmi26-sub-tp5'
SRC = os.path.join(HERE, 'casmi26-lb-0-495-network-fr-p5-tail')
NOTICE = f"""# CASMI26 {NAME}: "CASMI26 LB 0.495 Network + FR + P5 Tail" by Taeyang, without NIST-matched library rows

**Notice.** Derived from the public Kaggle notebook
[casmi26-lb-0-495-network-fr-p5-tail](https://www.kaggle.com/code/taeyangg4/casmi26-lb-0-495-network-fr-p5-tail)
by Taeyang (Apache-2.0), which builds on analyticaobscura, alex chilton, mathleteutd and others credited in the cells
below. **Changes:** this notice cell is added, and the external-library overlay drops the reference spectra whose
`source_library` names NIST (117 rows of GNPS-NIST14-MATCHES in taeyangg4/casmi26-extlib-gnps-full), because our team
uses no NIST-derived data. Nothing else is changed. Datasets keep their own licences.
"""
OLD = "            for idx, row in meta_df.iterrows():\n                try:\n                    mode = 1 if 'pos'"
NEW = ("            # ours: drop NIST-matched rows (index kept, so mz_all[idx] stays aligned)\n"
       "            if 'source_library' in meta_df.columns:\n"
       "                _nist = meta_df['source_library'].astype(str).str.contains('NIST', case=False)\n"
       "                print(f'[apex-extlib] dropped {int(_nist.sum())} NIST-matched reference rows')\n"
       "                meta_df = meta_df[~_nist]\n" + OLD)
src = lambda c: c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
fn = [f for f in os.listdir(SRC) if f.endswith('.ipynb')][0]
nb0 = json.load(open(os.path.join(SRC, fn), encoding='utf-8')); nb = copy.deepcopy(nb0)
cell = [c for c in nb['cells'] if c['cell_type'] == 'code' and 'def _apex_extlib_overlay' in src(c)]
assert len(cell) == 1; s = src(cell[0]); assert s.count(OLD) == 1
s = s.replace(OLD, NEW); compile(s, 'overlay', 'exec'); cell[0]['source'] = s.splitlines(keepends=True)
a = [src(c) for c in nb['cells'] if c['cell_type'] == 'code']; b = [src(c) for c in nb0['cells'] if c['cell_type'] == 'code']
assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1
for c in nb['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
nb['cells'] = [dict(cell_type='markdown', metadata={}, source=NOTICE.splitlines(keepends=True))] + nb['cells']
meta0 = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
meta = {k: v for k, v in meta0.items() if k != 'id_no'}
meta.update(id='huseyinemreaksoy/' + NAME, title=NAME, code_file=NAME + '.ipynb', is_private=True, keywords=[])
assert meta['enable_internet'] is False and meta['enable_gpu'] is True and all(meta['dataset_sources']) and all(meta['kernel_sources'])
d = os.path.join(HERE, 'sub', NAME); os.makedirs(d, exist_ok=True); out = os.path.join(d, NAME + '.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), ensure_ascii=False, indent=1)
json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8'), indent=1)
print(out, 'code cells', len(a), 'changed 1', 'sha', hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16])
