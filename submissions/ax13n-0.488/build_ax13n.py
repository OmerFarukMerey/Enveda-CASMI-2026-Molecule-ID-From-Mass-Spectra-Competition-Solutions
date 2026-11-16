"""ax13n / ax13tn: copy of analyticaobscura/casmi-2026 (public 0.488 = alexchilton v13 + N-1c in-test molecular-network
rerank), optionally (--t) with the ax13t train-blind guard in the external-library overlay.
Local evidence: scratch/land1113/log_n1c.txt (N-1c on the 600 clean lists: +0.0006, top-1 never changed).
usage: python -I scratch/land1113/build_ax13n.py [--t]"""
import copy, glob, hashlib, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); T = '--t' in sys.argv
SRC = os.path.join(HERE, 'ao'); NAME = 'casmi26-sub-ax13tn' if T else 'casmi26-sub-ax13n'
bt = open(os.path.join(HERE, 'build_ax13t.py'), encoding='utf-8').read()
g = {}; exec(bt[bt.index('PATCH = ['):bt.index('NOTICE = ')], g); PATCH = g['PATCH']
NOTICE = f"""# CASMI26 {NAME}: "Casmi-2026" by analyticaobscura{' with a train-blind guard' if T else ''}

**Notice.** Derived from the public Kaggle notebook [casmi-2026](https://www.kaggle.com/code/analyticaobscura/casmi-2026)
by analyticaobscura (Apache-2.0), itself derived from
[casmi26-lb-0-487-peaks-to-structures-scorer](https://www.kaggle.com/code/alexchilton/casmi26-lb-0-487-peaks-to-structures-scorer)
by alex chilton (Apache-2.0). The credits of the upstream authors stay in the cells below. **Changes:** this notice cell is
added{'; the external-library overlay promotes a library match only when its structure is absent from `train.parquet`' if T else ''}.
Nothing else is changed. Datasets and the attached notebook output keep their own licences.
"""
src = lambda c: c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
nb0 = json.load(open(os.path.join(SRC, 'casmi-2026.ipynb'), encoding='utf-8')); nb = copy.deepcopy(nb0)
if T:
    cell = [c for c in nb['cells'] if c['cell_type'] == 'code' and 'def _apex_extlib_overlay' in src(c)]
    assert len(cell) == 1; s = src(cell[0])
    for old, new in PATCH:
        assert s.count(old) == 1, old; s = s.replace(old, new)
    compile(s, 'overlay', 'exec'); cell[0]['source'] = s.splitlines(keepends=True)
a = [src(c) for c in nb['cells'] if c['cell_type'] == 'code']; b = [src(c) for c in nb0['cells'] if c['cell_type'] == 'code']
assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == int(T)
nb['cells'] = [dict(cell_type='markdown', metadata={}, source=NOTICE.splitlines(keepends=True))] + nb['cells']
meta0 = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
meta = {k: v for k, v in meta0.items() if k != 'id_no'}
meta.update(id='huseyinemreaksoy/' + NAME, title=NAME, code_file=NAME + '.ipynb', is_private=True, keywords=[])
assert meta['enable_internet'] is False and meta['enable_gpu'] is True and all(meta['dataset_sources']) and all(meta['kernel_sources'])
d = os.path.join(HERE, 'sub', NAME); os.makedirs(d, exist_ok=True); out = os.path.join(d, NAME + '.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), ensure_ascii=False, indent=1)
json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8'), indent=1)
print(out, 'code cells', len(a), 'changed', int(T), 'sha', hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16])
