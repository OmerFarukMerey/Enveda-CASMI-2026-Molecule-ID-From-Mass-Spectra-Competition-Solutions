"""gnx kernel folders from kozykappa's casmi26-lb-0-505 (scratch/land1123/pull/kozy505), built locally, never pushed here:
  casmi26-sub-gnx    kozy505 with its P5 cell (code cell 17, gfp5's) replaced by fpn's (gfpe code cell 18)  = gfpe + NPX
  casmi26-sub-gnx11  gnx with E180X 1.1 (gp505 code cell 23) in place of E180X 1.0 (code cell 22)
  casmi26-sub-gnx25  gnx11 with APRIOR _AP_LAM 0.25
Every other cell, markdown included, is kozy505's byte for byte. Metadata: kozy505's sources (= gfpe's), owner huseyinemreaksoy.
usage: python -I scratch/land1123/build_gnx.py"""
import copy, glob, hashlib, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
src = lambda c: c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
setsrc = lambda c, s: c.__setitem__('source', s.splitlines(keepends=True))
codes = lambda nb: [c for c in nb['cells'] if c['cell_type'] == 'code']
K = json.load(open(glob.glob(os.path.join(HERE, 'pull', 'kozy505', '*.ipynb'))[0], encoding='utf-8'))
GP = json.load(open(glob.glob(os.path.join(HERE, 'pull', 'gp505', '*.ipynb'))[0], encoding='utf-8'))
GE = json.load(open(os.path.join(ROOT, 'land1122', 'sub', 'casmi26-sub-gfpe', 'casmi26-sub-gfpe.ipynb'), encoding='utf-8'))
kmeta = json.load(open(os.path.join(HERE, 'pull', 'kozy505', 'kernel-metadata.json'), encoding='utf-8'))
gmeta = json.load(open(os.path.join(ROOT, 'land1122', 'sub', 'casmi26-sub-gfpe', 'kernel-metadata.json'), encoding='utf-8'))
KC, GC, EC = codes(K), codes(GP), codes(GE)
assert len(KC) == 23 and 'P5 tail fix' in src(KC[17])[:200] and 'NPX' in src(KC[21])[:100] and 'Ours: E180X -' in src(KC[22])[:60]
assert 'Arsokan fpn' in src(EC[18])[:200] and 'E180X 1.1' in src(GC[23])[:60]
assert set(gmeta['dataset_sources']) <= set(kmeta['dataset_sources']) and gmeta['kernel_sources'] == kmeta['kernel_sources'], 'P5 cell inputs'


def build(name, edits):
    nb = copy.deepcopy(K); C = codes(nb)
    for i, s in edits.items():
        setsrc(C[i], s)
    for c in C:
        compile(src(c), name, 'exec'); c['outputs'] = []; c['execution_count'] = None
    meta = {k: kmeta[k] for k in ('language', 'kernel_type', 'enable_gpu', 'enable_tpu', 'enable_internet', 'dataset_sources',
                                   'kernel_sources', 'competition_sources', 'model_sources')}
    meta = dict(id='huseyinemreaksoy/' + name, title=name, code_file=name + '.ipynb', is_private=True, **meta)
    assert meta['enable_internet'] is False and meta['enable_gpu'] is True
    d = os.path.join(HERE, 'sub', name); os.makedirs(d, exist_ok=True); out = os.path.join(d, name + '.ipynb')
    json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), ensure_ascii=False, separators=(',', ':'))
    json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8', newline='\n'), indent=1)
    size = os.path.getsize(out); assert size < 1_048_576, size
    back = codes(json.load(open(out, encoding='utf-8')))
    for c in back:
        compile(src(c), name, 'exec')
    diff = [i for i, (a, b) in enumerate(zip(back, KC)) if src(a) != src(b)]
    md_same = [src(c) for c in nb['cells'] if c['cell_type'] == 'markdown'] == [src(c) for c in K['cells'] if c['cell_type'] == 'markdown']
    print(f'{name}: code cells differing from kozy505 {diff}, markdown identical {md_same}, bytes {size}, sha {hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]}')


p5 = src(EC[18]); e11 = src(GC[23])
ap = src(KC[19]); assert ap.count('_AP_LAM = 0.5 ') == 1
build('casmi26-sub-gnx', {17: p5})
build('casmi26-sub-gnx11', {17: p5, 22: e11})
build('casmi26-sub-gnx25', {17: p5, 22: e11, 19: ap.replace('_AP_LAM = 0.5 ', '_AP_LAM = 0.25')})
