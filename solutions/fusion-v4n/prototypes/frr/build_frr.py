"""Build a private kernel: baidalinadilzhan's MIDX notebook (public 0.479) + our FRR overlay cell after its final layer.
usage: build_frr.py --base NB.ipynb --model model.txt --slug SLUG --mode all|top --lam LAMBDA [--title T] [--notice TEXT]"""
import argparse, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path('C:/Users/omerm/AppData/Local/Temp/claude/C--Users-omerm-Desktop-Kaggle/f6bb9c18-a985-4e93-9584-49242fbbeecc/scratchpad/ultra10/kernels')

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True); ap.add_argument('--model', required=True); ap.add_argument('--slug', required=True)
ap.add_argument('--mode', choices=['all', 'top'], required=True); ap.add_argument('--lam', type=float, required=True)
ap.add_argument('--title'); ap.add_argument('--notice', default='')
a = ap.parse_args()
base = Path(a.base); nb = json.loads(base.read_text(encoding='utf-8')); meta = json.loads((base.parent / 'kernel-metadata.json').read_text(encoding='utf-8'))
txt = Path(a.model).read_bytes()
cell = (HERE / 'frr_cell.py').read_text(encoding='utf-8')
fill = {'__FRR_MODE__': repr(a.mode), '__FRR_LAMBDA__': repr(a.lam), '__FRR_FEAT_SRC__': repr((HERE / 'fragfeat.py').read_text(encoding='utf-8')),
        '__FRR_MODEL_SHA__': repr(hashlib.sha256(txt).hexdigest())}
for k, v in fill.items():
    assert cell.count(k) == 1, k
    cell = cell.replace(k, v)
cells = nb['cells']
mi = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and 'def pxm_merge' in ''.join(c['source'])]
assert len(mi) == 1, mi
cells.insert(mi[0] + 1, {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': cell.splitlines(keepends=True)})
for c in cells:
    if c['cell_type'] == 'code':
        c['outputs'] = []; c['execution_count'] = None
if a.notice:
    cells.insert(1, {'cell_type': 'markdown', 'metadata': {}, 'source': a.notice.splitlines(keepends=True)})
out = OUT / a.slug; out.mkdir(parents=True, exist_ok=True)
(out / f'{a.slug}.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
newmeta = dict(id=f'omerfarukmerey/{a.slug}', title=a.title or a.slug, code_file=f'{a.slug}.ipynb', language='python', kernel_type='notebook',
               is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False, dataset_sources=meta['dataset_sources'] + ['omerfarukmerey/casmi26-om-frr'], kernel_sources=[],
               competition_sources=meta['competition_sources'], model_sources=[], docker_image=meta.get('docker_image'),
               machine_shape=meta.get('machine_shape', 'NvidiaTeslaT4'))
(out / 'kernel-metadata.json').write_text(json.dumps(newmeta, indent=2), encoding='utf-8')
print(out, f'cell {len(cell)} chars after cell {mi[0]}, model sha {hashlib.sha256(txt).hexdigest()[:12]}, mode {a.mode}, lambda {a.lam}')
