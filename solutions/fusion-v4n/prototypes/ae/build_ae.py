"""Build a private kernel: bgunshui's 0.480 neural ensemble + our analog-edit overlay cell after its final layer.
usage: build_ae.py --base NB_DIR --gate ae_gate_v1.txt --slug SLUG --tau 0.25 --lib-tau 0.95 [--notice TEXT] [--dataset ID]
Writes scratch/kernels/SLUG/ (notebook, kernel-metadata.json) and the filled cell ae_cell_filled.py (for the local test)."""
import argparse, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[3] / 'scratch' / 'kernels'  # repo-root scratch/ (not in git)

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True); ap.add_argument('--gate', required=True); ap.add_argument('--slug', required=True)
ap.add_argument('--tau', type=float, required=True); ap.add_argument('--lib-tau', type=float, required=True)
ap.add_argument('--notice', default=''); ap.add_argument('--dataset', default='omerfarukmerey/casmi26-om-ae')
a = ap.parse_args()
base = Path(a.base)
nb = json.loads(next(base.glob('*.ipynb')).read_text(encoding='utf-8'))
meta = json.loads((base / 'kernel-metadata.json').read_text(encoding='utf-8'))
cell = (HERE / 'ae_cell.py').read_text(encoding='utf-8')
fill = {'__AE_TAU__': repr(a.tau), '__AE_LIB_TAU__': repr(a.lib_tau), '__AE_CORE_SRC__': repr((HERE / 'ae_core.py').read_text(encoding='utf-8')),
        '__AE_GATE_SHA__': repr(hashlib.sha256(Path(a.gate).read_bytes()).hexdigest()), '__AE_DATASET__': a.dataset}
for k, v in fill.items():
    assert cell.count(k) == 1, k
    cell = cell.replace(k, v)
compile(cell, 'ae_cell', 'exec')
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
text = json.dumps(nb, ensure_ascii=False, indent=1)
(out / f'{a.slug}.ipynb').write_text(text, encoding='utf-8')
(out / 'ae_cell_filled.py').write_text(cell, encoding='utf-8')
newmeta = dict(id=f'omerfarukmerey/{a.slug}', title=a.slug, code_file=f'{a.slug}.ipynb', language='python', kernel_type='notebook',
               is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False, dataset_sources=meta['dataset_sources'] + [a.dataset],
               kernel_sources=meta['kernel_sources'], competition_sources=meta['competition_sources'], model_sources=[],
               docker_image=meta.get('docker_image'), machine_shape=meta.get('machine_shape', 'NvidiaTeslaT4'))
(out / 'kernel-metadata.json').write_text(json.dumps(newmeta, indent=2), encoding='utf-8')
print(out, f'overlay cell {len(cell):,} chars after cell {mi[0]}; notebook {len(text.encode()):,} bytes; tau {a.tau}, lib_tau {a.lib_tau}')
