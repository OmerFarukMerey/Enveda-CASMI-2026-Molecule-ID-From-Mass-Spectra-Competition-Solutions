"""Build a private kernel: bgunshui's 0.480 neural ensemble + our own seeds of the same 52-epoch recipe as extra members.
usage: build_ens3.py --base NB_DIR --models DIR --slug SLUG --w-pub 0.5 --w-bg 0.25 --w-om 0.25 [--notice TEXT]
DIR holds om_seed_*.pt (kit checkpoint format); their SHA-256s are pinned in the cell."""
import argparse, hashlib, json
from pathlib import Path

OUT = Path(__file__).resolve().parents[4] / 'scratch' / 'kernels'  # repo-root scratch/ (not in git)
DATASET = 'omerfarukmerey/casmi26-om-nn-b3'

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True); ap.add_argument('--models', required=True); ap.add_argument('--slug', required=True)
ap.add_argument('--w-pub', type=float, required=True); ap.add_argument('--w-bg', type=float, required=True); ap.add_argument('--w-om', type=float, required=True)
ap.add_argument('--notice', default='')
a = ap.parse_args()
base = Path(a.base)
nb = json.loads(next(base.glob('*.ipynb')).read_text(encoding='utf-8'))
meta = json.loads((base / 'kernel-metadata.json').read_text(encoding='utf-8'))
shas = sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(a.models).glob('om_seed_*.pt')))
assert shas, 'no om_seed_*.pt'

FUNC = f'''# ── Extra members (team Arsokan): our own seeds of the same recipe as the retrained model (sanpier's kit, 52 epochs
# on every identity, constant learning rate), trained by us on train.parquet. Each scores the final lists like the two
# models above; their per-molecule z-scores are averaged and enter the ensemble with weight W_OM. Pinned by SHA-256;
# if any is missing or fails, the ensemble is bgunshui's 0.5 / 0.5 unchanged.
W_PUB, W_BG, W_OM = {a.w_pub!r}, {a.w_bg!r}, {a.w_om!r}
OM_SHA = {shas!r}
OM_REPORT = dict(status='not run')


def _om_z(index, zf):
    import shutil as _s, hashlib as _h
    from pathlib import Path as _P
    import pandas as _pd
    global BUNDLE
    found = {{}}
    for p in _P('/kaggle/input').rglob('om_seed_*.pt'):
        d = _h.sha256(p.read_bytes()).hexdigest()
        if d in OM_SHA: found.setdefault(d, p)
    if len(found) != len(OM_SHA):
        OM_REPORT.update(status='models missing', found=len(found)); print('OM: models missing', len(found), flush=True); return None
    zs = []
    for d in OM_SHA:
        p = found[d]; b = _P('/kaggle/working') / ('om_bundle_' + p.stem); b.mkdir(parents=True, exist_ok=True)
        for n in ('model.py', 'features.py', 'config.json'):
            _s.copy2(_P(PUB_BUNDLE) / n, b / n)
        _s.copy2(p, b / 'final_model.pt')
        BUNDLE = str(b)
        if not score_final_lists():
            OM_REPORT.update(status='scoring failed', model=p.name); print('OM: scoring failed', p.name, flush=True); return None
        f = _pd.read_parquet(ROOT / 'neural_candidate_features.parquet').set_index(['molecule_id', 'smiles']).reindex(index)
        _s.copy2(ROOT / 'neural_candidate_features.parquet', ROOT / f'neural_{{p.stem}}.parquet')
        zs.append(zf(f['neural_score'].fillna(0.0)))
    OM_REPORT.update(status='applied', models=[found[d].name for d in OM_SHA])
    return zf(sum(zs) / len(zs))


'''
OLD_LINE = "    _a['neural_score'] = 0.5 * _z(_a['neural_score'].fillna(0.0)) + 0.5 * _z(_b['neural_score'].fillna(0.0))\n"
NEW_LINE = '''    try:
        _zo = _om_z(_a.index, _z)
    except Exception as _e:  # noqa: BLE001 - our members never break the base ensemble
        import traceback as _tb; _tb.print_exc(); OM_REPORT.update(status='error', error=repr(_e)[:300]); _zo = None
    if _zo is not None:
        _a['neural_score'] = W_PUB * _z(_a['neural_score'].fillna(0.0)) + W_BG * _z(_b['neural_score'].fillna(0.0)) + W_OM * _zo
        print(f'ENSEMBLE neural = {W_PUB}*z(public) + {W_BG}*z(retrained) + {W_OM}*z(mean of our {len(OM_SHA)} seeds)', flush=True)
    else:
        _a['neural_score'] = 0.5 * _z(_a['neural_score'].fillna(0.0)) + 0.5 * _z(_b['neural_score'].fillna(0.0))
        print('ENSEMBLE neural = 0.5*z(public) + 0.5*z(ours)', flush=True)
    (ROOT / 'om_ensemble_report.json').write_text(json.dumps(dict(OM_REPORT, w=[W_PUB, W_BG, W_OM], sha=OM_SHA), indent=1))
'''
cells = nb['cells']
hit = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and 'def _swap_to_ours' in ''.join(c['source'])]
assert len(hit) == 1, hit
s = ''.join(cells[hit[0]]['source'])
OLD_PRINT = "    print('ENSEMBLE neural = 0.5*z(public) + 0.5*z(ours)', flush=True)\n"
assert s.count(OLD_LINE) == 1 and s.count('PUB_BUNDLE = BUNDLE\n') == 1 and s.count(OLD_PRINT) == 1
s = s.replace(OLD_PRINT, '')  # the fallback branch prints it now
s = s.replace('PUB_BUNDLE = BUNDLE\n', FUNC + 'PUB_BUNDLE = BUNDLE\n').replace(OLD_LINE, NEW_LINE)
cells[hit[0]]['source'] = s.splitlines(keepends=True)
for c in cells:
    if c['cell_type'] == 'code':
        c['outputs'] = []; c['execution_count'] = None
if a.notice:
    cells.insert(1, {'cell_type': 'markdown', 'metadata': {}, 'source': a.notice.splitlines(keepends=True)})
out = OUT / a.slug; out.mkdir(parents=True, exist_ok=True)
(out / f'{a.slug}.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
newmeta = dict(id=f'omerfarukmerey/{a.slug}', title=a.slug, code_file=f'{a.slug}.ipynb', language='python', kernel_type='notebook',
               is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False, dataset_sources=meta['dataset_sources'] + [DATASET],
               kernel_sources=meta['kernel_sources'], competition_sources=meta['competition_sources'], model_sources=[],
               docker_image=meta.get('docker_image'), machine_shape=meta.get('machine_shape', 'NvidiaTeslaT4'))
(out / 'kernel-metadata.json').write_text(json.dumps(newmeta, indent=2), encoding='utf-8')
print(out, f'patched cell {hit[0]}, {len(shas)} pinned members, weights {a.w_pub}/{a.w_bg}/{a.w_om}')
