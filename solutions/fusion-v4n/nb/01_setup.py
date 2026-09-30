# setup: offline rdkit wheel, input discovery, MANIFEST check, embedded sources written to /kaggle/working/ours
import os, sys, glob, re, subprocess, time, json, hashlib, pickle
T0 = time.time()
WORK = '/kaggle/working'
STAGE = os.path.join(WORK, 'stage_cache')
OURS = os.path.join(WORK, 'ours')
os.makedirs(os.path.join(WORK, 'numba_cache'), exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = os.path.join(WORK, 'numba_cache')

def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]

def sha256(path, n=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(n), b''):
            h.update(b)
    return h.hexdigest()

def dump(name, obj):
    if CFG['DUMP_STAGES']:
        os.makedirs(STAGE, exist_ok=True)
        json.dump(obj, open(os.path.join(STAGE, f'{name}.json'), 'w'))

def log(*a):
    print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)

tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if tag in w]
log('rdkit wheel:', whl)
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])
import rdkit; log('rdkit', rdkit.__version__)
assert rdkit.__version__ == '2026.03.3', 'metric is pinned to RDKit 2026.03.3'

V4CODE = os.path.dirname(os.path.dirname(find('casmi26-v4b-models/**/casmi/engine.py')))
V4MOD = os.path.dirname(find('casmi26-v4b-models/**/MANIFEST.json'))
V3CODE = os.path.dirname(os.path.dirname(find('casmi26-v3-models/**/casmi/engine.py')))
V3MOD = os.path.dirname(find('casmi26-v3-models/**/ranker_0.pkl'))
POOL_DIR = os.path.dirname(find('pool_meta.parquet'))
PC_DIR = os.path.dirname(find('pc_smiles.npy'))
COMP = os.path.dirname(find('test.parquet'))
MAN = json.load(open(os.path.join(V4MOD, 'MANIFEST.json')))
bad = []
for rel, h in sorted(MAN['files'].items()):
    p = os.path.join(V4CODE, rel[len('code/'):]) if rel.startswith('code/') else os.path.join(V4MOD, rel)
    if not os.path.exists(p) or sha256(p) != h:
        bad.append(rel)
assert not bad, f'dataset files differ from MANIFEST.json: {bad}'
log('V4CODE', V4CODE, '| V4MOD', V4MOD, '| V3CODE', V3CODE, '| POOL', POOL_DIR, '| PC', PC_DIR, '| COMP', COMP)
log(f"{len(MAN['files'])} files match MANIFEST.json; families {MAN['families']}")

for _rel, _src in EMBED.items():        # EMBED is injected by tools/build_notebook.py from nb/embed/
    _p = os.path.join(OURS, _rel); os.makedirs(os.path.dirname(_p), exist_ok=True)
    open(_p, 'w').write(_src)
sys.path.insert(0, OURS)
import fusion_core
log('embedded sources:', sorted(EMBED))
json.dump(CFG, open(os.path.join(WORK, 'cfg.json'), 'w'), indent=1)
