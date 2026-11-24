"""casmi26-vh2-gnx11: PRIVATE harness kernel v2 on the 0.505 line (sub/casmi26-sub-gnx11), never submitted.
Same mechanics as build_vh_gfpe.py (one added cell after the session cell with the v2 builder vh2_cell_builder.py; the frozen
pipeline's three CASMI_VH_COMP lines; stage cells byte-identical) plus: snapshots around NPX and E180X 1.1, a scoring cell per
group (nphard / npref0 / e180) and an output-cleanup cell last. The leaderboard-field table cell (display only) is dropped.
Source notebook: submissions/gnx11-0.505/notebook.ipynb. Output: kernel/ next to this file.
usage: python -I solutions/fusion-v4n/kaggle/vh2/build_vh2.py"""
import base64, copy, hashlib, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'casmi26-vh2-gnx11'
SRC = os.path.join(HERE, '..', '..', '..', '..', 'submissions', 'gnx11-0.505')
src = lambda c: c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
lines = lambda s: s.splitlines(keepends=True)
mk = lambda s: dict(cell_type='code', metadata={}, source=lines(s), outputs=[], execution_count=None)


def rep(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:80])
    return s.replace(old, new)


g = json.load(open(os.path.join(SRC, 'notebook.ipynb'), encoding='utf-8'))
G = [c for c in g['cells'] if c['cell_type'] == 'code']
marks = {1: 'Session: paths, run mode', 3: 'The field as the Code tab', 14: '_NNET_NOTEBOOK = True', 15: '0.500 APEX OVERLAY',
         17: 'Arsokan fpn', 18: 'Arsokan fp5', 19: 'Ours: APRIOR', 20: '_UNIQ_NOTEBOOK = True', 21: 'ours: NPX', 22: 'Ours: E180X 1.1'}
assert len(G) == 23 and all(p in src(G[i])[:300] for i, p in marks.items()), [i for i, p in marks.items() if p not in src(G[i])[:300]]
assert 'FROZEN_B64 = ' in src(G[4]) and "COMP = next((p for p in" in src(G[1])
nb = copy.deepcopy(g); C = [c for c in nb['cells'] if c['cell_type'] == 'code']
s4 = src(C[4]); b64 = s4.split("FROZEN_B64 = '", 1)[1].split("'", 1)[0]; sha0 = s4.split("FROZEN_SHA256 = '", 1)[1].split("'", 1)[0]
raw = base64.b64decode(b64); assert hashlib.sha256(raw).hexdigest() == sha0
fz = json.loads(raw); FC = fz['cells']; fs = lambda i: src(FC[i])


def fset(i, t):
    FC[i]['source'] = lines(t) if isinstance(FC[i]['source'], list) else t
fset(4, rep(fs(4), 'DATA = locate_data_dir()\n',
            'DATA = (Path(os.environ["CASMI_VH_COMP"]).resolve() if os.environ.get("CASMI_VH_COMP") else locate_data_dir())  # VH\n'))
fset(26, rep(fs(26), "COMP = os.path.dirname(find('test.parquet'))\n",
             "COMP = os.path.dirname(find('test.parquet'))\nCOMP = os.environ.get('CASMI_VH_COMP') or COMP  # VH\n"))
fset(28, rep(fs(28), "    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}\n",
             "    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'} or comp == os.path.realpath(os.environ.get('CASMI_VH_COMP', '-')), comp  # VH\n"))
full = "_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)   # v4m\n"
fset(37, rep(fs(37), full, full.replace("_FULL = glob.glob(", "_FULL = [os.environ['CASMI_FPNET_FOLD0']] if os.environ.get('VH_CLEAN_MAIN') == '1' else glob.glob(", 1).rstrip('\n') + '  # VH clean main engine\n'))
for i in (4, 26, 28, 37):
    compile(fs(i), f'frozen{i}', 'exec')
raw2 = json.dumps(fz, ensure_ascii=False, separators=(',', ':')).encode('utf-8'); sha2 = hashlib.sha256(raw2).hexdigest()
s4 = rep(s4, b64, base64.b64encode(raw2).decode()); s4 = rep(s4, f"FROZEN_SHA256 = '{sha0}'", f"FROZEN_SHA256 = '{sha2}'")
C[4]['source'] = lines(s4)
sel = json.load(open(os.path.join(HERE, 'vh2_selection.json'), encoding='utf-8'))
builder = rep(open(os.path.join(HERE, 'vh2_cell_builder.py'), encoding='utf-8').read(), '__SEL_JSON__', json.dumps(sel, separators=(',', ':')))
scorer = open(os.path.join(HERE, 'vh2_cell_score.py'), encoding='utf-8').read()
cleanup = open(os.path.join(HERE, 'vh2_cell_cleanup.py'), encoding='utf-8').read()
preflight = open(os.path.join(HERE, 'vh2_cell_preflight.py'), encoding='utf-8').read()
snap = lambda tag: mk(f"vh_snap('{tag}')  # VH: copy of submission.csv at this stage boundary\n")
after = {14: 'b_network', 15: 'c_extlib', 17: 'd_p5', 18: 'e_fr', 19: 'f_aprior', 20: 'g_uniq', 21: 'h_npx', 22: 'i_e180x11'}
plan = []
for i, c in enumerate(C):
    if i == 3:
        continue
    if i == 14:
        plan.append(snap('a_pre_network'))
    plan.append(c)
    if i == 1:
        plan.append(mk(builder)); plan.append(mk(preflight))
    if i in after:
        plan.append(snap(after[i]))
plan += [mk(scorer), mk(cleanup)]
md = dict(cell_type='markdown', metadata={}, source=lines(
    '# casmi26-vh2-gnx11: private validation harness v2 (never submitted)\n\n'
    'gnx11 (the 0.505 line; public sources credited in the cells below) run in full mode on 300 held-out training molecules, with the main engine on the fold-safe FPNet fold0 model (VH_CLEAN_MAIN): '
    '150 natural products without any external reference spectrum (identity fold 0), the 35 enveda-np-examples without external '
    'reference, 115 enveda-180 controls, all from the fold that model never saw; every train row of those structures removed. Snapshots in /kaggle/working/vh/.'))
nb['cells'] = [md] + plan
for c in nb['cells']:
    if c['cell_type'] == 'code':
        compile(src(c), NAME, 'exec'); c['outputs'] = []; c['execution_count'] = None
orig = {src(c) for i, c in enumerate(G) if i not in (3, 4)}
assert orig <= {src(c) for c in nb['cells'] if c['cell_type'] == 'code'}, 'a stage cell changed'
meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json'), encoding='utf-8'))
meta.update(id='huseyinemreaksoy/' + NAME, title=NAME, code_file=NAME + '.ipynb', is_private=True)
assert meta['enable_internet'] is False and meta['enable_gpu'] is True
d = os.path.join(HERE, 'kernel'); os.makedirs(d, exist_ok=True); out = os.path.join(d, NAME + '.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), ensure_ascii=False, separators=(',', ':'))
json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8', newline='\n'), indent=1)
size = os.path.getsize(out); assert size < 1_048_576, size
for c in json.load(open(out, encoding='utf-8'))['cells']:
    if c['cell_type'] == 'code':
        compile(src(c), NAME, 'exec')
print(out, 'code cells', sum(c['cell_type'] == 'code' for c in nb['cells']), 'bytes', size, 'frozen sha', sha0[:12], '->', sha2[:12])
