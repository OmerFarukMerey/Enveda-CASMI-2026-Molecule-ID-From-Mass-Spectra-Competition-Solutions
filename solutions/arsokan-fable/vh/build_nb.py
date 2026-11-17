"""vh: build the private harness notebook = n0 notebook + data redirect + embedded held-out selection. argv: selection.json out.ipynb"""
import json, re, base64, hashlib, sys, copy
R='/home/user/Enveda-CASMI-2026-Molecule-ID-From-Mass-Spectra-Competition-Solutions/'
sel=json.load(open(sys.argv[1])); nb=json.load(open(R+'submissions/n0-0.492/notebook.ipynb'))
S=lambda c: ''.join(c['source']) if isinstance(c['source'],list) else c['source']
def setsrc(c,s): c['source']=s; c['outputs']=[]; c['execution_count']=None
cells=nb['cells']
def rep(s,old,new,count=1):
    assert s.count(old)==count, (s.count(old), old[:80]); return s.replace(old,new)
# 1. session cell: competition dir override
s=S(cells[3]); s=rep(s,"assert COMP is not None, 'Attach the competition data (enveda-CASMI26-molecule-id-mass-spectra)'\n",
 "assert COMP is not None, 'Attach the competition data (enveda-CASMI26-molecule-id-mass-spectra)'\n_HOV = os.environ.get('CASMI_COMP_DIR')          # VH: reduced train + held-out test\nif _HOV and all((Path(_HOV) / f).is_file() for f in ('train.parquet', 'test.parquet', 'sample_submission.csv')):\n    COMP = Path(_HOV).resolve()\nprint('VH COMP', COMP, flush=True)\n"); setsrc(cells[3],s)
# 2. frozen pipeline: the three path patches of the public fold0 harness (and nothing else)
s=S(cells[9]); b64=re.search(r"FROZEN_B64 = '([^']+)'",s).group(1); fz=json.loads(base64.b64decode(b64))
FS=lambda i: ''.join(fz['cells'][i]['source']) if isinstance(fz['cells'][i]['source'],list) else fz['cells'][i]['source']
def fset(i,t): fz['cells'][i]['source']=t
fset(4,rep(FS(4),'DATA = locate_data_dir()\n','_OV = os.environ.get("CASMI_DATA_DIR")\nDATA = (Path(_OV).resolve() if (_OV and all((Path(_OV) / n).exists() for n in\n        ("train.parquet", "test.parquet", "sample_submission.csv"))) else locate_data_dir())\n'))
fset(26,rep(FS(26),"COMP = os.path.dirname(find('test.parquet'))\n","COMP = os.path.dirname(find('test.parquet'))\n_HV = os.environ.get(\"CASMI_DATA_DIR\")      # VH: the reduced pool is the competition dir\nif _HV and os.path.isfile(os.path.join(_HV, \"test.parquet\")):\n    COMP = _HV\n"))
fset(28,rep(FS(28),"    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}\n","    assert (comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}) or os.environ.get('CASMI_DATA_DIR'), comp\n"))
raw=json.dumps(fz).encode(); nb64=base64.b64encode(raw).decode()
s=s.replace(b64,nb64); s=rep(s,re.search(r"FROZEN_SHA256 = '([^']+)'",s).group(1),hashlib.sha256(raw).hexdigest()); setsrc(cells[9],s)
# 3. N-1c and overlay: read the held-out test, not the visible one
s=S(cells[21]); s=rep(s,"    tp = sorted(glob.glob(inp + '/**/test.parquet', recursive=True), key=lambda p: ('mass-spectra' not in p.lower(), len(p)))\n","    tp = [str(COMP / 'test.parquet')]   # VH\n"); setsrc(cells[21],s)
s=S(cells[22]); s=rep(s,"    test_paths = sorted(glob.glob(inp + '/**/test.parquet', recursive=True),\n                       key=lambda p: ('mass-spectra' not in p.lower(), len(p)))\n","    test_paths = [str(COMP / 'test.parquet')]   # VH\n"); setsrc(cells[22],s)
# 4. builder cell (before the session cell) and scoring cell (last)
builder=open('cell_builder.py').read().replace('__SEL_JSON__',json.dumps(sel,separators=(',',':')))
scorer=open('cell_score.py').read()
mk=lambda src: dict(cell_type='code',metadata={},source=src,outputs=[],execution_count=None)
cells.insert(3,mk(builder)); cells.append(mk(scorer))
cells[0]=dict(cell_type='markdown',metadata={},source='# casmi-vh: private validation harness\n\nn0 (public 0.492 network-rerank notebook, credits in the cells below) run on held-out training molecules with a reduced train file. Not a submission notebook.')
for c in cells:
    if c['cell_type']=='code': c['outputs']=[]; c['execution_count']=None
json.dump(nb,open(sys.argv[2],'w')); print('cells',len(cells),'molecules',len(sel['mols']),'bytes',len(json.dumps(nb)))
