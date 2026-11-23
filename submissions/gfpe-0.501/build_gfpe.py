"""gfpe: Ömer's gfp5 (0.500) + goodpjw2008's E180X cell (public 0.501: demotes train-library structures that are not
natural products, run last) + our fpn P5 cell (12 parents / 300 products + NetGen network parents; fpn 0.496).
GNPS-NIST14-MATCHES rows kept (allowed by name by the host, thread 745832). Optional EXTRA cells for variants.
usage: python -I scratch/land1122/build_gfpe.py [name] [variant]"""
import copy, glob, hashlib, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
NAME = sys.argv[1] if len(sys.argv) > 1 else 'casmi26-sub-gfpe'; VAR = sys.argv[2] if len(sys.argv) > 2 else ''
src = lambda c: c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
codes = lambda nb: [c for c in nb['cells'] if c['cell_type'] == 'code']
setsrc = lambda c, s: c.__setitem__('source', s.splitlines(keepends=True))
L = lambda p: json.load(open(glob.glob(p)[0], encoding='utf-8'))
g = L(os.path.join(HERE, 'gfp5/notebook.ipynb')); gp = L(os.path.join(HERE, 'pull/enveda-casmi-stack-aprior-uniq-lb-0-498/*.ipynb'))
kr = L(os.path.join(HERE, 'pull/casmi26-with-180x/*.ipynb'))
fp5 = L(os.path.join(ROOT, 'land1120/om/fp5/*.ipynb')); fpn = L(os.path.join(ROOT, 'land1120/sub/casmi26-sub-fpn/*.ipynb'))
G, GP, KR, F, N = codes(g), codes(gp), codes(kr), codes(fp5), codes(fpn)
# goodpjw2008 v5 = gfp5 code cells 1-21 byte for byte + E180X as the last cell (cell 0 is its figure); koushikrudra's 0.501 is the same
assert len(G) == 22 and len(GP) == 23 and all(src(GP[i]) == src(G[i]) for i in range(1, 22)) and 'Ours: E180X' in src(GP[22])
assert all(src(KR[i]) == src(G[i]) for i in range(22)) and src(KR[22]) == src(GP[22])
assert src(G[18]) == src(F[17]) and src(N[18]) == src(F[18])
nb = copy.deepcopy(g); C = codes(nb)
setsrc(C[18], src(N[17]))                                   # fpn P5 cell
e = copy.deepcopy(gp['cells'][gp['cells'].index(GP[22])])   # E180X, appended last (after APRIOR and UNIQ)
nb['cells'].append(e)
note = ("\n\n**casmi26-sub-gfpe (team Arsokan, Hemre):** this notebook + goodpjw2008's E180X cell "
        "([Stack + E180X + APRIOR + UNIQ, LB 0.501](https://www.kaggle.com/code/goodpjw2008/enveda-casmi-stack-aprior-uniq-lb-0-498), "
        "Apache-2.0) as the last cell, and our P5 tail cell with 12 parents / 300 products plus NetGen network parents (fpn, 0.496).")
if VAR == 'g90':   # Ömer's g90 network gate (fpg), on the network cell 15
    cg = L(os.path.join(ROOT, 'land1120/om/cg/*.ipynb')); CG = codes(cg); O = codes(L(os.path.join(ROOT, 'land1120/casmi26-lb-0-495-network-fr-p5-tail/*.ipynb')))
    assert src(C[15]) == src(O[14]); setsrc(C[15], src(CG[14])); note += " Network cell with Ömer's g90 gate (library_max >= 0.90 keeps the pre-network leader)."
C = codes(nb)
for c in C:
    compile(src(c), NAME, 'exec'); c['outputs'] = []; c['execution_count'] = None
md = [c for c in nb['cells'] if c['cell_type'] == 'markdown'][0]; setsrc(md, src(md) + note + "\n")
nb['cells'] = [c for c in nb['cells'] if c['cell_type'] == 'code' or c is md]   # size: keep only the first markdown cell
meta = json.load(open(os.path.join(HERE, 'gfp5/kernel-metadata.json'), encoding='utf-8'))
meta.update(id='huseyinemreaksoy/' + NAME, title=NAME, code_file=NAME + '.ipynb', is_private=True)
assert meta['enable_internet'] is False and meta['enable_gpu'] is True and all(meta['dataset_sources'])
d = os.path.join(HERE, 'sub', NAME); os.makedirs(d, exist_ok=True); out = os.path.join(d, NAME + '.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), ensure_ascii=False, separators=(',', ':'))  # compact: Kaggle rejects > 1 MiB
json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8', newline='\n'), indent=1)
ch = [i for i in range(22) if src(C[i]) != src(G[i])]
print(out, 'code cells', len(C), 'changed vs gfp5', ch, '+ E180X', 'sha', hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16])
