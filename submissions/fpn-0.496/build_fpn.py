"""fpn: the team's three public-0.496 kernels merged (Ömer's fp5 + p12 + n5) and made NIST-free.
fp5 = taeyangg4's 0.495 notebook with FR 3-25 moved after the P5 tail; p12 = P5 runner with 12 parents / 300 products;
n5 = NetGen (network neighbours' top-2 guesses as extra P5 generation parents). Base = n5f; p12's runner edit is applied
on top (the two edits touch different lines of p5_generate); the external-library overlay drops the 117
GNPS-NIST14-MATCHES rows exactly as in tp5. usage: python -I scratch/land1120/build_fpn.py"""
import ast, copy, glob, hashlib, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); NAME = 'casmi26-sub-fpn'
src = lambda c: c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
load = lambda d: json.load(open(glob.glob(os.path.join(HERE, d, '*.ipynb'))[0], encoding='utf-8'))
codes = lambda nb: [c for c in nb['cells'] if c['cell_type'] == 'code']
sha = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()
fp5, p12, n5 = load('om/fp5'), load('om/p12f'), load('om/n5f')
F, P, N = codes(fp5), codes(p12), codes(n5)
assert len(F) == len(P) == len(N) == 19
# only cell 17 (P5 tail) differs between the three
assert [i for i in range(19) if src(F[i]) != src(P[i])] == [17] and [i for i in range(19) if src(F[i]) != src(N[i])] == [17]

def split(s):
    """cell 17 -> (runner text, its pinned sha, lines of the cell)"""
    lines = s.splitlines(keepends=True)
    r = [l for l in lines if l.startswith('_P5_RUNNER = ')]; h = [l for l in lines if l.startswith('_P5_RUNNER_SHA256 = ')]
    assert len(r) == len(h) == 1
    return ast.literal_eval(r[0][len('_P5_RUNNER = '):].rstrip('\n')), ast.literal_eval(h[0][len('_P5_RUNNER_SHA256 = '):].rstrip('\n'))

rF, hF = split(src(F[17])); rP, hP = split(src(P[17])); rN, hN = split(src(N[17]))
for r, h in ((rF, hF), (rP, hP), (rN, hN)):
    assert sha(r) == h, 'pinned sha is not sha256(utf-8 runner)'
# p12's two runner edits, re-applied on n5's runner
A0 = "VERSION = '1.0.0'\n"; A1 = A0 + "P12_N, P12_TOTAL = 12, 300   # Arsokan p12\n"
B0 = "        res = orig_generate(new, target, exclude_keys)\n"
B1 = ("        # Arsokan p12: up to P12_N transformable-first parents and P12_TOTAL products for this call only; the engine\n"
      "        # config is restored right after, so cfg0 and the P6 parent set (first 6 eligible analogs) are unchanged.\n"
      "        p5 = [int(a[0]) for a in new if a[1] >= ecfg.gen_min_sim][:P12_N]\n"
      "        _n0, _t0 = ecfg.gen_n_analog, ecfg.gen_max_total\n"
      "        ecfg.gen_n_analog, ecfg.gen_max_total = P12_N, P12_TOTAL\n"
      "        try:\n"
      "            res = orig_generate(new, target, exclude_keys)\n"
      "        finally:\n"
      "            ecfg.gen_n_analog, ecfg.gen_max_total = _n0, _t0\n")
assert rF.count(A0) == 1 and rF.count(B0) == 1 and rF.replace(A0, A1).replace(B0, B1) == rP, 'p12 edit not reproduced'
assert rN.count(A0) == 1 and rN.count(B0) == 1
rM = rN.replace(A0, A1).replace(B0, B1); compile(rM, 'runner', 'exec'); hM = sha(rM)

nb = copy.deepcopy(n5); C = codes(nb)
s = src(C[17]).replace('_P5_RUNNER = ' + repr(rN), '_P5_RUNNER = ' + repr(rM)).replace(repr(hN), repr(hM))
assert split(s) == (rM, hM)
s = ("# ===== Arsokan fpn: n5 (NetGen) and p12 (12 parents / 300 products) together in one P5 runner; runner sha256 re-pinned.\n" + s)
compile(s, 'cell17', 'exec'); C[17]['source'] = s.splitlines(keepends=True)
# NIST-free external library (same patch as tp5)
OLD = "            for idx, row in meta_df.iterrows():\n                try:\n                    mode = 1 if 'pos'"
NEW = ("            # ours: drop NIST-matched rows (index kept, so mz_all[idx] stays aligned)\n"
       "            if 'source_library' in meta_df.columns:\n"
       "                _nist = meta_df['source_library'].astype(str).str.contains('NIST', case=False)\n"
       "                print(f'[apex-extlib] dropped {int(_nist.sum())} NIST-matched reference rows')\n"
       "                meta_df = meta_df[~_nist]\n" + OLD)
o = src(C[15]); assert 'def _apex_extlib_overlay' in o and o.count(OLD) == 1
o = o.replace(OLD, NEW); compile(o, 'overlay', 'exec'); C[15]['source'] = o.splitlines(keepends=True)
ch = [i for i in range(19) if src(C[i]) != src(codes(n5)[i])]; assert ch == [15, 17], ch
for c in nb['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
md = nb['cells'][0]; t = src(md)
t = t.replace('casmi26-om-n5f', NAME, 1)
t += ("\n**fpn (Hemre):** Ömer's three public-0.496 kernels in one: fp5 (FR 3-25 after the P5 tail), p12 (P5 runner with 12 "
      "parents / 300 products) and n5 (NetGen). The external-library overlay drops the 117 GNPS-NIST14-MATCHES rows, "
      "because our team uses no NIST-derived data (as in casmi26-sub-tp5, which scored the same 0.495 as the source).\n")
md['source'] = t.splitlines(keepends=True)
meta = json.load(open(os.path.join(HERE, 'om/n5f/kernel-metadata.json'), encoding='utf-8'))
meta = {k: v for k, v in meta.items() if k != 'id_no'}
meta.update(id='huseyinemreaksoy/' + NAME, title=NAME, code_file=NAME + '.ipynb', is_private=True)
assert meta['enable_internet'] is False and meta['enable_gpu'] is True and all(meta['dataset_sources']) and all(meta['kernel_sources'])
d = os.path.join(HERE, 'sub', NAME); os.makedirs(d, exist_ok=True); out = os.path.join(d, NAME + '.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8', newline='\n'), ensure_ascii=False, indent=1)
json.dump(meta, open(os.path.join(d, 'kernel-metadata.json'), 'w', encoding='utf-8'), indent=1)
print(out, 'changed cells vs n5f', ch, 'runner sha', hM[:16], 'nb sha', hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16])
