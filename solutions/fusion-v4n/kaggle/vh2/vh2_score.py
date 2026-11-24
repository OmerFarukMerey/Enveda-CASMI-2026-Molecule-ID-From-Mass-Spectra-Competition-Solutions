"""Go / no-go table for a harness run (casmi26-vh2-gnx11 clean main engine, or v1): per group, the DUEL family (exact cell
cell_duel2.py, every mode), DUEL variants with extra gates, MAINLEAD, stage deltas from the snapshots and the two stage ablations
that replay exactly offline (NPX off, APRIOR off). One table sorted by the delta on the first group; FLAG = non-negative on every
group with up > 2 x down on the first group.
usage: python -I scratch/land1123/vh2_score.py [run_dir=scratch/land1123/vh2_out] [--train reduced_train.parquet]
       [--gnx11-chain --npx-train file]   (v1 outputs: rebuild the gnx11 final list g_uniq -> NPX -> E180X 1.1 first)"""
import argparse, csv, json, os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'solutions', 'fusion-v4n', 'tools'))
from metric import ik14 as mkey
ap = argparse.ArgumentParser()
ap.add_argument('run', nargs='?', default=os.path.join(HERE, 'vh2_out'))
ap.add_argument('--train', default=os.path.join(HERE, 'vh2_dry', 'comp', 'train.parquet'))
ap.add_argument('--gnx11-chain', action='store_true')
ap.add_argument('--npx-train', default=None)
a = ap.parse_args()
INP = os.path.join(REPO, 'data', 'input')
V = os.path.join(a.run, 'vh') if os.path.isdir(os.path.join(a.run, 'vh')) else a.run
cell = lambda p: (lambda g: (exec(compile(open(p, encoding='utf-8').read(), p, 'exec'), g), g)[1])({})
DU = cell(os.path.join(HERE, 'cell_duel2.py')); E11 = cell(os.path.join(HERE, 'cells_g', 'c23.py')); NPX = cell(os.path.join(HERE, 'cells_k', 'c21.py'))
AP = cell(os.path.join(HERE, 'cells', 'c20.py'))
ans = pd.read_csv(os.path.join(V, 'answers.csv'), dtype=str)
T = {r.molecule_id: mkey(r.smiles) for r in ans.itertuples()}; G = dict(zip(ans.molecule_id, ans.group))
GROUPS = [g for g in ('nphard', 'np', 'npref0', 'e180') if g in set(G.values())] + sorted(set(G.values()) - {'nphard', 'np', 'npref0', 'e180'})
read = lambda f: {r['molecule_id']: r['smiles'].split(';')[:25] for r in csv.DictReader(open(os.path.join(V, f), encoding='utf-8'))}
snapf = sorted(f for f in os.listdir(V) if f.startswith('sub_') and f.endswith('.csv'))
S = {f[4:-4]: read(f) for f in snapf}
evp = next(p for p in (os.path.join(V, 'frozen_candidate_evidence.parquet'), os.path.join(a.run, 'frozen_candidate_evidence.parquet')) if os.path.exists(p))
ev = pd.read_parquet(evp); ev['molecule_id'] = ev.molecule_id.astype(str)
main_key = ev[ev.main_rank == 1].drop_duplicates('molecule_id').set_index('molecule_id').metric_key.to_dict()
pre_key = ev[ev.prefusion_rank == 1].drop_duplicates('molecule_id').set_index('molecule_id').metric_key.to_dict()
lib = ev.groupby('molecule_id').library_max.first().to_dict()
EVS = {(r.molecule_id, r.smiles): r for r in ev.itertuples()}
if not os.path.exists(a.train):
    # the reduced train the kernel saw: rebuild it deterministically with the kernel's own builder cell (about 1 min, 4 GB)
    print('reduced train missing: rebuilding', a.train, flush=True)
    import json as _j
    from pathlib import Path as _P
    _nb = _j.load(open(os.path.join(HERE, 'sub', 'casmi26-vh2-gnx11', 'casmi26-vh2-gnx11.ipynb'), encoding='utf-8'))
    _b = [''.join(c['source']) for c in _nb['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).startswith('# ===== VH (Arsokan')][0]
    _g = {}; exec(compile(_b.split(chr(10) + "if 'COMP' in globals()")[0], 'vh_builder', 'exec'), _g)
    _out = _P(a.train).parent.parent
    _g['vh_build'](_P(INP) / 'competitions' / 'enveda-CASMI26-molecule-id-mass-spectra', _P(a.train).parent, _out / 'vh', _g['VH_SEL'])
    _chk = pd.read_csv(_out / 'vh' / 'answers.csv', dtype=str).set_index('molecule_id').inchikey14
    assert _chk.to_dict() == ans.set_index('molecule_id').inchikey14.to_dict(), 'rebuilt selection differs from the run'
A11, B11, _ = E11['ex_sets'](a.train, INP); k11 = E11['ex_keyer']()
tierA = lambda s: k11(s) in A11
mk = DU['du_metric_keyer']()


def perm_lists(L, fn):
    return {m: fn(m, c) for m, c in L.items()}


def e11(c):
    ks = [k11(s) for s in c]
    return E11['ex_reorder'](list(c), ks, A11, B11, True)


npx_model = None
def npx(c):
    global npx_model
    if npx_model is None:
        npx_model = NPX['npx_train'](a.npx_train or a.train, INP)
    model, feats, table, _ = npx_model
    from rdkit import Chem
    p = []
    for s in c:
        f = feats(s); p.append(None if f is None else float(model.predict_proba(f[None])[0, 1]))
    def pk(s):
        m = Chem.MolFromSmiles(s); return Chem.MolToInchiKey(m)[:14] if m else None
    return NPX['npx_reorder'](list(c), p, [pk(s) in table for s in c])


if 'i_e180x11' in S:
    FINAL = S['i_e180x11']; fin_name = 'i_e180x11'
elif a.gnx11_chain:
    FINAL = {m: e11(npx(c)) for m, c in S['g_uniq'].items()}; fin_name = 'gnx11 chain rebuilt from g_uniq'
else:
    fin_name = snapf[-1][4:-4]; FINAL = S[fin_name]
print('final list:', fin_name, '| snapshots:', list(S))
rr = lambda c, t: (1.0 / ([mkey(x) for x in c].index(t) + 1)) if t in [mkey(x) for x in c] else 0.0
BASE = {m: rr(c, T[m]) for m, c in FINAL.items()}
rows = []


def score(name, L, ref=None, ref_lists=None):
    ref = ref or BASE; ref_lists = ref_lists or FINAL
    r = dict(variant=name)
    for g in GROUPS:
        ms = [m for m in FINAL if G.get(m) == g]
        b = np.array([rr(L[m], T[m]) for m in ms]); a0 = np.array([ref[m] for m in ms]); d = b - a0
        r[g] = f'{d.mean():+.4f} ({int((d > 0).sum())}/{int((d < 0).sum())}, {100 * np.mean([L[m] != ref_lists[m] for m in ms]):.0f}%)'
        r[g + '_d'], r[g + '_up'], r[g + '_down'] = d.mean(), int((d > 0).sum()), int((d < 0).sum())
    rows.append(r)


for mode in ('DUEL', 'AGREE', 'TOP3', 'MAINLEAD'):
    out, _ = DU['du_lists'](FINAL, main_key, pre_key, lib, mk, tierA, mode)
    score(f'{mode} (exact cell)', out)
out, _ = DU['du_lists'](FINAL, pre_key, pre_key, lib, mk, tierA, 'DUEL')
score('DUEL-PRE (prefusion leader at rank 2)', out)
ungated = {m: ([x for x in c if mk(x) == main_key.get(m)][:1] + [x for x in c if mk(x) != main_key.get(m)]) if main_key.get(m) in [mk(x) for x in c[1:]] else c for m, c in FINAL.items()}
score('MAINLEAD ungated (no formula / library gate)', ungated)
duel, _ = DU['du_lists'](FINAL, main_key, pre_key, lib, mk, tierA, 'DUEL')
def gate(fn):
    return {m: (duel[m] if duel[m] != FINAL[m] and fn(m, FINAL[m]) else FINAL[m]) for m in FINAL}
def fwd_agree(m, c):
    a_, b_ = EVS.get((m, c[0])), EVS.get((m, c[1]))
    if a_ is None or b_ is None:
        return False
    return any(np.isfinite(getattr(b_, x)) and np.isfinite(getattr(a_, x)) and getattr(b_, x) > getattr(a_, x) for x in ('ice', 'gl'))
score('DUEL + ice or gl agrees', gate(fwd_agree))
score('DUEL + library_max < 0.5', gate(lambda m, c: lib.get(m, 1) < 0.5))
if 'a_pre_network' in S:
    score('DUEL + leader installed after the network snapshot', gate(lambda m, c: mk(c[0]) != mk(S['a_pre_network'][m][0])))
# stage deltas (measured) and exact ablations
names = list(S)
for x, y in zip(names[:-1], names[1:]):
    score(f'stage {y} (measured: {x} -> {y})', S[y], {m: rr(c, T[m]) for m, c in S[x].items()}, S[x])
if 'g_uniq' in S and 'h_npx' in S:
    score('ablation: NPX off (E180X 1.1 on g_uniq; vs final)', {m: e11(c) for m, c in S['g_uniq'].items()})
    ap_m = None
    try:
        te = pd.read_parquet(os.path.join(V, 'test.parquet'), columns=['molecule_id', 'adduct'])
        seen = te.groupby('molecule_id')['adduct'].agg(lambda x: set(map(str, x))).to_dict()
        uq = {m for m in FINAL if S['f_aprior'][m] != S['g_uniq'][m]}
        score('ablation: APRIOR off (e_fr -> [UNIQ kept] -> NPX -> E180X 1.1)',
              {m: (e11(npx(S['g_uniq'][m])) if m in uq else e11(npx(S['e_fr'][m]))) for m in FINAL})
    except Exception as e:
        print('APRIOR-off ablation skipped:', repr(e)[:200])
print('not replayable offline (model stages): network, extlib, P5, FR -> their measured stage deltas are shown instead')
R = pd.DataFrame(rows)
g0 = GROUPS[0]
R['FLAG'] = [('*' if all(r[g + '_d'] >= 0 for g in GROUPS) and r[g0 + '_up'] > 2 * r[g0 + '_down'] else '') for _, r in R.iterrows()]
R = R.sort_values(g0 + '_d', ascending=False)
pd.set_option('display.width', 260); pd.set_option('display.max_colwidth', 60)
print(f'base (final list) MRR: ' + ', '.join(f'{g} {np.mean([BASE[m] for m in FINAL if G.get(m) == g]):.4f} (n {sum(G.get(m) == g for m in FINAL)})' for g in GROUPS))
print('main-engine leader correct: ' + ', '.join(f'{g} {sum(main_key.get(m) == T[m] for m in FINAL if G.get(m) == g)}/{sum(G.get(m) == g for m in FINAL)}' for g in GROUPS))
print(R[['variant'] + GROUPS + ['FLAG']].to_string(index=False))
R.to_csv(os.path.join(a.run, 'vh2_score_table.csv'), index=False)
