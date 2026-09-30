"""fusion_core.build_submission must reproduce the reference v4n final stage (reference/v4n/notebook_main_stubbed.txt,
lines 266-400) exactly, for the default CFG. Runs both on random stage caches with deterministic mock re-rankers.

usage: python tests/test_fusion_equivalence.py   (or pytest)
"""
import copy, json, math, os, random, sys, tempfile, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'nb', 'embed'))
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import fusion_core

REF = os.path.join(ROOT, 'reference', 'v4n', 'notebook_main_stubbed.txt')


def formula(s):
    try:
        return CalcMolFormula(Chem.MolFromSmiles(s))
    except Exception:
        return s


def score_key(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def _order(smis, scs, terms, lams, top_n):
    n = min(top_n, len(smis))
    def val(i):
        v = scs[i] if i < len(scs) else 0.0
        for t, l in zip(terms, lams):
            x = t.get(smis[i])
            v += l * (x if x is not None else 0.0)
        return v
    head = sorted(range(n), key=lambda i: (-val(i), i))
    return head + list(range(n, len(smis)))


ice_fuse = types.SimpleNamespace(rerank=lambda smis, keys, scs, forms, ice, lam, top_n: _order(smis, scs, [ice], [lam], top_n))
gl_fuse = types.SimpleNamespace(rerank_multi=lambda smis, keys, scs, forms, terms, lams, top_n: _order(smis, scs, terms, lams, top_n))


def make_case(seed, n_mol=40):
    rng = random.Random(seed)
    pool = sorted({s for n in range(1, 16) for s in ('C' * n, 'C' * n + 'O', 'C' * n + 'N', 'O' + 'C' * n + 'O', 'N' + 'C' * n + 'O')})
    mids = [f'm_{i:04x}' for i in range(n_mol)]
    BASE, PC, ENG, ICE, GL = {}, {}, {}, {}, {}
    for m in mids:
        k = rng.choice([0, 3, 30, 60])
        smis = rng.sample(pool, min(k, len(pool)))
        BASE[m] = (smis, [score_key(s) for s in smis], rng.choice([0.3, 0.85, 0.95, 1.0]),
                   sorted([rng.random() for _ in smis], reverse=True), [formula(s) for s in smis])
        if rng.random() < 0.8:
            pc = rng.sample(pool, rng.choice([0, 5, 25]))
            fz = sorted([rng.uniform(0, 2000) for _ in pc], reverse=True)
            PC[m] = dict(pc=pc, pc_fz=fz, pc_keys=[score_key(s) for s in pc],
                         best_pool_fz=rng.choice([float('nan'), rng.uniform(0, 1500), rng.uniform(1500, 2500)]), target=300.0)
        if rng.random() < 0.7:
            es = rng.sample(pool, rng.choice([0, 10, 40]))
            ENG[m] = dict(smiles=es, keys=[score_key(s) if rng.random() > 0.05 else None for s in es])
        if rng.random() < 0.7:
            cand = set(BASE[m][0]) | set(PC.get(m, {}).get('pc', [])) | set(ENG.get(m, {}).get('smiles', []))
            ICE[m] = {s: (rng.gauss(0, 1) if rng.random() > 0.1 else None) for s in cand}
            if rng.random() < 0.7:
                GL[m] = {s: (rng.gauss(0, 1) if rng.random() > 0.1 else None) for s in cand}
    return mids, BASE, PC, ENG, ICE, GL


def run_reference(mids, BASE, PC, ENG, ICE, GL, workdir):
    src = open(REF, encoding='utf-8').read().split('\n')
    code = '\n'.join(src[265:349] + src[353:400])        # lines 266-349 (rerank+merge) and 354-400 (fusion)
    code = code.replace("'submission.csv'", repr(os.path.join(workdir, 'submission.csv')))
    code = code.replace("open('run_manifest.json', 'w')", f"open({os.path.join(workdir, 'rm.json')!r}, 'w')")
    PC = copy.deepcopy(PC)
    for m, p in PC.items():                               # what the reference ICE cell adds before this point
        if p.get('pc') and BASE[m][2] < 0.9:
            p['pc_form'] = [formula(x) for x in p['pc']]
    pd.DataFrame({'molecule_id': mids, 'smiles': 'CCO'}).to_csv(os.path.join(workdir, 'sample_submission.csv'), index=False)
    ns = dict(np=np, pd=pd, os=os, json=json, time=__import__('time'), T0=0.0, COMP=workdir,
              BASE=copy.deepcopy(BASE), PC=PC, ENG=copy.deepcopy(ENG), ICE_SCORES=copy.deepcopy(ICE),
              GL_SCORES=copy.deepcopy(GL), ice_fuse=ice_fuse, gl_fuse=gl_fuse, ICE_LAM=1.0, GL_LAM=1.0, TOPN=60,
              ICE_PC=True, LIB_TAU=0.9, REL_TH=600.0, SLOTS_AGG=[2, 4, 6, 8, 10], SLOTS_GENTLE=[4, 8, 12, 16, 20],
              gl_stats={}, ice_stats=dict(molecules=0, changed_top25=0, changed_top1=0),
              te=pd.DataFrame({'molecule_id': mids}), chem=types.SimpleNamespace(score_key=score_key),
              MAN=dict(families=[]), n_err=0, dev='cpu', torch=None)
    exec(compile(code, REF, 'exec'), ns)
    return pd.read_csv(os.path.join(workdir, 'submission.csv'))


def run_ours(mids, BASE, PC, ENG, ICE, GL):
    base = {m: list(v) for m, v in BASE.items()}
    sub, _ = fusion_core.build_submission(json.loads(json.dumps(base)), json.loads(json.dumps(PC)), ENG, ICE, GL,
                                          {}, mids, mids, ice_fuse=ice_fuse, gl_fuse=gl_fuse, score_key=score_key,
                                          formula=formula, log=lambda *a: None)
    return sub


def test_equivalence():
    for seed in range(8):
        case = make_case(seed)
        with tempfile.TemporaryDirectory() as d:
            ref = run_reference(*case, d)
        ours = run_ours(*case)
        assert list(ref.molecule_id) == list(ours.molecule_id)
        diff = [(m, a, b) for m, a, b in zip(ref.molecule_id, ref.smiles, ours.smiles) if a != b]
        assert not diff, f'seed {seed}: {len(diff)} molecules differ, first: {diff[0]}'
        fusion_core.validate(ours, case[0])
    print('fusion_core == reference v4n final stage on 8 random cases')


if __name__ == '__main__':
    test_equivalence()
