"""Evaluation sets of fpgen (DEV class N, set F, scaffold hold-out SH), the pool rows removed from every training set, and the leak assertions.

Removed from the generator's training universe: every pool row sharing a metric key, an InChIKey first block (also of up to 500 enumerated
tautomers) or a stereo-free canonical SMILES with an evaluation molecule; every row with the generic Murcko scaffold of an SH molecule;
every row sharing a key with a FINAL-half benchmark molecule (FINAL stays untouched). `leak_check()` prints the line every run stores.
out: $FPGEN_WORK/sets/{mols.parquet, removed.npy, leak.json}.   usage: python sets.py [--workers 5]
"""
import argparse, hashlib, json, os, pickle, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpaths as P, tok as T, chemrows as CR

SH_N, SH_PER_SCAF, SH_SCAF_CAP, SALT = 300, 3, 1000, 'fpgen-sh-1'
R_IDENT, R_SCAF, R_FINAL = 1, 2, 4
_W = {}


def _h(s):
    return hashlib.md5((SALT + str(s)).encode()).hexdigest()


def _init():
    CR._init()
    from rdkit.Chem.MolStandardize import rdMolStandardize
    te = rdMolStandardize.TautomerEnumerator(); te.SetMaxTautomers(500); te.SetMaxTransforms(500); _W['te'] = te


def ident(smiles):
    """Identity of an evaluation molecule: canonical stereo-free SMILES, IK14, generic scaffold, tautomer SMILES / IK14 sets, token count."""
    if 'te' not in _W:
        _init()
    Chem = CR._W['Chem']; m = Chem.MolFromSmiles(smiles); Chem.RemoveStereochemistry(m)
    can = Chem.MolToSmiles(m, isomericSmiles=False); ts = {can}; ti = {Chem.MolToInchiKey(m)[:14]}
    try:
        for x in _W['te'].Enumerate(m):
            ts.add(Chem.MolToSmiles(x, isomericSmiles=False)); ti.add(Chem.MolToInchiKey(x)[:14])
    except Exception:
        pass
    tk = T.tokenize(can)
    return dict(can=can, ik=Chem.MolToInchiKey(m)[:14], scaf=CR.scaffold(m), tsmi='|'.join(sorted(ts)), tik='|'.join(sorted(ti)),
                n_tok=len(tk) if tk else 0, usable=bool(tk) and T.usable(tk) and len(tk) <= CR.MAX_TOKENS)


def formulas_of(r, k=3):
    """The pipeline's first k distinct base-list formulas of a stored list record (as gen3's target_formulas)."""
    out = []
    for f in r['base_forms']:
        if isinstance(f, str) and f and f not in out:
            out.append(f)
            if len(out) >= k:
                break
    return out


def build(workers=5, log=print):
    import multiprocessing as mp
    P.mk(); t0 = time.time()
    PM = pd.read_parquet(P.WORK + '/pool_meta.parquet', columns=['key', 'formula'])
    PC = pd.read_parquet(P.CHEM + '/pool.parquet')
    assert len(PM) == len(PC) and (PC.rid.values == np.arange(len(PM))).all()
    ES = pd.read_parquet(P.G2 + '/eval_set.parquet'); FS = pd.read_parquet(P.G2 + '/eval_fair.parquet')
    B = pd.read_parquet(P.BENCH + '/mols.parquet')
    L = pickle.load(open(P.G3 + '/lists.pkl', 'rb')); rec = {(r['set'], r['view'], r['mid']): r for r in L}
    zdev = set(np.load(P.G2 + '/z_dev.npz')['mid'].tolist())
    scaf_n = PC.scaf[PC.scaf != ''].value_counts()
    with mp.Pool(workers, initializer=_init) as pool:
        cand = ES[(ES.nat != 'N') & (ES.np_catalogue | ES.panel) & ES.mid.isin(zdev)].copy()
        cand = cand[[('dev', 'c2J', m) in rec for m in cand.mid]].reset_index(drop=True)
        ci = pool.map(ident, list(cand.smiles), chunksize=20)
        cand['scaf'] = [x['scaf'] for x in ci]; cand['usable'] = [x['usable'] for x in ci]
        elig = cand[(cand.scaf != '') & cand.usable & (cand.scaf.map(scaf_n).fillna(0) <= SH_SCAF_CAP)]
        pick = []
        for s in sorted(elig.scaf.unique(), key=_h):
            g = elig[elig.scaf == s]
            pick += list(g.mid.values[np.argsort([_h(m) for m in g.mid])][:SH_PER_SCAF])
            if len(pick) >= SH_N:
                break
        SH = cand[cand.mid.isin(pick[:SH_N])]
        log('SH: %d DEV database natural products with logits and a list, %d eligible, %d picked, %d scaffolds'
            % (len(cand), len(elig), len(SH), SH.scaf.nunique()), flush=True)
        M = pd.concat([ES[ES.nat == 'N'][['mid', 'smiles', 'formula', 'key']].assign(eset='N'),
                       FS[['mid', 'smiles', 'formula', 'key']].assign(eset='F'),
                       SH[['mid', 'smiles', 'formula', 'key']].assign(eset='SH')], ignore_index=True)
        n0 = len(M); M = M.drop_duplicates('key', keep='first').reset_index(drop=True)      # one structure of F is also a class-N molecule: it stays in N
        log('evaluation molecules: %d, after dropping structures listed in two sets: %d' % (n0, len(M)), flush=True)
        assert M.mid.is_unique and M.key.is_unique, 'an evaluation molecule appears twice'
        idn = pool.map(ident, list(M.smiles), chunksize=10)
    for k in idn[0]:
        M[k] = [x[k] for x in idn]
    fml = []
    for r in M.itertuples():
        q = rec.get(('dev', 'N', r.mid)) or rec.get(('F', 'F', r.mid)) or rec.get(('dev', 'c2J', r.mid))
        fml.append('|'.join(formulas_of(q)) if q else '')
    M['forms3'] = fml; M['has_list'] = [bool(x) for x in fml]
    tsmi = set(); tik = set()
    for r in M.itertuples():
        tsmi |= set(r.tsmi.split('|')); tik |= set(r.tik.split('|'))
    rm = np.zeros(len(PM), np.int8)
    rm[np.isin(PM.key.values, list(set(M.key))) | np.isin(PC.ik.values, list(tik)) | np.isin(PC.can.values, list(tsmi))] |= R_IDENT
    rm[PC.scaf.isin(set(M.scaf[(M.eset == 'SH') & (M.scaf != '')])).values] |= R_SCAF
    rm[np.isin(PM.key.values, list(set(B.key[B.half == 'final'])))] |= R_FINAL
    np.save(P.SETS + '/removed.npy', rm); M.to_parquet(P.SETS + '/mols.parquet', index=False)
    info = dict(built=time.strftime('%Y-%m-%d %H:%M'), pool=len(PM), eval_molecules={k: int(v) for k, v in M.eset.value_counts().items()},
                sh_scaffolds=int(M.scaf[M.eset == 'SH'].nunique()), removed_total=int((rm > 0).sum()),
                removed_identity=int(((rm & R_IDENT) > 0).sum()), removed_scaffold=int(((rm & R_SCAF) > 0).sum()),
                removed_final_key=int(((rm & R_FINAL) > 0).sum()), with_pipeline_formulas=int(M.has_list.sum()),
                usable_tokens=int(M.usable.sum()), wall_s=round(time.time() - t0))
    json.dump(info, open(P.SETS + '/leak.json', 'w'), indent=1); log(json.dumps(info), flush=True)
    return M, rm


def load():
    return pd.read_parquet(P.SETS + '/mols.parquet'), np.load(P.SETS + '/removed.npy')


def leak_check(train_ids, log=print, extra=''):
    """Assert that no removed pool row is in the training set and print the line every run stores."""
    M, rm = load(); t = np.asarray(train_ids); bad = int((rm[t] > 0).sum())
    line = ('leak check%s: %d training structures, %d pool rows removed (%d identity or tautomer of an evaluation molecule, '
            '%d scaffold of a hold-out molecule, %d FINAL-half key); removed rows among the training structures: %d; evaluation molecules: %s'
            % (extra, len(t), int((rm > 0).sum()), int(((rm & R_IDENT) > 0).sum()), int(((rm & R_SCAF) > 0).sum()),
               int(((rm & R_FINAL) > 0).sum()), bad, ', '.join('%s %d' % (k, v) for k, v in M.eset.value_counts().items())))
    log(line, flush=True)
    assert bad == 0, line
    return line


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=5); a = ap.parse_args()
    P.low_priority(); build(a.workers)
