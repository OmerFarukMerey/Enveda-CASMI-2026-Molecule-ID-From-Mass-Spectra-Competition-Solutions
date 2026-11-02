"""Isomer ranking sets from train.parquet for the fragment-explanation reranker experiment.

Each set: one enveda-180 molecule (its spectra from enveda-180 rows, up to 6, random), the true structure and up to 15
same-formula decoys from the train molecule table: the 10 most Tanimoto-similar (Morgan r2, 2048) and 5 random others.
Writes sets.parquet (one row per candidate, with features) to OUT. Scaffold fold of the query from its Murcko scaffold.
usage: build_sets.py OUT N_QUERIES [SEED]
"""
import sys, hashlib, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, str(Path(__file__).parent))
import fragfeat as F

CACHE = Path('C:/Users/omerm/Desktop/Kaggle/casmi-local-work/cache')
TRAIN = Path('C:/Users/omerm/Desktop/Kaggle/enveda-casmi-2026/data/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet')
N_SIM, N_RAND, MAX_SPEC = 10, 5, 6
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


from functools import lru_cache


@lru_cache(maxsize=None)
def fp(s):
    m = Chem.MolFromSmiles(s)
    return _gen.GetFingerprint(m) if m is not None else None


def work(job):
    qkey, spectra, cands = job
    lv = [F.frag_levels(s) for s in cands]
    X = F.molecule_features(spectra, lv)
    return qkey, X


def main():
    out = Path(sys.argv[1]); nq = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    mol = pd.read_parquet(CACHE / 'molecules.parquet')
    env = mol[mol.libs.str.contains('enveda-180')]
    fcount = mol.formula.value_counts()
    env = env[env.formula.map(fcount) >= 6]
    q = env.sample(nq, random_state=seed)
    print('queries', len(q), flush=True)
    # spectra of the queries (enveda-180 rows only)
    t0 = time.time()
    tab = pq.read_table(TRAIN, columns=['ingest_lib', 'inchikey14', 'adduct', 'ionization_mode', 'ms2_mzs', 'ms2_normalized_intensities',
                                        'collision_energy_ev', 'num_peaks'],
                        filters=[('ingest_lib', '=', 'enveda-180')])
    tab = tab.filter(pc.is_in(tab['inchikey14'], value_set=__import__('pyarrow').array(q.key.tolist())))
    sp = tab.to_pandas()
    sp = sp[sp.num_peaks >= 3]
    print('spectra rows', len(sp), f'{time.time() - t0:.0f}s', flush=True)
    by_key = {k: g for k, g in sp.groupby('inchikey14')}
    by_formula = {f: g for f, g in mol.groupby('formula')}
    jobs, meta = [], []
    for r in q.itertuples(index=False):
        g = by_key.get(r.key)
        if g is None or not len(g):
            continue
        g = g.sample(min(MAX_SPEC, len(g)), random_state=int(rng.integers(1 << 30)))
        spectra = [dict(mz=x.ms2_mzs, it=x.ms2_normalized_intensities, adduct=x.adduct, mode=x.ionization_mode) for x in g.itertuples(index=False)]
        iso = by_formula[r.formula]
        iso = iso[iso.key != r.key]
        tf = fp(r.smiles)
        if tf is None or len(iso) < 5:
            continue
        fps = [fp(s) for s in iso.smiles]
        ok = [i for i, v in enumerate(fps) if v is not None]
        sims = np.array(DataStructs.BulkTanimotoSimilarity(tf, [fps[i] for i in ok]))
        order = [ok[i] for i in np.argsort(-sims)]
        top = order[:N_SIM]
        rest = order[N_SIM:]
        rnd = list(rng.choice(rest, size=min(N_RAND, len(rest)), replace=False)) if rest else []
        pick = top + rnd
        simmap = dict(zip(ok, sims))
        cands = [r.smiles] + [iso.smiles.iloc[i] for i in pick]
        keys = [r.key] + [iso.key.iloc[i] for i in pick]
        sim = [1.0] + [float(simmap[i]) for i in pick]
        kind = ['true'] + ['sim'] * len(top) + ['rand'] * len(rnd)
        jobs.append((r.key, spectra, cands))
        meta.append(dict(qkey=r.key, scaffold=r.scaffold, keys=keys, smiles=cands, sim=sim, kind=kind,
                         n_spec=len(spectra), adducts='|'.join(sorted({s['adduct'] for s in spectra}))))
    print('sets', len(jobs), f'{time.time() - t0:.0f}s', flush=True)
    with Pool(28) as pool:
        res = dict(pool.imap_unordered(work, jobs, chunksize=8))
    rows = []
    for m in meta:
        X = res[m['qkey']]
        fold = int(hashlib.sha256(('20261023fold' + str(m['scaffold'])).encode()).hexdigest(), 16) % 5
        for c in range(len(m['keys'])):
            rows.append(dict(qkey=m['qkey'], fold=fold, cand=m['keys'][c], smiles=m['smiles'][c], label=int(c == 0), sim=m['sim'][c],
                             kind=m['kind'][c], n_spec=m['n_spec'], adducts=m['adducts'], **dict(zip(F.FEATURES, X[c]))))
    df = pd.DataFrame(rows)
    df.to_parquet(out / 'sets.parquet')
    print('wrote', len(df), 'rows', df.qkey.nunique(), 'sets', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
