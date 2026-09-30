"""Compute MetFrag-parsimony scores (nb/embed/frag_rescore.py) for every candidate of a run's stage cache.

Candidates = base lists ∪ engine-2 lists ∪ PubChem lists; spectra come from the run's test.parquet (for VALIDATION
runs, rebuild it locally with the same VAL_* settings; build_validation is deterministic). Output:
<stage_cache>/frag.json = {molecule_id: {smiles: score | None}}, which tools/refuse.py picks up (knob FRAG_LAM).

usage: python tools/frag_scores.py <run_dir> <test.parquet> [--workers 4]
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nb', 'embed'))


def _one(args):
    import frag_rescore
    mid, spectra, cands = args
    return mid, dict(zip(cands, frag_rescore.score_molecule(spectra, cands)))


def molecule_spectra(te):
    out = {}
    for mid, sub in te.groupby('molecule_id', sort=False):
        out[mid] = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                         adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1)
                    for r in sub.itertuples()][:16]
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('test'); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    sc = next(d for d in glob.glob(os.path.join(a.run, '**', 'stage_cache'), recursive=True))
    ld = lambda n: json.load(open(os.path.join(sc, f'{n}.json'))) if os.path.exists(os.path.join(sc, f'{n}.json')) else {}
    base, eng, pc = ld('base'), ld('eng'), ld('pc')
    spec = molecule_spectra(pd.read_parquet(a.test))
    missing = [m for m in base if m not in spec]
    assert not missing, f'{len(missing)} run molecules missing from {a.test} (wrong test file?) e.g. {missing[:3]}'
    tasks = []
    for m in base:
        c = list(dict.fromkeys(list(base[m][0]) + list(eng.get(m, {}).get('smiles', [])) + list((pc.get(m) or {}).get('pc', []))))
        tasks.append((m, spec[m], c))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(_one, tasks, chunksize=4))
    json.dump(res, open(os.path.join(sc, 'frag.json'), 'w'))
    n = sum(len(v) for v in res.values())
    print(f'frag scores: {len(res)} molecules, {n} candidates, {time.time()-t0:.0f}s -> {sc}/frag.json')
