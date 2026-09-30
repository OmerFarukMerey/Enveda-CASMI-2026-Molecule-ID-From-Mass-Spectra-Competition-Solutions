"""Local MRR@25, same rules as the official metric (kaggle.com/code/metric/casmi-mean-reciprocal-rank):
tautomer-canonical InChIKey14 match, rank = position of the first correct guess, RDKit pinned to 2026.03.3.

usage: python tools/metric.py <solution.csv: molecule_id,smiles> <submission.csv> [--by COLUMN_IN_SOLUTION]
"""
import sys
import warnings

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')
if rdkit.__version__ != '2026.03.3':
    warnings.warn(f'RDKit {rdkit.__version__} != 2026.03.3 (metric pin): tautomer keys may differ')
_TAUT = rdMolStandardize.TautomerEnumerator()
_CACHE = {}


def ik14(smiles):
    if smiles in _CACHE:
        return _CACHE[smiles]
    k = None
    if isinstance(smiles, str):
        try:
            m = Chem.MolFromSmiles(smiles)
            if m is not None:
                k = Chem.MolToInchiKey(_TAUT.Canonicalize(m))[:14]
        except (ValueError, RuntimeError, TypeError):
            k = None
    _CACHE[smiles] = k
    return k


def per_molecule_rr(solution, submission, max_rank=25):
    """Series molecule_id -> reciprocal rank (0 if missed) over the solution's molecules."""
    guesses = {m: [g.strip() for g in str(s).split(';') if g.strip()][:max_rank]
               for m, s in zip(submission.molecule_id, submission.smiles)}
    out = {}
    for m, ans in zip(solution.molecule_id, solution.smiles):
        acc = {ik14(a) for a in str(ans).split(',') if a} - {None}
        rr = 0.0
        for r, g in enumerate(guesses.get(m, []), 1):
            if ik14(g) in acc:
                rr = 1.0 / r
                break
        out[m] = rr
    return pd.Series(out, name='rr')


def score(solution, submission, max_rank=25):
    return float(per_molecule_rr(solution, submission, max_rank).mean())


if __name__ == '__main__':
    sol, sub = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
    rr = per_molecule_rr(sol, sub)
    print(f'MRR@25 = {rr.mean():.4f}  | top1 {np.mean(rr == 1):.3f}  top25-hit {np.mean(rr > 0):.3f}  n={len(rr)}')
    if len(sys.argv) > 4 and sys.argv[3] == '--by':
        col = sys.argv[4]
        print(rr.to_frame().join(sol.set_index('molecule_id')[col]).groupby(col).rr.agg(['mean', 'size']))
