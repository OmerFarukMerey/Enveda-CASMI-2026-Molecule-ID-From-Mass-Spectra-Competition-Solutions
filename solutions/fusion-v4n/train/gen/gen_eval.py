"""Evaluate a score file on the generated candidate sets of the gen2 study (DEV molecules): rank of the answer among the first B generated structures.

Candidate file ($CASMI_DATA/gen2/gen_cands_dev.parquet, written by scratch/gen2/g8_eval.py): one row per generated structure of a molecule
  molecule_id, candidate_key (metric key), smiles, source_rule, step ('1' | 'move' | '2'), ord (1 = first in the pre-declared budget order),
  parent_pid, parent_rank, n_support, is_answer, eset (N = answer in no database; P / Lpanel / Lnp / C = leave-out proxies, answer removed by hand)
Score file: the format of train/joint/bench.py, a parquet of (molecule_id, candidate_key, score); higher is better; a missing candidate ranks last.
Per set and budget B: molecules, share with the answer among the first B, top-1 and MRR@25 over those (distinct metric keys),
and the expected value per molecule = share x MRR. `score=None` evaluates the budget order itself.

usage: python train/gen/gen_eval.py [--scores S.parquet] [--cands C.parquet] [--budgets 100,300,1000]
"""
import argparse, os

import numpy as np, pandas as pd

K = 25
DEFAULT = os.environ.get('CASMI_DATA', 'data').replace(chr(92), '/') + '/gen2/gen_cands_dev.parquet'


def pairs(cands=DEFAULT):
    """Unique (molecule_id, candidate_key, smiles) a model has to score."""
    return pd.read_parquet(cands, columns=['molecule_id', 'candidate_key', 'smiles']).drop_duplicates(['molecule_id', 'candidate_key']).reset_index(drop=True)


def evaluate(scores=None, cands=DEFAULT, budgets=(100, 300, 1000, 2000)):
    C = pd.read_parquet(cands)
    if scores is None:
        C['score'] = -C.ord.astype(np.float64)
    else:
        s = pd.read_parquet(scores) if isinstance(scores, str) else scores
        s = s.groupby(['molecule_id', 'candidate_key'], sort=False).score.max().reset_index()
        C = C.merge(s, on=['molecule_id', 'candidate_key'], how='left'); C['score'] = C.score.fillna(-np.inf)
    mf = os.path.join(os.path.dirname(cands), os.path.basename(cands).replace('gen_cands', 'gen_mols'))     # every evaluation molecule, also those without a candidate
    n_mol = pd.read_parquet(mf).groupby('eset').molecule_id.nunique() if os.path.exists(mf) and mf != cands else C.groupby('eset').molecule_id.nunique(); rows = []
    for B in budgets:
        c = C[C.ord <= B].sort_values(['molecule_id', 'score', 'ord'], ascending=[True, False, True], kind='stable')
        c = c.drop_duplicates(['molecule_id', 'candidate_key']); c['rank'] = c.groupby('molecule_id').cumcount() + 1
        a = c[c.is_answer]
        for e, g in a.groupby('eset'):
            rr = np.where(g['rank'] <= K, 1.0 / g['rank'], 0.0); n = int(n_mol[e])
            rows.append(dict(eset=e, B=B, molecules=n, answer_in_B=len(g), share=len(g) / n, top1=float((g['rank'] == 1).mean()), mrr_given=float(rr.mean()),
                             se=float(rr.std(ddof=1) / np.sqrt(len(rr))) if len(rr) > 1 else float('nan'), value_per_molecule=float(rr.sum() / n)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--scores'); ap.add_argument('--cands', default=DEFAULT); ap.add_argument('--budgets', default='100,300,1000,2000'); a = ap.parse_args()
    pd.set_option('display.width', 200); print(evaluate(a.scores, a.cands, tuple(int(x) for x in a.budgets.split(','))).round(3).to_string(index=False))
