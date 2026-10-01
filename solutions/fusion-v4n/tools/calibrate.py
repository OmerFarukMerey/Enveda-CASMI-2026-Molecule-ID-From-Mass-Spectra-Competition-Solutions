"""Acceptance test for a validation run: does local MRR reproduce the order of our public-LB results (lb_points.json)?

For each run it replays every LB point's config (tools/refuse.py; v5-style points need a VAL_EXTRAS run) and reports the
local MRR, the Spearman rank correlation with the LB, and the sign agreement over all pairs of LB points that differ
by at least --min-gap on the LB. A regime-weighted composite of several runs is scored the same way.
Trust a harness only if the correlation is clearly positive and the large LB gaps have the right sign.

usage: python tools/calibrate.py <run_dir> [<run_dir> ...] [--weights 0.5,0.5] [--shuffle-seeds N] [--min-gap 0.008]
  --shuffle-seeds N: evaluate through the isomer shuffle of sim_isomer_eval (only for runs whose FP bank saw the molecules)
"""
import argparse, itertools, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refuse
from paths import SOLUTION


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else float('nan')


def pair_agreement(lb, loc, min_gap):
    ok = tot = 0
    for i, j in itertools.combinations(range(len(lb)), 2):
        if abs(lb[i] - lb[j]) >= min_gap:
            tot += 1
            ok += int(np.sign(lb[i] - lb[j]) == np.sign(loc[i] - loc[j]))
    return ok, tot


def eval_run(run, points, mods, pool_pop, seeds):
    S = refuse.load_run(run)
    lab = pd.read_csv(S['val_labels']); lab = lab[lab.molecule_id.isin(S['meta']['mol_order'])]
    extras = bool(S.get('pc_pop')) and bool(S.get('gl_mh'))
    out = {}
    for p in points:
        if p.get('needs_extras') and not extras:
            out[p['name']] = None
            continue
        cfg = dict(S['meta'].get('cfg', {})); cfg.update(TOPN=60, FP_BANK=cfg.get('FP_BANK')); cfg.update(p['set'])
        if seeds:
            import sim_isomer_eval
            rr = [refuse.run_one(dict(S, base=sim_isomer_eval.shuffle_isomers(S['base'], np.random.RandomState(s), 0.0)),
                                 mods, cfg, lab, pool_pop)[2] for s in range(seeds)]
            out[p['name']] = pd.concat(rr, axis=1).mean(axis=1)
        else:
            out[p['name']] = refuse.run_one(S, mods, cfg, lab, pool_pop)[2]
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+'); ap.add_argument('--weights'); ap.add_argument('--shuffle-seeds', type=int, default=0)
    ap.add_argument('--min-gap', type=float, default=0.008)
    a = ap.parse_args()
    points = json.load(open(os.path.join(SOLUTION, 'lb_points.json'), encoding='utf-8'))['points']
    mods = refuse.load_modules(); pool_pop = refuse.load_pool_pop()
    table = pd.DataFrame({'config': [p['name'] for p in points], 'LB': [p['lb'] for p in points]})
    for run in a.runs:
        res = eval_run(run, points, mods, pool_pop, a.shuffle_seeds)
        table[os.path.basename(os.path.normpath(run))] = [None if res[p['name']] is None else round(float(res[p['name']].mean()), 4) for p in points]
    cols = list(table.columns[2:])
    if len(cols) > 1:
        w = [float(x) for x in a.weights.split(',')] if a.weights else [1 / len(cols)] * len(cols)
        table['composite'] = sum(wi * table[c].astype(float) for wi, c in zip(w, cols))
    print(table.to_string(index=False))
    for c in table.columns[2:]:
        t = table.dropna(subset=[c])
        ok, tot = pair_agreement(list(t.LB), list(t[c].astype(float)), a.min_gap)
        print(f'{c:12s} Spearman vs LB = {spearman(t.LB, t[c].astype(float)):+.2f} over {len(t)} points | '
              f'pairs with LB gap >= {a.min_gap}: {ok}/{tot} ordered correctly')
