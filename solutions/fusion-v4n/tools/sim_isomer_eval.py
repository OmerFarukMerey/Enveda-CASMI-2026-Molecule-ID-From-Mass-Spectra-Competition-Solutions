"""De-leaked local proxy: compare configs on a VALIDATION run after erasing the v1 ranker's memorised isomer order.

On train-derived holdouts the v1 ranker puts the truth top-1 ~94% of the time (its models saw those spectra), so every
re-scorer looks harmful. On the hidden test ~98% of ranking errors are between same-formula isomers (forum). This
script simulates that regime: inside each same-formula group of the base list the candidates are shuffled among the
group's slots with probability 1-RHO per molecule (scores stay with the slots), so the ranker carries no isomer
information and ICE / GLACIER / frag / popularity have to find the truth. Results are averaged over seeds and compared
paired with a reference config.

Calibrate before trusting it: the known public-LB order is v2 (0.404) > v1 (0.399) > v3b frag-all (0.390).

usage: python tools/sim_isomer_eval.py <spec.json> [--seeds 10] [--rho 0.0]
  spec = same format as compare_configs.py: {"reference": ..., "configs": {name: {"run": ..., "set": {...}}}}
"""
import argparse, copy, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refuse
import metric


def shuffle_isomers(base, rng, rho):
    """Per molecule (prob 1-rho): permute the candidates of every same-formula group among that group's slots."""
    out = {}
    for m, v in base.items():
        v = copy.deepcopy(v)
        if len(v[0]) > 1 and rng.random() >= rho:
            groups = {}
            for i, f in enumerate(v[4]):
                groups.setdefault(f, []).append(i)
            perm = list(range(len(v[0])))
            for idx in groups.values():
                if len(idx) > 1:
                    for slot, src in zip(idx, rng.permutation(idx)):
                        perm[slot] = int(src)
            for j in (0, 1, 4) + ((5,) if len(v) > 5 else ()):      # smiles, keys, formulas, pool ids move; scores stay
                v[j] = [v[j][i] for i in perm]
        out[m] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spec'); ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--rho', type=float, default=0.0)
    a = ap.parse_args()
    spec = json.load(open(a.spec, encoding='utf-8'))
    mods = refuse.load_modules(); pool_pop = refuse.load_pool_pop()
    runs = {}
    rr = {name: [] for name in spec['configs']}
    for seed in range(a.seeds):
        shuffled = {}
        for name, c in spec['configs'].items():
            if c['run'] not in runs:
                S = refuse.load_run(c['run'])
                lab = pd.read_csv(S['val_labels']); lab = lab[lab.molecule_id.isin(S['meta']['mol_order'])]
                runs[c['run']] = (S, lab)
            S, lab = runs[c['run']]
            if c['run'] not in shuffled:                              # same shuffle for every config on this run
                shuffled[c['run']] = shuffle_isomers(S['base'], np.random.RandomState(seed), a.rho)
            S2 = dict(S); S2['base'] = shuffled[c['run']]
            cfg = dict(S['meta'].get('cfg', {})); cfg.update(c.get('set', {}))
            rr[name].append(refuse.run_one(S2, mods, cfg, lab, pool_pop)[2])
        print(f'seed {seed}: ' + '  '.join(f'{n}={rr[n][-1].mean():.4f}' for n in rr), flush=True)
    mean = {n: pd.concat(v, axis=1).mean(axis=1) for n, v in rr.items()}     # per-molecule mean over seeds
    ref = mean[spec['reference']]
    rows = []
    for n, r in mean.items():
        d = (r.reindex(ref.index) - ref).values
        rows.append(dict(config=n, sim_mrr=round(float(r.mean()), 4), delta=round(float(d.mean()), 4),
                         se=round(float(d.std(ddof=1) / np.sqrt(len(d))), 4),
                         t=round(float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)), 1),
                         wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum())))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = os.path.splitext(a.spec)[0] + f'_sim_rho{a.rho}.json'
    df.to_json(out, orient='records', indent=1); print('wrote', out)


if __name__ == '__main__':
    main()
