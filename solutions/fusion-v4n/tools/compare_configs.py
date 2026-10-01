"""Compare named configs (each = a run's stage cache + CFG overrides) paired against a reference on the same molecules.

Reports MRR@25, paired delta ± SE and W/L vs the reference, plus how much each config changes the reference's
answers (top-1 changed, any of the top-25 changed). On leaked validation runs the MRR deltas are biased (see
EXPERIMENTS.md, val1); the change counts still show how big and how risky a change is.

usage: python tools/compare_configs.py <spec.json>
  spec = {"reference": "v2", "configs": {"v2": {"run": "runs/val1", "set": {...}}, "S1": {...}, ...}}
"""
import json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refuse
import metric


def main(spec_path):
    spec = json.load(open(spec_path, encoding='utf-8'))
    mods = refuse.load_modules(); pool_pop = refuse.load_pool_pop()
    res = {}
    for name, c in spec['configs'].items():
        S = refuse.load_run(c['run'])
        cfg = dict(S['meta'].get('cfg', {})); cfg.update(c.get('set', {}))
        labels = pd.read_csv(S['val_labels'] or c['labels'])
        labels = labels[labels.molecule_id.isin(S['meta']['mol_order'])]
        sub, st, rr = refuse.run_one(S, mods, cfg, labels, pool_pop)
        res[name] = dict(rr=rr, lists=dict(zip(sub.molecule_id, sub.smiles.str.split(';'))))
        print(f'{name:10s} MRR@25={rr.mean():.4f} top1={np.mean(rr == 1):.3f}  ({c["run"]}, {c.get("set", {})})', flush=True)
    ref = res[spec['reference']]
    rows = []
    for name, r in res.items():
        mids = ref['rr'].index
        d = (r['rr'].reindex(mids) - ref['rr']).values
        key = lambda s: metric.ik14(s) or s
        top1 = sum(key(r['lists'][m][0]) != key(ref['lists'][m][0]) for m in mids)
        top25 = sum([key(x) for x in r['lists'][m][:25]] != [key(x) for x in ref['lists'][m][:25]] for m in mids)
        rows.append(dict(config=name, mrr=round(float(r['rr'].mean()), 4), delta=round(float(d.mean()), 4),
                         se=round(float(d.std(ddof=1) / np.sqrt(len(d))), 4), wins=int((d > 0).sum()),
                         losses=int((d < 0).sum()), top1_changed=int(top1), top25_changed=int(top25), n=len(d)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = os.path.splitext(spec_path)[0] + '_results.json'
    df.to_json(out, orient='records', indent=1); print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1])
