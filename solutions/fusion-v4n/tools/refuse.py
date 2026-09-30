"""Re-run only the final fusion (fusion_core) locally on a notebook run's stage_cache, with CFG overrides, and score
it when labels exist. This is the fast iteration loop: the heavy stages run once on Kaggle (FULL_ON_COMMIT=True),
then fusion knobs are re-fused here in seconds.
CAVEAT: the only local labels are the visible placeholder test (drawn from train, all in the library). Use this to check
the plumbing and to see which molecules a knob moves; judge real gains on the leaderboard.

usage:
  python tools/refuse.py runs/<name> [--labels data/visible_labels.csv] [--set ICE_LAM=0.8 ...]
  python tools/refuse.py runs/<name> --grid FUSE_ALPHA=0.4,0.6,0.8 --grid FUSE_KRR=1,3,10
"""
import argparse, glob, itertools, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import SOLUTION, DATA, INPUT
sys.path.insert(0, os.path.join(SOLUTION, 'nb', 'embed'))
import pandas as pd
import fusion_core
import metric



def _pkg_dir(pattern):
    hits = sorted(glob.glob(os.path.join(INPUT, '**', pattern), recursive=True), key=len)
    return os.path.dirname(hits[0]) if hits else None


def load_modules():
    """ICEBERG fuse.py, GLACIER gl_fuse.py (score re-ranking) and the metric score key from the downloaded inputs.
    Missing packages degrade gracefully (that stage is skipped, like a failed stage on Kaggle)."""
    ice_fuse = gl_fuse = None
    d = _pkg_dir('ice_runner.py')
    if d:
        sys.path.append(d); import fuse as ice_fuse
    d = _pkg_dir('gl_runner.py')
    if d:
        sys.path.append(d); import gl_fuse
    score_key = metric.ik14            # fallback: metric's own key
    v4 = glob.glob(os.path.join(INPUT, '**', 'casmi26-v4b-models', '**', 'casmi', 'chem.py'), recursive=True)
    if v4:
        sys.path.insert(0, os.path.dirname(os.path.dirname(sorted(v4, key=len)[0])))
        try:
            from casmi import chem
            score_key = chem.score_key
        except Exception as e:
            print('casmi.chem import failed, using metric.ik14 as score key:', repr(e))
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    def formula(s):
        try:
            return CalcMolFormula(Chem.MolFromSmiles(s))
        except Exception:
            return s
    print('ice_fuse', bool(ice_fuse), '| gl_fuse', bool(gl_fuse), '| score_key', score_key.__module__)
    return ice_fuse, gl_fuse, score_key, formula


def load_run(run_dir):
    sc = next((d for d in glob.glob(os.path.join(run_dir, '**', 'stage_cache'), recursive=True)), None)
    assert sc, f'no stage_cache/ under {run_dir}'
    ld = lambda n: json.load(open(os.path.join(sc, f'{n}.json'))) if os.path.exists(os.path.join(sc, f'{n}.json')) else {}
    return {n: ld(n) for n in ['base', 'pc', 'eng', 'ice', 'gl', 'meta']}


def parse_val(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def run_one(S, mods, cfg, labels):
    ice_fuse, gl_fuse, score_key, formula = mods
    sub, st = fusion_core.build_submission(S['base'], S['pc'], S['eng'], S['ice'], S['gl'], cfg,
                                           S['meta']['mol_order'], S['meta']['sample_ids'], ice_fuse=ice_fuse,
                                           gl_fuse=gl_fuse, score_key=score_key, formula=formula, log=lambda *a: None)
    fusion_core.validate(sub, S['meta']['sample_ids'])
    s = metric.score(labels, sub) if labels is not None else None
    return sub, st, s


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--labels', default=os.path.join(DATA, 'visible_labels.csv'))
    ap.add_argument('--set', action='append', default=[]); ap.add_argument('--grid', action='append', default=[])
    ap.add_argument('--out')
    a = ap.parse_args()
    S = load_run(a.run)
    base_cfg = dict(S['meta'].get('cfg', {}))
    for kv in a.set:
        k, v = kv.split('=', 1); base_cfg[k] = parse_val(v)
    labels = pd.read_csv(a.labels) if os.path.exists(a.labels) else None
    if labels is not None:
        labels = labels[labels.molecule_id.isin(S['meta']['mol_order'])]
    print(f"run {a.run}: {len(S['base'])} molecules | labels: {0 if labels is None else len(labels)}")
    mods = load_modules()
    grid = [(kv.split('=', 1)[0], [parse_val(x) for x in kv.split('=', 1)[1].split(',')]) for kv in a.grid]
    rows = []
    for combo in itertools.product(*[vals for _, vals in grid]) if grid else [()]:
        cfg = dict(base_cfg); cfg.update({k: v for (k, _), v in zip(grid, combo)})
        sub, st, s = run_one(S, mods, cfg, labels)
        tag = ' '.join(f'{k}={v}' for (k, _), v in zip(grid, combo)) or 'cfg'
        rows.append(dict(setting=tag, mrr=s, **{k: v for (k, _), v in zip(grid, combo)}))
        print(f"{tag:40s} MRR@25={s if s is None else round(s, 4)}  merge={st['merge']}  fuse={st['fuse']}")
    if a.out:
        sub.to_csv(a.out, index=False); print('wrote', a.out)
    if grid and labels is not None:
        print(pd.DataFrame(rows).sort_values('mrr', ascending=False).head(10).to_string(index=False))
