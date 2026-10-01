"""Re-run only the final fusion (fusion_core) locally on a notebook run's stage_cache, with CFG overrides, and score it.
The heavy stages run once on Kaggle; fusion knobs (ICE_LAM, GL_LAM, FUSE_*, TOPN depth, POP_MU, gates) are then
re-fused here in seconds and compared PAIRED on the same molecules against the run's own config.
Labels: <run>/**/stage_cache/val_labels.csv (VALIDATION runs: held-out molecules, library purged) if present, else
--labels (the visible placeholder test: plumbing check only, all library hits).

usage:
  python tools/refuse.py runs/<name> [--set ICE_LAM=0.8 ...]
  python tools/refuse.py runs/<name> --grid FUSE_ALPHA=0.4,0.6,0.8 --grid FUSE_KRR=1,3,10 [--results out.json]
"""
import argparse, glob, itertools, json, os, sys

import numpy as np

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
    import functools                        # grids call these on the same SMILES thousands of times
    return ice_fuse, gl_fuse, functools.lru_cache(maxsize=None)(score_key), functools.lru_cache(maxsize=None)(formula)


def load_run(run_dir):
    sc = next((d for d in glob.glob(os.path.join(run_dir, '**', 'stage_cache'), recursive=True)), None)
    assert sc, f'no stage_cache/ under {run_dir}'
    ld = lambda n: json.load(open(os.path.join(sc, f'{n}.json'))) if os.path.exists(os.path.join(sc, f'{n}.json')) else {}
    S = {n: ld(n) for n in ['base', 'pc', 'eng', 'ice', 'gl', 'meta', 'frag', 'pc_pop', 'gl_mh']}   # last two: VAL_EXTRAS
    S['val_labels'] = os.path.join(sc, 'val_labels.csv') if os.path.exists(os.path.join(sc, 'val_labels.csv')) else None
    return S


def load_pool_pop():
    """log1p(PubChem SIDs) + log1p(PubMed ids) per v2-pool row, from casmi26-pubchem-popularity-prior (if downloaded)."""
    d = _pkg_dir('pool_lsid.npy')
    if not d:
        return None
    return np.load(os.path.join(d, 'pool_lsid.npy')).astype(np.float64) + np.load(os.path.join(d, 'pool_lpmid.npy')).astype(np.float64)


def parse_val(v):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def run_one(S, mods, cfg, labels, pool_pop=None):
    ice_fuse, gl_fuse, score_key, formula = mods
    # replay heavy-stage variants from the VAL_EXTRAS dumps: PubChem channel with popularity, GLACIER on [M+H]+ only
    pc = S['pc_pop'] if (cfg.get('PC_POP_LAM') or 0) > 0 and S.get('pc_pop') else S['pc']
    gl = S['gl_mh'] if cfg.get('GL_MH_ONLY') and S.get('gl_mh') else S['gl']
    sub, st = fusion_core.build_submission(S['base'], pc, S['eng'], S['ice'], gl, cfg,
                                           S['meta']['mol_order'], S['meta']['sample_ids'], ice_fuse=ice_fuse,
                                           gl_fuse=gl_fuse, score_key=score_key, formula=formula, log=lambda *a: None,
                                           pool_pop=pool_pop, frag_scores=S['frag'] or None)
    fusion_core.validate(sub, S['meta']['sample_ids'])
    rr = metric.per_molecule_rr(labels, sub) if labels is not None else None
    return sub, st, rr


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--labels', default=os.path.join(DATA, 'visible_labels.csv'))
    ap.add_argument('--set', action='append', default=[]); ap.add_argument('--grid', action='append', default=[])
    ap.add_argument('--out'); ap.add_argument('--results')
    a = ap.parse_args()
    S = load_run(a.run)
    base_cfg = dict(S['meta'].get('cfg', {}))
    for kv in a.set:
        k, v = kv.split('=', 1); base_cfg[k] = parse_val(v)
    lab_path = S['val_labels'] or a.labels
    labels = pd.read_csv(lab_path) if os.path.exists(lab_path) else None
    if labels is not None:
        labels = labels[labels.molecule_id.isin(S['meta']['mol_order'])]
    print(f"run {a.run}: {len(S['base'])} molecules | labels: {0 if labels is None else len(labels)} from {lab_path}")
    mods = load_modules()
    pool_pop = load_pool_pop()
    print('pool popularity prior:', None if pool_pop is None else pool_pop.shape, '| frag scores:', len(S['frag']), 'molecules')
    _, _, rr0 = run_one(S, mods, base_cfg, labels, pool_pop)
    if rr0 is not None:
        print(f"run config: MRR@25={rr0.mean():.4f} top1={np.mean(rr0 == 1):.3f} hit25={np.mean(rr0 > 0):.3f}")
    grid = [(kv.split('=', 1)[0], [parse_val(x) for x in kv.split('=', 1)[1].split(',')]) for kv in a.grid]
    rows = []
    for combo in itertools.product(*[vals for _, vals in grid]) if grid else [()]:
        cfg = dict(base_cfg); cfg.update({k: v for (k, _), v in zip(grid, combo)})
        sub, st, rr = run_one(S, mods, cfg, labels, pool_pop)
        tag = ' '.join(f'{k}={v}' for (k, _), v in zip(grid, combo)) or 'cfg'
        row = dict(setting=tag, **{k: v for (k, _), v in zip(grid, combo)})
        if rr is not None:
            d = (rr - rr0).values
            row.update(mrr=round(float(rr.mean()), 4), delta=round(float(d.mean()), 4),
                       se=round(float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0, 4),
                       wins=int((d > 0).sum()), losses=int((d < 0).sum()), top1=round(float(np.mean(rr == 1)), 3))
        rows.append(row)
        print(f"{tag:45s} " + (f"MRR={row['mrr']:.4f} d={row['delta']:+.4f}±{row['se']:.4f} W/L={row['wins']}/{row['losses']}"
                               if rr is not None else f"merge={st['merge']}"))
    if a.out:
        sub.to_csv(a.out, index=False); print('wrote', a.out)
    if grid and labels is not None:
        df = pd.DataFrame(rows).sort_values('mrr', ascending=False)
        print(df.head(15).to_string(index=False))
        if a.results:
            os.makedirs(os.path.dirname(os.path.abspath(a.results)), exist_ok=True)
            df.to_json(a.results, orient='records', indent=1); print('wrote', a.results)
