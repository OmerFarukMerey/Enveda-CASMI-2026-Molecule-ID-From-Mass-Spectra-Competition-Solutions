"""Same-formula isomer panel for nb/embed/frag_rescore.py: MRR@25 of the true structure among train isomers, per setting.

Queries are the enveda-np-examples structures (timsTOF) with their spectra restricted to the 10 test adducts;
candidates are the truth plus up to --n-cand other train structures (distinct inchikey14) with the same
molecular_formula; panels with fewer than 3 candidates are skipped. Ties count as the expected reciprocal rank
under random tie-breaking. The structures are in train, but the scorer is untrained, so there is no leak for it.

usage: python tools/eval_frag_panel.py [--data D:/casmi-data] [--n-cand 30] [--limit N] [--out runs/frag_panel/results.json]
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'nb', 'embed'))
import paths  # noqa: E402
import frag_rescore as fr  # noqa: E402

TEST_ADDUCTS = ['[M+H]+', '[M-H]-', '[M+Na]+', '[M+NH4]+', '[M-H2O+H]+', '[M+K]+', '[M+Cl]-', '[M+CH2O2-H]-',
                '[M-H2O-H]-', '[M-2H2O+H]+']
D = fr.DEFAULT_CFG
B = fr.BASELINE_CFG
SETTINGS = [  # (name, cfg overrides of DEFAULT_CFG)
    ('baseline (pv-style)', dict(B)),
    ('default (forum)', {}),
    ('default, w2=1.0', dict(w2=1.0)),
    ('default, sqrt int', dict(intensity='sqrt')),
    ('default, tol 0.01', dict(tol=0.01)),
    ('default, H -2..+2', dict(h_shifts=tuple(range(-2, 3)))),
    ('default, w2=0.4', dict(w2=0.4)),
    ('default, w2=0.8', dict(w2=0.8)),
    ('default, 1 break only', dict(max_breaks=1)),
    ('default, 3 breaks w3=0.3', dict(max_breaks=3, w3=0.3)),
    ('default, mz^0.5 weight', dict(mz_power=0.5)),
    ('default, tol 0.003', dict(tol=0.003)),
    ('default, H -3..+3', dict(h_shifts=tuple(range(-3, 4)))),
]
K = 25


def load_queries(train):
    cols = ['normalized_smiles', 'inchikey14', 'molecular_formula', 'ionization_mode', 'adduct', 'precursor_mz',
            'ms2_mzs', 'ms2_normalized_intensities']
    t = pq.read_table(train, columns=cols, filters=[('ingest_lib', '==', 'enveda-np-examples'),
                                                     ('adduct', 'in', TEST_ADDUCTS)])
    df = t.to_pandas()
    q = {}
    for r in df.itertuples(index=False):
        mz = np.asarray(r.ms2_mzs, float)
        it = np.asarray(r.ms2_normalized_intensities, float)
        if len(mz) == 0 or not np.isfinite(it).any() or np.nanmax(it) <= 0:
            continue
        e = q.setdefault(r.inchikey14, dict(smiles=[], formula=r.molecular_formula, spectra=[]))
        e['smiles'].append(r.normalized_smiles)
        e['spectra'].append(dict(mz=mz, it=it / np.nanmax(it), adduct=r.adduct, prec=float(r.precursor_mz),
                                 mode=1 if str(r.ionization_mode).startswith('pos') else -1))
    for e in q.values():
        e['smiles'] = sorted(e['smiles'])[0]
    return q


def load_isomers(train, formulas):
    t = pq.read_table(train, columns=['normalized_smiles', 'inchikey14', 'molecular_formula'],
                      filters=[('molecular_formula', 'in', sorted(formulas))])
    df = t.to_pandas().drop_duplicates()
    df = df.sort_values(['molecular_formula', 'inchikey14', 'normalized_smiles'])
    df = df.drop_duplicates(['molecular_formula', 'inchikey14'])
    return {f: list(zip(g.inchikey14, g.normalized_smiles)) for f, g in df.groupby('molecular_formula')}


def build_panels(q, iso, n_cand):
    panels, skipped = [], 0
    for ik in sorted(q):
        e = q[ik]
        others = [s for k, s in iso.get(e['formula'], []) if k != ik]
        if len(others) > n_cand:
            rng = np.random.default_rng(zlib.crc32(ik.encode()))
            others = [others[i] for i in sorted(rng.choice(len(others), n_cand, replace=False))]
        if len(others) + 1 < 3:
            skipped += 1
            continue
        panels.append(dict(ik=ik, formula=e['formula'], spectra=e['spectra'], cands=[e['smiles']] + others))
    return panels, skipped


def expected_rr(scores, eps=1e-12):
    """Expected reciprocal rank@K of candidate 0 under random tie-breaking (None scores rank last)."""
    s = np.array([-np.inf if x is None else x for x in scores], float)
    t = s[0]
    better = int((s > t + eps).sum())
    ties = int((np.abs(s - t) <= eps).sum()) - 1 if np.isfinite(t) else len(s) - 1 - better
    r = better + 1 + np.arange(ties + 1)
    rr = np.where(r <= K, 1.0 / r, 0.0)
    return float(rr.mean()), float((r == 1).mean()), ties


def random_rr(n):
    r = np.arange(1, n + 1)
    return float(np.where(r <= K, 1.0 / r, 0.0).mean()), 1.0 / n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=paths.DATA)
    ap.add_argument('--n-cand', type=int, default=30)
    ap.add_argument('--limit', type=int, default=0, help='only the first N structures (debug)')
    ap.add_argument('--out', default=os.path.join(paths.RUNS, 'frag_panel', 'results.json'))
    a = ap.parse_args()
    train = os.path.join(a.data, 'input', 'competitions', 'enveda-CASMI26-molecule-id-mass-spectra', 'train.parquet')

    t0 = time.time()
    q = load_queries(train)
    if a.limit:
        q = {k: q[k] for k in sorted(q)[: a.limit]}
    iso = load_isomers(train, {e['formula'] for e in q.values()})
    panels, skipped = build_panels(q, iso, a.n_cand)
    n_c = np.array([len(p['cands']) for p in panels])
    n_s = np.array([len(p['spectra']) for p in panels])
    print(f'{len(q)} structures, {len(panels)} panels (skipped {skipped} with <3 candidates), '
          f'candidates/panel mean {n_c.mean():.1f} median {np.median(n_c):.0f}, spectra/panel {n_s.mean():.1f}, '
          f'load {time.time() - t0:.1f}s', flush=True)

    fr.warmup()
    cache = {}
    # timing: the default setting on a cold cache (fragment enumeration + scoring), per panel
    per_panel = []
    t_all = time.time()
    for p in panels:
        t = time.time()
        fr.score_molecule(p['spectra'], p['cands'], None, cache)
        per_panel.append(time.time() - t)
    per_panel = np.array(per_panel)
    per_cand = per_panel.sum() / n_c.sum()
    t_score = time.time()
    for p in panels:
        fr.score_molecule(p['spectra'], p['cands'], None, cache)
    per_cand_warm = (time.time() - t_score) / n_c.sum()
    timing = dict(total_cold_s=round(time.time() - t_all, 2), per_panel_cold_ms_mean=round(float(1e3 * per_panel.mean()), 2),
                  per_panel_cold_ms_max=round(float(1e3 * per_panel.max()), 2), per_candidate_cold_ms=round(float(1e3 * per_cand), 3),
                  per_candidate_warm_ms=round(float(1e3 * per_cand_warm), 3), mean_spectra_per_panel=round(float(n_s.mean()), 2),
                  est_400x100_on_4_cores_min=round(float(400 * 100 * per_cand / 4 / 60), 2), numba=fr.HAVE_NUMBA)
    print('timing', timing, flush=True)

    rows, rrs = [], {}
    rnd = [random_rr(len(p['cands'])) for p in panels]
    for name, cfg in SETTINGS:
        t = time.time()
        res = [expected_rr(fr.score_molecule(p['spectra'], p['cands'], cfg, cache)) for p in panels]
        rrs[name] = np.array([x[0] for x in res])
        rows.append(dict(setting=name, cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                         mrr=float(rrs[name].mean()), top1=float(np.mean([x[1] for x in res])),
                         truth_tied_frac=float(np.mean([x[2] > 0 for x in res])), secs=round(time.time() - t, 2)))
    for r in rows:  # paired per-panel differences (mean, standard error)
        for ref, tag in ((SETTINGS[0][0], 'baseline'), (SETTINGS[1][0], 'default')):
            d = rrs[r['setting']] - rrs[ref]
            r[f'd_vs_{tag}'] = float(d.mean())
            r[f'd_vs_{tag}_se'] = float(d.std(ddof=1) / np.sqrt(len(d)))
    rows.append(dict(setting='random order', mrr=float(np.mean([x[0] for x in rnd])),
                     top1=float(np.mean([x[1] for x in rnd]))))

    print(f'\n{"setting":28s} {"MRR@25":>7s} {"top1":>6s} {"vs base":>8s} {"+-se":>6s} {"vs dflt":>8s} {"+-se":>6s} '
          f'{"tied":>5s}')
    for r in rows:
        if 'd_vs_baseline' in r:
            print(f'{r["setting"]:28s} {r["mrr"]:7.4f} {r["top1"]:6.3f} {r["d_vs_baseline"]:+8.4f} '
                  f'{r["d_vs_baseline_se"]:6.4f} {r["d_vs_default"]:+8.4f} {r["d_vs_default_se"]:6.4f} '
                  f'{r["truth_tied_frac"]:5.2f}')
        else:
            print(f'{r["setting"]:28s} {r["mrr"]:7.4f} {r["top1"]:6.3f}')

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = dict(n_structures=len(q), n_panels=len(panels), skipped_lt3=skipped, n_cand_cap=a.n_cand,
               cands_per_panel_mean=float(n_c.mean()), spectra_per_panel_mean=float(n_s.mean()),
               adducts=TEST_ADDUCTS, timing=timing, results=rows)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', a.out)


if __name__ == '__main__':
    main()
