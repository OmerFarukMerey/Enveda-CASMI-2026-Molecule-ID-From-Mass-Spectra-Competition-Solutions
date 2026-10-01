"""Evaluate named fusion configs in several validation regimes at once, paired against a reference config.

One table per spec: for every regime the MRR@25, paired delta +- SE, wins/losses and how many top-1 answers change
versus the reference, plus a composite delta for assumed class-1 shares of the hidden test
(composite = share * delta[class-1 regime] + (1 - share) * delta[class-2 regime]).

Regimes (default; add or replace them in the spec, a regime is just a run directory with a stage_cache/):
  class2   runs/val1, mode "sim": 250 held-out natural products, library purged. The v1 ranker memorised them, so the
           same-formula candidates of the base lists are shuffled (tools/sim_isomer_eval.shuffle_isomers) and the
           result is averaged over seeds. Biased: over-rewards popularity, blind to the v3b drop (EXPERIMENTS.md).
  library  runs/vis, mode "plain": 400 visible-test molecules, all exact library hits (lib_max = 1.0). No shuffle.
           Easier than real class-1 molecules: a "do no harm to library hits" check, not a class-1 estimate.

A regime may carry its own "set": overrides forced on every config in that regime only, e.g. {"LIB_TAU": 1.01} on
runs/vis switches every library gate off ("what if the gate does not fire on a class-1 molecule").

Everything is a local re-fusion of cached stage outputs (tools/refuse.run_one). Two caches under --cache-dir
(default runs/search/cache, git-ignored) make repeated calls cheap: keys.json (score key / formula / metric key per
SMILES, the slow tautomer canonicalisation) and results.sqlite (per-molecule results per regime, seed and effective
fusion config; invalidated when a stage file or fusion_core.py changes).

usage: python tools/regime_eval.py <spec.json> [--out PREFIX] [--seeds N] [--cache-dir DIR] [--no-result-cache]
  spec = {"reference": "v2",
          "base_set": {"TOPN": 60},                                  # applied to every config (optional)
          "configs": {"v2": {"set": {"POP_MU": 0.15, ...}}, ...},    # overrides on top of the run's own cfg
          "regimes": {"class2": {"run": "runs/val1", "mode": "sim", "seeds": 6, "rho": 0.0},
                      "library": {"run": "runs/vis", "mode": "plain"},
                      "class1": {"run": "runs/valc1", "mode": "plain", "set": {}}},   # optional, defaults above
          "composite": {"class1": "library", "class2": "class2", "shares": [0.2, 0.35, 0.5]}}   # optional
Writes PREFIX.json (table rows), PREFIX.md (the table) and PREFIX_rr_<regime>.csv (per-molecule reciprocal ranks).
"""
import argparse, hashlib, json, os, sqlite3, sys, time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import SOLUTION, REPO, RUNS
import refuse
import metric
import fusion_core
from sim_isomer_eval import shuffle_isomers

DEFAULT_REGIMES = {'class2': {'run': 'runs/val1', 'mode': 'sim', 'seeds': 6, 'rho': 0.0},
                   'library': {'run': 'runs/vis', 'mode': 'plain'}}
DEFAULT_COMPOSITE = {'class1': 'library', 'class2': 'class2', 'shares': [0.2, 0.35, 0.5]}
STAGES = ['base', 'pc', 'eng', 'ice', 'gl', 'frag', 'meta', 'val_labels']


def resolve(path):
    """Run directories may be given relative to the cwd, the repo root or the solution directory."""
    for root in ('', REPO, SOLUTION):
        p = os.path.join(root, path) if root else path
        if os.path.isdir(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f'run directory not found: {path}')


def _sha(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class KeyCache:
    """Disk-backed memo for the three per-SMILES functions that dominate a cold start (minutes -> a second)."""

    def __init__(self, path, score_key, formula):
        self.path, self.fns = path, {'score_key': score_key, 'formula': formula}
        d = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
        self.d = {'score_key': d.get('score_key', {}), 'formula': d.get('formula', {})}
        metric._CACHE.update(d.get('ik14', {}))
        self.n0 = self.size()

    def size(self):
        return len(self.d['score_key']) + len(self.d['formula']) + len(metric._CACHE)

    def fn(self, name):
        d, f = self.d[name], self.fns[name]

        def call(s):
            try:
                return d[s]
            except KeyError:
                v = d[s] = f(s)
                return v
        return call

    def save(self):
        if self.size() == self.n0:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(dict(self.d, ik14={k: v for k, v in metric._CACHE.items() if isinstance(k, str)}), f)
        os.replace(tmp, self.path)
        self.n0 = self.size()


class ResultCache:
    """sqlite: (regime signature, effective config signature, seed) -> per-molecule rr / top-1 key / top-25 hash."""

    def __init__(self, path):
        self.db = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.db = sqlite3.connect(path)
            self.db.execute('create table if not exists r (regime text, cfg text, seed int, payload text, '
                            'primary key (regime, cfg, seed))')

    def get(self, regime, cfg, seed):
        if self.db is None:
            return None
        row = self.db.execute('select payload from r where regime=? and cfg=? and seed=?', (regime, cfg, seed)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, regime, cfg, seed, payload):
        if self.db is not None:
            self.db.execute('insert or replace into r values (?,?,?,?)', (regime, cfg, seed, json.dumps(payload)))

    def commit(self):
        if self.db is not None:
            self.db.commit()


def effective_cfg(cfg):
    """Only the keys fusion_core reads decide the result (cache key)."""
    c = dict(fusion_core.DEFAULTS); c.update({k: v for k, v in cfg.items() if k in fusion_core.DEFAULTS})
    return c


class Regime:
    def __init__(self, name, spec, mods, pool_pop, rcache, seeds=None):
        self.name, self.mode = name, spec.get('mode', 'plain')
        assert self.mode in ('plain', 'sim'), f'regime {name}: mode must be "plain" or "sim"'
        self.run = resolve(spec['run'])
        self.S = refuse.load_run(self.run)
        lab_path = self.S['val_labels'] or spec.get('labels')
        assert lab_path and os.path.exists(lab_path), f'regime {name}: no val_labels.csv in the stage cache and no "labels"'
        lab = pd.read_csv(lab_path)
        self.labels = lab[lab.molecule_id.isin(self.S['meta']['mol_order'])]
        self.mids = list(self.labels.molecule_id)
        self.rho = float(spec.get('rho', 0.0))
        self.seeds = list(range(int(seeds or spec.get('seeds', 6)))) if self.mode == 'sim' else [-1]
        self.force = dict(spec.get('set', {}))               # regime-level overrides, applied last
        self.mods, self.pool_pop, self.rcache = mods, pool_pop, rcache
        self._shuffled = {}
        sc = os.path.dirname(self.S['val_labels']) if self.S['val_labels'] else self.run
        files = [(n, os.path.getsize(p), int(os.path.getmtime(p))) for n in STAGES
                 for p in [os.path.join(sc, n + ('.csv' if n == 'val_labels' else '.json'))] if os.path.exists(p)]
        code = hashlib.sha1(open(fusion_core.__file__, 'rb').read()).hexdigest()
        self.sig = _sha([files, self.mode, self.rho, code, lab_path if not self.S['val_labels'] else ''])
        libs = np.array([v[2] for v in self.S['base'].values()], float)
        self.info = dict(run=os.path.relpath(self.run, REPO).replace('\\', '/'), mode=self.mode, n=len(self.mids),
                         seeds=len(self.seeds) if self.mode == 'sim' else 0, set=self.force,
                         lib_hits=int((libs >= self.S['meta'].get('cfg', {}).get('LIB_TAU', 0.9)).sum()))

    def base(self, seed):
        if seed < 0:
            return self.S['base']
        if seed not in self._shuffled:                       # same shuffle for every config (paired comparison)
            self._shuffled[seed] = shuffle_isomers(self.S['base'], np.random.RandomState(seed), self.rho)
        return self._shuffled[seed]

    def evaluate(self, overrides):
        """-> dict(rr=DataFrame molecule x seed, top1=..., top25=...) for the run's cfg + overrides."""
        cfg = dict(self.S['meta'].get('cfg', {})); cfg.update(overrides); cfg.update(self.force)
        csig = _sha(effective_cfg(cfg))
        cols = {}
        for seed in self.seeds:
            hit = self.rcache.get(self.sig, csig, seed)
            if hit is None:
                S2 = dict(self.S); S2['base'] = self.base(seed)
                sub, _, rr = refuse.run_one(S2, self.mods, cfg, self.labels, self.pool_pop)
                lists = dict(zip(sub.molecule_id, sub.smiles.str.split(';')))
                key = lambda s: metric.ik14(s) or s          # same key as compare_configs.py
                hit = dict(rr=[float(rr[m]) for m in self.mids], top1=[key(lists[m][0]) for m in self.mids],
                           top25=[hashlib.md5('|'.join(key(x) for x in lists[m][:25]).encode()).hexdigest()[:12]
                                  for m in self.mids])
                self.rcache.put(self.sig, csig, seed, hit)
            cols[seed] = hit
        return dict(rr=pd.DataFrame({s: h['rr'] for s, h in cols.items()}, index=self.mids),
                    top1={s: h['top1'] for s, h in cols.items()}, top25={s: h['top25'] for s, h in cols.items()})


def paired(res, ref):
    """Paired stats on the per-molecule mean over seeds; change counts are averaged over seeds."""
    r, r0 = res['rr'].mean(axis=1), ref['rr'].mean(axis=1)
    d = (r - r0).values
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ch = lambda k: float(np.mean([sum(a != b for a, b in zip(res[k][s], ref[k][s])) for s in res[k]]))
    return dict(mrr=float(r.mean()), top1=float((res['rr'] == 1).values.mean()), delta=float(d.mean()), se=se,
                wins=int((d > 1e-9).sum()), losses=int((d < -1e-9).sum()), top1_changed=ch('top1'),
                top25_changed=ch('top25'), n=n)


def verdict(lib, c2):
    """The qualification rule of the search: 'sim+' = not worse on library hits (delta >= -1 SE) and better in the
    class-2 sim by more than 2 SE; 'lib+' = better on library hits by more than 2 SE without hurting the sim."""
    v = []
    if lib['delta'] >= -lib['se'] and c2['delta'] > 2 * c2['se'] and c2['delta'] > 0:
        v.append('sim+')
    if lib['delta'] > 2 * lib['se'] and lib['delta'] > 0 and c2['delta'] >= -c2['se']:
        v.append('lib+')
    return '&'.join(v) or '-'


def evaluate_spec(spec, seeds=None, cache_dir=None, use_result_cache=True, log=print):
    """-> (rows, per-regime rr DataFrames molecule x config, regime infos). Importable by search scripts."""
    cache_dir = cache_dir or os.path.join(RUNS, 'search', 'cache')
    ice_fuse, gl_fuse, score_key, formula = refuse.load_modules()
    kc = KeyCache(os.path.join(cache_dir, 'keys.json'), score_key, formula)
    mods = (ice_fuse, gl_fuse, kc.fn('score_key'), kc.fn('formula'))
    pool_pop = refuse.load_pool_pop()
    rcache = ResultCache(os.path.join(cache_dir, 'results.sqlite') if use_result_cache else None)
    regimes = {n: Regime(n, r, mods, pool_pop, rcache, seeds) for n, r in (spec.get('regimes') or DEFAULT_REGIMES).items()}
    comp = dict(DEFAULT_COMPOSITE); comp.update(spec.get('composite') or {})
    for n, g in regimes.items():
        log(f'regime {n}: {g.info}')
    base_set = spec.get('base_set', {})
    sets = {n: dict(base_set, **c.get('set', {})) for n, c in spec['configs'].items()}
    ref_name = spec['reference']
    assert ref_name in sets, f'reference {ref_name!r} is not one of the configs'
    order = [ref_name] + [n for n in sets if n != ref_name]
    res, rows, t_all = {}, [], time.time()
    for i, name in enumerate(order):
        t0 = time.time()
        res[name] = {rn: g.evaluate(sets[name]) for rn, g in regimes.items()}
        row = dict(config=name, set=spec['configs'][name].get('set', {}))
        for rn in regimes:
            for k, v in paired(res[name][rn], res[ref_name][rn]).items():
                row[f'{rn}.{k}'] = v
        c1, c2 = comp['class1'], comp['class2']
        if c1 in regimes and c2 in regimes:
            for sh in comp['shares']:
                row[f'comp@{sh:g}'] = sh * row[f'{c1}.delta'] + (1 - sh) * row[f'{c2}.delta']
                row[f'comp@{sh:g}.se'] = float(np.hypot(sh * row[f'{c1}.se'], (1 - sh) * row[f'{c2}.se']))
            row['verdict'] = '(ref)' if name == ref_name else verdict(
                {k: row[f'{c1}.{k}'] for k in ('delta', 'se')}, {k: row[f'{c2}.{k}'] for k in ('delta', 'se')})
        rows.append(row)
        if time.time() - t0 > 5:                               # cold caches: keep what was computed
            kc.save(); rcache.commit()
        log(f'[{i + 1}/{len(order)}] {name:28s} ' + '  '.join(
            f'{rn} {row[f"{rn}.mrr"]:.4f} ({row[f"{rn}.delta"]:+.4f}±{row[f"{rn}.se"]:.4f})' for rn in regimes)
            + f'  {time.time() - t0:.1f}s')
    kc.save(); rcache.commit()
    log(f'{len(order)} configs in {time.time() - t_all:.0f}s')
    rr = {rn: pd.DataFrame({n: res[n][rn]['rr'].mean(axis=1) for n in order}) for rn in regimes}
    return rows, rr, {n: g.info for n, g in regimes.items()}, comp


def table(rows, regimes, comp, markdown=False):
    """One line per config. Deltas are versus the reference; top-1 changed is the mean count over seeds."""
    out = []
    for r in rows:
        line = {'config': r['config']}
        for rn in regimes:
            line[f'{rn} MRR'] = f"{r[f'{rn}.mrr']:.4f}"
            line[f'{rn} Δ±SE'] = f"{r[f'{rn}.delta']:+.4f}±{r[f'{rn}.se']:.4f}"
            line[f'{rn} W/L'] = f"{r[f'{rn}.wins']}/{r[f'{rn}.losses']}"
            line[f'{rn} top1Δ'] = f"{r[f'{rn}.top1_changed']:.4g}"
        for sh in comp['shares']:
            if f'comp@{sh:g}' in r:
                line[f'comp@{sh:g}'] = f"{r[f'comp@{sh:g}']:+.4f}"
        if 'verdict' in r:
            line['verdict'] = r['verdict']
        out.append(line)
    df = pd.DataFrame(out)
    if not markdown:
        return df.to_string(index=False)
    head = '| ' + ' | '.join(df.columns) + ' |\n|' + '---|' * len(df.columns)
    return head + '\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in df.values)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('spec'); ap.add_argument('--out'); ap.add_argument('--seeds', type=int)
    ap.add_argument('--cache-dir'); ap.add_argument('--no-result-cache', action='store_true')
    a = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):                     # Windows consoles: the table uses ± and Δ
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    spec = json.load(open(a.spec, encoding='utf-8'))
    rows, rr, infos, comp = evaluate_spec(spec, a.seeds, a.cache_dir, not a.no_result_cache,
                                          log=lambda *x: print(*x, flush=True))
    print(table(rows, infos, comp))
    out = a.out or os.path.splitext(a.spec)[0] + '_regimes'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(dict(reference=spec['reference'], regimes=infos, composite=comp, rows=rows),
              open(out + '.json', 'w', encoding='utf-8'), indent=1)
    open(out + '.md', 'w', encoding='utf-8').write(table(rows, infos, comp, markdown=True) + '\n')
    for rn, df in rr.items():
        df.to_csv(f'{out}_rr_{rn}.csv', index_label='molecule_id')
    print('wrote', out + '.json', out + '.md', 'and per-molecule rr CSVs')


if __name__ == '__main__':
    main()
