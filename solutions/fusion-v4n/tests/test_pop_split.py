"""POP_SPLIT: the popularity prior without spectral-library deposits (nb/embed/pop_split.py, reviews/popsplit-2026-10-08.md).

Checks, without competition data (a synthetic pool and table): flag off is today's array bit for bit and the submission of
fusion_core is the reference bit for bit; with the table only the listed rows change, to log1p(max(sid - spec, 0)) + lpmid, and the
PubMed part is untouched; a table that does not fit the pool raises (the notebook cell then falls back to the standard prior and
records the fallback); the config default is off, cell 08 reads the table only behind the flag and after the standard array, the
preset differs from v33b in this flag only, and build_notebook attaches the dataset only with the flag.
usage: python tests/test_pop_split.py   (or pytest)
"""
import json, os, random, subprocess, sys, tempfile

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import fusion_core, pop_split
from test_join_fwd_a import Fwd
from test_pop_group import case, extras

N = 200
CFG = dict(ICE_LAM=0.75, GL_LAM=0.75, POP_MU=0.25)


def pool(seed=3):
    r = np.random.RandomState(seed); sid = r.randint(0, 400, N) * (r.rand(N) < 0.9); pm = r.randint(0, 50, N) * (r.rand(N) < 0.3)
    return np.log1p(sid).astype(np.float16), np.log1p(pm).astype(np.float16), sid          # float16, as the shipped arrays


def table(sid, seed=4, n_pool=N, d=None):
    r = np.random.RandomState(seed); pid = np.sort(r.choice(np.flatnonzero(sid > 0), 40, replace=False)); spec = np.minimum(r.randint(1, 4, len(pid)), sid[pid])
    d = d or tempfile.mkdtemp()
    pd.DataFrame(dict(pid=pid.astype(np.int32), sid=sid[pid].astype(np.uint32), spec=spec.astype(np.uint16))).to_parquet(os.path.join(d, 'pool_split.parquet'), index=False)
    json.dump(dict(n_pool=n_pool), open(os.path.join(d, 'meta.json'), 'w'))
    return d, pid, spec


def run(v, ice, gl, pp, pc=None, eng=None, mu=0.25):
    sub, st = fusion_core.build_submission({'m': v}, {'m': pc} if pc else {}, {'m': eng} if eng else {}, {'m': ice}, {'m': gl}, dict(CFG, POP_MU=mu), ['m'], ['m'],
                                           ice_fuse=Fwd, gl_fuse=Fwd, formula=lambda s: 'F', score_key=lambda s: s, log=lambda *a: None, pool_pop=pp)
    return sub.smiles[0].split(';')


def test_off_is_today_bit_for_bit():
    ls, lp, sid = pool(); ref = ls.astype(np.float64) + lp.astype(np.float64)          # the two lines of cell 08
    off, n = pop_split.pool_pop(ls, lp, None)
    assert n == 0 and off.dtype == np.float64 and np.array_equal(off, ref)
    rng = random.Random(0)
    for _ in range(300):
        v, ice, gl = case(rng); pc, eng = extras(rng, v)
        for kw in (dict(), dict(pc=pc), dict(pc=pc, eng=eng)):
            for mu in (0.15, 0.25, 0.35):
                assert run(v, ice, gl, off, mu=mu, **kw) == run(v, ice, gl, ref, mu=mu, **kw)


def test_only_listed_rows_change():
    ls, lp, sid = pool(); d, pid, spec = table(sid); ref = ls.astype(np.float64) + lp.astype(np.float64)
    on, n = pop_split.pool_pop(ls, lp, d); rest = np.setdiff1d(np.arange(N), pid)
    assert n == len(pid) == 40 and np.array_equal(on[rest], ref[rest])                    # every other row bit for bit
    assert np.allclose(on[pid], np.log1p(sid[pid] - spec) + lp.astype(np.float64)[pid], rtol=0, atol=1e-12)
    assert (on[pid] <= ref[pid] + 2e-3).all() and (on[pid] < ref[pid] - 1e-3).sum() > 20    # float16 rounding of the stored count aside, never higher
    lib_only = pid[sid[pid] == spec]
    assert np.array_equal(on[lib_only], lp.astype(np.float64)[lib_only])                  # nothing but library deposits: the PubMed part remains
    rng = random.Random(1); differ = 0
    for _ in range(400):
        v, ice, gl = case(rng)
        differ += int(run(v, ice, gl, on) != run(v, ice, gl, ref))
    assert 0 < differ < 400, differ                                                       # it acts, and not on every list


def test_wrong_table_raises():
    ls, lp, sid = pool()
    d, _, _ = table(sid, n_pool=N + 1)
    for bad in (lambda: pop_split.pool_pop(ls, lp, d), lambda: pop_split.pool_pop(ls, lp, tempfile.mkdtemp()),
                lambda: pop_split.split_values(ls, lp, np.array([0, 0]), np.array([3, 3]), np.array([1, 1]), N),
                lambda: pop_split.split_values(ls, lp, np.array([N]), np.array([3]), np.array([1]), N)):
        try:
            bad()
        except Exception:
            continue
        raise AssertionError('a table that does not fit the pool must raise')


def test_wiring():
    ns = {}; exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    assert ns['CFG']['POP_SPLIT'] is False
    src = open(os.path.join(SOL, 'nb', '08_submit.py'), encoding='utf-8').read()
    i = src.index("if CFG['POP_SPLIT'] and POOL_POP is not None:")
    assert src.count('pop_split') == src[i:].count('pop_split') and src.index("np.load(os.path.join(_pp, 'pool_lsid.npy')).astype(np.float64)") < i
    assert "FAILS.append('pop split: '" in src                                           # a missing table is a recorded fallback, not a silent one
    assert 'POP_SPLIT' not in fusion_core.DEFAULTS                                        # the final stage itself is unchanged
    a = json.load(open(os.path.join(SOL, 'configs', 'v33b-v24a-fsranker-gate07-rankavg.json'))); b = json.load(open(os.path.join(SOL, 'configs', 'v53a-v33b-popsplit.json')))
    assert {k: v for k, v in b.items() if k not in ('VERSION', 'POP_SPLIT')} == {k: v for k, v in a.items() if k != 'VERSION'} and b['POP_SPLIT'] is True


def _build(cfg):
    out = tempfile.mkdtemp()
    subprocess.run([sys.executable, os.path.join(SOL, 'tools', 'build_notebook.py'), '--user', 'someone', '--out', out, '--config', os.path.join(SOL, 'configs', cfg)],
                   check=True, capture_output=True)
    return json.load(open(os.path.join(out, 'kernel-metadata.json')))['dataset_sources']


def test_build_attaches_dataset_only_with_flag():
    assert 'someone/casmi26-pop-split' not in _build('v33b-v24a-fsranker-gate07-rankavg.json')
    ds = _build('v53a-v33b-popsplit.json')
    assert 'someone/casmi26-pop-split' in ds and 'dmitriigluzdov/casmi26-pubchem-popularity-prior' in ds


if __name__ == '__main__':
    test_off_is_today_bit_for_bit(); test_only_listed_rows_change(); test_wrong_table_raises(); test_wiring(); test_build_attaches_dataset_only_with_flag()
    print('POP_SPLIT OK: off == today bit for bit (array and submission), only listed rows change, a wrong table raises, wiring and preset checked')
