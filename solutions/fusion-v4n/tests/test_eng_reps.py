"""ENG_REPS: representative spectra per library structure in engine 2's analog search (default 1 = the reference rule).

Checks without competition data: the flag exists and is 1; the cell hands it to the runner and the runner replaces casmi_engine._reps only
when it is above 1; reps_k(E, 1) selects the rows of casmi_engine._reps on a synthetic library; reps_k(E, 2) keeps that row and adds the next
one of the same order (timsTOF first, then most peaks); top_analogs returns the best similarity per structure over its representatives.
The engine modules need numba and pandas; the functional part is skipped when they are not installed.
usage: python tests/test_eng_reps.py   (or pytest)
"""
import os, sys

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed', 'eng'))


def load_cfg():
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    return ns['CFG']


def test_flag_default_and_wiring():
    assert load_cfg()['ENG_REPS'] == 1
    cell = open(os.path.join(SOL, 'nb', '03_engine2.py'), encoding='utf-8').read()
    assert "REPS=CFG['ENG_REPS']" in cell
    run = open(os.path.join(SOL, 'nb', 'embed', 'eng', 'eng_runner.py'), encoding='utf-8').read()
    assert "REPS=1)" in run and "if int(C['REPS']) > 1:" in run and run.count('E._reps = ') == 1, 'the reference function is replaced only above 1'


def _library(np):
    rnd = np.random.RandomState(0); n = 400
    npk = rnd.randint(1, 30, n); off = np.r_[0, np.cumsum(npk)].astype(np.int64)
    mz = np.concatenate([np.sort(rnd.uniform(50, 500, k)) for k in npk]).astype(np.float32); it = rnd.uniform(0.01, 1, off[-1]).astype(np.float32)
    row_p = rnd.randint(0, 60, n).astype(np.int64); nm = 100.0 + row_p * 3.0 + rnd.uniform(0, 1e-4, n)
    return dict(off=off, mz=mz, it=it, row_p=row_p, nm=nm, tims=rnd.rand(n) < 0.4, pol=np.where(rnd.rand(n) < 0.5, 1, -1).astype(np.int8)), npk


def test_reps():
    try:
        import numpy as np, pandas as pd
        import casmi_engine as E, eng_runner
    except ImportError as e:
        print('engine modules not importable (%s): skipped' % e); return
    L, npk = _library(np); rows = np.arange(len(npk))
    a = E._reps(L, rows); b = eng_runner.reps_k(E, 1)(L, rows)
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    c = eng_runner.reps_k(E, 2)(L, rows)
    assert set(a['rep']) <= set(c['rep']) and len(c['rep']) == sum(min(2, v) for v in pd.Series(L['row_p']).value_counts())
    for p in range(60):
        r = rows[L['row_p'] == p]
        want = sorted(r, key=lambda i: (-int(L['tims'][i]), -int(npk[i]), i))[:2]
        assert sorted(c['rep'][c['rep_p'] == p]) == sorted(want), p
    assert np.all(np.diff(c['rep_nm']) >= 0), 'representatives stay sorted by neutral mass'
    assert len(eng_runner.reps_k(E, 10 ** 9)(L, rows)['rep']) == len(rows)
    n0 = E.CFG.N_ANALOG
    try:
        E.CFG.N_ANALOG = 5
        assert E.top_analogs(np.array([7, 3, 7, 3, 9]), np.array([0.2, 0.5, 0.9, 0.1, 0.4], np.float32)) == [(7, np.float32(0.9)), (3, 0.5), (9, np.float32(0.4))]
    finally:
        E.CFG.N_ANALOG = n0


if __name__ == '__main__':
    for n, f in sorted(globals().items()):
        if n.startswith('test_') and callable(f):
            f(); print('ok', n)
