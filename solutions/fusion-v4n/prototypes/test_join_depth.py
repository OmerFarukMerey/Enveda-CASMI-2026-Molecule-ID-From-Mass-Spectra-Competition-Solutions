"""Tests of join_depth.py without competition data: cache-clear interval, runtime budget, depth read-out.

    python solutions/fusion-v4n/prototypes/test_join_depth.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import join_depth as JD


def test_clear_every():
    assert JD.clear_every(0) == 50
    assert JD.clear_every(50) == 48 and JD.clear_every(100) == 24 and JD.clear_every(200) == 12
    assert JD.clear_every(1000) == 2 and JD.clear_every(100000) == 1
    for n in (25, 50, 100, 200, 500, 1000):
        assert JD.clear_every(n) * JD.MB_PER_ROW * n <= 1600.0 + JD.MB_PER_ROW * n


def test_budget_monotone_and_within_limit():
    b = [JD.budget(n) for n in (50, 100, 200, 500, 1000)]
    assert all(x['run_minutes'] < y['run_minutes'] for x, y in zip(b, b[1:]))
    assert all(x['cache_mb_peak'] <= 1600.0 for x in b)
    assert b[1]['run_minutes'] < 15 and b[-1]['run_minutes'] < 150          # N 100: minutes; N 1000: still inside the 9 h
    assert JD.budget(0)['run_minutes'] == 0.0 and JD.budget(0)['cache_mb_peak'] == 0.0
    assert JD.budget(200, pc_workers=1)['channel_s_per_molecule'] > JD.budget(200, pc_workers=4)['channel_s_per_molecule']


def test_depth_readout():
    pc = [f'P{i}' for i in range(120)]
    base = ['B1', 'P0', 'P60', 'B2', 'P110'] + [f'B{i}' for i in range(3, 30)]
    final = ['B1', 'P60', 'X']
    d = JD.depth_readout(pc, base, final)
    assert d['base25_1_25'] == 1 and d['base25_26_50'] == 0 and d['base25_51_100'] == 1 and d['base25_101_200'] == 1
    assert d['final25_1_25'] == 0 and d['final25_51_100'] == 1
    assert 'base25_201_500' not in d                                         # the list is only 120 deep
    assert JD.depth_readout([], base) == {}
    assert set(JD.depth_readout(pc[:10], base)) == {'base25_1_25'}


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); n += 1
    print(f'{n} tests passed')
