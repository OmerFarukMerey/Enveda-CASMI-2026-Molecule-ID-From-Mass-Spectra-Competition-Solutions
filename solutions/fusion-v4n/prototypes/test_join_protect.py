"""Tests for join_protect.py (no competition data): the four list-assembly rules, the margins, apply_all and exposure()."""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import join_protect as JP


def entry(pids, scores=None):
    n = len(pids)
    scores = list(scores) if scores is not None else [float(n - i) for i in range(n)]
    return [[f's{i}' for i in range(n)], [f'k{i}' for i in range(n)], 0.4, scores, ['C6H6'] * n, list(pids)]


J = JP.JOINED


def test_cap():
    e = entry([J, J, J, 7, J, 8])                       # three joined rows above the first pool row (k3)
    assert JP.protect(e, 'cap', 0)[1] == ['k3', 'k0', 'k1', 'k2', 'k4', 'k5']
    assert JP.protect(e, 'cap', 1)[1] == ['k0', 'k3', 'k1', 'k2', 'k4', 'k5']
    assert JP.protect(e, 'cap', 3)[1] == e[1] and JP.protect(e, 'cap', 5)[1] == e[1]
    # a generated row (pool id -1) is not a joined row: it counts as the pool's first candidate
    assert JP.protect(entry([J, -1, 5]), 'cap', 0)[1] == ['k1', 'k0', 'k2']


def test_entry_stays_consistent():
    e = entry([J, J, 4, J, 9], [5.0, 4.0, 3.5, 1.0, 0.5])
    o = JP.protect(e, 'cap', 0)
    assert o[3] == sorted(e[3], reverse=True) and o[2] == e[2]                 # scores monotone, lib_max kept
    assert sorted(o[1]) == sorted(e[1]) and len(set(o[0])) == len(e[0])       # a permutation
    assert [o[5][o[1].index(k)] for k in e[1]] == e[5]                        # pool ids travel with their rows
    assert e[1] == [f'k{i}' for i in range(5)]                                # input not mutated


def test_pin_uses_the_unjoined_top():
    e = entry([J, 3, J, 4, 5])
    assert JP.protect(e, 'pin', 0, ref_key='k3')[1] == ['k3', 'k0', 'k1', 'k2', 'k4']     # the unjoined top was k3, not k1
    assert JP.protect(e, 'pin', 1, ref_key='k3')[1] == ['k0', 'k3', 'k1', 'k2', 'k4']
    assert JP.protect(e, 'pin', 1, ref_key='gone')[1] == e[1]                             # falls back to the first pool row (already at 2)
    assert JP.protect(e, 'pin', 3, ref_key='k3')[1] == e[1]


def test_alt():
    e = entry([J, J, J, 1, 2, 3, J])
    assert JP.protect(e, 'alt', 2)[1] == ['k3', 'k0', 'k4', 'k1', 'k2', 'k5', 'k6']
    assert JP.protect(entry([1, 2, 3, J]), 'alt', 3)[1] == ['k0', 'k3', 'k1', 'k2']      # fewer joined rows than K


def test_conf_margin():
    # pool rows 3.0, 1.0, 0.9 -> margin 2 / sd(3, 1, 0.9) = 2.06; joined rows do not enter the margin
    e = entry([J, 5, 6, 7], [9.0, 3.0, 1.0, 0.9])
    m = JP.pool_margin(e[3], [True, False, False, False])
    assert abs(m - 2.0 / np.std([3.0, 1.0, 0.9])) < 1e-6
    assert JP.protect(e, 'conf', 0, tau=1.5)[1] == ['k1', 'k0', 'k2', 'k3']
    assert JP.protect(e, 'conf', 0, tau=2.5)[1] == e[1]                       # not confident: free
    # the unjoined list's margin, when supplied, decides instead
    assert JP.protect(e, 'conf', 0, tau=1.5, margin=0.2)[1] == e[1]
    assert JP.protect(e, 'conf', 0, tau=2.5, margin=3.0)[1] == ['k1', 'k0', 'k2', 'k3']
    assert np.isnan(JP.pool_margin([1.0, 0.5], [False, False]))               # too few pool rows: never confident
    assert JP.protect(entry([J, 5, 6]), 'conf', 0, tau=0.0)[1] == ['k0', 'k1', 'k2']


def test_unjoined_ref():
    r = JP.unjoined_ref(['a', 'b', 'a', 'c'], [1.0, 3.0, 2.5, 0.0])           # rows in any order, duplicate key 'a'
    assert r['key'] == 'b' and abs(r['margin'] - 0.5 / np.std([3.0, 2.5, 0.0])) < 1e-6
    assert JP.unjoined_ref([], [])['key'] is None


def test_noops():
    e = entry([1, 2, 3])
    assert JP.protect(e, 'cap', 0) == e and JP.protect(entry([J, J]), 'cap', 0)[1] == ['k0', 'k1']
    assert JP.protect(entry([J, 1]), None) == entry([J, 1])
    assert JP.protect([['a'], ['k'], 0.1, [1.0], ['C']], 'cap', 0) == [['a'], ['k'], 0.1, [1.0], ['C']]   # no pool ids
    assert JP.protect([[], [], 0.0, [], [], []], 'cap', 0) == [[], [], 0.0, [], [], []]
    try:
        JP.protect(entry([J, 1]), 'nonsense'); raise AssertionError
    except ValueError:
        pass


def test_apply_all():
    base = {'m1': entry([J, 5, 6, 7], [9.0, 3.0, 1.0, 0.9]), 'm2': entry([1, 2, 3]), 'm3': entry([J, 5, 6, 7], [9.0, 3.0, 2.9, 2.8])}
    assert JP.apply_all(base, {}) is base and JP.apply_all(base, dict(PC_JOIN_PROTECT=None)) is base
    msgs = []
    out = JP.apply_all(base, dict(PC_JOIN_PROTECT='conf', PC_JOIN_PROTECT_K=0, PC_JOIN_PROTECT_TAU=1.5, PC_JOIN_PROTECT_REF='joined'), log=lambda *a: msgs.append(a))
    assert out['m1'][1][0] == 'k1' and out['m2'] == base['m2'] and out['m3'][1] == base['m3'][1]
    assert '1 of 3 lists re-ordered' in msgs[-1][0]
    # unjoined reference: m3 was confident before the join, m1 was not; a molecule without a reference uses its joined list
    ref = {'m3': dict(key='k1', margin=4.0), 'm1': dict(key='k1', margin=0.1)}
    out = JP.apply_all(base, dict(PC_JOIN_PROTECT='conf', PC_JOIN_PROTECT_TAU=1.5), ref=ref)
    assert out['m3'][1][0] == 'k1' and out['m1'][1] == base['m1'][1]
    out = JP.apply_all(base, dict(PC_JOIN_PROTECT='conf', PC_JOIN_PROTECT_TAU=1.5), ref={'m3': dict(key='k1', margin=4.0)})
    assert out['m1'][1][0] == 'k1'
    # a broken entry never loses the molecule
    bad = {'m': [['a', 'b'], ['k0', 'k1'], 0.1, ['x', 'y'], ['C', 'C'], [J, 1]]}
    assert JP.apply_all(bad, dict(PC_JOIN_PROTECT='cap'))['m'][1] in (['k1', 'k0'], ['k0', 'k1'])


def test_exposure():
    ex = JP.exposure({'a': 1.0, 'b': 1.0, 'c': 0.5, 'd': 1.0}, {'a': 1.0, 'b': 0.5, 'c': 1.0, 'd': 0.0})
    assert ex['first_before'] == 3 and ex['lost'] == 2 and ex['out_of_25'] == 1 and abs(ex['rr_lost'] - 1.5) < 1e-12
    assert ex['median_rank_after'] == 2.0


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
