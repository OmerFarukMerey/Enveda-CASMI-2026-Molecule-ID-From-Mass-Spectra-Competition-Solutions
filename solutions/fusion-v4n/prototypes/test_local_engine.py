"""Tests of prototypes/local_engine.py helpers that need no competition data (masks, list order, safe unpickling).

usage: python solutions/fusion-v4n/prototypes/test_local_engine.py
"""
import io
import os
import pickle
import sys
import tempfile
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_engine as LE


def test_truth_mask_by_metric_key_and_library():
    struct_key = np.array(['K1', 'K2', 'K1', 'K3'], object)          # sid 0 and 2 are tautomer twins (same metric key)
    sid = np.array([0, 1, 2, 2, 3, -1, 0]); lib = np.array([0, 0, 3, 9, 0, 0, 9])
    assert LE.truth_mask(struct_key, sid, lib, 'K1').tolist() == [True, False, True, True, False, False, True]     # twins purged too
    assert LE.truth_mask(struct_key, sid, lib, 'K1', lib=9).tolist() == [False, False, False, True, False, False, True]
    assert not LE.truth_mask(struct_key, sid, lib, 'absent').any()


def test_dedup_order_and_base_list():
    score = np.array([0.1, 0.9, 0.9, 0.5, 0.7]); keys = ['a', 'b', 'c', 'b', 'd']
    assert LE.dedup_order(score, keys) == [1, 2, 4, 0]                 # stable for ties, key 'b' kept once (best score)
    assert LE.dedup_order(score, keys, topn=2) == [1, 2]
    C = types.SimpleNamespace(pid=np.array([5, 6, -1, 6, 8]), key=keys, smiles=list('ABCDE'), formula=['f'] * 5)
    b = LE.base_list(dict(C=C, score=score, lib_max=0.25), topn=3)
    assert b == [['B', 'C', 'E'], ['b', 'c', 'd'], 0.25, [0.9, 0.9, 0.7], ['f', 'f', 'f'], [6, -1, 8]]
    empty = types.SimpleNamespace(pid=np.zeros(0, int), key=[], smiles=[], formula=[])
    assert LE.base_list(dict(C=empty, score=np.zeros(0), lib_max=0.0)) == [[], [], 0.0, [], [], []]


def test_target_mass_and_spectra():
    assert LE.target_mass([np.nan, 100.0, 102.0, 101.0]) == 101.0 and LE.target_mass([np.nan]) is None
    rows = pd.DataFrame(dict(ms2_mzs=[np.array([50.0, 60.0])], ms2_normalized_intensities=[np.array([1.0, 0.5])], ionization_mode=['negative'],
                             adduct=['[M-H]-'], precursor_mz=[99.0], collision_energy_ev=[np.array([10.0, 30.0])]))
    s = LE.spectra_from_rows(rows)[0]
    assert s['mode'] == -1 and s['ce'] == 20.0 and s['ce_n'] == 2 and s['prec'] == 99.0
    rows['collision_energy_ev'] = [None]
    s = LE.spectra_from_rows(rows)[0]
    assert np.isnan(s['ce']) and s['ce_n'] == 1


def test_restricted_unpickler():
    ok = dict(features=['a', 'b'], boosters=['tree'], arr=np.arange(3, dtype=np.float32), t=(1, 2.5, None))
    back = LE.safe_pickle_load(io.BytesIO(pickle.dumps(ok)))
    assert back['features'] == ['a', 'b'] and back['arr'].tolist() == [0.0, 1.0, 2.0]

    class Evil:
        def __reduce__(self):
            return (os.system, ('echo should-not-run',))
    try:
        LE.safe_pickle_loads(pickle.dumps(Evil()))
        raise AssertionError('a callable global was unpickled')
    except pickle.UnpicklingError:
        pass


def test_link_work_reports_missing():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'src'); os.makedirs(src)
        for f in LE.WORK_FILES[:3]:
            open(os.path.join(src, f), 'wb').write(b'x')
        miss = LE.link_work(os.path.join(d, 'work'), [src])
        assert sorted(miss) == sorted(LE.WORK_FILES[3:])
        assert all(os.path.exists(os.path.join(d, 'work', f)) for f in LE.WORK_FILES[:3])


def test_logit_cache_shares_items_between_banks_and_runs():
    calls = []

    def fake(items):                               # stands in for ModelBank.logits_el: one row of "logits" per item
        calls.append(len(items))
        return np.array([[it['prec'], len(it['mz'])] for it in items], np.float32), np.zeros((len(items), 10), np.float32)

    def item(prec, noise=0.0):
        return dict(mz=np.array([50.0, 60.5], np.float32) + noise, it=np.array([1.0, 0.5], np.float32), prec=prec, adduct_ix=0, ce=0.0,
                    ce_known=0.0, n_merged=1, mode=1.0)
    with tempfile.TemporaryDirectory() as d:
        c = LE.LogitCache(d); f = c.wrap(fake, 'ckpt')
        z, el = f([item(100.0), item(200.0)])
        assert z[:, 0].tolist() == [100.0, 200.0] and calls == [2]
        z, _ = f([item(200.0), item(300.0), item(100.0, noise=1e-5)])        # float-noise twin of the first item is a hit
        assert z[:, 0].tolist() == [200.0, 300.0, 100.0] and calls == [2, 1] and c.hits == 2 and c.misses == 3
        c2 = LE.LogitCache(d); g = c2.wrap(lambda items: (_ for _ in ()).throw(AssertionError('recomputed')), 'ckpt')
        assert g([item(300.0)])[0][0, 0] == 300.0                             # a second process reads the entry from disk
        e = c2.wrap(lambda items: tuple(x[None] for x in fake(items)), 'other', each=True)   # fpnet6 API: (1, n, nbits)
        assert e([item(100.0)])[0].shape == (1, 1, 2) and calls[-1] == 1      # another checkpoint tag does not share


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
