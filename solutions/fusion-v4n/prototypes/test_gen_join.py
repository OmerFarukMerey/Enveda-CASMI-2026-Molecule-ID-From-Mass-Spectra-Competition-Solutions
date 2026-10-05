"""Tests of gen_join.py without competition data: parent selection, record building, and one real analog_derive call.

    python solutions/fusion-v4n/prototypes/test_gen_join.py
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_join as GJ


def test_parents():
    smi = ['A', 'B', 'C', 'D']
    an = [(2, 0.9, 0, 0), (0, 0.5, 0, 0), (3, 0.31, 0, 0), (1, 0.2, 0, 0)]
    assert GJ.parents(an, smi, 0.3, 20) == [('C', 0.9), ('A', 0.5), ('D', 0.31)]
    assert GJ.parents(an, smi, 0.3, 2) == [('C', 0.9), ('A', 0.5)]
    assert GJ.parents(an, smi, 0.3, 20, skip_sids=[2]) == [('A', 0.5), ('D', 0.31)]
    assert GJ.parents([], smi) == []


def _fake_propose(par, target, ppm=10.0, top=None, exclude_keys=(), budget=400):
    return [SimpleNamespace(smiles=s, support=sup) for s, sup in (('s1', 3.0), ('s2', 2.0), ('s2b', 1.5), ('bad', 1.2), ('far', 1.1), ('s3', 1.0), ('s4', 0.5))]


def _rec(s):
    if s == 'bad':
        return None
    return dict(smiles=s, key={'s2b': 'K_s2'}.get(s, 'K_' + s), mass=300.0 + (0.05 if s == 'far' else 0.0))


def test_records_order_dedup_cap():
    r = GJ.records([('P', 0.8)], 300.0, {'K_s1'}, 3, _rec, propose=_fake_propose)
    assert [x['key'] for x in r] == ['K_s2', 'K_s3', 'K_s4']             # s1 excluded, s2b a twin, bad unparsable, far off-mass
    assert all(x['src'] == 'gen' for x in r) and [x['gen_rank'] for x in r] == [1, 5, 6] and r[0]['gen_support'] == 2.0
    assert len(GJ.records([('P', 0.8)], 300.0, set(), 2, _rec, propose=_fake_propose)) == 2
    assert GJ.records([('P', 0.8)], 300.0, set(), 0, _rec, propose=_fake_propose) == []
    assert GJ.records([], 300.0, set(), 5, _rec, propose=_fake_propose) == []


def test_with_analog_derive():
    try:
        from rdkit import Chem
        from rdkit.Chem.Descriptors import ExactMolWt
    except Exception:
        print('rdkit missing: skipped'); return
    parent = 'COc1ccc(C=CC(=O)O)cc1O'                                      # isoferulic acid
    truth = 'COc1ccc(C=CC(=O)OC)cc1O'                                      # its methyl ester
    target = ExactMolWt(Chem.MolFromSmiles(truth))
    key = lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14]
    rec = lambda s: dict(smiles=s, key=key(s), mass=ExactMolWt(Chem.MolFromSmiles(s)))
    r = GJ.records([(parent, 0.9)], target, set(), 30, rec)
    assert r and key(truth) in {x['key'] for x in r} and len({x['key'] for x in r}) == len(r)
    assert all(abs(x['mass'] - target) <= target * 30e-6 for x in r)
    assert key(truth) not in {x['key'] for x in GJ.records([(parent, 0.9)], target, {key(truth)}, 30, rec)}


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); n += 1
    print(f'{n} tests passed')
