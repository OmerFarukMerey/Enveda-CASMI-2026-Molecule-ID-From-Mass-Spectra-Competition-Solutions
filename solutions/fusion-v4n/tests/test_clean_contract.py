"""Behaviour pins for the clean-chain rewrite (clean/chemcore.py, clean/pool_build.py): data-free rule tests, and golden
tests against stored outputs of the current chain ($CASMI_DATA/clean/golden, built by clean/golden.py; skipped if absent)."""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'clean'))
import chemcore as C

GOLD = os.path.join(os.environ.get('CASMI_DATA', 'data'), 'clean', 'golden')
SKIP = []


def _gold(name):
    p = os.path.join(GOLD, name)
    if not os.path.exists(p):
        SKIP.append(name); return None
    return p


# ---- data-free
def test_adduct_arithmetic():
    H, e = C.mono('H'), C.ELECTRON
    assert abs(C.parse_adduct('[M+H]+')[2] - (H - e)) < 1e-12
    assert abs(C.parse_adduct('[M-H]-')[2] - (-H + e)) < 1e-12
    n, z, d, s = C.parse_adduct('[2M+Na]+'); assert (n, z, s) == (2, 1, 1) and abs(d - (C.mono('Na') - e)) < 1e-12
    n, z, d, s = C.parse_adduct('[M+2H]2+'); assert (n, z) == (1, 2) and abs(d - (2 * H - 2 * e)) < 1e-12
    assert abs(C.parse_adduct('[M-2H2O+H]+')[2] - (H - 2 * C.formula_mass('H2O') - e)) < 1e-12
    assert C.parse_adduct('[M+CH2O2-H]-')[2] == C.parse_adduct('[M+HCOO]-')[2]
    assert C.parse_adduct('M+H') is None and np.isnan(C.neutral_mass(100.0, 'junk'))
    assert abs(C.neutral_mass(195.0877, '[M+H]+') - 194.0804) < 1e-3          # caffeine


def test_clean_peaks_rules():
    mz = np.array([50.0, 60.0, 100.0, 0.0, 103.0]); it = np.array([0.5, 0.0004, 1.0, 0.9, 5.0])
    m, v = C.clean_peaks(mz, it, 100.0)             # 103 is above precursor + 2; 60 is below 0.1% of the base; m/z 0 dropped
    assert m.tolist() == [50.0, 100.0] and v.tolist() == [0.5, 1.0] and m.dtype == np.float32
    m, v = C.clean_peaks(np.arange(1, 601.0), np.arange(1, 601.0), 1000.0)
    assert len(m) == 512 and m[0] == 89.0 and abs(v.max() - 1.0) < 1e-7 and np.all(np.diff(m) > 0)
    assert len(C.clean_peaks(np.array([500.0]), np.array([1.0]), 100.0)[0]) == 0


def test_fragment_rules():
    f = C.fragment_masses('CO')                     # methanol: parent, CH3, OH
    assert np.allclose(f, [15.023, 17.003, 32.026], atol=1e-3)
    f1, f2 = C.fragment_masses('C1CCCCC1', 1), C.fragment_masses('C1CCCCC1', 2)
    assert len(f1) == 1 and len(f2) == 6            # one ring bond never splits a ring; two give CH2 chains of 1..5 + parent
    assert np.allclose(f2[:2], [14.016, 28.031], atol=1e-3)


# ---- golden (stored outputs of the current chain)
def test_gold_neutral_mass():
    p = _gold('adducts.json')
    if not p:
        return
    n = bad = 0
    for r in json.load(open(p)):
        got = C.neutral_mass(r['mz'], r['adduct'])
        if r['nm'] is None:
            continue                                 # the current chain has no entry for this adduct; ours may parse it
        n += 1; bad += not abs(got - r['nm']) < 2e-6
    assert n > 100 and bad == 0, (n, bad)


def test_gold_clean_peaks():
    p = _gold('spectra.npz')
    if not p:
        return
    z = np.load(p); bad = 0
    for j in range(len(z['rows'])):
        m, v = C.clean_peaks(z[f'mz{j}'], z[f'it{j}'], float(z['prec'][j]))
        bad += not (len(m) == len(z[f'cmz{j}']) and np.array_equal(m, z[f'cmz{j}']) and np.allclose(v, z[f'cit{j}'], atol=1e-7))
    assert bad == 0, bad


def test_gold_fragments():
    p = _gold('frags.npz')
    if not p:
        return
    z = np.load(p); bad = 0; n = len(z['pid'])
    for j in range(0, n, 4):                         # 100 structures
        f = C.fragment_masses(str(z['smiles'][j])); g = z[f'f{j}'].astype(np.float64)
        bad += not (len(f) == len(g) and np.abs(f - g).max() < 6e-4)
    assert bad == 0, bad


def test_gold_pool_rows():
    p = _gold('pool_sample.parquet')
    if not p:
        return
    import pandas as pd, pool_build as P
    d = pd.read_parquet(p).head(300); ref = np.load(os.path.join(GOLD, 'pool_sample_fp.npy'))[:300]; bits = np.load(os.path.join(GOLD, 'fp_bits_ref.npy'))
    bad = dict(key=0, formula=0, mass=0, fp=0, smiles=0)
    for i, r in enumerate(d.itertuples()):
        st = P.standardise(r.smiles); rec, fp = P.record(st[0])
        bad['smiles'] += st[0] != r.smiles; bad['key'] += rec['key'] != r.key; bad['formula'] += rec['formula'] != r.formula
        bad['mass'] += abs(rec['mass'] - r.mass) > 1e-6
        bad['fp'] += not np.array_equal(np.unpackbits(fp)[bits], np.unpackbits(ref[i])[:len(bits)])
    assert not any(bad.values()), bad


if __name__ == '__main__':
    for k, v in sorted(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
    print('skipped (no golden fixture):', SKIP or 'none')
