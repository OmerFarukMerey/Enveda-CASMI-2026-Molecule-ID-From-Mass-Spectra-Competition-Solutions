"""Unit tests for nb/embed/frag_rescore.py (MetFrag-lite re-scorer); synthetic spectra only, no competition data.

usage: python tests/test_frag_rescore.py   (or pytest)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'nb', 'embed'))
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

import frag_rescore as fr

A = 'CCOC(=O)c1ccc(O)cc1'   # ethyl 4-hydroxybenzoate, C9H10O3
B = 'COC(=O)Cc1ccc(O)cc1'   # methyl 4-hydroxyphenylacetate, same formula
ETOH = 'CCO'
M_ETOH = 2 * 12.0 + 6 * fr.H + fr.AMU['O']


def spec(mz, it=None, adduct='[M+H]+', prec=None, mode=1):
    mz = np.asarray(mz, float)
    it = np.ones(len(mz)) if it is None else np.asarray(it, float)
    return dict(mz=mz, it=it / it.max(), adduct=adduct, prec=prec if prec is not None else float(mz.max()), mode=mode)


def _all_ions(smi, cfg=None):
    c = fr.resolve_cfg(cfg)
    lv = fr.candidate_levels(smi, c)
    m = np.concatenate(lv)
    return np.sort(np.concatenate([m + fr.PROTON + dh * fr.H for dh in c['h_shifts']]))


def _far_from(x, ref, tol):
    i = np.searchsorted(ref, x)
    lo = np.abs(ref[np.clip(i - 1, 0, len(ref) - 1)] - x)
    hi = np.abs(ref[np.clip(i, 0, len(ref) - 1)] - x)
    return np.minimum(lo, hi) > tol


def test_isomers_same_formula():
    assert CalcMolFormula(Chem.MolFromSmiles(A)) == CalcMolFormula(Chem.MolFromSmiles(B))


def test_parse_failure_is_none():
    s = [spec([M_ETOH + fr.PROTON])]
    out = fr.score_molecule(s, ['not_a_smiles', ETOH, '', 'C1CC'])
    assert out[0] is None and out[2] is None and out[3] is None
    assert isinstance(out[1], float) and out[1] > 0.99


def test_levels_match_reference_enumeration():
    """Same fragment mass set as the pv.py MetFrag-lite (pv.fragment_masses) for <=2 cleavages."""
    try:
        sys.path.insert(0, os.path.join(ROOT, 'nb', 'embed', 'eng'))
        import pv
    except Exception:
        return  # pv needs numba; the check is optional
    for smi in (A, B, 'O=C(O)C1(O)CC(O)C(O)C(O)C1'):
        ours = np.sort(np.concatenate(fr.fragment_levels(smi)))
        ref = np.unique(np.round(pv.fragment_masses(smi, max_bonds=500), 5))
        assert np.array_equal(np.unique(np.round(ours, 5)), ref)


def test_own_fragments_beat_isomer():
    lv = fr.candidate_levels(A)
    peaks = np.concatenate([lv[0], lv[1]]) + fr.PROTON
    # the isomer explains most of these too (shared pieces, H shifts); some must be A-only or the test is vacuous
    only_a = _far_from(peaks, _all_ions(B), 0.01)
    assert only_a.sum() >= 2
    rng = np.random.default_rng(0)
    it = rng.uniform(0.2, 1.0, len(peaks))
    s = [spec(peaks, it, prec=float(lv[0][0] + fr.PROTON))]
    for cfg in (None, fr.BASELINE_CFG):
        sa, sb = fr.score_molecule(s, [A, B], cfg)
        assert abs(sa - 1.0) < 1e-9 and sb < sa - 0.1, (cfg, sa, sb)
    # only the A-specific peaks: B explains none of them
    sa, sb = fr.score_molecule([spec(peaks[only_a])], [A, B])
    assert abs(sa - 1.0) < 1e-9 and sb == 0.0


def test_two_bond_fragments_get_w2_credit():
    lv = fr.candidate_levels(A)
    one = np.sort(np.concatenate([lv[0], lv[1]]))
    ions1 = np.sort(np.concatenate([one + fr.PROTON + dh * fr.H for dh in range(-2, 4)]))
    cand = lv[2] + fr.PROTON
    cand = cand[_far_from(cand, ions1, 0.01)]
    assert len(cand) > 0
    s = [spec(cand[:1])]
    assert abs(fr.score_molecule(s, [A])[0] - 0.6) < 1e-9
    assert abs(fr.score_molecule(s, [A], dict(w2=1.0))[0] - 1.0) < 1e-9
    assert fr.score_molecule(s, [A], dict(max_breaks=1))[0] == 0.0


def test_tolerance():
    s = [spec([M_ETOH + fr.PROTON + 0.007])]
    assert fr.score_molecule(s, [ETOH], dict(tol=0.005))[0] == 0.0
    assert fr.score_molecule(s, [ETOH], dict(tol=0.01))[0] == 1.0
    assert fr.score_molecule(s, [ETOH])[0] == 0.0  # default tol is 0.005


def test_h_shift_range():
    s = [spec([M_ETOH + fr.PROTON + 3 * fr.H])]
    assert fr.score_molecule(s, [ETOH])[0] == 1.0                    # default -2..+3
    assert fr.score_molecule(s, [ETOH], fr.BASELINE_CFG)[0] == 0.0   # baseline -2..+2


def test_adduct_carriers():
    na = M_ETOH + fr.AMU['Na'] - fr.E
    assert fr.score_molecule([spec([na], adduct='[M+Na]+')], [ETOH])[0] == 1.0
    assert fr.score_molecule([spec([na], adduct='[M+H]+')], [ETOH])[0] == 0.0
    neg = M_ETOH - fr.PROTON
    assert fr.score_molecule([spec([neg], adduct='[M-H]-', mode=-1)], [ETOH])[0] == 1.0
    cl = M_ETOH + fr.AMU['Cl'] + fr.E
    assert fr.score_molecule([spec([cl], adduct='[M+Cl]-', mode=-1)], [ETOH])[0] == 1.0
    assert fr.score_molecule([spec([cl], adduct='[M-H]-', mode=-1)], [ETOH])[0] == 0.0
    # formate adduct keeps the formate on the precursor ion, and unknown adducts fall back to the ion mode
    fa = M_ETOH + fr.FORMATE - fr.PROTON
    assert fr.score_molecule([spec([fa], adduct='[M+CH2O2-H]-', mode=-1)], [ETOH])[0] == 1.0
    assert fr.score_molecule([spec([neg], adduct='[M+X]-', mode=-1)], [ETOH])[0] == 1.0
    # [M-H2O+H]+ precursor is explained by an OH-loss fragment with an H shift
    assert fr.score_molecule([spec([M_ETOH - fr.H2O + fr.PROTON], adduct='[M-H2O+H]+')], [ETOH])[0] == 1.0


def test_intensity_modes_and_mean_over_spectra():
    good = M_ETOH + fr.PROTON
    s1 = spec([good, 300.0], [1.0, 0.25])
    lin = fr.score_molecule([s1], [ETOH])[0]
    sq = fr.score_molecule([s1], [ETOH], dict(intensity='sqrt'))[0]
    assert abs(lin - 1 / 1.25) < 1e-9 and abs(sq - 1 / 1.5) < 1e-9
    s2 = spec([300.0])
    assert abs(fr.score_molecule([s1, s2], [ETOH])[0] - lin / 2) < 1e-9
    empty = dict(mz=np.zeros(0), it=np.zeros(0), adduct='[M+H]+', prec=47.0, mode=1)
    assert abs(fr.score_molecule([s1, empty], [ETOH])[0] - lin) < 1e-9  # empty spectra are ignored
    assert fr.score_molecule([empty], [ETOH, 'xx']) == [0.0, None]


def test_deterministic_and_cache():
    lv = fr.candidate_levels(A)
    rng = np.random.default_rng(1)
    mz = np.concatenate([np.concatenate(lv[:3])[:40] + fr.PROTON, rng.uniform(50, 200, 20)])
    s = [spec(mz, rng.uniform(0.01, 1, len(mz))), spec(mz[::2], adduct='[M+Na]+')]
    cands = [A, B, ETOH, 'bad', 'CCCCCCCCO']
    r1 = fr.score_molecule(s, cands)
    cache = {}
    r2 = fr.score_molecule(s, cands, cache=cache)
    r3 = fr.score_molecule(s, cands, cache=cache)
    r4 = fr.score_molecule(s, cands[::-1])[::-1]
    assert r1 == r2 == r3 == r4
    assert len(cache) == len(cands)


def test_cfg_validation():
    try:
        fr.resolve_cfg(dict(w_2=0.5))
    except KeyError:
        pass
    else:
        raise AssertionError('unknown cfg key accepted')
    assert fr.resolve_cfg(None)['h_shifts'] == (-2, -1, 0, 1, 2, 3)


def test_large_molecule_is_capped():
    big = 'C' * 120  # 119 bonds > max_bonds_2: only single cleavages, still scores
    lv = fr.fragment_levels(big)
    assert len(lv) == 3 and len(lv[2]) == 0 and len(lv[1]) > 0
    assert fr.score_molecule([spec([200.0])], [big])[0] >= 0.0


if __name__ == '__main__':
    n = 0
    for name, f in sorted(globals().items()):
        if name.startswith('test_') and callable(f):
            f()
            n += 1
            print('ok', name)
    print(f'{n} tests passed (numba={fr.HAVE_NUMBA})')
