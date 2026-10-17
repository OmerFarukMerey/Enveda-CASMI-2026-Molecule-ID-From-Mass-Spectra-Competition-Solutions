"""Unit tests for clean/chemrules.py (knowledge-driven same-formula re-order); synthetic spectra only, no competition data.

usage: python tests/test_chemrules.py   (or pytest)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'clean'))
import numpy as np

import chemrules as cr

O_GLC = 'OC[C@H]1O[C@@H](Oc2cc(O)c3c(c2)oc(-c2ccc(O)cc2)cc3=O)[C@H](O)[C@@H](O)[C@@H]1O'   # apigenin 7-O-glucoside
C_GLC = 'OC[C@H]1O[C@@H](c2c(O)cc(O)c3c(=O)cc(-c4ccc(O)cc4)oc23)[C@H](O)[C@@H](O)[C@@H]1O'   # vitexin (8-C-glucoside)
OTHER = 'CCCCCCCCCCCCCCCC(=O)O'                                                                  # palmitic acid, other formula
PROT = cr.PROTON
M_APG7G = 432.10565                                                                               # C21H20O10


def spec(peaks, adduct='[M+H]+', prec=M_APG7G + PROT, mode=1):
    mz = np.array([p[0] for p in peaks] + [prec]); it = np.array([p[1] for p in peaks] + [0.3])
    return dict(mz=mz, it=it, adduct=adduct, prec=prec, mode=mode)


def test_masses():
    assert abs(cr.L_HEX - 162.0528) < 1e-3 and abs(cr.L_DHEX - 146.0579) < 1e-3 and abs(cr.L_URON - 176.0321) < 1e-3
    assert abs(cr.ACYLS['caffeoyl'] - 162.0317) < 1e-3 and abs(cr.ACYLS['galloyl'] - 152.0110) < 1e-3
    assert abs(cr.L_SO3 - 79.9568) < 1e-3 and abs(cr.IONS['pchol'][0] - 184.0733) < 1e-3


def test_profile_sugars_and_acyls():
    a, b = cr.profile(O_GLC), cr.profile(C_GLC)
    assert a['formula'] == b['formula'] == 'C21H20O10'
    assert a['gly_hex'] == 1 and a['gly_O'] == 1 and a['gly_C'] == 0
    assert b['gly_C'] == 1 and b['gly_O'] == 0
    rha = cr.profile('C[C@@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@H](O)[C@H]1O')
    assert rha['gly_dhex'] == 1
    assert cr.profile('OCC1OC(O)C(O)C(O)C1O')['gly_O'] == 0          # free glucose: no glycosidic link
    cga = cr.profile('O=C(/C=C/c1ccc(O)c(O)c1)O[C@@H]1C[C@](O)(C(=O)O)C[C@@H](O)[C@H]1O')
    assert any(abs(x - cr.ACYLS['caffeoyl']) < 1e-3 for x in cga['acyl'])
    assert any(abs(x - cr.ACYLS['quinoyl']) < 1e-3 for x in cga['acyl'])
    assert cr.profile('CC(=O)Nc1ccc(OS(=O)(=O)O)cc1')['sulfate'] == 1
    assert cr.profile('not a smiles') is None


def test_evidence_hexose_loss():
    ev = cr.evidence([spec([(M_APG7G + PROT - cr.L_HEX, 1.0)])])
    assert ev['L_hex'] >= 0.99 and ev['any_sugar'] >= 0.99 and ev['L_CGLY'] == 0
    ev2 = cr.evidence([spec([(M_APG7G + PROT - 120.0423, 1.0), (M_APG7G + PROT - 90.0317, 0.5)])])
    assert ev2['L_hex'] == 0 and ev2['L_CGLY'] >= 0.99


def test_reorder_promotes_o_glycoside_on_hexose_loss():
    rows = [C_GLC, OTHER, O_GLC]
    new, info = cr.reorder(rows, [spec([(M_APG7G + PROT - cr.L_HEX, 1.0), (153.0182, 0.2)])])
    assert new == [O_GLC, OTHER, C_GLC]            # group slots 0 and 2 swap; the other formula keeps slot 1
    assert info['changed'] and 'L_HEX' in info['groups'][0]['rules']


def test_reorder_promotes_c_glycoside_on_cross_ring_losses():
    rows = [O_GLC, C_GLC]
    new, _ = cr.reorder(rows, [spec([(M_APG7G + PROT - 120.0423, 1.0), (M_APG7G + PROT - 90.0317, 0.5)])])
    assert new == [C_GLC, O_GLC]


def test_gate_keeps_order_without_strong_evidence():
    rows = [C_GLC, O_GLC]
    assert cr.reorder(rows, [spec([(200.0, 1.0)])])[0] == rows                              # no diagnostic evidence
    weak = [spec([(M_APG7G + PROT - cr.L_HEX, 0.1), (200.0, 1.0)])]                          # weak evidence: weight 0.5 < theta
    assert cr.reorder(rows, weak)[0] == rows
    assert cr.reorder(rows, weak, dict(theta=0.5))[0] == [O_GLC, C_GLC]
    assert cr.reorder(rows, [spec([(M_APG7G + PROT - cr.L_HEX, 1.0)])], dict(top=1))[0] == rows   # outside the window


def test_rule_selection_and_cfg():
    rows = [C_GLC, O_GLC]
    s = [spec([(M_APG7G + PROT - cr.L_HEX, 1.0)])]
    assert cr.reorder(rows, s, dict(rules=['L_CO2N']))[0] == rows
    assert set(cr.DEFAULT_RULES) <= set(cr.RULES)
    try:
        cr.resolve_cfg(dict(thta=1.0))
        raise AssertionError('typo accepted')
    except KeyError:
        pass


def test_low_resolution_spectra_ignored():
    lowres = dict(mz=np.array([271.0, 433.0]), it=np.array([1.0, 0.3]), adduct='[M+H]+', prec=433.0, mode=1)
    assert cr.evidence([lowres])['n'] == 0


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
