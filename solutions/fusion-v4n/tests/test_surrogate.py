"""Tests for nb/embed/surrogate.py (adduct surrogate rows for the forward models); no competition data needed.

usage: python tests/test_surrogate.py   (or pytest)
"""
import os, sys

import numpy as np
import pandas as pd

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed'))
import surrogate


def frame():
    rows = [
        ('a', '[M+H]+', 'positive', 301.0, [100.0, 200.0, 301.0]),
        ('a', '[M+NH4]+', 'positive', 318.026549, [100.0, 301.0, 301.4, 302.0, 318.03]),
        ('b', '[M-H2O+H]+', 'positive', 283.0, [100.0, 283.0]),
        ('b', '[M+K]+', 'positive', 339.0, [100.0, 339.0]),
        ('c', '[M-H]-', 'negative', 299.0, [100.0, 299.0]),
        ('d', '[M+NH4]+', 'positive', 60.0, [59.0, 60.0]),          # every peak above the [M+H]+ m/z -> row dropped
        ('e', '[M-2H2O+H]+', 'positive', 265.0, [50.0]),
    ]
    return pd.DataFrame(dict(molecule_id=[r[0] for r in rows], adduct=[r[1] for r in rows],
                             ionization_mode=[r[2] for r in rows], precursor_mz=[r[3] for r in rows],
                             ms2_mzs=[np.array(r[4]) for r in rows],
                             ms2_normalized_intensities=[np.ones(len(r[4])) for r in rows]))


def test_rewrite_selects_and_relabels():
    out = surrogate.rewrite(frame())
    assert list(out.molecule_id) == ['a', 'b', 'e']
    assert set(out.adduct) == {'[M+H]+'}


def test_nh4_shifts_precursor_and_cuts_peaks():
    r = surrogate.rewrite(frame()).iloc[0]
    assert abs(r.precursor_mz - 301.0) < 1e-6
    assert list(r.ms2_mzs) == [100.0, 301.0, 301.4]                 # <= 301.0 + 0.5 kept, NH4 precursor removed
    assert len(r.ms2_normalized_intensities) == 3


def test_water_loss_only_relabelled():
    r = surrogate.rewrite(frame()).iloc[1]
    assert r.precursor_mz == 283.0 and list(r.ms2_mzs) == [100.0, 283.0]


def test_input_frame_untouched_and_empty_ok():
    df = frame(); before = df.adduct.tolist()
    surrogate.rewrite(df)
    assert df.adduct.tolist() == before and df.precursor_mz[1] == 318.026549
    assert len(surrogate.rewrite(df[df.adduct == '[M+H]+'])) == 0


def test_extend_items_gap_and_all():
    items = [dict(mid='a', spectra=[{'adduct': '[M+H]+'}], cands=['C']), dict(mid='b', spectra=[], cands=['C']),
             dict(mid='c', spectra=[], cands=['C'])]
    extra = [dict(mid='a', spectra=[{'s': 1}], cands=['C']), dict(mid='b', spectra=[{'s': 2}], cands=['C']),
             dict(mid='c', spectra=[], cands=['C'])]
    gap, n = surrogate.extend_items(items, extra, 'gap')
    assert n == 1 and len(gap[0]['spectra']) == 1 and gap[1]['spectra'] == [{'s': 2}] and gap[2]['spectra'] == []
    al, n = surrogate.extend_items(items, extra, 'all')
    assert n == 2 and len(al[0]['spectra']) == 2
    assert items[0]['spectra'] == [{'adduct': '[M+H]+'}] and items[1]['spectra'] == []      # inputs not mutated


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f()
    print('surrogate OK')
