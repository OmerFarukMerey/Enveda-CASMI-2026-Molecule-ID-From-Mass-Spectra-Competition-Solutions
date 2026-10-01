"""tools/build_ext_val.py: the matched cosine must be a one-to-one 0.01 Da match, the peak-list hash order-free and rounding-exact.

usage: python tests/test_build_ext_val.py   (or pytest)
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import numpy as np
from build_ext_val import matched_cosine, peak_hashes, unit_sqrt


def cos(qmz, qit, libs):
    lmz = np.concatenate([np.asarray(m, float) for m, _ in libs])
    lw = np.concatenate([unit_sqrt(np.asarray(i, float)) for _, i in libs])
    sid = np.repeat(np.arange(len(libs)), [len(m) for m, _ in libs])
    return matched_cosine(np.asarray(qmz, float), unit_sqrt(np.asarray(qit, float)), lmz, lw, sid, len(libs))


def test_cosine_tolerance_and_reference():
    q = ([100.0, 150.0, 200.0], [1.0, 4.0, 9.0])
    c = cos(*q, [q, ([100.009, 150.0, 200.0], [1.0, 4.0, 9.0]), ([100.02, 150.0, 200.0], [1.0, 4.0, 9.0]), ([300.0], [1.0])])
    assert np.allclose(c[:2], 1.0) and c[3] == 0.0
    assert np.isclose(c[2], (2 * 2 + 3 * 3) / 14)          # sqrt intensities 1,2,3; 100.02 is outside 0.01 Da


def test_cosine_is_one_to_one():
    # two library peaks inside the tolerance of one query peak may not both count: the cosine must stay a valid matching
    c = cos([100.0], [1.0], [([99.995, 100.005], [1.0, 1.0])])
    assert np.isclose(c[0], np.sqrt(0.5))
    c = cos([99.995, 100.005], [1.0, 1.0], [([100.0], [1.0])])
    assert np.isclose(c[0], np.sqrt(0.5))


def test_peak_hash_rounding_and_order():
    h = peak_hashes(np.array([0, 0, 1, 1, 2, 2, 3]), np.array([50.004, 80.0, 80.001, 49.996, 50.0, 80.02, 50.0]), 5)
    assert h[0] == h[1] != h[2]                 # same 0.01 Da set in another order -> equal; 80.02 -> different
    assert h[3] != h[0] and h[4] == 0           # subset differs; spectrum without peaks hashes to 0


if __name__ == '__main__':
    for f in (test_cosine_tolerance_and_reference, test_cosine_is_one_to_one, test_peak_hash_rounding_and_order):
        f()
    print('build_ext_val: cosine and peak-hash tests pass')
