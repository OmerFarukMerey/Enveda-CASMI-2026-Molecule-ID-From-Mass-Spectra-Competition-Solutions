"""Tests of the GLACIER fine-tune helpers (train/glacierft): the pygmtools stand-in, the runner-identical CE rules; no competition data.

usage: python tests/test_glacierft.py   (or pytest)
"""
import importlib.util
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); SOL = os.path.dirname(HERE)
GF = os.path.join(SOL, 'train', 'glacierft')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


PG = _load('pygmtools_shim', os.path.join(GF, 'shim', 'pygmtools', '__init__.py'))


def _sinkhorn_ref(s, n, tau, iters=10):
    """Reference: plain numpy log-domain Sinkhorn on the [:n, :n] block, rows first."""
    x = s[:n, :n] / tau
    for i in range(iters):
        ax = 1 if i % 2 == 0 else 0
        mx = x.max(axis=ax, keepdims=True)
        x = x - (mx + np.log(np.exp(x - mx).sum(axis=ax, keepdims=True)))
    out = np.zeros_like(s)
    out[:n, :n] = np.exp(x)
    return out


def test_sinkhorn_matches_reference_and_masks_padding():
    import torch
    rng = np.random.default_rng(0)
    s = rng.normal(size=(3, 6, 6)).astype(np.float64)
    n = [6, 4, 2]
    got = PG.sinkhorn(torch.from_numpy(s), n1=torch.tensor(n), n2=torch.tensor(n), tau=0.05, backend='pytorch').numpy()
    for b in range(3):
        ref = _sinkhorn_ref(s[b], n[b], 0.05)
        assert np.allclose(got[b], ref, atol=1e-6)
        assert np.all(got[b, n[b]:, :] == 0) and np.all(got[b, :, n[b]:] == 0)
        assert np.allclose(got[b, :n[b], :n[b]].sum(0), 1.0, atol=1e-6)      # last (10th) step normalised columns


def test_sinkhorn_is_differentiable():
    import torch
    s = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True)
    p = PG.sinkhorn(s, n1=torch.tensor([4, 3]), n2=torch.tensor([4, 3]), tau=0.5)[:, 0, 0].sum()
    p.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0


def test_hungarian_is_maximum_assignment():
    import torch
    rng = np.random.default_rng(1)
    s = rng.normal(size=(2, 4, 5))
    got = PG.hungarian(torch.from_numpy(s), n2=torch.tensor([5, 3])).numpy()
    for b, nc in enumerate((5, 3)):
        best = max(sum(s[b, i, p[i]] for i in range(4) if p[i] < nc) for p in itertools.permutations(range(5), 4)
                   if sum(p[i] < nc for i in range(4)) == min(4, nc))
        assert abs((got[b] * s[b]).sum() - best) < 1e-9
        assert got[b, :, nc:].sum() == 0 and got[b].sum() == min(4, nc)


def test_ce_rules_match_the_runner():
    sys.path.insert(0, GF)
    import data as D
    import build_data as BD
    assert D.ce_bucket(42.4) == 40.0 and D.ce_bucket(2.0) == 5.0 and D.ce_bucket(47.6) == 50.0 and np.isnan(D.ce_bucket(np.nan))
    assert BD.nce('50% (nominal)') == 50.0 and abs(BD.nce('30,45,60') - 45.0) < 1e-12 and np.isnan(BD.nce(''))


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
