"""Deterministic merged-adduct choice for the public v1 engine, and the hash-seed facts of a run (ADDUCT_TIE; off by default).

Why. casmi/engine.py (casmi26-v4b-models, verbatim on Kaggle) labels the merged spectrum of one polarity with

    ad = max(set(adducts), key=adducts.count)

When two adducts share the highest count the winner is the first of them in SET iteration order, and the order of a set of
strings depends on the interpreter's hash seed (random per process unless PYTHONHASHSEED is set). The engine bank's merged-item
logits, every f.z feature and the ranker score of such a molecule therefore differ between two sessions of the same notebook
(reviews/determ-2026-10-07.md). The notebook kernel's seed is fixed when the kernel starts and cannot be changed from a cell, so
the tie is broken here by an explicit key instead. Nothing of the public package is edited: `install` puts a module-level `max`
into casmi.engine that answers exactly this one call and hands every other call to the builtin.

Rules (a molecule without a tie is untouched by all of them: the builtin is called with the same arguments):
  'train'  the upstream authors' own rule in fe_v4/v1engine.py::merged_adduct (most frequent adduct in the training spectra, then
           alphabetical): the engine bank then reads the same adduct as the three model_views models of the same pass
  'first'  the tied adduct that occurs first in the molecule's spectra (the rule of nb/embed/pc/probe_core2.py)
  'mean'   the merged item's logits are averaged over the tied adducts (the expectation over hash seeds at the logit level);
           needs `wrap_bank` on every bank the engine may hold
  callable(adducts, tied) -> adduct   (studies)

    determ.install(casmi.engine, E, 'train', merged_adduct=v1engine.merged_adduct)
    determ.wrap_bank(bank, chem.adduct_index)            # 'mean' only

Pure python + numpy; covered by tests/test_determ.py.
"""
from __future__ import annotations

import builtins
import inspect
import os
import sys
from typing import Callable, List, Optional, Sequence

import numpy as np

RULES = ('train', 'first', 'mean')
EXPR = 'max(set(adducts), key=adducts.count)'
_STATE = dict(rule=None, merged=None, pending=[], n_tie=0, n_mean=0)


def tied(adducts: Sequence[str]) -> List[str]:
    """The most frequent adducts, in order of first occurrence (one entry = no tie)."""
    cnt: dict = {}
    for a in adducts:
        cnt[a] = cnt.get(a, 0) + 1
    if not cnt:
        return []
    b = builtins.max(cnt.values())
    return [a for a in cnt if cnt[a] == b]


def pick(adducts: Sequence[str], rule, merged: Optional[Callable] = None) -> str:
    """The merged item's adduct under `rule`; without a tie the single most frequent adduct."""
    t = tied(adducts)
    if len(t) == 1:
        return t[0]
    if callable(rule):
        r = rule(list(adducts), list(t)); assert r in t, (r, t)
        return r
    if rule == 'train':
        assert merged is not None, "ADDUCT_TIE 'train' needs v1engine.merged_adduct"
        r = merged(list(adducts)); assert r in t, (r, t)
        return r
    if rule in ('first', 'mean'):
        return t[0]
    raise ValueError(f'unknown ADDUCT_TIE rule {rule!r}')


def groups(spectra: Sequence[dict], max_query: int = 16) -> List[List[str]]:
    """Tied adducts of each polarity group in the engine's order (positive, negative; first max_query spectra)."""
    sp = list(spectra)[:max_query]; out = []
    for md in (1, -1):
        g = [s['adduct'] for s in sp if s['mode'] == md]
        if g:
            out.append(tied(g))
    return out


def _max(*a, **k):
    key = k.get('key')
    if len(a) == 1 and len(k) == 1 and getattr(key, '__name__', '') == 'count' and isinstance(getattr(key, '__self__', None), list) \
            and isinstance(a[0], (set, frozenset)) and _STATE['rule'] is not None:
        adducts = key.__self__
        if a[0] == set(adducts) and len(tied(adducts)) > 1:
            _STATE['n_tie'] += 1
            return pick(adducts, _STATE['rule'], _STATE['merged'])
    return builtins.max(*a, **k)


def install(engine_mod, E, rule, merged: Optional[Callable] = None):
    """Break the merged-adduct tie of `engine_mod` (casmi.engine) by `rule`; None restores the public behaviour (set order).
    E.run is wrapped once to note the molecule's tied groups for 'mean'. Raises if the engine does not hold the expression."""
    if rule is None:
        _STATE.update(rule=None, merged=None, pending=[])
        engine_mod.__dict__.pop('max', None)
        return
    assert callable(rule) or rule in RULES, f'ADDUCT_TIE must be one of {RULES} or None, got {rule!r}'
    src = inspect.getsource(engine_mod)
    assert src.count(EXPR) == 1, f'casmi.engine does not hold `{EXPR}` exactly once: the tie rule would do nothing'
    _STATE.update(rule=rule, merged=merged, pending=[])
    engine_mod.max = _max
    if E is not None and not getattr(E, '_determ_run', False):
        run0 = E.run

        def run(spectra, *a, **k):
            _STATE['pending'] = groups(spectra, getattr(E.cfg, 'max_query_spectra', 16)) if _STATE['rule'] == 'mean' else []
            return run0(spectra, *a, **k)
        E.run = run; E._determ_run = True


def wrap_bank(bank, adduct_index: Callable):
    """'mean': the bank's logits of a merged item with tied adducts become the mean over the tied adducts' logits (the item is
    evaluated once per distinct adduct index). Every other call, and every call under another rule, passes through untouched."""
    if getattr(bank, '_determ_wrapped', False):
        return bank
    orig = bank.logits_el

    def logits_el(items, *a, **k):
        pend = _STATE['pending']
        if _STATE['rule'] != 'mean' or len(items) != len(pend) or not any(len(t) > 1 for t in pend):
            return orig(items, *a, **k)            # the per-spectrum call (a tie needs two spectra, so it is longer than `pend`)
        _STATE['pending'] = []
        ext, owner = [], []
        for j, (it, t) in enumerate(zip(items, pend)):
            ixs = list(dict.fromkeys(int(adduct_index(x)) for x in t)) if len(t) > 1 else [int(it['adduct_ix'])]
            for ix in ixs:
                ext.append(dict(it, adduct_ix=ix)); owner.append(j)
        z, el = orig(ext, *a, **k)
        owner = np.asarray(owner)
        Z = np.stack([np.asarray(z)[owner == j].mean(0) for j in range(len(items))]).astype(np.float32)
        EL = np.stack([np.asarray(el)[owner == j].mean(0) for j in range(len(items))]).astype(np.float32)
        _STATE['n_mean'] += int(len(ext) > len(items))
        return Z, EL
    bank.logits_el = logits_el; bank._determ_wrapped = True
    return bank


def counts() -> dict:
    """Ties broken by a rule since the start (one per polarity group) and merged calls averaged ('mean')."""
    return dict(ties=_STATE['n_tie'], mean_calls=_STATE['n_mean'])


def hash_facts() -> dict:
    """What decides set order in THIS interpreter: PYTHONHASHSEED as the process sees it and whether hashing is randomised."""
    return dict(PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED'), hash_randomization=int(sys.flags.hash_randomization),
                probe=builtins.max({'[M+H]+', '[M+Na]+'}, key=['[M+H]+', '[M+Na]+'].count))
