"""List assembly after the PubChem join: rules that keep the pool's best candidate near the top of a joined base list (flag PC_JOIN_PROTECT).

The join (pc_join.py) lets the ranker place PubChem-only rows among the pool candidates. Leaderboard: 100 joined rows
0.431, 200 rows 0.406, the gate at lib_max < 0.8 instead of 0.7 0.416. The local regime does not reproduce those drops
(reviews/lbcal-2026-10-04.md), but it shows their only visible mechanism: a pool answer that was first loses the first
place to a joined row, and locally engine-2 repairs it. The rules here bound that exposure on the base list itself,
before any fusion, without a label and without a second engine run:

    'cap'    at most K joined rows above the first row that is not joined (K = 0: a joined row is never first)
    'pin'    the top candidate of the UNJOINED list (ref_key) keeps a rank no worse than K + 1 (needs that list)
    'alt'    the first K pool rows and the first K joined rows alternate in the first 2K positions, pool row first
    'conf'   'cap' with K when the pool rows are confident (top-score margin in standard deviations of the first 25
             scores, above TAU); free otherwise. The margin is that of the UNJOINED list when nb/06 supplies it
             (PC_JOIN_PROTECT_REF='unjoined': one more engine pass per joined molecule), else that of the non-joined
             rows of the joined list ('joined': no extra work, weaker)

A joined row is a row with pool id -2 (nb/06_base_lists.py writes that). Rows keep their relative order apart from the
moved ones; the score column is re-assigned in descending order so that it stays monotone for the stages that z-score it.

Local result (reviews/lbcal-2026-10-04.md, N 100 / 200, gate 0.7): 'conf' K 0 TAU 1.25 with the unjoined margin keeps 92 to
95% of the gain on PubChem-only answers and removes about two thirds of the pool answers that lose the first place on
the v1 side; with the joined-list margin (TAU 1.5) 95% and a quarter to 40%; 'cap' K 0 keeps half and removes three
quarters; 'cap' K >= 1 and 'pin' keep 82 to 94% and remove none of the lost first places; 'alt' keeps half and adds losses.

Config (defaults = off):  PC_JOIN_PROTECT=None | 'cap' | 'pin' | 'alt' | 'conf',  PC_JOIN_PROTECT_K=0,  PC_JOIN_PROTECT_TAU=1.25,
                          PC_JOIN_PROTECT_REF='unjoined' | 'joined'
Notebook edit (nb/08_submit.py, before fusion_core.build_submission):
    BASE = join_protect.apply_all(BASE, CFG, ref=JOIN_REF, log=log)
Pure functions, no competition data; tests in test_join_protect.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULTS = dict(PC_JOIN_PROTECT=None, PC_JOIN_PROTECT_K=0, PC_JOIN_PROTECT_TAU=1.25, PC_JOIN_PROTECT_REF='unjoined')
JOINED = -2


def pool_margin(scores: Sequence[float], joined: Sequence[bool], n: int = 25) -> float:
    """(best - second) score of the rows that are not joined, in standard deviations of their first n scores.
    nan when fewer than three such rows exist (no confidence statement possible)."""
    s = np.asarray([x for x, j in zip(scores, joined) if not j][:n], float)
    if len(s) < 3:
        return float('nan')
    return float((s[0] - s[1]) / (s.std() + 1e-9))


def _move_up(order: List[int], src: int, dst: int) -> List[int]:
    """Row at position src goes to position dst (dst <= src); the rows between shift down by one."""
    if dst >= src:
        return order
    o = list(order); o.insert(dst, o.pop(src))
    return o


def unjoined_ref(keys: Sequence[str], scores: Sequence[float], n: int = 25) -> dict:
    """From the candidate rows of the engine run WITHOUT injection (any order, duplicate keys allowed): the top key and
    the top-score margin of the key-deduplicated list. nb/06 stores this per joined molecule (JOIN_REF[mid])."""
    seen, ks, ss = set(), [], []
    for i in np.argsort(-np.asarray(scores, float), kind='stable'):
        if keys[i] in seen:
            continue
        seen.add(keys[i]); ks.append(keys[i]); ss.append(float(scores[i]))
        if len(ks) >= n:
            break
    return dict(key=ks[0] if ks else None, margin=pool_margin(ss, [False] * len(ss), n))


def protect_order(joined: Sequence[bool], rule: str, k: int = 0, scores: Optional[Sequence[float]] = None, tau: float = 1.25,
                  ref_pos: Optional[int] = None, margin: Optional[float] = None) -> List[int]:
    """New order (indices into the current list) under a rule. joined[i]: row i is a joined PubChem row.
    ref_pos: position of the unjoined list's top candidate in this list ('pin'); None = the first non-joined row.
    margin: the unjoined list's top-score margin ('conf'); None = the margin of the non-joined rows of this list."""
    n = len(joined); order = list(range(n))
    pool = [i for i in range(n) if not joined[i]]; inj = [i for i in range(n) if joined[i]]
    if not rule or not pool or not inj:
        return order
    if rule == 'conf':
        m = margin if margin is not None else (pool_margin(scores, joined) if scores is not None else float('nan'))
        if m is None or not (np.isfinite(m) and m > tau):
            return order
        rule = 'cap'
    if rule == 'cap':
        return _move_up(order, pool[0], min(pool[0], max(0, int(k))))
    if rule == 'pin':
        src = ref_pos if ref_pos is not None and 0 <= ref_pos < n else pool[0]
        return _move_up(order, src, min(src, max(0, int(k))))
    if rule == 'alt':
        head = []
        for a, b in zip(pool[:k], inj[:k]):
            head += [a, b]
        head += pool[len(head) // 2:k] if len(inj) < k else []
        hs = set(head)
        return head + [i for i in order if i not in hs]
    raise ValueError(f'unknown PC_JOIN_PROTECT rule {rule!r}')


def protect(entry: Sequence, rule: Optional[str], k: int = 0, tau: float = 1.25, ref_key: Optional[str] = None,
            margin: Optional[float] = None) -> list:
    """BASE entry [smiles, keys, lib_max, scores, formulas, pool ids] -> the same entry re-ordered under the rule.
    Entries without a joined row (pool id -2), without pool ids, or with rule None come back unchanged."""
    if not rule or len(entry) < 6 or not entry[0]:
        return list(entry)
    smis, keys, lib_max, scs, forms, pids = entry[:6]
    joined = [int(p) == JOINED for p in pids]
    ref_pos = None
    if ref_key is not None and ref_key in keys:
        ref_pos = list(keys).index(ref_key)
    o = protect_order(joined, rule, k, scs, tau, ref_pos, margin)
    if o == list(range(len(o))):
        return list(entry)
    s_sorted = sorted((float(x) for x in scs), reverse=True)
    return [[smis[i] for i in o], [keys[i] for i in o], lib_max, s_sorted, [forms[i] for i in o], [pids[i] for i in o]] + list(entry[6:])


def apply_all(base: Dict, cfg: Dict, ref: Optional[Dict] = None, log=None) -> Dict:
    """Every BASE entry under CFG['PC_JOIN_PROTECT'] (None = unchanged dict). Never raises on a single entry: a molecule
    whose entry cannot be re-ordered keeps its list. ref: {mid: unjoined_ref(...)} = top key ('pin') and margin ('conf')
    of the unjoined list; used when PC_JOIN_PROTECT_REF is 'unjoined', a molecule without it falls back to the joined list."""
    rule = cfg.get('PC_JOIN_PROTECT')
    if not rule:
        return base
    k = int(cfg.get('PC_JOIN_PROTECT_K', DEFAULTS['PC_JOIN_PROTECT_K'])); tau = float(cfg.get('PC_JOIN_PROTECT_TAU', DEFAULTS['PC_JOIN_PROTECT_TAU']))
    use_ref = cfg.get('PC_JOIN_PROTECT_REF', DEFAULTS['PC_JOIN_PROTECT_REF']) == 'unjoined' and bool(ref)
    out, n_changed, n_top, n_err = {}, 0, 0, 0
    for m, e in base.items():
        try:
            r = ((ref.get(m) or ref.get(str(m))) if use_ref else None) or {}
            e2 = protect(e, rule, k, tau, r.get('key'), r.get('margin'))
        except Exception as ex:
            if log:
                log('PC_JOIN_PROTECT failed -> list unchanged', m, repr(ex))
            e2 = e; n_err += 1
        n_changed += int(list(e2[1]) != list(e[1])); n_top += int(bool(e[1]) and e2[1][:1] != list(e[1])[:1])
        out[m] = e2
    if log:
        log(f'PC_JOIN_PROTECT {rule} K {k} tau {tau}: {n_changed} of {len(base)} lists re-ordered, top-1 changed in {n_top}, failures {n_err}')
    return out


def exposure(rr0: Dict[str, float], rr: Dict[str, float]) -> dict:
    """Of the lists whose answer was first before (rr0 == 1): how many lose the first place, and how far they fall."""
    was = [m for m, v in rr0.items() if v == 1.0 and m in rr]
    lost = [m for m in was if rr[m] < 1.0]
    after = np.array([rr[m] for m in lost], float)
    return dict(first_before=len(was), lost=len(lost), lost_share=len(lost) / max(1, len(was)),
                mean_rr_after=float(after.mean()) if len(after) else float('nan'),
                out_of_25=int((after == 0).sum()), median_rank_after=float(np.median(1.0 / after[after > 0])) if (after > 0).any() else float('nan'),
                rr_lost=float((1.0 - after).sum()))
