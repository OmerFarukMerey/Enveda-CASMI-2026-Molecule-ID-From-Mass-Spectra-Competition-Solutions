"""Prototype stage-B merge of the PubChem-only list: evidence tiers (forward-model agreement, FPNet gate) pick the slots.

Drop-in for the stage-B branch of nb/embed/fusion_core.py (same inputs: base list after stage A, its keys, lib_max and
the PubChem dict after stage A), as a pure function. Not wired into the notebook; see reviews/pubchem-2026-10-02.md.

Why it looks like this (all numbers in the review):
  * Popularity never orders the list. z(f.z) + 0.25 * pop only gains when the answer is documented (pop above 1.6 to
    2.4); a PubChem structure that is NOT in the pool has a median pop of 1.1 (>= 5 in 0.36% of cases), and at 1.1
    the prior takes away 17 to 37% of what the channel returns, 49 to 76% at 0. The only popularity use left is
    optional and promote-only: a proposal with pop >= POP_HI that is already among the first POP_K moves to the front
    of the PubChem list and gets the early slots (no measured cost; pays only if a documented answer is missing from
    the pool).
  * FPNet quantities (rel = fz_top - best_pool_fz, score margins) do not tell "the best PubChem proposal is the
    answer" from "the answer is in the pool" (AUC 0.63 to 0.72), so nothing is promoted to rank 1 on them. rel keeps
    its old job: early slots when rel > REL_TH.
  * ICEBERG / GLACIER compared ACROSS the two lists do better (AUC 0.84, optimistic: both models have very likely seen
    the validation molecules): when the best PubChem proposal beats every same-formula candidate of the base top
    FWD_N in every forward model that scored both, the answer is rarely in the pool. That evidence picks the tier.
  * Without any evidence the proposals sit one place earlier than in the reference (3, 6, 9, 12, 15).

Tiers -> slots (1-based positions of the PubChem proposals in the final 25):
  0  no evidence                                                        SLOTS_GENTLE
  1  rel > REL_TH, forward margin > FWD_T1, or a documented proposal    SLOTS_AGG
  2  forward margin > FWD_T2 (off by default)                           SLOTS_TOP (the proposal leads the list)
"""
import math

DEFAULTS = dict(
    LIB_TAU=0.9,                        # untouched at or above this library similarity (as in fusion_core)
    REL_TH=600.0,                       # FPNet gate of the reference notebook
    SLOTS_GENTLE=[3, 6, 9, 12, 15],     # tier 0 (reference: 4, 8, 12, 16, 20)
    SLOTS_AGG=[2, 4, 6, 8, 10],         # tier 1 (reference "aggressive")
    SLOTS_TOP=[1, 3, 5, 7, 9],          # tier 2
    FWD_N=25,                           # base candidates the proposal is compared with
    FWD_T1=0.0,                         # forward margin for tier 1: the proposal beats every compared base candidate (None = off)
    FWD_T2=None,                        # forward margin for tier 2 (None = never promote to rank 1; 0.05 was the best tested value)
    FWD_MODELS=1,                       # minimum number of forward models that must have scored both sides
    FWD_SAME_FORMULA=True,              # compare inside the proposal's formula group when `formula` is given
    FWD_LIB_MAX=None,                   # forward evidence only below this lib_max (None = LIB_TAU)
    POP_HI=8.0,                         # popularity that moves a proposal of the first POP_K to the front; acts only when `pop` is passed
    POP_K=5,
    N=25,
)


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def insert_slots(base, base_keys, pc, pc_keys, slots, n=25):
    """PubChem proposals (not already in base) go into `slots` (1-based); base fills the rest. = fusion_core.merge."""
    bk = set(base_keys)
    pcs = [s for s, k in zip(pc, pc_keys) if k not in bk]
    out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs):
            out.append(pcs[pj]); pj += 1
        elif bi < len(base):
            out.append(base[bi]); bi += 1
        elif pj < len(pcs):
            out.append(pcs[pj]); pj += 1
    return out


def forward_margin(prop, base, scores, n=25, formula=None):
    """min over forward models of score(proposal) - max score(compared base candidates), and the number of models used.
    scores: list of {smiles: float | None} (ICEBERG, GLACIER, ...). With `formula` (callable or {smiles: formula}) only
    base candidates with the proposal's formula are compared. A model counts only if it scored the proposal and at
    least one compared base candidate. Returns (margin or None, models used)."""
    cand = list(base[:n])
    if formula is not None:
        f = formula.get if isinstance(formula, dict) else formula
        fp = f(prop)
        cand = [b for b in cand if f(b) == fp]
    margins = []
    for d in scores:
        if not d:
            continue
        v = d.get(prop)
        if not _finite(v):
            continue
        bs = [x for x in (d.get(b) for b in cand) if _finite(x)]
        if bs:
            margins.append(v - max(bs))
    return (min(margins) if margins else None), len(margins)


def merge_pubchem(base, base_keys, lib_max, p, cfg=None, pop=None, ice=None, gl=None, formula=None):
    """Final list (<= N SMILES) and an info dict.

    base, base_keys   base list after stage A and its score keys
    lib_max           best library similarity of the molecule
    p                 PubChem dict after stage A: pc, pc_keys, pc_fz, best_pool_fz, optional fz_top
    pop               optional popularity of the proposals: {smiles: value}, or a list aligned with p['pc'] AS PASSED
                      (stage A re-orders p['pc'], so a list made before stage A is misaligned; prefer the dict)
    ice, gl           optional {smiles: score | None} of the forward models (base and PubChem candidates)
    formula           optional callable or {smiles: formula}; p['pc_form'] is NOT used (stage A re-orders p['pc'] but
                      leaves pc_form in the original order)
    """
    c = dict(DEFAULTS); c.update(cfg or {})
    info = dict(tier=None, rel=None, fwd=None, fwd_models=0, pop_front=False)
    if not p or not p.get('pc') or lib_max >= c['LIB_TAU']:
        return list(base[:c['N']]), info
    pc, pc_keys = list(p['pc']), list(p['pc_keys'])
    bk = set(base_keys)
    order = [i for i, k in enumerate(pc_keys) if k not in bk]
    if not order:
        return list(base[:c['N']]), info

    # optional popularity: promote-only, inside the first POP_K proposals, never a full re-sort
    if pop is not None and c['POP_HI'] is not None:
        pv = (lambda i: pop.get(pc[i])) if isinstance(pop, dict) else (lambda i: pop[i] if i < len(pop) else None)
        hi = [i for i in order[:c['POP_K']] if _finite(pv(i)) and pv(i) >= c['POP_HI']]
        if hi:
            order = hi + [i for i in order if i not in hi]; info['pop_front'] = True

    bp = p.get('best_pool_fz')
    top = p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]
    rel = top - bp if _finite(bp) else 1e9
    info['rel'] = rel
    tier = 1 if rel > c['REL_TH'] else 0
    if info['pop_front']:
        tier = max(tier, 1)

    fwd_lib = c['LIB_TAU'] if c['FWD_LIB_MAX'] is None else c['FWD_LIB_MAX']
    if lib_max < fwd_lib and (c['FWD_T1'] is not None or c['FWD_T2'] is not None):
        fwd, nm = forward_margin(pc[order[0]], base, [ice, gl], n=c['FWD_N'], formula=formula if c['FWD_SAME_FORMULA'] else None)
        info['fwd'], info['fwd_models'] = fwd, nm
        if fwd is not None and nm >= c['FWD_MODELS']:
            if c['FWD_T2'] is not None and fwd > c['FWD_T2']:
                tier = 2
            elif c['FWD_T1'] is not None and fwd > c['FWD_T1']:
                tier = max(tier, 1)
    info['tier'] = tier
    slots = (c['SLOTS_GENTLE'], c['SLOTS_AGG'], c['SLOTS_TOP'])[tier]
    final = insert_slots(base, base_keys, [pc[i] for i in order], [pc_keys[i] for i in order], slots, c['N'])
    return final, info
