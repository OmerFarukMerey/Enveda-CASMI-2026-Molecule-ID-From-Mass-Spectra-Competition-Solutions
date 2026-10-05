"""Prototype same-formula isomer re-ranker `rerank_conf`: gl_fuse.rerank_multi plus confidence knobs and per-spectrum scores.

Drop-in for `gl_fuse.rerank_multi` (casmi26-glacier) in nb/embed/fusion_core.py: same arguments, returns a permutation,
same-formula groups keep their slots. With the default knobs it is EXACTLY rerank_multi (test_isomer_rank.py; re-fusing
the five validation runs through it changes no list). Not wired into the notebook. Evidence: reviews/isomer-2026-10-02.md.

    order = rerank_conf(smis, keys, scores, formulas, [ice, gl], [1.0, 1.0], top_n=60, **RECOMMENDED)

The study found NO rule that orders isomers clearly better than the z-sum: the knobs below are what survived, and their
whole-pipeline effect is of the size of existing knobs (ICE_LAM = GL_LAM = 0.75, the soft library gate). All off by default.

1. Confidence. The re-scorers "agree" on a group when every contributing one puts the same member first (ICEBERG and
   GLACIER agree on 54 to 79% of covered groups and are then right in 81 to 92%, against 30 to 43% when they disagree).
     w_disagree   multiplies the re-scorer weights on groups without agreement (and, if margin_tau is set, with a
                  joint z-margin below it). Helps where the prior is strong (class-1-like molecules below the library
                  gate), costs a little where it is weak. 0 keeps the prior order: do not use.
     agree_bonus  z units added to the member all re-scorers put first. Helps where the prior is weak, hurts where it
                  is strong (library hits: keep the ICE_LAM_LIB / GL_LAM_LIB gate, which sets the weights to 0 and
                  thereby switches both knobs off).
   RECOMMENDED = both: non-negative on all five validation runs in both regimes with the v4b library gate on.
2. sd_floor: z of a re-scorer divides by sqrt(sd^2 + sd_floor^2), so a group in which a model sees no difference
   contributes little. prior_w: weight of z(ranker score). Neither is recommended (no honest gain).
3. Per-spectrum scores. A re-scorer may give a list with one similarity per covered query spectrum instead of their
   mean (the runners would have to dump it). `spec_weights[k]` holds one weight per spectrum of re-scorer k
   (`adduct_weights`), `spec_agg` is 'mean' (weighted mean = the runner's score when all weights are 1) or 'zmean'
   (mean of per-spectrum z). Measured with local predictions: no aggregation beats the plain mean, and dropping
   [M+Na]+ spectra does not help ICEBERG; the path exists so that a later study does not need a new function.

Pure Python, no numpy / rdkit, like fuse.py and gl_fuse.py.
"""
import math

__all__ = ['rerank_conf', 'adduct_weights', 'RECOMMENDED']

RECOMMENDED = dict(w_disagree=0.5, agree_bonus=1.0)


def adduct_weights(adducts, primary='[M+H]+', other=0.0):
    """Per-spectrum weights from the adducts of the covered spectra: `primary` spectra 1, the rest `other`; if the
    molecule has no `primary` spectrum, every spectrum gets `other` (0 = the re-scorer is dropped for it)."""
    return [1.0 if a == primary else float(other) for a in adducts]


def _clean(v):
    try:
        return float(v) if v is not None and math.isfinite(float(v)) else None
    except (TypeError, ValueError):
        return None


def _z(vals, sd_floor=0.0):
    """fuse._z: (x - mean) / std(ddof=1) over the non-missing members; missing / n < 2 / std 0 -> 0.
    sd_floor > 0 divides by sqrt(sd^2 + sd_floor^2) instead."""
    xs = [v for v in vals if v is not None]
    n = len(xs)
    if n < 2:
        return [0.0] * len(vals)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    if not sd > 0:
        return [0.0] * len(vals)
    if sd_floor > 0:
        sd = math.sqrt(sd * sd + sd_floor * sd_floor)
    return [((v - mu) / sd) if v is not None else 0.0 for v in vals]


def _getter(smis, keys, sc):
    if isinstance(sc, dict):
        def of(i):
            v = sc.get(smis[i])
            if v is None and keys is not None:
                v = sc.get(keys[i])
            return v
    else:
        def of(i):
            return sc[i] if sc is not None and i < len(sc) else None
    return of


def _aggregate(raw, weights, how):
    """Group column from raw per-member values (number | None | list of per-spectrum numbers) -> list of float | None.
    Scalars pass through, so today's caches (one mean per candidate) behave exactly as in rerank_multi."""
    if not any(isinstance(v, (list, tuple)) for v in raw):
        return [_clean(v) for v in raw]
    ns = max(len(v) for v in raw if isinstance(v, (list, tuple)))
    w = [1.0] * ns if weights is None else [float(x) for x in list(weights)[:ns]] + [1.0] * max(0, ns - len(weights))
    M = [[_clean(v[j]) if isinstance(v, (list, tuple)) and j < len(v) else None for j in range(ns)] for v in raw]
    if how == 'zmean':
        Z = [[None] * ns for _ in raw]
        for j in range(ns):
            col = [row[j] for row in M]
            if sum(x is not None for x in col) >= 2:
                zc = _z(col)
                for i, x in enumerate(col):
                    Z[i][j] = zc[i] if x is not None else None
        M = Z
    elif how != 'mean':
        raise ValueError(f'spec_agg {how!r}')
    out = []
    for row in M:
        num = sum(w[j] * x for j, x in enumerate(row) if x is not None and w[j] > 0)
        den = sum(w[j] for j, x in enumerate(row) if x is not None and w[j] > 0)
        out.append(num / den if den > 0 else None)
    return out


def rerank_conf(smis, keys, scores, formulas, score_dicts, lams, top_n=60, min_covered=2, tie_eps=1e-9,
                w_disagree=1.0, margin_tau=None, agree_bonus=0.0, sd_floor=0.0, prior_w=1.0,
                spec_agg='mean', spec_weights=None):
    """New order (list of indices into smis) after re-scoring same-formula groups inside the top_n.

    Contract of gl_fuse.rerank_multi: smis / keys / scores / formulas are aligned lists in the current ranked order,
    score_dicts a list of {smiles: value} (a key from `keys` is also accepted) or lists aligned with smis, lams one
    weight per re-scorer. A group (same formula, >= 2 members among the first top_n, at least one re-scorer covering
    >= min_covered members) is re-sorted by
        prior_w * z(score) + m * sum_k lam_k * z_k + agree_bonus * [member is every re-scorer's first choice]
    and its members are put back into the same positions; m = w_disagree on groups where the contributing re-scorers
    do not agree on the first member (and the margin of their joint pick is below margin_tau, if given), else 1.
    A group with a single contributing re-scorer has no agreement to check: m = 1 unless margin_tau is set.
    Re-scorers with weight 0 do not contribute to the confidence. Ties keep the current order."""
    n = len(smis)
    order = list(range(n))
    if len(score_dicts) != len(lams):
        raise ValueError('one lam per score dict')
    sw = list(spec_weights) if spec_weights is not None else [None] * len(score_dicts)
    if len(sw) != len(score_dicts):
        raise ValueError('one spec_weights entry (or None) per score dict')
    use = [(_getter(smis, keys, sc), float(l), w) for sc, l, w in zip(score_dicts, lams, sw) if sc]
    if n == 0 or not use:
        return order
    m = min(int(top_n), n)
    groups = {}
    for i in range(m):
        groups.setdefault(formulas[i], []).append(i)
    for f, idx in groups.items():
        if len(idx) < 2:
            continue
        cols = []
        for of, lam, w in use:
            v = _aggregate([of(i) for i in idx], w, spec_agg)
            cols.append((v, lam, sum(x is not None for x in v)))
        if not any(c >= max(2, min_covered) for _, _, c in cols):
            continue
        zs = [(_z(v, sd_floor), lam) for v, lam, c in cols if c >= 2]
        act = [(zk, lam) for zk, lam in zs if lam != 0.0]            # re-scorers that carry weight: the confidence voters
        mult, top = 1.0, None
        if act and (w_disagree != 1.0 or agree_bonus != 0.0):
            firsts = {max(range(len(idx)), key=lambda k: (zk[k], -k)) for zk, _ in act}
            joint = [sum(zk[k] for zk, _ in act) for k in range(len(idx))]
            srt = sorted(joint, reverse=True)
            margin = srt[0] - srt[1]
            agree = len(act) >= 2 and len(firsts) == 1
            if agree:
                top = next(iter(firsts))
            if len(act) >= 2:
                confident = agree or (margin_tau is not None and margin >= margin_tau)
            else:
                confident = margin_tau is None or margin >= margin_tau
            if not confident:
                mult = float(w_disagree)
        zp = _z([_clean(scores[i]) for i in idx])
        fused = [prior_w * a for a in zp] if prior_w != 1.0 else zp
        for zk, lam in zs:
            wk = lam * mult if mult != 1.0 else lam
            fused = [a + wk * b for a, b in zip(fused, zk)]
        if top is not None and agree_bonus:
            fused[top] += agree_bonus
        if len(zs) >= 2 and tie_eps > 0:
            fused = [round(x / tie_eps) for x in fused]
        new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
        for slot, src in zip(idx, new):
            order[slot] = src
    return order
