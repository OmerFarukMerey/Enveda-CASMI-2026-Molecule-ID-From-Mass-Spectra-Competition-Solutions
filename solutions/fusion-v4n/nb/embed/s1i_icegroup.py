"""s1i cell (ICEBERG formula-group fallback in s1's final E006 overlay), inserted verbatim before s1's cell 53; cell 53 gets one call.

Mechanism in s1 (cell 53 `rerank_row`, ice_fallback=True): when a molecule has both ICEBERG scores (`ICE_SCORES[mid]`) and GLACIER scores
(`gl_of(mid)`, non-empty if any value is not None), the 40-row window is re-ordered by `gl_fuse.rerank_multi(..., [ice if N1 proxy else {},
gl], [ICE_LAM, GL_LAM])`. Outside the N1 proxy molecules the ICE column is `{}`, and rerank_multi re-orders a same-formula group only if
GLACIER covers >= 2 of its members. A group with ZERO finite GLACIER scores but >= 2 finite ICEBERG scores therefore keeps its fused order
although ICEBERG scored it. Codex (debate round 4) rule: in exactly those groups, and only there, apply the chain's own ICE fallback
(`ice_fuse.rerank(..., lam=ICE_LAM)`, slot-preserving, the arithmetic of cell 53's `ice_fallback` branch); every other group keeps
GLACIER's order. Not touched: N1 proxy molecules (ICE already joint), the E004 control rebuild (ice_fallback=False), the frag-gap and
ICE-fallback branches. Groups with exactly one finite GLACIER score are only counted (`groups_one_gl`).
Switch S1I (on False = cell 53 runs s1's arithmetic: submission bytes unchanged). Self-test on synthetic lists first (fails -> off);
per-molecule error -> that molecule keeps GLACIER's order. Log: one S1I_PATCH {...} line, printed at the end of cell 53.
"""
import math

S1I = dict(on=True)


def s1i_new_stats(sw=None):
    return dict(dict(sw or {}), selftest='not run', calls=0, molecules_eligible=0, groups_eligible=0, groups_one_gl=0,
                groups_moved=0, molecules_reordered=0, molecules_top25_changed=0, molecules_top1_changed=0, fallback=0,
                eligible_mids=[])


S1I_STATS = s1i_new_stats(S1I)


def s1i_finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def s1i_lookup(d, smiles, key):
    """gl_fuse._getter / fuse.rerank semantics: the SMILES entry, else the key entry."""
    if not isinstance(d, dict):
        return None
    v = d.get(smiles)
    if v is None and key is not None:
        v = d.get(key)
    return v


def s1i_group_order(candidates, keys, scores, forms, ice, gl, gl_order, rerank, lam):
    """Pure. candidates / keys / scores / forms: the window in its fused order (before the GLACIER permutation); gl_order: rerank_multi's
    permutation. Returns (order, info): gl_order with every eligible group (>= 2 members, 0 finite GLACIER, >= 2 finite ICEBERG)
    replaced by rerank(..., ICE restricted to the eligible groups, lam=lam) on the same slots."""
    n = len(candidates)
    gl_order = list(gl_order)
    info = dict(groups_eligible=0, groups_one_gl=0, groups_moved=0, eligible_slots=[])
    if n == 0:
        return gl_order, info
    groups = {}
    for i in range(n):
        groups.setdefault(forms[i], []).append(i)
    k = keys if keys is not None else [None] * n
    elig = []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        n_gl = sum(s1i_finite(s1i_lookup(gl, candidates[i], k[i])) for i in idx)
        n_ice = sum(s1i_finite(s1i_lookup(ice, candidates[i], k[i])) for i in idx)
        if n_ice >= 2 and n_gl == 0:
            if all(gl_order[i] == i for i in idx):       # GLACIER left the group in place (it must: it covers none of it)
                elig.append(idx)
        elif n_ice >= 2 and n_gl == 1:
            info['groups_one_gl'] += 1
    if not elig:
        return gl_order, info
    ice_list = [None] * n                                   # aligned list (fuse.rerank accepts one): ICE only inside eligible groups
    for idx in elig:
        for i in idx:
            v = s1i_lookup(ice, candidates[i], k[i])
            ice_list[i] = float(v) if s1i_finite(v) else None
    o_ice = list(rerank(candidates, keys, scores, forms, ice_list, lam=lam, top_n=n))
    new = list(gl_order)
    for idx in elig:
        for i in idx:
            new[i] = o_ice[i]
        info['groups_moved'] += int(any(o_ice[i] != i for i in idx))
        info['eligible_slots'].extend(idx)
    info['groups_eligible'] = len(elig)
    assert sorted(new) == list(range(n)), 's1i order is not a permutation'
    assert all(new[i] == gl_order[i] for i in range(n) if i not in set(info['eligible_slots'])), 's1i touched a non-eligible slot'
    return new, info


def s1i_apply(scope, mid, candidates, keys, scores, order, ice, gl):
    """Called from cell 53's GLACIER branch (ice_fallback=True, not an N1 proxy molecule). Returns the permutation to use."""
    st = scope.get('S1I_STATS')
    sw = scope.get('S1I') or {}
    if not isinstance(st, dict) or not sw.get('on') or st.get('selftest') != 'ok':
        return order
    st['calls'] += 1
    try:
        forms = [scope['_form2'](s) for s in candidates]
        new, info = s1i_group_order(candidates, keys, scores, forms, ice, gl, order, scope['ice_fuse'].rerank, scope['ICE_LAM'])
        st['groups_eligible'] += info['groups_eligible']; st['groups_one_gl'] += info['groups_one_gl']
        st['groups_moved'] += info['groups_moved']
        if info['groups_eligible']:
            st['molecules_eligible'] += 1
            if len(st['eligible_mids']) < 200:
                st['eligible_mids'].append(str(mid))
        if new != list(order):
            st['molecules_reordered'] += 1
            st['molecules_top25_changed'] += int(new[:25] != list(order)[:25])
            st['molecules_top1_changed'] += int(new[:1] != list(order)[:1])
        return new
    except Exception as exc:
        st['fallback'] += 1
        print('S1I_FALLBACK', mid, repr(exc)[:200], flush=True)
        return order


def s1i_selftest(rerank):
    """Synthetic window: group A (GLACIER covers 2: kept as GLACIER ordered it), group B (0 GLACIER, 3 ICEBERG: re-ordered by ICE),
    group C (1 GLACIER, 2 ICEBERG: untouched, counted), group D (0 GLACIER, 1 ICEBERG: untouched)."""
    cands = ['a1', 'a2', 'b1', 'b2', 'b3', 'c1', 'c2', 'd1', 'd2']
    forms = ['A', 'A', 'B', 'B', 'B', 'C', 'C', 'D', 'D']
    keys = ['K' + c for c in cands]
    scores = [9.0, 8.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4]
    gl = {'a1': 0.1, 'a2': 0.9, 'c1': 0.3, 'd1': None}
    ice = {'a1': 1.0, 'a2': 0.0, 'b1': 0.0, 'Kb2': 0.5, 'b3': 5.0, 'c1': 0.0, 'c2': 9.0, 'd2': 3.0}   # 'Kb2': lookup by key
    gl_order = [1, 0, 2, 3, 4, 5, 6, 7, 8]
    new, info = s1i_group_order(cands, keys, scores, forms, ice, gl, gl_order, rerank, 1.0)
    assert new == [1, 0, 2, 4, 3, 5, 6, 7, 8], new
    assert (info['groups_eligible'], info['groups_one_gl'], info['groups_moved']) == (1, 1, 1), info
    same = rerank(cands[2:5], keys[2:5], scores[2:5], forms[2:5], {'b1': 0.0, 'b2': 0.5, 'b3': 5.0}, lam=1.0, top_n=3)
    assert [x + 2 for x in same] == new[2:5], same                      # = the chain's ICE fallback on that group alone
    new2, info2 = s1i_group_order(cands, keys, scores, forms, {}, gl, gl_order, rerank, 1.0)
    assert new2 == gl_order and info2['groups_eligible'] == 0
    new3, _ = s1i_group_order(cands, keys, scores, forms, ice, dict(gl, b1=0.2), gl_order, rerank, 1.0)
    assert new3 == gl_order                                             # one GLACIER score in B -> not eligible
    nan = float('nan')
    new4, _ = s1i_group_order(cands, keys, scores, forms, dict(ice, b3=nan, Kb2=nan), gl, gl_order, rerank, 1.0)
    assert new4 == gl_order                                             # < 2 finite ICE in B
    assert s1i_group_order([], [], [], [], ice, gl, [], rerank, 1.0)[0] == []
    return 'ok'


if __name__ == '__main__':
    S1I_STATS = s1i_new_stats(S1I)
    if S1I.get('on'):
        try:
            S1I_STATS['selftest'] = s1i_selftest(ice_fuse.rerank)   # noqa: F821 (cell 42's ICEBERG module)
        except Exception as _s1i_exc:
            S1I_STATS['selftest'] = 'FAILED ' + repr(_s1i_exc)[:200]
            print('S1I_SELFTEST_FAILED patch switched off', repr(_s1i_exc)[:300], flush=True)
    print('S1I_SWITCH', {k: S1I_STATS[k] for k in ('on', 'selftest') if k in S1I_STATS}, flush=True)
