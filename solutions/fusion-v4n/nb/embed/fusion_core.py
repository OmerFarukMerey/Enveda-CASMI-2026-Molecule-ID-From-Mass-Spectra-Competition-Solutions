"""Final-stage fusion for casmi26-ours: ICE/GLACIER re-ranking -> gated PubChem merge -> engine-2 RRF fusion.

Pure function over the stage caches, so the Kaggle notebook and tools/refuse.py run the exact same code.
With the default CFG this reproduces the reference notebook v4n (seyitkaangunes, LB 0.399) step for step:
  stage A  z(ranker) + ICE_LAM z(ICEBERG) + GL_LAM z(GLACIER) inside same-formula groups (base + PubChem lists)
  stage B  gated PubChem merge: untouched if lib_max >= LIB_TAU, else PubChem-only structures into fixed slots
  stage C  weighted reciprocal-rank fusion with engine-2 lists, then the ICE/GL re-rank again on the fused list

Inputs (all JSON-able, as dumped to stage_cache/):
  base  {mid: [smis, keys, lib_max, scores, formulas]}      v1 engine + ranker, top TOPN
  pc    {mid: {pc, pc_fz, pc_keys, best_pool_fz, target}}  PubChem-only channel
  eng   {mid: {smiles, keys}}                               engine-2 lists
  ice / gl  {mid: {smiles: score | None}}                   re-scoring terms
"""
import copy

import numpy as np
import pandas as pd

DEFAULTS = dict(LIB_TAU=0.9, REL_TH=600.0, SLOTS_AGG=[2, 4, 6, 8, 10], SLOTS_GENTLE=[4, 8, 12, 16, 20], TOPN=60,
                ICE_LAM=1.0, GL_LAM=1.0, ICE_PC=True, FUSE_ALPHA=0.6, FUSE_KRR=3.0, FUSE_N=40, FALLBACK='CCO',
                POP_MU=0.0, FUSE_ENG_K=40, FRAG_LAM=0.0, FRAG_MODE='all',
                POST_ICE_LAM=None, POST_GL_LAM=None,   # stage-C re-rank weights; None = ICE_LAM / GL_LAM
                FUSE_SKIP_LIB=None,                    # skip engine-2 fusion when lib_max >= this; None = never
                ICE_LAM_LIB=None, GL_LAM_LIB=None,     # ICE/GL weights for library hits (lib_max >= LIB_TAU); None = same
                FILL_25=False,                         # top up lists shorter than 25 from the other channels
                KEY_NOSTEREO=False,                    # de-duplicate the final list by the metric key of 2026-10-07 (stereo stripped first)
                FUSE_KEY_VOTE=None,                    # engine fusion keyed by that metric key, one vote per key and engine: 'best' = a key
                                                       #   votes at its first rank in each list, later rows of the key are skipped; 'compact' =
                                                       #   the skipped rows do not consume a rank either (flexonafft v80). None = off
                                                       #   (reviews/keyvote-2026-10-09.md)
                POP_LIB_OFF=False, FRAG_LIB_OFF=False, # no popularity / frag term for library hits
                GATE_TAU=None,                         # lib_max threshold of the *_LIB gates; None = LIB_TAU
                FUSE_MIN_LIB=None,                     # fuse engine-2 only when lib_max >= this (it needs library analogs)
                PC_FWD_T1=None, PC_FWD_N=25,           # early PubChem slots when the best proposal beats the base
                                                       #   candidates in the forward models by more than this; None = off
                FUSE_INJ_CREDIT=0.0,                   # engine fusion: a joined PubChem row (pool id -2) gets this share of the
                                                       #   engine-2 weight at its own rank, because engine-2 cannot list it
                JOIN_FWD_A=None,                       # stage A: ICEBERG / GLACIER terms of joined PubChem rows (pool id -2) times
                                                       #   this, other rows keep 1.0; None = off (reviews/calib-2026-10-05.md)
                FWD_A_SCALE=None,                      # stage A, base list only: [ice, gl] factors on ICE_LAM / GL_LAM; the PubChem
                                                       #   list and stage C keep the full weights; None = off (reviews/fsranker2-2026-10-07.md)
                POP_GROUP_MU=None,                     # stage A: in-group popularity term, mu x z(popularity) inside each same-formula
                                                       #   group of pool rows, added to the stage-A sum; POP_MU keeps acting across the
                                                       #   list (0 = in-group only). None = off (reviews/popgroup-2026-10-08.md)
                POP_GROUP_MAX_LIB=None,                # ... only for molecules with lib_max below this; None = every molecule
                NP_MU=0.0,                             # natural-product-likeness term (np_score, structure only) on pool rows, added where
                                                       #   the popularity prior acts: f = z + POP_MU pop + NP_MU s; 0 = off
                NP_JOIN_MU=0.0,                        # ... on joined PubChem rows (pool id -2): f = z + NP_JOIN_MU s, re-sorted among the
                                                       #   joined rows' own slots; 0 = off (reviews/nplike-2026-10-08.md)
                ADD_JOIN_MU=0.0,                       # adduct-evidence term (add_score, per molecule) on joined PubChem rows, same
                                                       #   arithmetic and slots as NP_JOIN_MU; 0 = off (reviews/newinfo-2026-10-08.md)
                FWD_GL_ALONE=False)                    # robustness: a molecule WITHOUT an ICEBERG score but WITH GLACIER scores (ICEBERG crashed
                                                       #   or stopped at its budget) is re-ranked by GLACIER alone in all three re-ranks and keeps
                                                       #   the fragment 'gap' term out; off = the reference: no forward re-rank at all for it
                                                       #   (docs/landscape/canqiang-e006-2026-10-10.md)


def popularity_reorder(smis, keys, scs, forms, pids, pool_pop, mu):
    """bobthebot369 v10 prior: f = z(ranker) + mu * pop[pid] for pool candidates; they are re-sorted among the pool
    slots, generated candidates (pid < 0) keep their slots and get a '|gen' formula tag (own same-formula group)."""
    s = np.asarray(scs, np.float64); z = (s - s.mean()) / (s.std() + 1e-9)
    pid = np.asarray(pids); isg = pid < 0
    f = z + mu * np.where(isg, 0.0, pool_pop[np.maximum(pid, 0)])
    pool_i = np.where(~isg)[0]; o = np.empty(len(pid), np.int64)
    o[isg] = np.where(isg)[0]; o[~isg] = pool_i[np.argsort(-f[pool_i], kind='stable')]
    return ([smis[i] for i in o], [keys[i] for i in o], [float(f[i]) for i in o],
            [forms[i] + ('|gen' if isg[i] else '') for i in o], [int(pid[i]) for i in o])


def np_reorder(smis, keys, scs, forms, pids, pool_pop, mu, np_vals, np_mu, np_join_mu):
    """popularity_reorder with a natural-product-likeness term s (np_vals, one value per row, 0 = unknown): pool rows
    f = z + mu * pop + np_mu * s, re-sorted among the pool slots; joined PubChem rows (pid == -2) f = z + np_join_mu * s,
    re-sorted among the joined rows' slots, so a joined row never passes a pool row here; generator products (pid == -1)
    keep their slots. Tags as in popularity_reorder. mu = 0 or pool_pop None: no popularity term."""
    s = np.asarray(scs, np.float64); z = (s - s.mean()) / (s.std() + 1e-9)
    pid = np.asarray(pids); isg = pid < 0; isj = pid == -2
    v = np.nan_to_num(np.asarray(np_vals, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    pop = pool_pop[np.maximum(pid, 0)] if (mu and pool_pop is not None) else np.zeros(len(pid))
    f = z + np.where(isg, 0.0, mu * pop + np_mu * v) + np.where(isj, np_join_mu * v, 0.0)
    o = np.arange(len(pid), dtype=np.int64)
    pool_i = np.where(~isg)[0]; o[pool_i] = pool_i[np.argsort(-f[pool_i], kind='stable')]
    if np_join_mu:
        j_i = np.where(isj)[0]; o[j_i] = j_i[np.argsort(-f[j_i], kind='stable')]
    return ([smis[i] for i in o], [keys[i] for i in o], [float(f[i]) for i in o],
            [forms[i] + ('|gen' if isg[i] else '') for i in o], [int(pid[i]) for i in o])


def pop_group_rerank(order_std, smis, keys, scores, formulas, pids, score_dicts, lams, pool_pop, mu, top_n, tie_eps=1e-9):
    """POP_GROUP_MU: every same-formula group of pool rows (pid >= 0, two or more) among the first top_n rows is re-sorted by
    z(score) + sum_k lam_k z(forward_k) + mu z(popularity), all z inside the group (the arithmetic of gl_fuse.rerank_multi
    with the popularity as one more re-scorer), and put back into the slots of the group. order_std = the standard stage-A
    order (indices into smis); groups without two pool rows keep it. Returns the new order."""
    def num(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    n = len(smis); order = list(order_std); use = [(d, float(l)) for d, l in zip(score_dicts, lams) if d]
    groups = {}
    for i in range(min(int(top_n), n)):
        groups.setdefault(formulas[i], []).append(i)
    for idx in groups.values():
        pv = [float(pool_pop[pids[i]]) if pids[i] >= 0 else None for i in idx]
        if len(idx) < 2 or sum(x is not None for x in pv) < 2:
            continue
        fused = _zs([float(scores[i]) for i in idx]); n_act = 1
        for d, lam in use:
            v = [num(d.get(smis[i]) if d.get(smis[i]) is not None else (d.get(keys[i]) if keys is not None else None)) for i in idx]
            if sum(x is not None for x in v) >= 2:
                fused = [a + lam * b for a, b in zip(fused, _zs(v))]; n_act += 1
        fused = [a + mu * b for a, b in zip(fused, _zs(pv))]
        if n_act >= 2 and tie_eps > 0:
            fused = [round(x / tie_eps) for x in fused]
        new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
        for slot, src in zip(idx, new):
            order[slot] = src
    return order


def merge(base, base_keys, pc, pc_keys, slots, n=25):
    """PubChem-only structures (not already in base) go into `slots` (1-based); base fills the rest."""
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
    """min over the forward models of score(proposal) - max score(base top-n candidates with the proposal's formula),
    or None. A model counts only if it scored the proposal and at least one compared base candidate
    (reviews/pubchem-2026-10-02.md: a positive margin means the answer is rarely in the pool)."""
    cand = list(base[:n])
    if formula is not None:
        fp = formula(prop)
        cand = [b for b in cand if formula(b) == fp]
    ok = lambda x: isinstance(x, (int, float)) and np.isfinite(x)
    margins = []
    for d in scores:
        v = d.get(prop) if d else None
        bs = [x for x in (d.get(b) for b in cand) if ok(x)] if ok(v) else []
        if bs:
            margins.append(v - max(bs))
    return min(margins) if margins else None


def _zs(vals):
    """z over the finite values (ddof 1); missing, a single value or zero spread -> 0 (as fuse._z / gl_fuse._z)."""
    xs = [float(v) for v in vals if v is not None]
    if len(xs) < 2:
        return [0.0] * len(vals)
    mu = sum(xs) / len(xs); sd = float(np.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)))
    if not sd > 0:
        return [0.0] * len(vals)
    return [((float(v) - mu) / sd) if v is not None else 0.0 for v in vals]


def rerank_joined(smis, keys, scores, formulas, score_dicts, lams, top_n, joined, w, tie_eps=1e-9):
    """gl_fuse.rerank_multi with the forward terms of the rows flagged in `joined` multiplied by w (JOIN_FWD_A).
    Inside each same-formula group of the first top_n rows the members are re-sorted by z(score) + sum_k lam_k z_k
    (x w for a joined row) and put back into the group's positions; w = 1 gives rerank_multi's order."""
    def num(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    n = len(smis); order = list(range(n))
    use = [(d, float(l)) for d, l in zip(score_dicts, lams) if d]
    if n == 0 or not use:
        return order
    groups = {}
    for i in range(min(int(top_n), n)):
        groups.setdefault(formulas[i], []).append(i)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        cols = []
        for d, lam in use:
            v = [num(d.get(smis[i]) if d.get(smis[i]) is not None else (d.get(keys[i]) if keys is not None else None)) for i in idx]
            if sum(x is not None for x in v) >= 2:
                cols.append((_zs(v), lam))
        if not cols:
            continue
        fused = _zs([float(scores[i]) for i in idx])
        for z, lam in cols:
            fused = [a + lam * b * (w if joined[i] else 1.0) for a, b, i in zip(fused, z, idx)]
        if len(cols) >= 2 and tie_eps > 0:
            fused = [round(x / tie_eps) for x in fused]
        new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
        for slot, src in zip(idx, new):
            order[slot] = src
    return order


def _marked(s):
    return isinstance(s, str) and ('@' in s or '/' in s or chr(92) in s)


def fuse_rrf(v_smis, e_smis, e_keys, score_key, alpha, krr, n, credit=None, credit_w=0.0, key_fn=None, compact=False):
    """sum 1/(krr + r) over the v4 list plus alpha/(krr + r) over the engine list, keyed by the metric key.
    credit: SMILES of the v4 list that engine-2 could not have listed (joined PubChem rows); each gets
    credit_w * alpha / (krr + r) on top, i.e. is treated as if engine-2 ranked it where the v4 list does.
    key_fn (FUSE_KEY_VOTE): the stereo-free metric key; every key gets ONE vote per list, at its first rank there (compact: at
    its rank among the list's distinct keys). An engine row without a stereo mark keeps its delivered key (the same function)."""
    sc, smi_of = {}, {}
    if key_fn is not None:
        seen = set()
        for r, s in enumerate(v_smis, 1):
            k = key_fn(s) or s
            if k in seen:
                continue
            seen.add(k)
            w = 1.0 + (credit_w * alpha if (credit and s in credit) else 0.0)
            sc[k] = sc.get(k, 0.0) + w / (krr + (len(seen) if compact else r)); smi_of.setdefault(k, s)
        seen = set()
        for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
            k = (key_fn(s) or k) if _marked(s) else k
            if not k or k in seen:
                continue
            seen.add(k)
            sc[k] = sc.get(k, 0.0) + alpha / (krr + (len(seen) if compact else r)); smi_of.setdefault(k, s)
        order = sorted(sc, key=lambda k: -sc[k])[:n]
        return [smi_of[k] for k in order], order, [sc[k] for k in order]
    for r, s in enumerate(v_smis, 1):
        k = score_key(s) or s
        w = 1.0 + (credit_w * alpha if (credit and s in credit) else 0.0)
        sc[k] = sc.get(k, 0.0) + w / (krr + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + alpha / (krr + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]


def build_submission(base, pc, eng, ice_scores, gl_scores, cfg, mol_order, sample_ids,
                     ice_fuse=None, gl_fuse=None, score_key=None, formula=None, log=print, pool_pop=None,
                     frag_scores=None, np_score=None, add_score=None, nostereo_key=None):
    """Returns (submission DataFrame in sample order, stats dict). Inputs are not mutated."""
    c = dict(DEFAULTS); c.update({k: v for k, v in cfg.items() if k in DEFAULTS})
    base = copy.deepcopy(base); pc = copy.deepcopy(pc) if isinstance(pc, dict) else {}
    eng = eng or {}; ice_scores = ice_scores or {}; gl_scores = gl_scores or {}
    ICE_LAM, GL_LAM, TOPN, LIB_TAU = c['ICE_LAM'], c['GL_LAM'], c['TOPN'], c['LIB_TAU']
    stats = dict(ice=dict(molecules=0, changed_top25=0, changed_top1=0), gl=dict(molecules=0),
                 merge=dict(untouched=0, gentle=0, aggressive=0, no_pc=0), fuse=dict(top1_changed=0, ice_reordered=0))

    def gl_of(m):
        g = gl_scores.get(str(m), {}) if isinstance(gl_scores, dict) else {}
        return g if isinstance(g, dict) and any(v is not None for v in g.values()) else {}

    def gl_rerank(order0, smis, keys, scs, forms, ice, gl, top_n, tag=''):
        if not gl or gl_fuse is None:
            return order0
        try:
            o = gl_fuse.rerank_multi(smis, keys, scs, forms, [ice, gl], [ICE_LAM, GL_LAM], top_n=top_n)
        except Exception as e:
            log('gl rerank failed', tag, repr(e)); return order0
        g = stats['gl']
        for k, v in (('molecules', 1), ('changed_vs_ice_top25', int(o[:25] != order0[:25])),
                     ('changed_vs_ice_top1', int(o[:1] != order0[:1]))):
            g[tag + k] = g.get(tag + k, 0) + v
        return o

    def covered(d):
        return bool(d) and any(v is not None for v in d.values())

    def frag_rerank(m, smis, keys, scs, forms, ice, gl, top_n):
        """MetFrag-parsimony term (frag_rescore.py). 'all': one joint z-sum with ICE/GL; 'gap': only molecules ICE
        does not cover (e.g. negative-mode or [M+NH4]+ only). None -> leave the current order."""
        fr = (frag_scores or {}).get(str(m), {})
        if c['FRAG_LAM'] <= 0 or not covered(fr) or gl_fuse is None:
            return None
        if c['FRAG_LIB_OFF'] and lib_of_m.get(m, lib_of_m.get(str(m), 0.0)) >= GATE_TAU:
            return None
        if c['FRAG_MODE'] == 'gap':
            if covered(ice) or (c['FWD_GL_ALONE'] and covered(gl)):
                return None
            terms, lams = [fr], [c['FRAG_LAM']]
        else:
            terms = [d for d in (ice, gl) if covered(d)] + [fr]
            lams = [l for d, l in ((ice, ICE_LAM), (gl, GL_LAM)) if covered(d)] + [c['FRAG_LAM']]
        stats.setdefault('frag', {'molecules': 0})['molecules'] += 1
        return gl_fuse.rerank_multi(smis, keys, scs, forms, terms, lams, top_n=top_n)

    GATE_TAU = LIB_TAU if c['GATE_TAU'] is None else c['GATE_TAU']   # LIB_TAU itself keeps steering the PubChem merge

    def lam_pair(lib_max):
        lib = lib_max >= GATE_TAU
        il = c['ICE_LAM_LIB'] if (lib and c['ICE_LAM_LIB'] is not None) else c['ICE_LAM']
        gll = c['GL_LAM_LIB'] if (lib and c['GL_LAM_LIB'] is not None) else c['GL_LAM']
        return il, gll

    extra = {}                                   # FILL_25: re-ranked base candidates beyond position 25
    inj_of = {}                                  # FUSE_INJ_CREDIT: joined PubChem rows (pool id -2) per molecule
    lib_of_m = {m: v[2] for m, v in base.items()}

    # ---- stage A: ICE (+GL) re-rank of the base lists and of the PubChem lists of gated molecules
    for m in list(base):
        v = base[m]
        smis, keys, lib_max, scs, forms = list(v[0][:TOPN]), list(v[1][:TOPN]), v[2], list(v[3][:TOPN]), list(v[4][:TOPN])
        pids = list(v[5][:TOPN]) if len(v) > 5 else None
        ICE_LAM, GL_LAM = lam_pair(lib_max)       # closures (gl_rerank, frag_rerank) read these per molecule
        lam_full = (ICE_LAM, GL_LAM)
        if c['FWD_A_SCALE'] is not None:          # the base list's re-rank with scaled forward weights
            ICE_LAM, GL_LAM = ICE_LAM * float(c['FWD_A_SCALE'][0]), GL_LAM * float(c['FWD_A_SCALE'][1])
        pop_on = c['POP_MU'] > 0 and not (c['POP_LIB_OFF'] and lib_max >= GATE_TAU)
        pg_on = (c['POP_GROUP_MU'] is not None and c['POP_GROUP_MU'] > 0 and pool_pop is not None and pids is not None and len(scs) > 1
                 and not (c['POP_GROUP_MAX_LIB'] is not None and lib_max >= c['POP_GROUP_MAX_LIB']))
        np_on = (c['NP_MU'] > 0 or c['NP_JOIN_MU'] > 0) and np_score is not None and pids is not None and len(scs) > 1
        if np_on:                                 # NP-likeness term; carries the popularity term as well when that is on
            try:
                nv = np_score(smis) if callable(np_score) else [np_score.get(x) for x in smis]
                nv = [0.0 if x is None else float(x) for x in nv]
                smis, keys, scs, forms, pids = np_reorder(smis, keys, scs, forms, pids, pool_pop,
                                                          c['POP_MU'] if (pop_on and pool_pop is not None) else 0.0, nv, c['NP_MU'], c['NP_JOIN_MU'])
                stats.setdefault('nplike', {'molecules': 0})['molecules'] += 1
            except Exception as e:
                log('NP-likeness term failed', m, repr(e)); np_on = False
        if (not np_on and c['ADD_JOIN_MU'] > 0 and add_score is not None and pids is not None and len(scs) > 1
                and sum(1 for x in pids if x == -2) > 1):
            try:                                  # adduct evidence on joined rows; add_score(m, smis) -> one value per row, or None
                av = add_score(m, smis)
                if av is not None:
                    av = [0.0 if x is None else float(x) for x in av]
                    smis, keys, scs, forms, pids = np_reorder(smis, keys, scs, forms, pids, pool_pop,
                                                              c['POP_MU'] if (pop_on and pool_pop is not None) else 0.0, av, 0.0, c['ADD_JOIN_MU'])
                    stats.setdefault('adduct', {'molecules': 0})['molecules'] += 1; np_on = True
            except Exception as e:
                log('adduct-evidence term failed', m, repr(e))
        if not np_on and (pop_on or pg_on) and pool_pop is not None and pids is not None and len(scs) > 1:
            try:                                  # pg_on without POP_MU: weight 0, the rows still get their '|gen' tags
                smis, keys, scs, forms, pids = popularity_reorder(smis, keys, scs, forms, pids, pool_pop, c['POP_MU'] if pop_on else 0.0)
                stats.setdefault('pop', {'molecules': 0})['molecules'] += 1
            except Exception as e:
                log('popularity prior failed', m, repr(e))
        if pids is not None and c['FUSE_INJ_CREDIT'] > 0:
            inj_of[m] = {s for s, p in zip(smis, pids) if p == -2}
        ice = ice_scores.get(str(m), {}) if ice_scores else {}
        gl = gl_of(m)
        smis0, keys0, scs0, forms0 = smis, keys, scs, forms
        if ice_fuse is not None and ice and any(v is not None for v in ice.values()):
            try:
                jrow = [p == -2 for p in pids] if (c['JOIN_FWD_A'] is not None and pids is not None) else None
                if jrow and any(jrow):                # joined rows: forward terms at a reduced weight
                    order = rerank_joined(smis, keys, scs, forms, [ice, gl if gl_fuse is not None else {}],
                                          [ICE_LAM, GL_LAM], TOPN, jrow, float(c['JOIN_FWD_A']))
                    stats['ice']['join_fwd_a'] = stats['ice'].get('join_fwd_a', 0) + 1
                else:
                    order = ice_fuse.rerank(smis, keys, scs, forms, ice, lam=ICE_LAM, top_n=TOPN)
                    order = gl_rerank(order, smis, keys, scs, forms, ice, gl, TOPN)
                s2 = [smis[i] for i in order]; k2 = [keys[i] for i in order]
                st = stats['ice']; st['molecules'] += 1
                st['changed_top25'] += int(s2[:25] != smis[:25]); st['changed_top1'] += int(bool(s2) and s2[0] != smis[0])
                smis, keys = s2, k2
            except Exception as e:
                log('rerank failed', m, repr(e))
        elif c['FWD_GL_ALONE'] and gl and gl_fuse is not None:   # no ICEBERG score for this molecule: GLACIER alone
            try:
                jrow = [p == -2 for p in pids] if (c['JOIN_FWD_A'] is not None and pids is not None) else None
                if jrow and any(jrow):
                    order = rerank_joined(smis, keys, scs, forms, [{}, gl], [ICE_LAM, GL_LAM], TOPN, jrow, float(c['JOIN_FWD_A']))
                else:
                    order = gl_fuse.rerank_multi(smis, keys, scs, forms, [{}, gl], [ICE_LAM, GL_LAM], top_n=TOPN)
                s2 = [smis[i] for i in order]; k2 = [keys[i] for i in order]
                st = stats.setdefault('gl_alone', {'molecules': 0, 'changed_top1': 0, 'pc': 0, 'fused': 0})
                st['molecules'] += 1; st['changed_top1'] += int(bool(s2) and s2[0] != smis[0])
                smis, keys = s2, k2
            except Exception as e:
                log('GLACIER-alone rerank failed', m, repr(e))
        try:
            o = frag_rerank(m, smis0, keys0, scs0, forms0, ice, gl, TOPN)
            if o is not None:
                smis, keys = [smis0[i] for i in o], [keys0[i] for i in o]
        except Exception as e:
            log('frag rerank failed', m, repr(e))
        if pg_on:                                 # in-group popularity term on top of the standard stage-A order
            try:
                ix = {s_: i_ for i_, s_ in enumerate(smis0)}
                fr = (frag_scores or {}).get(str(m), {})
                fr_on = (c['FRAG_LAM'] > 0 and covered(fr) and gl_fuse is not None
                         and not (c['FRAG_LIB_OFF'] and lib_max >= GATE_TAU))
                if fr_on and c['FRAG_MODE'] != 'gap':
                    terms = [d for d in (ice, gl) if covered(d)] + [fr]
                    lams = [l for d, l in ((ice, ICE_LAM), (gl, GL_LAM)) if covered(d)] + [c['FRAG_LAM']]
                elif fr_on and not covered(ice):
                    terms, lams = [fr], [c['FRAG_LAM']]
                elif ice_fuse is not None and covered(ice):
                    terms, lams = [ice] + ([gl] if gl and gl_fuse is not None else []), [ICE_LAM, GL_LAM]
                else:
                    terms, lams = [], []
                if len(ix) == len(smis0) and len(smis) == len(smis0):
                    o = pop_group_rerank([ix[s_] for s_ in smis], smis0, keys0, scs0, forms0, pids, terms, lams, pool_pop,
                                         float(c['POP_GROUP_MU']), TOPN)
                    smis, keys = [smis0[i] for i in o], [keys0[i] for i in o]
                    stats.setdefault('pop_group', {'molecules': 0})['molecules'] += 1
            except Exception as e:
                log('in-group popularity term failed', m, repr(e))
        base[m] = (smis[:25], keys[:25], lib_max)
        extra[m] = smis[25:]
        ICE_LAM, GL_LAM = lam_full                # the PubChem list below keeps the full weights
        p = pc.get(m) if c['ICE_PC'] else None
        gl_alone = bool(c['FWD_GL_ALONE'] and gl and gl_fuse is not None and not covered(ice))
        if ice_fuse is not None and p and p.get('pc') and lib_max < LIB_TAU and (ice or gl_alone):
            try:
                if not p.get('pc_form'):
                    p['pc_form'] = [formula(x) for x in p['pc']]
                o = ice_fuse.rerank(p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, lam=ICE_LAM, top_n=25)
                o = gl_rerank(o, p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, gl, 25, 'pc_')
                fz0 = p['pc_fz'][0]          # the gate keeps the ORIGINAL top f.z, so gate decisions do not move
                p['pc'] = [p['pc'][i] for i in o]; p['pc_keys'] = [p['pc_keys'][i] for i in o]
                p['pc_fz'] = [fz0] + [p['pc_fz'][i] for i in o][1:]
                stats['ice']['pc_changed'] = stats['ice'].get('pc_changed', 0) + int(o[:1] != [0])
                if gl_alone:
                    stats.setdefault('gl_alone', {'molecules': 0, 'changed_top1': 0, 'pc': 0, 'fused': 0})['pc'] += 1
            except Exception as e:
                log('pc rerank failed', m, repr(e))

    # ---- stage B: gated PubChem merge
    rows = {}
    for mid in mol_order:
        smis, keys, lib_max = base.get(mid, ([], [], 0.0))
        p = pc.get(mid)
        ms = stats['merge']
        if p is None or not p.get('pc'):
            ms['no_pc'] += 1; final = smis
        elif lib_max >= LIB_TAU:
            ms['untouched'] += 1; final = smis
        else:
            bp = p.get('best_pool_fz')
            top = p['fz_top'] if p.get('fz_top') is not None else p['pc_fz'][0]   # fz_top: raw f.z when the list is popularity-sorted
            rel = top - bp if bp is not None and np.isfinite(bp) else 1e9
            aggressive = rel > c['REL_TH']
            if not aggressive and c['PC_FWD_T1'] is not None:
                bk = set(keys)
                new = [s for s, k in zip(p['pc'], p['pc_keys']) if k not in bk]
                fwd = forward_margin(new[0], smis, [ice_scores.get(str(mid), {}), gl_of(mid)], c['PC_FWD_N'],
                                     formula) if new else None
                if fwd is not None and fwd > c['PC_FWD_T1']:
                    aggressive = True; ms['fwd_early'] = ms.get('fwd_early', 0) + 1
            ms['aggressive' if aggressive else 'gentle'] += 1
            final = merge(smis, keys, p['pc'], p['pc_keys'], c['SLOTS_AGG'] if aggressive else c['SLOTS_GENTLE'])
        if not final:
            final = [c['FALLBACK']]
        rows[mid] = ';'.join(final[:25])
    v4 = [(mid, rows.get(mid, c['FALLBACK'])) for mid in sample_ids]
    lib_of = {m: v[2] for m, v in base.items()}

    NSK = nostereo_key if (c['KEY_NOSTEREO'] and nostereo_key is not None) else None
    KVF = None                                   # FUSE_KEY_VOTE: the key function of the engine fusion
    if c['FUSE_KEY_VOTE'] is not None:
        assert c['FUSE_KEY_VOTE'] in ('best', 'compact'), c['FUSE_KEY_VOTE']
        KVF = nostereo_key if nostereo_key is not None else make_nostereo_key(score_key)

    def ns_cut(full):
        """KEY_NOSTEREO: rows that share the stereo-stripped metric key with an earlier row are dropped BEFORE the cut to 25."""
        d = dedup_by(full, NSK)
        st = stats.setdefault('nostereo', {'molecules': 0, 'removed': 0, 'removed_top25': 0})
        st['molecules'] += int(len(d) < len(full)); st['removed'] += len(full) - len(d); st['removed_top25'] += len(full[:25]) - len(dedup_by(full[:25], NSK))
        return d

    def fill(mid, lst):
        """FILL_25: append engine-2, PubChem and deeper base candidates (metric-key dedup) until 25."""
        if not c['FILL_25'] or len(lst) >= 25:
            return lst
        fk_ = NSK or score_key
        seen = {fk_(x) or x for x in lst}; out = list(lst)
        pool = list((eng.get(str(mid)) or {}).get('smiles', [])) + list((pc.get(mid) or {}).get('pc', [])) + list(extra.get(mid, []))
        for x in pool:
            k = fk_(x) or x
            if k not in seen:
                seen.add(k); out.append(x)
                if len(out) >= 25:
                    break
        stats.setdefault('fill', {'molecules': 0, 'added': 0})
        stats['fill']['molecules'] += int(len(out) > len(lst)); stats['fill']['added'] += len(out) - len(lst)
        return out

    # ---- stage C: engine-2 fusion (+ ICE/GL on the fused list)
    if not eng:
        if c['FILL_25']:
            v4 = [(mid, ';'.join(fill(mid, [x for x in s.split(';') if x and x != c['FALLBACK']])) or c['FALLBACK'])
                  for mid, s in v4]
        if NSK is not None:
            v4 = [(mid, ';'.join(fill(mid, ns_cut([x for x in s.split(';') if x and x != c['FALLBACK']]))[:25]) or c['FALLBACK']) for mid, s in v4]
        return pd.DataFrame(v4, columns=['molecule_id', 'smiles']), stats
    out = []
    for mid, s in v4:
        vs = [x for x in s.split(';') if x and x != c['FALLBACK']]
        ICE_LAM, GL_LAM = lam_pair(lib_of.get(mid, 0.0))
        P_ICE = ICE_LAM if c['POST_ICE_LAM'] is None else c['POST_ICE_LAM']
        P_GL = GL_LAM if c['POST_GL_LAM'] is None else c['POST_GL_LAM']
        e = eng.get(str(mid))
        if c['FUSE_SKIP_LIB'] is not None and lib_of.get(mid, 0.0) >= c['FUSE_SKIP_LIB']:
            e = None
            stats['fuse']['skipped_lib'] = stats['fuse'].get('skipped_lib', 0) + 1
        if c['FUSE_MIN_LIB'] is not None and lib_of.get(mid, 0.0) < c['FUSE_MIN_LIB']:
            e = None
            stats['fuse']['skipped_nolib'] = stats['fuse'].get('skipped_nolib', 0) + 1
        if e and e.get('smiles'):
            K = int(c['FUSE_ENG_K'])
            fsm, fk, fsc = fuse_rrf(vs, e['smiles'][:K], e['keys'][:K], score_key, c['FUSE_ALPHA'], c['FUSE_KRR'], c['FUSE_N'],
                                    inj_of.get(mid), c['FUSE_INJ_CREDIT'], *(() if KVF is None else (KVF, c['FUSE_KEY_VOTE'] == 'compact')))
            ice = ice_scores.get(str(mid), {}) if isinstance(ice_scores, dict) else {}
            fsm0, fk0, fsc0 = fsm, fk, fsc
            gl_alone = bool(c['FWD_GL_ALONE'] and gl_fuse is not None and not covered(ice) and gl_of(mid))
            if (ice or gl_alone) and ice_fuse is not None:
                try:
                    forms = [formula(x) for x in fsm]
                    o = ice_fuse.rerank(fsm, fk, fsc, forms, ice, lam=P_ICE, top_n=len(fsm))
                    g = gl_of(mid)
                    if g and gl_fuse is not None:
                        o = gl_fuse.rerank_multi(fsm, fk, fsc, forms, [ice, g], [P_ICE, P_GL], top_n=len(fsm))
                    stats['fuse']['ice_reordered'] += int(o[:25] != list(range(min(25, len(o)))))
                    if gl_alone:
                        stats.setdefault('gl_alone', {'molecules': 0, 'changed_top1': 0, 'pc': 0, 'fused': 0})['fused'] += 1
                    fsm = [fsm[i] for i in o]
                except Exception as ex:
                    log('fused ICE rerank failed', mid, repr(ex))
            if c['FRAG_LAM'] > 0:
                try:
                    o = frag_rerank(mid, fsm0, fk0, fsc0, [formula(x) for x in fsm0], ice, gl_of(mid), len(fsm0))
                    if o is not None:
                        fsm = [fsm0[i] for i in o]
                except Exception as ex:
                    log('fused frag rerank failed', mid, repr(ex))
            stats['fuse']['top1_changed'] += int(bool(vs) and fsm[:1] != vs[:1])
            vs = fsm[:25] if NSK is None else ns_cut(fsm)[:25]
        elif NSK is not None:
            vs = ns_cut(vs)
        vs = fill(mid, vs[:25])
        out.append((mid, ';'.join(vs[:25]) if vs else c['FALLBACK']))
    return pd.DataFrame(out, columns=['molecule_id', 'smiles']), stats


def make_nostereo_key(score_key):
    """The metric's key since 2026-10-07: RemoveStereochemistry, then the tautomer canonicalisation, then InChIKey14.
    A SMILES without a stereo mark has nothing to strip, so it keeps `score_key` (no second canonicalisation)."""
    cache = {}; rd = {}; bs = chr(92)                  # the three stereo marks of SMILES: @, / and the backslash

    def key(s):
        if not isinstance(s, str) or not ('@' in s or '/' in s or bs in s):
            return score_key(s)
        if s not in cache:
            try:
                if not rd:
                    from rdkit import Chem
                    from rdkit.Chem.MolStandardize import rdMolStandardize
                    rd['Chem'] = Chem; rd['taut'] = rdMolStandardize.TautomerEnumerator()
                mol = rd['Chem'].MolFromSmiles(s)
                if mol is None:
                    cache[s] = None
                else:
                    rd['Chem'].RemoveStereochemistry(mol)
                    ik = rd['Chem'].MolToInchiKey(rd['taut'].Canonicalize(mol))
                    cache[s] = ik[:14] if ik else None
            except Exception:
                cache[s] = None
        return cache[s]
    return key


def dedup_by(smis, key):
    """First row of every key, order kept (a row without a key is its own key)."""
    seen = set(); out = []
    for s in smis:
        k = key(s) or s
        if k not in seen:
            seen.add(k); out.append(s)
    return out


def validate(sub, sample_ids, max_rank=25):
    assert len(sub) == len(sample_ids) and sub.molecule_id.is_unique and sub.smiles.notna().all()
    assert set(sub.molecule_id) == set(sample_ids)
    assert sub.smiles.map(lambda s: len([x for x in s.split(';') if x])).min() >= 1
    assert sub.smiles.map(lambda s: len(s.split(';'))).max() <= max_rank
