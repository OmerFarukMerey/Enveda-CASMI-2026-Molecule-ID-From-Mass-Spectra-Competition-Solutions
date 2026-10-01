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
                POP_LIB_OFF=False, FRAG_LIB_OFF=False, # no popularity / frag term for library hits
                GATE_TAU=None,                         # lib_max threshold of the *_LIB gates; None = LIB_TAU
                FUSE_MIN_LIB=None)                     # fuse engine-2 only when lib_max >= this (it needs library analogs)


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


def fuse_rrf(v_smis, e_smis, e_keys, score_key, alpha, krr, n):
    """sum 1/(krr + r) over the v4 list plus alpha/(krr + r) over the engine list, keyed by the metric key."""
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = score_key(s) or s
        sc[k] = sc.get(k, 0.0) + 1.0 / (krr + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + alpha / (krr + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]


def build_submission(base, pc, eng, ice_scores, gl_scores, cfg, mol_order, sample_ids,
                     ice_fuse=None, gl_fuse=None, score_key=None, formula=None, log=print, pool_pop=None,
                     frag_scores=None):
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
            if covered(ice):
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
    lib_of_m = {m: v[2] for m, v in base.items()}

    # ---- stage A: ICE (+GL) re-rank of the base lists and of the PubChem lists of gated molecules
    for m in list(base):
        v = base[m]
        smis, keys, lib_max, scs, forms = list(v[0][:TOPN]), list(v[1][:TOPN]), v[2], list(v[3][:TOPN]), list(v[4][:TOPN])
        pids = list(v[5][:TOPN]) if len(v) > 5 else None
        ICE_LAM, GL_LAM = lam_pair(lib_max)       # closures (gl_rerank, frag_rerank) read these per molecule
        pop_on = c['POP_MU'] > 0 and not (c['POP_LIB_OFF'] and lib_max >= GATE_TAU)
        if pop_on and pool_pop is not None and pids is not None and len(scs) > 1:
            try:
                smis, keys, scs, forms, pids = popularity_reorder(smis, keys, scs, forms, pids, pool_pop, c['POP_MU'])
                stats.setdefault('pop', {'molecules': 0})['molecules'] += 1
            except Exception as e:
                log('popularity prior failed', m, repr(e))
        ice = ice_scores.get(str(m), {}) if ice_scores else {}
        gl = gl_of(m)
        smis0, keys0, scs0, forms0 = smis, keys, scs, forms
        if ice_fuse is not None and ice and any(v is not None for v in ice.values()):
            try:
                order = ice_fuse.rerank(smis, keys, scs, forms, ice, lam=ICE_LAM, top_n=TOPN)
                order = gl_rerank(order, smis, keys, scs, forms, ice, gl, TOPN)
                s2 = [smis[i] for i in order]; k2 = [keys[i] for i in order]
                st = stats['ice']; st['molecules'] += 1
                st['changed_top25'] += int(s2[:25] != smis[:25]); st['changed_top1'] += int(bool(s2) and s2[0] != smis[0])
                smis, keys = s2, k2
            except Exception as e:
                log('rerank failed', m, repr(e))
        try:
            o = frag_rerank(m, smis0, keys0, scs0, forms0, ice, gl, TOPN)
            if o is not None:
                smis, keys = [smis0[i] for i in o], [keys0[i] for i in o]
        except Exception as e:
            log('frag rerank failed', m, repr(e))
        base[m] = (smis[:25], keys[:25], lib_max)
        extra[m] = smis[25:]
        p = pc.get(m) if c['ICE_PC'] else None
        if ice_fuse is not None and p and p.get('pc') and lib_max < LIB_TAU and ice:
            try:
                if not p.get('pc_form'):
                    p['pc_form'] = [formula(x) for x in p['pc']]
                o = ice_fuse.rerank(p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, lam=ICE_LAM, top_n=25)
                o = gl_rerank(o, p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, gl, 25, 'pc_')
                fz0 = p['pc_fz'][0]          # the gate keeps the ORIGINAL top f.z, so gate decisions do not move
                p['pc'] = [p['pc'][i] for i in o]; p['pc_keys'] = [p['pc_keys'][i] for i in o]
                p['pc_fz'] = [fz0] + [p['pc_fz'][i] for i in o][1:]
                stats['ice']['pc_changed'] = stats['ice'].get('pc_changed', 0) + int(o[:1] != [0])
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
            ms['aggressive' if aggressive else 'gentle'] += 1
            final = merge(smis, keys, p['pc'], p['pc_keys'], c['SLOTS_AGG'] if aggressive else c['SLOTS_GENTLE'])
        if not final:
            final = [c['FALLBACK']]
        rows[mid] = ';'.join(final[:25])
    v4 = [(mid, rows.get(mid, c['FALLBACK'])) for mid in sample_ids]
    lib_of = {m: v[2] for m, v in base.items()}

    def fill(mid, lst):
        """FILL_25: append engine-2, PubChem and deeper base candidates (metric-key dedup) until 25."""
        if not c['FILL_25'] or len(lst) >= 25:
            return lst
        seen = {score_key(x) or x for x in lst}; out = list(lst)
        pool = list((eng.get(str(mid)) or {}).get('smiles', [])) + list((pc.get(mid) or {}).get('pc', [])) + list(extra.get(mid, []))
        for x in pool:
            k = score_key(x) or x
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
            fsm, fk, fsc = fuse_rrf(vs, e['smiles'][:K], e['keys'][:K], score_key, c['FUSE_ALPHA'], c['FUSE_KRR'], c['FUSE_N'])
            ice = ice_scores.get(str(mid), {}) if isinstance(ice_scores, dict) else {}
            fsm0, fk0, fsc0 = fsm, fk, fsc
            if ice and ice_fuse is not None:
                try:
                    forms = [formula(x) for x in fsm]
                    o = ice_fuse.rerank(fsm, fk, fsc, forms, ice, lam=P_ICE, top_n=len(fsm))
                    g = gl_of(mid)
                    if g and gl_fuse is not None:
                        o = gl_fuse.rerank_multi(fsm, fk, fsc, forms, [ice, g], [P_ICE, P_GL], top_n=len(fsm))
                    stats['fuse']['ice_reordered'] += int(o[:25] != list(range(min(25, len(o)))))
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
            vs = fsm[:25]
        vs = fill(mid, vs[:25])
        out.append((mid, ';'.join(vs[:25]) if vs else c['FALLBACK']))
    return pd.DataFrame(out, columns=['molecule_id', 'smiles']), stats


def validate(sub, sample_ids, max_rank=25):
    assert len(sub) == len(sample_ids) and sub.molecule_id.is_unique and sub.smiles.notna().all()
    assert set(sub.molecule_id) == set(sample_ids)
    assert sub.smiles.map(lambda s: len([x for x in s.split(';') if x])).min() >= 1
    assert sub.smiles.map(lambda s: len(s.split(';'))).max() <= max_rank
