"""genprobe: capture of engine 1's raw list in the stack1 notebook and the frozen generated-candidate probes (G-current / G-first / G-floor3).

Inserted verbatim as a new cell of stack1 (omerfarukmerey/casmi26-om-stack1 v1) after its commit switch (cell 11) by
scratch/genprobe/build_genprobe.py; three short cells call it later:
  after cell 18  gp_install(globals())       wraps V.run and rank_score (same arguments, same return values) to keep, per molecule, engine 1's
                                             deduplicated top-60 in RAW ranker order with pid, generated flag and parent similarity (C.gen_sim)
  after cell 20  gp_check_capture(globals()) rebuilds cell 20's popularity re-rank from the capture and requires it to equal BASE[m] exactly
                                             (lists, scores, formulas); a molecule that does not match has no frozen candidate (a fallback)
  last cell      gp_apply(globals())         frozen rule on stack1's final file (after E2R2), engine-2 membership, the variant, the log line
Engine 2's full candidate window (its 10 / 30 ppm pool window, the list it ranks) is written by its own runner (one appended block in cell 13,
active only when $GENPROBE_E2WIN is set) as raw SMILES; membership is decided here with stack1's metric_identity.

Frozen rule (scratch/genprobe/PREREG.md, fixed before any outcome): per active molecule, g = the generated row (pid < 0) with the best raw rank
among raw ranks 1-3 of engine 1's list; g is frozen iff lib_max < 0.5, g's parent similarity >= 0.6, and g's metric identity is not at final
rank 1-3 (absent counts as below). At most one per molecule; no fall-through to a second generated row.

GENPROBE['variant']: 'off' (nothing written: the submission is stack1's byte for byte), 'current' (frozen candidates at their current final
rank, 25 placeholders otherwise), 'first' (frozen candidate at rank 1, placeholders after it), 'floor3' (stack1's list with exactly the frozen
candidates moved to rank 3). Inactive rows (visible smoke placeholders) are never touched. With a variant on, a failed self-test, a missing
capture or a file-level error RAISES (commit and scoring run): a probe that silently fell back to stack1's file would be misread.
Log: GENPROBE_PATCH {...} (one line) and GENPROBE_MOL {...} per frozen molecule.
Pure functions are imported by solutions/fusion-v4n/tests/test_genprobe.py (the notebook-only lines run under __main__).
"""
import json
import os

import numpy as np

GENPROBE = dict(variant='off')
GP_VARIANTS = ('off', 'current', 'first', 'floor3')
GP_LIB_MAX, GP_SIM_MIN, GP_TOPK, GP_FLOOR, GP_LIMIT = 0.5, 0.6, 3, 3, 25
GP_E2WIN = '/tmp/genprobe_e2win.json'
GP_DUMP = 'genprobe_capture.json'
GP_E2_BUDGET_S = 900.0
# 25 distinct metric identities, every one below 100 Da (all pool / PubChem-tier candidates are >= 150 Da): never an answer
GP_PLACEHOLDERS = ['C', 'CC', 'CCC', 'CCCC', 'CC(C)C', 'CCCCC', 'C=C', 'C#C', 'CC=C', 'O', 'CO', 'N', 'CN', 'CCN', 'S', 'CS',
                   'C=O', 'CC=O', 'OC=O', 'CC(=O)O', 'CC#N', 'C1CC1', 'C1CCC1', 'CCl', 'c1ccccc1']


# ------------------------------------------------------------------------------------------------------------------------------
# metric identity (same definition as stack1 cell 28; the notebook cross-checks it against stack1's own function)
# ------------------------------------------------------------------------------------------------------------------------------
def gp_metric_identity(smiles):
    """(InChIKey14 after stereo removal + RDKit tautomer canonicalisation, heavy-atom composition) or None."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        counts = {}
        for a in mol.GetAtoms():
            if a.GetSymbol() != 'H':
                counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
        Chem.RemoveStereochemistry(mol)
        ik = Chem.MolToInchiKey(rdMolStandardize.TautomerEnumerator().Canonicalize(mol))[:14]
        return (ik, tuple(sorted(counts.items()))) if ik else None
    except (ValueError, RuntimeError, TypeError):
        return None


# ------------------------------------------------------------------------------------------------------------------------------
# capture: cell 20's dedup in raw order and its popularity re-rank, rebuilt
# ------------------------------------------------------------------------------------------------------------------------------
def gp_compact(pid, key, smiles, formula, gen_sim, score, topn):
    """Cell 20's loop: raw ranker order (stable), first occurrence of each C.key, at most topn rows. Rows in raw order."""
    score = np.asarray(score)
    order = np.argsort(-score, kind='stable')
    seen, rows = set(), []
    for i in order:
        k = key[i]
        if k in seen:
            continue
        seen.add(k)
        p = int(pid[i])
        rows.append(dict(smiles=smiles[i], key=k, formula=formula[i], raw=float(score[i]), pid=p, gen=p < 0, raw_rank=len(rows) + 1,
                         sim=(float(gen_sim[i]) if gen_sim is not None else None)))
        if len(rows) >= topn:
            break
    return rows


def gp_poolpop(rows, mu, pool_pop):
    """Cell 20's popularity re-rank of the raw list. Returns (smis, keys, scs, forms, order, pop) where order[j] = raw index at base position j
    and pop[i] = the re-rank score of raw row i (the raw score when the re-rank does not run)."""
    smis = [r['smiles'] for r in rows]; keys = [r['key'] for r in rows]; scs = [r['raw'] for r in rows]; forms = [r['formula'] for r in rows]
    order = list(range(len(rows))); pop = list(scs)
    if mu > 0 and len(scs) > 1:
        try:
            s_ = np.asarray(scs, np.float64); z_ = (s_ - s_.mean()) / (s_.std() + 1e-9)
            pid_ = np.asarray([r['pid'] for r in rows]); isg_ = pid_ < 0
            f_ = z_ + mu * np.where(isg_, 0.0, pool_pop[np.maximum(pid_, 0)])
            pool_i = np.where(~isg_)[0]; o_ = np.empty(len(pid_), np.int64)
            o_[isg_] = np.where(isg_)[0]; o_[~isg_] = pool_i[np.argsort(-f_[pool_i], kind='stable')]
            smis = [smis[i] for i in o_]; keys = [keys[i] for i in o_]; scs = [float(f_[i]) for i in o_]
            forms = [forms[i] + ('|gen' if isg_[i] else '') for i in o_]
            order = [int(i) for i in o_]; pop = [float(v) for v in f_]
        except Exception:  # cell 20 keeps the raw list when its re-rank fails
            smis = [r['smiles'] for r in rows]; keys = [r['key'] for r in rows]; scs = [r['raw'] for r in rows]
            forms = [r['formula'] for r in rows]; order = list(range(len(rows))); pop = list(scs)
    return smis, keys, scs, forms, order, pop


def gp_match(rows, lib_max, base_entry, mu, pool_pop):
    """Capture rows vs BASE[m] = (smis, keys, lib_max, scs, forms) -> capture dict, or raises AssertionError naming the first difference."""
    smis, keys, scs, forms, order, pop = gp_poolpop(rows, mu, pool_pop)
    b = base_entry
    assert len(b) == 5, 'BASE entry is not cell 20 shaped'
    assert list(b[0]) == smis, 'smiles differ'
    assert list(b[1]) == keys, 'keys differ'
    assert float(b[2]) == float(lib_max), 'lib_max differs'
    assert [float(x) for x in b[3]] == [float(x) for x in scs], 'scores differ'
    assert list(b[4]) == forms, 'formulas differ'
    pos = {i: j + 1 for j, i in enumerate(order)}
    out = [dict(r, pop=pop[i], pos=pos[i]) for i, r in enumerate(rows)]
    return dict(lib_max=float(lib_max), rows=out)


# ------------------------------------------------------------------------------------------------------------------------------
# the frozen rule and the three list transforms (pure)
# ------------------------------------------------------------------------------------------------------------------------------
def gp_final_rank(toks, k, ident):
    """1-based rank of the first token whose identity is k, None if absent."""
    for i, t in enumerate(toks, 1):
        if ident(t) == k:
            return i
    return None


def gp_select(cap, toks, ident):
    """(frozen candidate dict or None, reason). reason: no_capture, no_gen_top3, lib, sim, no_identity, in_top3, frozen."""
    if not cap:
        return None, 'no_capture'
    gens = [r for r in cap.get('rows') or [] if r.get('gen') and r.get('raw_rank', 99) <= GP_TOPK]
    if not gens:
        return None, 'no_gen_top3'
    g = min(gens, key=lambda r: r['raw_rank'])
    lm = cap.get('lib_max')
    if lm is None or not float(lm) < GP_LIB_MAX:
        return None, 'lib'
    if g.get('sim') is None or not float(g['sim']) >= GP_SIM_MIN:
        return None, 'sim'
    k = ident(g['smiles'])
    if k is None:
        return None, 'no_identity'
    r = gp_final_rank(toks, k, ident)
    if r is not None and r <= GP_FLOOR:
        return None, 'in_top3'
    return dict(g, ident=k, final_rank=r), 'frozen'


def gp_token(toks, sel):
    """The string written for the frozen candidate: stack1's own token when it is listed, else engine 1's SMILES."""
    r = sel.get('final_rank')
    return toks[r - 1] if r else sel['smiles']


def gp_current(toks, sel, ph=GP_PLACEHOLDERS, limit=GP_LIMIT):
    """Placeholders everywhere; a listed frozen candidate keeps its exact final rank."""
    out = list(ph[:limit])
    if sel and sel.get('final_rank'):
        r = sel['final_rank']
        out[r - 1] = toks[r - 1]
    return out


def gp_first(toks, sel, ph=GP_PLACEHOLDERS, limit=GP_LIMIT):
    """The frozen candidate alone at rank 1, placeholders after it; 25 placeholders without one."""
    if not sel:
        return list(ph[:limit])
    return [gp_token(toks, sel)] + list(ph[:limit - 1])


def gp_floor3(toks, sel, ident, limit=GP_LIMIT, floor=GP_FLOOR):
    """stack1's list with exactly the frozen candidate at rank `floor` (ranks above kept, the rest in order, cut to `limit`)."""
    toks = list(toks)
    if not sel:
        return toks
    k = sel['ident']; tok = gp_token(toks, sel)
    rest = [t for t in toks if ident(t) != k]
    return (rest[:floor - 1] + [tok] + rest[floor - 1:])[:limit]


def gp_transform(variant, toks, sel, ident, ph=GP_PLACEHOLDERS):
    if variant == 'current':
        return gp_current(toks, sel, ph)
    if variant == 'first':
        return gp_first(toks, sel, ph)
    if variant == 'floor3':
        return gp_floor3(toks, sel, ident)
    if variant == 'off':
        return list(toks)
    raise ValueError(f'unknown variant {variant!r}')


def gp_check_placeholders(ident, ph=GP_PLACEHOLDERS):
    """25 distinct parseable identities, every one below 100 Da (average and monoisotopic), no ';'."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    assert len(ph) == GP_LIMIT and len(set(ph)) == GP_LIMIT
    ks = []
    for s in ph:
        assert ';' not in s and s.strip() == s and s, s
        m = Chem.MolFromSmiles(s)
        assert m is not None, s
        assert Descriptors.MolWt(m) < 100.0 and Descriptors.ExactMolWt(m) < 100.0, s
        k = ident(s)
        assert k is not None, s
        ks.append(k)
    assert len(set(ks)) == GP_LIMIT, 'placeholder identities collide'
    assert len({k[0] if isinstance(k, tuple) else k for k in ks}) == GP_LIMIT, 'placeholder InChIKey14 collide'
    return ks


def gp_selftest(ident=None):
    """Synthetic rows for every branch of the rule and the transforms; raises on any mismatch."""
    ident = ident or gp_metric_identity
    gp_check_placeholders(ident)
    key = lambda s: s.lower() if isinstance(s, str) and s.strip() else None

    def row(s, rr, gen, sim=0.9):
        return dict(smiles=s, key=s, raw=1.0 / rr, pid=-1 if gen else rr, gen=gen, raw_rank=rr, sim=sim if gen else 0.0, formula='F')
    cap = dict(lib_max=0.3, rows=[row('A', 1, False), row('G', 2, True), row('H', 3, True), row('B', 4, False)])
    toks = ['A', 'B', 'C', 'D', 'G', 'E']
    sel, why = gp_select(cap, toks, key)
    assert why == 'frozen' and sel['smiles'] == 'G' and sel['final_rank'] == 5, (sel, why)
    assert gp_select(dict(cap, lib_max=0.5), toks, key)[1] == 'lib'
    assert gp_select(dict(cap, lib_max=float('nan')), toks, key)[1] == 'lib'
    c2 = dict(cap, rows=[row('A', 1, False), row('G', 2, True, 0.59), row('H', 3, True, 0.99)])
    assert gp_select(c2, toks, key)[1] == 'sim'                                   # select first, then test: no fall-through to H
    assert gp_select(cap, ['A', 'G', 'B'], key)[1] == 'in_top3'
    assert gp_select(cap, ['A', 'B', 'C', 'g'], key)[0]['final_rank'] == 4       # identity, not string
    s3, _ = gp_select(cap, ['A', 'B'], key); assert s3['final_rank'] is None    # absent
    assert gp_select(dict(cap, rows=[row('A', 1, False), row('B', 2, False), row('C', 3, False), row('G', 4, True)]), toks, key)[1] == 'no_gen_top3'
    assert gp_select(None, toks, key)[1] == 'no_capture'
    ph = [f'p{i}' for i in range(GP_LIMIT)]
    T = [f't{i}' for i in range(1, 26)]; T[6] = 'G'
    s, _ = gp_select(cap, T, key); assert s['final_rank'] == 7
    cur = gp_current(T, s, ph); assert len(cur) == 25 and cur[6] == 'G' and cur[:6] == ph[:6] and cur[7:] == ph[7:]
    assert gp_current(T, None, ph) == ph
    fi = gp_first(T, s, ph); assert fi == ['G'] + ph[:24]
    fl = gp_floor3(T, s, key); assert fl[:3] == ['t1', 't2', 'G'] and fl[3:] == [t for t in T if t != 'G'][2:] and len(fl) == 25
    s_abs, _ = gp_select(cap, [f't{i}' for i in range(1, 26)], key)
    fl2 = gp_floor3([f't{i}' for i in range(1, 26)], s_abs, key)
    assert fl2[:3] == ['t1', 't2', 'G'] and fl2[3:] == [f't{i}' for i in range(3, 25)] and len(fl2) == 25   # rank 25 dropped
    assert gp_first([f't{i}' for i in range(1, 26)], s_abs, ph)[0] == 'G'
    assert gp_current([f't{i}' for i in range(1, 26)], s_abs, ph) == ph
    assert gp_floor3(T, None, key) == T and gp_transform('off', T, s, key) == T
    assert gp_floor3(['A', 'G'], dict(s, final_rank=2), key) == ['A', 'G']           # short list: after the rows it has
    # capture rebuild: popularity re-rank keeps generated rows in their slots
    rows = gp_compact([5, -1, 7, 5, -1], ['a', 'g', 'b', 'a', 'h'], ['A', 'G', 'B', 'A2', 'H'], ['F1', 'F2', 'F1', 'F1', 'F2'],
                      [0.0, 0.7, 0.0, 0.0, 0.4], [3.0, 2.5, 1.0, 2.9, 0.5], 60)
    assert [r['smiles'] for r in rows] == ['A', 'G', 'B', 'H'] and [r['raw_rank'] for r in rows] == [1, 2, 3, 4]
    pp = np.zeros(10); pp[7] = 50.0
    smis, keys, scs, forms, order, pop = gp_poolpop(rows, 0.15, pp)
    assert smis == ['B', 'G', 'A', 'H'] and forms == ['F1', 'F2|gen', 'F1', 'F2|gen'] and order == [2, 1, 0, 3]
    cp = gp_match(rows, 0.2, (smis, keys, 0.2, scs, forms), 0.15, pp)
    assert [r['pos'] for r in cp['rows']] == [3, 2, 1, 4]
    try:
        gp_match(rows, 0.2, (smis[::-1], keys, 0.2, scs, forms), 0.15, pp)
        raise RuntimeError('mismatch not detected')
    except AssertionError:
        pass
    return 'ok'


# ------------------------------------------------------------------------------------------------------------------------------
# notebook drivers
# ------------------------------------------------------------------------------------------------------------------------------
def gp_log(tag, **kw):
    print(tag, json.dumps(kw, default=str), flush=True)


def gp_install(scope):
    """Wrap V.run and rank_score (cell 18's objects) so cell 20 runs unchanged while the raw list is kept per molecule (scope['mid'])."""
    raw = scope.setdefault('GP_RAW', {})
    st = scope.setdefault('GP_STAT', dict(install='not run', run_err=0, rank_err=0, sim_source={}))
    V = scope['V']; run0 = V.run; rank0 = scope['rank_score']; topn = int(scope['TOPN'])
    scope['GP_ORIG'] = (V, run0, rank0)

    def run(*a, **k):
        out = run0(*a, **k)
        try:
            for m_ in list(raw):
                raw[m_].pop('C', None)                    # an unranked previous molecule (empty candidate set) never holds its arrays
            C, X, F, names, info = out
            m = str(scope.get('mid'))
            raw[m] = dict(C=C, lib_max=float(info.get('lib_max', 0.0)), rows=None)
            scope['GP_LAST'] = m
        except Exception as e:
            st['run_err'] += 1; st['run_last'] = repr(e)[:200]
        return out

    def rank_score(Xc, names):
        s = rank0(Xc, names)
        try:
            rec = raw.get(scope.get('GP_LAST'))
            C = rec.pop('C', None) if rec else None
            if C is not None:
                nm = list(names)
                sim_c = getattr(C, 'gen_sim', None)
                sim_x = np.asarray(Xc)[:, nm.index('gen_parent_sim')] if 'gen_parent_sim' in nm else None
                sim = sim_c if sim_c is not None else sim_x
                src = 'C.gen_sim' if sim_c is not None else ('X.gen_parent_sim' if sim_x is not None else 'none')
                st['sim_source'][src] = st['sim_source'].get(src, 0) + 1
                if sim_c is not None and sim_x is not None:
                    d = float(np.max(np.abs(np.asarray(sim_c, np.float64) - np.asarray(sim_x, np.float64)))) if len(sim_x) else 0.0
                    st['sim_c_vs_x_maxdiff'] = max(st.get('sim_c_vs_x_maxdiff', 0.0), d)
                rec['rows'] = gp_compact(C.pid, C.key, C.smiles, C.formula, sim, np.asarray(s), topn)
        except Exception as e:
            st['rank_err'] += 1; st['rank_last'] = repr(e)[:200]
        return s

    V.run = run
    scope['rank_score'] = rank_score
    st['install'] = 'ok'
    gp_log('GENPROBE_CAPTURE', stage='install', variant=scope['GENPROBE']['variant'], topn=topn)


def gp_check_capture(scope):
    """After cell 20: restore the wrapped functions, rebuild BASE from the capture, keep only exact matches in GP_CAP."""
    st = scope.setdefault('GP_STAT', dict(install='not run'))
    orig = scope.get('GP_ORIG')
    if orig:
        V, run0, rank0 = orig
        try:
            if 'run' in vars(V):
                del V.run
        except Exception:
            pass
        scope['rank_score'] = rank0
    raw = scope.get('GP_RAW') or {}
    cap, fails = {}, {}
    base = scope['BASE']; mu = float(scope['POOLPOP_MU']); pp = scope['POOL_POP']
    for m, b in base.items():
        m = str(m); rec = raw.get(m)
        try:
            if rec is None or rec.get('rows') is None:
                assert not b[0], 'no capture for a non-empty list'
                cap[m] = dict(lib_max=float(b[2]), rows=[])
            else:
                cap[m] = gp_match(rec['rows'], rec['lib_max'], b, mu, pp)
        except Exception as e:
            fails[m] = repr(e)[:160]
    scope['GP_CAP'] = cap; scope['GP_CAPFAIL'] = fails
    scope.pop('GP_RAW', None)
    n_gen3 = sum(any(r['gen'] and r['raw_rank'] <= GP_TOPK for r in c['rows']) for c in cap.values())
    st.update(check='ok', base=len(base), captured=len(cap), mismatched=len(fails), gen_top3=n_gen3)
    gp_log('GENPROBE_CAPTURE', stage='check', **{k: v for k, v in st.items()}, fail_examples=dict(list(fails.items())[:3]))


def gp_apply(scope, workdir='.'):
    """Last cell: rule, engine-2 membership, the variant, checks, log. 'off' writes nothing."""
    import hashlib
    import io
    import time
    from collections import Counter
    from pathlib import Path
    import pandas as pd
    t0 = time.time()
    variant = str((scope.get('GENPROBE') or {}).get('variant', 'off'))
    on = variant != 'off'
    stat = dict(variant=variant, selftest='not run', status='not run', rows=0, active=0, fallback=0, frozen=0, changed=0, changed_top1=0)
    root = Path(workdir); sub = root / 'submission.csv'; tmp = root / 'submission_genprobe.tmp.csv'
    cur = sub.read_bytes() if sub.exists() else None
    problems, mols = [], []
    try:
        assert variant in GP_VARIANTS, f'unknown variant {variant!r}'
        ident = scope['metric_identity']; hac = scope['heavy_atom_composition']
        stat['selftest'] = gp_selftest(ident)
        probe = GP_PLACEHOLDERS + ['CC(=O)Oc1ccccc1C(=O)O', 'C[C@H](N)C(=O)O', 'Oc1ccccn1', 'O=c1cccc[nH]1']
        assert all(gp_metric_identity(s) == ident(s) for s in probe), 'module identity differs from stack1 metric_identity'
        cap_all = scope.get('GP_CAP')
        assert isinstance(cap_all, dict), 'capture missing (GP_CAP not built)'
        A = pd.read_csv(io.BytesIO(cur), dtype=str, keep_default_na=False)
        assert list(A.columns) == ['molecule_id', 'smiles']
        active = set(scope['te'].molecule_id.astype(str))
        assert not active or cap_all, 'capture empty while molecules are active'
        e2win = None
        try:
            if os.path.exists(GP_E2WIN):
                e2win = json.load(open(GP_E2WIN))
        except Exception as e:
            stat['e2win_error'] = repr(e)[:160]
        eng = scope.get('ENG') or {}
        reasons, rows, mols, dump = Counter(), [], [], {}
        e2c = Counter(); e2_deadline = t0 + GP_E2_BUDGET_S
        for mid, s in zip(A.molecule_id, A.smiles):
            old = s.split(';')
            if mid not in active:
                rows.append((mid, s)); continue
            stat['active'] += 1
            sel = None
            try:
                cap = cap_all.get(mid)
                if cap is None:
                    stat['fallback'] += 1; why = 'capture_mismatch' if mid in (scope.get('GP_CAPFAIL') or {}) else 'no_capture'
                else:
                    sel, why = gp_select(cap, old, ident)
            except Exception as e:
                stat['fallback'] += 1; why = 'error'; sel = None
                gp_log('GENPROBE_FALLBACK', molecule_id=mid, error=repr(e)[:200])
            reasons[why] += 1
            # capture record: generated rows of engine 1's raw top 3, final rank and engine-2 membership by metric identity
            rec = dict(lib_max=(cap_all.get(mid) or {}).get('lib_max'), reason=why, gen_top3=[])
            try:
                e2top = {ident(x) for x in ((eng.get(mid) or {}).get('smiles') or [])}
                wl = (e2win or {}).get(mid); wh = None
                for r in ((cap_all.get(mid) or {}).get('rows') or []):
                    if not (r['gen'] and r['raw_rank'] <= GP_TOPK):
                        continue
                    k = ident(r['smiles'])
                    in_win = None
                    if k is not None and wl is not None and time.time() < e2_deadline:
                        if wh is None:
                            wh = [hac(x) for x in wl]
                        in_win = any(h == k[1] and ident(x) == k for x, h in zip(wl, wh))
                    rec['gen_top3'].append(dict(smiles=r['smiles'], raw_rank=r['raw_rank'], pos=r.get('pos'), raw=r['raw'], pop=r.get('pop'),
                                                sim=r['sim'], final_rank=gp_final_rank(old, k, ident) if k else None,
                                                in_e2_top40=(k in e2top) if k else None, in_e2_window=in_win, e2_window_n=len(wl) if wl is not None else None))
                    e2c['gen_top3_rows'] += 1; e2c['in_e2_window'] += in_win is True; e2c['in_e2_top40'] += bool(k in e2top)
                    e2c['window_unknown'] += in_win is None
            except Exception as e:
                rec['e2_error'] = repr(e)[:160]; e2c['errors'] += 1
            dump[mid] = rec
            new = gp_transform(variant, old, sel, ident)
            if sel:
                stat['frozen'] += 1
                g3 = next((x for x in rec['gen_top3'] if x['smiles'] == sel['smiles']), {})
                mols.append(dict(molecule_id=mid, raw_rank=sel['raw_rank'], pos=sel.get('pos'), sim=round(sel['sim'], 4),
                                 lib_max=round(float(rec['lib_max']), 4), final_rank=sel['final_rank'] or 0,
                                 in_e2_window=g3.get('in_e2_window'), in_e2_top40=g3.get('in_e2_top40')))
            if on and new != old:
                ks = [ident(t) for t in new]
                assert 1 <= len(new) <= GP_LIMIT and all(t and t.strip() == t and ';' not in t for t in new), f'{mid}: invalid list'
                if variant in ('current', 'first'):
                    assert len(new) == GP_LIMIT and None not in ks and len(set(ks)) == GP_LIMIT, f'{mid}: identities'
                    kept = [t for t in new if t not in GP_PLACEHOLDERS]
                    assert len(kept) == (1 if (sel and (variant == 'first' or sel.get('final_rank'))) else 0), f'{mid}: kept rows'
                    if kept:
                        assert ident(kept[0]) == sel['ident'], f'{mid}: wrong kept row'
                        assert new.index(kept[0]) + 1 == (1 if variant == 'first' else sel['final_rank']), f'{mid}: kept rank'
                else:
                    assert sel and ks[GP_FLOOR - 1] == sel['ident'] and new[:GP_FLOOR - 1] == [t for t in old if ident(t) != sel['ident']][:GP_FLOOR - 1]
                    assert sum(k == sel['ident'] for k in ks) == 1, f'{mid}: twin'
                    assert [t for t in new if ident(t) != sel['ident']] == [t for t in old if ident(t) != sel['ident']][:len(new) - 1]
            if not on:
                new = old
            stat['changed'] += int(new != old); stat['changed_top1'] += int(new[:1] != old[:1])
            rows.append((mid, ';'.join(new)))
        stat.update(rows=len(rows), reasons=dict(reasons), e2=dict(e2c), e2win_loaded=e2win is not None,
                    frozen_final_ranks=sorted(m['final_rank'] for m in mols),
                    frozen_in_e2_window=sum(m['in_e2_window'] is True for m in mols), frozen_in_e2_top40=sum(bool(m['in_e2_top40']) for m in mols),
                    capture_mismatch=len(scope.get('GP_CAPFAIL') or {}), capture=dict((scope.get('GP_STAT') or {})),
                    base_sha256=hashlib.sha256(cur).hexdigest())
        try:
            (root / GP_DUMP).write_text(json.dumps(dict(variant=variant, molecules=dump, frozen=mols,
                                                        capture={m: c for m, c in cap_all.items() if m in active}), default=str))
        except Exception as e:
            stat['dump_error'] = repr(e)[:160]
        if on and stat['changed']:
            out = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
            out.to_csv(tmp, index=False)
            back = pd.read_csv(tmp, dtype=str, keep_default_na=False)
            assert list(back.columns) == ['molecule_id', 'smiles'] and list(back.molecule_id) == list(A.molecule_id)
            assert list(back.smiles) == list(out.smiles) and back.molecule_id.is_unique
            assert back.smiles.map(lambda v: 1 <= len(v.split(';')) <= GP_LIMIT).all() and (back.smiles.str.strip() != '').all()
            os.replace(tmp, sub)
        stat['status'] = 'applied' if on else 'off (nothing written)'
    except Exception as exc:
        problems.append(repr(exc)[:300])
        try:
            if cur is not None and sub.read_bytes() != cur:
                sub.write_bytes(cur)
            if tmp.exists():
                tmp.unlink()
        except Exception as exc2:
            problems.append('restore failed ' + repr(exc2)[:200])
        stat['status'] = 'FAILED ' + repr(exc)[:200]
    stat['submission_sha256'] = hashlib.sha256(sub.read_bytes()).hexdigest() if sub.exists() else None
    stat['secs'] = round(time.time() - t0, 1)
    for m in mols:
        gp_log('GENPROBE_MOL', **m)
    gp_log('GENPROBE_PATCH', **stat)
    if on and problems:
        raise RuntimeError(f'GENPROBE {variant}: {problems}')
    return stat


if __name__ == '__main__':      # notebook cell (inserted after the commit switch, before engine 2 runs): no CUDA, nothing but the switch
    assert GENPROBE['variant'] in GP_VARIANTS, GENPROBE
    if os.path.exists(GP_E2WIN):
        os.remove(GP_E2WIN)
    os.environ['GENPROBE_E2WIN'] = GP_E2WIN
    gp_log('GENPROBE_SWITCH', **GENPROBE, e2win=GP_E2WIN)
