# ICEBERG + GLACIER scoring of same-formula candidate groups (base top-N ∪ engine-2 top-K ∪ gated PubChem lists).
# Scores only; the re-ranking itself happens in fusion_core. Any failure -> empty scores -> ranker order kept.
import gc
from rdkit import Chem as _Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF

def formula_of(s):
    try:
        return _CMF(_Chem.MolFromSmiles(s))
    except Exception:
        return s

ICE_SCORES, GL_SCORES, items, items_native, ice_fuse, gl_fuse = {}, {}, None, None, None, None
TOPN, LIB_TAU = CFG['TOPN'], CFG['LIB_TAU']
if CFG['USE_ICE']:
    try:
        ICE_PKG = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))
        sys.path.append(ICE_PKG)
        import fuse as ice_fuse
        ICE_RUN_PKG = ICE_PKG
        if CFG['ICE_FT']:                        # writable copy of the package with the fine-tuned intensity model
            import shutil
            ICE_RUN_PKG = os.path.join(WORK, 'ice_pkg_ft')
            if not os.path.exists(ICE_RUN_PKG):
                shutil.copytree(ICE_PKG, ICE_RUN_PKG)
            _ft = find('casmi26-ice-ft/**/inten.pt')
            _dst = os.path.join(ICE_RUN_PKG, 'ckpt', 'inten.pt')
            os.chmod(_dst, 0o644); shutil.copyfile(_ft, _dst)
            log('ICE_FT: intensity model', _ft, os.path.getsize(_dst), 'bytes')
        del V, E, bank; gc.collect(); torch.cuda.empty_cache()
        mids = sorted(BASE, key=lambda m: BASE[m][2])        # molecules without a library match first
        cands = {m: ice_fuse.ice_candidates(BASE[m][0], BASE[m][4], TOPN) for m in mids}
        if ENG and CFG['ICE_UNION']:
            K, n_add = CFG['ICE_UNION_K'], 0
            for m in mids:
                e = ENG.get(str(m))
                if not e or not e.get('smiles'):
                    continue
                bs, bk, bf = list(BASE[m][0][:TOPN]), set(BASE[m][1][:TOPN]), list(BASE[m][4][:TOPN])
                es = [s for s, k in zip(e['smiles'][:K], e['keys'][:K]) if k not in bk]
                extra = ice_fuse.ice_candidates(bs + es, bf + [formula_of(s) for s in es], len(bs) + len(es))
                before = len(cands.get(m, []))
                cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))
                n_add += len(cands[m]) - before
            log('ICE_UNION: engine-2 candidates added:', n_add)
        if CFG['ICE_PC']:
            for m in mids:
                p = PC.get(m) if isinstance(PC, dict) else None
                if p and p.get('pc') and BASE[m][2] < LIB_TAU:
                    p['pc_form'] = [formula_of(x) for x in p['pc']]
                    extra = ice_fuse.ice_candidates(p['pc'], p['pc_form'], 25)
                    cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))
        for m in mids:                           # VAL_EXTRAS: score the popularity-channel proposals as well
            p = PC_ALT.get(m) if isinstance(PC_ALT, dict) else None
            if p and p.get('pc') and BASE[m][2] < LIB_TAU:
                extra = ice_fuse.ice_candidates(p['pc'], [formula_of(x) for x in p['pc']], 25)
                cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))
        cands = {m: c for m, c in cands.items() if c}
        items = items_native = ice_fuse.build_ice_input(te, cands)
        if CFG['ICE_SURROGATE']:                 # [M+NH4]+ / water-loss spectra scored against the [M+H]+ prediction
            import surrogate
            items, _n = surrogate.extend_items(items_native, ice_fuse.build_ice_input(surrogate.rewrite(te), cands),
                                               CFG['ICE_SURROGATE'])
            log('ICE_SURROGATE', CFG['ICE_SURROGATE'], ': molecules with surrogate spectra', _n)
        log('ICE input molecules', len(items), 'candidates', sum(len(it['cands']) for it in items))
        ICE_SCORES = ice_fuse.run_ice(ICE_RUN_PKG, items, workdir=os.path.join(WORK, 'ice_work'), device='cuda',
                                      budget_s=ICE_BUDGET, site=os.path.join(WORK, 'ice_site'))
        meta_f = os.path.join(WORK, 'ice_work', 'ice_out.json.meta.json')
        if os.path.exists(meta_f):
            print('ICE meta', open(meta_f).read()[:1500])
            if '"status": "budget"' in open(meta_f).read()[:200]:      # the wall-clock budget cut the work: which molecules are scored
                FAILS.append('iceberg stopped at ICE_BUDGET: the last molecules have no ICEBERG score')   # then depends on the machine's speed
    except Exception as e:
        ICE_SCORES = {}
        print('ICE FAILED -> ranker order kept:', repr(e)); FAILS.append('iceberg: ' + repr(e)[:200])
if CFG['USE_GL'] and items:
    try:
        GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))
        sys.path.append(GL_PKG)
        import gl_fuse
        gc.collect(); torch.cuda.empty_cache()
        gl_items = items if CFG['GL_SURROGATE'] else items_native    # surrogate spectra are untested with GLACIER
        if CFG['GL_MH_ONLY']:                # GLACIER on [M+H]+ spectra only (wangpenghua v29: +0.006 on the LB)
            gl_items = [dict(it, spectra=[s for s in it['spectra'] if s.get('adduct') == '[M+H]+']) for it in items]
            gl_items = [it for it in gl_items if it['spectra']]
            log('GLACIER [M+H]+ only: molecules', len(gl_items), 'of', len(items))
        GL_SCORES = gl_fuse.run_gl(GL_PKG, gl_items, workdir=os.path.join(WORK, 'gl_work'), device='cuda',
                                   budget_s=CFG['GL_BUDGET'], site=os.path.join(WORK, 'ice_site'),
                                   wheels=os.path.join(ICE_PKG, 'wheels'))
        meta_f = os.path.join(WORK, 'gl_work', 'gl_out.json.meta.json')
        print('GL meta', open(meta_f).read()[:1500] if os.path.exists(meta_f) else 'missing')
        if os.path.exists(meta_f) and '"status": "budget"' in open(meta_f).read()[:200]:
            FAILS.append('glacier stopped at GL_BUDGET: the last molecules have no GLACIER score')
    except Exception as e:
        GL_SCORES = {}
        print('GL FAILED -> ICE only:', repr(e)); FAILS.append('glacier: ' + repr(e)[:200])
FRAG = {}                                   # MetFrag-parsimony term (frag_rescore.py), CPU, ~1 ms per candidate
if CFG['FRAG_LAM'] > 0:
    try:
        import frag_rescore
        _t0 = time.time()
        for mid, _sub in te.groupby('molecule_id', sort=False):
            if mid not in BASE:
                continue
            _spec = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                          adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1)
                     for r in _sub.itertuples()][:16]
            _c = list(dict.fromkeys(list(BASE[mid][0]) + list((ENG.get(str(mid)) or {}).get('smiles', [])) +
                                    list((PC.get(mid) or {}).get('pc', []))))
            FRAG[mid] = dict(zip(_c, frag_rescore.score_molecule(_spec, _c)))
        log('frag scores', len(FRAG), 'molecules', f'{time.time()-_t0:.0f}s')
    except Exception as e:
        FRAG = {}
        print('FRAG FAILED -> off:', repr(e)); FAILS.append('frag: ' + repr(e)[:200])
GL_MH = {}                                    # VAL_EXTRAS: GLACIER on [M+H]+ spectra only, for local replay
if CFG['VALIDATION'] and CFG['VAL_EXTRAS'] and CFG['USE_GL'] and items and not CFG['GL_MH_ONLY'] and not IS_RERUN:
    try:
        _mh = [dict(it, spectra=[s for s in it['spectra'] if s.get('adduct') == '[M+H]+']) for it in items_native]
        _mh = [it for it in _mh if it['spectra']]
        gc.collect(); torch.cuda.empty_cache()
        GL_MH = gl_fuse.run_gl(GL_PKG, _mh, workdir=os.path.join(WORK, 'gl_work_mh'), device='cuda',
                               budget_s=CFG['GL_BUDGET'], site=os.path.join(WORK, 'ice_site'),
                               wheels=os.path.join(ICE_PKG, 'wheels'))
    except Exception as e:
        GL_MH = {}
        print('GL [M+H]+ EXTRA FAILED:', repr(e))
    log('GLACIER [M+H]+ only (extras): molecules', len(GL_MH))
    dump('gl_mh', GL_MH)
log('ICE molecules', len(ICE_SCORES), '| GL molecules', len(GL_SCORES), '| FRAG molecules', len(FRAG))
if CFG['USE_ICE'] and not ICE_SCORES:
    FAILS.append('iceberg returned no scores')
if CFG['USE_GL'] and not GL_SCORES:
    FAILS.append('glacier returned no scores')
dump('ice', ICE_SCORES); dump('gl', GL_SCORES); dump('frag', FRAG); dump('pc', PC)   # pc again: now carries pc_form
