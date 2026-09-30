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

ICE_SCORES, GL_SCORES, items, ice_fuse, gl_fuse = {}, {}, None, None, None
TOPN, LIB_TAU = CFG['TOPN'], CFG['LIB_TAU']
if CFG['USE_ICE']:
    try:
        ICE_PKG = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))
        sys.path.append(ICE_PKG)
        import fuse as ice_fuse
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
        cands = {m: c for m, c in cands.items() if c}
        items = ice_fuse.build_ice_input(te, cands)
        log('ICE input molecules', len(items), 'candidates', sum(len(it['cands']) for it in items))
        ICE_SCORES = ice_fuse.run_ice(ICE_PKG, items, workdir=os.path.join(WORK, 'ice_work'), device='cuda',
                                      budget_s=ICE_BUDGET, site=os.path.join(WORK, 'ice_site'))
        meta_f = os.path.join(WORK, 'ice_work', 'ice_out.json.meta.json')
        if os.path.exists(meta_f):
            print('ICE meta', open(meta_f).read()[:1500])
    except Exception as e:
        ICE_SCORES = {}
        print('ICE FAILED -> ranker order kept:', repr(e))
if CFG['USE_GL'] and items:
    try:
        GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))
        sys.path.append(GL_PKG)
        import gl_fuse
        gc.collect(); torch.cuda.empty_cache()
        GL_SCORES = gl_fuse.run_gl(GL_PKG, items, workdir=os.path.join(WORK, 'gl_work'), device='cuda',
                                   budget_s=CFG['GL_BUDGET'], site=os.path.join(WORK, 'ice_site'),
                                   wheels=os.path.join(ICE_PKG, 'wheels'))
        meta_f = os.path.join(WORK, 'gl_work', 'gl_out.json.meta.json')
        print('GL meta', open(meta_f).read()[:1500] if os.path.exists(meta_f) else 'missing')
    except Exception as e:
        GL_SCORES = {}
        print('GL FAILED -> ICE only:', repr(e))
log('ICE molecules', len(ICE_SCORES), '| GL molecules', len(GL_SCORES))
dump('ice', ICE_SCORES); dump('gl', GL_SCORES); dump('pc', PC)   # pc again: now carries pc_form
