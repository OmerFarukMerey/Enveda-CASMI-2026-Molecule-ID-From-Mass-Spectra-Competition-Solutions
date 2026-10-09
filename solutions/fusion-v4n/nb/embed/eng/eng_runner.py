"""Our engine (two-ranker, BIO + AFIX) -> {molecule_id: {smiles: [...], keys: [...]}} (blend order, top 40)."""
import os, sys, json, time
import numpy as np


def select_fp_models(paths, extra=()):
    """The reference's two models (casmi26-fp-models-v2) plus, with ENG_EXTRA_FPNET, the extra checkpoints. pv_fp.load_fp_models
    sorts each file into the per-spectrum or the merged group by its name, and logits_batch averages inside a group."""
    ref = sorted(p for p in paths if 'casmi26-fp-models-v2' in p)
    return ref + [p for p in sorted(extra) if p not in ref]


def reps_k(E, k):
    """casmi_engine._reps with the first k spectra of every structure instead of one (same order: timsTOF first, then most raw peaks,
    then the row number); k = 1 selects the rows of the original. analog_search / top_analogs already take the maximum per structure,
    so an analog's similarity becomes the best over its k representatives (ENG_REPS, reviews/keyvote-2026-10-09.md)."""
    import pandas as pd

    def _reps(L, rows):
        npk = np.diff(L["off"])[rows]
        df = pd.DataFrame({"row": rows, "p": L["row_p"][rows], "npk": npk, "tims": L["tims"][rows]})
        df = df.sort_values(["p", "tims", "npk", "row"], ascending=[True, False, False, True])
        df = df[df.groupby("p", sort=False).cumcount() < int(k)]
        rep = df.row.values; rep_nm = L["nm"][rep]
        o = np.argsort(rep_nm, kind="mergesort"); rep = rep[o]
        C = E.CFG
        roff, rmz, rit = E.pv.clean_store(rep, L["off"], L["mz"], L["it"], C.INT_FLOOR, C.MAX_PEAKS, C.INT_POWER, C.ENT_WEIGHT)
        return dict(rep=rep, rep_p=L["row_p"][rep], rep_nm=L["nm"][rep], roff=roff, rmz=rmz, rit=rit)
    return _reps


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    test, train, sample, out = sys.argv[1:5]
    import casmi_engine as E
    # ours: knobs come from the notebook CFG via CASMI_ENG_CFG (defaults = reference v4n)
    C = dict(W_A=(0.35, 0.55), SEEDS=(0, 1), N_ANALOG=200, W_PV=0.88, ORDER_K=80, TOPK=40, EXTRA_FP=(), REPS=1)
    C.update(json.loads(os.environ.get('CASMI_ENG_CFG', '{}')))
    _orig_find = E.find
    def _find(n):                                  # v4g ships its own fp_bits.npy (10,226 bits): take prvsiyan's 6,930
        if n == 'fp_bits.npy':
            return os.path.join(os.path.dirname(_orig_find('coco_fp.npy')), 'fp_bits.npy')
        return _orig_find(n)
    E.find = _find
    E.RANK.W_A = tuple(C['W_A']); E.RANK.SEEDS = tuple(C['SEEDS']); E.CFG.N_ANALOG = int(C['N_ANALOG'])
    if int(C['REPS']) > 1:                         # ours: more representative spectra per library structure in the analog search
        E._reps = reps_k(E, int(C['REPS'])); print('engine representatives per structure', int(C['REPS']), flush=True)
    import glob
    fp_models = select_fp_models(glob.glob('/kaggle/input/**/fp_*.pt', recursive=True), C['EXTRA_FP'])
    print('engine fp models', fp_models, flush=True)
    T0 = time.time()
    subs, recs, S = E.main(test, train, sample, E.find('sim_rank_rows_nofp.npz'), E.find('rank_train.npz'), fp_models,
                           workers=os.cpu_count(), w_pv=C['W_PV'])
    bl = E.blend_scores(S['pv'], S['ours'], C['W_PV'])
    res = {}
    for r in recs:
        mid = r['mid']
        if mid not in bl:
            continue
        order = np.argsort(-bl[mid], kind='mergesort')[:int(C['ORDER_K'])]
        smis, keys, seen = [], [], set()
        for c in r['cand'][order]:
            s = E.POOL['smiles'][c]; k = E.canon_key(s)
            if k is None or k in seen:
                continue
            seen.add(k); smis.append(s); keys.append(k)
            if len(smis) >= int(C['TOPK']):
                break
        res[str(mid)] = dict(smiles=smis, keys=keys)
    json.dump(res, open(out, 'w'))
    print('engine lists', len(res), f'{time.time()-T0:.0f}s', flush=True)
