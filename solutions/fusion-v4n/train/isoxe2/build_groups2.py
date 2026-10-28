"""isoxe v2 data: natural-product-only same-formula groups (positive + pipeline-like isomer negatives), then graph-aware features.

v2 (2026-10-28, reviews/isoxe2-2026-10-28.md) changes against v1 (train/isoxe/build_groups.py):
  positives   train structures whose pool row is a natural product (pool_np > 0: COCONUT / LOTUS / NPAtlas), any library
  excluded    (positives and negatives) the keys in --exclude (one plain InChIKey14 per line: evaluation answers, their scaffold
              mates, ...) and the V scaffold groups
  V           whole Murcko scaffold groups (<= --v_group_max train structures, acyclic = own group) until --n_val NP positives
  negatives   like the pipeline's NP lists (about half pool / COCONUT isomers, half PubChem-window isomers, both chosen by
              similarity to the answer): up to K_PC PubChem rows of the w10 / w10_hold hard tables (most similar first, same
              formula), K_SIM most ECFP4-similar same-formula pool isomers + K_RAND random other pool isomers, then more similar
              pool isomers up to K_MAX. No list rank or pipeline score is used anywhere.
  features    --peaks most intense peaks (40), broken-bond indices per peak and the candidate's atom / bond graph (isoxe_feat
              bonds=True) for the graph channel of model2.py
Stages:
    python train/isoxe2/build_groups2.py select --exclude <keys.txt> [--n_val 1200]
    python train/isoxe2/build_groups2.py feat --split train|val [--workers 4]
Out: $ISOXE2_W (default D:/casmi-data/isoxe2)/groups_<split>.pkl, feat_<split>/chunk_*.npz (resumable)
"""
import argparse, os, pickle, sys, time

import numpy as np

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
HERE = os.path.dirname(os.path.abspath(__file__)); TRAIN = os.path.dirname(HERE); SOL = os.path.dirname(TRAIN)
sys.path.insert(0, os.path.join(SOL, 'clean')); sys.path.insert(0, os.path.join(TRAIN, 'isoxe'))
import build_groups as BG  # noqa: E402  (v1: Index, pick_spectra, below_normal, formula_of, ik14_of)

DATA = BG.DATA
OUT = os.environ.get('ISOXE2_W', 'D:/casmi-data/isoxe2').replace('\\', '/')
os.environ.setdefault('NUMBA_CACHE_DIR', OUT + '/numba_cache')
POOL_NP = DATA + '/input/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior/pool_np.npy'
SCAF = os.environ.get('ISOXE2_SCAF', 'C:/Users/hemre/casmi-work/clean/scaffolds.parquet')
K_PC, K_SIM, K_RAND, K_MAX = 7, 6, 2, 15
PEAKS = 40


def ecfp(smiles):
    from rdkit import Chem, DataStructs, RDLogger  # noqa: F401
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return None
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(m)


def tanimoto(fp, fps):
    from rdkit import DataStructs
    ok = [i for i, f in enumerate(fps) if f is not None]
    out = np.full(len(fps), -1.0)
    if fp is not None and ok:
        out[ok] = DataStructs.BulkTanimotoSimilarity(fp, [fps[i] for i in ok])
    return out


def order_pool(pos_smiles, iso_rows, smiles_of, rng, k_sim=K_SIM, k_rand=K_RAND, cap=40):
    """Pool isomer rows in use order: k_sim most similar, k_rand random others, then the rest by similarity (at most cap)."""
    iso_rows = list(iso_rows)
    if not iso_rows:
        return []
    sim = tanimoto(ecfp(pos_smiles), [ecfp(smiles_of(r)) for r in iso_rows])
    o = [iso_rows[i] for i in np.argsort(-sim, kind='stable')]
    head, rest = o[:k_sim], o[k_sim:]
    rnd = [rest[i] for i in sorted(rng.permutation(len(rest))[:k_rand])] if rest else []
    rs = set(rnd); tail = [r for r in rest if r not in rs]
    return (head + rnd + tail)[:cap]


# ------------------------------------------------------------------------------------------------ select ------------
_S = {}


def _sel_init(excl_pid):
    BG.below_normal()
    _S['ix'] = BG.Index(); _S['excl_pid'] = excl_pid


def _sel_task(task):
    """task: list of (sid, pid, smiles, seed) -> list of (sid, pool_rows ordered, pc_rows)."""
    ix = _S['ix']; out = []
    for sid, pid, smi, seed in task:
        rng = np.random.default_rng(seed)
        f = int(ix.pool_form[pid]); iso = np.asarray(ix.form_rows[ix.form_ptr[f]:ix.form_ptr[f + 1]])
        iso = [int(x) for x in iso if x != pid and int(x) not in _S['excl_pid']]
        if len(iso) > 400:                                   # very common formulas: similarity among a random 400
            iso = [iso[i] for i in sorted(rng.permutation(len(iso))[:400])]
        pool = order_pool(smi, iso, ix.pool_smiles_of, rng)
        pc = []
        for which in ('train', 'hold'):
            m, hard, rows = ix.pcneg(which)
            s = m.get(int(pid))
            if s is not None:
                pc = [int(rows[j]) for j in np.asarray(hard[s]) if j >= 0]
                break
        out.append((sid, pool, pc))
    return out


def select(a):
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    ix = BG.Index(); t0 = time.time()
    st = pd.read_parquet(BG.WORK + '/train_structs.parquet', columns=['inchikey14', 'smiles', 'formula', 'sid'])
    assert (st.sid.values == np.arange(len(st))).all()
    sc = pd.read_parquet(SCAF, columns=['sid', 'scaffold']).set_index('sid').scaffold.reindex(np.arange(len(st))).fillna('').values
    scaf = np.array([s if s else 'ik14:' + k for s, k in zip(sc, st.inchikey14.values)], dtype=object)
    npf = np.load(POOL_NP)
    sid_pid = np.asarray(ix.sid_pid)
    excl = {l.strip() for l in open(a.exclude) if l.strip()}
    ik = st.inchikey14.values
    is_np = np.array([p >= 0 and npf[p] > 0 for p in sid_pid])
    has_sp = np.array([len(ix.spectra_of(s)) > 0 for s in range(len(st))]) if not a.limit else np.ones(len(st), bool)
    cand = np.flatnonzero(is_np & has_sp & ~np.array([k in excl for k in ik]))
    print('NP structures', int(is_np.sum()), 'with spectra', int((is_np & has_sp).sum()), 'not excluded', len(cand), f'{time.time() - t0:.0f}s', flush=True)
    # V: whole scaffold groups
    size = pd.Series(scaf).value_counts()
    rng = np.random.default_rng(0)
    groups_np = pd.Series(cand).groupby(scaf[cand]).apply(list)
    order = [g for g in rng.permutation(groups_np.index.values) if size[g] <= a.v_group_max]
    val_scaf, nv = set(), 0
    for g in order:
        if nv >= a.n_val:
            break
        val_scaf.add(g); nv += len(groups_np[g])
    in_v = np.array([s in val_scaf for s in scaf])  # set lookups (np.isin on object arrays is quadratic)
    v_keys = set(ik[in_v])
    excl_all = excl | v_keys
    pm = pd.read_parquet(DATA + '/input/datasets/ahmedberatozer/casmi26-v2-pool/pool_meta.parquet', columns=['key'])
    pool_key = pm.key.values
    excl_pid = {i for i, k in enumerate(pool_key) if k in excl_all}
    val = [int(s) for s in cand if in_v[s]]; tr = [int(s) for s in cand if not in_v[s]]
    if a.limit:
        tr = tr[:a.limit]; val = val[:max(1, a.limit // 5)]
    print('V scaffolds', len(val_scaf), 'V positives', len(val), 'train positives', len(tr), 'excluded pool rows', len(excl_pid), flush=True)
    tasks = []
    for split, sids in (('val', val), ('train', tr)):
        for i in range(0, len(sids), 200):
            tasks.append((split, [(s, int(sid_pid[s]), st.smiles.values[s], 1000003 * s + 7) for s in sids[i:i + 200]]))
    res = {}
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(a.workers, initializer=_sel_init, initargs=(excl_pid,)) as P:
        for j, r in enumerate(P.imap(_sel_task, [t[1] for t in tasks])):
            for sid, pool, pc in r:
                res[sid] = (pool, pc)
            if (j + 1) % 20 == 0:
                print('select', j + 1, len(tasks), f'{time.time() - t0:.0f}s', flush=True)
    G = {}
    for split, sids, kspec, seed in (('val', val, a.max_spec_eval, 1), ('train', tr, a.max_spec, 3)):
        r2 = np.random.default_rng(seed); out = []; drop = 0
        for s in sids:
            pool, pc = res[s]
            if len(pool) + len(pc) < 2:
                drop += 1; continue
            out.append(dict(split=split, sid=s, pid=int(sid_pid[s]), ik14=ik[s], smiles=st.smiles.values[s], formula=st.formula.values[s],
                            scaffold=scaf[s], spec=BG.pick_spectra(ix, ix.spectra_of(s), r2, kspec), pool_rows=pool, pc_rows=pc, e180=False))
        for i, g in enumerate(out):
            g['gid'] = i
        G[split] = out
        print(split, len(out), 'dropped (< 2 negatives)', drop, flush=True)
    for k, v in G.items():
        pickle.dump(v, open(f'{OUT}/groups_{k}.pkl', 'wb'), protocol=4)
    open(f'{OUT}/excluded_keys.txt', 'w').write('\n'.join(sorted(excl_all)) + '\n')
    print('SELECT_DONE', {k: len(v) for k, v in G.items()}, f'{time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------------------------------------ featurisation -----
_W = {}


def _init():
    BG.below_normal()
    _W['ix'] = BG.Index()
    import isoxe_feat as X
    X.warmup(); _W['X'] = X
    p = f'{OUT}/excluded_keys.txt'
    _W['excl'] = {l.strip() for l in open(p) if l.strip()} if os.path.exists(p) else set()


def group_candidates(ix, g, excl=frozenset()):
    """[positive] + <= K_MAX negatives with distinct InChIKey14: <= K_PC PubChem rows (same formula), then pool rows in order."""
    seen = {g['ik14']}; pc, pool = [], []
    for r in g['pc_rows']:
        if len(pc) >= K_PC:
            break
        s = ix.tier_smiles(r)
        if BG.formula_of(s) != g['formula']:
            continue
        k = BG.ik14_of(s)
        if k is None or k in seen or k in excl:
            continue
        seen.add(k); pc.append(s)
    for p in g['pool_rows']:
        if len(pool) >= K_MAX - len(pc):
            break
        s = ix.pool_smiles_of(p)
        k = BG.ik14_of(s)
        if k is None or k in seen or k in excl:
            continue
        seen.add(k); pool.append(s)
    return [g['smiles']] + pc + pool, [0] + [1] * len(pc) + [2] * len(pool)


def featurize_groups2(groups, cand_fn, spec_fn, require_pos=True, peaks=PEAKS):
    """Pack groups with graph data: cand_fn(g) -> (smiles, source, label); spec_fn(g) -> prepared spectra (peaks = `peaks`)."""
    X = _W['X'] if 'X' in _W else __import__('isoxe_feat')
    cache = {}
    keys = ('g_ncand', 'g_nspec', 'g_gid', 'cand_g', 'cand_src', 'cand_lab', 'cand_smiles', 'cand_na', 'cand_nb', 'spec_f', 'spec_mask',
            'spec_meta', 'spec_ce', 'ann_i', 'ann_f', 'ann_b', 'atom_f', 'bond_f', 'bond_e')
    out = {k: [] for k in keys}
    for g in groups:
        sps = [s for s in spec_fn(g) if s is not None]
        if not sps:
            continue
        smis, src, lab = cand_fn(g)
        feats = []
        for s, so, la in zip(smis, src, lab):
            r = X.featurize_candidate(s, sps, cache, peaks=peaks, bonds=True)
            if r is not None:
                feats.append((s, so, la, r))
        if len(feats) < (2 if require_pos else 1) or (require_pos and not any(f[2] for f in feats)):
            continue
        f, mask, meta, ce = X.spectra_block(sps, peaks=peaks)
        out['g_ncand'].append(len(feats)); out['g_nspec'].append(len(sps)); out['g_gid'].append(g['gid'])
        out['spec_f'].append(f.astype(np.float16)); out['spec_mask'].append(mask); out['spec_meta'].append(meta); out['spec_ce'].append(ce)
        for s, so, la, (cg, ai, af, ab, gr) in feats:
            out['cand_g'].append(cg); out['cand_src'].append(so); out['cand_lab'].append(la); out['cand_smiles'].append(s)
            out['ann_i'].append(ai); out['ann_f'].append(af.astype(np.float16)); out['ann_b'].append(ab)
            out['cand_na'].append(gr['n']); out['cand_nb'].append(len(gr['ea']))
            out['atom_f'].append(gr['afeat']); out['bond_f'].append(gr['bfeat'])
            out['bond_e'].append(np.stack([gr['ea'], gr['eb']], 1).astype(np.int16).reshape(-1, 2))
        if len(cache) > 20000:
            cache.clear()
    if not out['g_ncand']:
        return None
    C = np.concatenate
    return dict(g_ncand=np.array(out['g_ncand'], np.int32), g_nspec=np.array(out['g_nspec'], np.int32), g_gid=np.array(out['g_gid'], np.int64),
                cand_g=np.stack(out['cand_g']), cand_src=np.array(out['cand_src'], np.int8), cand_lab=np.array(out['cand_lab'], np.int8),
                cand_smiles=np.array(out['cand_smiles'], dtype=object), cand_na=np.array(out['cand_na'], np.int32),
                cand_nb=np.array(out['cand_nb'], np.int32), spec_f=C(out['spec_f']), spec_mask=C(out['spec_mask']), spec_meta=C(out['spec_meta']),
                spec_ce=C(out['spec_ce']), ann_i=C(out['ann_i']), ann_f=C(out['ann_f']), ann_b=C(out['ann_b']),
                atom_f=C(out['atom_f']), bond_f=C(out['bond_f']), bond_e=C(out['bond_e']))


def save_chunk(path, D):
    np.savez(path + '.tmp.npz', **{k: v for k, v in D.items() if k != 'cand_smiles'}, cand_smiles=D['cand_smiles'].astype(str))
    os.replace(path + '.tmp.npz', path)


def _feat_chunk(task):
    path, groups = task
    if os.path.exists(path):
        return path, 0.0, -1
    t0 = time.time(); ix = _W['ix']

    def cands(g):
        smis, src = group_candidates(ix, g, _W['excl'] - {g['ik14']})
        return smis, src, [1] + [0] * (len(smis) - 1)
    D = featurize_groups2(groups, cands, lambda g: [prepared(ix, r) for r in g['spec']])
    if D is not None:
        save_chunk(path, D)
    return path, time.time() - t0, 0 if D is None else len(D['g_ncand'])


def prepared(ix, r, peaks=PEAKS):
    X = _W['X']
    mz, it = ix.peaks(r)
    ce = float(ix.spec_ce[r])
    return X.prep_spectrum(mz, it, float(ix.spec_prec[r]), X.ADDUCTS[int(ix.spec_add[r])], 1 if int(ix.spec_mode[r]) > 0 else -1,
                           [ce] if np.isfinite(ce) else None, int(ix.spec_instr[r]), peaks=peaks)


def feat(a):
    groups = pickle.load(open(f'{OUT}/groups_{a.split}.pkl', 'rb'))
    if a.limit:
        groups = groups[:a.limit]
    d = f'{OUT}/feat_{a.split}'; os.makedirs(d, exist_ok=True)
    tasks = [(f'{d}/chunk_{i // a.chunk:05d}.npz', groups[i:i + a.chunk]) for i in range(0, len(groups), a.chunk)]
    tasks = [t for t in tasks if not os.path.exists(t[0])]
    print(a.split, 'chunks to do', len(tasks), flush=True); t0 = time.time()
    BG.below_normal()
    if a.workers <= 1:
        _init()
        for t in tasks:
            print(*_feat_chunk(t), flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(a.workers, initializer=_init) as pool:
            for i, r in enumerate(pool.imap_unordered(_feat_chunk, tasks)):
                print(i + 1, len(tasks), os.path.basename(r[0]), f'{r[1]:.0f}s', r[2], f'total {time.time() - t0:.0f}s', flush=True)
    print('FEAT_DONE', a.split, f'{time.time() - t0:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['select', 'feat'])
    ap.add_argument('--exclude', default=''); ap.add_argument('--n_val', type=int, default=1200); ap.add_argument('--v_group_max', type=int, default=50)
    ap.add_argument('--max_spec', type=int, default=6); ap.add_argument('--max_spec_eval', type=int, default=16)
    ap.add_argument('--split', default='train'); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=200); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    BG.below_normal()
    select(a) if a.stage == 'select' else feat(a)


if __name__ == '__main__':
    main()
