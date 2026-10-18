"""isoxe data: same-formula candidate groups (positive + isomer negatives) with spectra, then cross-encoder features per group.

Stage `select` (labels only, no featurisation): training positives from the identity-disjoint split's training part (joint index
`tr_sid`), V / T from the `hold` fold, minus anything sharing a Murcko scaffold (acyclic: InChIKey14) with an evaluation molecule
(the 250-molecule Enveda panel, the 380 mixed-regime replay molecules, V, T). Negatives: PubChem window rows of the w10 tables
(most similar first; formula re-checked at featurisation) and same-formula pool rows (random, held-out rows excluded for training).
Spectra: test adducts only, timsTOF first, at most --max_spec (training) / --max_spec_eval (V, T).
Stage `feat`: featurises the groups in chunks with --workers low-priority processes (isoxe_feat), one npz per chunk (resumable).

    python train/isoxe/build_groups.py select [--n_tims 20000 --n_other 20000 --n_val 3000 --n_test 2000]
    python train/isoxe/build_groups.py feat --split train|val|test [--workers 4]
Out: $ISOXE_W (default C:/Users/hemre/casmi-work/isoxe)/groups_<split>.pkl, feat_<split>/chunk_*.npz
"""
import argparse, glob, os, pickle, sys, time

import numpy as np

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
HERE = os.path.dirname(os.path.abspath(__file__)); SOL = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(SOL, 'clean'))
DATA = os.environ.get('CASMI_DATA', 'D:/casmi-data').replace('\\', '/')
WORK = DATA + '/directions/run/work'; IDX = DATA + '/joint/index'; TIER = DATA + '/pubchem_study/tier'
PCNEG = dict(train=DATA + '/hardneg/w10', hold=DATA + '/hardneg/w10_hold')
OUT = os.environ.get('ISOXE_W', 'C:/Users/hemre/casmi-work/isoxe').replace('\\', '/')
os.environ.setdefault('NUMBA_CACHE_DIR', OUT + '/numba_cache')
PANEL_Q = DATA + '/panel2/regime/queries.parquet'; MIXED_Q = DATA + '/mixed_regime/queries.parquet'
N_TEST_ADDUCTS = 10
K_PC, K_POOL, K_MAX = 8, 7, 15


def below_normal():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def scaffold(smiles, ik14):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    try:
        m = Chem.MolFromSmiles(smiles)
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else ''
    except Exception:
        s = ''
    return s if s else 'ik14:' + str(ik14)


class Index:
    def __init__(self):
        L = lambda n: np.load(f'{IDX}/{n}.npy', mmap_mode='r')
        for n in ('spec_sid', 'spec_add', 'spec_mode', 'spec_ce', 'spec_instr', 'spec_prec', 'sid_ptr', 'sid_rows', 'sid_pid', 'tr_sid',
                  'pool_form', 'form_ptr', 'form_rows', 'pool_held', 'pool_smiles', 'pool_smiles_off'):
            setattr(self, n, L(n))
        self.off = np.load(WORK + '/spec_off.npy', mmap_mode='r'); self.mz = np.load(WORK + '/spec_mz.npy', mmap_mode='r')
        self.it = np.load(WORK + '/spec_it.npy', mmap_mode='r'); self.npk = np.diff(np.asarray(self.off))
        self.t_off = np.load(TIER + '/pc_off.npy', mmap_mode='r'); self.t_smi = np.load(TIER + '/pc_smiles.npy', mmap_mode='r')
        self.pc = {}

    def pcneg(self, which):
        if which not in self.pc:
            d = PCNEG[which]; pid = np.load(d + '/pid.npy'); m = {int(p): i for i, p in enumerate(pid)}
            self.pc[which] = (m, np.load(d + '/hard.npy', mmap_mode='r'), np.load(d + '/rows.npy', mmap_mode='r'))
        return self.pc[which]

    def pool_smiles_of(self, i):
        return bytes(self.pool_smiles[self.pool_smiles_off[i]:self.pool_smiles_off[i + 1] - 1]).decode('ascii')

    def tier_smiles(self, r):
        return bytes(self.t_smi[int(self.t_off[r]):int(self.t_off[r + 1])]).decode('ascii')

    def spectra_of(self, sid):
        rows = np.asarray(self.sid_rows[self.sid_ptr[sid]:self.sid_ptr[sid + 1]])
        rows = rows[(np.asarray(self.spec_add[rows]) < N_TEST_ADDUCTS) & (self.npk[rows] > 0)] if len(rows) else rows
        return rows

    def peaks(self, r):
        return np.asarray(self.mz[self.off[r]:self.off[r + 1]], np.float64), np.asarray(self.it[self.off[r]:self.off[r + 1]], np.float64)

    def prepared(self, r):
        import isoxe_feat as X
        mz, it = self.peaks(r)
        ce = float(self.spec_ce[r])
        return X.prep_spectrum(mz, it, float(self.spec_prec[r]), X.ADDUCTS[int(self.spec_add[r])], 1 if int(self.spec_mode[r]) > 0 else -1,
                               [ce] if np.isfinite(ce) else None, int(self.spec_instr[r]))


def pick_spectra(ix, rows, rng, k):
    rows = np.asarray(rows)
    if len(rows) == 0:
        return []
    tims = rows[np.asarray(ix.spec_instr[rows]) == 0]; other = rows[np.asarray(ix.spec_instr[rows]) != 0]
    out = list(rng.permutation(tims)) + list(rng.permutation(other))
    return [int(x) for x in out[:k]]


def negatives(ix, pid, which, rng, allow_held):
    m, hard, rows = ix.pcneg(which)
    pc = []
    s = m.get(int(pid))
    if s is not None:
        pc = [int(rows[j]) for j in np.asarray(hard[s]) if j >= 0]
    f = int(ix.pool_form[pid]); iso = np.asarray(ix.form_rows[ix.form_ptr[f]:ix.form_ptr[f + 1]]); iso = iso[iso != pid]
    if not allow_held and len(iso):
        iso = iso[~np.asarray(ix.pool_held[iso]).astype(bool)]
    iso = [int(x) for x in rng.permutation(iso)[:K_MAX]]
    return pc, iso


def select(a):
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    ix = Index(); t0 = time.time()
    st = pd.read_parquet(WORK + '/train_structs.parquet', columns=['inchikey14', 'smiles', 'formula', 'sid'])
    sp = pd.read_parquet(WORK + '/split.parquet'); fold = dict(zip(sp.sid, sp.fold))
    lib = pd.read_parquet(WORK + '/spec_meta.parquet', columns=['lib']).lib.values
    sid_of_ik = dict(zip(st.inchikey14, st.sid))
    ev_ik = set(pd.read_parquet(PANEL_Q, columns=['inchikey14']).inchikey14) | set(pd.read_parquet(MIXED_Q, columns=['inchikey14']).inchikey14)
    ev_sid = {sid_of_ik[k] for k in ev_ik if k in sid_of_ik}
    mixed_ik = set(pd.read_parquet(MIXED_Q, columns=['inchikey14']).inchikey14)
    print('evaluation structures (panel + mixed):', len(ev_sid), flush=True)
    rng = np.random.default_rng(0)

    def is_e180(sid):
        r = ix.spectra_of(sid)
        return len(r) > 0 and bool((lib[r] == 0).any())

    # V and T from the hold fold (not mixed replay molecules)
    hold = np.array([s for s, f in fold.items() if f == 'hold' and st.inchikey14.values[s] not in mixed_ik], np.int64)
    hold = rng.permutation(hold)
    val, test = [], []
    nv_t = nv_o = 0; half = a.n_val // 2
    for s in hold:
        if nv_t >= half and nv_o >= a.n_val - half and len(test) >= a.n_test:
            break
        if ix.sid_pid[s] < 0 or len(ix.spectra_of(s)) == 0:
            continue
        if is_e180(s):
            if nv_t < half:
                val.append(int(s)); nv_t += 1
            elif len(test) < a.n_test:
                test.append(int(s))
        elif nv_o < a.n_val - half:
            val.append(int(s)); nv_o += 1
    print('V', len(val), 'T', len(test), f'{time.time() - t0:.0f}s', flush=True)
    ev_all = sorted(ev_sid | set(val) | set(test))
    scaf_ev = {scaffold(st.smiles.values[s], st.inchikey14.values[s]) for s in ev_all}
    print('evaluation scaffolds', len(scaf_ev), flush=True)

    def build(sids, which, rng, kspec, allow_held, split, need):
        out = []; drop = dict(scaffold=0, spectra=0, negs=0)
        for s in sids:
            if need is not None and len(out) >= need:
                break
            pid = int(ix.sid_pid[s])
            if pid < 0:
                continue
            if split == 'train' and scaffold(st.smiles.values[s], st.inchikey14.values[s]) in scaf_ev:
                drop['scaffold'] += 1; continue
            rows = ix.spectra_of(s)
            if len(rows) == 0:
                drop['spectra'] += 1; continue
            pc, iso = negatives(ix, pid, which, rng, allow_held)
            if len(pc) + len(iso) < 2:
                drop['negs'] += 1; continue
            out.append(dict(split=split, sid=int(s), pid=pid, ik14=st.inchikey14.values[s], smiles=st.smiles.values[s], formula=st.formula.values[s],
                            spec=pick_spectra(ix, rows, rng, kspec), pc_rows=pc, pool_rows=iso, e180=bool((lib[rows] == 0).any())))
        print(split, len(out), drop, f'{time.time() - t0:.0f}s', flush=True)
        return out

    G = dict(val=build(val, 'hold', np.random.default_rng(1), a.max_spec_eval, True, 'val', None),
             test=build(test, 'hold', np.random.default_rng(2), a.max_spec_eval, True, 'test', None))
    tr = np.asarray(ix.tr_sid)
    tr = tr[~np.isin(tr, list(ev_sid))]
    tr = np.random.default_rng(0).permutation(tr)
    e = np.array([is_e180(int(s)) for s in tr[:min(len(tr), 6 * (a.n_tims + a.n_other))]])
    tr = tr[:len(e)]
    G['train'] = build([int(s) for s in tr[e]], 'train', np.random.default_rng(3), a.max_spec, False, 'train', a.n_tims) + \
        build([int(s) for s in tr[~e]], 'train', np.random.default_rng(4), a.max_spec, False, 'train', a.n_other)
    for k, v in G.items():
        for i, g in enumerate(v):
            g['gid'] = i
        pickle.dump(v, open(f'{OUT}/groups_{k}.pkl', 'wb'), protocol=4)
    print('SELECT_DONE', {k: len(v) for k, v in G.items()}, f'{time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------------------------------------ featurisation ----
_W = {}


def _init():
    below_normal()
    _W['ix'] = Index()
    import isoxe_feat as X
    X.warmup(); _W['X'] = X


def formula_of(smiles):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    try:
        return CalcMolFormula(Chem.MolFromSmiles(smiles))
    except Exception:
        return None


def ik14_of(smiles):
    from rdkit import Chem
    try:
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14]
    except Exception:
        return None


def group_candidates(ix, g):
    """[positive] + <= K_MAX same-formula negatives with distinct InChIKey14 (PubChem first, then pool)."""
    seen = {g['ik14']}; pc, pool = [], []
    for r in g['pc_rows']:
        if len(pc) >= K_PC:
            break
        s = ix.tier_smiles(r)
        if formula_of(s) != g['formula']:
            continue
        k = ik14_of(s)
        if k is None or k in seen:
            continue
        seen.add(k); pc.append(s)
    for p in g['pool_rows']:
        if len(pool) >= K_MAX - len(pc):
            break
        s = ix.pool_smiles_of(p)
        k = ik14_of(s)
        if k is None or k in seen:
            continue
        seen.add(k); pool.append(s)
    return [g['smiles']] + pc + pool, [0] + [1] * len(pc) + [2] * len(pool)


def featurize_groups(groups, cand_fn, spec_fn, require_pos=True):
    """Pack groups: cand_fn(g) -> (smiles list, source list, label list); spec_fn(g) -> prepared spectra. Returns a dict of arrays."""
    X = _W['X']; cache = {}
    out = dict(g_ncand=[], g_nspec=[], g_e180=[], g_gid=[], cand_g=[], cand_src=[], cand_lab=[], cand_smiles=[], spec_f=[], spec_mask=[], spec_meta=[],
               spec_ce=[], ann_i=[], ann_f=[])
    for g in groups:
        sps = [s for s in spec_fn(g) if s is not None]
        if not sps:
            continue
        smis, src, lab = cand_fn(g)
        feats = []
        for s, so, la in zip(smis, src, lab):
            r = X.featurize_candidate(s, sps, cache)
            if r is not None:
                feats.append((s, so, la, r))
        if len(feats) < (2 if require_pos else 1) or (require_pos and not any(f[2] for f in feats)):
            continue
        f, mask, meta, ce = X.spectra_block(sps)
        out['g_ncand'].append(len(feats)); out['g_nspec'].append(len(sps)); out['g_e180'].append(bool(g.get('e180', False))); out['g_gid'].append(g['gid'])
        out['spec_f'].append(f.astype(np.float16)); out['spec_mask'].append(mask); out['spec_meta'].append(meta); out['spec_ce'].append(ce)
        for s, so, la, (cg, ai, af) in feats:
            out['cand_g'].append(cg); out['cand_src'].append(so); out['cand_lab'].append(la); out['cand_smiles'].append(s)
            out['ann_i'].append(ai); out['ann_f'].append(af.astype(np.float16))
        if len(cache) > 20000:
            cache.clear()
    if not out['g_ncand']:
        return None
    return dict(g_ncand=np.array(out['g_ncand'], np.int32), g_nspec=np.array(out['g_nspec'], np.int32), g_e180=np.array(out['g_e180']),
                g_gid=np.array(out['g_gid'], np.int64), cand_g=np.stack(out['cand_g']), cand_src=np.array(out['cand_src'], np.int8),
                cand_lab=np.array(out['cand_lab'], np.int8), cand_smiles=np.array(out['cand_smiles'], dtype=object),
                spec_f=np.concatenate(out['spec_f']), spec_mask=np.concatenate(out['spec_mask']), spec_meta=np.concatenate(out['spec_meta']),
                spec_ce=np.concatenate(out['spec_ce']), ann_i=np.concatenate(out['ann_i']), ann_f=np.concatenate(out['ann_f']))


def _feat_chunk(task):
    path, groups = task
    if os.path.exists(path):
        return path, 0.0, -1
    t0 = time.time(); ix = _W['ix']
    def cands(g):
        smis, src = group_candidates(ix, g)
        return smis, src, [1] + [0] * (len(smis) - 1)
    D = featurize_groups(groups, cands, lambda g: [ix.prepared(r) for r in g['spec']])
    if D is not None:
        np.savez(path + '.tmp.npz', **{k: v for k, v in D.items() if k != 'cand_smiles'}, cand_smiles=D['cand_smiles'].astype(str))
        os.replace(path + '.tmp.npz', path)
    return path, time.time() - t0, 0 if D is None else len(D['g_ncand'])


def feat(a):
    groups = pickle.load(open(f'{OUT}/groups_{a.split}.pkl', 'rb'))
    if a.limit:
        groups = groups[:a.limit]
    d = f'{OUT}/feat_{a.split}'; os.makedirs(d, exist_ok=True)
    tasks = [(f'{d}/chunk_{i // a.chunk:05d}.npz', groups[i:i + a.chunk]) for i in range(0, len(groups), a.chunk)]
    tasks = [t for t in tasks if not os.path.exists(t[0])]
    print(a.split, 'chunks to do', len(tasks), flush=True); t0 = time.time()
    below_normal()
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
    ap.add_argument('--n_tims', type=int, default=20000); ap.add_argument('--n_other', type=int, default=20000)
    ap.add_argument('--n_val', type=int, default=3000); ap.add_argument('--n_test', type=int, default=2000)
    ap.add_argument('--max_spec', type=int, default=4); ap.add_argument('--max_spec_eval', type=int, default=6)
    ap.add_argument('--split', default='train'); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=250); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    below_normal()
    select(a) if a.stage == 'select' else feat(a)


if __name__ == '__main__':
    main()
