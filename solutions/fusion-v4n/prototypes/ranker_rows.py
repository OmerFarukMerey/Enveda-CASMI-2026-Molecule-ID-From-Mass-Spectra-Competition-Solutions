"""Ranker rows of the v1 engine: load local-engine dumps, add leak-free extra features, score lists, train LightGBM.

A "row" is one candidate of one molecule (pool structure or generated derivative); a "group" is one molecule. Rows come
from scratch/ranker/run_engine.py (prototypes/local_engine.py): X = the engine's 94 features, F = the fe_v4 family
features, rk = the public ranker's score, y = candidate metric key equals the truth's.

    R = load_rows('D:/casmi-data/ranker_study/rows/c2_fold0', pool_meta)       # Rows
    E = extra_features(R, pop=dict(lsid=..., lpmid=..., np=...))               # popularity + confidence features
    rr = list_rr(score, R.y, R.start)                                          # whole-list reciprocal rank @25
    rk, n = ingroup_rank(score, R.y, R.start, R.fg)                            # truth's rank in its formula group

Pure numpy / pandas; LightGBM is imported only by train_ranker. Covered by test_ranker_rows.py (synthetic rows).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---- feature bookkeeping ---------------------------------------------------------------------------------------------
# blocks of the engine's 94 base features (casmi/engine.py FEATURES), used for ablations and permutation tests
BASE_BLOCKS = {
    'lib': ['lib_max', 'lib_mean', 'lib_top3', 'lib_nspec', 'lib_cos', 'lib_max_rank', 'lib_gap', 'lib_grp_max', 'lib_hit', 'lib_same_adduct'],
    'analog': ['ap', 'a1', 'best_tan', 'top_tan', 'mean_tan', 'ap_rank', 'ap_gap', 'ap_grp_max', 'top_sim', 'ap_np', 'best_tan_np', 'top_sim_np',
               'ap_np_gap', 'ap_sum', 'n_analog_hi', 'tan_zero_shift'],
    'fpnet': ['fz', 'fz_z', 'fz_rank', 'fz_gap', 'fz_norm', 'fz_top', 'fz_softmax', 'fz_grp_ent', 'fz_grp_gap', 'fz_single', 'fz_single_gap',
              'fz_merged', 'fz_merged_gap', 'fp_cos', 'fp_tan', 'fp_ll', 'fp_ll_gap', 'el_c', 'el_n', 'el_o', 'el_sum', 'el_rank'],
    'frag': ['fr_int', 'fr_int_mean', 'fr_cnt', 'fr_top10', 'fr_strict', 'fr_rank', 'fr_gap', 'fr_z', 'fr_n'],
    'subformula': ['sf_int', 'sf_cnt', 'sf_top10', 'sf_rank', 'sf_gap', 'formula_share', 'n_formula', 'formula_is_top'],
    'context': ['mass_ppm', 'mass_ppm_rank', 'n_heavy', 'fp_bits', 'n_cand', 'n_query', 'n_pos', 'n_neg', 'target_mass', 'q_npeaks'],
    'cross': ['lib_x_fz', 'ap_x_fz', 'agree_lib', 'agree_ap', 'fr_x_fz'],
    'gen': ['is_gen', 'gen_parent_sim', 'gen_steps', 'gen_parent_tan', 'n_gen', 'gen_rank_fz'],
    'xenc': ['xs_max', 'xs_mean', 'xs_merged', 'xs_gap', 'xs_rank', 'xs_z', 'xs_top', 'xs_grp_gap'],
}
FAMILIES = ['derivation', 'frag2', 'analog_struct', 'model_views', 'fragnet']
# features that identify the candidate or the query rather than describe evidence: a ranker trained on the same
# molecules can memorise (target_mass, n_cand) -> (n_heavy, fp_bits, mass_ppm) of the truth. Kept out of our ranker.
IDENTIFIERS = ['target_mass', 'mass_ppm', 'fp_bits', 'fr_n']


def block_of(name: str) -> str:
    if '__' in name:
        return name.split('__')[0]
    for b, fs in BASE_BLOCKS.items():
        if name in fs:
            return b
    return 'other'


@dataclass
class Rows:
    X: np.ndarray                 # (n, n_x) engine features
    F: np.ndarray                 # (n, n_f) family features
    rk: np.ndarray                # public ranker score
    pid: np.ndarray               # pool id, -1 = generated
    y: np.ndarray                 # bool, candidate is the truth
    mol: np.ndarray               # molecule index of each row (contiguous blocks)
    start: np.ndarray             # (n_mol + 1,) row offsets
    names_x: List[str]
    names_f: List[str]
    meta: pd.DataFrame            # one row per molecule (mid, n_cand, lib_max, ...), same order as `start`
    fg: np.ndarray = field(default=None)       # formula-group id of each row (unique within the molecule)
    key: Optional[np.ndarray] = None           # metric key of each row (object)
    smiles: Optional[np.ndarray] = None

    @property
    def n_mol(self) -> int:
        return len(self.start) - 1

    def col(self, name: str) -> np.ndarray:
        if name in self.names_x:
            return self.X[:, self.names_x.index(name)]
        return self.F[:, self.names_f.index(name)]

    def matrix(self, names: Sequence[str]) -> np.ndarray:
        return np.stack([self.col(n) for n in names], 1)


def load_rows(path: str, pool_meta: Optional[pd.DataFrame] = None, mids: Optional[set] = None, strings: bool = True) -> Rows:
    """Concatenate the chunk files of one run. pool_meta (pid-indexed: key, smiles, formula) resolves pool candidates;
    generated candidates carry their own strings in the chunk."""
    Xs, Fs, rks, pids, ys, mols, metas, gens = [], [], [], [], [], [], [], []
    names_x = names_f = None; nm = 0
    for f in sorted(glob.glob(os.path.join(path, '*.parquet'))):
        m = pd.read_parquet(f); z = np.load(f[:-8] + '.npz')
        if len(z['names_f']) == 0:                     # chunk without a single candidate
            m['row_n'] = 0; metas.append(m); nm += len(m); continue
        if names_x is None:
            names_x, names_f = [str(x) for x in z['names_x']], [str(x) for x in z['names_f']]
        g = z['g']; cnt = np.bincount(g, minlength=len(m))
        m['row_n'] = cnt
        Xs.append(z['X']); Fs.append(z['F']); rks.append(z['rk']); pids.append(z['pid']); ys.append(z['y']); mols.append(g + nm)
        gj = json.loads(str(z['gen_json']))
        if gj:
            gd = pd.DataFrame(gj); gd['mol'] = gd.mid.map(dict(zip(m.mid, np.arange(len(m)) + nm))); gens.append(gd)
        metas.append(m); nm += len(m)
    meta = pd.concat(metas, ignore_index=True)
    X, F, rk, pid, y, mol = (np.concatenate(v) for v in (Xs, Fs, rks, pids, ys, mols))
    start = np.r_[0, np.cumsum(meta.row_n.values)]
    assert start[-1] == len(y) and (np.diff(mol) >= 0).all()
    R = Rows(X=X, F=F, rk=rk, pid=pid, y=y, mol=mol, start=start, names_x=names_x, names_f=names_f, meta=meta)
    if pool_meta is not None:
        pos = np.where(pid >= 0)[0]
        form = np.empty(len(pid), object); key = np.empty(len(pid), object); smi = np.empty(len(pid), object)
        form[pos] = pool_meta.formula.values[pid[pos]]; key[pos] = pool_meta.key.values[pid[pos]]
        if strings:
            smi[pos] = pool_meta.smiles.values[pid[pos]]
        if gens:
            G = pd.concat(gens, ignore_index=True)
            ridx = start[G.mol.values] + G.j.values
            assert (pid[ridx] < 0).all()
            form[ridx] = G.formula.values; key[ridx] = G.key.values; smi[ridx] = G.smiles.values
        R.key = key; R.smiles = smi
        R.fg = formula_groups(form, mol)
    if mids is not None:
        R = subset(R, np.flatnonzero(meta.mid.isin(mids).values))
    return R


def formula_groups(formula: np.ndarray, mol: np.ndarray) -> np.ndarray:
    """Integer id per row, equal for rows of the same molecule with the same formula."""
    codes, _ = pd.factorize(pd.Series(formula, dtype=object).fillna('?'))
    return pd.factorize(mol.astype(np.int64) * (codes.max() + 2) + codes)[0].astype(np.int64)


def subset(R: Rows, mols: np.ndarray) -> Rows:
    """Rows of the given molecule indices (order kept)."""
    mols = np.asarray(mols, np.int64)
    idx = np.concatenate([np.arange(R.start[m], R.start[m + 1]) for m in mols]) if len(mols) else np.zeros(0, np.int64)
    cnt = np.array([R.start[m + 1] - R.start[m] for m in mols], np.int64)
    new_mol = np.repeat(np.arange(len(mols)), cnt)
    take = lambda v: None if v is None else v[idx]
    return Rows(X=R.X[idx], F=R.F[idx], rk=R.rk[idx], pid=R.pid[idx], y=R.y[idx], mol=new_mol, start=np.r_[0, np.cumsum(cnt)], names_x=R.names_x,
                names_f=R.names_f, meta=R.meta.iloc[mols].reset_index(drop=True), fg=take(R.fg), key=take(R.key), smiles=take(R.smiles))


# ---- scoring ---------------------------------------------------------------------------------------------------------
def _rank_of_truth(s: np.ndarray, y: np.ndarray) -> float:
    """1 + number of candidates scored above the truth; ties with the truth count half (expected rank)."""
    t = s[y][0]
    return float((s > t).sum() + ((s == t).sum() - 1) / 2.0 + 1.0)


def list_rank(score: np.ndarray, y: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Rank of the truth in the whole candidate list per molecule (inf if the truth is not a candidate)."""
    out = np.full(len(start) - 1, np.inf)
    for m in range(len(start) - 1):
        a, b = start[m], start[m + 1]
        if b > a and y[a:b].any():
            out[m] = _rank_of_truth(np.nan_to_num(score[a:b], nan=-np.inf), y[a:b])
    return out


def list_rr(score, y, start, k: int = 25) -> np.ndarray:
    r = list_rank(score, y, start)
    return np.where(r <= k, 1.0 / np.maximum(np.floor(r + 0.5), 1.0), 0.0)      # a tied truth takes the middle slot


def ingroup_rank(score, y, start, fg, restrict: Optional[np.ndarray] = None):
    """Rank of the truth among the candidates sharing its formula (and passing `restrict`, a row mask that always keeps
    the truth). Returns (rank, group size); nan / 0 where the truth is absent."""
    rk = np.full(len(start) - 1, np.nan); n = np.zeros(len(start) - 1, np.int64)
    for m in range(len(start) - 1):
        a, b = start[m], start[m + 1]
        yy = y[a:b]
        if b == a or not yy.any():
            continue
        g = fg[a:b] == fg[a:b][yy][0]
        if restrict is not None:
            g &= restrict[a:b] | yy
        s = np.nan_to_num(score[a:b][g], nan=-np.inf)
        rk[m] = _rank_of_truth(s, yy[g]); n[m] = int(g.sum())
    return rk, n


def summarize(rank: np.ndarray, n: Optional[np.ndarray] = None, k: int = 25) -> Dict[str, float]:
    """top-1 / MRR (with standard errors) of a vector of truth ranks; groups of size 1 are dropped when n is given."""
    ok = np.isfinite(rank) if n is None else (np.isfinite(rank) & (n >= 2))
    r = rank[ok]
    if not len(r):
        return dict(n=0)
    t1 = (r < 1.5).astype(float); rr = np.where(r <= k, 1.0 / np.maximum(np.floor(r + 0.5), 1), 0.0)
    se = lambda v: float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan')
    out = dict(n=int(len(r)), top1=float(t1.mean()), top1_se=se(t1), mrr=float(rr.mean()), mrr_se=se(rr))
    if n is not None:
        out['random_top1'] = float((1.0 / n[ok]).mean()); out['size'] = float(n[ok].mean())
    return out


def paired(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Mean and standard error of the paired difference a - b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return dict(delta=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan'),
                wins=int((d > 1e-12).sum()), losses=int((d < -1e-12).sum()), n=int(len(d)))


def group_z(v: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """z-score inside groups (gid = molecule or formula-group id); constant groups -> 0. NaN stay NaN."""
    s = pd.Series(v, dtype=float); g = s.groupby(gid)
    sd = g.transform('std').fillna(0).values; mu = g.transform('mean').values
    return np.where(sd > 1e-9, (v - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0 * v)


def group_rank01(v: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """0 = best (largest) .. 1 = worst inside the group, average ties; single-member groups -> 0."""
    s = pd.Series(np.nan_to_num(v, nan=-np.inf))
    r = s.groupby(gid).rank(ascending=False, method='average').values
    n = s.groupby(gid).transform('size').values
    return np.where(n > 1, (r - 1) / np.maximum(n - 1, 1), 0.0)


# ---- extra features --------------------------------------------------------------------------------------------------
def extra_features(R: Rows, pop: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """Features the public ranker never saw. Popularity (pool rows only; NaN for generated candidates): PubChem
    substance count, PubMed count, NP-catalogue count, and their position in the window / formula group. Confidence:
    size of the candidate's formula group, FPNet margins inside it, how many candidates the library and the analog
    channel vouch for. All are computable inside the scored notebook from data it already mounts."""
    n = len(R.y); mol = R.mol; fg = R.fg
    E = {}
    fz = R.col('fz').astype(np.float64) * 100.0
    s_fg = pd.Series(fz).groupby(fg)
    E['fg_size'] = np.log(s_fg.transform('size').values.astype(float))
    E['fg_share'] = s_fg.transform('size').values / pd.Series(fz).groupby(mol).transform('size').values
    E['fz_fg_gap'] = (fz - s_fg.transform('max').values) / 100.0
    E['fz_fg_z'] = group_z(fz, fg)
    E['fz_fg_rank'] = group_rank01(fz, fg)
    top2 = s_fg.transform(lambda x: np.sort(x.values)[-2] if len(x) > 1 else x.values[0]).values
    E['fz_fg_margin'] = (s_fg.transform('max').values - top2) / 100.0          # group-level: how clear the FPNet pick is
    lib = R.col('lib_max').astype(np.float64)
    E['lib_fg_max'] = pd.Series(lib).groupby(fg).transform('max').values
    E['n_lib_hit'] = np.log1p(pd.Series((lib > 0).astype(float)).groupby(mol).transform('sum').values)
    E['n_lib_hit_fg'] = np.log1p(pd.Series((lib > 0).astype(float)).groupby(fg).transform('sum').values)
    lib2 = pd.Series(lib).groupby(mol).transform(lambda x: np.sort(x.values)[-2] if len(x) > 1 else 0.0).values
    E['lib_margin'] = pd.Series(lib).groupby(mol).transform('max').values - lib2
    if pop is not None:
        ok = R.pid >= 0; p = np.where(ok, R.pid, 0)
        ls = np.where(ok, pop['lsid'][p].astype(np.float64), np.nan); lp = np.where(ok, pop['lpmid'][p].astype(np.float64), np.nan)
        tot = ls + lp
        E['pop_lsid'] = ls; E['pop_lpmid'] = lp; E['pop'] = tot
        E['pop_np'] = np.where(ok, pop['np'][p].astype(np.float64), np.nan)
        t0 = np.nan_to_num(tot, nan=0.0)
        E['pop_fg_gap'] = t0 - pd.Series(t0).groupby(fg).transform('max').values
        E['pop_fg_rank'] = group_rank01(t0, fg)
        E['pop_fg_z'] = group_z(t0, fg)
        E['pop_win_rank'] = group_rank01(t0, mol)
        E['pop_fg_max'] = pd.Series(t0).groupby(fg).transform('max').values
        E['np_fg_share'] = pd.Series(np.nan_to_num(E['pop_np'], nan=0.0) > 0).astype(float).groupby(fg).transform('mean').values
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in E.items()})


def descriptors(smiles: Sequence[str]) -> pd.DataFrame:
    """Cheap RDKit descriptors that differ between isomers (formula-level counts are constant in a formula group)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors as D
    RDLogger.DisableLog('rdApp.*')
    cols = ['d_rings', 'd_arom_rings', 'd_fsp3', 'd_rot', 'd_hbd', 'd_stereo', 'd_ring_sys_max', 'd_arom_atoms', 'd_oh', 'd_co', 'd_ether', 'd_amide']
    pats = [Chem.MolFromSmarts(s) for s in ('[OX2H]', '[CX3]=[OX1]', '[OD2]([#6])[#6]', 'C(=O)[NX3]')]
    out = np.full((len(smiles), len(cols)), np.nan, np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            continue
        ri = m.GetRingInfo()
        big = max((len(r) for r in ri.AtomRings()), default=0)
        out[i] = [ri.NumRings(), D.CalcNumAromaticRings(m), D.CalcFractionCSP3(m), D.CalcNumRotatableBonds(m), D.CalcNumHBD(m),
                  D.CalcNumAtomStereoCenters(m), big, sum(a.GetIsAromatic() for a in m.GetAtoms())] + [len(m.GetSubstructMatches(p)) for p in pats]
    return pd.DataFrame(out, columns=cols)


# ---- training --------------------------------------------------------------------------------------------------------
LGB_PARAMS = dict(objective='lambdarank', metric='None', learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, lambdarank_truncation_level=30, label_gain=[0, 1], verbose=-1,
                  num_threads=2)       # few threads: on a shared machine OpenMP barriers cost a scheduler quantum each


def train_ranker(X: np.ndarray, y: np.ndarray, sizes: np.ndarray, names: Sequence[str], rounds: int = 400, params: Optional[dict] = None,
                 seed: int = 0, weight: Optional[np.ndarray] = None):
    """LightGBM lambdarank on contiguous groups (sizes = rows per molecule)."""
    import lightgbm as lgb
    p = dict(LGB_PARAMS); p.update(params or {}); p.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    ds = lgb.Dataset(X, label=y.astype(np.int32), group=sizes, weight=weight, feature_name=list(names), free_raw_data=False)
    return lgb.train(p, ds, num_boost_round=rounds)


def grouped_folds(labels: Sequence, n_fold: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per molecule such that equal labels (scaffold, formula, structure key ...) share a fold."""
    u, inv = np.unique(np.asarray(labels, object).astype(str), return_inverse=True)
    f = np.random.RandomState(seed).permutation(len(u)) % n_fold
    return f[inv]


def cv_predict(X: np.ndarray, y: np.ndarray, start: np.ndarray, fold: np.ndarray, names: Sequence[str], rounds: int = 400,
               params: Optional[dict] = None, seeds: Sequence[int] = (0,), train_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Out-of-fold scores. fold: per molecule. train_mask: per molecule, molecules allowed in training (default all);
    molecules without a positive row are skipped in training automatically."""
    n_mol = len(start) - 1
    sizes = np.diff(start); mol = np.repeat(np.arange(n_mol), sizes)
    has_pos = np.array([y[start[m]:start[m + 1]].any() for m in range(n_mol)])
    ok = has_pos if train_mask is None else (has_pos & train_mask)
    pred = np.zeros(len(y))
    for f in np.unique(fold):
        tr_m = (fold != f) & ok; te_m = fold == f
        tr = tr_m[mol]; te = te_m[mol]
        ps = [train_ranker(X[tr], y[tr], sizes[tr_m], names, rounds, params, seed=s).predict(X[te]) for s in seeds]
        pred[te] = np.mean(ps, axis=0)
    return pred
