"""Training batches and the Enveda validation lists of the ownneural model.

Reads the joint study's memory-mapped spectrum cache / pool / PubChem tables (train/joint/data.py JointData) plus this study's
split.parquet (roles), negs.npz (most similar pool isomers / near-mass analogues) and banned.npy (pool rows never drawn).

A batch: B molecules drawn with probability proportional to a weight (w_env for structures with Enveda timsTOF spectra), each with
  spectra   a view of 1..max_spec of its spectra; with probability p_env only its Enveda spectra (the test's instrument) when it has them
  negatives K rows, in this order (each kind capped, the rest filled in the same order):
            0 most similar same-formula pool isomers (top 16 by Tanimoto)   1 random same-formula pool isomers
            2 PubChem window hard (most similar)                             3 PubChem window random
            4 most similar near-mass pool rows of another formula (+-10 ppm) 5 random near-mass pool rows   6 random pool rows
  w         per-molecule loss weight (w_envview when the view is Enveda-only)
"""
import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import onpaths as P  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'joint'))
sys.path.insert(0, P.CLEAN)
import data as JD  # noqa: E402  (train/joint/data.py: JointData, warm)
import onnet  # noqa: E402

CAPS = dict(iso_sim=6, iso_rand=4, pc_hard=8, pc_rand=5, win_sim=4, win_rand=2)
ENV_LIBS = (0, 9)


class OnData:
    """Everything a batch needs, memory-mapped; cheap to open in each worker."""

    def __init__(self):
        self.J = JD.JointData(neg_hold=True, pc_neg=True)
        S = pd.read_parquet(P.ON + '/split.parquet'); sid_pid = np.asarray(self.J.sid_pid); nspec = np.diff(np.asarray(self.J.sid_ptr))
        ok = (sid_pid >= 0) & (nspec > 0)
        self.tr = np.flatnonzero((S.role.values == 'train') & ok).astype(np.int64); self.val = np.flatnonzero((S.role.values == 'val') & ok).astype(np.int64)
        self.env_struct = S.env.values
        N = np.load(P.ON + '/negs.npz'); self.n_pid = N['pid']; self.n_iso = N['iso']; self.n_win = N['win']
        self.banned = np.load(P.ON + '/banned.npy')
        lib = pd.read_parquet(P.WORK + '/spec_meta.parquet', columns=['lib']).lib.values; self.spec_env = np.isin(lib, ENV_LIBS)

    def neg_row(self, pid):
        j = np.searchsorted(self.n_pid, pid)
        return j if j < len(self.n_pid) and self.n_pid[j] == pid else -1

    def negatives(self, pid, k, rng, caps=CAPS):
        """(src [k] 0 pool / 1 PubChem row, idx [k], kind [k] (-1 none)) for one positive pool row."""
        J = self.J; src = np.zeros(k, np.int8); idx = np.zeros(k, np.int64); kind = np.full(k, -1, np.int8); n = 0; used = {pid}
        j = self.neg_row(pid)
        iso_sim = self.n_iso[j][self.n_iso[j] >= 0] if j >= 0 else np.zeros(0, np.int64)
        win_sim = self.n_win[j][self.n_win[j] >= 0] if j >= 0 else np.zeros(0, np.int64)
        f = int(J.pool_form[pid]); iso = np.asarray(J.form_rows[J.form_ptr[f]:J.form_ptr[f + 1]]); iso = iso[(iso != pid) & ~self.banned[iso]]
        if len(iso) > 400:
            iso = rng.choice(iso, 400, replace=False)
        pj = int(J.pc_of_pid[pid]) if J.pc else -1; hard = rnd = np.zeros(0, np.int64)
        if pj >= 0:
            hard = np.asarray(J.pc_hard[pj]); hard = hard[hard >= 0]; rnd = np.asarray(J.pc_rand[pj]); rnd = rnd[rnd >= 0]
        m = float(J.pool_mass[pid]); lo = np.searchsorted(J.pool_mass, m * (1 - 1e-5)); hi = np.searchsorted(J.pool_mass, m * (1 + 1e-5))
        near = np.arange(lo, hi); near = near[(np.asarray(J.pool_form[lo:hi]) != f) & ~self.banned[lo:hi]]
        pcu = set()

        def take(cand, s, kd, limit):
            nonlocal n
            if n >= k or limit <= 0 or len(cand) == 0:
                return
            cand = np.asarray([c for c in cand if (c not in used if s == 0 else c not in pcu)], np.int64)
            if len(cand) == 0:
                return
            mm = min(limit, k - n, len(cand)); c = cand if len(cand) == mm else rng.choice(cand, mm, replace=False)
            src[n:n + mm] = s; idx[n:n + mm] = c; kind[n:n + mm] = kd; n += mm
            (used if s == 0 else pcu).update(int(x) for x in c)
        plan = [(iso_sim, 0, 0, caps['iso_sim']), (iso, 0, 1, caps['iso_rand']), (hard, 1, 2, caps['pc_hard']), (rnd, 1, 3, caps['pc_rand']),
                (win_sim, 0, 4, caps['win_sim']), (near, 0, 5, caps['win_rand'])]
        for cand, s, kd, lim in plan:
            take(cand, s, kd, lim)
        for cand, s, kd, _ in plan:                                              # fill in the same order
            take(cand, s, kd, k)
        if n < k:
            r = rng.integers(0, len(J.pool_mass), 4 * (k - n)); r = r[~self.banned[r]]; take(r, 0, 6, k)
        return src, idx, kind

    def view(self, sid, rng, max_spec, p_env):
        """Spectrum rows of one molecule's view and whether it is Enveda-only."""
        r = np.asarray(self.J.spectra_of(int(sid))); env_only = False
        if p_env > 0 and rng.random() < p_env:
            e = r[self.spec_env[r]]
            if len(e):
                r = e; env_only = True
        m = int(rng.integers(1, min(len(r), max_spec) + 1)); r = r if len(r) == m else rng.choice(r, m, replace=False)
        return np.sort(r), env_only

    def arrays(self, rows, owner, max_peaks):
        J = self.J
        return onnet.spectra_arrays([(J.mz[J.off[r]:J.off[r + 1]], J.it[J.off[r]:J.off[r + 1]]) for r in rows], np.asarray(J.spec_prec[rows], np.float32),
                                    np.asarray(J.spec_add[rows], np.int64), np.asarray(J.spec_mode[rows], np.int64), np.asarray(J.spec_ce[rows], np.float32),
                                    np.asarray(J.spec_instr[rows], np.int64), owner, max_peaks)


class BatchDataset:
    """Item i = batch i of the stream (deterministic in seed, i). Use with DataLoader(batch_size=None)."""

    def __init__(self, bs=192, k=31, max_spec=8, max_peaks=96, steps=1000, seed=0, offset=0, w_env=1.5, p_env=0.7, w_envview=1.5):
        self.a = dict(bs=bs, k=k, max_spec=max_spec, max_peaks=max_peaks, steps=steps, seed=seed, offset=offset, w_env=w_env, p_env=p_env, w_envview=w_envview)
        self.D = None

    def __len__(self):
        return self.a['steps']

    def _open(self):
        if self.D is None:
            self.D = OnData(); w = np.where(self.D.env_struct[self.D.tr], self.a['w_env'], 1.0); self.p = w / w.sum()
        return self.D

    def __getitem__(self, i):
        a = self.a; D = self._open(); rng = np.random.default_rng([a['seed'], 7, i + a['offset']])
        sids = D.tr[np.unique(rng.choice(len(D.tr), int(a['bs'] * 1.05) + 4, p=self.p))]
        sids = rng.permutation(sids)[:a['bs']]; B = len(sids)
        rows, owner, w = [], [], np.ones(B, np.float32)
        for b, s in enumerate(sids):
            r, env_only = D.view(s, rng, a['max_spec'], a['p_env']); rows.append(r); owner.append(np.full(len(r), b)); w[b] = a['w_envview'] if env_only else 1.0
        out = D.arrays(np.concatenate(rows), np.concatenate(owner), a['max_peaks'])
        pid = np.asarray(D.J.sid_pid)[sids].astype(np.int64); K = a['k']
        ns = np.zeros((B, K), np.int8); ni = np.zeros((B, K), np.int64); nk = np.zeros((B, K), np.int8)
        for b in range(B):
            ns[b], ni[b], nk[b] = D.negatives(int(pid[b]), K, rng)
        out.update(pid=pid, neg_kind=nk, neg_ok=nk >= 0, w=w, pos_fp=D.J.fp(np.zeros(B, np.int8), pid), neg_fp=D.J.fp(ns.ravel(), ni.ravel()).reshape(B, K, -1))
        return out


def val_lists(D, n_neg=24, seed=123, max_spec=16):
    """Fixed Enveda validation lists: per val structure its Enveda spectra (up to max_spec, file order) and 1 + n_neg candidates (true first),
    negatives drawn like training but with fixed caps (8 similar isomers, 4 random isomers, 6 PubChem hard, 2 PubChem random, 4 analogues)."""
    rng = np.random.default_rng(seed); caps = dict(iso_sim=8, iso_rand=4, pc_hard=6, pc_rand=2, win_sim=4, win_rand=0)
    rows, own, cs, ci, co = [], [], [], [], []; kept = []
    for s in D.val:
        r = np.asarray(D.J.spectra_of(int(s))); r = r[D.spec_env[r]][:max_spec]
        if not len(r):
            continue
        b = len(kept); kept.append(s); rows.append(r); own.append(np.full(len(r), b))
        pid = int(D.J.sid_pid[s]); src, idx, kind = D.negatives(pid, n_neg, rng, caps); ok = kind >= 0
        cs.append(np.r_[0, src[ok]]); ci.append(np.r_[pid, idx[ok]]); co.append(np.full(1 + ok.sum(), b))
    return dict(sids=np.asarray(kept), rows=np.concatenate(rows), owner=np.concatenate(own), src=np.concatenate(cs).astype(np.int8), idx=np.concatenate(ci).astype(np.int64),
                cowner=np.concatenate(co))


def list_mrr(score, cowner, first_is_true=True):
    """MRR of the true candidate (first of each group) with ties averaged; groups = consecutive equal cowner values."""
    st = np.r_[0, np.flatnonzero(np.diff(cowner)) + 1, len(cowner)]; rr = []
    for a, b in zip(st[:-1], st[1:]):
        v = score[a:b]; rr.append(1.0 / (1.0 + (v[1:] > v[0]).sum() + 0.5 * (v[1:] == v[0]).sum()))
    return float(np.mean(rr)), np.asarray(rr)
