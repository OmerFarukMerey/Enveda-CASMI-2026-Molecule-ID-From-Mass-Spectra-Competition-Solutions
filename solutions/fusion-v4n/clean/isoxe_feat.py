"""Isomer cross-encoder featurizer: annotate every peak of a molecule's spectra with the bond-cleavage fragments of ONE candidate.

Ours (2026-10-18, isoxe pilot; reviews/isoxe-2026-10-18.md). Same fragment idea as nb/embed/frag_rescore.py (MetFrag-lite), new
code that also keeps WHAT explains a peak, so a model can learn which cleavages matter on an instrument:

  candidate graph   heavy atoms (mass incl. implicit H, element / aromatic / ring flags) and bonds with a hashed bond-type id
                    (elements, aromaticity, degree, H count and ring flag of both ends, bond order, ring bond)
  fragments         connected pieces left after removing 1..depth bonds (depth 3 if <= MAX_B3 bonds, 2 if <= MAX_B2, else 1),
                    plus the intact molecule (k = 0); per unique mass (1e-4 Da) the cheapest (fewest cleavages) piece is kept
                    with its heavy-atom, N, O, ring- and aromatic-atom counts and the type ids of the bonds it broke
  peak annotation   neutral mass needed = m/z - carrier - dh * H for the carriers the adduct allows and dh in -2..+3; best
                    explanation = min(k + 0.25 |dh| + 0.1 [non-proton carrier]) within tol = max(TOL_DA, TOL_PPM); also the
                    number of distinct fragments within tol (ambiguity)

    g = cand_graph(smiles)                      -> dict | None
    F = fragments(g)                            -> dict of arrays (sorted by mass)
    sp = prep_spectrum(mz, it, prec, adduct, mode, ce, instr)   -> dict (top PEAKS peaks, spectrum-side features)
    ai, af = annotate(F, g, sp)                 -> int16 [PEAKS, N_INT], float32 [PEAKS, N_FLT]
    cg = cand_global(g, F)                      -> float32 [N_CAND]

isoxe v2 additions (2026-10-28, reviews/isoxe2-2026-10-28.md; v1 outputs unchanged): cand_graph also returns per-atom
(`afeat`, N_AF int8) and per-bond (`bfeat`, N_BF int8) categorical features for a graph network; fragments return the raw
broken-bond indices (`bi`); annotate(..., bonds=True) and featurize_candidate(..., bonds=True) also return them per peak
(int16, -1 = none); featurize_candidate / spectra_block take `peaks` (default PEAKS = v1's 32).

Only NumPy, RDKit and (optionally) numba. Spectra are expected cleaned like the library cache (chemcore.clean_peaks:
floor 0.001, top 512, m/z <= precursor + 2), which `clean_spectrum` applies.
"""
from __future__ import annotations

import zlib

import numpy as np

AMU = {'C': 12.0, 'H': 1.00782503207, 'N': 14.0030740048, 'O': 15.9949146196, 'P': 30.97376163, 'S': 31.97207100,
       'F': 18.99840322, 'Cl': 34.96885268, 'Br': 78.9183371, 'I': 126.904473, 'Na': 22.9897692809, 'K': 38.96370668,
       'Si': 27.9769265325, 'B': 11.0093054, 'Se': 79.9165213}
H = AMU['H']
E = 0.00054857990
PROTON = H - E
NH4 = AMU['N'] + 4 * H
FORMATE = AMU['C'] + 2 * H + 2 * AMU['O']
CARRIERS = {
    '[M+H]+': (PROTON,), '[M-H2O+H]+': (PROTON,), '[M-2H2O+H]+': (PROTON,),
    '[M+NH4]+': (PROTON, NH4 - E), '[M+Na]+': (PROTON, AMU['Na'] - E), '[M+K]+': (PROTON, AMU['K'] - E),
    '[M-H]-': (-PROTON,), '[M-H2O-H]-': (-PROTON,), '[M+CH2O2-H]-': (-PROTON, FORMATE - PROTON),
    '[M+Cl]-': (-PROTON, AMU['Cl'] + E),
}
ADDUCTS = ('[M+H]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+', '[M+Na]+', '[M+K]+', '[M-H]-', '[M-H2O-H]-', '[M+CH2O2-H]-',
           '[M+Cl]-')                                     # index 0..9; 10 = other
DH = np.arange(-2, 4)                                     # hydrogen shifts
PEAKS = 32                                                # peaks kept per spectrum (most intense)
TOL_DA, TOL_PPM = 0.01, 20e-6
MAX_B3, MAX_B2 = 50, 150
NBT = 4096                                                # bond-type hash buckets (0 = no bond)
N_INT = 6                                                 # k+1 (0 = unexplained), dh+3 (7 = none), carrier (3 = none), bt1, bt2, bt3
N_FLT = 8                                                 # err/tol, heavy frac, N frac, O frac, ring frac, arom frac, log1p(nmatch)/3, explained
N_SPF = 6                                                 # spectrum-side per peak: mz/1000, mz/prec, (prec-mz)/200, log10 it, rank, sqrt it
N_CAND = 6                                                # candidate-global features
VERSION = 1
ELEM = ('C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I')    # atom element ids 1..9, 10 = other, 0 = padding
N_AF = 6                                                  # element, aromatic, ring, degree (0..5), total H (0..4), charge + 1 (0..2)
N_BF = 3                                                  # order (0 single, 1 double, 2 triple, 3 aromatic), ring, conjugated
AF_SIZES = (11, 2, 2, 6, 5, 3)
BF_SIZES = (4, 2, 2)
_BO = {1.0: 0, 2.0: 1, 3.0: 2, 1.5: 3}


# ------------------------------------------------------------------------------------------------ candidate graph ---
def _atype(a):
    return f"{a.GetSymbol()}{'a' if a.GetIsAromatic() else ''}{min(a.GetDegree(), 4)}h{min(a.GetTotalNumHs(), 3)}{'r' if a.IsInRing() else ''}"


def bond_type_id(b) -> int:
    """Stable hashed bond-type id in 1..NBT-1 (both atom types sorted, bond order, ring bond)."""
    x, y = sorted((_atype(b.GetBeginAtom()), _atype(b.GetEndAtom())))
    s = f'{x}|{y}|{b.GetBondTypeAsDouble():.1f}|{int(b.IsInRing())}'
    return 1 + zlib.crc32(s.encode('ascii')) % (NBT - 1)


def cand_graph(smiles):
    """Heavy-atom graph of a candidate or None if it does not parse."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    except Exception:
        m = None
    if m is None or m.GetNumAtoms() == 0:
        return None
    pt = Chem.GetPeriodicTable()
    n = m.GetNumAtoms()
    w = np.empty(n, np.float64)
    isN = np.zeros(n, np.float64); isO = np.zeros(n, np.float64); ring = np.zeros(n, np.float64); arom = np.zeros(n, np.float64)
    for a in m.GetAtoms():
        s = a.GetSymbol(); i = a.GetIdx()
        am = AMU.get(s)
        if am is None:
            am = pt.GetMostCommonIsotopeMass(s)
        w[i] = am + a.GetTotalNumHs() * H
        isN[i] = s == 'N'; isO[i] = s == 'O'; ring[i] = a.IsInRing(); arom[i] = a.GetIsAromatic()
    bonds = list(m.GetBonds())
    afeat = np.array([[ELEM.index(a.GetSymbol()) + 1 if a.GetSymbol() in ELEM else len(ELEM) + 1, int(a.GetIsAromatic()), int(a.IsInRing()),
                       min(a.GetDegree(), 5), min(a.GetTotalNumHs(), 4), min(max(a.GetFormalCharge(), -1), 1) + 1] for a in m.GetAtoms()],
                     np.int8).reshape(-1, N_AF)
    bfeat = np.array([[_BO.get(b.GetBondTypeAsDouble(), 0), int(b.IsInRing()), int(b.GetIsConjugated())] for b in bonds],
                     np.int8).reshape(-1, N_BF)
    ea = np.array([b.GetBeginAtomIdx() for b in bonds], np.int64)
    eb = np.array([b.GetEndAtomIdx() for b in bonds], np.int64)
    bt = np.array([bond_type_id(b) for b in bonds], np.int64)
    return dict(n=n, w=w, isN=isN, isO=isO, ring=ring, arom=arom, ea=ea, eb=eb, bt=bt, afeat=afeat, bfeat=bfeat,
                npieces=len(Chem.GetMolFrags(m)), nring=m.GetRingInfo().NumRings(), nN=float(isN.sum()), nO=float(isO.sum()))


# ------------------------------------------------------------------------------------------------ enumeration -------
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _emit(n, A, ea, eb, removed, parent, acc, out, ob, cnt, k, b1, b2, b3):
    """Union the kept bonds; if the graph splits, append each piece's sums (columns of A) with cleavage count and bonds."""
    for a in range(n):
        parent[a] = a
    ncomp = n
    for e in range(ea.shape[0]):
        if removed[e]:
            continue
        ra = _find(parent, ea[e]); rb = _find(parent, eb[e])
        if ra != rb:
            parent[ra] = rb
            ncomp -= 1
    if ncomp < 2:
        return cnt
    for a in range(n):
        for c in range(A.shape[1]):
            acc[a, c] = 0.0
    for a in range(n):
        r = _find(parent, a)
        for c in range(A.shape[1]):
            acc[r, c] += A[a, c]
    for a in range(n):
        if parent[a] == a:
            for c in range(A.shape[1]):
                out[cnt, c] = acc[a, c]
            out[cnt, A.shape[1]] = k
            ob[cnt, 0] = b1; ob[cnt, 1] = b2; ob[cnt, 2] = b3
            cnt += 1
    return cnt


def _enumerate(n, A, ea, eb, depth, npieces):
    nb = ea.shape[0]
    n2 = nb * (nb - 1) // 2 if depth >= 2 else 0
    n3 = nb * (nb - 1) * (nb - 2) // 6 if depth >= 3 else 0
    cap = (npieces + 1) * nb + (npieces + 2) * n2 + (npieces + 3) * n3 + 1
    out = np.empty((cap, A.shape[1] + 1), np.float64)
    ob = np.full((cap, 3), -1, np.int64)
    parent = np.empty(n, np.int64)
    acc = np.empty((n, A.shape[1]), np.float64)
    removed = np.zeros(nb, np.bool_)
    cnt = 0
    for i in range(nb):
        removed[i] = True
        cnt = _emit(n, A, ea, eb, removed, parent, acc, out, ob, cnt, 1, i, -1, -1)
        if depth >= 2:
            for j in range(i + 1, nb):
                removed[j] = True
                cnt = _emit(n, A, ea, eb, removed, parent, acc, out, ob, cnt, 2, i, j, -1)
                if depth >= 3:
                    for l in range(j + 1, nb):
                        removed[l] = True
                        cnt = _emit(n, A, ea, eb, removed, parent, acc, out, ob, cnt, 3, i, j, l)
                        removed[l] = False
                removed[j] = False
        removed[i] = False
    return out[:cnt], ob[:cnt]


def _match(q, ns, npk, tol, fm, fk, sdh, scar):
    """Best fragment per peak over all (carrier, dh) shifts. q: [ns * npk] neutral masses needed (shift-major)."""
    best = np.full(npk, -1, np.int64); bs = np.full(npk, -1, np.int64); berr = np.zeros(npk); nm = np.zeros(npk, np.int64)
    bc = np.full(npk, 1e9)
    for p in range(npk):
        t = tol[p]
        for s in range(ns):
            x = q[s * npk + p]
            lo = np.searchsorted(fm, x - t)
            hi = np.searchsorted(fm, x + t, side='right')
            for j in range(lo, hi):
                nm[p] += 1
                c = fk[j] + 0.25 * abs(sdh[s]) + 0.1 * (scar[s] > 0) + 0.01 * abs(fm[j] - x) / t
                if c < bc[p]:
                    bc[p] = c; best[p] = j; bs[p] = s; berr[p] = fm[j] - x
    return best, bs, berr, nm


try:
    from numba import njit
    _find = njit(cache=True, nogil=True)(_find)
    _emit = njit(cache=True, nogil=True)(_emit)
    _enumerate = njit(cache=True, nogil=True)(_enumerate)
    _match = njit(cache=True, nogil=True)(_match)
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


def depth_for(nb, max_b3=MAX_B3, max_b2=MAX_B2):
    return 3 if nb <= max_b3 else 2 if nb <= max_b2 else 1


def fragments(g, depth=None):
    """Unique fragment masses of a candidate graph with their cheapest explanation. Arrays sorted by mass:
    m (neutral mass), k (cleavages, 0 = intact), heavy, N, O, ring, arom (atom counts), b (bond-type ids [n, 3], 0 = none)."""
    nb = len(g['ea'])
    d = depth_for(nb) if depth is None else int(depth)
    A = np.stack([g['w'], np.ones(g['n']), g['isN'], g['isO'], g['ring'], g['arom']], 1).astype(np.float64)
    intact = A.sum(0)
    rows = [np.r_[intact, 0.0][None, :]]; bix = [np.full((1, 3), -1, np.int64)]
    if nb:
        out, ob = _enumerate(g['n'], A, g['ea'], g['eb'], d, g['npieces'])
        rows.append(out); bix.append(ob)
    R = np.concatenate(rows); B = np.concatenate(bix)
    key = np.round(R[:, 0], 4)
    o = np.lexsort((R[:, -1], key))
    R, B, key = R[o], B[o], key[o]
    first = np.ones(len(key), bool); first[1:] = key[1:] != key[:-1]
    R, B = R[first], B[first]
    bt = np.where(B >= 0, g['bt'][np.maximum(B, 0)], 0) if nb else np.zeros((len(R), 3), np.int64)
    return dict(m=R[:, 0].copy(), heavy=R[:, 1], N=R[:, 2], O=R[:, 3], ring=R[:, 4], arom=R[:, 5], k=R[:, 6].astype(np.int64),
                b=bt.astype(np.int64), bi=B.astype(np.int64), depth=d)


# ------------------------------------------------------------------------------------------------ spectra -----------
def clean_spectrum(mz, it, prec, floor=0.001, topk=512, over=2.0):
    """chemcore.clean_peaks (the library cache's cleaning), repeated here so this module stands alone."""
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    inr = (mz <= prec + over) & np.isfinite(mz) & np.isfinite(it)
    if not inr.any() or it[inr].max() <= 0:
        return np.empty(0), np.empty(0)
    mx = it[inr].max()
    idx = np.flatnonzero(inr & (it >= floor * mx) & (mz > 0))
    if len(idx) > topk:
        idx = idx[np.argsort(it[idx], kind='stable')[len(idx) - topk:]]
    o = np.argsort(mz[idx], kind='stable')
    return mz[idx][o], (it[idx] / mx)[o]


def adduct_index(adduct):
    try:
        return ADDUCTS.index(adduct)
    except ValueError:
        return len(ADDUCTS)


def carriers(adduct, mode):
    c = CARRIERS.get(adduct)
    if c is not None:
        return c
    if isinstance(adduct, str) and adduct.endswith('-'):
        return (-PROTON,)
    if isinstance(adduct, str) and adduct.endswith('+'):
        return (PROTON,)
    return (PROTON,) if (mode or 1) > 0 else (-PROTON,)


def prep_spectrum(mz, it, prec, adduct, mode, ce=None, instr=0, peaks=PEAKS):
    """Top `peaks` peaks (by intensity, then m/z order) of an already cleaned spectrum and its spectrum-side features.
    Returns None for an empty spectrum. meta = [adduct index, mode (0 neg / 1 pos), instrument id, ce known]; ce in eV."""
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[ok], it[ok]
    if len(mz) == 0:
        return None
    if len(mz) > peaks:
        k = np.argsort(-it, kind='stable')[:peaks]; k.sort(); mz, it = mz[k], it[k]
    it = it / it.max()
    prec = float(prec) if prec and np.isfinite(prec) else float(mz.max())
    rank = np.empty(len(it)); rank[np.argsort(-it, kind='stable')] = np.arange(len(it))
    f = np.stack([mz / 1000.0, mz / prec, (prec - mz) / 200.0, np.log10(np.maximum(it, 1e-4)) / 4.0 + 1.0,
                  rank / peaks, np.sqrt(it)], 1).astype(np.float32)
    car = carriers(adduct, mode)
    sh, sdh, scar = [], [], []
    for ci, c in enumerate(car):
        for d in DH:
            sh.append(c + d * H); sdh.append(int(d)); scar.append(ci)
    pos = 1 if (mode if mode is not None else (1 if str(adduct).endswith('+') else -1)) > 0 else 0
    cev = float(np.mean(ce)) if ce is not None and np.size(ce) and np.all(np.isfinite(np.asarray(ce, float))) else np.nan
    return dict(mz=mz, it=it, prec=prec, f=f, shifts=np.array(sh), sdh=np.array(sdh, np.int64), scar=np.array(scar, np.int64),
                tol=np.maximum(TOL_DA, mz * TOL_PPM), meta=np.array([adduct_index(adduct), pos, int(instr), int(np.isfinite(cev))], np.int16),
                ce=np.float32(0.0 if not np.isfinite(cev) else cev / 100.0))


def annotate(F, g, sp, bonds=False):
    """Candidate-specific annotation of the spectrum's peaks: int16 [n, N_INT], float32 [n, N_FLT] (n = peaks of sp);
    bonds=True: also int16 [n, 3] broken-bond indices of the best explanation (-1 = none)."""
    npk = len(sp['mz']); ns = len(sp['shifts'])
    q = (sp['mz'][None, :] - sp['shifts'][:, None]).ravel()
    best, bs, berr, nm = _match(q, ns, npk, sp['tol'], F['m'], F['k'], sp['sdh'], sp['scar'])
    ai = np.zeros((npk, N_INT), np.int16); af = np.zeros((npk, N_FLT), np.float32)
    ai[:, 1] = 7; ai[:, 2] = 3
    hit = best >= 0
    if hit.any():
        j = best[hit]; s = bs[hit]
        ai[hit, 0] = F['k'][j] + 1
        ai[hit, 1] = sp['sdh'][s] + 3
        ai[hit, 2] = sp['scar'][s]
        ai[hit, 3:6] = F['b'][j]
        nh = max(float(g['n']), 1.0)
        af[hit, 0] = berr[hit] / sp['tol'][hit]
        af[hit, 1] = F['heavy'][j] / nh
        af[hit, 2] = F['N'][j] / (1.0 + g['nN'])
        af[hit, 3] = F['O'][j] / (1.0 + g['nO'])
        af[hit, 4] = F['ring'][j] / np.maximum(F['heavy'][j], 1.0)
        af[hit, 5] = F['arom'][j] / np.maximum(F['heavy'][j], 1.0)
        af[hit, 7] = 1.0
    af[:, 6] = np.log1p(nm) / 3.0
    if bonds:
        bi = np.full((npk, 3), -1, np.int16)
        if hit.any():
            bi[hit] = F['bi'][best[hit]]
        return ai, af, bi
    return ai, af


def cand_global(g, F):
    nb = len(g['ea'])
    return np.array([np.log1p(len(F['m'])) / 10.0, g['n'] / 50.0, nb / 50.0, g['nring'] / 10.0, F['depth'] / 3.0,
                     float((F['k'] == 1).sum()) / max(nb, 1)], np.float32)


def featurize_candidate(smiles, spectra, cache=None, peaks=PEAKS, bonds=False):
    """(cand_global [N_CAND], ann_int [S, P, N_INT], ann_flt [S, P, N_FLT]) for prepared spectra (padded to `peaks`), or None.
    bonds=True: (cand_global, ann_int, ann_flt, ann_bonds [S, P, 3] int16, graph dict incl. afeat / bfeat / ea / eb)."""
    if cache is not None and smiles in cache:
        g, F = cache[smiles]
    else:
        g = cand_graph(smiles)
        F = fragments(g) if g is not None else None
        if cache is not None:
            cache[smiles] = (g, F)
    if g is None:
        return None
    S = len(spectra)
    AI = np.zeros((S, peaks, N_INT), np.int16); AF = np.zeros((S, peaks, N_FLT), np.float32)
    AI[:, :, 1] = 7; AI[:, :, 2] = 3
    AB = np.full((S, peaks, 3), -1, np.int16) if bonds else None
    for i, sp in enumerate(spectra):
        if bonds:
            ai, af, bi = annotate(F, g, sp, bonds=True)
            AB[i, :len(bi)] = bi
        else:
            ai, af = annotate(F, g, sp)
        AI[i, :len(ai)] = ai; AF[i, :len(af)] = af
    if bonds:
        return cand_global(g, F), AI, AF, AB, g
    return cand_global(g, F), AI, AF


def spectra_block(spectra, peaks=PEAKS):
    """Spectrum-side arrays for prepared spectra: f [S, P, N_SPF] float32, mask [S, P] bool, meta [S, 4] int16, ce [S] float32."""
    S = len(spectra)
    f = np.zeros((S, peaks, N_SPF), np.float32); mask = np.zeros((S, peaks), bool)
    meta = np.zeros((S, 4), np.int16); ce = np.zeros(S, np.float32)
    for i, sp in enumerate(spectra):
        n = len(sp['mz']); f[i, :n] = sp['f']; mask[i, :n] = True; meta[i] = sp['meta']; ce[i] = sp['ce']
    return f, mask, meta, ce


def warmup():
    g = cand_graph('CCOC(=O)c1ccccc1O')
    fragments(g)
    sp = prep_spectrum(np.array([121.03, 149.06, 167.07]), np.array([1.0, 0.5, 0.2]), 167.0703, '[M+H]+', 1, [20.0])
    annotate(fragments(g), g, sp)
    return HAVE_NUMBA
