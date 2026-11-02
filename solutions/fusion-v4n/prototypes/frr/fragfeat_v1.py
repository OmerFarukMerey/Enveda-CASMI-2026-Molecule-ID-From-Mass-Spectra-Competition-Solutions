"""Fragment-explanation features for one molecule's spectra against candidate structures (our own code).

Fragments: neutral masses of the connected pieces left after removing 1 or 2 bonds of the heavy-atom graph (implicit H
stay on their atom). Only cuts that split the graph give new masses: a single bridge; two bridges; two non-bridge bonds
of one ring system that together disconnect it (one bridge plus one ring bond gives nothing new). Level 0 is the
intact molecule, level 1 one cut, level 2 two cuts (masses not already reachable with fewer cuts).

A peak at m/z is explained at level L when some level-<=L neutral mass lies within `tol` of
m/z - carrier - dh * H for a charge carrier the adduct allows and a hydrogen shift dh in -2..+3.
Chance level: the same match with every peak moved by +-5 or +-9 Da (mean of the four), shifts no H shift or carrier explains
and that keep the mass defect, so the chance rate is comparable.
"""
import numpy as np
from rdkit import Chem

HM = 1.00782503207
E = 0.00054857990
PROTON = HM - E
_PT = Chem.GetPeriodicTable()
NA, K, CL = 22.9897692809, 38.96370668, 34.96885268
FORMATE = 12.0 + 2 * HM + 2 * 15.9949146196
NH4 = 14.0030740048 + 4 * HM
CARRIERS = {
    '[M+H]+': (PROTON,), '[M-H2O+H]+': (PROTON,), '[M-2H2O+H]+': (PROTON,), '[2M+H]+': (PROTON,),
    '[M+NH4]+': (PROTON, NH4 - E), '[M+Na]+': (PROTON, NA - E), '[2M+Na]+': (PROTON, NA - E), '[M+K]+': (PROTON, K - E),
    '[M-H]-': (-PROTON,), '[M-H2O-H]-': (-PROTON,), '[2M-H]-': (-PROTON,), '[M+CH2O2-H]-': (-PROTON, FORMATE - PROTON),
    '[M+Cl]-': (-PROTON, CL + E),
}
SHIFTS = np.arange(-2, 4)
DECOY_OFFSETS = (5.0, -5.0, 9.0, -9.0)   # non-chemical shifts that keep the mass defect: the chance-match level


def carriers(adduct, mode):
    c = CARRIERS.get(adduct)
    if c is not None:
        return c
    if isinstance(adduct, str) and adduct.endswith('-'):
        return (-PROTON,)
    if isinstance(adduct, str) and adduct.endswith('+'):
        return (PROTON,)
    return (PROTON,) if str(mode).startswith('pos') else (-PROTON,)


def _flood(seed, adj):
    comp = frontier = 1 << seed
    while frontier:
        nxt = 0
        f = frontier
        while f:
            low = f & -f
            nxt |= adj[low.bit_length() - 1]
            f ^= low
        frontier = nxt & ~comp
        comp |= frontier
    return comp


def _bridges(n, nbrs, bonds):
    """Bridge flags per bond (iterative Tarjan)."""
    idx_of = {}
    for k, (i, j) in enumerate(bonds):
        idx_of[(i, j)] = idx_of[(j, i)] = k
    disc = [-1] * n; low = [0] * n; t = 0; br = [False] * len(bonds)
    for root in range(n):
        if disc[root] >= 0:
            continue
        stack = [(root, -1, iter(nbrs[root]))]
        disc[root] = low[root] = t; t += 1
        while stack:
            v, pe, it = stack[-1]
            adv = False
            for w in it:
                e = idx_of[(v, w)]
                if e == pe:
                    continue
                if disc[w] < 0:
                    disc[w] = low[w] = t; t += 1
                    stack.append((w, e, iter(nbrs[w]))); adv = True; break
                low[v] = min(low[v], disc[w])
            if not adv:
                stack.pop()
                if stack:
                    u = stack[-1][0]
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        br[pe] = True
    return br


def frag_levels(smi, max_ring_pairs=400):
    """[level0, level1, level2] sorted arrays of neutral fragment masses, plus structure counts; None if unparsable."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    n = m.GetNumAtoms()
    if n == 0:
        return None
    w = np.array([_PT.GetMostCommonIsotopeMass(a.GetSymbol()) + a.GetTotalNumHs() * HM for a in m.GetAtoms()])
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    nbrs = [[] for _ in range(n)]
    adj = [0] * n
    for i, j in bonds:
        nbrs[i].append(j); nbrs[j].append(i); adj[i] |= 1 << j; adj[j] |= 1 << i
    nbyte = (n + 7) // 8
    tab = np.zeros((nbyte, 256))
    for c in range(nbyte):
        ws = np.zeros(8); seg = w[c * 8:(c + 1) * 8]; ws[:len(seg)] = seg
        bits = (np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1
        tab[c] = bits @ ws

    def mass(mask):
        s = 0.0
        for c in range(nbyte):
            s += tab[c][(mask >> (8 * c)) & 255]
        return s

    total = float(w.sum()); full = (1 << n) - 1
    br = _bridges(n, nbrs, bonds)
    l1 = set(); l2 = set()
    bridge_side = []
    for k, (i, j) in enumerate(bonds):
        if not br[k]:
            continue
        a2 = list(adj); a2[i] &= ~(1 << j); a2[j] &= ~(1 << i)
        ci = _flood(i, a2)
        bridge_side.append((k, ci))
        mi = mass(ci); l1.add(round(mi, 5)); l1.add(round(total - mi, 5))
    # two bridges: three pieces, from the one-cut sides
    for x in range(len(bridge_side)):
        for y in range(x + 1, len(bridge_side)):
            s1, s2 = bridge_side[x][1], bridge_side[y][1]
            for p in (s1 & s2, s1 & ~s2 & full, ~s1 & s2 & full, ~s1 & ~s2 & full):
                if p:
                    l2.add(round(mass(p), 5))
    # two ring bonds of one ring system
    ring_bonds = [k for k in range(len(bonds)) if not br[k]]
    if ring_bonds:
        # ring systems = connected components of the graph with only non-bridge bonds
        radj = [0] * n
        for k in ring_bonds:
            i, j = bonds[k]; radj[i] |= 1 << j; radj[j] |= 1 << i
        sys_of = {}
        for k in ring_bonds:
            i = bonds[k][0]
            sys_of.setdefault(_flood(i, radj), []).append(k)
        pairs = 0
        for ks in sys_of.values():
            for x in range(len(ks)):
                for y in range(x + 1, len(ks)):
                    pairs += 1
                    if pairs > max_ring_pairs:
                        break
                    (a, b), (c, d) = bonds[ks[x]], bonds[ks[y]]
                    a2 = list(adj)
                    a2[a] &= ~(1 << b); a2[b] &= ~(1 << a); a2[c] &= ~(1 << d); a2[d] &= ~(1 << c)
                    ca = _flood(a, a2)
                    if ca != full:
                        mi = mass(ca); l2.add(round(mi, 5)); l2.add(round(total - mi, 5))
    l2 -= l1
    l1.discard(round(total, 5)); l2.discard(round(total, 5))
    ri = m.GetRingInfo()
    info = dict(n_atoms=n, n_bonds=len(bonds), n_bridges=int(sum(br)), n_rings=ri.NumRings(),
                n_arom=sum(1 for r in ri.AtomRings() if all(m.GetAtomWithIdx(a).GetIsAromatic() for a in r)),
                n_l1=len(l1), n_l2=len(l2))
    return [np.array([total]), np.array(sorted(l1)), np.array(sorted(l2))], info


def prep(mz, it, adduct, mode, offset=0.0):
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[ok] + offset, it[ok]
    if len(mz) == 0:
        return None
    sh = np.array([c + dh * HM for c in carriers(adduct, mode) for dh in SHIFTS])
    q = (mz[None, :] - sh[:, None])
    return dict(it=it, q=q, top3=np.argsort(-it)[:3])


def peak_levels(levels, p, tol):
    """Fewest cuts explaining each peak (0, 1, 2) or 3 = unexplained."""
    q = p['q']; best = np.full(q.shape[1], 3)
    for L in (2, 1, 0):
        arr = levels[L]
        if len(arr) == 0:
            continue
        a = np.searchsorted(arr, q - tol, 'left'); b = np.searchsorted(arr, q + tol, 'right')
        hit = (b > a).any(axis=0)
        best = np.where(hit, L, best)
    return best


def spectrum_features(levels, p, pdec, tol=0.005):
    it = p['it']; lin = it / it.sum(); sq = np.sqrt(it); sq = sq / sq.sum()
    lv = peak_levels(levels, p, tol)
    lv10 = peak_levels(levels, p, 0.010); lv3 = peak_levels(levels, p, 0.003)
    f = dict(
        lin1=float(lin @ (lv <= 1)), lin2=float(lin @ (lv <= 2)), sq1=float(sq @ (lv <= 1)), sq2=float(sq @ (lv <= 2)),
        cnt2=float((lv <= 2).mean()), top3=float((lv[p['top3']] <= 2).mean()),
        rd=float(lin @ np.select([lv <= 1, lv == 2], [1.0, 0.6], 0.0)),
        sq2_t10=float(sq @ (lv10 <= 2)), sq2_t3=float(sq @ (lv3 <= 2)), lin1_t3=float(lin @ (lv3 <= 1)),
    )
    dl, ds = [], []
    for d in pdec:
        if d is None:
            continue
        lvd = peak_levels(levels, d, tol)
        dl.append(float(lin @ (lvd <= 2))); ds.append(float(sq @ (lvd <= 2)))
    f['dlin2'] = float(np.mean(dl)) if dl else 0.0
    f['dsq2'] = float(np.mean(ds)) if ds else 0.0
    f['clin2'] = f['lin2'] - f['dlin2']; f['csq2'] = f['sq2'] - f['dsq2']
    return f


SPEC_KEYS = ['lin1', 'lin2', 'sq1', 'sq2', 'cnt2', 'top3', 'rd', 'sq2_t10', 'sq2_t3', 'lin1_t3', 'dlin2', 'dsq2', 'clin2', 'csq2']
INFO_KEYS = ['n_bonds', 'n_bridges', 'n_rings', 'n_arom', 'n_l1', 'n_l2']


def molecule_features(spectra, cand_levels):
    """spectra: list of dict(mz, it, adduct, mode). cand_levels: list of frag_levels() results (or None).
    Returns a float array [n_cand, len(FEATURES)] (NaN rows for unparsable candidates)."""
    preps = []
    for s in spectra:
        p = prep(s['mz'], s['it'], s['adduct'], s['mode'])
        if p is None:
            continue
        preps.append((p, [prep(s['mz'], s['it'], s['adduct'], s['mode'], o) for o in DECOY_OFFSETS]))
    out = np.full((len(cand_levels), len(FEATURES)), np.nan)
    if not preps:
        return out
    for c, cl in enumerate(cand_levels):
        if cl is None:
            continue
        levels, info = cl
        rows = [spectrum_features(levels, p, pd_) for p, pd_ in preps]
        v = []
        for k in SPEC_KEYS:
            a = np.array([r[k] for r in rows])
            v += [a.mean(), a.max()]
        v += [info[k] for k in INFO_KEYS]
        out[c] = v
    return out


FEATURES = [f'{k}_{a}' for k in SPEC_KEYS for a in ('mean', 'max')] + INFO_KEYS


# ---- model inputs: raw features plus within-group z-scores and rank percentiles (same function offline and on Kaggle) ----
GROUPED = [f for f in FEATURES if f not in INFO_KEYS] + ['n_l1', 'n_l2']
MODEL_FEATURES = [f for f in FEATURES if f not in INFO_KEYS] + ['z_' + f for f in GROUPED] + ['r_' + f for f in GROUPED]


def _rank_pct(v):
    """pandas rank(method='average', pct=True) for one column without NaN."""
    order = np.argsort(v, kind='mergesort'); r = np.empty(len(v)); i = 0
    sv = v[order]
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r / len(v)


def model_matrix(X):
    """X: [n, len(FEATURES)] raw rows of ONE candidate group (no NaN rows). Returns [n, len(MODEL_FEATURES)]."""
    col = {f: X[:, k] for k, f in enumerate(FEATURES)}
    out = [col[f] for f in FEATURES if f not in INFO_KEYS]
    zs, rs = [], []
    for f in GROUPED:
        v = col[f]
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        zs.append((v - v.mean()) / (sd + 1e-6)); rs.append(_rank_pct(v))
    return np.column_stack(out + zs + rs)
