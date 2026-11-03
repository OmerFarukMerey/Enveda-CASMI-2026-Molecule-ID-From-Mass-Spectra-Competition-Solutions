"""Spectral analog edits (idea: heliosli's r3p overlay; this implementation is ours).

A query whose spectrum matches a library spectrum after shifting by a known natural-product modification (modified
cosine) is probably that library structure with the modification at one site. We enumerate every single-site edit of
the analog for that modification and rank the sites by fragment evidence: matched peaks that moved by the modification
mass should come from analog fragments that contain the site, unmoved ones from fragments that do not.

    q = query arrays (mz, it, n, prec, group);  L = library (from build_lib.py)
    best = analog_search(q, L, exclude_sidx)                 -> (score, lib row, kind, direction, delta)
    prods = edits(analog_smiles, kind, direction)            -> {key14: (smiles, [site, ...])}
    ranked = rank_sites(analog_mol, prods, analog peaks, query peaks, delta, group)
"""
from itertools import combinations
import numpy as np
try:
    import numba
    _njit = numba.njit(cache=False)
except ImportError:  # pure Python fallback (slow)
    def _njit(f):
        return f
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')
GROUPS = ['[M+H]+', '[M-H]-', '[M+Na]+']
H, E, NA = 1.00782503207, 0.00054858, 22.98976928

OH = '[OX2H1;!$(O[S,P])]'
NH = '[NX3;H1,H2;!$(NC=O);!$(N=*);!a]'
CH_AR = '[cH1]'
CH_SP3 = '[CX4;H1,H2,H3]'
# kind: (net mass change, [(site SMARTS, group SMILES bonded to the site through its first atom), ...])
MODS = {
    'CH2': (14.015650, [(OH, 'C'), (NH, 'C'), (CH_AR, 'C')]),
    'O': (15.994915, [(CH_AR, 'O'), (CH_SP3, 'O')]),
    'CH2O': (30.010565, [(CH_AR, 'OC')]),
    'C2H2O': (42.010565, [(OH, 'C(C)=O'), (NH, 'C(C)=O')]),
    'CO2': (43.989829, [(CH_AR, 'C(=O)O')]),
    'C5H8': (68.062600, [(CH_AR, 'CC=C(C)C'), (OH, 'CC=C(C)C')]),
    'SO3': (79.956815, [(OH, 'S(=O)(=O)O')]),
    'C3H2O3': (86.000394, [(OH, 'C(=O)CC(=O)O')]),
    'C7H4O': (104.026215, [(OH, 'C(=O)c1ccccc1')]),
    'C5H8O4': (132.042259, [(OH, 'C1OCC(O)C(O)C1O')]),
    'C6H10O4': (146.057909, [(OH, 'C1OC(C)C(O)C(O)C1O')]),
    'C9H6O2': (146.036779, [(OH, 'C(=O)C=Cc1ccc(O)cc1')]),
    'C7H4O4': (152.010959, [(OH, 'C(=O)c1cc(O)c(O)c(O)c1')]),
    'C6H10O5': (162.052824, [(OH, 'C1OC(CO)C(O)C(O)C1O'), (CH_AR, 'C1OC(CO)C(O)C(O)C1O')]),
    'C9H6O3': (162.031694, [(OH, 'C(=O)C=Cc1ccc(O)c(O)c1')]),
    'C6H8O6': (176.032088, [(OH, 'C1OC(C(=O)O)C(O)C(O)C1O')]),
    'C10H8O3': (176.047344, [(OH, 'C(=O)C=Cc1ccc(O)c(OC)c1')]),
    'C11H10O4': (206.057909, [(OH, 'C(=O)C=Cc1cc(OC)c(O)c(OC)c1')]),
    'C12H20O10': (324.105647, [(OH, 'C1OC(COC2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O')]),
}
PATT = {s: Chem.MolFromSmarts(s) for v in MODS.values() for s, _ in v[1]}
GRP = {g: Chem.MolFromSmiles(g) for v in MODS.values() for _, g in v[1]}
TARGETS = sorted([(k, v[0], 'add') for k, v in MODS.items()] + [(k, -v[0], 'remove') for k, v in MODS.items()],
                 key=lambda t: t[1])
PT = Chem.GetPeriodicTable()


@_njit
def modcos(qm, qi, nq, lm, li, nl, shift, tol):
    """Modified cosine: greedy one-to-one matching of peaks equal or shifted by `shift` (library + shift = query)."""
    if nq == 0 or nl == 0:
        return 0.0
    cs = np.empty(nq * nl * 2); ci = np.empty(nq * nl * 2, np.int64); cj = np.empty(nq * nl * 2, np.int64)
    c = 0
    for i in range(nq):
        for j in range(nl):
            d = qm[i] - lm[j]
            if abs(d) <= tol or abs(d - shift) <= tol:
                cs[c] = qi[i] * li[j]; ci[c] = i; cj[c] = j; c += 1
    if c == 0:
        return 0.0
    order = np.argsort(-cs[:c])
    ui = np.zeros(nq, np.bool_); uj = np.zeros(nl, np.bool_)
    num = 0.0
    for k in range(c):
        o = order[k]
        if ui[ci[o]] or uj[cj[o]]:
            continue
        ui[ci[o]] = True; uj[cj[o]] = True; num += cs[o]
    a = 0.0
    for i in range(nq):
        a += qi[i] * qi[i]
    b = 0.0
    for j in range(nl):
        b += li[j] * li[j]
    return num / np.sqrt(a * b)


class Library:
    def __init__(self, npz, struct):
        Z = npz if isinstance(npz, dict) else np.load(npz)
        self.mz, self.it, self.n, self.prec = Z['pk_mz'], Z['pk_it'].astype(np.float64), Z['pk_n'], Z['prec']
        self.grp, self.sidx = Z['grp'], Z['sidx']
        self.keys = struct.key.to_numpy(); self.smiles = struct.smiles.to_numpy()
        self.by = {}
        for g in range(len(GROUPS)):
            rows = np.flatnonzero(self.grp == g); o = np.argsort(self.prec[rows], kind='stable')
            self.by[g] = (rows[o], self.prec[rows[o]])


def analog_search(qm, qi, nq, qprec, g, L, exclude=(), prec_tol=0.01, tol=0.01, min_peaks=3):
    """Best (score, row, kind, direction, delta) over every modification and library spectrum of group g."""
    best = (0.0, -1, None, None, 0.0)
    if nq < min_peaks:
        return best
    rows, precs = L.by[g]
    ex = set(exclude)
    for kind, delta, direction in TARGETS:
        lo, hi = np.searchsorted(precs, [qprec - delta - prec_tol, qprec - delta + prec_tol])
        for r in rows[lo:hi]:
            if L.n[r] < min_peaks or L.sidx[r] in ex:
                continue
            s = modcos(qm, qi, nq, L.mz[r], L.it[r], L.n[r], delta, tol)
            if s > best[0]:
                best = (s, int(r), kind, direction, float(delta))
    return best


def key14(m):
    try:
        return Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None


def edits(smiles, kind, direction, max_sites=200):
    """Every single-site edit of `smiles` for `kind`: {key14: (smiles, [site, ...])}. A site is ('a', atom index) for
    an addition and ('r', frozenset of removed atom indices) for a removal. Products are checked by exact mass."""
    m = Chem.MolFromSmiles(smiles)
    out = {}
    if m is None:
        return out
    mass, rules = MODS[kind]
    mw = Descriptors.ExactMolWt(m)
    if direction == 'add':
        for smarts, group in rules:
            p, gm = PATT[smarts], GRP[group]
            for match in m.GetSubstructMatches(p, maxMatches=max_sites):
                site = match[0]
                rw = Chem.RWMol(Chem.CombineMols(m, gm))
                rw.AddBond(site, m.GetNumAtoms(), Chem.BondType.SINGLE)
                a = rw.GetAtomWithIdx(site)
                if a.GetNumExplicitHs():
                    a.SetNumExplicitHs(a.GetNumExplicitHs() - 1)
                a.SetNoImplicit(False)
                try:
                    Chem.SanitizeMol(rw)
                except Exception:
                    continue
                if abs(Descriptors.ExactMolWt(rw) - mw - mass) > 0.002:
                    continue
                k = key14(rw)
                if k:
                    out.setdefault(k, (Chem.MolToSmiles(rw), []))[1].append(('a', site))
    else:
        n = m.GetNumAtoms()
        for b in m.GetBonds():
            if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
                continue
            rw = Chem.RWMol(m)
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            rw.RemoveBond(i, j)
            for x in (i, j):
                at = rw.GetAtomWithIdx(x)
                at.SetNoImplicit(False)
                if at.GetIsAromatic() and at.GetSymbol() == 'N':
                    at.SetNumExplicitHs(at.GetNumExplicitHs() + 1)
            frags = Chem.GetMolFrags(rw, asMols=False, sanitizeFrags=False)
            if len(frags) != 2:
                continue
            for keep_atoms in frags:
                removed = frozenset(range(n)) - frozenset(keep_atoms)
                sub = Chem.RWMol(rw)
                for x in sorted(removed, reverse=True):
                    sub.RemoveAtom(x)
                try:
                    Chem.SanitizeMol(sub)
                except Exception:
                    continue
                if abs(mw - Descriptors.ExactMolWt(sub) - mass) > 0.002:
                    continue
                k = key14(sub)
                if k:
                    out.setdefault(k, (Chem.MolToSmiles(sub), []))[1].append(('r', removed))
    return out


def fragments(m, max_bonds=60):
    """Atom sets left after cutting one or two bonds: [(bitmask, neutral mass, cuts)] plus the whole molecule."""
    n = m.GetNumAtoms()
    am = [PT.GetMostCommonIsotopeMass(a.GetAtomicNum()) + a.GetTotalNumHs() * H for a in m.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()][:max_bonds]
    res = {(1 << n) - 1: (sum(am), 0)}
    for k in (1, 2):
        for rem in combinations(range(len(bonds)), k):
            par = list(range(n))

            def f(x):
                while par[x] != x:
                    par[x] = par[par[x]]; x = par[x]
                return x
            for bi, (a, b) in enumerate(bonds):
                if bi not in rem:
                    ra, rb = f(a), f(b)
                    if ra != rb:
                        par[ra] = rb
            comp = {}
            for a in range(n):
                comp.setdefault(f(a), []).append(a)
            if len(comp) == 1:
                continue
            for c in comp.values():
                mask = 0
                for a in c:
                    mask |= 1 << a
                if mask not in res:
                    res[mask] = (sum(am[a] for a in c), k)
    masks = list(res)
    return masks, np.array([res[x][0] for x in masks]), np.array([res[x][1] for x in masks])


def ion_offsets(g):
    if g == 0:
        return [h * H - E for h in range(-2, 4)]
    if g == 1:
        return [h * H + E for h in range(-3, 3)]
    return [h * H - E for h in range(-2, 4)] + [NA - E + h * H for h in range(-2, 2)]


def annotate(mz, fr, g, tol=0.01):
    """For each peak, the fragment masks that explain it with the fewest cuts."""
    masks, mass, cuts = fr
    offs = np.array(ion_offsets(g))
    out = []
    for x in mz:
        hit = np.flatnonzero((np.abs(mass[:, None] + offs[None, :] - x) <= tol).any(1))
        if not len(hit):
            out.append([]); continue
        c = cuts[hit].min()
        out.append([masks[i] for i in hit if cuts[i] == c])
    return out


def align(qm, qi, am, ai, delta, tol=0.01):
    """Greedy one-to-one query / analog peak pairs: (analog peak index, shifted?, weight)."""
    cand = []
    for i in range(len(qm)):
        for j in range(len(am)):
            w = float(qi[i] * ai[j])
            if abs(qm[i] - am[j]) <= tol:
                cand.append((w, i, j, False))
            elif abs(qm[i] - am[j] - delta) <= tol:
                cand.append((w, i, j, True))
    cand.sort(key=lambda x: -x[0])
    ui, uj, out = set(), set(), []
    for w, i, j, sh in cand:
        if i in ui or j in uj:
            continue
        ui.add(i); uj.add(j); out.append((j, sh, w))
    return out


def site_score(site, pairs, ann):
    """Weighted share of matched peaks consistent with the site: moved peaks from fragments that hold the site,
    unmoved peaks from fragments without it. Uninformative peaks (no fragment, or every fragment both ways) count 0.5."""
    num = tot = 0.0
    for j, sh, w in pairs:
        F = ann[j]
        if not F:
            continue
        if site[0] == 'a':
            bit = 1 << site[1]
            inside = sum(1 for f in F if f & bit) / len(F)
            outside = 1 - inside
        else:
            gm = 0
            for a in site[1]:
                gm |= 1 << a
            inside = sum(1 for f in F if f & gm == gm) / len(F)
            outside = sum(1 for f in F if f & gm == 0) / len(F)
        num += w * (inside if sh else outside); tot += w
    return num / tot if tot > 0 else 0.5


def rank_products(analog_smiles, prods, a_mz, a_it, q_mz, q_it, delta, g):
    """Products sorted by their best site's score: [(score, key14, smiles)]."""
    m = Chem.MolFromSmiles(analog_smiles)
    fr = fragments(m)
    ann = annotate(a_mz, fr, g)
    pairs = align(q_mz, q_it, a_mz, a_it, delta)
    out = [(max(site_score(s, pairs, ann) for s in sites), k, smi) for k, (smi, sites) in prods.items()]
    return sorted(out, key=lambda t: -t[0])


FEATS = ['score', 'abs_delta', 'is_add', 'n_qspec', 'n_spec_hit', 'mean_score', 'second', 'top_score', 'n_prod', 'site1',
         'site_margin', 'prec', 'grp', 'nq', 'na', 'n_match', 'n_shift', 'w_shift', 'w_unshift', 'cand_rank']
NPK, PER, TOPK, MIN_SCORE = 32, 3, 5, 0.6


def prep_peaks(mz, it, prec):
    """Library preprocessing: finite positive peaks at or below the precursor + 1 Da, top 32 by intensity,
    sqrt(intensity / max). None if fewer than 3 peaks."""
    mz, it = np.asarray(mz, np.float64), np.asarray(it, np.float64)
    if len(mz) != len(it) or not len(mz) or not np.isfinite(prec):
        return None
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz <= prec + 1.0)
    mz, it = mz[ok], it[ok]
    if len(mz) < 3:
        return None
    o = np.argsort(-it, kind='stable')[:NPK]
    return mz[o], np.sqrt(it[o] / it[o].max())


def search_all(qm, qi, nq, qprec, g, L, exclude=-1, prec_tol=0.01, tol=0.01):
    """Every (score, row, kind, direction, delta) with modified cosine >= MIN_SCORE."""
    rows, precs = L.by[g]
    out = []
    for kind, delta, direction in TARGETS:
        lo, hi = np.searchsorted(precs, [qprec - delta - prec_tol, qprec - delta + prec_tol])
        for r in rows[lo:hi]:
            if L.n[r] < 3 or L.sidx[r] == exclude:
                continue
            s = modcos(qm, qi, nq, L.mz[r], L.it[r], L.n[r], delta, tol)
            if s >= MIN_SCORE:
                out.append((s, int(r), kind, direction, float(delta)))
    return out


def pair_stats(qm, qi, am, ai, delta):
    pairs = align(qm, qi, am, ai, delta)
    tot = float(np.sqrt((qi ** 2).sum() * (ai ** 2).sum()))
    sh = [w for _, s, w in pairs if s]; un = [w for _, s, w in pairs if not s]
    return dict(n_match=len(pairs), n_shift=len(sh), w_shift=sum(sh) / tot, w_unshift=sum(un) / tot)


def molecule_candidates(queries, L, exclude=-1, drop_keys=frozenset(), keep_key=None):
    """queries: [(mz, it, prec, group)] (prep_peaks output, at most PER per group). Rows for the top TOPK analogs
    (distinct structure x modification) that keep at least one edit after dropping drop_keys (keep_key stays)."""
    hits = {}
    for qi_, (qm, qi, qp, g) in enumerate(queries):
        for sc, r, kind, direction, delta in search_all(qm, qi, len(qm), qp, g, L, exclude):
            k = (int(L.sidx[r]), kind, direction)
            h = hits.setdefault(k, dict(best=0.0, spectra=set(), scores=[]))
            h['scores'].append(sc); h['spectra'].add(qi_)
            if sc > h['best']:
                h.update(best=sc, row=r, q=qi_, delta=delta)
    out = []
    ranked = sorted(hits.items(), key=lambda kv: -kv[1]['best'])
    for rank, ((a, kind, direction), h) in enumerate(ranked[:TOPK]):
        prods = {k: v for k, v in edits(L.smiles[a], kind, direction).items() if k not in drop_keys or k == keep_key}
        if not prods:
            continue
        qm, qi, qp, g = queries[h['q']]; ar = h['row']
        am, ai = L.mz[ar][:L.n[ar]], L.it[ar][:L.n[ar]]
        try:
            rp = rank_products(L.smiles[a], prods, am, ai, qm, qi, h['delta'], g)
        except Exception:  # noqa: BLE001
            continue
        out.append(dict(cand_rank=rank, analog=L.keys[a], kind=kind, direction=direction, is_add=int(direction == 'add'),
                        abs_delta=abs(h['delta']), score=h['best'], n_qspec=len(queries), n_spec_hit=len(h['spectra']),
                        mean_score=float(np.mean(h['scores'])), second=ranked[rank + 1][1]['best'] if rank + 1 < len(ranked) else 0.0,
                        top_score=ranked[0][1]['best'], n_prod=len(prods), site1=rp[0][0],
                        site_margin=rp[0][0] - (rp[1][0] if len(rp) > 1 else 0.0), prec=float(qp), grp=int(g),
                        nq=len(qm), na=int(L.n[ar]), **pair_stats(qm, qi, am, ai, h['delta']),
                        pick=rp[0][1], pick_smiles=rp[0][2], prods=[k for _, k, _ in rp]))
    return out


def build_library(train_path, log=print):
    """build_lib.py in memory: ({pk_mz, pk_it, pk_n, prec, grp, sidx}, DataFrame key / smiles, set of every training key)."""
    import pandas as pd
    import pyarrow.parquet as pq
    f = pq.ParquetFile(train_path)
    cols = ['inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']
    keys, smiles, all_keys, recs = {}, [], set(), []
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=cols)
        all_keys.update(t.column('inchikey14').to_pylist())
        ad = t.column('adduct').to_pylist()
        keep = np.array([a in GROUPS for a in ad])
        if not keep.any():
            continue
        t = t.filter(keep)
        ad = t.column('adduct').to_pylist(); ik = t.column('inchikey14').to_pylist(); sm = t.column('normalized_smiles').to_pylist()
        pr = t.column('precursor_mz').to_numpy()
        mzc, itc = t.column('ms2_mzs').combine_chunks(), t.column('ms2_normalized_intensities').combine_chunks()
        mzo, ito = mzc.offsets.to_numpy(), itc.offsets.to_numpy()
        mzv, itv = mzc.values.to_numpy(zero_copy_only=False), itc.values.to_numpy(zero_copy_only=False)
        for i in range(len(ad)):
            pk = prep_peaks(mzv[mzo[i]:mzo[i + 1]], itv[ito[i]:ito[i + 1]], pr[i])
            if pk is None:
                continue
            k = ik[i]
            if k not in keys:
                keys[k] = len(smiles); smiles.append(sm[i])
            mzp = np.zeros(NPK, np.float64); itp = np.zeros(NPK, np.float32); mzp[:len(pk[0])] = pk[0]; itp[:len(pk[0])] = pk[1]
            recs.append((keys[k], GROUPS.index(ad[i]), len(pk[0]), float(pr[i]), mzp, itp))
        log(f'library row group {rg + 1}/{f.num_row_groups}: {len(recs):,} spectra')
        del t, mzc, itc, mzv, itv
    sidx = np.array([r[0] for r in recs], np.int32); grp = np.array([r[1] for r in recs], np.int8)
    npk = np.array([r[2] for r in recs], np.int16)
    order = np.lexsort((-npk, grp, sidx))
    pair = sidx[order].astype(np.int64) * 4 + grp[order]
    first = np.r_[0, np.flatnonzero(np.diff(pair)) + 1]
    rank = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
    sel = np.sort(order[rank < PER])
    Z = dict(pk_mz=np.stack([recs[i][4] for i in sel]), pk_it=np.stack([recs[i][5] for i in sel]), pk_n=npk[sel],
             prec=np.array([recs[i][3] for i in sel]), grp=grp[sel], sidx=sidx[sel])
    return Z, pd.DataFrame(dict(key=list(keys), smiles=smiles)), all_keys


def test_queries(rows):
    """rows: one molecule's test spectra as (adduct, precursor_mz, mzs, intensities). At most PER per group, most peaks first."""
    by = {}
    for ad, prec, mz, it in rows:
        if ad not in GROUPS:
            continue
        pk = prep_peaks(mz, it, float(prec))
        if pk is not None:
            by.setdefault(GROUPS.index(ad), []).append((pk[0], pk[1], float(prec)))
    out = []
    for g in sorted(by):
        for m, i, p in sorted(by[g], key=lambda x: -len(x[0]))[:PER]:
            out.append((m, i, p, g))
    return out
