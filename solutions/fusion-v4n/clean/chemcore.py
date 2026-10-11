"""Own chemistry core for the clean chain: adduct parsing, neutral mass, spectrum cleaning, bond-cleavage fragment masses.

Written from the behaviour contract in clean/SPEC.md (sections 2 to 4); no third-party code is imported except NumPy and
RDKit (BSD-3-Clause). Element masses come from RDKit's periodic table, the electron mass is the CODATA value.
"""
import re

import numpy as np

ELECTRON = 0.00054857990907
_PT = {}


def mono(el):
    """Monoisotopic mass of an element symbol (most common isotope), cached."""
    if el not in _PT:
        from rdkit import Chem
        pt = Chem.GetPeriodicTable()
        _PT[el] = 2.01410177812 if el == 'D' else float(pt.GetMostCommonIsotopeMass(el))
    return _PT[el]


_TOK = re.compile(r'([A-Z][a-z]?)(\d*)')
_GRP = re.compile(r'([+-])(\d*)([A-Za-z][A-Za-z0-9]*)')
_ADD = re.compile(r'^\[(\d*)M((?:[+-]\d*[A-Za-z][A-Za-z0-9]*)*)\](\d*)([+-])$')


def formula_mass(f):
    """Monoisotopic mass of a plain formula string such as 'CH2O2' (no brackets, no charges)."""
    pos, m = 0, 0.0
    for t in _TOK.finditer(f):
        if t.start() != pos:
            raise ValueError(f)
        pos = t.end(); m += mono(t.group(1)) * (int(t.group(2)) if t.group(2) else 1)
    if pos != len(f) or not f:
        raise ValueError(f)
    return m


def parse_adduct(a):
    """'[nM+X-Y]z+' -> (n, z, delta, sign) with  m/z * z = n * M + delta  (delta includes the electron correction)."""
    m = _ADD.match(a)
    if not m:
        return None
    n = int(m.group(1)) if m.group(1) else 1; z = int(m.group(3)) if m.group(3) else 1; sign = 1 if m.group(4) == '+' else -1
    d = 0.0
    for sg, k, f in _GRP.findall(m.group(2)):
        d += (1 if sg == '+' else -1) * (int(k) if k else 1) * formula_mass(f)
    return n, z, d - sign * z * ELECTRON, sign


def neutral_mass(mz, adduct):
    """Neutral monoisotopic mass of M from a precursor m/z and an adduct string; NaN for an adduct that does not parse."""
    p = parse_adduct(adduct)
    return float('nan') if p is None else (float(mz) * p[1] - p[2]) / p[0]


def clean_peaks(mz, it, prec_mz, floor=0.001, topk=512, over=2.0):
    """Library-cache cleaning of one spectrum -> (m/z float32 ascending, intensity float32 with base peak 1).

    Kept: peaks with 0 < m/z <= precursor + `over` and intensity >= `floor` x the largest kept-range intensity; if more
    than `topk` remain, the `topk` most intense; empty if no positive intensity is in range.
    """
    mz = np.asarray(mz, np.float64); it = np.asarray(it, np.float64)
    inr = mz <= prec_mz + over
    if not inr.any() or it[inr].max() <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = it[inr].max()
    idx = np.flatnonzero(inr & (it >= floor * mx) & (mz > 0))
    if len(idx) > topk:
        idx = idx[np.argsort(it[idx], kind='stable')[len(idx) - topk:]]
    o = np.argsort(mz[idx], kind='stable')
    return mz[idx][o].astype(np.float32), (it[idx] / mx).astype(np.float32)[o]


def fragment_masses(smiles, max_depth=2):
    """Sorted unique masses (rounded to 0.001 Da) of the molecule and of every connected piece left after removing one
    or two bonds of the heavy-atom graph (any bond, ring bonds included); each atom carries its hydrogens."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(0)
    n = m.GetNumAtoms()
    w = np.array([mono(a.GetSymbol()) + a.GetTotalNumHs() * mono('H') for a in m.GetAtoms()])
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    out = [w.sum()]
    nbr = [[] for _ in range(n)]
    for e, (a, b) in enumerate(bonds):
        nbr[a].append((b, e)); nbr[b].append((a, e))

    def pieces(d1, d2):
        comp = [-1] * n; sums = []
        for s in range(n):
            if comp[s] >= 0:
                continue
            c = len(sums); comp[s] = c; tot = 0.0; st = [s]
            while st:
                u = st.pop(); tot += w[u]
                for v, e in nbr[u]:
                    if e != d1 and e != d2 and comp[v] < 0:
                        comp[v] = c; st.append(v)
            sums.append(tot)
        return sums
    E = len(bonds)
    for e1 in range(E):
        p = pieces(e1, -1)
        if len(p) > 1:
            out += p
        if max_depth >= 2:
            for e2 in range(e1 + 1, E):
                p = pieces(e1, e2)
                if len(p) > 1:
                    out += p
    return np.unique(np.floor(np.array(out) * 1000.0 + 0.5) / 1000.0)
