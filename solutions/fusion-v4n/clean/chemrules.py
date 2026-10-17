"""Knowledge-driven MS/MS chemistry rules that re-order same-formula candidates (no training, a few hand-set weights).

Licence: ours. Written from textbook mass-spectrometry knowledge (diagnostic neutral losses and ions of glycosides, acyl
esters, sulfates, phosphates, choline lipids, amino acids; adduct chemistry); only NumPy, RDKit and clean/chemcore.py.

Idea. A rule pairs SPECTRAL EVIDENCE (a diagnostic loss from the precursor, a diagnostic ion, the precursor adduct
itself) with a STRUCTURAL REQUIREMENT on the candidate (a SMARTS-level group). Evidence is pooled over all spectra of
the molecule (cross-spectrum: present in any spectrum; absent only if absent in every informative one). A candidate that
does not meet the requirement of a rule whose evidence is present gets the rule's violation weight; a candidate with a
labile group whose expected loss is absent everywhere gets a smaller weight. A rule counts inside a same-formula group
only when it separates the group (some members meet it, some do not). The re-order permutes each same-formula group
among its own slots in the first `top` rows, by violation tier (floor((v - v_min) / theta)) then original order, so it
acts only when the evidence difference reaches `theta` and never moves rows of other formulas.

    profile(smiles) -> dict | None                 structural facts of one candidate (cached by the caller)
    evidence(spectra, cfg=None) -> dict            rule -> strength in [0, 1] (0 = no evidence), plus 'informative'
    violations(prof, ev, cfg=None) -> dict         rule -> weight for one candidate
    reorder(rows, spectra, cfg=None, cache=None) -> (new_rows, info)

`spectra`: list of dicts {mz, it (base peak = 1), adduct, prec, mode (+1/-1)}. `rows`: ranked SMILES. info holds, per
same-formula group that changed, the rules that separated it. Rules (keys of RULES): see the table below.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

H = 1.00782503207
E = 0.00054857990907
PROTON = H - E
M = dict(C=12.0, H=H, N=14.0030740048, O=15.9949146196, P=30.97376163, S=31.97207100)


def fmass(**k):
    return sum(M[e] * n for e, n in k.items())


# diagnostic neutral losses (neutral formulas)
L_HEX = fmass(C=6, H=10, O=5)          # 162.0528 hexose
L_DHEX = fmass(C=6, H=10, O=4)         # 146.0579 deoxyhexose
L_PENT = fmass(C=5, H=8, O=4)          # 132.0423 pentose
L_URON = fmass(C=6, H=8, O=6)          # 176.0321 uronic acid
L_CO2 = fmass(C=1, O=2)                # 43.9898
L_CH3 = fmass(C=1, H=3)                # 15.0235 methyl radical
L_NH3 = fmass(N=1, H=3)                # 17.0265
L_SO3 = fmass(S=1, O=3)                # 79.9568
L_H2SO4 = fmass(S=1, O=4, H=2)         # 97.9674
L_HPO3 = fmass(H=1, P=1, O=3)          # 79.9663
L_H3PO4 = fmass(H=3, P=1, O=4)         # 97.9769
L_TMA = fmass(C=3, H=9, N=1)           # 59.0735 trimethylamine
L_CGLY = (fmass(C=4, H=8, O=4), fmass(C=3, H=6, O=3))   # 120.0423, 90.0317 cross-ring cleavage of C-hexosides
L_H2O = fmass(H=2, O=1)
L_HCOOH = fmass(C=1, H=2, O=2)
L_HCL = fmass(H=1) + 34.96885268
L_NH3_ADD = L_NH3
# named acyl units: ketene (acyl - H) losses; the acid (ketene + H2O) is accepted as the same evidence
ACYLS = {
    'galloyl': fmass(C=7, H=4, O=4), 'coumaroyl': fmass(C=9, H=6, O=2), 'caffeoyl': fmass(C=9, H=6, O=3),
    'feruloyl': fmass(C=10, H=8, O=3), 'sinapoyl': fmass(C=11, H=10, O=4), 'malonyl': fmass(C=3, H=2, O=3),
    'benzoyl': fmass(C=7, H=4, O=1), 'cinnamoyl': fmass(C=9, H=6, O=1), 'quinoyl': fmass(C=7, H=10, O=5),
}
# diagnostic ions (m/z of the singly charged ion), mode, requirement key
IONS = {
    'oxo_hex': (fmass(C=6, H=11, O=5) - E, +1, 'gly_hex'),     # 163.0601 hexose oxonium (B1)
    'oxo_dhex': (fmass(C=6, H=11, O=4) - E, +1, 'gly_dhex'),   # 147.0652
    'oxo_pent': (fmass(C=5, H=9, O=4) - E, +1, 'gly_pent'),    # 133.0495
    'hso4': (fmass(H=1, S=1, O=4) + E, -1, 'sulfate'),        # 96.9601
    'h2po4': (fmass(H=2, P=1, O=4) + E, -1, 'phosphate'),     # 96.9696
    'po3': (fmass(P=1, O=3) + E, -1, 'phosphate'),            # 78.9591
    'choline': (fmass(C=5, H=14, N=1, O=1) - E, +1, 'choline'),        # 104.1070
    'pchol': (fmass(C=5, H=15, N=1, O=4, P=1) - E, +1, 'choline'),     # 184.0733
    'trop': (fmass(C=7, H=7) - E, +1, 'benzyl'),                       # 91.0542 tropylium
    'imm_phe': (fmass(C=8, H=10, N=1) - E, +1, 'aa_phe'),              # 120.0808
    'imm_tyr': (fmass(C=8, H=10, N=1, O=1) - E, +1, 'aa_tyr'),         # 136.0757
    'imm_trp': (fmass(C=10, H=11, N=2) - E, +1, 'aa_trp'),             # 159.0917
    'imm_his': (fmass(C=5, H=8, N=3) - E, +1, 'aa_his'),               # 110.0718
}

DEFAULT_CFG = dict(
    top=10,              # rows of the list that may move
    theta=1.0,           # violation difference needed for a candidate to drop a tier
    tol_da=0.005,        # m/z tolerance: tol_da + tol_ppm * m/z
    tol_ppm=5.0,
    weak=0.05,           # product / ion relative intensity counted as evidence
    strong=0.2,          # ... counted as strong evidence (weight 1; weak evidence weight w_weak)
    w_weak=0.5,
    w_absent=0.5,        # labile group present, its loss absent in every informative spectrum
    w_adduct=1.0,        # adduct-implied requirement violated
    xpk_int=0.3,         # unexplained-peak rule: peaks at least this intense must be explainable
    rules=None,          # None = DEFAULT_RULES; else an iterable of rule names (any of RULES)
)

RULES = ('L_HEX', 'L_DHEX', 'L_PENT', 'L_URON', 'L_CGLY', 'A_GLY', 'L_ACYL', 'A_ACYL', 'L_SULF', 'A_SULF', 'L_PHOS',
         'L_CO2N', 'L_CH3', 'L_NH3', 'L_TMA', 'I_OXO', 'I_CHOL', 'I_TROP', 'I_IMM', 'AD_H2O', 'AD_NEG', 'XPK')
# frozen 2026-10-17 on the development split (scratch/chemrules/PREREG.md): the six rules below lost there and are off by
# default -- I_TROP (C7H7+ also from indoles, terpenoids, phenethyl groups), I_IMM (immonium m/z shared by other CnHmN+
# ions), L_NH3 (secondary ring amines lose NH3 too), AD_H2O / AD_NEG (library adduct labels follow the mass, not the
# chemistry: guaiacol as [M-H2O+H]+, lactones and sulfones as [M-H]-), XPK (hard unexplained-peak veto = MetFrag-lite
# as a gate: misses rearrangement ions of the true structure)
DEFAULT_RULES = ('L_HEX', 'L_DHEX', 'L_PENT', 'L_URON', 'L_CGLY', 'A_GLY', 'I_OXO', 'L_ACYL', 'A_ACYL', 'L_SULF', 'A_SULF',
                 'L_PHOS', 'L_CO2N', 'L_CH3', 'L_TMA', 'I_CHOL')


def resolve_cfg(cfg=None):
    out = dict(DEFAULT_CFG)
    for k, v in (cfg or {}).items():
        if k not in out:
            raise KeyError(f'chemrules: unknown cfg key {k!r}')
        out[k] = v
    return out


# ---------------------------------------------------------------------------------------------------------------------
# candidate structure
# ---------------------------------------------------------------------------------------------------------------------
_PAT = {}


def _p(s):
    from rdkit import Chem
    if s not in _PAT:
        _PAT[s] = Chem.MolFromSmarts(s)
    return _PAT[s]


SMARTS = {
    'cooh': '[CX3](=O)[OX2H1,OX1-]',
    'carboxyl_any': '[CX3](=O)[OX2,OX1-]',            # acid, ester, lactone, carbonate, carbamate O side
    'me_het': '[CH3][O,N,S,n]',
    'nh2': '[NX3H2,NX4H3+]',
    'nme3': '[NX4+,NX3]([CH3])([CH3])[CH3]',
    'sulfate': '[#6,#7][OX2,NX3]S(=O)(=O)[OX2H1,OX1-]',
    'sulfonic': '[#6]S(=O)(=O)[OX2H1,OX1-]',
    'phosphate': '[OX2]P(=O)([OX2H1,OX1-,OX2])',
    'choline': '[CH3][NX4+]([CH3])([CH3])[CH2][CH2][OX2]',
    'aa_phe': '[cH]1[cH][cH][cH][cH]c1[CH2][CX4][NX3,NX4+]',
    'aa_tyr': '[OX2H1]c1[cH][cH]c([cH][cH]1)[CH2][CX4][NX3,NX4+]',
    'aa_trp': 'c1ccc2c(c1)c(c[nH]2)[CH2][CX4][NX3,NX4+]',
    'aa_his': 'c1nc[nH]c1[CH2][CX4][NX3,NX4+]',
    'aliph_oh': '[CX4][OX2H1]',
    'acidic_h': '[$([OX2H1]),$([SX2H1]),$([nH]),$([NX3H1,NX3H2][CX3,SX4]=O),$([NX3H1](c)),$([NX4+;!H0])]',
}


def _formula_counts(atoms_idx, mol):
    c = {}
    for i in atoms_idx:
        a = mol.GetAtomWithIdx(i)
        c[a.GetSymbol()] = c.get(a.GetSymbol(), 0) + 1
        c['H'] = c.get('H', 0) + a.GetTotalNumHs()
    return c


def _counts_mass(c):
    from chemcore import mono
    return sum(mono(e) * n for e, n in c.items())


def _component(mol, start, cut):
    """Atom indices reachable from `start` without crossing bond `cut` (a (i, j) pair)."""
    seen, st = {start}, [start]
    while st:
        u = st.pop()
        for nb in mol.GetAtomWithIdx(u).GetNeighbors():
            v = nb.GetIdx()
            if {u, v} == set(cut) or v in seen:
                continue
            seen.add(v); st.append(v)
    return seen


def _sugar_units(mol):
    """Pyranose / furanose units: list of (kind, link) with kind in hex/dhex/pent/uron/other and link O/C/N/free."""
    from rdkit import Chem
    out = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) not in (5, 6):
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if sum(a.GetSymbol() == 'O' for a in atoms) != 1 or sum(a.GetSymbol() == 'C' for a in atoms) != len(ring) - 1:
            continue
        if any(a.GetSymbol() == 'C' and (a.GetIsAromatic() or a.GetHybridization() != Chem.HybridizationType.SP3) for a in atoms):
            continue
        k = [a.GetSymbol() for a in atoms].index('O')
        path = list(ring[k:]) + list(ring[:k])           # ring O first, then the carbons in cyclic order
        rs = set(ring)

        def exo(i):
            return [n for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() not in rs]

        def has_exo_o(i):
            return any(n.GetSymbol() in ('O', 'N') for n in exo(i))
        best, best_sc = None, -1
        for p in (path[1:], path[1:][::-1]):
            ano, term, mid = p[0], p[-1], p[1:-1]
            ex_ano = exo(ano)
            link = None
            for n in ex_ano:
                if n.GetSymbol() == 'O':
                    link = 'O' if n.GetDegree() > 1 else ('free' if link is None else link)
                elif n.GetSymbol() == 'N':
                    link = 'N'
            if link is None:
                # C-glycoside: the anomeric carbon carries an aromatic (or carbonyl) carbon, not a CH2OH / CH3 / COOH
                for n in ex_ano:
                    if n.GetSymbol() == 'C' and (n.GetIsAromatic() or n.GetHybridization() == Chem.HybridizationType.SP2):
                        if not any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetSymbol() == 'O' and
                                   any(x.GetSymbol() == 'O' and x.GetIdx() != b.GetOtherAtom(n).GetIdx() for x in n.GetNeighbors())
                                   for b in n.GetBonds()):
                            link = 'C'
            if link is None:
                continue
            n_o = sum(has_exo_o(i) for i in mid)
            if (len(ring) == 6 and n_o < 2) or (len(ring) == 5 and n_o < 1):
                continue
            ex_t = [n for n in exo(term) if n.GetSymbol() == 'C']
            if not ex_t:
                kind = 'pent' if len(ring) == 6 else 'other'
            else:
                c = ex_t[0]
                nbo = [n for n in c.GetNeighbors() if n.GetIdx() != term]
                if c.GetTotalNumHs() == 3:
                    kind = 'dhex' if len(ring) == 6 else 'other'
                elif any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(c).GetSymbol() == 'O' for b in c.GetBonds()):
                    kind = 'uron' if len(ring) == 6 else 'other'
                elif c.GetTotalNumHs() == 2 and len(nbo) == 1 and nbo[0].GetSymbol() == 'O':
                    kind = 'hex' if len(ring) == 6 else 'pent'
                elif len(ring) == 5 and any(n.GetSymbol() == 'C' for n in nbo):
                    kind = 'hex'
                else:
                    kind = 'other'
            sc = {'O': 3, 'N': 3, 'free': 2, 'C': 1}[link] + (kind != 'other')
            if sc > best_sc:
                best, best_sc = (kind, link), sc
        if best is not None:
            out.append(best)
    return out


def _acyl_losses(mol):
    """Ketene-loss masses of every acyclic ester / amide acyl unit (cut C(=O)-X, X = O or N)."""
    out = []
    patt = _p('[CX3;!R](=O)-[OX2,NX3]')
    for c, o, x in mol.GetSubstructMatches(patt):
        side = _component(mol, c, (c, x))
        if x in side:
            continue
        cnt = _formula_counts(side, mol)
        out.append(_counts_mass(cnt) - H)            # acyl piece carries the carbonyl; ketene = acyl - H
    # quinic acid esters lose quinic acid - H2O (C7H10O5) from the alcohol side
    q = _p('[CX3](=O)O[CX4;R]')
    for c, o1, o, r in mol.GetSubstructMatches(q):
        side = _component(mol, r, (o, r))
        cnt = _formula_counts(side, mol)
        out.append(_counts_mass(cnt) - H)            # alcohol side as alkene (alcohol - H2O)
    return out


def profile(smiles):
    """Structural facts used by the rules, or None if the SMILES does not parse."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog('rdApp.*')
    try:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    except Exception:
        mol = None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    f = {k: len(mol.GetSubstructMatches(_p(s))) for k, s in SMARTS.items()}
    sug = _sugar_units(mol)
    pr = dict(formula=CalcMolFormula(mol), smiles=smiles, **f)
    for kind in ('hex', 'dhex', 'pent', 'uron'):
        pr['gly_' + kind] = sum(1 for k, l in sug if k == kind and l == 'O')
    pr['gly_O'] = sum(1 for k, l in sug if l == 'O')
    pr['gly_C'] = sum(1 for k, l in sug if l == 'C')
    pr['acyl'] = _acyl_losses(mol)
    pr['benzyl'] = int(_benzyl(mol))
    pr['sulf_any'] = f['sulfate'] + f['sulfonic']
    pr['h2o_sites'] = f['aliph_oh'] + f['cooh']
    pr['_mol'] = mol
    return pr


def _benzyl(mol):
    """A benzene ring with only carbon substituents (at most two), one of them sp3: the C7H7+ (tropylium) source."""
    from rdkit import Chem
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        at = [mol.GetAtomWithIdx(i) for i in ring]
        if not all(a.GetSymbol() == 'C' and a.GetIsAromatic() for a in at):
            continue
        rs = set(ring); subs = []
        for a in at:
            for n in a.GetNeighbors():
                if n.GetIdx() not in rs:
                    subs.append(n)
        if not subs or len(subs) > 2 or any(n.GetSymbol() != 'C' for n in subs):
            continue
        # fused ring systems are not benzyl sources
        if any(mol.GetRingInfo().NumAtomRings(i) > 1 for i in ring):
            continue
        if any(n.GetHybridization() == Chem.HybridizationType.SP3 for n in subs):
            return True
    return False


# ---------------------------------------------------------------------------------------------------------------------
# spectral evidence
# ---------------------------------------------------------------------------------------------------------------------
def _prep(s):
    mz = np.asarray(s.get('mz', ()), np.float64).ravel(); it = np.asarray(s.get('it', ()), np.float64).ravel()
    n = min(len(mz), len(it)); mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[ok], it[ok]
    if len(mz) == 0:
        return None
    it = it / it.max()
    prec = float(s.get('prec') or 0.0)
    # low-resolution spectra (most peaks at <= 2 decimals) carry no usable mass differences
    if np.mean(np.abs(mz * 100 - np.round(mz * 100)) < 1e-6) > 0.5:
        return None
    # measured precursor if present
    if prec > 0:
        near = np.abs(mz - prec) < 0.02
        if near.any():
            prec = float(mz[near][np.argmax(it[near])])
    add = s.get('adduct') or ''
    mode = s.get('mode') or (1 if add.endswith('+') else -1)
    refs = [prec]
    if add == '[M+NH4]+':
        refs.append(prec - L_NH3_ADD)
    elif add == '[M+CH2O2-H]-':
        refs.append(prec - L_HCOOH)
    elif add == '[M+Cl]-':
        refs.append(prec - L_HCL)
    frag = mz < (prec - 1.5) if prec > 0 else np.ones(len(mz), bool)
    informative = bool((it[frag] >= 0.05).any())
    return dict(mz=mz, it=it, prec=prec, refs=refs, add=add, mode=int(np.sign(mode)), informative=informative)


def _int_at(p, target, cfg):
    tol = cfg['tol_da'] + cfg['tol_ppm'] * 1e-6 * abs(target)
    k = np.abs(p['mz'] - target) <= tol
    return float(p['it'][k].max()) if k.any() else 0.0


def _loss_int(p, losses, cfg, chain=()):
    """Max relative intensity of a product ref - L (L in losses) over the reference ions (plus a sugar-loss chain)."""
    best = 0.0
    refs = list(p['refs'])
    if chain:
        frontier = list(refs)
        for _ in range(3):
            nxt = []
            for r in frontier:
                for L in chain:
                    if _int_at(p, r - L, cfg) >= cfg['weak']:
                        nxt.append(r - L)
            refs += nxt; frontier = nxt
            if not frontier:
                break
    for r in refs:
        for L in losses:
            if r - L > 40:
                best = max(best, _int_at(p, r - L, cfg))
    return best


SUGAR_L = (L_HEX, L_DHEX, L_PENT, L_URON)


def evidence(spectra, cfg=None):
    """rule key -> evidence strength (max product / ion intensity over the molecule's spectra; 0 = none)."""
    c = resolve_cfg(cfg)
    P = [p for p in (_prep(s) for s in (spectra or [])) if p is not None]
    ev = {'informative': any(p['informative'] for p in P), 'n': len(P)}

    def mx(fn):
        return max([fn(p) for p in P] + [0.0])
    for name, L in (('hex', L_HEX), ('dhex', L_DHEX), ('pent', L_PENT), ('uron', L_URON)):
        ev['L_' + name] = mx(lambda p, L=L: _loss_int(p, (L, L + H) if p['mode'] < 0 else (L,), c, chain=SUGAR_L))
    ev['L_CGLY'] = mx(lambda p: _loss_int(p, L_CGLY, c))
    ev['any_sugar'] = max(ev['L_hex'], ev['L_dhex'], ev['L_pent'], ev['L_uron'])
    for k, (mz, mode, req) in IONS.items():
        ev['I_' + k] = mx(lambda p, mz=mz, mode=mode: _int_at(p, mz, c) if p['mode'] == mode else 0.0)
    ev['any_sugar'] = max(ev['any_sugar'], ev['I_oxo_hex'], ev['I_oxo_dhex'], ev['I_oxo_pent'])
    ev['acyl'] = {k: mx(lambda p, L=L: _loss_int(p, (L, L + L_H2O), c)) for k, L in ACYLS.items()}
    ev['L_SULF'] = max(mx(lambda p: _loss_int(p, (L_SO3, L_H2SO4), c)), ev['I_hso4'])
    ev['L_PHOS'] = max(mx(lambda p: _loss_int(p, (L_HPO3, L_H3PO4), c)), ev['I_h2po4'], ev['I_po3'])
    ev['L_CO2N'] = mx(lambda p: _loss_int(p, (L_CO2,), c) if p['mode'] < 0 else 0.0)
    ev['L_CH3'] = mx(lambda p: _loss_int(p, (L_CH3,), c))
    ev['L_NH3'] = mx(lambda p: _loss_int(p, (L_NH3,), c) if p['mode'] > 0 and p['add'] in ('[M+H]+',) else 0.0)
    ev['L_TMA'] = mx(lambda p: _loss_int(p, (L_TMA,), c) if p['mode'] > 0 else 0.0)
    adds = [str(s.get('adduct') or '') for s in (spectra or [])]
    ev['AD_H2O'] = 2 if '[M-2H2O+H]+' in adds else 1 if ('[M-H2O+H]+' in adds or '[M-H2O-H]-' in adds) else 0
    ev['AD_NEG'] = 1 if '[M-H]-' in adds else 0
    ev['_P'] = P
    return ev


def _lvl(x, c):
    """Evidence intensity -> weight (0, w_weak, 1)."""
    return 1.0 if x >= c['strong'] else c['w_weak'] if x >= c['weak'] else 0.0


def violations(prof, ev, cfg=None, frag_cache=None):
    """rule -> violation weight for one candidate profile (rules listed in RULES; absent key = no violation)."""
    c = resolve_cfg(cfg)
    use = set(DEFAULT_RULES if c['rules'] is None else c['rules'])
    v = {}
    tolL = c['tol_da'] * 2
    if prof is None:
        return v
    for name in ('hex', 'dhex', 'pent', 'uron'):
        r = 'L_' + name.upper()
        if r in use and prof['gly_' + name] == 0:
            w = _lvl(ev['L_' + name], c)
            if w:
                v[r] = w
    if 'L_CGLY' in use and prof['gly_C'] == 0:
        w = _lvl(ev['L_CGLY'], c)
        if w:
            v['L_CGLY'] = w
    if 'A_GLY' in use and prof['gly_O'] > 0 and ev['informative'] and ev['any_sugar'] < c['weak']:
        v['A_GLY'] = c['w_absent']
    if 'I_OXO' in use:
        w = max([_lvl(ev['I_' + k], c) for k in ('oxo_hex', 'oxo_dhex', 'oxo_pent') if prof[IONS[k][2]] == 0] + [0.0])
        if w:
            v['I_OXO'] = w
    if 'L_ACYL' in use or 'A_ACYL' in use:
        have = prof['acyl']
        miss = 0.0
        for k, L in ACYLS.items():
            x = ev['acyl'][k]
            if x >= c['weak'] and not any(abs(a - L) <= tolL for a in have):
                miss = max(miss, _lvl(x, c))
        if 'L_ACYL' in use and miss:
            v['L_ACYL'] = miss
        if 'A_ACYL' in use and ev['informative']:
            named = [k for k, L in ACYLS.items() if any(abs(a - L) <= tolL for a in have)]
            if named and all(ev['acyl'][k] < c['weak'] for k in named):
                v['A_ACYL'] = c['w_absent']
    if 'L_SULF' in use and prof['sulf_any'] == 0:
        w = _lvl(ev['L_SULF'], c)
        if w:
            v['L_SULF'] = w
    if 'A_SULF' in use and prof['sulfate'] > 0 and ev['informative'] and ev['L_SULF'] < c['weak']:
        v['A_SULF'] = c['w_absent']
    if 'L_PHOS' in use and prof['phosphate'] == 0:
        w = _lvl(ev['L_PHOS'], c)
        if w:
            v['L_PHOS'] = w
    for r, key in (('L_CO2N', 'carboxyl_any'), ('L_CH3', 'me_het'), ('L_NH3', 'nh2'), ('L_TMA', 'nme3')):
        if r in use and prof[key] == 0:
            w = _lvl(ev[r], c)
            if w:
                v[r] = w
    if 'I_CHOL' in use and prof['choline'] == 0:
        w = max(_lvl(ev['I_choline'], c), _lvl(ev['I_pchol'], c))
        if w:
            v['I_CHOL'] = w
    if 'I_TROP' in use and prof['benzyl'] == 0:
        w = _lvl(ev['I_trop'], c)
        if w:
            v['I_TROP'] = w
    if 'I_IMM' in use:
        w = max([_lvl(ev['I_imm_' + a], c) for a in ('phe', 'tyr', 'trp', 'his') if prof['aa_' + a] == 0] + [0.0])
        if w:
            v['I_IMM'] = w
    if 'AD_H2O' in use and ev['AD_H2O'] and prof['h2o_sites'] < ev['AD_H2O']:
        v['AD_H2O'] = c['w_adduct']
    if 'AD_NEG' in use and ev['AD_NEG'] and prof['acidic_h'] == 0:
        v['AD_NEG'] = c['w_adduct']
    if 'XPK' in use:
        w = _unexplained(prof, ev, c, frag_cache)
        if w:
            v['XPK'] = w
    return v


def _unexplained(prof, ev, c, frag_cache):
    """1.0 if some peak >= xpk_int (below the precursor) is not a <=2-cleavage piece +- H of the candidate, else 0."""
    from chemcore import fragment_masses
    key = prof['smiles']
    if frag_cache is not None and key in frag_cache:
        fm = frag_cache[key]
    else:
        fm = fragment_masses(key, 2)
        if frag_cache is not None:
            frag_cache[key] = fm
    if len(fm) == 0:
        return 0.0
    for p in ev['_P']:
        car = [PROTON] if p['mode'] > 0 else [-PROTON]
        if p['add'] == '[M+Na]+':
            car.append(22.9897692809 - E)
        if p['add'] == '[M+K]+':
            car.append(38.96370668 - E)
        sel = (p['it'] >= c['xpk_int']) & (p['mz'] < p['prec'] - 1.5) & (p['mz'] >= 50)
        for mz in p['mz'][sel]:
            tol = c['tol_da'] + c['tol_ppm'] * 1e-6 * mz + 0.001
            ok = False
            for cr in car:
                q = mz - cr - H * np.arange(-2, 4)
                a = np.searchsorted(fm, q - tol); b = np.searchsorted(fm, q + tol, 'right')
                if (b > a).any():
                    ok = True
                    break
            if not ok:
                return 1.0
    return 0.0


# ---------------------------------------------------------------------------------------------------------------------
# re-order
# ---------------------------------------------------------------------------------------------------------------------
def reorder(rows, spectra, cfg=None, cache=None, ev=None):
    """Permute same-formula groups among their own slots in rows[:top] by violation tier; returns (rows, info).

    info: dict(groups=[{slots, formula, v: [per member total], rules: {rule: n violators}, moved: bool}], changed=bool)
    """
    c = resolve_cfg(cfg)
    cache = {} if cache is None else cache
    rows = list(rows)
    top = min(int(c['top']), len(rows))
    info = dict(groups=[], changed=False)
    if top < 2:
        return rows, info
    profs = []
    for s in rows[:top]:
        if s not in cache:
            cache[s] = profile(s)
        profs.append(cache[s])
    groups = {}
    for i, pr in enumerate(profs):
        if pr is not None:
            groups.setdefault(pr['formula'], []).append(i)
    groups = {f: ix for f, ix in groups.items() if len(ix) >= 2}
    if not groups:
        return rows, info
    if ev is None:
        ev = evidence(spectra, c)
    fc = cache.setdefault('__frag__', {}) if isinstance(cache, dict) else None
    out = list(rows)
    for f, ix in groups.items():
        V = [violations(profs[i], ev, c, fc) for i in ix]
        # a rule counts only if it separates the group
        rules = set().union(*[set(v) for v in V])
        sep = {r for r in rules if any(r not in v for v in V)}
        tot = [sum(w for r, w in v.items() if r in sep) for v in V]
        vmin = min(tot)
        tier = [int(np.floor((t - vmin) / c['theta'] + 1e-9)) for t in tot]
        order = sorted(range(len(ix)), key=lambda k: (tier[k], k))
        moved = order != list(range(len(ix)))
        g = dict(formula=f, slots=list(ix), v=tot, tier=tier, rules={r: sum(r in v for v in V) for r in sep}, moved=moved,
                 viol=[{r: w for r, w in v.items() if r in sep} for v in V])
        info['groups'].append(g)
        if moved:
            for slot, k in zip(ix, order):
                out[slot] = rows[ix[k]]
            info['changed'] = True
    return out, info
