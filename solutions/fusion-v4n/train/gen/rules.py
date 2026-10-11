"""Biotransformation rule table (RDKit reaction SMARTS with formula deltas) and formula-constrained analog enumeration from known parents.

Study code of reviews/gen2-2026-10-10.md (reachability of answers that are in no database). Nothing here runs on Kaggle.
A rule = (reaction SMARTS on one reactant, formula delta, group). `enumerate_target` takes a target formula and a parent index
(formula string -> parent ids, SMILES by id) and returns every product with the target formula reachable in
  step '1'    one rule applied to a parent whose formula is target - delta(rule)
  step 'move' a substituent removed and re-attached elsewhere on a same-formula parent (positional isomerisation)
  step '2'    two rules in either order from a parent whose formula is target - delta(r1) - delta(r2)
Products are stereo-free canonical SMILES of single-component neutral molecules; the caller removes structures it already knows.
"""
import time
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

RDLogger.DisableLog('rdApp.*')

_OH = '[O;H1;!$(OC=O);!$(O[S,P]):1]'           # alcohol / phenol / sugar hydroxyl
_HEX = 'C1OC(CO)C(O)C(O)C1O'; _DHEX = 'C1OC(C)C(O)C(O)C1O'; _PENP = 'C1OCC(O)C(O)C1O'; _PENF = 'C1OC(CO)C(O)C1O'; _GLCA = 'C1OC(C(=O)O)C(O)C(O)C1O'
_SP3 = '[C;H1,H2,H3;!$(C=*);!$(C#*):1]'

# name: (SMARTS, formula delta, group)
RULES = {
    # methylation / demethylation (CH2)
    'methyl_O': (_OH + '>>[O:1]C', dict(C=1, H=2), 'methyl'),
    'methyl_Oacid': ('[O;H1;$(OC=O):1]>>[O:1]C', dict(C=1, H=2), 'methyl'),
    'methyl_N': ('[N;H1,H2;!$(N=*);!$(N#*):1]>>[N:1]C', dict(C=1, H=2), 'methyl'),
    'methyl_c': ('[c;H1:1]>>[c:1]C', dict(C=1, H=2), 'methyl'),
    'methyl_C': (_SP3 + '>>[C:1]C', dict(C=1, H=2), 'methyl'),
    'ch2_insert': ('[C:1]-;!@[C;H2;!R:2]>>[C:1]C[C:2]', dict(C=1, H=2), 'methyl'),
    'demethyl_O': ('[O;!$(OC=O):1][CH3]>>[O:1]', dict(C=-1, H=-2), 'methyl'),
    'demethyl_Oacid': ('[O;$(OC=O):1][CH3]>>[O:1]', dict(C=-1, H=-2), 'methyl'),
    'demethyl_N': ('[N:1][CH3]>>[N:1]', dict(C=-1, H=-2), 'methyl'),
    'demethyl_c': ('[c:1][CH3]>>[c:1]', dict(C=-1, H=-2), 'methyl'),
    'demethyl_C': ('[C;!$(C=O):1][CH3]>>[C:1]', dict(C=-1, H=-2), 'methyl'),
    'ch2_delete': ('[C:1]-;!@[CH2;!R]-;!@[C:2]>>[C:1][C:2]', dict(C=-1, H=-2), 'methyl'),
    # hydroxylation / dehydroxylation (O)
    'hydroxyl_c': ('[c;H1:1]>>[c:1]O', dict(O=1), 'hydroxyl'),
    'hydroxyl_C': (_SP3 + '>>[C:1]O', dict(O=1), 'hydroxyl'),
    'epoxide': ('[C;!$(C=O):1]=[C;!$(C=O):2]>>[C:1]1O[C:2]1', dict(O=1), 'hydroxyl'),
    'N_oxide_hydroxyl': ('[N;H1;!$(NC=O):1]>>[N:1]O', dict(O=1), 'hydroxyl'),
    'dehydroxyl_c': ('[c:1][OH]>>[c:1]', dict(O=-1), 'hydroxyl'),
    'dehydroxyl_C': ('[C;!$(C=O):1][OH]>>[C:1]', dict(O=-1), 'hydroxyl'),
    'deepoxide': ('[C:1]1O[C:2]1>>[C:1]=[C:2]', dict(O=-1), 'hydroxyl'),
    # hydrogenation / dehydrogenation (H2)
    'hydrogenate_CC': ('[C:1]=[C:2]>>[C:1][C:2]', dict(H=2), 'H2'),
    'hydrogenate_CO': ('[C;!$(C(=O)[O,N,S]):1]=[O:2]>>[C:1][O:2]', dict(H=2), 'H2'),
    'dehydrogenate_CC': ('[C;H1,H2,H3;!$(C=*):1]-[C;H1,H2,H3;!$(C=*):2]>>[C:1]=[C:2]', dict(H=-2), 'H2'),
    'oxidize_CHOH': ('[C;H1,H2:1][O;H1:2]>>[C:1]=[O:2]', dict(H=-2), 'H2'),
    # acetylation
    'acetyl_O': (_OH + '>>[O:1]C(C)=O', dict(C=2, H=2, O=1), 'acetyl'),
    'acetyl_N': ('[N;H1,H2;!$(NC=O);!$(N=*):1]>>[N:1]C(C)=O', dict(C=2, H=2, O=1), 'acetyl'),
    'deacetyl_O': ('[O:1]C(=O)[CH3]>>[O:1]', dict(C=-2, H=-2, O=-1), 'acetyl'),
    'deacetyl_N': ('[N:1]C(=O)[CH3]>>[N:1]', dict(C=-2, H=-2, O=-1), 'acetyl'),
    # glycosylation / deglycosylation: O- (alcohol, phenol, sugar hydroxyl), acyl-O, C- (aromatic CH)
    'hexose_O': (_OH + '>>[O:1]' + _HEX, dict(C=6, H=10, O=5), 'glycosyl'),
    'hexose_Oacid': ('[O;H1;$(OC=O):1]>>[O:1]' + _HEX, dict(C=6, H=10, O=5), 'glycosyl'),
    'hexose_C': ('[c;H1:1]>>[c:1]' + _HEX, dict(C=6, H=10, O=5), 'glycosyl'),
    'deoxyhexose_O': (_OH + '>>[O:1]' + _DHEX, dict(C=6, H=10, O=4), 'glycosyl'),
    'deoxyhexose_C': ('[c;H1:1]>>[c:1]' + _DHEX, dict(C=6, H=10, O=4), 'glycosyl'),
    'pentose_O': (_OH + '>>[O:1]' + _PENP, dict(C=5, H=8, O=4), 'glycosyl'),
    'pentosef_O': (_OH + '>>[O:1]' + _PENF, dict(C=5, H=8, O=4), 'glycosyl'),
    'pentose_C': ('[c;H1:1]>>[c:1]' + _PENP, dict(C=5, H=8, O=4), 'glycosyl'),
    'glucuronyl_O': (_OH + '>>[O:1]' + _GLCA, dict(C=6, H=8, O=6), 'glycosyl'),
    'dehexose_O': ('[O:1]C1O[CH]([CH2][OH])[CH]([OH])[CH]([OH])[CH]1[OH]>>[O:1]', dict(C=-6, H=-10, O=-5), 'glycosyl'),
    'dehexose_C': ('[c:1]C1O[CH]([CH2][OH])[CH]([OH])[CH]([OH])[CH]1[OH]>>[c:1]', dict(C=-6, H=-10, O=-5), 'glycosyl'),
    'dedeoxyhexose_O': ('[O:1]C1O[CH]([CH3])[CH]([OH])[CH]([OH])[CH]1[OH]>>[O:1]', dict(C=-6, H=-10, O=-4), 'glycosyl'),
    'dedeoxyhexose_C': ('[c:1]C1O[CH]([CH3])[CH]([OH])[CH]([OH])[CH]1[OH]>>[c:1]', dict(C=-6, H=-10, O=-4), 'glycosyl'),
    'depentose_O': ('[O:1]C1O[CH2][CH]([OH])[CH]([OH])[CH]1[OH]>>[O:1]', dict(C=-5, H=-8, O=-4), 'glycosyl'),
    'depentosef_O': ('[O:1]C1O[CH]([CH2][OH])[CH]([OH])[CH]1[OH]>>[O:1]', dict(C=-5, H=-8, O=-4), 'glycosyl'),
    'depentose_C': ('[c:1]C1O[CH2][CH]([OH])[CH]([OH])[CH]1[OH]>>[c:1]', dict(C=-5, H=-8, O=-4), 'glycosyl'),
    'deglucuronyl_O': ('[O:1]C1O[CH](C(=O)[OH])[CH]([OH])[CH]([OH])[CH]1[OH]>>[O:1]', dict(C=-6, H=-8, O=-6), 'glycosyl'),
    # malonylation
    'malonyl_O': (_OH + '>>[O:1]C(=O)CC(=O)O', dict(C=3, H=2, O=3), 'malonyl'),
    'demalonyl_O': ('[O:1]C(=O)[CH2]C(=O)[OH]>>[O:1]', dict(C=-3, H=-2, O=-3), 'malonyl'),
    # prenylation
    'prenyl_c': ('[c;H1:1]>>[c:1]CC=C(C)C', dict(C=5, H=8), 'prenyl'),
    'prenyl_O': (_OH + '>>[O:1]CC=C(C)C', dict(C=5, H=8), 'prenyl'),
    'deprenyl_c': ('[c:1][CH2][CH]=C([CH3])[CH3]>>[c:1]', dict(C=-5, H=-8), 'prenyl'),
    'deprenyl_O': ('[O:1][CH2][CH]=C([CH3])[CH3]>>[O:1]', dict(C=-5, H=-8), 'prenyl'),
    # sulfation
    'sulfate_O': (_OH + '>>[O:1]S(=O)(=O)O', dict(S=1, O=3), 'sulfate'),
    'desulfate_O': ('[O:1]S(=O)(=O)[OH]>>[O:1]', dict(S=-1, O=-3), 'sulfate'),
    # dehydration / hydration
    'dehydrate': ('[C;!$(C=*):1]([OH])-[C;H1,H2,H3;!$(C=*):2]>>[C:1]=[C:2]', dict(H=-2, O=-1), 'water'),
    'hydrate': ('[C;!$(C=O):1]=[C:2]>>[C:1](O)[C:2]', dict(H=2, O=1), 'water'),
    # methoxy as one unit, common acyl groups (not in the task's list; kept from casmi.derive, reported as their own groups)
    'methoxy_c': ('[c;H1:1]>>[c:1]OC', dict(C=1, H=2, O=1), 'methoxy'),
    'demethoxy_c': ('[c:1]O[CH3]>>[c:1]', dict(C=-1, H=-2, O=-1), 'methoxy'),
    'coumaroyl_O': (_OH + '>>[O:1]C(=O)C=Cc1ccc(O)cc1', dict(C=9, H=6, O=2), 'acyl'),
    'feruloyl_O': (_OH + '>>[O:1]C(=O)C=Cc1ccc(O)c(OC)c1', dict(C=10, H=8, O=3), 'acyl'),
    'caffeoyl_O': (_OH + '>>[O:1]C(=O)C=Cc1ccc(O)c(O)c1', dict(C=9, H=6, O=3), 'acyl'),
    'galloyl_O': (_OH + '>>[O:1]C(=O)c1cc(O)c(O)c(O)c1', dict(C=7, H=4, O=4), 'acyl'),
    'benzoyl_O': (_OH + '>>[O:1]C(=O)c1ccccc1', dict(C=7, H=4, O=1), 'acyl'),
    'decoumaroyl_O': ('[O:1]C(=O)[CH]=[CH]c1ccc([OH])cc1>>[O:1]', dict(C=-9, H=-6, O=-2), 'acyl'),
    'deferuloyl_O': ('[O:1]C(=O)[CH]=[CH]c1ccc([OH])c(O[CH3])c1>>[O:1]', dict(C=-10, H=-8, O=-3), 'acyl'),
    'decaffeoyl_O': ('[O:1]C(=O)[CH]=[CH]c1ccc([OH])c([OH])c1>>[O:1]', dict(C=-9, H=-6, O=-3), 'acyl'),
    'degalloyl_O': ('[O:1]C(=O)c1cc([OH])c([OH])c([OH])c1>>[O:1]', dict(C=-7, H=-4, O=-4), 'acyl'),
    'debenzoyl_O': ('[O:1]C(=O)c1ccccc1>>[O:1]', dict(C=-7, H=-4, O=-1), 'acyl'),
}

# positional isomerisation: (removal, re-attachment) on a same-formula parent; name of the moved substituent
MOVES = [(r, a, sub) for sub, rem, add in [
    ('methyl', ['demethyl_O', 'demethyl_Oacid', 'demethyl_N', 'demethyl_c', 'demethyl_C'], ['methyl_O', 'methyl_Oacid', 'methyl_N', 'methyl_c', 'methyl_C']),
    ('hydroxyl', ['dehydroxyl_c', 'dehydroxyl_C'], ['hydroxyl_c', 'hydroxyl_C']),
    ('methoxy', ['demethoxy_c'], ['methoxy_c']),
    ('hexose', ['dehexose_O', 'dehexose_C'], ['hexose_O', 'hexose_Oacid', 'hexose_C']),
    ('deoxyhexose', ['dedeoxyhexose_O', 'dedeoxyhexose_C'], ['deoxyhexose_O', 'deoxyhexose_C']),
    ('pentose', ['depentose_O', 'depentosef_O', 'depentose_C'], ['pentose_O', 'pentosef_O', 'pentose_C']),
    ('glucuronyl', ['deglucuronyl_O'], ['glucuronyl_O']),
    ('acetyl', ['deacetyl_O', 'deacetyl_N'], ['acetyl_O', 'acetyl_N']),
    ('malonyl', ['demalonyl_O'], ['malonyl_O']),
    ('prenyl', ['deprenyl_c', 'deprenyl_O'], ['prenyl_c', 'prenyl_O']),
    ('sulfate', ['desulfate_O'], ['sulfate_O']),
    ('double bond', ['hydrogenate_CC'], ['dehydrogenate_CC']),
] for r in rem for a in add]
_MOVE_PAIRS = {(r, a) for r, a, _ in MOVES}
GROUP = {n: v[2] for n, v in RULES.items()}
ELEMS = ['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']
_RXN = {}


def rxn(name):
    if name not in _RXN:
        _RXN[name] = AllChem.ReactionFromSmarts(RULES[name][0])
    return _RXN[name]


def parse(f):
    import re
    out = {}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', f):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def fstr(c):
    """Hill-order formula string as RDKit's CalcMolFormula writes it for neutral molecules; None when a count is negative."""
    if any(v < 0 for v in c.values()):
        return None
    c = {k: v for k, v in c.items() if v > 0}
    if 'C' in c:
        order = ['C'] + (['H'] if 'H' in c else []) + sorted(k for k in c if k not in ('C', 'H'))
    else:
        order = sorted(c)
    return ''.join(f'{k}{c[k] if c[k] != 1 else ""}' for k in order)


def minus(c, *deltas):
    out = dict(c)
    for d in deltas:
        for k, v in d.items():
            out[k] = out.get(k, 0) - v
    return out


def apply(mol, name, cap=64):
    """Distinct single-component products (mol, smiles) of one rule on one molecule, every matching site."""
    try:
        prods = rxn(name).RunReactants((mol,), maxProducts=cap * 8)
    except Exception:
        return []
    out = {}
    for p in prods:
        m = p[0]
        try:
            Chem.SanitizeMol(m); s = Chem.MolToSmiles(m)
        except Exception:
            continue
        if '.' in s or s in out:
            continue
        out[s] = m
        if len(out) >= cap:
            break
    return [(m, s) for s, m in out.items()]


def one_step_formulas(target):
    """{parent formula: [rule, ...]} for one rule."""
    out = defaultdict(list)
    for n, (_, d, _) in RULES.items():
        f = fstr(minus(target, d))
        if f:
            out[f].append(n)
    return out


def two_step_formulas(target):
    """{parent formula: [(r1, r2), ...]} for ordered rule pairs that are not a positional move and do not undo each other's formula with the same group... (net-zero pairs of
    different sites are kept: they are isomerisations outside the move table)."""
    out = defaultdict(list); names = list(RULES)
    for a in names:
        for b in names:
            if (a, b) in _MOVE_PAIRS or all(RULES[a][1].get(e, 0) + RULES[b][1].get(e, 0) == 0 for e in ELEMS):
                continue
            f = fstr(minus(target, RULES[a][1], RULES[b][1]))
            if f:
                out[f].append((a, b))
    return out


def products_one(psmi, rules, target_f, cap=64):
    """[(smiles, rule)] of one-step products of a parent that have the target formula."""
    m = Chem.MolFromSmiles(psmi)
    if m is None:
        return []
    out = []
    for r in rules:
        for pm, s in apply(m, r, cap):
            if rdMolDescriptors.CalcMolFormula(pm) == target_f:
                out.append((s, r))
    return out


def products_pairs(psmi, pairs, target_f, cap1=24, cap2=16, budget=600, deadline=None):
    """[(smiles, 'r1+r2')] of two-step products (ordered pairs) of a parent with the target formula; the parent itself is dropped."""
    m = Chem.MolFromSmiles(psmi)
    if m is None:
        return []
    p0 = Chem.MolToSmiles(m); first = {}; out = []
    for a, b in pairs:
        if a not in first:
            first[a] = apply(m, a, cap1)
        for im, _ in first[a]:
            for pm, s in apply(im, b, cap2):
                budget -= 1
                if s != p0 and rdMolDescriptors.CalcMolFormula(pm) == target_f:
                    out.append((s, a + '+' + b))
            if budget <= 0 or (deadline is not None and time.time() > deadline):
                return out
    return out


def products_move(psmi, target_f, cap1=24, cap2=24, budget=1500, deadline=None):
    return products_pairs(psmi, [(r, a) for r, a, _ in MOVES], target_f, cap1, cap2, budget, deadline)


def neighbours(smiles, steps=1, cap=64, cap2=32, max_out=200000):
    """Rule neighbourhood of a structure, any formula: {smiles: 'r1' or 'r1+r2'} (used backwards from an answer to find candidate parents)."""
    m = Chem.MolFromSmiles(smiles); out = {}
    if m is None:
        return out
    lvl1 = []
    for r in RULES:
        for pm, s in apply(m, r, cap):
            if s not in out:
                out[s] = r; lvl1.append((pm, r))
    if steps >= 2:
        for pm, r in lvl1:
            for r2 in RULES:
                for _, s in apply(pm, r2, cap2):
                    if s not in out:
                        out[s] = r + '+' + r2
            if len(out) >= max_out:
                break
    return out
