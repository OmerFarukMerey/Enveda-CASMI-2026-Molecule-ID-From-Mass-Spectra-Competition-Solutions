"""Analog generator channel (GEN_ANALOG; off by default): one rule step from the best fingerprint-ranked pool parents, at most 25 new structures per molecule, placed by fixed slots or a gated probability merge.

Why. About 45% of the hidden answers are in none of our 25 rows and most of those are in no database (reviews/budget-2026-10-10.md).
reviews/gen2-2026-10-10.md measured that one biotransformation step from the pool rows the fingerprint network ranks first reaches
57 to 66% of such answers when a close relative is known (10 to 17% when not), with tens of candidates, and that placement decides the
sign: a generated row at position 1 costs database answers 0.35, five rows from position 6 cost 0.003. This module is that bounded
build; reviews/gen3-2026-10-10.md is its measurement. Pure functions, RDKit and numpy only, no network, no file access.

    z    = spectrum_logits(spectra, bank, fpnet, chem, merge_spectra)        # FPNet logits of the molecule, as the v1 engine aggregates them
    pool = PoolIndex(smiles, keys, formulas, packed_fp)                      # the v2 pool (pool_meta.parquet, pool_fp.npy)
    rows = generate(pool, z, base_formulas, exclude_keys, fp_fn=..., key_fn=...)   # <= 25 dicts: smiles, key, features
    out  = place(final_smiles, rows, 'slots', pos=6, k=5)                    # policy 'slots'
    out  = place(final_smiles, rows, 'gate', s1=s1, lib_max=lib_max)         # policy 'gate' (a list whose gate is closed is returned as it is)

Generator (fixed by scratch/gen3/PREREG.md). Target formulas = the first KF distinct formulas of the molecule's base list. Parents =
for every formula `target - delta(rule)` the M pool rows with the highest f.z (stored pool fingerprints x logits; nothing is
fingerprinted to choose a parent), and the M best same-formula rows for positional moves. Products = one rule (66 reaction SMARTS)
or one move (61 removal / re-attachment pairs) with exactly the target formula, sanitised, one component, stereo-free; pool rows and
structures whose metric key is already a candidate are dropped. Order = best parent rank, one step before move, number of parents,
parent f.z; formulas interleaved with position x formula index. The first KEY_CAP products get a metric key, the first ROW_CAP distinct
new keys are returned with their features (parent rank and f.z, parents and rules that support the product, step, rule group, own f.z,
Tanimoto to the parent).

The rule table is a copy of train/gen/rules.py (tests/test_gen_analog.py asserts that the two are equal).
"""
from __future__ import annotations

import math, re, time
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

KF, M_PARENTS, KEY_CAP, ROW_CAP, T_ENUM, T_KEY_SLOW = 3, 3, 60, 25, 10.0, 0.04     # pre-registered caps (scratch/gen3/PREREG.md)
MODES = ('slots', 'gate')

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
_MOVE_PAIRS = [(r, a) for r, a, _ in MOVES]
GROUP = {n: v[2] for n, v in RULES.items()}
GROUPS = ('methyl', 'H2', 'hydroxyl', 'glycosyl')                      # rule groups with a column of their own in the placement model
_RXN: Dict[str, object] = {}
_RD: Dict[str, object] = {}


def _rd():
    if not _RD:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem, rdFingerprintGenerator, rdMolDescriptors
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog('rdApp.*')
        _RD.update(Chem=Chem, AllChem=AllChem, rdMD=rdMolDescriptors, DS=DataStructs, taut=rdMolStandardize.TautomerEnumerator(),
                   morgan=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))
    return _RD


# ------------------------------------------------------------------------------------------------ rules
def rxn(name: str):
    if name not in _RXN:
        _RXN[name] = _rd()['AllChem'].ReactionFromSmarts(RULES[name][0])
    return _RXN[name]


def parse(f: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', f):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def fstr(c: Dict[str, int]) -> Optional[str]:
    """Hill-order formula string as RDKit's CalcMolFormula writes it for neutral molecules; None when a count is negative."""
    if any(v < 0 for v in c.values()):
        return None
    c = {k: v for k, v in c.items() if v > 0}
    if 'C' in c:
        order = ['C'] + (['H'] if 'H' in c else []) + sorted(k for k in c if k not in ('C', 'H'))
    else:
        order = sorted(c)
    return ''.join(f'{k}{c[k] if c[k] != 1 else ""}' for k in order)


def minus(c: Dict[str, int], *deltas) -> Dict[str, int]:
    out = dict(c)
    for d in deltas:
        for k, v in d.items():
            out[k] = out.get(k, 0) - v
    return out


def apply_rule(mol, name: str, cap: int = 64) -> list:
    """Distinct single-component products [(mol, smiles)] of one rule on one molecule, every matching site."""
    Chem = _rd()['Chem']
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


def one_step_formulas(target: Dict[str, int]) -> Dict[str, List[str]]:
    """{parent formula: [rule, ...]}: the formulas from which one rule leads to the target formula."""
    out = defaultdict(list)
    for n, (_, d, _) in RULES.items():
        f = fstr(minus(target, d))
        if f:
            out[f].append(n)
    return out


def products_one(psmi: str, rules: Sequence[str], target_f: str, cap: int = 64) -> List[tuple]:
    """[(smiles, rule)] of the one-step products of a parent that have the target formula."""
    r = _rd(); m = r['Chem'].MolFromSmiles(psmi)
    if m is None:
        return []
    out = []
    for name in rules:
        for pm, s in apply_rule(m, name, cap):
            if r['rdMD'].CalcMolFormula(pm) == target_f:
                out.append((s, name))
    return out


def products_move(psmi: str, target_f: str, cap1: int = 24, cap2: int = 24, budget: int = 1500, deadline: Optional[float] = None) -> List[tuple]:
    """[(smiles, 'removal+attachment')] of the positional isomers of a same-formula parent (the parent itself is dropped)."""
    r = _rd(); m = r['Chem'].MolFromSmiles(psmi)
    if m is None:
        return []
    p0 = r['Chem'].MolToSmiles(m); first = {}; out = []
    for a, b in _MOVE_PAIRS:
        if a not in first:
            first[a] = apply_rule(m, a, cap1)
        for im, _ in first[a]:
            for pm, s in apply_rule(im, b, cap2):
                budget -= 1
                if s != p0 and r['rdMD'].CalcMolFormula(pm) == target_f:
                    out.append((s, a + '+' + b))
            if budget <= 0 or (deadline is not None and time.time() > deadline):
                return out
    return out


# ------------------------------------------------------------------------------------------------ pool, logits, keys
class PoolIndex:
    """The candidate pool as the generator needs it: SMILES, metric keys, formulas and packed fingerprints by pool id."""

    def __init__(self, smiles: Sequence[str], keys: Sequence[str], formulas: Sequence[str], packed_fp: np.ndarray):
        self.smiles = np.asarray(smiles, object); self.key = np.asarray(keys, object); self.fp = packed_fp
        f2p: Dict[str, list] = defaultdict(list)
        for i, f in enumerate(formulas):
            f2p[f].append(i)
        self.f2p = {f: np.asarray(v, np.int64) for f, v in f2p.items()}
        self._win: Dict[str, tuple] = {}

    def window(self, formula: str) -> np.ndarray:
        return self.f2p.get(formula, np.zeros(0, np.int64))

    def known(self, formula: str) -> tuple:
        """(SMILES set, key set) of the pool rows with this formula: what a product must not be."""
        if formula not in self._win:
            if len(self._win) > 64:
                self._win.clear()
            w = self.window(formula); self._win[formula] = (set(self.smiles[w]), set(self.key[w]))
        return self._win[formula]

    def ranked(self, formula: str, z: np.ndarray, m: int, banned: Optional[set] = None) -> List[tuple]:
        """The m rows of a formula window with the highest f.z: [(pool id, 1-based rank, f.z / 100, window size)]."""
        w = self.window(formula)
        if banned:
            w = np.asarray([p for p in w if int(p) not in banned], np.int64)
        if len(w) == 0:
            return []
        f = np.unpackbits(np.asarray(self.fp[w]), axis=1)[:, :len(z)].astype(np.float32) @ np.asarray(z, np.float32)
        o = np.argsort(-f, kind='stable')[:m]
        return [(int(w[j]), r + 1, float(f[j]) / 100.0, len(w)) for r, j in enumerate(o)]


def spectrum_logits(spectra: Sequence[dict], bank, fpnet, chem, merge_spectra) -> np.ndarray:
    """Fingerprint logits of one molecule: 0.5 x (mean over its spectra + mean over the merged spectrum of each polarity), the
    aggregation of casmi.engine.Engine.run (the `fz` feature of a candidate is its fingerprint times this vector)."""
    items_s = []
    for s in spectra:
        pm, pi = fpnet.prep_peaks(s['mz'], s['it'], s['prec']); ok = s['ce'] is not None and not np.isnan(s['ce'])
        items_s.append(dict(mz=pm, it=pi, prec=float(s['prec']), adduct_ix=chem.adduct_index(s['adduct']), ce=float(s['ce']) if ok else 0.0,
                            ce_known=1.0 if ok else 0.0, n_merged=min(int(s.get('ce_n', 1)), 8), mode=float(s['mode'])))
    items_m = []
    for md in (1, -1):
        grp = [s for s in spectra if s['mode'] == md]
        if not grp:
            continue
        mm, ii = merge_spectra([(s['mz'], s['it']) for s in grp]); prec = float(np.median([s['prec'] for s in grp])); pm, pi = fpnet.prep_peaks(mm, ii, prec)
        ces = [s['ce'] for s in grp if s['ce'] is not None and not np.isnan(s['ce'])]; ads = [s['adduct'] for s in grp]
        ad = sorted(set(ads), key=lambda a: (-ads.count(a), a))[0]          # most frequent adduct; ties by name (no hash-seed dependence)
        items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad), ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                            n_merged=min(sum(max(1, int(s.get('ce_n', 1))) for s in grp), 8), mode=float(md)))
    zs, _ = bank.logits_el(items_s); zm, _ = bank.logits_el(items_m)
    return (0.5 * (np.asarray(zs).mean(0) + np.asarray(zm).mean(0))).astype(np.float32)


def metric_key(smiles: str) -> Optional[str]:
    """Tautomer-canonical InChIKey first block (the metric's key; the products are stereo-free already)."""
    r = _rd()
    try:
        mol = r['Chem'].MolFromSmiles(smiles)
        if mol is None:
            return None
        ik = r['Chem'].MolToInchiKey(r['taut'].Canonicalize(mol))
        return ik[:14] if ik else None
    except Exception:
        return None


def plain_key(smiles: str) -> Optional[str]:
    r = _rd()
    try:
        ik = r['Chem'].MolToInchiKey(r['Chem'].MolFromSmiles(smiles))
        return ik[:14] if ik else None
    except Exception:
        return None


def tanimoto(a: str, b: str) -> float:
    r = _rd()
    try:
        fa = r['morgan'].GetFingerprint(r['Chem'].MolFromSmiles(a)); fb = r['morgan'].GetFingerprint(r['Chem'].MolFromSmiles(b))
        return float(r['DS'].TanimotoSimilarity(fa, fb))
    except Exception:
        return 0.0


# ------------------------------------------------------------------------------------------------ the generator
def target_formulas(base_formulas: Iterable[Optional[str]], kf: int = KF) -> List[str]:
    """The first kf distinct formulas of the molecule's base list, in the list's order."""
    out: List[str] = []
    for f in base_formulas:
        if f and f not in out:
            out.append(f)
            if len(out) >= kf:
                break
    return out


def enumerate_formula(pool: PoolIndex, z: np.ndarray, formula: str, m: int = M_PARENTS, banned: Optional[set] = None, deadline: Optional[float] = None,
                      extra_move_parents: Optional[Sequence[tuple]] = None) -> List[dict]:
    """Products with the formula `formula`, one row per distinct SMILES, in the pre-declared order (best parent rank, one step before
    move, number of distinct parents, parent f.z): dicts smiles, step ('1' | 'move'), rule, pid, prank, pfz, psmi, nsup, nrule.
    extra_move_parents: [(smiles, f.z / 100)] of same-formula structures outside the pool (joined PubChem rows) used as move parents,
    ranked together with the pool's own (pid -2)."""
    raw = []
    wins = sorted(((f, rl, pool.ranked(f, z, m, banned)) for f, rl in one_step_formulas(parse(formula)).items()), key=lambda x: x[0])
    for depth in range(m):                                  # rank-1 parents of every window first: a time cut is a cut in M
        for f, rl, ps in wins:
            if depth < len(ps):
                pid, rk, fz, _ = ps[depth]; psmi = pool.smiles[pid]
                raw += [(s, 0, r, pid, rk, fz, psmi) for s, r in products_one(psmi, rl, formula)]
            if deadline is not None and time.time() > deadline:
                break
    mp = [(pid, fz, pool.smiles[pid]) for pid, _, fz, _ in pool.ranked(formula, z, m if not extra_move_parents else 10 ** 6, banned)]
    if extra_move_parents:
        mp = sorted(mp + [(-2, float(fz), s) for s, fz in extra_move_parents], key=lambda x: -x[1])[:m]
    for rk, (pid, fz, psmi) in enumerate(mp, 1):
        if deadline is not None and time.time() > deadline:
            break
        raw += [(s, 1, r, pid, rk, fz, psmi) for s, r in products_move(psmi, formula, deadline=deadline)]
    by: Dict[str, dict] = {}
    for s, so, r, pid, rk, fz, psmi in raw:
        d = by.get(s)
        if d is None:
            d = by[s] = dict(smiles=s, so=so, rule=r, pid=pid, prank=rk, pfz=fz, psmi=psmi, parents={psmi}, rules={r})
        else:
            d['parents'].add(psmi); d['rules'].add(r)
            if (so, rk, -fz) < (d['so'], d['prank'], -d['pfz']):
                d.update(so=so, rule=r, pid=pid, prank=rk, pfz=fz, psmi=psmi)
    out = []
    for d in by.values():
        d['nsup'] = len(d.pop('parents')); d['nrule'] = len(d.pop('rules')); d['step'] = 'move' if d.pop('so') else '1'; d['formula'] = formula; out.append(d)
    out.sort(key=lambda d: (d['prank'], d['step'] == 'move', -d['nsup'], -d['pfz'], d['smiles']))
    return out


def interleave(per_formula: Sequence[List[dict]]) -> List[dict]:
    """Formulas merged into one order: position = (order inside the formula) x (formula index), ties to the lower index."""
    allr = []
    for fi, rows in enumerate(per_formula, 1):
        for o, d in enumerate(rows, 1):
            allr.append((o * fi, fi, o, d))
    allr.sort(key=lambda x: x[:3])
    return [dict(d, fidx=fi) for _, fi, _, d in allr]


def finish(products: Sequence[dict], pool: PoolIndex, z: np.ndarray, exclude_keys: Iterable[str], fp_fn: Optional[Callable] = None,
           key_fn: Callable = metric_key, fast_key_fn: Callable = plain_key, key_cap: int = KEY_CAP, row_cap: int = ROW_CAP,
           removed_keys: Optional[set] = None) -> List[dict]:
    """Keys, novelty and features for the first products of the order: at most key_cap products are keyed, the first row_cap distinct
    keys that are neither a pool row of the formula nor in exclude_keys are returned with ord = 1.. and the feature fields.
    fp_fn(smiles) -> 0/1 vector in the pool's bit order (None: the own-f.z features are 0). When the first 8 keys cost more than
    T_KEY_SLOW each (large amide- or sugar-rich structures) the plain InChIKey first block is used for the rest.
    removed_keys (replays only): keys of pool rows that were taken out of the pool, so a product with such a key is new."""
    seen = set(exclude_keys); out = []; n_key = 0; t0 = time.time(); slow = False; rem = removed_keys or ()
    for d in products:
        if n_key >= key_cap or len(out) >= row_cap:
            break
        ksmi, kkey = pool.known(d['formula'])
        if d['smiles'] in ksmi and not rem:
            continue
        if n_key == 8 and (time.time() - t0) / 8 > T_KEY_SLOW:
            slow = True
        k = (fast_key_fn if slow else key_fn)(d['smiles']); n_key += 1
        if k is None or k in seen or (k in kkey and k not in rem):
            continue
        seen.add(k); r = dict(d, key=k, ord=len(out) + 1, key_plain=slow)
        r['tan'] = tanimoto(d['smiles'], d['psmi']); r['fz'] = 0.0; r['fzn'] = 0.0
        if fp_fn is not None:
            f = fp_fn(d['smiles'])
            if f is not None:
                f = np.asarray(f, np.float32)[:len(z)]; r['fz'] = float(f @ np.asarray(z, np.float32)) / 100.0; r['fzn'] = r['fz'] * 10.0 / math.sqrt(max(1.0, float(f.sum())))
        out.append(r)
    return out


def generate(pool: PoolIndex, z: np.ndarray, base_formulas: Iterable[Optional[str]], exclude_keys: Iterable[str], fp_fn: Optional[Callable] = None,
             key_fn: Callable = metric_key, kf: int = KF, m: int = M_PARENTS, banned: Optional[set] = None, t_enum: float = T_ENUM,
             extra_move_parents: Optional[Dict[str, list]] = None, stats: Optional[dict] = None, removed_keys: Optional[set] = None) -> List[dict]:
    """The molecule's generated rows (at most ROW_CAP), best first. base_formulas: formulas of its base list in list order.
    exclude_keys: metric keys that are candidates already. banned: pool ids that must not be parents (replays that remove the answer)."""
    t0 = time.time(); fs = target_formulas(base_formulas, kf); per = []
    for f in fs:
        per.append(enumerate_formula(pool, z, f, m, banned, t0 + t_enum, (extra_move_parents or {}).get(f)))
    prods = interleave(per); t1 = time.time()
    rows = finish(prods, pool, z, exclude_keys, fp_fn, key_fn, removed_keys=removed_keys)
    if stats is not None:
        stats.update(n_formulas=len(fs), n_products=len(prods), n_rows=len(rows), t_enum=t1 - t0, t_finish=time.time() - t1, capped=bool(t1 - t0 >= t_enum))
    return rows


# ------------------------------------------------------------------------------------------------ placement
def place_slots(final: Sequence[str], gen: Sequence[str], pos: int = 6, k: int = 5, n_out: int = 25) -> List[str]:
    """Policy 'slots': the first k generated structures take positions pos .. pos + k - 1 (1-based); the rows from pos on move down.
    A list shorter than pos - 1 gets them at its end. No generated row = the list as it is."""
    final = list(final); g = list(gen)[:max(0, k)]
    if not g:
        return final[:n_out]
    p = min(max(1, int(pos)) - 1, len(final))
    return (final[:p] + g + final[p:])[:n_out]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _lin(model: dict, x: np.ndarray) -> np.ndarray:
    return ((x - np.asarray(model['mean'])) / np.asarray(model['std'])) @ np.asarray(model['coef']) + model['intercept']


def db_features(n: int, s1: float) -> np.ndarray:
    """Rows 1..n of a database list: log rank, first row, s1 (top score of the fold-safe ranker, clipped), s1 x first row."""
    r = np.arange(1, n + 1, dtype=np.float64); s = float(np.clip(s1, -6.0, 8.0)); f = (r == 1).astype(np.float64)
    return np.stack([np.log(r), f, np.full(n, s), s * f], 1)


GEN_FEATURES = ['log_ord', 'log_prank', 'move', 'log_nsup', 'log_nrule', 'tan', 'dfz', 'fzn_z', 'fidx', 'log_n', 's1', 'lib_max'] + ['g_' + g for g in GROUPS]


def gen_features(rows: Sequence[dict], s1: float, lib_max: float) -> np.ndarray:
    """Feature matrix of a molecule's generated rows (columns GEN_FEATURES)."""
    n = len(rows)
    if n == 0:
        return np.zeros((0, len(GEN_FEATURES)))
    fzn = np.array([r['fzn'] for r in rows], np.float64); sd = fzn.std(); fz_z = (fzn - fzn.mean()) / sd if sd > 1e-9 else np.zeros(n)
    s = float(np.clip(s1, -6.0, 8.0)); X = []
    for i, r in enumerate(rows):
        g = GROUP.get(r['rule'].split('+')[-1], '')
        X.append([math.log(r['ord']), math.log(r['prank']), float(r['step'] == 'move'), math.log(r['nsup']), math.log(r['nrule']), r['tan'], r['fz'] - r['pfz'],
                  fz_z[i], float(r['fidx']), math.log(n), s, float(lib_max)] + [float(g == x) for x in GROUPS])
    return np.asarray(X, np.float64)


def gate_open(lib_max: float, s1: Optional[float], model: dict, use_s1: bool = True) -> bool:
    """Policy 'gate': generated rows compete only where the list has no library evidence and the ranker does not believe its top row."""
    if s1 is None or not np.isfinite(s1) or lib_max >= model['max_lib']:
        return False
    return (not use_s1) or float(s1) < model['s1_max']


def place_merge(final: Sequence[str], rows: Sequence[dict], s1: float, lib_max: float, model: dict, n_out: int = 25) -> List[str]:
    """Policy 'gate' for a molecule whose gate is open: every row gets a probability of being the answer (database rows from their
    rank and s1, generated rows from their features) and the list is re-sorted by it; database rows keep their mutual order and
    come first among ties."""
    final = list(final)
    if not rows:
        return final[:n_out]
    pd_ = _sig(_lin(model['db'], db_features(len(final), s1))) if final else np.zeros(0)
    pd_ = np.minimum.accumulate(pd_) if len(pd_) else pd_                       # the list's own order is never broken
    pg = _sig(_lin(model['gen'], gen_features(rows, s1, lib_max)))
    items = [(-float(p), 0, i, s) for i, (s, p) in enumerate(zip(final, pd_))] + [(-float(p), 1, i, r['smiles']) for i, (r, p) in enumerate(zip(rows, pg))]
    items.sort(key=lambda x: x[:3])
    return [x[3] for x in items][:n_out]


def place(final: Sequence[str], rows: Sequence[dict], mode: str, pos: int = 6, k: int = 5, s1: Optional[float] = None, lib_max: float = 0.0,
          model: Optional[dict] = None, use_s1: bool = True) -> List[str]:
    """The final list with the generated rows placed by the chosen policy. rows must already exclude the final list's keys."""
    if mode == 'slots':
        return place_slots(final, [r['smiles'] for r in rows], pos, k)
    if mode == 'gate':
        model = model or MODEL
        if model is None or not gate_open(lib_max, s1, model, use_s1):
            return list(final)[:25]
        return place_merge(final, rows, s1, lib_max, model)
    raise ValueError(mode)


def run_channel(mol_ids: Sequence, lists: Dict, base: Dict, zs: Dict, s1s: Dict, pool: PoolIndex, mode: str, pos: int = 6, k: int = 5,
                fp_fn: Optional[Callable] = None, key_fn: Callable = metric_key, known_keys: Optional[Dict[str, str]] = None, model: Optional[dict] = None,
                budget_s: float = 1500.0, strict: bool = False, force_gate: bool = False, log: Callable = print):
    """The channel over a whole run: {mid: final SMILES list} -> ({mid: new list}, stats). base[mid] = the notebook's base list
    [smiles, keys, lib_max, scores, formulas, pool ids]; zs[mid] = logits; s1s[mid] = top score of the fold-safe ranker (absent =
    no fold-safe pass: the gate of 'gate' stays closed). A molecule without logits or a base list, a failure (unless strict) or
    the time budget leaves its list as it is; with mode 'gate' nothing is generated for a closed gate. force_gate (commit runs,
    whose molecules are library hits): every molecule with a score is merged, so a bug fails the commit instead of the rerun."""
    assert mode in MODES, mode
    model = model or MODEL; known = known_keys or {}; out = {}; t0 = time.time()
    st = dict(mode=mode, molecules=0, generated_for=0, rows=0, changed=0, gate_open=0, failed=0, skipped_budget=0, no_context=0, secs=0.0)
    for mid in mol_ids:
        fin = list(lists[mid]); out[mid] = fin; st['molecules'] += 1
        b = base.get(mid); z = zs.get(mid)
        if b is None or z is None or not len(b[0]):
            st['no_context'] += 1; continue
        lib = float(b[2]); s1 = s1s.get(mid)
        if mode == 'gate':
            if model is None or s1 is None or not (force_gate or gate_open(lib, s1, model)):
                continue
            st['gate_open'] += 1
        if time.time() - t0 > budget_s:
            st['skipped_budget'] += 1; continue
        try:
            excl = set(b[1])
            for s in fin:
                kk = known.get(s) or key_fn(s)
                if kk:
                    excl.add(kk)
            rows = generate(pool, z, b[4], excl, fp_fn, key_fn)
            new = place_slots(fin, [r['smiles'] for r in rows], pos, k) if mode == 'slots' else place_merge(fin, rows, s1, lib, model)
            assert len(new) <= 25 and all(new)
            st['generated_for'] += 1; st['rows'] += len(rows); st['changed'] += int(new != fin); out[mid] = new
        except Exception as e:
            if strict:
                raise
            st['failed'] += 1; log('GEN_ANALOG FAILED -> list unchanged', mid, repr(e)[:200])
    st['secs'] = round(time.time() - t0, 1)
    return out, st


# Placement model of policy 'gate': two logistic regressions and the s1 threshold, fitted on the natural list of every DEV molecule of the
# joint benchmark below lib_max 0.7 (16597 database rows, 16444 generated rows, 56 of them answers; scratch/gen3/g3_eval.py). The figures of
# reviews/gen3-2026-10-10.md are out of fold by scaffold; these constants are the fit on all of DEV. MODEL = None switches 'gate' off.
MODEL = dict(
    max_lib=0.7, s1_max=2.0307,                 # gate: lib_max below max_lib and the fold-safe ranker's top score below s1_max
    db=dict(mean=[2.316547, 0.040429, 1.312928, 0.052352],
            std=[0.819393, 0.196963, 1.925913, 0.464757],
            coef=[-1.285364, 0.003295, -0.221307, 0.460738], intercept=-4.956658),       # log rank, first row, s1, s1 x first row
    gen=dict(mean=[2.315091, 0.020732, 0.047859, 0.170244, 0.161129, 0.691394, -0.140305, -0.0, 1.274568, 3.213105, 1.314375, 0.35315, 0.371199, 0.135673, 0.201289, 0.068171],
             std=[0.81931, 0.123782, 0.213469, 0.33588, 0.314226, 0.136156, 0.174429, 0.99997, 0.583635, 0.076115, 1.914551, 0.22393, 0.483126, 0.342441, 0.400964, 0.252039],
             coef=[-0.787836, 0.184314, -0.324449, -0.415162, 0.765576, -0.087066, 0.440504, 0.991087, -0.498083, 0.246995, -0.803155, 0.075823, 0.230254, 0.39424, -0.262783, -0.328981],
             intercept=-8.124222))                 # columns GEN_FEATURES
