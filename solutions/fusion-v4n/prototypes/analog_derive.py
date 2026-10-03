"""Prototype: rule-based derivatives of analog structures for class-3 molecules, ranked by multi-parent support.

Not wired into the notebook. Own code (RDKit only), written for the coverage study of 2026-10-02
(reviews/coverage-2026-10-02.md). It differs from the v1 engine's generator (casmi/derive.py of casmi26-v4b-models,
46 hand-written reactions, 6 parents, first-come cap of 150) in three ways:
  1. substituents are a table (fragment SMILES x attachment site); every entry gives an addition and its exact
     reversal, so the rule set is symmetric and easy to extend (112 operations);
  2. all parents are expanded first, then products are ranked by support (sum of parent weight^4 over the distinct
     parents that derive the product, fewer steps first) and cut, instead of keeping the first 150 found;
  3. the mass filter is a ppm window on the product's exact mass (default 10 ppm).

    from analog_derive import propose
    cands = propose([(parent_smiles, weight), ...], target_mass, ppm=10.0, top=100)   # -> list of Candidate

    python analog_derive.py --self-test
    python analog_derive.py --parent "COc1ccc(C=CC(=O)O)cc1O" --target 356.1107
"""
from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt

RDLogger.DisableLog('rdApp.*')
H_MASS = 1.00782503223

# attachment sites: name -> SMARTS of the atom that carries a hydrogen to be replaced (map number 1)
SITES: Dict[str, str] = {
    'O': '[OX2H1;!$(OC=O);!$(O[S,P]):1]',                 # alcohol / phenol
    'N': '[NX3;H1,H2;!$(NC=[O,S,N]);!$(N[O,N,S]):1]',      # amine
    'c': '[cH1:1]',                                        # aromatic CH
    'C': '[CX4;H1,H2,H3;!$(C[O,N,S]):1]',                   # aliphatic CH without a heteroatom neighbour
    'A': '[CX3:1](=O)[OX2H1]',                             # carboxylic acid (ester / amide formation at the OH)
}
# what the reversal must see on the other side of the bond (site atom, map number 1)
SITE_BACK: Dict[str, str] = {'O': '[OX2:1]', 'N': '[NX3:1]', 'c': '[c:1]', 'C': '[CX4:1]'}

# substituent: (name, fragment SMILES starting at the attaching atom, sites it is offered on)
SUBST: List[Tuple[str, str, str]] = [
    ('hydroxy', 'O', 'cC'), ('methyl', 'C', 'ONc'), ('ethyl', 'CC', 'O'), ('methoxy', 'OC', 'c'),
    ('hydroxymethyl', 'CO', 'c'), ('formyl', 'C=O', 'cN'), ('carboxy', 'C(=O)O', 'c'), ('amino', 'N', 'c'),
    ('chloro', 'Cl', 'c'), ('bromo', 'Br', 'c'),
    ('acetyl', 'C(C)=O', 'ON'), ('propionyl', 'C(=O)CC', 'O'), ('isobutyryl', 'C(=O)C(C)C', 'O'),
    ('isovaleryl', 'C(=O)CC(C)C', 'O'), ('methylbutyryl', 'C(=O)C(C)CC', 'O'), ('tigloyl', 'C(=O)C(C)=CC', 'O'),
    ('senecioyl', 'C(=O)C=C(C)C', 'O'), ('benzoyl', 'C(=O)c1ccccc1', 'O'), ('hydroxybenzoyl', 'C(=O)c1ccc(O)cc1', 'O'),
    ('vanilloyl', 'C(=O)c1ccc(O)c(OC)c1', 'O'), ('galloyl', 'C(=O)c1cc(O)c(O)c(O)c1', 'O'),
    ('cinnamoyl', 'C(=O)C=Cc1ccccc1', 'O'), ('coumaroyl', 'C(=O)C=Cc1ccc(O)cc1', 'O'),
    ('caffeoyl', 'C(=O)C=Cc1ccc(O)c(O)c1', 'O'), ('feruloyl', 'C(=O)C=Cc1ccc(O)c(OC)c1', 'O'),
    ('sinapoyl', 'C(=O)C=Cc1cc(OC)c(O)c(OC)c1', 'O'), ('malonyl', 'C(=O)CC(=O)O', 'O'), ('succinyl', 'C(=O)CCC(=O)O', 'O'),
    ('hmg', 'C(=O)CC(C)(O)CC(=O)O', 'O'),
    ('hexose', 'C1OC(CO)C(O)C(O)C1O', 'OcN'), ('deoxyhexose', 'C1OC(C)C(O)C(O)C1O', 'O'), ('pentose', 'C1OCC(O)C(O)C1O', 'O'),
    ('hexuronic', 'C1OC(C(=O)O)C(O)C(O)C1O', 'O'), ('glcnac', 'C1OC(CO)C(O)C(O)C1NC(C)=O', 'O'),
    ('sulfate', 'S(=O)(=O)O', 'O'), ('phosphate', 'P(=O)(O)O', 'O'),
    ('prenyl', 'CC=C(C)C', 'Oc'), ('geranyl', 'CC=C(C)CCC=C(C)C', 'c'),
    ('methyl_ester', 'OC', 'A'), ('ethyl_ester', 'OCC', 'A'), ('amide', 'N', 'A'), ('glycine_amide', 'NCC(=O)O', 'A'),
]

# skeleton / redox moves that are not substituent exchanges: name -> (reaction SMARTS, mass change)
C12, O16 = 12.0, 15.99491461957
REDOX: Dict[str, Tuple[str, float]] = {
    'reduce_CC': ('[C:1]=[C:2]>>[C:1][C:2]', 2 * H_MASS),
    'reduce_CO': ('[C;!$(C[O,N]):1]=[O:2]>>[C:1][OH:2]', 2 * H_MASS),
    'desaturate': ('[CX4;H1,H2:1][CX4;H1,H2:2]>>[C:1]=[C:2]', -2 * H_MASS),
    'oxidise_alcohol': ('[CX4;H1,H2:1][OX2H1:2]>>[C:1]=[O:2]', -2 * H_MASS),
    'keto_at_CH2': ('[CX4;H2;!$(C[O,N]):1]>>[C:1]=O', O16 - 2 * H_MASS),
    'deketo': ('[CX3;!$(C[O,N]):1]=O>>[CH2:1]', -(O16 - 2 * H_MASS)),
    'aldehyde_to_acid': ('[CX3H1:1]=[O:2]>>[C:1](=[O:2])O', O16),
    'methyl_to_acid': ('[CX4H3:1]>>[C:1](=O)O', 2 * O16 - 2 * H_MASS),
    'epoxidise': ('[C:1]=[C:2]>>[C:1]1[C:2]O1', O16),
    'hydrate': ('[C:1]=[C:2]>>[C:1][C:2]O', O16 + 2 * H_MASS),
    'dehydrate': ('[CX4;H1,H2:1][C:2][OX2H1]>>[C:1]=[C:2]', -(O16 + 2 * H_MASS)),
    'decarboxylate': ('[#6:1]C(=O)[OX2H1]>>[#6:1]', -(C12 + 2 * O16)),
}


@dataclass(frozen=True)
class Op:
    name: str
    smarts: str
    dmass: float


@dataclass
class Candidate:
    smiles: str
    key: str
    mass: float
    support: float = 0.0
    n_parents: int = 0
    steps: int = 9
    routes: List[Tuple[int, Tuple[str, ...]]] = field(default_factory=list)


def _frag_dmass(frag: str) -> float:
    """Mass change when a hydrogen is replaced by the fragment = mass(H-fragment) - 2 H."""
    return ExactMolWt(Chem.MolFromSmiles('[H]' + frag)) - 2 * H_MASS


def _exact_fragment_smarts(frag: str) -> str:
    """SMARTS of the fragment in which every heavy atom keeps exactly its degree and hydrogen count, so the reversal
    only fires on an unmodified substituent (a methylated sugar is not a hexose). Starts at the attaching atom."""
    m = Chem.RWMol(Chem.MolFromSmiles('[*:9]' + frag))
    for a in list(m.GetAtoms()):
        if a.GetAtomMapNum() == 9:
            continue
        sym = a.GetSymbol()
        if a.GetIsAromatic():
            sym = sym.lower()
        m.ReplaceAtom(a.GetIdx(), Chem.AtomFromSmarts(f'[{sym};D{a.GetDegree()};H{a.GetTotalNumHs()}]'))
    sma = Chem.MolToSmarts(m)
    head = sma.index(']') + 1                               # the dummy is written first
    assert sma.startswith('[#0') and Chem.MolFromSmarts('[*]' + sma[head:]) is not None, sma
    return sma[head:]                                       # begins with the bond symbol to the site atom


_SITE_SYMBOL = {'O': 'O', 'N': 'N', 'c': 'c', 'C': 'C'}


def build_ops() -> List[Op]:
    ops: List[Op] = []
    for name, frag, sites in SUBST:
        dm = _frag_dmass(frag)
        back = _exact_fragment_smarts(frag)
        for s in sites:
            if s == 'A':                                    # acid OH replaced by O-R / N-R
                ops.append(Op(f'+{name}@A', f'[CX3:1](=[O:2])[OX2H1]>>[C:1](=[O:2]){frag}', dm - O16))
                ops.append(Op(f'-{name}@A', f'[CX3:1](=[O:2]){back}>>[C:1](=[O:2])O', -(dm - O16)))
            else:
                ops.append(Op(f'+{name}@{s}', f'{SITES[s]}>>[{_SITE_SYMBOL[s]}:1]{frag}', dm))
                ops.append(Op(f'-{name}@{s}', f'{SITE_BACK[s]}{back}>>[{_SITE_SYMBOL[s]}:1]', -dm))
    for name, (sm, dm) in REDOX.items():
        ops.append(Op(name, sm, dm))
    return ops


_OPS: Optional[List[Op]] = None
_RXN: Dict[str, object] = {}
_PAIRS: Optional[Tuple[List[float], List[Tuple[int, int]]]] = None


def ops() -> List[Op]:
    global _OPS
    if _OPS is None:
        _OPS = build_ops()
    return _OPS


def _rxn(op: Op):
    r = _RXN.get(op.name)
    if r is None:
        r = AllChem.ReactionFromSmarts(op.smarts)
        _RXN[op.name] = r
    return r


def _pairs():
    """All unordered pairs of operations sorted by total mass change (for a binary search on the target delta)."""
    global _PAIRS
    if _PAIRS is None:
        o = ops(); items = []
        for i in range(len(o)):
            for j in range(i, len(o)):
                items.append((o[i].dmass + o[j].dmass, (i, j)))
        items.sort()
        _PAIRS = ([x[0] for x in items], [x[1] for x in items])
    return _PAIRS


def apply(mol, op: Op, limit: int = 32) -> list:
    out, seen = [], set()
    try:
        prods = _rxn(op).RunReactants((mol,))
    except Exception:
        return out
    for p in prods:
        m = p[0]
        try:
            Chem.SanitizeMol(m)
            s = Chem.MolToSmiles(m)
        except Exception:
            continue
        if s in seen:
            continue
        seen.add(s); out.append(Chem.MolFromSmiles(s))
        if len(out) >= limit:
            break
    return [m for m in out if m is not None]


def fast_key(mol) -> Optional[str]:
    """First InChIKey block (connectivity; stereo-free). The metric key adds a tautomer canonicalisation."""
    try:
        return Chem.MolToInchiKey(mol)[:14]
    except Exception:
        return None


def derive(parent_smiles: str, target_mass: float, ppm: float = 10.0, max_steps: int = 2, budget: int = 600,
           per_op: int = 24, fan: int = 16):
    """Products of 1 or 2 operations on the parent whose exact mass is within ppm of the target.
    Yields (mol, route) with route = tuple of operation names."""
    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None:
        return
    tol = target_mass * ppm * 1e-6
    delta = target_mass - ExactMolWt(parent)
    o = ops()
    for op in o:                                            # one step
        if abs(op.dmass - delta) <= tol:
            for m in apply(parent, op, per_op):
                yield m, (op.name,)
            budget -= 1
    if max_steps < 2:
        return
    masses, idx = _pairs()
    lo = bisect.bisect_left(masses, delta - tol); hi = bisect.bisect_right(masses, delta + tol)
    first_cache: Dict[int, list] = {}
    for i, j in idx[lo:hi]:
        for a, b in ((i, j), (j, i)) if i != j else ((i, j),):
            if budget <= 0:
                return
            if a not in first_cache:
                first_cache[a] = apply(parent, o[a], fan); budget -= 1
            for mid in first_cache[a]:
                if budget <= 0:
                    return
                budget -= 1
                for m in apply(mid, o[b], per_op):
                    yield m, (o[a].name, o[b].name)


def propose(parents: Sequence[Tuple[str, float]], target_mass: float, ppm: float = 10.0, max_steps: int = 2,
            top: Optional[int] = 100, exclude_keys: Iterable[str] = (), per_parent: Optional[int] = None,
            budget: int = 600) -> List[Candidate]:
    """parents: (smiles, weight) with weight = spectral similarity of the analog (or any 0..1 confidence).
    Returns candidates sorted by support = sum over distinct parents of weight^4 (ties: fewer steps, more parents)."""
    excl = set(exclude_keys); tol = target_mass * ppm * 1e-6
    out: Dict[str, Candidate] = {}
    for pi, (psmi, w) in enumerate(parents):
        pk = None
        pm = Chem.MolFromSmiles(psmi)
        if pm is not None:
            pk = fast_key(pm)
        got = set()
        for m, route in derive(psmi, target_mass, ppm, max_steps, budget):
            k = fast_key(m)
            if k is None or k in excl or k == pk:
                continue
            mass = ExactMolWt(m)
            if abs(mass - target_mass) > tol:
                continue
            c = out.get(k)
            if c is None:
                c = out[k] = Candidate(Chem.MolToSmiles(m), k, mass)
            if k not in got:
                got.add(k); c.support += float(w) ** 4; c.n_parents += 1
            c.steps = min(c.steps, len(route))
            if len(c.routes) < 4:
                c.routes.append((pi, route))
            if per_parent and len(got) >= per_parent:
                break
    res = sorted(out.values(), key=lambda c: (-c.support, c.steps, -c.n_parents, c.smiles))
    return res[:top] if top else res


def self_test() -> int:
    """Every operation must change the exact mass by its declared amount on at least one probe molecule, and every
    substituent addition must be undone by its reversal. Returns the number of failures."""
    probes = ['OC(=O)C=Cc1ccc(O)c(OC)c1', 'CC(=O)OC1CCC(N)CC1C=C', 'NCCc1ccc(O)cc1', 'OCC1OC(Oc2ccccc2C(C)=O)C(O)C(O)C1O',
              'CC(C)=CCc1ccccc1OS(=O)(=O)O', 'O=CC1CCCCC1C(=O)NCC(=O)O', 'COC(=O)c1cc(Cl)ccc1OC(=O)C=Cc1ccc(O)cc1']
    probes = [Chem.MolFromSmiles(s) for s in probes]
    bad = 0
    o = ops(); byname = {x.name: x for x in o}
    for op in o:
        ok = None
        for p in probes:
            pr = apply(p, op, 4)
            if pr:
                ok = all(abs(ExactMolWt(m) - ExactMolWt(p) - op.dmass) < 1e-4 for m in pr)
                break
        if ok is False:
            bad += 1; print('mass mismatch', op.name)
        if op.name.startswith('+'):
            rev = byname['-' + op.name[1:]]
            done = False
            for p in probes:
                for m in apply(p, op, 4):
                    back = {fast_key(x) for x in apply(m, rev, 16)}
                    if fast_key(p) not in back:
                        bad += 1; print('not reversible', op.name, Chem.MolToSmiles(p), '->', Chem.MolToSmiles(m))
                    done = True
                    break
                if done:
                    break
            if not done:
                print('note: no probe exercises', op.name)
    print(f'{len(o)} operations, {bad} failures')
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--parent', action='append', default=[]); ap.add_argument('--target', type=float)
    ap.add_argument('--ppm', type=float, default=10.0); ap.add_argument('--top', type=int, default=25)
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(1 if self_test() else 0)
    if a.parent and a.target:
        for c in propose([(p, 1.0) for p in a.parent], a.target, a.ppm, top=a.top):
            print(f'{c.support:6.2f} parents {c.n_parents} steps {c.steps} {c.mass:.4f} {c.smiles}  {c.routes[0][1]}')
