"""Adduct-evidence term for joined PubChem rows (flag ADD_JOIN_MU, off by default; reviews/newinfo-2026-10-08.md).

Which adducts a molecule was acquired with says something about its functional groups ([M-H]- with acids, the formate
adduct without, [M+NH4]+ / [M+Na]+ without a basic nitrogen ...). Seven LightGBM classifiers (dataset
casmi26-ours-adduct-prior, fitted on the Enveda timsTOF molecules of train.parquet only) predict from a structure
whether each adduct is observed; a candidate's score is the Bernoulli log-likelihood of the query's observed adduct
set, standardised over the rows of its list. fusion_core adds it to joined rows only (pool id -2), inside their slots.
"""
import os

import numpy as np

TARGETS = ('H', 'mH', 'Na', 'NH4', 'FA', 'K', 'Cl')
ADDUCTS = {'H': ('[M+H]+',), 'mH': ('[M-H]-',), 'Na': ('[M+Na]+', '[2M+Na]+'), 'NH4': ('[M+NH4]+',), 'FA': ('[M+CH2O2-H]-',),
           'K': ('[M+K]+',), 'Cl': ('[M+Cl]-',)}
SMARTS = dict(
    oh_ali='[OX2H][CX4]', oh_phen='[OX2H]c', cooh='[CX3](=O)[OX2H1,OX1-]', ester='[CX3](=O)[OX2][#6]', lactone='[CX3;R](=O)[OX2;R]',
    ketone='[#6][CX3](=O)[#6]', aldehyde='[CX3H1](=O)[#6]', ether='[OD2]([#6])[#6]', ome='[OX2]([CH3])[#6]', amide='[CX3](=O)[NX3]',
    amine1='[NX3;H2;!$(NC=[O,S,N]);!$(Na)]', amine2='[NX3;H1;!$(NC=[O,S,N]);!$(Na);!$(N-a)]',
    amine3='[NX3;H0;!$(NC=[O,S,N]);!$(N-a);!$(N=*)]', aniline='[NX3;!$(NC=O)]c', n_arom='[nX2]', nh_arom='[nH]',
    quatn='[N+,n+;!$([N+][O-])]', nitro='[N+](=O)[O-]', nitrile='C#N', sulfonamide='S(=O)(=O)N', sulfo='S(=O)(=O)[OH,O-]',
    thio='[#16X2]', halo='[F,Cl,Br,I]', acetal='[CX4]([OX2])([OX2])', enone='C=CC(=O)', epoxide='C1OC1', guan='NC(=N)N',
    phosph='P(=O)', enol='[OX2H]C=C', carbonyl='[CX3]=O', c_dbl='C=C', urea='NC(=O)N', carbamate='NC(=O)O')
SUGAR = '[OX2;R]1[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R]([OX2])[CX4;R][CX4;R]1'
N_DESC, N_FP = len(SMARTS) + 16, 1024
_RD = {}


def _rd():
    if not _RD:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Crippen, rdFingerprintGenerator, rdMolDescriptors
        RDLogger.DisableLog('rdApp.*')
        _RD.update(Chem=Chem, Crippen=Crippen, md=rdMolDescriptors, pats=[Chem.MolFromSmarts(v) for v in SMARTS.values()],
                   sugar=Chem.MolFromSmarts(SUGAR), gen=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_FP))
    return _RD


def featurize(smiles):
    """(n, 49 + 1024) float32: functional-group counts, simple descriptors, hashed Morgan counts (radius 2, clipped at
    255). An unparseable SMILES gives NaN descriptors and zero counts (LightGBM takes NaN as missing)."""
    r = _rd(); Chem, md = r['Chem'], r['md']
    X = np.zeros((len(smiles), N_DESC + N_FP), np.float32); X[:, :N_DESC] = np.nan
    for j, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            continue
        v = [len(m.GetSubstructMatches(p)) for p in r['pats']]
        el = [a.GetSymbol() for a in m.GetAtoms()]
        v += [m.GetNumHeavyAtoms(), el.count('C'), el.count('N'), el.count('O'), el.count('S'), md.CalcNumRings(m),
              md.CalcNumAromaticRings(m), md.CalcFractionCSP3(m), md.CalcTPSA(m), r['Crippen'].MolLogP(m), md.CalcNumHBD(m),
              md.CalcNumHBA(m), md.CalcNumRotatableBonds(m), Chem.GetFormalCharge(m),
              len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)),
              float(len(m.GetSubstructMatches(r['sugar'])))]
        X[j, :N_DESC] = v
        X[j, N_DESC:] = np.minimum(r['gen'].GetCountFingerprintAsNumPy(m), 255)
    return X


def observed(adducts):
    """Presence vector (one value per target) of a molecule's observed adduct strings."""
    a = set(str(x) for x in adducts)
    return np.array([float(any(x in a for x in ADDUCTS[t])) for t in TARGETS])


def log_likelihood(P, y):
    """P (n rows, 7) predicted presence probabilities, y (7,) observed presence -> (n,) Bernoulli log-likelihood."""
    P = np.clip(np.asarray(P, np.float64), 1e-4, 1 - 1e-4)
    return (y[None, :] * np.log(P) + (1 - y[None, :]) * np.log(1 - P)).sum(1)


def standardise(v):
    v = np.asarray(v, np.float64)
    return np.nan_to_num((v - v.mean()) / (v.std() + 1e-9))


class Scorer:
    """add_score callable of fusion_core.build_submission: (molecule id, list SMILES) -> one standardised value per row,
    or None when the molecule has no adduct record (the list then takes the standard path)."""

    def __init__(self, models, adducts_of):
        self.models = models                      # {target: object with predict(X) -> probability}
        self.y = {m: observed(a) for m, a in adducts_of.items()}
        self.cache = {}; self.n = 0

    def __call__(self, mid, smiles):
        y = self.y.get(mid)
        if y is None or len(smiles) < 2:
            return None
        new = [s for s in dict.fromkeys(smiles) if s not in self.cache]
        if new:
            X = featurize(new)
            P = np.stack([np.asarray(self.models[t].predict(X), np.float64) for t in TARGETS], 1)
            self.cache.update(zip(new, P))
        self.n += 1
        return [float(x) for x in standardise(log_likelihood(np.stack([self.cache[s] for s in smiles]), y))]


def load(folder, adducts_of):
    """Scorer from the dataset folder (adduct_<target>.txt LightGBM model files)."""
    import lightgbm as lgb
    return Scorer({t: lgb.Booster(model_file=os.path.join(folder, f'adduct_{t}.txt')) for t in TARGETS}, adducts_of)
