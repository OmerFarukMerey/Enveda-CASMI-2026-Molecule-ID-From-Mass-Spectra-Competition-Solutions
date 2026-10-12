"""SMILES tokeniser of fpgen: atom-level tokens, the heavy-element vector of every token, vocabulary, formula parsing and counting."""
import json, re

import numpy as np

ELEMS = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']            # heavy elements that are tracked while decoding; others are dropped from the data
EIX = {e: i for i, e in enumerate(ELEMS)}
NE = len(ELEMS)
PAD, BOS, EOS = 0, 1, 2
SPECIAL = ['<pad>', '<bos>', '<eos>']
_TOK = re.compile(r'\[[^\]]+\]|Br|Cl|%\d\d|[BCNOSPFI]|[bcnosp]|[=#()\d]|[-+/\\.:~@*$]')
_BR = re.compile(r'^\[\d*([A-Z][a-z]?|[a-z]{1,2})')
_F = re.compile(r'([A-Z][a-z]?)(\d*)')
_AROM = {'b': 'B', 'c': 'C', 'n': 'N', 'o': 'O', 's': 'S', 'p': 'P', 'se': 'Se', 'as': 'As', 'te': 'Te'}


def tokenize(s):
    """SMILES -> list of tokens; None when a character is not covered."""
    t = _TOK.findall(s)
    return t if sum(map(len, t)) == len(s) else None


def token_element(t):
    """Element symbol of an atom token, None for bonds, branches and ring closures."""
    if t[0] == '[':
        m = _BR.match(t)
        if not m:
            return '?'
        e = m.group(1)
        return _AROM.get(e, e) if e[0].islower() else e
    if t in ('Br', 'Cl') or t in 'BCNOSPFI':
        return t
    return _AROM.get(t)


def parse_formula(f):
    """'C17H23NO4' (a trailing charge is ignored) -> {element: count}; None when it does not parse."""
    if not isinstance(f, str) or not f:
        return None
    f = re.sub(r'[+-]\d*$', '', f); c = {}; n = 0
    for e, k in _F.findall(f):
        c[e] = c.get(e, 0) + (int(k) if k else 1); n += len(e) + len(k)
    return c if n == len(f) and c else None


def counts_vec(c):
    """{element: count} -> (heavy counts in ELEMS order, hydrogens, True when another heavy element is present)."""
    v = np.zeros(NE, np.int16)
    for e, k in c.items():
        if e in EIX:
            v[EIX[e]] = k
    return v, int(c.get('H', 0)), any(e not in EIX and e != 'H' for e in c)


def mol_counts(mol):
    """RDKit molecule -> {element: count} with implicit hydrogens."""
    c = {}; h = 0
    for a in mol.GetAtoms():
        s = a.GetSymbol(); c[s] = c.get(s, 0) + 1; h += a.GetTotalNumHs()
    if h:
        c['H'] = c.get('H', 0) + h
    return c


class Vocab:
    """Token table with the per-token heavy-element matrix (V, NE)."""

    def __init__(self, tokens):
        self.itos = list(SPECIAL) + [t for t in tokens if t not in SPECIAL]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.elem = np.zeros((len(self.itos), NE), np.int8)
        for i, t in enumerate(self.itos[3:], 3):
            e = token_element(t)
            if e is not None:
                assert e in EIX, f'token {t} has an untracked element'
                self.elem[i, EIX[e]] = 1

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        """SMILES -> token ids without BOS / EOS; None when a token is unknown."""
        t = tokenize(s)
        if t is None:
            return None
        try:
            return [self.stoi[x] for x in t]
        except KeyError:
            return None

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            if i > EOS:
                out.append(self.itos[i])
        return ''.join(out)

    def save(self, path):
        json.dump(self.itos[3:], open(path, 'w'))

    @staticmethod
    def load(path):
        return Vocab(json.load(open(path)))


def usable(tokens):
    """True when every atom token carries a tracked element and the string is one component."""
    for t in tokens:
        if t == '.' or t == '*' or t == '$':
            return False
        e = token_element(t)
        if e is not None and e not in EIX:
            return False
    return True
