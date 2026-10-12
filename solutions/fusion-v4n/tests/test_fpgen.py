"""Tests of the fingerprint-conditioned generator (train/fpgen): tokeniser, formula parsing, noise model, model shapes and the formula-constrained decoder. Run without competition data."""
import os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'train', 'fpgen'))

import tok as T  # noqa: E402
import model as MD  # noqa: E402
import noise as NZ  # noqa: E402


def test_tokenizer_roundtrip():
    for s in ['CC(=O)Oc1ccccc1C(=O)O', 'C1CC2CCC1C2', 'CC[NH3+].[Cl-]'.split('.')[0], 'c1ccc2[nH]ccc2c1', 'CC(C)/C=C/C', 'C%12CCCCC%12']:
        t = T.tokenize(s)
        assert t is not None and ''.join(t) == s, s
    assert T.tokenize('CC(') is not None                      # the tokeniser does not judge grammar
    assert T.token_element('[nH]') == 'N' and T.token_element('Br') == 'Br' and T.token_element('c') == 'C'
    assert T.token_element('=') is None and T.token_element('1') is None


def test_vocab_elements_and_codec():
    v = T.Vocab(['C', 'c', 'N', 'O', '(', ')', '=', '1', 'Cl'])
    ids = v.encode('CC(=O)Oc1ccccc1'.replace('b', 'C'))
    assert ids is None or v.decode(ids) == 'CC(=O)Oc1ccccc1'
    assert v.elem[v.stoi['C'], T.EIX['C']] == 1 and v.elem[v.stoi['Cl'], T.EIX['Cl']] == 1
    assert v.elem[v.stoi['(']].sum() == 0
    assert v.encode('CBr') is None                            # a token outside the vocabulary


def test_usable_rejects_untracked_elements_and_fragments():
    assert T.usable(T.tokenize('CC(=O)O'))
    assert not T.usable(T.tokenize('CC.CC'))
    assert not T.usable(T.tokenize('C[Se]C'))


def test_formula_parsing_and_counts():
    c = T.parse_formula('C17H23NO4')
    assert c == {'C': 17, 'H': 23, 'N': 1, 'O': 4}
    v, h, other = T.counts_vec(c)
    assert h == 23 and v[T.EIX['C']] == 17 and v[T.EIX['O']] == 4 and not other
    assert T.parse_formula('C5H12NO+') == {'C': 5, 'H': 12, 'N': 1, 'O': 1}
    assert T.parse_formula('not a formula') is None
    _, _, other = T.counts_vec(T.parse_formula('C6H12Se'))
    assert other


def test_noise_corrupt_respects_scale_and_rates():
    rng = np.random.default_rng(0); nb = 200
    M = dict(mu1=np.full(nb, 2.0, np.float32), mu0=np.full(nb, -2.0, np.float32), sd=np.full(nb, 1.0, np.float32), freq=np.full(nb, 0.1, np.float32))
    M['w'], M['c'] = NZ.params(M['mu1'], M['mu0'], M['sd'], M['freq'])
    B = (rng.random((500, nb)) < 0.1).astype(np.uint8)
    assert np.array_equal(NZ.corrupt(B, M, rng, 0.0), B.astype(np.float32))
    p1 = NZ.corrupt(B, M, rng, 1.0); p3 = NZ.corrupt(B, M, rng, 3.0)
    on1 = p1[B > 0].mean(); off1 = p1[B == 0].mean()
    assert on1 > 0.5 > off1                                    # a well separated bit keeps its value on average
    assert abs(p3[B > 0].mean() - p3[B == 0].mean()) < abs(on1 - off1)   # more noise = less separation


def test_model_shapes_and_causality():
    torch.manual_seed(0)
    v = T.Vocab(['C', 'N', 'O', '(', ')', '=', '1'])
    m = MD.FPGen(len(v), nbits=64, d=32, layers=2, heads=2, drop=0.0, n_fp=2).eval()
    fp = torch.rand(3, 64); cnt = torch.zeros(3, T.NE + 1); cnt[:, T.EIX['C']] = 4
    x = torch.randint(3, len(v), (3, 7))
    z = m(fp, cnt, x)
    assert z.shape == (3, 7, len(v))
    y = x.clone(); y[:, -1] = (y[:, -1] + 1) % len(v) + 3 * 0
    z2 = m(fp, cnt, y)
    assert torch.allclose(z[:, :-2], z2[:, :-2], atol=1e-5)    # a change of the last token cannot move earlier predictions


def test_sampler_obeys_the_formula_constraint():
    torch.manual_seed(0)
    v = T.Vocab(['C', 'N', 'O', '(', ')', '=', '1'])
    m = MD.FPGen(len(v), nbits=64, d=32, layers=2, heads=2, drop=0.0, n_fp=2).eval()
    cnt = torch.zeros(4, T.NE + 1); cnt[:, T.EIX['C']] = 3; cnt[:, T.EIX['O']] = 1; cnt[:, T.NE] = 8
    ids, ll = m.sample(torch.rand(4, 64), cnt, torch.from_numpy(v.elem.astype('float32')), max_len=24, hard=True)
    for row in ids:
        s = v.decode(list(row))
        t = T.tokenize(s); e = [T.token_element(x) for x in t]
        assert e.count('C') <= 3 and e.count('O') <= 1
        if T.EOS in list(row):
            assert e.count('C') == 3 and e.count('O') == 1     # EOS is masked until every heavy atom is used
    assert ll.shape == (4,) and np.all(ll <= 0)


def test_sampler_without_the_constraint_is_unconstrained():
    torch.manual_seed(0)
    v = T.Vocab(['C', 'N', 'O'])
    m = MD.FPGen(len(v), nbits=16, d=16, layers=1, heads=2, drop=0.0, n_fp=1).eval()
    cnt = torch.zeros(2, T.NE + 1); cnt[:, T.EIX['C']] = 1
    ids, _ = m.sample(torch.rand(2, 16), cnt, torch.from_numpy(v.elem.astype('float32')), max_len=12, hard=False)
    assert ids.shape[0] == 2
