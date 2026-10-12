"""Training data of fpgen: the leak-safe structure universe, its vocabulary, and batches of (conditioning fingerprint, formula, SMILES token ids).

Universe: pool rows (train structures and COCONUT) that `sets.py` did not remove, tokenisable and at most MAX_TOKENS tokens; arm
`--data` chooses train-only, train + COCONUT (default) or plus a PubChem natural-product-like subset (pubchem.py).
Each sample: a canonical or a randomised SMILES of the structure (augmentation), its element counts, and a conditioning fingerprint
drawn by `noise.corrupt` from the structure's true pool bits.
usage: python data.py --vocab   (build the vocabulary)  |  python data.py --stats
"""
import argparse, json, os, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fpaths as P, tok as T, sets as S, noise as NZ

MAX_TOKENS = 160


def universe(arm='pool', log=print):
    """-> DataFrame(rid, can, rnd, pid) of the structures a run may train on, with the leak line printed."""
    PC = pd.read_parquet(P.CHEM + '/pool.parquet'); PM = pd.read_parquet(P.WORK + '/pool_meta.parquet', columns=['src', 'formula'])
    _, rm = S.load()
    keep = PC.ok.values & (rm == 0)
    if arm == 'train':
        keep &= np.isin(PM.src.values, (1, 3))
    elif arm == 'coconut':
        keep &= np.isin(PM.src.values, (2, 3))
    elif arm != 'pool':
        raise SystemExit('unknown arm ' + arm)
    D = PC[keep][['rid', 'can', 'rnd']].rename(columns={'rid': 'pid'}).reset_index(drop=True)
    D['nH'] = [(T.parse_formula(f) or {}).get('H', 0) for f in PM.formula.values[keep]]
    D['pcrow'] = -1
    S.leak_check(D.pid.values, log=log, extra=' (data arm %s)' % arm)
    return D


def with_pubchem(D, name='pc_mix', log=print):
    """Append the PubChem subset of pubchem.py (its own fingerprint file); its leak removals were applied when it was built."""
    Q = pd.read_parquet('%s/%s.parquet' % (P.CHEM, name)).reset_index(drop=True)
    Q = Q.assign(pid=-1, pcrow=np.arange(len(Q)))[['pid', 'can', 'rnd', 'nH', 'pcrow']]
    log('training universe: %d pool structures + %d PubChem structures (%s)' % (len(D), len(Q), name), flush=True)
    return pd.concat([D[['pid', 'can', 'rnd', 'nH', 'pcrow']], Q], ignore_index=True)


def encodable(D, vocab, max_len=MAX_TOKENS, log=print):
    """Rows whose canonical SMILES encodes with this vocabulary and is not longer than max_len (dropped: rare tokens)."""
    ok = np.array([(lambda e: e is not None and len(e) <= max_len)(vocab.encode(s)) for s in D.can.values])
    log('encodable with the vocabulary: %d of %d structures' % (int(ok.sum()), len(D)), flush=True)
    return D[ok].reset_index(drop=True)


def build_vocab(D, log=print):
    out = P.TOK + '/vocab.json'
    if os.path.exists(out):
        return T.Vocab.load(out)
    c = {}
    for s in D.can.values:
        for t in T.tokenize(s):
            c[t] = c.get(t, 0) + 1
    keep = sorted([t for t, n in c.items() if n >= 20], key=lambda t: -c[t])
    v = T.Vocab(keep); v.save(out)
    log('vocabulary: %d tokens (%d distinct seen, %d dropped below 20 occurrences)' % (len(v), len(c), len(c) - len(keep)), flush=True)
    return v


class Batches:
    """Infinite stream of training batches from a structure table (one worker process holds one of these)."""

    def __init__(self, D, vocab, seed=0, bs=256, aug=1.0, noise_scale=1.0, nbits=None, max_len=MAX_TOKENS):
        self.D, self.V, self.bs, self.aug, self.scale, self.max_len = D.reset_index(drop=True), vocab, bs, aug, noise_scale, max_len
        self.rng = np.random.default_rng(seed)
        self.M = NZ.load(); self.nbits = nbits or len(self.M['w'])
        self.fp = np.load(P.WORK + '/pool_fp.npy', mmap_mode='r')
        self.pid = self.D.pid.values
        self.pcrow = self.D.pcrow.values if 'pcrow' in self.D.columns else np.full(len(self.D), -1)
        self.pcfp = np.load(P.CHEM + '/pc_mix_fp.npy', mmap_mode='r') if (self.pcrow >= 0).any() else None
        self.can = self.D.can.values; self.rnd = [r.split(' ') if r else [] for r in self.D.rnd.values]
        self.H = self.D.nH.values.astype(np.float32)

    def one(self, i):
        s = self.can[i]
        if self.aug and self.rnd[i] and self.rng.random() < self.aug:
            s = self.rnd[i][self.rng.integers(len(self.rnd[i]))]
        ids = self.V.encode(s)
        return ids if ids is not None and len(ids) <= self.max_len else self.V.encode(self.can[i])

    def batch(self):
        ix = self.rng.integers(len(self.D), size=self.bs)
        seqs = [self.one(int(i)) for i in ix]
        keep = [k for k, s in enumerate(seqs) if s]
        ix = ix[keep]; seqs = [seqs[k] for k in keep]
        L = max(len(s) for s in seqs) + 2
        x = np.zeros((len(seqs), L), np.int16)
        for k, s in enumerate(seqs):
            x[k, 0] = T.BOS; x[k, 1:1 + len(s)] = s; x[k, 1 + len(s)] = T.EOS
        pk = np.zeros((len(ix), self.fp.shape[1]), np.uint8)
        a = self.pid[ix] >= 0
        if a.any():
            pk[a] = self.fp[self.pid[ix[a]]]
        if (~a).any():
            pk[~a] = self.pcfp[self.pcrow[ix[~a]]]
        B = np.unpackbits(pk, axis=1)[:, :self.nbits]
        fp = NZ.corrupt(B, self.M, self.rng, self.scale)
        cnt = np.zeros((len(seqs), T.NE + 1), np.float32)
        cnt[:, :T.NE] = self.V.elem[np.clip(x, 0, None)].sum(1)   # heavy counts of the written string = the molecule's formula
        cnt[:, T.NE] = self.H[ix]                                 # hydrogens of the structure (from the stored formula)
        return fp.astype(np.float32), cnt, x


def formula_counts(formulas):
    """Formula strings -> (n, NE + 1) float32 of heavy-element counts and hydrogens; rows of unparsable formulas are zero."""
    out = np.zeros((len(formulas), T.NE + 1), np.float32)
    for i, f in enumerate(formulas):
        c = T.parse_formula(f) if isinstance(f, str) else None
        if c:
            v, h, _ = T.counts_vec(c); out[i, :T.NE] = v; out[i, T.NE] = h
    return out


def counts_from_smiles(smiles_list):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    out = np.zeros((len(smiles_list), T.NE + 1), np.float32)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        v, h, _ = T.counts_vec(T.mol_counts(m)); out[i, :T.NE] = v; out[i, T.NE] = h
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='pool'); ap.add_argument('--pc', action='store_true'); ap.add_argument('--vocab', action='store_true'); ap.add_argument('--stats', action='store_true'); a = ap.parse_args()
    P.mk(); P.low_priority(); t0 = time.time()
    D = universe(a.arm)
    if a.arm == 'pool' and a.pc:
        D = with_pubchem(D)
    print('universe %s: %d structures' % (a.arm, len(D)), flush=True)
    V = build_vocab(D)
    if a.stats:
        n = [len(T.tokenize(s)) for s in D.can.values[:20000]]
        print(json.dumps(dict(vocab=len(V), tokens_mean=float(np.mean(n)), tokens_p90=float(np.quantile(n, .9)), tokens_max=int(np.max(n)),
                              randomised_per_structure=float(np.mean([len(r.split(' ')) if r else 0 for r in D.rnd.values[:20000]])), wall_s=round(time.time() - t0))), flush=True)
