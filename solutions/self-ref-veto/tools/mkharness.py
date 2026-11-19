"""Build a local stand-in for /kaggle/working + /kaggle/input so final-stage cells can be crash-tested on CPU.

For each visible-test molecule: its true structure (recovered by exact spectrum match to train) plus 24 decoys from
train with the same molecular formula when possible (else nearest mass), shuffled with the truth at a random rank.
Writes: <out>/working/submission.csv, frozen_candidate_evidence.parquet, neural_*.parquet; <out>/input/comp/{test,train}
are symlink-free copies by path (train is read in place). Labels go to <out>/truth.csv. Only for plumbing tests:
the visible test is drawn from train, so any score computed here is leaked and meaningless as a quality estimate.
usage: python -I mkharness.py <competition_dir> <out_dir>
"""
import os, sys, random
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

sys.stdout.reconfigure(encoding="utf-8")
D, OUT = sys.argv[1], sys.argv[2]
rng = random.Random(7)
W = os.path.join(OUT, "working"); os.makedirs(W, exist_ok=True)
te = pd.read_parquet(f"{D}/test.parquet")
train = ds.dataset(f"{D}/train.parquet")
st = train.to_table(columns=["inchikey14", "normalized_smiles", "molecular_formula", "precursor_mz", "adduct"]).to_pandas()
st = st.drop_duplicates("inchikey14")
byf = {f: g.normalized_smiles.tolist() for f, g in st.groupby("molecular_formula")}

# truth by exact spectrum signature
def sig(p, m):
    a = np.asarray(m, float); return (round(float(p), 4), len(a), tuple(np.round(a[:6], 4)))
want = {sig(p, m) for p, m in zip(te.precursor_mz, te.ms2_mzs)}
pset = {k[0] for k in want}
truth_sig = {}
for b in train.scanner(columns=["normalized_smiles", "inchikey14", "molecular_formula", "precursor_mz", "ms2_mzs"],
                       filter=(pc.field("precursor_mz") >= min(pset) - 0.001) & (pc.field("precursor_mz") <= max(pset) + 0.001),
                       batch_size=65536).to_batches():
    t = b.to_pandas(); t = t[t.precursor_mz.round(4).isin(pset)]
    for r in t.itertuples(index=False):
        k = sig(r.precursor_mz, r.ms2_mzs)
        if k in want: truth_sig.setdefault(k, (r.normalized_smiles, r.inchikey14, r.molecular_formula))
rows, ev, truth = [], [], []
for mid, g in te.groupby("molecule_id", sort=False):
    hits = [truth_sig.get(sig(p, m)) for p, m in zip(g.precursor_mz, g.ms2_mzs)]
    hits = [h for h in hits if h]
    if not hits:
        continue
    smi, k14, f = hits[0]
    pool = [s for s in byf.get(f, []) if s != smi]
    rng.shuffle(pool)
    dec = pool[:24]
    if len(dec) < 24:  # pad with random structures
        dec += rng.sample(st.normalized_smiles.tolist(), 24 - len(dec))
    lst = dec[:]; pos = rng.randrange(0, 6) if rng.random() < 0.8 else rng.randrange(6, 25); lst.insert(pos, smi)
    lst = list(dict.fromkeys(lst))[:25]
    rows.append((str(mid), ";".join(lst))); truth.append((str(mid), smi, k14))
    lib = rng.choice([0.3, 0.6, 0.8, 0.95])
    for r, s in enumerate(lst, 1):
        ev.append(dict(molecule_id=str(mid), smiles=s, metric_key=None, baseline_rank=r, processed=True, library_max=lib,
                       proxy_gl=False, ice=rng.random(), gl=rng.random(), main_rank=r, prefusion_rank=r, engine2_rank=0,
                       pubchem_rank=0, neural_ok=True, neural_score=rng.gauss(0, 1), neural_cos=rng.random(),
                       neural_fp=rng.random(), neural_std=0.1, n_spectra=len(g), n_adducts=g.adduct.nunique()))
sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
sub.to_csv(os.path.join(W, "submission.csv"), index=False)
evd = pd.DataFrame(ev)
evd.drop(columns=["neural_ok", "neural_score", "neural_cos", "neural_fp", "neural_std", "n_spectra", "n_adducts"]).to_parquet(
    os.path.join(W, "frozen_candidate_evidence.parquet"), index=False)
for name in ("neural_public.parquet", "neural_ours.parquet", "neural_candidate_features.parquet"):
    evd.to_parquet(os.path.join(W, name), index=False)
pd.DataFrame(truth, columns=["molecule_id", "smiles", "ik14"]).to_csv(os.path.join(OUT, "truth.csv"), index=False)
te[te.molecule_id.astype(str).isin(set(sub.molecule_id))].to_parquet(os.path.join(OUT, "test.parquet"), index=False)
print("harness molecules", len(sub), "candidate rows", len(evd), "->", OUT)
