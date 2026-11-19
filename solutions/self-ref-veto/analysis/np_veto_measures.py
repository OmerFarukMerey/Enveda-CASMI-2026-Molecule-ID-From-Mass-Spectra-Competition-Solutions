"""Which spectrum similarity best separates 'same compound, other instrument' from 'wrong isomer' on NP timsTOF queries?
Measures (query = enveda-np-examples spectra of one structure, reference = train spectra from other libraries):
  merged  : TA/N-1c style merged spectra (0.01 Da bins, max intensity, top 120, sqrt), greedy cosine (current V-1)
  entropy : spectral entropy similarity (Li et al. 2021) on the same merged, unweighted spectra
  pairmax : max greedy sqrt-cosine over individual (query spectrum, reference spectrum) pairs of the same adduct
For each measure: pick the threshold at which 3% of true structures are vetoed, report isomer veto rates by Tanimoto.
usage: python -I np_veto_measures.py <competition_dir>
"""
import random, sys, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
sys.stdout.reconfigure(encoding="utf-8")
D = sys.argv[1]; ADD = ("[M+H]+", "[M-H]-"); t0 = time.time()
ds = pads.dataset(f"{D}/train.parquet")
meta = ds.to_table(columns=["inchikey14", "normalized_smiles", "molecular_formula", "ingest_lib", "adduct"]).to_pandas()
np_keys = sorted(set(meta.loc[meta.ingest_lib == "enveda-np-examples", "inchikey14"]))
others = meta[(meta.ingest_lib != "enveda-np-examples") & meta.adduct.isin(ADD)]
form = meta.drop_duplicates("inchikey14").set_index("inchikey14").molecular_formula.to_dict()
smi = meta.drop_duplicates("inchikey14").set_index("inchikey14").normalized_smiles.to_dict()
byf = others.drop_duplicates("inchikey14").groupby("molecular_formula").inchikey14.apply(list).to_dict()
rng = random.Random(5); iso = {}
for k in np_keys:
    pool = [x for x in byf.get(form[k], []) if x != k]; rng.shuffle(pool); iso[k] = pool[:40]
need = sorted(set(np_keys) | {x for v in iso.values() for x in v})
cols = ["inchikey14", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
Q = ds.to_table(columns=cols, filter=pc.field("ingest_lib") == "enveda-np-examples").to_pandas()
flt = pc.field("inchikey14").isin(pa.array(need)) & pc.field("adduct").isin(pa.array(list(ADD))) & (pc.field("ingest_lib") != "enveda-np-examples")
Rf = pd.concat([b.to_pandas() for b in ds.scanner(columns=cols, filter=flt, batch_size=8192).to_batches() if b.num_rows])
Rf = Rf.groupby(["inchikey14", "adduct"]).head(int(sys.argv[2]) if len(sys.argv) > 2 else 6)


def clean(mz, it, pr):
    mz = np.asarray(mz, float); it = np.asarray(it, float); it = it / max(float(it.max()) if len(it) else 1.0, 1e-9)
    k = mz < pr - 1.5; return mz[k], it[k]


def merged(g):
    pr = float(np.median(g.precursor_mz)); parts = [clean(a, b, pr) for a, b in zip(g.ms2_mzs, g.ms2_normalized_intensities)]
    mz = np.concatenate([p[0] for p in parts]); it = np.concatenate([p[1] for p in parts])
    if not len(mz): return None
    u, inv = np.unique(np.round(mz, 2), return_inverse=True); w = np.zeros(len(u)); np.maximum.at(w, inv, it)
    o = np.argsort(-w)[:120]; u, w = u[o], w[o]; oo = np.argsort(u); return u[oo], w[oo]


def single(g):
    out = []
    for a, b, p in zip(g.ms2_mzs, g.ms2_normalized_intensities, g.precursor_mz):
        mz, it = clean(a, b, float(p))
        if len(mz): o = np.argsort(mz); out.append((mz[o], it[o]))
    return out


def gcos(a, b, sq=True):
    (ma, wa), (mb, wb) = a, b
    if sq: wa, wb = np.sqrt(wa), np.sqrt(wb)
    wa = wa / max(np.linalg.norm(wa), 1e-12); wb = wb / max(np.linalg.norm(wb), 1e-12)
    lo = np.searchsorted(mb, ma - 0.0101); hi = np.searchsorted(mb, ma + 0.0101, "right"); cand = []
    for i in np.flatnonzero(hi > lo):
        for j in range(lo[i], hi[i]): cand.append((wa[i] * wb[j], i, j))
    cand.sort(reverse=True); ui, uj, s = set(), set(), 0.0
    for v, i, j in cand:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s += v
    return s


def entropy_sim(a, b):
    (ma, wa), (mb, wb) = a, b
    pa_ = wa / wa.sum(); pb_ = wb / wb.sum()
    lo = np.searchsorted(mb, ma - 0.0101); hi = np.searchsorted(mb, ma + 0.0101, "right")
    # match peaks greedily by intensity product, then build the merged distribution
    cand = []
    for i in np.flatnonzero(hi > lo):
        for j in range(lo[i], hi[i]): cand.append((pa_[i] * pb_[j], i, j))
    cand.sort(reverse=True); ui, uj, pairs = set(), set(), []
    for v, i, j in cand:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); pairs.append((i, j))
    H = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
    mix = [(pa_[i] + pb_[j]) / 2 for i, j in pairs] + [pa_[i] / 2 for i in range(len(pa_)) if i not in ui] + [pb_[j] / 2 for j in range(len(pb_)) if j not in uj]
    return 1 - (2 * H(np.array(mix)) - H(pa_) - H(pb_)) / np.log(4)


QM = {k: {ad: merged(g) for ad, g in grp.groupby("adduct")} for k, grp in Q.groupby("inchikey14")}
QS = {k: {ad: single(g) for ad, g in grp.groupby("adduct")} for k, grp in Q.groupby("inchikey14")}
RM = {k: {ad: merged(g) for ad, g in grp.groupby("adduct")} for k, grp in Rf.groupby("inchikey14")}
RS = {k: {ad: single(g) for ad, g in grp.groupby("adduct")} for k, grp in Rf.groupby("inchikey14")}
print(f"spectra prepared, {time.time() - t0:.0f}s", flush=True)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); fpc = {}
def fp(k):
    if k not in fpc:
        m = Chem.MolFromSmiles(smi[k]) if isinstance(smi.get(k), str) else None; fpc[k] = gen.GetFingerprint(m) if m is not None else None
    return fpc[k]
rows = []
for k in np_keys:
    if k not in QM: continue
    for kind, c in [("truth", k)] + [("isomer", x) for x in iso[k]]:
        if c not in RM: continue
        ads = [ad for ad in QM[k] if ad in RM[c] and QM[k][ad] is not None and RM[c][ad] is not None]
        if not ads: continue
        m1 = max(gcos(QM[k][ad], RM[c][ad]) for ad in ads)
        m2 = max(entropy_sim(QM[k][ad], RM[c][ad]) for ad in ads)
        m3 = max((gcos(a, b) for ad in ads for a in QS[k].get(ad, []) for b in RS[c].get(ad, [])), default=0.0)
        tan = DataStructs.TanimotoSimilarity(fp(k), fp(c)) if fp(k) is not None and fp(c) is not None else np.nan
        rows.append((kind, k, c, m1, m2, m3, tan))
R = pd.DataFrame(rows, columns=["kind", "query", "cand", "merged", "entropy", "pairmax", "tan"])
R.to_csv(f"np_veto_measures_cap{sys.argv[2] if len(sys.argv) > 2 else 6}.csv", index=False)
print(f"pairs: truth {int((R.kind == 'truth').sum())}, isomer {int((R.kind == 'isomer').sum())}, {time.time() - t0:.0f}s")
for meas in ("merged", "entropy", "pairmax"):
    for err in (0.02, 0.03, 0.05):
        tau = float(R[R.kind == "truth"][meas].quantile(err))
        parts = []
        for lo, hi in ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
            I = R[(R.kind == "isomer") & (R.tan >= lo) & (R.tan < hi)]
            parts.append(f"[{lo:.1f},{hi:.1f}) {(I[meas] < tau).mean():.2f}")
        print(f"{meas:8s} truth-error {err:.2f} (tau {tau:.3f}): isomer vetoed by Tanimoto " + " | ".join(parts))
