"""How well does train-analog (TA-1) evidence separate natural-product isomers on timsTOF queries, leak-free?

Queries: the 250 enveda-np-examples structures (timsTOF, same instrument as the hidden test), [M+H]+ / [M-H]- spectra.
Candidates: the truth + up to 24 train structures with the same molecular formula (random sample, seed fixed).
Evidence exactly as TA-1: for each candidate, its 8 most similar train structures with Morgan Tanimoto in [0.40, 0.85),
score = sum(sim * modified-cosine(query, analog spectrum)) over analogs with >= 4 matched peaks.
Leak control: every spectrum of the TRUE structure is removed from the reference side for its own query (class-2
situation), and all enveda-np-examples rows are removed from the reference side for every query.
Reports: MRR / top-1 of TA evidence alone vs chance, by isomer-group size, and how often evidence is zero.
usage: python -I np_ta_power.py <competition_dir> [max_queries]
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
D = sys.argv[1]; MAXQ = int(sys.argv[2]) if len(sys.argv) > 2 else 250
ADD = ("[M+H]+", "[M-H]-")
t0 = time.time()
ds = pads.dataset(f"{D}/train.parquet")
meta = ds.to_table(columns=["inchikey14", "normalized_smiles", "molecular_formula", "ingest_lib"]).to_pandas()
npk = set(meta.loc[meta.ingest_lib == "enveda-np-examples", "inchikey14"])
st = meta[meta.ingest_lib != "enveda-np-examples"].drop_duplicates("inchikey14").reset_index(drop=True)
npst = meta[meta.ingest_lib == "enveda-np-examples"].drop_duplicates("inchikey14").reset_index(drop=True)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps, keys = [], []
for k, s in zip(st.inchikey14, st.normalized_smiles):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    if m is not None:
        fps.append(gen.GetFingerprint(m)); keys.append(k)
print(f"reference structures (no np-examples) {len(fps)}, {time.time() - t0:.0f}s", flush=True)
byf = st.groupby("molecular_formula").apply(lambda g: list(zip(g.inchikey14, g.normalized_smiles))).to_dict()


def build_sp(df, idcol):
    SP = {}
    for (m, ad), g in df.groupby([idcol, "adduct"]):
        if str(ad) not in ADD: continue
        pr = float(np.median(pd.to_numeric(g.precursor_mz, errors="coerce")))
        mz = np.concatenate([np.asarray(x, float) for x in g.ms2_mzs])
        it = np.concatenate([np.asarray(x, float) / max(float(np.max(x)) if len(x) else 1.0, 1e-9) for x in g.ms2_normalized_intensities])
        k = mz < pr - 1.5; mz, it = mz[k], it[k]
        if not len(mz): continue
        u, inv = np.unique(np.round(mz, 2), return_inverse=True); w = np.zeros(len(u)); np.maximum.at(w, inv, it)
        o = np.argsort(-w)[:120]; u, w = u[o], np.sqrt(w[o]); oo = np.argsort(u)
        SP.setdefault(str(m), {})[str(ad)] = (u[oo], (w / max(np.linalg.norm(w), 1e-12))[oo], pr)
    return SP


def mcos(a, b):
    (ma, ia_, pa_), (mb, ib, pb) = a, b; sh = pa_ - pb; cand = []
    for off in ((0.0, sh) if abs(sh) > 0.02 else (0.0,)):
        lo = np.searchsorted(mb, ma - off - 0.0101); hi = np.searchsorted(mb, ma - off + 0.0101, "right")
        for i in np.flatnonzero(hi > lo):
            for j in range(lo[i], hi[i]): cand.append((ia_[i] * ib[j], i, j))
    cand.sort(reverse=True); ui, uj, s, n = set(), set(), 0.0, 0
    for v, i, j in cand:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s += v; n += 1
    return s, n


# queries: np-examples spectra
q = ds.to_table(columns=["inchikey14", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"],
                filter=pc.field("ingest_lib") == "enveda-np-examples").to_pandas()
SPQ = build_sp(q, "inchikey14")
rng = random.Random(11)
queries = [r for r in npst.itertuples(index=False) if r.inchikey14 in SPQ][:MAXQ]
tasks = []  # (query key, [(cand key, smiles, is_truth)])
for r in queries:
    pool = [x for x in byf.get(r.molecular_formula, []) if x[0] != r.inchikey14]
    rng.shuffle(pool)
    cands = [(r.inchikey14, r.normalized_smiles, True)] + [(k, s, False) for k, s in pool[:24]]
    tasks.append((r.inchikey14, cands))
print(f"queries {len(tasks)}, mean candidates {np.mean([len(c) for _, c in tasks]):.1f}, {time.time() - t0:.0f}s", flush=True)
# analogs per candidate (the truth structure is excluded from every candidate's analog set of its own query)
NB = {}
for qk, cands in tasks:
    for k, s, _ in cands:
        if (qk, k) in NB: continue
        m = Chem.MolFromSmiles(s)
        if m is None: NB[(qk, k)] = []; continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), fps))
        o = np.argsort(-sims)[:40]
        NB[(qk, k)] = [(keys[i], float(sims[i])) for i in o if 0.40 <= sims[i] < 0.85 and keys[i] != qk][:8]
need = sorted({a for v in NB.values() for a, _ in v})
print(f"analog structures needed {len(need)}, {time.time() - t0:.0f}s", flush=True)
rows = []
flt = pc.field("inchikey14").isin(pa.array(need)) & pc.field("adduct").isin(pa.array(list(ADD))) & (pc.field("ingest_lib") != "enveda-np-examples")
for b in ds.scanner(columns=["inchikey14", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"], filter=flt,
                    batch_size=8192).to_batches():
    if b.num_rows: rows.append(b.to_pandas().groupby(["inchikey14", "adduct"]).head(6))
SPT = build_sp(pd.concat(rows).groupby(["inchikey14", "adduct"]).head(6), "inchikey14") if rows else {}
print(f"analog spectra {len(SPT)}, {time.time() - t0:.0f}s", flush=True)
res = []
for qk, cands in tasks:
    qq = SPQ[qk]; ev = []
    for k, s, tru in cands:
        e = 0.0
        for a, sim in NB.get((qk, k), []):
            tt = SPT.get(a)
            if tt is None: continue
            for ad in set(qq) & set(tt):
                sc, n = mcos(qq[ad], tt[ad])
                if n >= 4: e += sim * sc
        ev.append(e)
    ev = np.array(ev); n = len(ev)
    # rank of truth (index 0) with ties broken pessimistically / averaged
    better = int((ev > ev[0]).sum()); ties = int((ev == ev[0]).sum()) - 1
    rank_avg = better + 1 + ties / 2
    res.append(dict(q=qk, n=n, rank_avg=rank_avg, top1=int(better == 0 and ties == 0), ev_truth=ev[0], zero_truth=ev[0] == 0,
                    all_zero=bool((ev == 0).all())))
R = pd.DataFrame(res)
R["rr"] = 1.0 / R.rank_avg
R["rr_chance"] = R.n.map(lambda n: np.mean(1.0 / np.arange(1, n + 1)))
print(f"\nqueries {len(R)}; candidates median {R.n.median():.0f}")
print(f"TA evidence alone: MRR {R.rr.mean():.3f} (chance {R.rr_chance.mean():.3f}), top-1 {R.top1.mean():.3f} "
      f"(chance {np.mean(1.0 / R.n):.3f})")
print(f"truth has zero evidence {R.zero_truth.mean():.2f}; all candidates zero {R.all_zero.mean():.2f}")
for lo, hi in ((2, 5), (5, 15), (15, 26)):
    s = R[(R.n >= lo) & (R.n < hi)]
    if len(s):
        print(f"  groups of {lo}-{hi - 1}: n={len(s)}, MRR {s.rr.mean():.3f} vs chance {s.rr_chance.mean():.3f}, top-1 {s.top1.mean():.3f}")
R.to_csv("np_ta_power.csv", index=False)
print(f"done {time.time() - t0:.0f}s")
