"""Veto safety / power on NP timsTOF queries, with peak-count guards and isomer power stratified by Tanimoto to the truth.
Same data and spectrum preparation as np_veto_power.py (see its docstring).
usage: python -I np_veto_power2.py <competition_dir>
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
D = sys.argv[1]
ADD = ("[M+H]+", "[M-H]-")
t0 = time.time()
ds = pads.dataset(f"{D}/train.parquet")
meta = ds.to_table(columns=["inchikey14", "normalized_smiles", "molecular_formula", "ingest_lib", "adduct"]).to_pandas()
npq = meta[meta.ingest_lib == "enveda-np-examples"]
np_keys = sorted(set(npq.inchikey14))
others = meta[(meta.ingest_lib != "enveda-np-examples") & meta.adduct.isin(ADD)]
form = meta.drop_duplicates("inchikey14").set_index("inchikey14").molecular_formula.to_dict()
smi = meta.drop_duplicates("inchikey14").set_index("inchikey14").normalized_smiles.to_dict()
byf = others.drop_duplicates("inchikey14").groupby("molecular_formula").inchikey14.apply(list).to_dict()
rng = random.Random(5)
iso = {}
for k in np_keys:
    pool = [x for x in byf.get(form[k], []) if x != k]; rng.shuffle(pool); iso[k] = pool[:40]
need = sorted(set(np_keys) | {x for v in iso.values() for x in v})


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


def cos(a, b):
    (ma, ia_, _), (mb, ib, _) = a, b; cand = []
    lo = np.searchsorted(mb, ma - 0.0101); hi = np.searchsorted(mb, ma + 0.0101, "right")
    for i in np.flatnonzero(hi > lo):
        for j in range(lo[i], hi[i]): cand.append((ia_[i] * ib[j], i, j))
    cand.sort(reverse=True); ui, uj, s = set(), set(), 0.0
    for v, i, j in cand:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s += v
    return s


cols = ["inchikey14", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
SPQ = build_sp(ds.to_table(columns=cols, filter=pc.field("ingest_lib") == "enveda-np-examples").to_pandas(), "inchikey14")
flt = pc.field("inchikey14").isin(pa.array(need)) & pc.field("adduct").isin(pa.array(list(ADD))) & (pc.field("ingest_lib") != "enveda-np-examples")
rows = [b.to_pandas() for b in ds.scanner(columns=cols, filter=flt, batch_size=8192).to_batches() if b.num_rows]
SPR = build_sp(pd.concat(rows).groupby(["inchikey14", "adduct"]).head(6), "inchikey14")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048); fpc = {}
def fp(k):
    if k not in fpc:
        m = Chem.MolFromSmiles(smi[k]) if isinstance(smi.get(k), str) else None
        fpc[k] = gen.GetFingerprint(m) if m is not None else None
    return fpc[k]
out = []
for k in np_keys:
    qq = SPQ.get(k)
    if not qq: continue
    for kind, c in [("truth", k)] + [("isomer", x) for x in iso[k]]:
        rr = SPR.get(c, {}); ads = set(qq) & set(rr)
        if not ads: continue
        best = max(((cos(qq[ad], rr[ad]), min(len(qq[ad][0]), len(rr[ad][0]))) for ad in ads))
        tan = DataStructs.TanimotoSimilarity(fp(k), fp(c)) if fp(k) is not None and fp(c) is not None else np.nan
        out.append((kind, k, c, best[0], best[1], tan))
R = pd.DataFrame(out, columns=["kind", "query", "cand", "cos", "minpk", "tan"])
R.to_csv("np_veto_power2.csv", index=False)
print(f"pairs: truth {int((R.kind == 'truth').sum())}, isomer {int((R.kind == 'isomer').sum())}, {time.time() - t0:.0f}s")
for guard in (0, 5, 8):
    G = R[R.minpk >= guard]
    print(f"\n== both merged spectra >= {guard} peaks: truths {int((G.kind == 'truth').sum())}, isomers {int((G.kind == 'isomer').sum())}")
    for tau in (0.15, 0.2, 0.25, 0.3):
        pt = (G[G.kind == "truth"].cos < tau).mean()
        parts = []
        for lo, hi in ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
            I = G[(G.kind == "isomer") & (G.tan >= lo) & (G.tan < hi)]
            parts.append(f"tan[{lo:.1f},{hi:.1f}) {(I.cos < tau).mean():.2f} n={len(I)}")
        print(f"tau {tau:.2f}: truth vetoed {pt:.3f} | isomer vetoed: " + " | ".join(parts))
low = R[(R.kind == "truth") & (R.cos < 0.25)]
print("\nlow-cosine truths (cos < 0.25):", len(low), "| their min peak counts:", sorted(low.minpk.tolist()))
