"""Is 'one 2D answer per test molecule' a safe assumption, and how isomer-rich is Enveda's NP data?
1) enveda-np-examples: structures per IK14 vs full InChIKey (are stereoisomers kept as separate entries?),
   formulas shared by >= 2 structures (isomer density).
2) visible test placeholder: recover each molecule's structure by exact spectrum match to train, then check whether two
   molecule_ids ever share an IK14 (how the host groups spectra into molecules).
usage: python -I uniq_check.py <competition_dir>
"""
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

sys.stdout.reconfigure(encoding="utf-8")
D = sys.argv[1]
train = ds.dataset(f"{D}/train.parquet")
cols = ["inchikey", "inchikey14", "molecular_formula", "ingest_lib", "adduct", "precursor_mz", "ms2_mzs"]
np_ex = train.to_table(columns=cols, filter=pc.field("ingest_lib") == "enveda-np-examples").to_pandas()
print("enveda-np-examples rows", len(np_ex), "IK14", np_ex.inchikey14.nunique(), "full IK", np_ex.inchikey.nunique())
per14 = np_ex.groupby("inchikey14").inchikey.nunique()
print("  IK14 with >1 full InChIKey (stereo variants as separate entries):", int((per14 > 1).sum()))
f = np_ex.drop_duplicates("inchikey14").molecular_formula.value_counts()
print("  formulas:", len(f), "| formulas with >=2 structures:", int((f >= 2).sum()),
      "| structures in such formulas:", int(f[f >= 2].sum()), "of", int(f.sum()))
print("  largest isomer groups:", f.head(8).to_dict())

# 2) visible test: map each spectrum to a train row by (precursor_mz, first 5 peaks) signature
te = pd.read_parquet(f"{D}/test.parquet")
print("\nvisible test rows", len(te), "molecules", te.molecule_id.nunique())
def sig(prec, mzs):
    a = np.asarray(mzs, float)
    return (round(float(prec), 4), len(a), tuple(np.round(a[:6], 4)))
want = {sig(p, m) for p, m in zip(te.precursor_mz, te.ms2_mzs)}
precs = sorted({k[0] for k in want})
lo, hi = min(precs) - 0.001, max(precs) + 0.001
hits = {}
scanner = train.scanner(columns=["inchikey", "inchikey14", "precursor_mz", "ms2_mzs", "ingest_lib"],
                        filter=(pc.field("precursor_mz") >= lo) & (pc.field("precursor_mz") <= hi), batch_size=65536)
pset = set(precs)
for b in scanner.to_batches():
    t = b.to_pandas()
    t = t[t.precursor_mz.round(4).isin(pset)]
    for r in t.itertuples(index=False):
        k = sig(r.precursor_mz, r.ms2_mzs)
        if k in want:
            hits.setdefault(k, set()).add((r.inchikey14, r.inchikey, r.ingest_lib))
te["sig"] = [sig(p, m) for p, m in zip(te.precursor_mz, te.ms2_mzs)]
te["ik14"] = [";".join(sorted({h[0] for h in hits.get(s, set())})) for s in te.sig]
te["ik"] = [";".join(sorted({h[1] for h in hits.get(s, set())})) for s in te.sig]
te["lib"] = [";".join(sorted({h[2] for h in hits.get(s, set())})) for s in te.sig]
print("spectra matched to train:", int((te.ik14 != "").sum()), "of", len(te))
mol = te[te.ik14 != ""].groupby("molecule_id").agg(ik14=("ik14", lambda s: set(";".join(s).split(";"))),
                                                    ik=("ik", lambda s: set(";".join(s).split(";"))),
                                                    lib=("lib", lambda s: set(";".join(s).split(";"))))
print("molecules with recovered structure:", len(mol))
print("  molecules whose spectra map to >1 IK14:", int((mol.ik14.map(len) > 1).sum()))
print("  molecules whose spectra map to >1 full InChIKey:", int((mol.ik.map(len) > 1).sum()))
owner = defaultdict(set)
for m, s in mol.ik14.items():
    for k in s:
        owner[k].add(m)
dup = {k: v for k, v in owner.items() if len(v) > 1}
print("  IK14 shared by >1 molecule_id:", len(dup))
print("  source libraries of visible test:", Counter(l for s in mol.lib for l in s).most_common(6))

# 3) acquisition pattern of the visible test (adducts, spectra per molecule, coverage of the [M+H]+/[M-H]- channels)
print("\nadduct counts (spectra):", te.adduct.value_counts().to_dict())
per = te.groupby("molecule_id").agg(n=("spectrum_id", "size"), ads=("adduct", lambda s: set(s)))
print("spectra per molecule: median", per.n.median(), "mean", round(per.n.mean(), 2), "max", per.n.max())
has_main = per.ads.map(lambda a: bool(a & {"[M+H]+", "[M-H]-"}))
print("molecules without any [M+H]+ / [M-H]- spectrum:", int((~has_main).sum()), "of", len(per))
print("molecules with both polarities:", int(per.ads.map(lambda a: any(x.endswith("+") for x in a) and any(x.endswith("-") for x in a)).sum()))
print("instrument_type:", te.instrument_type.value_counts().to_dict())
print("collision_energy_ev examples:", te.collision_energy_ev.head(5).tolist())
