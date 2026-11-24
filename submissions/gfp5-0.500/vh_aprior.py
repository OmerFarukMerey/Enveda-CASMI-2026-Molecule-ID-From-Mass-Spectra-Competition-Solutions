"""APRIOR offline study (private Kaggle CPU kernel casmi-vh-aprior, never submitted): test-file thinning of adducts, weight and training-size grid.

Held-out enveda-180 molecules ranked against random same-formula decoys; results quoted in README.txt. Random decoys are easy,
so gains are ~10x the public board's and the best weight came out too high (0.75 scored 0.499 vs 0.5 at 0.500).
"""
# casmi-vh-aprior: offline study of goodpjw2008's APRIOR adduct-pattern prior. PRIVATE harness, never submitted.
# 1) visible test vs its train copies: which recorded adducts survive into the test file (observation process q_a)
# 2) APRIOR variants on held-out enveda-180 molecules (test-like thinned patterns) and on other timsTOF libraries,
#    ranked against same-formula decoys from train: model A (60k, current) vs B (all), plain vs observation-aware
#    log-likelihood, lambda grid, gamma 1.5, rank 1 kept.
import glob, json, math, os, subprocess, sys, time
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


try:
    import rdkit  # noqa
except ImportError:
    tag = "cp%d%d" % sys.version_info[:2]
    whl = [w for w in glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True) if tag in w]
    log("rdkit wheel", whl)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", whl[0]], check=True)
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

TRAIN = sorted(Path("/kaggle/input").rglob("train.parquet"), key=lambda p: ("mass-spectra" not in str(p).lower(), len(str(p))))[0]
TEST = TRAIN.with_name("test.parquet")
OUT = {}
_AP_TA = ("[M+H]+", "[M-H]-", "[M+CH2O2-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+Cl]-")
_AP_OBS = ("[M+H]+", "[M-H]-", "[M+CH2O2-H]-", "[M+Na]+")
_AP_SMARTS = {
    "cooh": "[CX3](=O)[OX2H1]", "phenol": "c[OX2H1]", "sulfo": "S(=O)(=O)[OH]", "phos": "P(=O)[OH]",
    "amine_basic": "[NX3;H2,H1,H0;!$(NC=O);!$(N-a);!$(N-S(=O)=O);!$(N-C=N)]", "aniline": "[NX3;H2,H1;$(N-a)]",
    "pyridine": "[nX2]", "amide": "C(=O)[NX3]", "imidazole": "c1ncc[nH]1", "guan": "NC(=N)N",
    "ester": "[CX3](=O)[OX2][#6]", "lactone": "[C;R](=O)[O;R]", "glyco": "[OX2H][CX4][CX4][OX2H]", "enol": "[OX2H]C=C",
    "oh_any": "[OX2H]", "nh_any": "[#7;!H0]", "quat": "[N+;!$(N=O);!$(N-[O-])]", "nitro": "[N+](=O)[O-]", "thiol": "[SX2H]",
    "carbonyl": "[CX3]=O", "halogen": "[Cl,Br,I]",
    "acid_any": "[$([CX3](=O)[OX2H1]),$(S(=O)(=O)[OH]),$(P(=O)[OH]),$(c[OX2H1])]",
}
PATS = [Chem.MolFromSmarts(v) for v in _AP_SMARTS.values()]
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def feats_mol(m):  # identical to the APRIOR cell's featurizer
    fp = np.asarray(GEN.GetCountFingerprintAsNumPy(m), dtype=np.float32)
    sm = np.array([len(m.GetSubstructMatches(p)) for p in PATS], np.float32)
    d = np.array([Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m), rdMolDescriptors.CalcNumHBD(m),
                  rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumAromaticRings(m), m.GetNumHeavyAtoms(),
                  sum(a.GetSymbol() == "N" for a in m.GetAtoms()), sum(a.GetSymbol() == "O" for a in m.GetAtoms()),
                  sum(a.GetSymbol() == "S" for a in m.GetAtoms()), Chem.GetFormalCharge(m)], np.float32)
    return np.concatenate([fp, sm, d])


# ---------------------------------------------------------------- load
schema = pq.read_schema(TRAIN).names
log("train columns", schema)
want = [c for c in ("ingest_lib", "inchikey14", "normalized_smiles", "adduct", "precursor_mz", "base_peak_intensity",
                    "collision_energy_orig", "instrument_type") if c in schema]
tr = pq.read_table(TRAIN, columns=want).to_pandas()
log("train rows", len(tr))
libs = tr.groupby("ingest_lib").agg(rows=("inchikey14", "size"), mols=("inchikey14", "nunique"),
                                    inst=("instrument_type", lambda s: ",".join(sorted(set(map(str, s)))[:3])))
log("libraries:\n" + libs.sort_values("rows", ascending=False).to_string())
OUT["libraries"] = libs.reset_index().to_dict("records")

# ---------------------------------------------------------------- 1) visible test vs its train copies
te = pd.read_parquet(TEST)
q_est = {}
try:
    keys = ["precursor_mz", "base_peak_intensity", "adduct"]
    tk = tr[keys + ["inchikey14", "ingest_lib", "collision_energy_orig"]].copy()
    mt = te[["molecule_id"] + keys].merge(tk, on=keys, how="left")
    hit = mt.dropna(subset=["inchikey14"])
    log("visible test spectra %d, matched %d (%.1f%%)" % (len(te), hit.drop_duplicates(["molecule_id"] + keys).shape[0],
                                                          100.0 * hit.drop_duplicates(["molecule_id"] + keys).shape[0] / len(te)))
    m2k = hit.groupby("molecule_id")["inchikey14"].agg(lambda s: s.value_counts().index[0])
    log("test molecules mapped", len(m2k), "of", te["molecule_id"].nunique(), "| libs of matches",
        hit["ingest_lib"].value_counts().head(5).to_dict())
    rec = tr[tr["inchikey14"].isin(set(m2k.values))]
    rec_e = rec[rec["ingest_lib"] == "enveda-180"]
    tset = te.groupby("molecule_id")["adduct"].agg(lambda s: set(s))
    rows = []
    for mid, k in m2k.items():
        r = rec_e[rec_e["inchikey14"] == k]
        if r.empty:
            continue
        rset = set(r["adduct"])
        rows.append(dict(mid=mid, k=k, rec=rset, obs=tset[mid], n_rec=len(r), n_obs=int((te["molecule_id"] == mid).sum()),
                         rec_by=r.groupby("adduct").size().to_dict()))
    log("mapped molecules with enveda-180 records", len(rows))
    for a in _AP_TA:
        den = sum(a in x["rec"] for x in rows)
        num = sum(a in x["rec"] and a in x["obs"] for x in rows)
        extra = sum(a not in x["rec"] and a in x["obs"] for x in rows)
        q_est[a] = (num / den) if den else None
        log("adduct %-14s recorded %4d kept %4d q=%s | observed-but-not-recorded %d" % (a, den, num, None if not den else round(num / den, 3), extra))
    # does keeping depend on how many spectra the adduct had, or on the molecule's number of recorded adducts?
    for nad in (1, 2, 3):
        sub = [x for x in rows if len(set(x["rec"]) & set(_AP_OBS)) == nad]
        if sub:
            kept = np.mean([len(x["obs"] & x["rec"]) / len(x["rec"]) for x in sub])
            log("molecules with %d recorded adducts: %d, mean fraction of adducts kept %.3f" % (nad, len(sub), kept))
    for a in ("[M+H]+", "[M-H]-"):
        for n in range(1, 9):
            sub = [x for x in rows if x["rec_by"].get(a, 0) == n]
            if len(sub) >= 5:
                log("  %s with %d recorded spectra: %d molecules, kept %.3f" % (a, n, len(sub), np.mean([a in x["obs"] for x in sub])))
    OUT["q_est"] = q_est
    OUT["n_mapped"] = len(rows)
    OUT["spectra_ratio"] = float(np.mean([x["n_obs"] / x["n_rec"] for x in rows])) if rows else None
    log("mean kept spectra fraction", OUT["spectra_ratio"])
except Exception as e:
    log("visible test analysis failed", repr(e))
Q = {a: (q_est.get(a) if q_est.get(a) else d) for a, d in zip(_AP_OBS, (0.98, 0.72, 0.7, 0.7))}
Q = {a: min(0.999, max(0.3, v)) for a, v in Q.items()}
log("q used", Q)

# ---------------------------------------------------------------- 2) data
e = tr[(tr["ingest_lib"] == "enveda-180") & tr["adduct"].isin(_AP_TA)]
pat = e.groupby("inchikey14")["adduct"].agg(lambda s: frozenset(s))
smi_all = tr.drop_duplicates("inchikey14").set_index("inchikey14")["normalized_smiles"].to_dict()
keys = np.array(sorted(pat.index))
perm = np.random.default_rng(0).permutation(len(keys))
keys = keys[perm]
HOLD = 6000
eval_keys, pool = keys[-HOLD:], keys[:-HOLD]
log("enveda-180 molecules", len(keys), "pool", len(pool), "held out", len(eval_keys))

# formulas of every train structure (decoy universe)
t = time.time()
formula, mols = {}, {}
for k, s in smi_all.items():
    try:
        m = Chem.MolFromSmiles(s)
    except Exception:
        m = None
    if m is not None:
        formula[k] = rdMolDescriptors.CalcMolFormula(m)
by_f = {}
for k, f in formula.items():
    by_f.setdefault(f, []).append(k)
log("formulas", len(formula), "distinct", len(by_f), "%.0fs" % (time.time() - t))

FEAT = {}


def feat(k):
    if k not in FEAT:
        try:
            m = Chem.MolFromSmiles(smi_all[k])
            FEAT[k] = feats_mol(m) if m is not None else None
        except Exception:
            FEAT[k] = None
    return FEAT[k]


def build_X(klist):
    X, ok = [], []
    for k in klist:
        f = feat(k)
        if f is not None:
            X.append(f); ok.append(k)
    return np.stack(X), ok


import lightgbm as lgb


def fit(train_keys, trees=300, seed=0):
    t = time.time()
    X, ok = build_X(train_keys)
    Y = np.array([[a in pat[k] for a in _AP_OBS] for k in ok], np.float32)
    ms = [lgb.LGBMClassifier(n_estimators=trees, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.5, random_state=seed, n_jobs=4, verbose=-1).fit(X, Y[:, j])
          for j in range(len(_AP_OBS))]
    log("fit %d molecules, %d trees: %.0fs, label rates %s" % (len(ok), trees, time.time() - t, Y.mean(0).round(3)))
    return ms


# eval sets ------------------------------------------------------------
rng = np.random.default_rng(1)
pool_set = set(pool)


def thin(rset):
    r = [a for a in rset if a in _AP_OBS]
    for _ in range(50):
        kept = {a for a in r if rng.random() < Q[a]}
        if kept:
            return kept
    return set(r)


def make_lists(eval_items):
    """eval_items: list of (key, observed set). Returns lists of 25 candidate keys (true included)."""
    out = []
    for k, obs in eval_items:
        f = formula.get(k)
        if f is None or not (set(obs) & set(_AP_OBS)):
            continue
        dec = [d for d in by_f.get(f, []) if d != k]
        if len(dec) < 5:
            continue
        dec = list(rng.choice(dec, size=min(24, len(dec)), replace=False))
        out.append((k, obs, dec))
    return out


E = {}
E["e180_thin"] = make_lists([(k, thin(pat[k])) for k in eval_keys])
E["e180_full"] = make_lists([(k, set(pat[k])) for k in eval_keys])
other = tr[(tr["ingest_lib"] != "enveda-180") & tr["instrument_type"].astype(str).str.contains("tims", case=False)
           & tr["adduct"].isin(_AP_TA)]
for lib, g in other.groupby("ingest_lib"):
    p2 = g.groupby("inchikey14")["adduct"].agg(lambda s: set(s))
    p2 = p2[[k not in pool_set for k in p2.index]]
    if len(p2) >= 50:
        E["lib_" + str(lib)] = make_lists(list(p2.items()))
for n, L in E.items():
    log("eval set %-30s %d molecules" % (n, len(L)))
OUT["eval_sizes"] = {n: len(L) for n, L in E.items()}

# position prior of the true candidate in ranks 2-25 (rank 1 is a decoy and is kept): P(r) ~ 1/r
R = np.arange(2, 26)
PR = (1.0 / R) / (1.0 / R).sum()


def evaluate(models, tagm):
    res = {}
    for n, L in E.items():
        if not L:
            continue
        allk = sorted({k for k, _, d in L for k in [k] + d})
        X, ok = build_X(allk)
        P = np.clip(np.stack([m.predict_proba(X)[:, 1] for m in models], 1), 1e-3, 1 - 1e-3)
        idx = {k: i for i, k in enumerate(ok)}
        rr = np.random.default_rng(7)
        # build lists once: true at simulated rank r, decoys elsewhere
        lists, ys, base_rr = [], [], []
        for k, obs, dec in L:
            r = int(rr.choice(R, p=PR))
            r = min(r, len(dec) + 1)
            cand = list(dec[:r - 1]) + [k] + list(dec[r - 1:])
            cand = [c for c in cand if c in idx]
            if k not in cand or cand.index(k) == 0 or len(cand) < 3:
                continue
            lists.append((cand, cand.index(k)))
            ys.append(np.array([float(a in obs) for a in _AP_OBS]))
        base = np.mean([1.0 / (p + 1) for _, p in lists])
        # observation log-loss of the true structure's pattern
        for mode in ("plain", "oa"):
            qv = np.array([Q[a] for a in _AP_OBS]) if mode == "oa" else np.ones(4)
            lls = []
            for (cand, p), y in zip(lists, ys):
                pp = P[[idx[c] for c in cand]] * qv
                lls.append((y * np.log(pp) + (1 - y) * np.log(1 - pp)).sum(1))
            true_ll = np.mean([ll[p] for ll, (c, p) in zip(lls, lists)])
            auc = np.mean([np.mean(ll[p] > np.delete(ll, p)) + 0.5 * np.mean(ll[p] == np.delete(ll, p)) for ll, (c, p) in zip(lls, lists)])
            for lam in (0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
                mrr = []
                for ll, (cand, p) in zip(lls, lists):
                    score = -1.5 * np.log(np.arange(1, len(cand) + 1)) + lam * ll
                    order = sorted(range(len(cand)), key=lambda i: (-score[i], i))
                    order = [0] + [i for i in order if i != 0]
                    mrr.append(1.0 / (order.index(p) + 1))
                res["%s|%s|%s|%.2f" % (n, tagm, mode, lam)] = round(float(np.mean(mrr)) - base, 5)
            res["%s|%s|%s|true_ll" % (n, tagm, mode)] = round(float(true_ll), 4)
            res["%s|%s|%s|auc" % (n, tagm, mode)] = round(float(auc), 4)
        res["%s|base_mrr" % n] = round(float(base), 5)
        res["%s|n" % n] = len(lists)
    return res


RES = {}
mA = fit(pool[:60000], 300)
RES.update(evaluate(mA, "A60k"))
log("A done", json.dumps({k: v for k, v in RES.items() if "|0.50" in k or "auc" in k}))
mB = fit(pool, 300)
RES.update(evaluate(mB, "Ball"))
log("B done")
mC = fit(pool, 600)
RES.update(evaluate(mC, "Ball600"))
log("C done")
OUT["results"] = RES
Path("/kaggle/working/vh_aprior.json").write_text(json.dumps(OUT, indent=1, default=str))
# compact table
for n in E:
    log("==== %s (n=%s, base MRR %s)" % (n, RES.get(n + "|n"), RES.get(n + "|base_mrr")))
    for tagm in ("A60k", "Ball", "Ball600"):
        for mode in ("plain", "oa"):
            row = [RES.get("%s|%s|%s|%.2f" % (n, tagm, mode, lam)) for lam in (0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)]
            log("  %-8s %-5s auc %s  dMRR@lam(.25 .35 .5 .75 1 1.5 2) %s" % (tagm, mode, RES.get("%s|%s|%s|auc" % (n, tagm, mode)), row))
log("done")
