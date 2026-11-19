"""Build a private Kaggle kernel folder = the public 0.492 notebook (mathleteutd, Apache-2.0) + our patches.

usage: python -I solutions/self-ref-veto/tools/build.py <variant> [...]   (variants: VARIANTS below)
Every patch is an exact string replacement that must match once, or a new cell inserted after a named anchor cell.
Output: solutions/self-ref-veto/kaggle/<slug>/<slug>.ipynb + kernel-metadata.json (private, T4, internet off, same inputs as the base).
"""
import copy, hashlib, json, os, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # solutions/self-ref-veto
BASE_DIR = os.path.join(HERE, "reference")
BASE_NB = os.path.join(BASE_DIR, "casmi26-0-492-test-set-network-rerank.ipynb")
BASE_META = os.path.join(BASE_DIR, "kernel-metadata.json")
CELLS = os.path.join(HERE, "cells")
OWNER = "omerfarukmerey"

src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def code_cell(text):
    return dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=text.splitlines(keepends=True))


def md_cell(text):
    return dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True))


def find(nb, needle):
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and needle in src(c)]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def replace(nb, needle_cell, old, new):
    i = find(nb, needle_cell)
    s = src(nb["cells"][i])
    assert s.count(old) == 1, (needle_cell, old, s.count(old))
    nb["cells"][i]["source"] = s.replace(old, new).splitlines(keepends=True)


def insert_after(nb, needle_cell, text):
    i = find(nb, needle_cell)
    nb["cells"].insert(i + 1, code_cell(text))


def insert_before(nb, needle_cell, text):
    i = find(nb, needle_cell)
    nb["cells"].insert(i, code_cell(text))


def cell_file(name):
    return open(os.path.join(CELLS, name), encoding="utf-8").read()


N1C = "def _n1c("
OVL = "def _apex_extlib_overlay("
CFG = "COMMIT_MODE = 'fast'"


def smoke(nb):
    # commit run executes the full pipeline on 12 visible molecules, so every new cell runs on real lists before we submit
    replace(nb, CFG, "COMMIT_MODE = 'fast'", "COMMIT_MODE = 'smoke'")


VARIANTS = {}


def variant(fn):
    VARIANTS[fn.__name__] = fn
    return fn


NOTICE = """# CASMI26 {slug}

**Notice.** Built on the public notebook [CASMI26 0.492 | Test-Set Network Rerank](https://www.kaggle.com/code/mathleteutd/casmi26-0-492-test-set-network-rerank)
(Apache-2.0), which is [rogersjohnson/casmi26-l4-lam035](https://www.kaggle.com/code/rogersjohnson/casmi26-l4-lam035) =
[analyticaobscura/casmi-2026](https://www.kaggle.com/code/analyticaobscura/casmi-2026) with the network weight 0.35. The
credits of every upstream author stay in the cells below. **Our changes:** {changes}
"""


def build(name):
    nb = json.load(open(BASE_NB, encoding="utf-8"))
    nb0 = copy.deepcopy(nb)
    slug, changes = VARIANTS[name](nb)
    nb["cells"].insert(0, md_cell(NOTICE.format(slug=slug, changes=changes)))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile(src(c), slug, "exec")  # every code cell must at least parse
    n_new = sum(1 for c in nb["cells"] if c["cell_type"] == "code") - sum(1 for c in nb0["cells"] if c["cell_type"] == "code")
    meta0 = json.load(open(BASE_META, encoding="utf-8"))
    meta = {k: v for k, v in meta0.items() if k not in ("id_no",)}
    meta.update(id=f"{OWNER}/{slug}", title=slug, code_file=f"{slug}.ipynb", is_private=True, keywords=[])
    meta["machine_shape"] = "NvidiaTeslaT4"
    assert meta["enable_internet"] is False and meta["enable_gpu"] is True
    out = os.path.join(HERE, "kaggle", slug)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{slug}.ipynb")
    json.dump(nb, open(path, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8"), indent=1)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    print(f"built {slug}: {len(nb['cells'])} cells ({n_new:+d} code cells vs base), sha {sha}, inputs "
          f"{len(meta['dataset_sources'])} datasets + {meta['kernel_sources']}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# variants (each returns (slug, one-paragraph description of the change))
# ---------------------------------------------------------------------------------------------------------------------
def _ta(nb, lam, freeze, margin):
    head = f"TA_LAM, TA_FREEZE_TOP1, TA_TOP1_MARGIN = {lam}, {freeze}, {margin}\n"
    insert_after(nb, N1C, head + cell_file("ta_param.py"))

def _d1(nb):
    insert_after(nb, OVL, cell_file("dedup_after_overlay.py"))

@variant
def tas(nb):  # leader-safe TA-1: rank 1 never moves, TA evidence re-orders ranks 2-25
    smoke(nb); _ta(nb, 0.75, True, 0.0); _d1(nb)
    return "casmi26-om-tas", ("train-analog evidence (the team's TA-1 stage: candidates whose closest training analogs "
                              "fragment like the query move up) applied to ranks 2-25 only, rank 1 never moves; plus a "
                              "metric-key deduplication after the external-library overlay.")

@variant
def vpd(nb):  # best-spectrum-pair veto (tau 0.30) + post-overlay dedup
    smoke(nb); insert_after(nb, N1C, cell_file("v1_veto_pm.py")); _d1(nb)
    return "casmi26-om-vpd", ("a self-reference contradiction veto after the network rerank: a candidate that is a "
                              "training structure has its own measured spectra in train.parquet; when none of them "
                              "(same adduct, any energy or library) matches any query spectrum (best single-pair sqrt "
                              "cosine < 0.30), it moves below every candidate that is not contradicted. Lists with "
                              "library_max >= 0.9 are untouched and the external-library overlay keeps the last word. "
                              "Measured on the 250 timsTOF natural-product standards against their references from "
                              "other libraries: a true structure is vetoed 2% of the time, a wrong same-formula isomer "
                              "22-66% of the time depending on how similar it is. Plus a metric-key deduplication "
                              "after the overlay.")

def _veto(nb, tau, cap, conf):
    head = f"V1_TAU, V1_CAP, V2_ON = {tau}, {cap}, {conf}\n"
    insert_after(nb, N1C, head + cell_file("veto.py")); _d1(nb)

VETO_TXT = ("a self-reference contradiction veto after the network rerank: a candidate that is a training structure has "
            "its own measured spectra in train.parquet (up to {cap} per adduct); when none of them matches any query "
            "spectrum (best single-pair sqrt cosine < {tau}) it moves below every candidate that is not contradicted.{c} "
            "Lists with library_max >= 0.9 are untouched, the external-library overlay keeps the last word, and a "
            "metric-key deduplication follows it.")

CONF_TXT = (" A known leader that matches its own spectra only weakly also gives way to a known candidate in ranks 2-10 "
            "that matches its own spectra strongly (>= 0.85 and 0.20 better).")

@variant
def w30(nb):  # veto, 30 refs per adduct, tau 0.40 (2.2% truth error on the NP panel), + dedup
    smoke(nb); _veto(nb, 0.40, 30, False)
    return "casmi26-om-w30", VETO_TXT.format(cap=30, tau=0.40, c="")

@variant
def w30c(nb):  # w30 + confirmation
    smoke(nb); _veto(nb, 0.40, 30, True)
    return "casmi26-om-w30c", VETO_TXT.format(cap=30, tau=0.40, c=CONF_TXT)

def _n1c_leader_safe(nb, lam_mid):
    # rank 1 exactly as the 0.492 notebook (LAM 0.35); ranks 2-25 ordered with the stronger network weight LAM_MID
    replace(nb, N1C, "def _n1c(work='/kaggle/working', inp='/kaggle/input', LAM=0.35, TAU=0.6):",
            f"def _n1c(work='/kaggle/working', inp='/kaggle/input', LAM=0.35, TAU=0.6, LAM_MID={lam_mid}):")
    replace(nb, N1C, "o = np.argsort(-(z(base) + LAM * z(N)), kind='stable'); nl = [lst[i] for i in o]",
            "o = np.argsort(-(z(base) + LAM * z(N)), kind='stable'); o2 = np.argsort(-(z(base) + LAM_MID * z(N)), kind='stable')\n"
            "            nl = [lst[o[0]]] + [lst[i] for i in o2 if i != o[0]]   # ours: leader-safe network rerank")

@variant
def nls(nb):  # N-1c leader-safe: rank 1 at LAM 0.35 (unchanged), ranks 2-25 at LAM_MID 0.70
    smoke(nb); _n1c_leader_safe(nb, 0.70)
    return "casmi26-om-nls", ("a leader-safe in-test network rerank: rank 1 is chosen exactly as in the 0.492 notebook "
                              "(network weight 0.35), ranks 2-25 are ordered with a stronger network weight (0.70) - the "
                              "same split that lifted the neural scorer from 0.468 to 0.472 (weight 0.25 for rank 1, 0.75 "
                              "for ranks 2-25).")

@variant
def nlt(nb):  # leader-safe network (0.35 / 0.70) + leader-safe TA-1 (ranks 2-25 only) + dedup
    smoke(nb); _n1c_leader_safe(nb, 0.70); _ta(nb, 0.75, True, 0.0); _d1(nb)
    return "casmi26-om-nlt", ("two leader-safe re-orderings of ranks 2-25 (rank 1 stays exactly as in the 0.492 "
                              "notebook): the in-test network rerank at weight 0.70 for ranks 2-25 (0.35 for rank 1, as "
                              "before), then train-analog evidence (the team's TA-1: candidates whose closest training "
                              "analogs fragment like the query move up) on ranks 2-25 only; plus a metric-key "
                              "deduplication after the external-library overlay.")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        build(name)
