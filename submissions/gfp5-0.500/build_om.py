"""Builder of Ömer's private casmi26-om-* kernels: a public notebook (taeyangg4 0.495 or goodpjw2008 0.498) + exact-string patches.

Copy of the session builder (scratchpad work/b495.py); it reads the pulled public notebooks from the scratchpad
(pub/<owner>_<slug>/), which are not in git, and writes work/kernels/<slug>/. gfp5 = GVARIANTS['gfp5'] = p_fp5 on g498.
usage: python -I build_om.py <variant> [<variant> ...]      (variants in VARIANTS / GVARIANTS below)
Every patch is an exact string replacement that must match once, or a new cell inserted after a named anchor cell.
Cell 0 becomes a short notice '# CASMI26 <slug>: ...'; every other base cell is kept byte for byte.
"""
import copy, hashlib, json, os, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(HERE)
BASE_DIR = os.path.join(SCR, "pub", "taeyangg4__casmi26-lb-0-495-network-fr-p5-tail")
BASE_NB = os.path.join(BASE_DIR, "casmi26-lb-0-495-network-fr-p5-tail.ipynb")
BASE_META = os.path.join(BASE_DIR, "kernel-metadata.json")
REPO = r"C:\Users\omerm\Desktop\PersonalRepositories\Enveda-CASMI-2026-Molecule-ID-From-Mass-Spectra-Competition-Solutions"
VETO = os.path.join(REPO, "solutions", "self-ref-veto", "cells", "veto.py")
CELLS = os.path.join(HERE, "cells495")
OWNER = "omerfarukmerey"

src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def code_cell(text):
    return dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=text.splitlines(keepends=True))


def find(nb, needle):
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and needle in src(c)]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def replace(nb, needle_cell, old, new):
    i = find(nb, needle_cell)
    s = src(nb["cells"][i])
    assert s.count(old) == 1, (needle_cell, old[:80], s.count(old))
    nb["cells"][i]["source"] = s.replace(old, new).splitlines(keepends=True)


def insert_after(nb, needle_cell, text):
    i = find(nb, needle_cell)
    nb["cells"].insert(i + 1, code_cell(text))


NET = "_NNET_NOTEBOOK = True"          # D-L1 neighbour-network cell
OVL = "def _apex_extlib_overlay("      # external-library rank-1 overlay cell

# ---- patch: confirmation-only rank-1 rule (veto off) after the network, before the overlay ----
CF_HEAD = (
    "# ===== Arsokan cf5: own-spectrum CONFIRMATION only (the contradiction veto is off: V1_TAU < 0) =====\n"
    "# In lists whose library_max < 0.90, a known (train) leader whose own spectra match the query weakly gives way to a\n"
    "# known candidate from ranks 2-10 whose own spectra match strongly (best single pair >= 0.85 and >= 0.20 better).\n"
    "# Unknown leaders and candidates without own spectra never move. Any failure leaves submission.csv unchanged.\n"
    "V1_TAU, V1_CAP, V2_ON, V2_CONF, V2_MARGIN, V1_LIB_TAU = -1.0, 30, True, 0.85, 0.20, 0.90\n")


def p_cf5(nb):
    insert_after(nb, NET, CF_HEAD + open(VETO, encoding="utf-8").read())


# ---- patch: library gate on the network's rank-1 change ----
GATE_CONST_OLD = "_NNET_LOCK1 = True\n"
GATE_CONST_NEW = ("_NNET_LOCK1 = True\n"
                  "_NNET_LIB_GATE = 0.90   # Arsokan g90: lists with library_max >= this keep their pre-network leader at rank 1\n")
GATE_OLD = "            changed = sum(output[mid] != lists[mid] for mid in lists)\n"
GATE_NEW = (
    "            _g = globals().get(\"_NNET_LIB_GATE\", None)\n"
    "            if _g is not None:\n"
    "                # Arsokan g90: a strong spectral-library match outranks network evidence for rank 1. Lists with\n"
    "                # library_max >= _g keep the leader they had before this cell; ranks 2-25 keep the network order.\n"
    "                try:\n"
    "                    _ev = pd.read_parquet(Path(work) / \"frozen_candidate_evidence.parquet\", columns=[\"molecule_id\", \"library_max\"])\n"
    "                    _lm = _ev.assign(molecule_id=_ev[\"molecule_id\"].astype(str)).groupby(\"molecule_id\")[\"library_max\"].max().to_dict()\n"
    "                    _gated = [mid for mid in lists if lists[mid] and float(_lm.get(mid, 0.0)) >= float(_g)\n"
    "                              and output[mid][0] != lists[mid][0]]\n"
    "                    for mid in _gated:\n"
    "                        _lead = lists[mid][0]\n"
    "                        output[mid] = [_lead] + [c for c in output[mid] if c != _lead]\n"
    "                    audit.update(lib_gate=float(_g), lib_gate_molecules=len(_lm),\n"
    "                                 lib_gate_at_or_above=sum(float(_lm.get(mid, 0.0)) >= float(_g) for mid in lists),\n"
    "                                 lib_gate_restored=len(_gated))\n"
    "                except Exception as _gerr:\n"
    "                    audit.update(lib_gate_error=repr(_gerr)[:300])\n"
    + GATE_OLD)


def p_g90(nb):
    replace(nb, NET, GATE_CONST_OLD, GATE_CONST_NEW)
    replace(nb, NET, GATE_OLD, GATE_NEW)


FR = "_F3_VERSION"                     # FR 3-25 learned fragment re-ranker cell
P5 = "_P5_VERSION"                     # P5 tail cell
FP5_HEAD = (
    "# ===== Arsokan fp5: this FR 3-25 cell runs AFTER the P5 tail (moved, otherwise unchanged), so the learned fragment\n"
    "# re-ranker also orders the generated candidates P5 inserted into ranks 3-25. Ranks 1-2 stay locked. Session guard: past\n"
    "# 8.2 h after NB_T0 the re-rank is skipped (P5's output is kept as is).\n"
    "import time as _fp5_time\n"
    "_FP5_SKIP = isinstance(globals().get('NB_T0'), (int, float)) and (_fp5_time.time() - NB_T0) > 8.2 * 3600\n"
    "if _FP5_SKIP:\n"
    "    print('[fp5] session guard: FR 3-25 after P5 skipped', flush=True)\n")


def p_fp5(nb):
    i = find(nb, FR)
    cell = nb["cells"].pop(i)
    s = src(cell)
    old = "    if globals().get('PIPELINE_DONE'):\n        _f3_parent_bytes = _f3_sub.read_bytes()\n"
    assert s.count(old) == 1
    s = FP5_HEAD + s.replace(old, "    if globals().get('PIPELINE_DONE') and not _FP5_SKIP:\n        _f3_parent_bytes = _f3_sub.read_bytes()\n")
    j = find(nb, P5)
    assert j >= i - 1, (i, j)
    nb["cells"].insert(j + 1, code_cell(s))


P5_SHA_OLD = "39c0d4c0271b26c199f4831ec56921c9fbe3f609deade013d631d4ab4ed81df2"
P12_GEN_OLD = ("        new, p6, p5, n_tr = p5_order(analogs, ecfg.gen_min_sim, ecfg.gen_n_analog, transformable)\n"
               "        res = orig_generate(new, target, exclude_keys)\n")
P12_GEN_NEW = ("        new, p6, p5, n_tr = p5_order(analogs, ecfg.gen_min_sim, ecfg.gen_n_analog, transformable)\n"
               "        # Arsokan p12: up to P12_N transformable-first parents and P12_TOTAL products for this call only; the engine\n"
               "        # config is restored right after, so cfg0 and the P6 parent set (first 6 eligible analogs) are unchanged.\n"
               "        p5 = [int(a[0]) for a in new if a[1] >= ecfg.gen_min_sim][:P12_N]\n"
               "        _n0, _t0 = ecfg.gen_n_analog, ecfg.gen_max_total\n"
               "        ecfg.gen_n_analog, ecfg.gen_max_total = P12_N, P12_TOTAL\n"
               "        try:\n"
               "            res = orig_generate(new, target, exclude_keys)\n"
               "        finally:\n"
               "            ecfg.gen_n_analog, ecfg.gen_max_total = _n0, _t0\n")


def p_p12(nb, n_par=12, total=300):
    import ast
    i = find(nb, P5)
    s = src(nb["cells"][i])
    lines = [l for l in s.splitlines() if l.startswith("_P5_RUNNER = ")]
    assert len(lines) == 1
    runner = ast.literal_eval(lines[0][len("_P5_RUNNER = "):])
    assert hashlib.sha256(runner.encode("utf-8")).hexdigest() == P5_SHA_OLD
    assert runner.count(P12_GEN_OLD) == 1 and runner.count("VERSION = '1.0.0'\n") == 1
    runner = runner.replace(P12_GEN_OLD, P12_GEN_NEW).replace(
        "VERSION = '1.0.0'\n", f"VERSION = '1.0.0'\nP12_N, P12_TOTAL = {n_par}, {total}   # Arsokan p12\n")
    sha_new = hashlib.sha256(runner.encode("utf-8")).hexdigest()
    assert s.count(lines[0]) == 1 and s.count(f"_P5_RUNNER_SHA256 = '{P5_SHA_OLD}'") == 1
    s = s.replace(lines[0], "_P5_RUNNER = " + repr(runner)).replace(
        f"_P5_RUNNER_SHA256 = '{P5_SHA_OLD}'", f"_P5_RUNNER_SHA256 = '{sha_new}'")
    s = ("# ===== Arsokan p12: the P5 tail runner below uses up to 12 transformable-first analog parents and 300 products per\n"
         "# molecule (P5: 6 / 150); only candidates generated from parents outside the original first 6 eligible analogs are\n"
         "# inserted, into ranks 3-25 by Engine 1 position, exactly as in P5. Runner sha256 re-pinned.\n") + s
    nb["cells"][i]["source"] = s.splitlines(keepends=True)


# ---- patch: NetGen - network neighbours' top candidates as extra generation parents inside the P5 runner ----
N5_RUNNER_HEAD_OLD = "VERSION = '1.0.0'\n"
N5_RUNNER_HEAD_NEW = (
    "VERSION = '1.0.0'\n"
    "N5_PER_PARENT, N5_MAX = 10, 16   # Arsokan n5: products per network parent / per molecule\n"
    "N5_PAR = {}\n"
    "if os.environ.get('P5N_PARENTS') and os.path.isfile(os.environ['P5N_PARENTS']):\n"
    "    N5_PAR = json.load(open(os.environ['P5N_PARENTS'], encoding='utf-8'))\n")
N5_GEN_OLD = ("        res = orig_generate(new, target, exclude_keys)\n"
              "        G.update(called=True, p6=p6, p5=p5, n_tr=n_tr, gen=[[str(g['key']), str(g['smiles']), int(g['parent'])] for g in res])\n")
N5_GEN_NEW = (
    "        res = orig_generate(new, target, exclude_keys)\n"
    "        npar = N5_PAR.get(str(scope.get('mid')), []) if N5_PAR else []\n"
    "        if npar:\n"
    "            res = list(res) + n5_products(npar, target, set(exclude_keys) | {g['key'] for g in res}, tol)\n"
    "        G.update(called=True, p6=p6, p5=p5, n_tr=n_tr, gen=[[str(g['key']), str(g['smiles']), int(g['parent'])] for g in res])\n")
N5_FUNC_ANCHOR = "    E.generate = p5_generate\n"
N5_FUNC = (
    "    def n5_products(npar, target, seen, tol):\n"
    "        \"\"\"Arsokan n5: derivatives of network neighbours' top candidates whose mass gap to the query is a casmi.derive\n"
    "        transform combination (<= 2 steps), built exactly like Engine.generate's rows; parent id -2-j marks them new.\"\"\"\n"
    "        import numpy as _np\n"
    "        from casmi import chem as _c, frag as _fm\n"
    "        out = []\n"
    "        for j, (psmi, psim) in enumerate(npar):\n"
    "            try:\n"
    "                pp = _c.mol_props(psmi)\n"
    "                if pp is None:\n"
    "                    continue\n"
    "                delta = target - float(pp[1])\n"
    "                if abs(delta) < tol or not dmod.matching_transforms(delta, tol, 2):\n"
    "                    continue\n"
    "                pfp = _c.raw_fingerprint(psmi)\n"
    "                if pfp is None:\n"
    "                    continue\n"
    "                pfp = pfp[E.pool.bits].astype(_np.float32)\n"
    "                prods = dmod.derive(psmi, delta, tol=tol, max_steps=2, max_per_analog=N5_PER_PARENT)\n"
    "            except Exception:\n"
    "                continue\n"
    "            for s in prods:\n"
    "                try:\n"
    "                    std = _c.standardize_smiles(s)\n"
    "                    if std is None:\n"
    "                        continue\n"
    "                    k = _c.score_key(std)\n"
    "                    if k is None or k in seen:\n"
    "                        continue\n"
    "                    p = _c.mol_props(std)\n"
    "                    if p is None or abs(p[1] - target) > tol:\n"
    "                        continue\n"
    "                    fp = _c.raw_fingerprint(std)\n"
    "                    if fp is None:\n"
    "                        continue\n"
    "                    fp = fp[E.pool.bits].astype(_np.float32)\n"
    "                    inter = float((fp * pfp).sum()); tan = inter / (fp.sum() + pfp.sum() - inter + 1e-9)\n"
    "                    seen.add(k)\n"
    "                    out.append(dict(smiles=std, key=k, formula=p[0], mass=p[1], n_heavy=p[2], fp=fp,\n"
    "                                    frags=_fm.fragments_for_smiles(std), sim=float(psim), steps=1, tan=tan, parent=-2 - j))\n"
    "                except Exception:\n"
    "                    continue\n"
    "                if len(out) >= N5_MAX:\n"
    "                    return out\n"
    "        return out\n"
    + N5_FUNC_ANCHOR)
# P5 cell: write the network-parent file right before the Engine 1 rerun (scoring run and fast commit alike)
N5_CELL_OLD = "        _p5_e1 = _p5_rerun(_p5_deadline - _P5_MERGE_S, _p5_root)\n"
N5_CELL_NEW = ("        _n5_write_parents(_p5_raw, _p5_root)\n" + N5_CELL_OLD)
N5_CELL_FAST_OLD = "        _p5_e1 = _p5_rerun(_p5_deadline - 60, _p5_root)\n"
N5_CELL_FAST_NEW = ("        _n5_write_parents(_p5_raw, _p5_root)\n" + N5_CELL_FAST_OLD)
N5_CELL_DEFS = (
    "# ===== Arsokan n5 (NetGen): network neighbours' top candidates become extra generation parents in the P5 Engine 1\n"
    "# rerun. Neighbours = the D-L1 cell's N-1c graph (same adduct, modified cosine >= 0.6, >= 6 matched peaks); parents = each\n"
    "# neighbour's top 2 guesses, ordered by edge cosine; a parent is used only when the mass gap to the query is a casmi.derive\n"
    "# transform combination (<= 2 steps). Products enter Engine 1 as generated rows (parent id < 0 = 'new') and reach the list\n"
    "# only through P5's unchanged tail rule (ranks 3-25 by Engine 1 position). Any failure: no parent file, plain P5.\n"
    "_N5_TOP, _N5_MAXPAR = 2, 12\n"
    "def _n5_write_parents(raw, root):\n"
    "    import os as _o, json as _j\n"
    "    _o.environ.pop('P5N_PARENTS', None)\n"
    "    try:\n"
    "        import pandas as _pd\n"
    "        comp = globals().get('COMP')\n"
    "        te = _pd.read_parquet(_o.path.join(str(comp), 'test.parquet'))\n"
    "        spectra = build_spectra(te, set(raw))\n"
    "        graph = build_graph(spectra)\n"
    "        par = {}\n"
    "        for m, nbrs in graph.items():\n"
    "            cand = []\n"
    "            for n, cos in sorted(nbrs, key=lambda x: -x[1]):\n"
    "                toks = [t for t in str(raw.get(n, '')).split(';') if t][:_N5_TOP]\n"
    "                cand += [(t, float(cos)) for t in toks]\n"
    "            seen, keep = set(), []\n"
    "            for t, c in cand:\n"
    "                if t not in seen:\n"
    "                    seen.add(t); keep.append([t, c])\n"
    "            if keep:\n"
    "                par[str(m)] = keep[:_N5_MAXPAR]\n"
    "        path = _o.path.join(str(root), 'p5n_parents.json')\n"
    "        _j.dump(par, open(path, 'w', encoding='utf-8'))\n"
    "        _o.environ['P5N_PARENTS'] = path\n"
    "        print('[n5] network parents for', len(par), 'molecules, edges', sum(len(v) for v in graph.values()) // 2, flush=True)\n"
    "    except Exception as _e:\n"
    "        print('[n5] parent file skipped:', repr(_e)[:200], flush=True)\n\n")


def p_n5(nb):
    import ast
    i = find(nb, P5)
    s = src(nb["cells"][i])
    lines = [l for l in s.splitlines() if l.startswith("_P5_RUNNER = ")]
    assert len(lines) == 1
    runner = ast.literal_eval(lines[0][len("_P5_RUNNER = "):])
    sha_old = hashlib.sha256(runner.encode("utf-8")).hexdigest()
    for old, new in ((N5_RUNNER_HEAD_OLD, N5_RUNNER_HEAD_NEW), (N5_GEN_OLD, N5_GEN_NEW), (N5_FUNC_ANCHOR, N5_FUNC)):
        assert runner.count(old) == 1, old[:60]
        runner = runner.replace(old, new)
    sha_new = hashlib.sha256(runner.encode("utf-8")).hexdigest()
    assert s.count(f"_P5_RUNNER_SHA256 = '{sha_old}'") == 1
    s = s.replace(lines[0], "_P5_RUNNER = " + repr(runner)).replace(
        f"_P5_RUNNER_SHA256 = '{sha_old}'", f"_P5_RUNNER_SHA256 = '{sha_new}'")
    for old, new in ((N5_CELL_OLD, N5_CELL_NEW), (N5_CELL_FAST_OLD, N5_CELL_FAST_NEW)):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    anchor = "# ---- run (everything above is definitions) ----\n"
    assert s.count(anchor) == 1
    s = s.replace(anchor, N5_CELL_DEFS + anchor)
    nb["cells"][i]["source"] = s.splitlines(keepends=True)


def p_n5_after_p12(nb):
    """NetGen on a runner already patched by p_p12 (anchors on the G.update line instead of the generate call)."""
    global N5_GEN_OLD, N5_GEN_NEW
    upd = "        G.update(called=True, p6=p6, p5=p5, n_tr=n_tr, gen=[[str(g['key']), str(g['smiles']), int(g['parent'])] for g in res])\n"
    saved = (N5_GEN_OLD, N5_GEN_NEW)
    N5_GEN_OLD = upd
    N5_GEN_NEW = ("        npar = N5_PAR.get(str(scope.get('mid')), []) if N5_PAR else []\n"
                  "        if npar:\n"
                  "            res = list(res) + n5_products(npar, target, set(exclude_keys) | {g['key'] for g in res}, tol)\n" + upd)
    try:
        p_n5(nb)
    finally:
        N5_GEN_OLD, N5_GEN_NEW = saved


# ---- patch: FR weight W 3 -> 5 (FR cell, wherever it sits) ----
def p_w5(nb):
    replace(nb, FR, "_F3_W = float('3.0')\n",
            "_F3_W = float('5.0')   # Arsokan fw5: W 3.0 -> 5.0 (the FR score weighs more against the parent rank)\n")


# ---- patch: FR over an extended pool (needs p_fp5: the FR cell runs after P5) ----
FX_DEFS = (
    "# ===== Arsokan fx: FR 3-25 over an extended pool =====\n"
    "# P5 cuts every list at 25 after its insertions, so the parent guesses it pushed past rank 25 never reach FR, and P5's\n"
    "# new generated rows whose tail rank was above 25 were never inserted. Here FR also scores up to _FX_MAX_EXTRA such\n"
    "# candidates per list (displaced parent guesses in parent order first, then the new P5 rows in Engine 1 order; metric\n"
    "# identity not already in the list), appended at ranks 26.. so they start below every listed guess, and the list keeps\n"
    "# its length after the FR order. Ranks 1-2 stay locked. Without a P5 result in this session there is no extra candidate.\n"
    "_FX_MAX_EXTRA = 15\n"
    "\n"
    "\n"
    "def _fx_pool(m, lst):\n"
    "    extra = []\n"
    "    try:\n"
    "        if not str((globals().get('_p5_audit') or {}).get('status', '')).startswith('applied'):\n"
    "            return []\n"
    "        ident = globals().get('_p5_ident')\n"
    "        if ident is None:\n"
    "            return []\n"
    "        have = {ident(t) for t in lst} - {None}\n"
    "        seen = set(lst)\n"
    "        cands = []\n"
    "        par = (globals().get('_p5_raw') or {}).get(m)\n"
    "        if par:\n"
    "            cands += par.split(';')\n"
    "        d = ((globals().get('_p5_e1') or {}).get('molecules_data') or {}).get(m)\n"
    "        if d:\n"
    "            cands += [str(r[0]) for r in d['base'] if r[3]]\n"
    "        for t in cands:\n"
    "            if len(extra) >= _FX_MAX_EXTRA:\n"
    "                break\n"
    "            if not t or t in seen:\n"
    "                continue\n"
    "            k = ident(t)\n"
    "            if k is None or k in have:\n"
    "                continue\n"
    "            extra.append(t)\n"
    "            seen.add(t)\n"
    "            have.add(k)\n"
    "    except Exception:\n"
    "        return []\n"
    "    return extra\n"
    "\n"
    "\n"
    "def _fx_same(old, ext, new):\n"
    "    return (len(new) == len(old) and new[:_F3_KEEP] == old[:_F3_KEEP] and len(set(new)) == len(new)\n"
    "            and set(new) <= set(ext) and len(new) <= _F3_MAX_CAND)\n"
    "\n"
    "\n")
FX_RUN = "# ---- run (everything above is definitions) ----\n"
FX_SCORE_OLD = "            per, npk, errs = _f3_score_list(ff, nets, spectra.get(m), lst, device)\n"
FX_SCORE_NEW = ("            _fx_ext = lst + _fx_pool(m, lst)\n"
                "            n['fx_extra'] += len(_fx_ext) - len(lst)\n"
                "            per, npk, errs = _f3_score_list(ff, nets, spectra.get(m), _fx_ext, device)\n")
FX_BLEND_OLD = "            new = _f3_blend(lst, sc, w)\n            if not _f3_same(lst, new):\n"
FX_BLEND_NEW = ("            new = _f3_blend(_fx_ext, sc, w)[:len(lst)]\n"
                "            if not _fx_same(lst, _fx_ext, new):\n")
FX_COUNT_OLD = "                n['rank3_changed'] += int(new[_F3_KEEP] != lst[_F3_KEEP])\n"
FX_COUNT_NEW = (FX_COUNT_OLD + "                n['fx_rescued'] += len(set(new) - set(lst))\n")
FX_ASSERT_OLD = "            assert _f3_same(_f3_raw[_f3_m].split(';'), _f3_new[_f3_m].split(';')), _f3_m\n"
FX_ASSERT_NEW = ("            _fx_o, _fx_n = _f3_raw[_f3_m].split(';'), _f3_new[_f3_m].split(';')\n"
                 "            assert len(_fx_n) == len(_fx_o) and _fx_n[:_F3_KEEP] == _fx_o[:_F3_KEEP] and len(set(_fx_n)) == len(_fx_n) \\\n"
                 "                and all(_fx_n), _f3_m\n")


def p_fx(nb):
    replace(nb, FR, FX_RUN, FX_DEFS + FX_RUN)
    replace(nb, FR, FX_SCORE_OLD, FX_SCORE_NEW)
    replace(nb, FR, FX_BLEND_OLD, FX_BLEND_NEW)
    replace(nb, FR, FX_COUNT_OLD, FX_COUNT_NEW)
    replace(nb, FR, FX_ASSERT_OLD, FX_ASSERT_NEW)



def p_w2(nb):
    replace(nb, FR, "_F3_W = float('3.0')\n",
            "_F3_W = float('2.0')   # Arsokan fw2: W 3.0 -> 2.0 (class-3 FR-validation harness: W 2 best of 1-12)\n")


def p_d10(nb):
    replace(nb, FR, "_F3_KEEP = 2                    # parent ranks 1 and 2 never move\n",
            "_F3_KEEP = 2                    # parent ranks 1 and 2 never move\n"
            "_F3_DEPTH = 10                  # Arsokan d10: FR re-orders ranks 3-10 only (z over ranks 1-10); 11-25 keep their order\n")
    replace(nb, FR, "            new = _f3_blend(lst, sc, w)\n",
            "            new = _f3_blend(lst[:_F3_DEPTH], sc[:_F3_DEPTH], w) + lst[_F3_DEPTH:]   # Arsokan d10\n")


def p_o95(nb):
    """Arsokan o95: the extlib overlay promotes a reference structure to rank 1 only on a near-identical spectrum
    (cosine >= 0.95, or cosine >= 0.90 with >= 8 matched peaks); was cosine >= 0.80 with >= 4 peaks."""
    replace(nb, OVL, "    TAU = 0.80          # Minimum cosine similarity for direct Rank-1 hit\n",
            "    TAU = 0.95          # Arsokan o95: was 0.80 (held-out NP run: all 9 promotions at 0.80-0.95 demoted the right answer)\n"
            "    TAU_RICH, K_RICH = 0.90, 8   # Arsokan o95: or cosine >= 0.90 with >= 8 matched peaks\n")
    replace(nb, OVL, "            if match_smi is not None and best_cos >= TAU:\n",
            "            if match_smi is not None and (best_cos >= TAU or (best_cos >= TAU_RICH and best_k >= K_RICH)):   # Arsokan o95\n")


def p_ooff(nb):
    """Arsokan ooff (probe): extlib overlay never promotes (TAU above any cosine); the cell still runs and writes its audit."""
    replace(nb, OVL, "    TAU = 0.80          # Minimum cosine similarity for direct Rank-1 hit\n",
            "    TAU = 9.0           # Arsokan ooff: overlay off (was 0.80); probe of the overlay's public effect\n")


VARIANTS = {
    "ooff": ([p_fp5, p_ooff], "FR after the P5 tail + extlib overlay off (probe)"),
    "o95":([p_fp5, p_o95], "FR after the P5 tail + extlib overlay only on near-identical spectra (cos >= 0.95, or >= 0.90 with 8 peaks)"),
    "fw2d10o95": ([p_fp5, p_w2, p_d10, p_o95], "fw2d10 + extlib overlay only on near-identical spectra (cos >= 0.95, or >= 0.90 with 8 peaks)"),
    "fw5": ([p_fp5, p_w5], "FR 3-25 after the P5 tail, FR weight W 5.0 (was 3.0)"),
    "fw2": ([p_fp5, p_w2], "FR 3-25 after the P5 tail, FR weight W 2.0 (was 3.0)"),
    "fw2d10": ([p_fp5, p_w2, p_d10], "FR after the P5 tail, W 2.0, re-ordering ranks 3-10 only (11-25 keep their order)"),
    "fx": ([p_fp5, p_fx], "FR 3-25 after the P5 tail over an extended pool (P5-displaced and uninserted P5 candidates)"),
    "fxw5": ([p_fp5, p_fx, p_w5], "fx + FR weight W 5.0"),
    "pn5fx": ([p_p12, p_n5_after_p12, p_fp5, p_fx], "P5 tail with 12 parents + NetGen parents + FR 3-25 after it over an extended pool"),
    "pn5f": ([p_p12, p_n5_after_p12, p_fp5], "P5 tail with 12 parents + NetGen parents + FR 3-25 after it"),
    "n5": ([p_n5], "NetGen: network neighbours' top candidates as extra P5 generation parents"),
    "n5f": ([p_n5, p_fp5], "NetGen in the P5 tail + FR 3-25 moved after it"),
    "p12": ([p_p12], "P5 tail with up to 12 transformable-first parents (300 products)"),
    "p12f": ([p_p12, p_fp5], "P5 tail with 12 parents + FR 3-25 moved after it"),
    "fp5": ([p_fp5], "FR 3-25 moved after the P5 tail (it also orders P5's inserted candidates)"),
    "cf5": ([p_cf5], "own-spectrum confirmation only (veto off) after the D-L1 network"),
    "g90": ([p_g90], "network rank-1 change blocked where library_max >= 0.90 (ranks 2-25 keep the network order)"),
    "cg": ([p_cf5, p_g90], "cf5 + g90 (disjoint lists: library_max < 0.90 vs >= 0.90)"),
    "s3": ([p_cf5, p_g90, p_fp5], "cf5 + g90 + FR 3-25 after the P5 tail"),
    "s4": ([p_cf5, p_g90, p_p12, p_fp5], "cf5 + g90 + P5 tail with 12 parents + FR 3-25 after it"),
}


G_DIR = os.path.join(SCR, "pub", "goodpjw2008_enveda-casmi-stack-aprior-uniq-lb-0-498")
BASES = {
    "t495": (BASE_NB, BASE_META, "taeyangg4's public 0.495 notebook",
             "taeyangg4/casmi26-lb-0-495-network-fr-p5-tail"),
    "g498": (os.path.join(G_DIR, "enveda-casmi-stack-aprior-uniq-lb-0-498.ipynb"), os.path.join(G_DIR, "kernel-metadata.json"),
             "goodpjw2008's public 0.498 notebook (taeyangg4 0.495 + overlay checks + APRIOR + UNIQ)",
             "goodpjw2008/enveda-casmi-stack-aprior-uniq-lb-0-498"),
}
AP = "_AP_VERSION"                     # goodpjw2008's APRIOR cell


def p_ap75(nb):
    replace(nb, AP, "_AP_LAM = 0.5            # weight of the pattern log-likelihood\n",
            "_AP_LAM = 0.75           # Arsokan ap75: was 0.5 (weight of the pattern log-likelihood)\n")


def p_apall(nb):
    """APRIOR trained on every enveda-180 molecule instead of a 60k sample (and a cap that leaves room for it)."""
    replace(nb, AP, "_AP_N_TRAIN = 60000      # enveda-180 molecules used for training (seeded sample)\n",
            "_AP_N_TRAIN = 250000     # Arsokan apall: was 60000 (enveda-180 molecules used for training; 250000 = all)\n")
    replace(nb, AP, "_AP_CAP_S = 1500.0\n", "_AP_CAP_S = 3000.0       # Arsokan apall: was 1500.0 (room for the full training set)\n")


def p_ap600(nb):
    replace(nb, AP, "_AP_TREES = 300\n", "_AP_TREES = 600          # Arsokan ap600: was 300\n")


AP_Q = None   # observation-aware APRIOR: P(adduct kept in the test file | recorded), filled from casmi-vh-aprior


def p_apoa(nb):
    """Observation-aware likelihood: the test file keeps a subset of each molecule's spectra, so an adduct the structure
    is recorded with is observed with probability q_a * p_a(s). Only the 'not observed' terms change."""
    q = AP_Q
    assert q and len(q) == 4
    replace(nb, AP, "_AP_CLIP = 1e-3\n",
            "_AP_CLIP = 1e-3\n_AP_Q = %r   # Arsokan apoa: P(adduct kept in the test file | recorded), per _AP_OBS\n" % (tuple(q),))
    replace(nb, AP, "    P = np.clip(np.stack([m.predict_proba(X)[:, 1] for m in models], 1), _AP_CLIP, 1 - _AP_CLIP)\n",
            "    P = np.clip(np.stack([m.predict_proba(X)[:, 1] for m in models], 1), _AP_CLIP, 1 - _AP_CLIP)\n"
            "    P = P * np.asarray(_AP_Q, np.float64)  # Arsokan apoa: observed in the test file = recorded and kept\n")


NIST_OLD = "            for idx, row in meta_df.iterrows():\n                try:\n                    mode = 1 if 'pos'"


def p_nist(nb):
    """Team no-NIST rule: the external-library overlay drops the 117 GNPS-NIST14-MATCHES rows (Hemre's tp5/fpn edit)."""
    replace(nb, OVL, NIST_OLD,
            "            # Arsokan nist: drop NIST-matched rows (index kept, so mz_all[idx] stays aligned)\n"
            "            if 'source_library' in meta_df.columns:\n"
            "                _nist = meta_df['source_library'].astype(str).str.contains('NIST', case=False)\n"
            "                print(f'[apex-extlib] dropped {int(_nist.sum())} NIST-matched reference rows')\n"
            "                meta_df = meta_df[~_nist]\n" + NIST_OLD)


GVARIANTS = {   # variants on goodpjw2008's 0.498 stack
    "gfp5": ([p_fp5], "FR 3-25 moved after the P5 tail (P5 -> FR -> APRIOR -> UNIQ)"),
    "gfp5n": ([p_fp5, p_nist], "FR 3-25 after the P5 tail + external library without the 117 GNPS-NIST14-MATCHES rows"),
    "gfp5a75": ([p_fp5, p_ap75], "FR 3-25 after the P5 tail + APRIOR weight 0.75 (was 0.5)"),
    "gfp5ab": ([p_fp5, p_apall, p_ap600],
               "FR 3-25 after the P5 tail + APRIOR model trained on all enveda-180 molecules with 600 trees (was 60k, 300)"),
    "gfp5a75b": ([p_fp5, p_ap75, p_apall, p_ap600],
                 "FR 3-25 after the P5 tail + APRIOR weight 0.75 + APRIOR model on all enveda-180 molecules, 600 trees"),
}


def build(name):
    base = "g498" if name in GVARIANTS else "t495"
    patches, what = (GVARIANTS if base == "g498" else VARIANTS)[name]
    base_nb, base_meta, base_what, base_ref = BASES[base]
    slug = f"casmi26-om-{name}"
    nb = json.load(open(base_nb, encoding="utf-8"))
    base_code = [src(c) for c in nb["cells"] if c["cell_type"] == "code"]
    for p in patches:
        p(nb)
    notice = (
        f"# CASMI26 {slug}: {base_what} + {what}\n\n"
        f"Private test kernel of team Arsokan. Base: [{base_ref}](https://www.kaggle.com/code/{base_ref}); every base code "
        "cell is kept byte for byte, the cells marked 'Arsokan' are ours. The base's own write-up follows below.\n"
    ).splitlines(keepends=True)
    if nb["cells"][0]["cell_type"] == "markdown":
        nb["cells"][0]["source"] = notice
    else:
        nb["cells"].insert(0, dict(cell_type="markdown", metadata={}, source=notice))
    new_code = [src(c) for c in nb["cells"] if c["cell_type"] == "code"]
    kept = sum(1 for s in base_code if s in new_code)
    meta = json.load(open(base_meta, encoding="utf-8"))
    for k in ("id_no",):
        meta.pop(k, None)
    meta.update(id=f"{OWNER}/{slug}", title=slug, code_file=f"{slug}.ipynb", is_private=True, enable_gpu=True,
                enable_internet=False, keywords=[])
    out = os.path.join(HERE, "kernels", slug)
    os.makedirs(out, exist_ok=True)
    raw = json.dumps(nb, ensure_ascii=False, indent=1)
    open(os.path.join(out, f"{slug}.ipynb"), "w", encoding="utf-8", newline="\n").write(raw)
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8"), indent=1)
    print(f"{slug}: base code cells kept {kept}/{len(base_code)}, code cells now {len(new_code)}, "
          f"sha {hashlib.sha256(raw.encode()).hexdigest()[:12]}")


if __name__ == "__main__":
    for v in sys.argv[1:]:
        build(v)
