"""Second pipeline inside our notebook (E006_FUSE): the public notebook "CASMI26 E006 Window 40 ICE Fallback True" run as a script, and the
fusion of its submission table with ours.

The notebook (canqiang/casmi26-e006-window-40-ice-fallback-true, version 1, public score 0.440, Apache-2.0; snapshot and notice in
reference/e006/) is the v18 family: PubChem channel with a popularity prior in fixed slots, pool popularity 0.15, promotion rules, GLACIER on
[M+H]+ only, one vote per metric key, tail fill. Its code cells are embedded unchanged by tools/build_notebook.py (only when E006_FUSE is
set) and turned into one script here with three marked text patches:
  1. every literal '/kaggle/working' becomes the stage's own folder (the script also runs with that folder as its working directory, so its
     relative and absolute paths agree as they do in the original notebook);
  2. its engine-2 subprocess call goes through `_ours_engine2`, which hands it the engine-2 lists of our own run when the environment names
     them (OURS_E006_ENG) and otherwise runs the engine exactly as the notebook does. The two notebooks embed the same engine-2 code
     (pv.py and pv_fp.py byte-identical, casmi_engine.py identical except one comment) and run it with the same arguments;
  3. OURS_E006_FULL makes it process every molecule on a commit run (our FULL_ON_COMMIT); never set on a scoring rerun, where it does so anyway.
`display` (IPython) is defined as a no-op. Nothing else of the notebook's code is touched.

Fusion rules (pre-registered in scratch/e006/PREREG.md, reviews/e006-2026-10-10.md), all on the metric key (`metric_key`: stereo stripped,
tautomer canonical, first InChIKey block, computed from the SMILES like the official metric) with the first SMILES of a key as its
representative (ours first), 25 rows:
  'rrf'    score 1/(k + rank in ours) + w/(k + rank in theirs), E006_RRF_K (default 3), E006_RRF_W (default 1 = equal weights, the
           pre-registered rule); ties: ours' order, then theirs
  'fill'   ours' rows 1 .. p-1, then their rows that are not among ours' 25 (their order), then ours' remaining rows; E006_FILL_POS (default 6)
  'agree'  keys that both lists hold in their first five rows, by 1/rank + 1/rank; then the rest in the 'rrf' order
A molecule without a list of theirs (stage failed, placeholder row) keeps ours' row byte for byte.

"e006 improved" (E006_BASE, reviews/e006imp-2026-10-10.md): `script_improved` is the same script with four more marked patches. Each is inert
unless its environment variable is set, so the notebook's own behaviour is the default of every one:
  4. base lists: for a molecule that OURS_E006_BASE (a JSON file {molecule: [smiles, keys, lib_max, scores, formulas, pool ids]}) holds, the
     notebook's engine call is skipped and our base list (public ranker; below the gate the rank average with the fitted fold-safe ranker,
     with the joined PubChem rows) takes its place; the notebook's own pool-popularity block (0.15) then acts on it as on its own list.
     A molecule the file does not hold is computed by the notebook's own engine as before;
  5. OURS_E006_PROM_REL: the `rel > 250` branch of its promotion rule gets this threshold (1e12 = branch off; the slots rule keeps 600);
  6. OURS_E006_GL_ALL: GLACIER scores every covered spectrum instead of [M+H]+ only;
  7. OURS_E006_PROM_FINAL: a row promoted by the popularity branch (S > 6 and popularity >= 5) leads the FINAL list as well (the notebook's
     later engine fusion and key vote let pool rows pass it again); applied after its last cell, on its own metric identity.
Two more, for the validation mode and the fallback run (reviews/diag67-2026-10-12.md), inert without their variable as well:
  8. OURS_E006_COMP: the competition folder (test.parquet, train.parquet, sample_submission.csv) is this folder instead of the one found
     below /kaggle/input, and its pre-flight check accepts it: our validation mode's held-out test and purged library;
  9. OURS_E006_PC: its PubChem-channel subprocess is replaced by a copy of this file (the lists an earlier run of the SAME script wrote for
     the same test: the channel does not depend on the base lists), so that a second run after a failed one does not pay the channel twice.
E006_BASE = 'e006' runs the script with none of the variables 4 to 7 set: the public notebook itself (the baseline profile).

Diagnostics: `run_script` starts a `Monitor` thread next to the script that writes a time-stamped copy of its log (`<log>.timed`) and a
resource trace (`<log>.res.jsonl`: resident memory of the kernel and of the script's process tree, available RAM, free disk, GPU memory);
`phase_times` reads the stage clock at the script's own milestones from the time-stamped log; `mrr_table` scores a table against
validation labels on the metric key.
"""
import json, os, signal, subprocess, sys, threading, time

RULES = ('rrf', 'fill', 'agree')
WORK_LITERAL = '/kaggle/working'
ENG_CALL = "    rr = subprocess.run([sys.executable, os.path.join(ENG_DIR, 'eng_runner.py'),"
ENG_CALL_NEW = "    rr = _ours_engine2([sys.executable, os.path.join(ENG_DIR, 'eng_runner.py'),"
RERUN_LINE = "IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'"
RERUN_LINE_NEW = RERUN_LINE + " or bool(os.getenv('OURS_E006_FULL'))   # ours: patch 3"
ENG_REFERENCE = dict(ENG_W_PV=0.88, ENG_W_A=[0.35, 0.55], ENG_SEEDS=[0, 1], ENG_N_ANALOG=200, ENG_ORDER_K=80, ENG_TOPK=40, ENG_REPS=1,
                     ENG_EXTRA_FPNET=None)
PREAMBLE = '''# generated by nb/embed/e006_stage.py from the code cells of the public notebook
# canqiang/casmi26-e006-window-40-ice-fallback-true (version 1, Apache License 2.0). The cells follow unchanged except for the three
# patches described in e006_stage.py (work folder, shared engine-2 lists, full run on request).
import os, shutil, subprocess, sys, types


def display(*a, **k):                         # IPython's display: the notebook shows one audit table with it
    pass


def _ours_engine2(cmd, **kw):                 # ours: patch 2. cmd = [python, eng_runner.py, test, train, sample, out]
    src = os.environ.get('OURS_E006_ENG')
    if src and os.path.exists(src):
        shutil.copyfile(src, cmd[5])
        return types.SimpleNamespace(stdout='E006_STAGE engine-2 lists shared with the first pipeline: ' + src, stderr='', returncode=0)
    print('E006_STAGE engine 2 runs in this pipeline (no shared lists)', flush=True)
    return subprocess.run(cmd, **kw)


'''


def script_from_cells(cells, work):
    """The notebook's code cells (list of source strings, in order) -> one Python script that works inside the folder `work`."""
    assert work.startswith(WORK_LITERAL + '/') and "'" not in work
    src = '\n\n'.join('# ---- cell %d\n%s' % (i, c.rstrip('\n')) for i, c in enumerate(cells)) + '\n'
    for old, new, n in ((ENG_CALL, ENG_CALL_NEW, 1), (RERUN_LINE, RERUN_LINE_NEW, 1)):
        assert src.count(old) == n, ('the notebook source changed: patch does not apply', old, src.count(old))
        src = src.replace(old, new)
    assert src.count(WORK_LITERAL) >= 15, src.count(WORK_LITERAL)
    src = src.replace(WORK_LITERAL, work)                       # patch 1
    return PREAMBLE + src


BASE_CALL = "            C, X, F, names, info = V.run(spectra, target)\n            lib_max = float(info.get('lib_max', 0.0))\n"
BASE_CALL_NEW = ("            _ob = _OURS_BASE.get(str(mid))                 # ours: patch 4\n"
                 "            if _ob is not None:\n"
                 "                smis, keys, lib_max, scs, forms, pids_ = _ours_base_rows(_ob, TOPN); C = types.SimpleNamespace(pid=[])\n"
                 "            else:\n"
                 "                C, X, F, names, info = V.run(spectra, target)\n"
                 "                lib_max = float(info.get('lib_max', 0.0))\n")
REL_TEXT = "pop_top >= POP_TAU) or (rel > 250.0):\n            stats['promoted'] += 1\n"
REL_TEXT_NEW = ("pop_top >= POP_TAU) or (rel > _OURS_PROM_REL):   # ours: patch 5\n"
                "            stats['promoted'] += 1; _ours_note_promotion(mid, p, S, pop_top, S_TAU, POP_TAU)   # ours: patch 7\n")
GL_LINE = "    gl_items = [it for it in gl_items if it['spectra']]\n"
GL_LINE_NEW = GL_LINE + ("    if os.environ.get('OURS_E006_GL_ALL'):             # ours: patch 6\n"
                         "        gl_items = [it for it in items if it['spectra']]; print('E006_STAGE GLACIER on every covered spectrum:', len(gl_items), flush=True)\n")
COMP_LINE = "COMP = os.path.dirname(find('test.parquet'))\n"
COMP_LINE_NEW = COMP_LINE + "COMP = os.environ.get('OURS_E006_COMP') or COMP   # ours: patch 8\n"
PRE_ASSERT = "    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}\n"
PRE_ASSERT_NEW = "    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'} or os.environ.get('OURS_E006_COMP')   # ours: patch 8\n"
PC_CALL = "    rr = subprocess.run([sys.executable, '%s/pc_runner.py', V3CODE, V3MOD, POOL_DIR, PC_DIR,\n"
PC_CALL_NEW = "    rr = _ours_pc_channel([sys.executable, '%s/pc_runner.py', V3CODE, V3MOD, POOL_DIR, PC_DIR,   # ours: patch 9\n"
PREAMBLE_IMP = '''
# "e006 improved" (E006_BASE): four more marked patches by nb/embed/e006_stage.py (4 base lists handed over, 5 threshold of the promotion's
# rel branch, 6 GLACIER on every covered spectrum, 7 popularity-branch promotion kept in the final list). Every changed line of the cells below
# is tagged '# ours: patch N'; patches 5 to 7 are inert unless their environment variable is set.
import json as _ours_json

_OURS_BASE = {}                               # ours: patch 4. {molecule: [smiles, keys, lib_max, scores, formulas, pool ids]}
if os.environ.get('OURS_E006_BASE') and os.path.exists(os.environ['OURS_E006_BASE']):
    _OURS_BASE = _ours_json.load(open(os.environ['OURS_E006_BASE']))
print('E006_STAGE base lists handed over by the first pipeline:', len(_OURS_BASE), flush=True)
_OURS_PROM_REL = float(os.environ.get('OURS_E006_PROM_REL') or 250.0)   # ours: patch 5 (250.0 = the notebook's own value)
_OURS_PROM = {}                               # ours: patch 7. {molecule: promoted row} for promotions by the popularity branch


def _ours_pc_channel(cmd, **kw):              # ours: patch 9. cmd = [python, pc_runner.py, V3CODE, V3MOD, POOL_DIR, PC_DIR, test, out, workers]
    src = os.environ.get('OURS_E006_PC')
    if src and os.path.exists(src):
        shutil.copyfile(src, cmd[7])
        return types.SimpleNamespace(stdout='E006_STAGE PubChem lists of the earlier run of this script reused: ' + src, stderr='', returncode=0)
    return subprocess.run(cmd, **kw)


def _ours_base_rows(b, topn):
    n = min(int(topn), len(b[0]))
    return (list(b[0][:n]), list(b[1][:n]), float(b[2]), [float(x) for x in b[3][:n]], [str(f).split('|')[0] for f in b[4][:n]],
            [int(x) for x in b[5][:n]])


def _ours_note_promotion(mid, p, S, pop_top, s_tau, pop_tau):
    try:
        if S is not None and float(S) > s_tau and pop_top >= pop_tau and p.get('pc'):
            _OURS_PROM[str(mid)] = p['pc'][0]
    except Exception as e:
        print('E006_STAGE promotion note failed', mid, repr(e), flush=True)


def _ours_promote_rows(L, row, ident, n=25):
    """row leads L; a row of L with the same identity is moved to the front instead of being duplicated."""
    k = ident(row)
    same = [x for x in L if (ident(x) == k if k is not None else x == row)]
    return ([same[0] if same else row] + [x for x in L if not (ident(x) == k if k is not None else x == row)])[:n]


def _ours_promote_final(path, ident):
    """Patch 7: every row promoted by the popularity branch leads its final list (the notebook's metric identity; at most 25 rows)."""
    import pandas as _pd
    t = _pd.read_csv(path, dtype=str, keep_default_na=False); n = moved = 0; out = []
    for m, s in zip(t.molecule_id, t.smiles):
        row = _OURS_PROM.get(str(m)); L = [x for x in str(s).split(';') if x]
        if row is not None and L and L != ['CCO']:
            L2 = _ours_promote_rows(L, row, ident); n += 1; moved += int(L2[:1] != L[:1]); L = L2
        out.append(';'.join(L))
    t['smiles'] = out; t.to_csv(path, index=False)
    print('E006_STAGE final promotion: popularity-branch promotions', n, '| first row changed', moved, flush=True)
'''
POSTAMBLE_IMP = '''

# ---- ours: after the notebook's last cell (patch 7 and the hand-over count)
print('E006_STAGE base lists used:', sum(1 for _m in BASE if str(_m) in _OURS_BASE), 'of', len(BASE), '| promotions by the popularity branch:',
      len(_OURS_PROM), flush=True)
if os.environ.get('OURS_E006_PROM_FINAL'):
    _ours_promote_final('submission.csv', metric_identity)
'''


def script_improved(cells, work):
    """script_from_cells plus the four patches of "e006 improved" (module docstring); every patch is asserted to match the source once."""
    src = script_from_cells(cells, work)
    assert src.startswith(PREAMBLE)
    body = src[len(PREAMBLE):]
    for old, new in improved_patches(work):
        assert body.count(old) == 1, ('the notebook source changed: patch does not apply', old, body.count(old))
        body = body.replace(old, new)
    return PREAMBLE + PREAMBLE_IMP + body + POSTAMBLE_IMP


def improved_patches(work):
    """(old, new) text pairs of the patches 4 to 9, in the order they are applied (the PubChem call names the work folder)."""
    return ((BASE_CALL, BASE_CALL_NEW), (REL_TEXT, REL_TEXT_NEW), (GL_LINE, GL_LINE_NEW), (COMP_LINE, COMP_LINE_NEW), (PRE_ASSERT, PRE_ASSERT_NEW),
            (PC_CALL % work, PC_CALL_NEW % work))


def improved_env(cfg):
    """Environment variables of the patches 5 to 7 for a CFG (only those that are switched on)."""
    env = {}
    if cfg.get('E006_PROM_REL') is not None:
        env['OURS_E006_PROM_REL'] = repr(float(cfg['E006_PROM_REL']))
    if cfg.get('E006_GL_ALL'):
        env['OURS_E006_GL_ALL'] = '1'
    if cfg.get('E006_PROM_FINAL'):
        env['OURS_E006_PROM_FINAL'] = '1'
    return env


def base_for_handover(base, join_pop=None, join_mu=0.0, reorder=None):
    """Our BASE {mid: [smiles, keys, lib_max, scores, formulas, pool ids]} as the JSON-able dict of patch 4 (molecules with rows only).
    join_mu > 0 (E006_JOIN_POP_MU): the joined rows (pool id -2) are first re-sorted among their own slots by z + join_mu x popularity
    (reorder = fusion_core.popularity_reorder_join, join_pop = {mid: {smiles: popularity}}); formulas stay untagged.
    Returns (dict, number of lists the popularity term touched)."""
    out = {}; n_jp = 0
    for m, v in base.items():
        if len(v) < 6 or not v[0]:
            continue
        smis, keys, scs = list(v[0]), list(v[1]), [float(x) for x in v[3]]
        forms, pids = [str(f).split('|')[0] for f in v[4]], [int(x) for x in v[5]]
        if join_mu > 0 and reorder is not None and join_pop is not None and len(scs) > 1 and any(p == -2 for p in pids):
            jp = join_pop.get(m) or {}
            smis, keys, scs, forms, pids = reorder(smis, keys, scs, forms, pids, None, 0.0,
                                                   [jp.get(s) if p == -2 else None for s, p in zip(smis, pids)], float(join_mu), False)
            forms = [f.split('|')[0] for f in forms]; n_jp += 1
        out[str(m)] = [smis, keys, float(v[2]), scs, forms, pids]
    return out, n_jp


def check_table(theirs, sample_ids, base_ids, max_rank=25):
    """The improved table must cover the sample's ids, hold 1 to 25 rows each, and hold real rows for (nearly) every molecule that has a
    base list. Returns the number of such molecules with real rows; raises AssertionError otherwise."""
    assert set(theirs) == set(map(str, sample_ids)), (len(theirs), len(sample_ids))
    assert all(1 <= len(v) <= max_rank for v in theirs.values()), 'a list is empty or longer than 25'
    real = sum(1 for m in base_ids if not is_placeholder(theirs.get(str(m)) or []))
    assert real >= 0.98 * len(base_ids), (real, len(base_ids))
    return real


def engine_shareable(cfg):
    """True when our engine-2 run used exactly the reference's arguments, i.e. produced what the second notebook's own run would."""
    return bool(cfg.get('USE_ENG')) and all(cfg.get(k) == v for k, v in ENG_REFERENCE.items())


MARKS = (('engine-2 lists', r'^engine lists \d+|^E006_STAGE engine 2 runs'), ('PubChem channel', r'^pubchem lists \d+'), ('spectrum cache', r'^cache built'),
         ('engine ready', r'^engine ready'), ('base lists', r'^errors \d+ \| popularity re-ranked'), ('ICEBERG input', r'^ICE input molecules'),
         ('ICEBERG', r'^ICE meta|^ICE FAILED'), ('GLACIER', r'^GL meta|^GL FAILED'), ('merge', r'^merge stats'), ('engine fusion', r'^V1 inference complete'),
         ('tail fill', r'^E004_VERIFICATION'), ('overlay', r'^FACTORIAL_VERIFICATION'), ('end', r'^E006_STAGE base lists used'))
LAST_MONITOR = {}


class Monitor(threading.Thread):
    """Next to the running script: a time-stamped copy of its log and a resource trace. Reads only; never touches the script's process."""

    def __init__(self, pid, work, log_path, every=10.0, res_every=30.0):
        super().__init__(daemon=True)
        self.pid, self.work, self.log_path, self.every, self.res_every = pid, work, log_path, every, res_every
        self.t0 = time.time(); self.stop_ev = threading.Event(); self.off = 0; self.tail = ''; self.rows = []; self.last_line = ''

    def snapshot(self):
        d = dict(t=round(time.time() - self.t0))
        try:
            import psutil, shutil
            d['kernel_rss_gb'] = round(psutil.Process().memory_info().rss / 2 ** 30, 2)
            rss = n = 0
            try:
                p = psutil.Process(self.pid)
                for q in [p] + p.children(recursive=True):
                    try:
                        rss += q.memory_info().rss; n += 1
                    except Exception:
                        pass
            except Exception:
                pass
            vm = psutil.virtual_memory()
            d.update(stage_rss_gb=round(rss / 2 ** 30, 2), stage_procs=n, ram_avail_gb=round(vm.available / 2 ** 30, 2), ram_total_gb=round(vm.total / 2 ** 30, 1),
                     disk_free_gb=round(shutil.disk_usage(self.work).free / 2 ** 30, 2))
        except Exception as e:
            d['err'] = repr(e)[:120]
        try:
            o = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15).stdout
            d['gpu_mb'] = [int(x) for x in o.split()]
        except Exception:
            pass
        d['line'] = self.last_line[:100]
        return d

    def pump(self):
        try:
            with open(self.log_path, 'rb') as fh:
                fh.seek(self.off); new = fh.read(); self.off += len(new)
            text = self.tail + new.decode('utf-8', errors='replace'); lines = text.split('\n'); self.tail = lines.pop()
            if lines:
                t = time.time() - self.t0
                with open(self.log_path + '.timed', 'a', encoding='utf-8') as out:
                    for l in lines:
                        out.write('[+%6.0fs] %s\n' % (t, l))
                keep = [l for l in lines if l.strip()]
                if keep:
                    self.last_line = keep[-1]
        except Exception:
            pass

    def run(self):
        nxt = 0.0
        while not self.stop_ev.wait(self.every):
            self.pump()
            if time.time() - self.t0 >= nxt:
                nxt = time.time() - self.t0 + self.res_every
                r = self.snapshot(); self.rows.append(r)
                try:
                    with open(self.log_path + '.res.jsonl', 'a') as out:
                        out.write(json.dumps(r) + '\n')
                except Exception:
                    pass

    def finish(self):
        self.stop_ev.set(); self.join(30); self.pump(); self.rows.append(self.snapshot())

        def ext(k, f):
            v = [r[k] for r in self.rows if r.get(k) is not None]
            return f(v) if v else None
        g = [max(r['gpu_mb']) for r in self.rows if r.get('gpu_mb')]
        return dict(samples=len(self.rows), peak_stage_rss_gb=ext('stage_rss_gb', max), peak_kernel_rss_gb=ext('kernel_rss_gb', max),
                    min_ram_avail_gb=ext('ram_avail_gb', min), ram_total_gb=ext('ram_total_gb', max), min_disk_free_gb=ext('disk_free_gb', min),
                    peak_gpu_mb=max(g) if g else None)


def phase_times(timed_text):
    """{milestone: seconds on the stage clock when the script first printed it}, from the time-stamped log (resolution = the monitor's 10 s)."""
    import re
    out = {}
    for line in timed_text.split('\n'):
        m = re.match(r'\[\+\s*(\d+)s\] (.*)', line)
        if not m:
            continue
        for name, pat in MARKS:
            if name not in out and re.search(pat, m.group(2)):
                out[name] = int(m.group(1))
    return out


def run_script(script, work, env, timeout_s, log_path, monitor=True):
    """Run the script in its own process group; kill the whole group at the timeout. -> (return code or None, seconds, 'ok' | 'timeout' | 'failed').
    monitor: a Monitor thread runs next to it (time-stamped log, resource trace); its summary is left in LAST_MONITOR."""
    t0 = time.time(); mon = None; LAST_MONITOR.clear()
    for ext in ('.timed', '.res.jsonl'):
        if os.path.exists(log_path + ext):
            os.remove(log_path + ext)
    with open(log_path, 'w') as fh:
        kw = dict(start_new_session=True) if os.name == 'posix' else {}
        p = subprocess.Popen([sys.executable, '-u', script], cwd=work, env=env, stdout=fh, stderr=subprocess.STDOUT, **kw)
        if monitor:
            try:
                mon = Monitor(p.pid, work, log_path); mon.start()
            except Exception:
                mon = None
        try:
            rc = p.wait(timeout=max(1.0, float(timeout_s)))
            st = 'ok' if rc == 0 else 'failed'
        except subprocess.TimeoutExpired:
            try:
                if os.name == 'posix':
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                else:
                    p.kill()
            except Exception:
                pass
            p.wait()
            rc, st = None, 'timeout'
    if mon is not None:
        try:
            LAST_MONITOR.update(mon.finish())
        except Exception as e:
            LAST_MONITOR.update(error=repr(e)[:200])
    return rc, time.time() - t0, st


def mrr_table(table, labels, key=None, max_rank=25):
    """Validation score of one table {molecule: [smiles]} against {molecule: answer SMILES} on the metric key.
    -> (dict(mrr, top1, hit, n), {molecule: reciprocal rank})."""
    key = key or metric_key
    rr = {}
    for m, ans in labels.items():
        t = key(ans); r = 0.0
        for i, s in enumerate((table.get(str(m)) or [])[:max_rank], 1):
            if t is not None and key(s) == t:
                r = 1.0 / i; break
        rr[str(m)] = r
    n = max(1, len(rr)); v = list(rr.values())
    return dict(mrr=round(sum(v) / n, 4), top1=round(sum(x == 1.0 for x in v) / n, 4), hit=round(sum(x > 0 for x in v) / n, 4), n=len(rr)), rr


# ------------------------------------------------------------------------------------------------ fusion ----
def dedup(smis, key):
    """[(key, smiles)] of the first row of every key, order kept; rows without a key are dropped (they cannot score)."""
    out, seen = [], set()
    for s in smis:
        k = key(s)
        if k is None or k in seen:
            continue
        seen.add(k); out.append((k, s))
    return out


def _rrf_order(o, e, k, w=1.0):
    ro = {q: r for r, (q, _) in enumerate(o, 1)}; re_ = {q: r for r, (q, _) in enumerate(e, 1)}
    sc = {q: (1.0 / (k + ro[q]) if q in ro else 0.0) + (w / (k + re_[q]) if q in re_ else 0.0) for q in list(ro) + [q for q in re_ if q not in ro]}
    big = 10 ** 6
    return sorted(sc, key=lambda q: (-sc[q], ro.get(q, big), re_.get(q, big)))


def fuse_lists(ours, theirs, key, rule, rrf_k=3.0, fill_pos=6, n=25, rrf_w=1.0):
    """One molecule: two ranked SMILES lists -> the fused list (at most n rows, one per metric key)."""
    assert rule in RULES, rule
    o = dedup(ours, key)[:n]; e = dedup(theirs, key)[:n]
    rep = {}
    for q, s in o + e:
        rep.setdefault(q, s)
    if rule == 'rrf':
        order = _rrf_order(o, e, float(rrf_k), float(rrf_w))
    elif rule == 'fill':
        p = max(1, int(fill_pos)); ok = [q for q, _ in o]; have = set(ok)
        order = ok[:p - 1] + [q for q, _ in e if q not in have] + ok[p - 1:]
    else:
        ro = {q: r for r, (q, _) in enumerate(o, 1)}; re_ = {q: r for r, (q, _) in enumerate(e, 1)}
        both = [q for q in ro if ro[q] <= 5 and re_.get(q, 99) <= 5]
        both.sort(key=lambda q: (-(1.0 / ro[q] + 1.0 / re_[q]), ro[q]))
        hb = set(both)
        order = both + [q for q in _rrf_order(o, e, 3.0) if q not in hb]
    return [rep[q] for q in order[:n]]


def is_placeholder(smis):
    return not smis or all(s == 'CCO' for s in smis)


def fuse_tables(ours, theirs, key, rule, rrf_k=3.0, fill_pos=6, n=25, rrf_w=1.0):
    """{mid: [smiles]} x {mid: [smiles]} -> ({mid: [smiles]}, stats). A molecule without a list of theirs keeps ours' list unchanged."""
    out = {}; st = dict(rule=rule, molecules=len(ours), fused=0, theirs_missing=0, changed=0, changed_top1=0, new_rows=0, short=0)
    for mid, o in ours.items():
        e = theirs.get(mid) or []
        if is_placeholder(e) or not dedup(e, key):                # no usable second list: ours stays byte for byte
            out[mid] = list(o); st['theirs_missing'] += 1
            continue
        f = fuse_lists([] if is_placeholder(o) else o, e, key, rule, rrf_k=rrf_k, fill_pos=fill_pos, n=n, rrf_w=rrf_w)
        if not f:
            out[mid] = list(o); st['theirs_missing'] += 1
            continue
        ko = {key(s) for s in o[:n]}
        st['fused'] += 1; st['changed'] += int(f != list(o)); st['changed_top1'] += int(f[:1] != list(o[:1]))
        st['new_rows'] += sum(key(s) not in ko for s in f); st['short'] += int(len(f) < n)
        out[mid] = f
    return out, st


# ------------------------------------------------------------------------------------------- metric key ----
_RD = {}


def metric_key(smiles):
    """The official metric's key (metric notebook v14+): stereo removed, tautomer-canonical form, first InChIKey block; None if no parse."""
    try:
        if not _RD:
            from rdkit import Chem, RDLogger
            from rdkit.Chem.MolStandardize import rdMolStandardize
            RDLogger.DisableLog('rdApp.*')
            _RD['Chem'] = Chem; _RD['taut'] = rdMolStandardize.TautomerEnumerator()
        Chem = _RD['Chem']
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        ik = Chem.MolToInchiKey(_RD['taut'].Canonicalize(mol))
        return ik[:14] if ik else None
    except Exception:
        return None


def metric_keys(smiles, workdir=None, workers=4, min_rows=200):
    """{smiles: key}. With a work folder and enough rows the keys are computed by a fresh Python process with `workers` workers (this file run
    as a script), so that the notebook kernel itself is never forked; any problem there falls back to computing them in this process."""
    uniq = list(dict.fromkeys(s for s in smiles if isinstance(s, str)))
    if workdir and workers > 1 and len(uniq) >= min_rows:
        try:
            os.makedirs(workdir, exist_ok=True)
            fi, fo = os.path.join(workdir, 'keys_in.json'), os.path.join(workdir, 'keys_out.json')
            json.dump(uniq, open(fi, 'w'))
            if os.path.exists(fo):
                os.remove(fo)
            r = subprocess.run([sys.executable, os.path.abspath(__file__), fi, fo, str(int(workers))], capture_output=True, text=True, timeout=3600)
            keys = json.load(open(fo))
            assert len(keys) == len(uniq), (len(keys), len(uniq), r.stderr[-300:])
            return dict(zip(uniq, keys))
        except Exception as e:
            print('metric_keys: worker process failed, computing here:', repr(e)[:300], flush=True)
    return {s: metric_key(s) for s in uniq}


def read_table(path):
    import pandas as pd
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {str(m): [x for x in str(s).split(';') if x] for m, s in zip(t.molecule_id, t.smiles)}


if __name__ == '__main__':                    # python e006_stage.py keys_in.json keys_out.json WORKERS  (see metric_keys)
    _fi, _fo, _nw = sys.argv[1], sys.argv[2], int(sys.argv[3])
    _sm = json.load(open(_fi))
    import multiprocessing as _mp
    with _mp.Pool(_nw) as _p:
        _keys = _p.map(metric_key, _sm, chunksize=40)
    json.dump(_keys, open(_fo + '.tmp', 'w')); os.replace(_fo + '.tmp', _fo)
