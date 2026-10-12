"""E006_BASE: no silent degradation (E006_BASE_ON_FAIL), the baseline mode 'e006', the stage monitor, the mixed validation set.

Checks, without competition data (reviews/diag67-2026-10-12.md):
  * defaults: E006_BASE_ON_FAIL is 'fail'; the deadlines leave room (first run + fallback run inside the hard deadline, hard deadline inside 9 h);
  * the stage cell and the final cell with a stubbed script run:
      - 'fail': a failed / late stage raises in the stage cell, no submission.csv is left, the log names the policy;
      - 'e006': the script runs a second time WITHOUT the hand-over variables, with the PubChem lists of the first run (OURS_E006_PC) and the
        shared engine-2 lists; its table is the submission and the log says that it is e006's own table; both runs failing raises;
      - 'ours': our own table stays and the log says so (the 2026-10-11 behaviour, now explicit);
      - a healthy stage whose table cannot be written by the final cell raises instead of leaving the stand-in as the submission;
      - STRICT with any fallback raises at the very end, after the report;
      - mode 'e006': no hand-over, no first pipeline's names needed, the script's table is the submission; a failure raises;
      - validation: the script is pointed at the validation folder (OURS_E006_COMP, OURS_E006_FULL) and every table is scored;
  * a build with E006_BASE 'e006' holds none of our cells 03 to 09 and the datasets of casmi26-e006r;
  * patches 8 / 9 in isolation (COMP override, PubChem lists reused) and inert without their variables;
  * run_script on real child processes: return code, timeout kill, the time-stamped log, the resource trace, phase_times;
  * mrr_table; build_validation_mix on a synthetic train.parquet: classes, purged rows, test spectra removed from the library.
usage: python tests/test_e006_onfail.py   (or pytest)
"""
import hashlib, json, os, subprocess, sys, tempfile, time, types

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.join(SOL, 'tools'))
import e006_stage as ES
import fusion_core as FC
import build_notebook as BN


def load_cfg():
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    return ns['CFG']


def snapshot_cells():
    nb = json.load(open(BN.E006_NB, encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


IDS = ['m1', 'm2', 'm3']
GOOD_LOG = 'x\nE006_STAGE base lists used: 3 of 3 | promotions by the popularity branch: 0\n'


def run_cells(tmp, cfg_over, results, labels=None, break_final=False, is_rerun=True):
    """nb/e006imp_stage.py and nb/e006imp_final.py with a stubbed script run. results = one (returncode, status, table or None, log text) per
    run of the script, in order. -> (namespace, log lines, env of every run, final table or None, the exception that ended the cells or None)."""
    import pandas as pd
    cfg = dict(dict(load_cfg(), E006_BASE='ours', FWD_GL_ALONE=True), **cfg_over)
    work = os.path.join(tmp, 'work'); comp = os.path.join(tmp, 'comp'); ours = os.path.join(tmp, 'ours'); stage = os.path.join(work, 'stage_cache')
    for d in (work, comp, os.path.join(ours, 'e006'), stage):
        os.makedirs(d, exist_ok=True)
    json.dump(snapshot_cells(), open(os.path.join(ours, 'e006', 'cells.json'), 'w'))
    pd.DataFrame(dict(molecule_id=IDS, smiles='CCO')).to_csv(os.path.join(comp, 'sample_submission.csv'), index=False)
    if labels:
        pd.DataFrame(dict(molecule_id=list(labels), smiles=list(labels.values()), val_class='all')).to_csv(os.path.join(stage, 'val_labels.csv'), index=False)
    base = {'m1': [['A', 'B'], ['ka', 'kb'], 0.3, [2.0, 1.0], ['F', 'F'], [5, -2]], 'm2': [['C'], ['kc'], 0.95, [1.0], ['G'], [3]], 'm3': [['D'], ['kd'], 0.5, [1.0], ['G'], [4]]}
    envs, todo = [], list(results)

    def fake_run(script, w, env, timeout_s, log_path, monitor=True):
        envs.append(dict({k: v for k, v in env.items() if k.startswith('OURS_E006_')}, _timeout=timeout_s))
        compile(open(script).read(), 'script', 'exec')
        rc, st, table, lg = todo.pop(0)
        open(log_path, 'w').write(GOOD_LOG if lg is None else lg)
        open(log_path + '.timed', 'w').write('[+     3s] pubchem lists 3 2s\n[+     9s] E006_STAGE base lists used: 2 of 2\n')
        json.dump({m: dict(pc=['P']) for m in IDS}, open(os.path.join(w, 'pc_lists.json'), 'w'))
        if table is not None:
            pd.DataFrame(dict(molecule_id=IDS, smiles=table)).to_csv(os.path.join(w, 'submission.csv'), index=False)
        ES.LAST_MONITOR.clear(); ES.LAST_MONITOR.update(samples=1)
        return rc, 5.0, st
    real = ES.run_script; ES.run_script = fake_run
    real_si = ES.script_improved; ES.script_improved = lambda cells, w: real_si(cells, '/kaggle/working/e006')   # the work folder must be Kaggle's
    logs = []
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
    ns = dict(CFG=cfg, FAILS=[], WORK=work, COMP=comp, OURS=ours, STAGE=stage, T0=time.time(), IS_RERUN=is_rerun, os=os, json=json, time=time, pd=pd,
              fusion_core=FC, log=lambda *a: logs.append(' '.join(map(str, a))), dump=lambda *a: None, find=None, PC_DIR=None, resources=lambda t: None)
    if cfg['E006_BASE'] == 'ours':
        ns.update(BASE=base, PC={}, ENG={'m1': 1}, torch=torch)
    err = None
    try:
        exec(compile(open(os.path.join(SOL, 'nb', 'e006imp_stage.py'), encoding='utf-8').read(), 'stage', 'exec'), ns)
        ns['cfg_after_stage'] = dict(ns['CFG'])
        if cfg['E006_BASE'] == 'ours':             # cell 08: the stand-in / our own table
            ns['samp'] = pd.read_csv(os.path.join(comp, 'sample_submission.csv')); ns['sub'] = pd.DataFrame(dict(molecule_id=IDS, smiles=['o1;o2', 'o3', 'o4']))
            ns['sub'].to_csv(os.path.join(work, 'submission.csv'), index=False); json.dump({}, open(os.path.join(work, 'run_manifest.json'), 'w'))
        if break_final:
            ns['fusion_core'] = types.SimpleNamespace(validate=lambda *a: (_ for _ in ()).throw(AssertionError('broken')), popularity_reorder_join=None)
        exec(compile(open(os.path.join(SOL, 'nb', 'e006imp_final.py'), encoding='utf-8').read(), 'final', 'exec'), ns)
    except Exception as e:
        err = e
    finally:
        ES.run_script = real; ES.script_improved = real_si
    f = os.path.join(work, 'submission.csv')
    return ns, logs, envs, (list(pd.read_csv(f, dtype=str).smiles) if os.path.exists(f) else None), err


def test_defaults():
    cfg = load_cfg()
    assert cfg['E006_BASE'] is None and cfg['E006_BASE_ON_FAIL'] == 'fail' and cfg['VAL_AS_RERUN'] is False
    assert cfg['E006_HARD_DEADLINE_S'] <= 9 * 3600 - 15 * 60 and 0 < cfg['E006_FALLBACK_S'] < cfg['E006_HARD_DEADLINE_S']
    assert cfg['E006_BASE_DEADLINE_S'] < cfg['E006_HARD_DEADLINE_S']


def test_policy_fail():
    for res in ((1, 'failed', ['e1', 'e2', 'e3'], None), (None, 'timeout', None, None), (0, 'ok', ['e1', 'e2', 'e3'], 'x\nGL FAILED -> v4g\n' + GOOD_LOG)):
        with tempfile.TemporaryDirectory() as t:
            ns, logs, envs, final, err = run_cells(t, dict(E006_BASE_ON_FAIL='fail'), [res])
        assert isinstance(err, RuntimeError) and 'E006_BASE stage failed' in str(err), err
        assert final is None and len(envs) == 1                                  # no table, no second run
        assert any('E006IMP SUBMISSION = NONE' in l for l in logs) and ns['E6I']['status'].startswith('failed')
        assert 'cfg_after_stage' not in ns                                        # the notebook stopped in the stage cell
    cfg = load_cfg()
    with tempfile.TemporaryDirectory() as t:                                      # 'fail' gives the first run the whole budget
        ns, logs, envs, final, err = run_cells(t, dict(E006_BASE_ON_FAIL='fail'), [(0, 'ok', ['e1', 'e2', 'e3'], None)])
    assert err is None and final == ['e1', 'e2', 'e3'] and cfg['E006_BASE_DEADLINE_S'] < envs[0]['_timeout'] <= cfg['E006_HARD_DEADLINE_S']
    assert any("E006IMP SUBMISSION = e006 improved" in l for l in logs) and not ns['FAILS']
    assert (ns['cfg_after_stage']['USE_ICE'], ns['cfg_after_stage']['USE_GL']) == (False, False)


def test_policy_e006():
    cfg = load_cfg()
    with tempfile.TemporaryDirectory() as t:                                      # first run fails, the untouched script succeeds
        ns, logs, envs, final, err = run_cells(t, dict(E006_BASE_ON_FAIL='e006', E006_PROM_FINAL=True),
                                               [(1, 'failed', None, None), (0, 'ok', ['f1;f2', 'f3', 'f4'], 'no hand-over line\n')])
    assert err is None and final == ['f1;f2', 'f3', 'f4'] and len(envs) == 2
    assert 'OURS_E006_BASE' in envs[0] and envs[0].get('OURS_E006_PROM_FINAL') == '1' and 'OURS_E006_PC' not in envs[0]
    assert not any(k in envs[1] for k in ('OURS_E006_BASE', 'OURS_E006_PROM_FINAL', 'OURS_E006_PROM_REL', 'OURS_E006_GL_ALL'))   # the public notebook untouched
    assert envs[1].get('OURS_E006_PC', '').endswith('pc_lists_first_run.json') and 'OURS_E006_ENG' in envs[1]
    assert envs[0]['_timeout'] <= cfg['E006_HARD_DEADLINE_S'] - cfg['E006_FALLBACK_S'] and envs[1]['_timeout'] <= cfg['E006_HARD_DEADLINE_S']
    assert ns['E6I']['status'] == 'ok' and ns['E6I']['fallback'] == 'e006 own table' and len(ns['FAILS']) == 1
    assert any("E006IMP SUBMISSION = E006'S OWN UNTOUCHED TABLE" in l for l in logs) and any('E006IMP attempt fallback' in l for l in logs)
    assert (ns['cfg_after_stage']['USE_ICE'], ns['cfg_after_stage']['USE_GL']) == (False, False)        # cells 07 / 08 only write a stand-in
    with tempfile.TemporaryDirectory() as t:                                      # the same with STRICT: the table is written, then the run fails loudly
        ns, logs, envs, final, err = run_cells(t, dict(E006_BASE_ON_FAIL='e006', STRICT=True), [(1, 'failed', None, None), (0, 'ok', ['f1', 'f3', 'f4'], '')])
    assert isinstance(err, RuntimeError) and str(err).startswith('STRICT: ') and final == ['f1', 'f3', 'f4']
    with tempfile.TemporaryDirectory() as t:                                      # both runs fail -> nothing is submitted
        ns, logs, envs, final, err = run_cells(t, dict(E006_BASE_ON_FAIL='e006'), [(1, 'failed', None, None), (None, 'timeout', None, None)])
    assert isinstance(err, RuntimeError) and final is None and len(envs) == 2 and ns['E6I']['status'].startswith('failed twice')
    with tempfile.TemporaryDirectory() as t:                                      # a healthy first run never starts the second
        ns, logs, envs, final, err = run_cells(t, dict(E006_BASE_ON_FAIL='e006'), [(0, 'ok', ['e1', 'e2', 'e3'], None)])
    assert err is None and final == ['e1', 'e2', 'e3'] and len(envs) == 1 and 'fallback' not in ns['E6I']


def test_policy_ours_and_final():
    with tempfile.TemporaryDirectory() as t:
        ns, logs, envs, final, err = run_cells(t, dict(E006_BASE_ON_FAIL='ours'), [(None, 'timeout', None, None)])
    assert err is None and final == ['o1;o2', 'o3', 'o4'] and len(ns['FAILS']) == 1
    assert any('E006IMP SUBMISSION = OUR OWN TABLE' in l for l in logs)
    assert ns['cfg_after_stage']['USE_ICE'] is True and ns['cfg_after_stage']['USE_GL'] is True          # our own forward stage runs
    with tempfile.TemporaryDirectory() as t:                                      # STRICT: the same table is written, then the run fails
        ns, logs, envs, final, err = run_cells(t, dict(E006_BASE_ON_FAIL='ours', STRICT=True), [(None, 'timeout', None, None)])
    assert isinstance(err, RuntimeError) and str(err).startswith('STRICT: ') and final == ['o1;o2', 'o3', 'o4']
    for pol in ('fail', 'e006', 'ours'):                                          # healthy stage, broken final cell: never the stand-in
        with tempfile.TemporaryDirectory() as t:
            ns, logs, envs, final, err = run_cells(t, dict(E006_BASE_ON_FAIL=pol), [(0, 'ok', ['e1', 'e2', 'e3'], None)], break_final=True)
        assert isinstance(err, RuntimeError) and 'no stand-in is submitted' in str(err) and final is None, (pol, err, final)
        assert any('E006IMP SUBMISSION = NONE' in l for l in logs)


def test_mode_e006_and_validation():
    with tempfile.TemporaryDirectory() as t:                                      # commit run: the script's own smoke, no variable set
        ns, logs, envs, final, err = run_cells(t, dict(E006_BASE='e006'), [(0, 'ok', ['e1', 'e2', 'e3'], 'plain log\n')], is_rerun=False)
    assert err is None and final == ['e1', 'e2', 'e3'] and [k for k in envs[0] if k != '_timeout'] == [], envs
    assert ns['E6I']['mode'] == 'e006' and any("E006IMP SUBMISSION = e006's own table" in l for l in logs)
    with tempfile.TemporaryDirectory() as t:
        ns, logs, envs, final, err = run_cells(t, dict(E006_BASE='e006'), [(1, 'failed', None, None)])
    assert isinstance(err, RuntimeError) and final is None and len(envs) == 1
    with tempfile.TemporaryDirectory() as t:                                      # 'ours' needs a first pipeline: refused in mode 'e006'
        err = run_cells(t, dict(E006_BASE='e006', E006_BASE_ON_FAIL='ours'), [])[4]
    assert isinstance(err, AssertionError)
    lab = {'m1': 'CCN', 'm2': 'CCC', 'm3': 'c1ccccc1'}
    with tempfile.TemporaryDirectory() as t:                                      # validation rehearsal: validation folder, every molecule, tables scored
        ns, logs, envs, final, err = run_cells(t, dict(VALIDATION=True, VAL_AS_RERUN=True, E006_BASE_ON_FAIL='ours'),
                                               [(0, 'ok', ['NCC;CCO', 'CCO;CC(C)C;CCC', 'CCCC'], None)], labels=lab, is_rerun=False)
        rr = json.load(open(os.path.join(t, 'work', 'stage_cache', 'val_rr.json')))
    assert err is None and envs[0]['OURS_E006_COMP'].endswith('comp') and envs[0]['OURS_E006_FULL'] == '1'
    assert rr['rr']['FINAL_submission'] == {'m1': 1.0, 'm2': 1 / 3, 'm3': 0.0} and rr['rr']['stage_oursbase'] == rr['rr']['FINAL_submission']
    assert set(rr['rr']) == {'cell08_standin', 'stage_oursbase', 'FINAL_submission'} and rr['lib_max'] == {'m1': 0.3, 'm2': 0.95, 'm3': 0.5}
    assert sum('VALIDATION (metric key)' in l for l in logs) == 3
    with tempfile.TemporaryDirectory() as t:                                      # a validation run without VAL_AS_RERUN is refused
        err = run_cells(t, dict(VALIDATION=True), [], is_rerun=False)[4]
    assert isinstance(err, AssertionError)


def test_build_mode_e006():
    with tempfile.TemporaryDirectory() as t:
        r = subprocess.run([sys.executable, os.path.join(SOL, 'tools', 'build_notebook.py'), '--user', 'u', '--slug', 's', '--config',
                            os.path.join(SOL, 'configs', 'e006-through-stage.json'), '--out', t], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        nb = json.load(open(os.path.join(t, 's.ipynb'), encoding='utf-8')); meta = json.load(open(os.path.join(t, 'kernel-metadata.json')))
    assert [c['id'] for c in nb['cells']] == ['embed', '00-config', '01-setup', '02-smoke', 'e006imp-stage', 'e006imp-final']
    ref = json.load(open(os.path.join(os.path.dirname(os.path.dirname(SOL)), 'submissions', 'e006r-0.443', 'kernel-metadata.json')))
    assert meta['dataset_sources'] == ref['dataset_sources'] and meta['docker_image'] == ref['docker_image'] and meta['is_private'] is True
    assert meta['enable_internet'] is False


def test_patches_8_9():
    cells = snapshot_cells(); work = '/kaggle/working/e006'
    s1 = ES.script_improved(cells, work)
    assert s1.count('# ours: patch 8') == 2 and s1.count('# ours: patch 9') == 2          # the two patched lines + the helper's comment / the COMP line
    ns = {}
    env = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith('OURS_E006_')}
    try:
        exec(compile(ES.PREAMBLE + ES.PREAMBLE_IMP, 'pre', 'exec'), ns)
        with tempfile.TemporaryDirectory() as t:
            out = os.path.join(t, 'out.json'); src = os.path.join(t, 'first.json'); open(src, 'w').write('{"m": 1}')
            cmd = [sys.executable, 'x', 'a', 'b', 'c', 'd', 'test', out, '4']
            os.environ['OURS_E006_PC'] = src
            r = ns['_ours_pc_channel'](cmd, capture_output=True, text=True)
            assert r.returncode == 0 and 'reused' in r.stdout and open(out).read() == '{"m": 1}'
            os.environ.pop('OURS_E006_PC'); os.remove(out)
            calls = []
            ns['subprocess'] = types.SimpleNamespace(run=lambda c, **k: calls.append(c) or 'RAN')
            assert ns['_ours_pc_channel'](cmd, timeout=1) == 'RAN' and calls == [cmd]     # no variable: the notebook's own call
        # patch 8: the COMP line and the pre-flight assertion
        g = dict(os=os, find=lambda p: '/kaggle/input/competitions/x/test.parquet')
        exec(ES.COMP_LINE_NEW, g); assert g['COMP'] == '/kaggle/input/competitions/x'
        os.environ['OURS_E006_COMP'] = '/kaggle/working/val_comp'
        exec(ES.COMP_LINE_NEW, g); assert g['COMP'] == '/kaggle/working/val_comp'
        slug = 'enveda-CASMI26-molecule-id-mass-spectra'
        exec(ES.PRE_ASSERT_NEW.strip(), dict(os=os, slug=slug, comp='/kaggle/working/val_comp'))
        os.environ.pop('OURS_E006_COMP')
        exec(ES.PRE_ASSERT_NEW.strip(), dict(os=os, slug=slug, comp='/kaggle/input/competitions/' + slug))
        try:
            exec(ES.PRE_ASSERT_NEW.strip(), dict(os=os, slug=slug, comp='/kaggle/working/val_comp')); bad = False
        except AssertionError:
            bad = True
        assert bad                                                                  # without the variable the notebook's own check holds
    finally:
        os.environ.update(env)


def test_run_script_monitor():
    with tempfile.TemporaryDirectory() as t:
        sc = os.path.join(t, 's.py'); lp = os.path.join(t, 'e.log')
        open(sc, 'w').write("import time\nprint('pubchem lists 3 1s', flush=True)\ntime.sleep(0.6)\nprint('ICE meta {}', flush=True)\ntime.sleep(0.6)\n"
                            "print('E006_STAGE base lists used: 3 of 3', flush=True)\n")
        real = ES.Monitor.__init__

        def fast(self, pid, work, log_path, every=0.1, res_every=0.2):
            real(self, pid, work, log_path, every, res_every)
        ES.Monitor.__init__ = fast
        try:
            rc, secs, st = ES.run_script(sc, t, dict(os.environ), 60, lp)
            timed = open(lp + '.timed').read(); res = [json.loads(l) for l in open(lp + '.res.jsonl')]
            assert (rc, st) == (0, 'ok') and open(lp).read().count('\n') == 3 and timed.count('[+') == 3
            ph = ES.phase_times(timed)
            assert list(ph) == ['PubChem channel', 'ICEBERG', 'end'] and ph['PubChem channel'] <= ph['ICEBERG'] <= ph['end']
            assert res and all('t' in r for r in res) and ES.LAST_MONITOR['samples'] >= 2
            if 'kernel_rss_gb' in res[0]:                                          # psutil present: the resource columns are filled
                assert ES.LAST_MONITOR['peak_kernel_rss_gb'] > 0 and ES.LAST_MONITOR['min_disk_free_gb'] > 0 and ES.LAST_MONITOR['min_ram_avail_gb'] > 0
            open(sc, 'w').write("import sys\nprint('boom'); sys.exit(3)\n")
            assert ES.run_script(sc, t, dict(os.environ), 60, lp)[::2] == (3, 'failed')
            open(sc, 'w').write("import time\nprint('start', flush=True)\ntime.sleep(60)\n")
            t0 = time.time(); rc, secs, st = ES.run_script(sc, t, dict(os.environ), 1.5, lp)
            assert (rc, st) == (None, 'timeout') and time.time() - t0 < 30 and 'start' in open(lp + '.timed').read()
            rc, secs, st = ES.run_script(sc, t, dict(os.environ), 1.0, lp, monitor=False)
            assert st == 'timeout' and not os.path.exists(lp + '.timed') and ES.LAST_MONITOR == {}
        finally:
            ES.Monitor.__init__ = real
    assert ES.phase_times('[+    12s] engine lists 12 0s\n[+   100s] pubchem lists 12 99s\n[+   200s] errors 0 | popularity re-ranked lists 12\n'
                          '[+   300s] GL FAILED -> x\n') == {'engine-2 lists': 12, 'PubChem channel': 100, 'base lists': 200, 'GLACIER': 300}


def test_mrr_table():
    key = lambda s: s.lower()
    st, rr = ES.mrr_table({'a': ['X', 'y', 'z'], 'b': ['q'], 'c': ['w'] * 30}, {'a': 'Y', 'b': 'Q', 'c': 'n', 'd': 'k'}, key)
    assert rr == {'a': 0.5, 'b': 1.0, 'c': 0.0, 'd': 0.0} and st == dict(mrr=0.375, top1=0.25, hit=0.5, n=4)
    st, rr = ES.mrr_table({'a': ['r%d' % i for i in range(25)] + ['Y']}, {'a': 'y'}, key)
    assert rr == {'a': 0.0}                                                        # rank 26 does not count


def test_validation_mix():
    import numpy as np, pandas as pd
    src = open(os.path.join(SOL, 'nb', '02_smoke.py'), encoding='utf-8').read()
    a = src.index('def build_validation_mix('); b = src.index('def build_validation(')
    rows = []
    for k in range(8):                                   # structures K0..K7 of library L, 5 spectra each; K0 also has 2 rows in library M
        for j in range(5):
            rows.append(('K%d' % k, 'L', '[M+H]+' if j else '[M+K]+', 5000.0))
    rows += [('K0', 'M', '[M+H]+', 5000.0)] * 2 + [('T0', 'M', '[M+H]+', 5000.0)] * 3 + [('Z9', 'M', '[M+H]+', 5000.0)] * 4
    rows += [('K%d' % k, 'L', '[M+H]+', 10.0) for k in range(8)]                    # weak spectra: never test spectra
    df = pd.DataFrame(rows, columns=['inchikey14', 'ingest_lib', 'adduct', 'base_peak_intensity'])
    for c in ('ms2_mzs', 'ms2_normalized_intensities'):
        df[c] = [[1.0, 2.0]] * len(df)
    for c, v in (('ionization_mode', 'positive'), ('instrument_type', 'timsTOF'), ('precursor_mz', 100.0), ('collision_energy_ev', 20.0), ('collision_energy_orig', '20'),
                 ('collision_energy_orig_units', 'eV')):
        df[c] = v
    df['normalized_smiles'] = ['S_' + k for k in df.inchikey14]
    with tempfile.TemporaryDirectory() as t:
        comp = os.path.join(t, 'comp'); pool = os.path.join(t, 'pool'); out = os.path.join(t, 'val'); os.makedirs(comp); os.makedirs(pool)
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), os.path.join(comp, 'train.parquet'), row_group_size=7)
        keys = ['K%d' % k for k in range(8)] + ['T0', 'Z9']
        pd.DataFrame(dict(inchikey14=keys, key=['MK%d' % k for k in range(8)] + ['MK0', 'MZ'])).to_parquet(os.path.join(pool, 'train_structs.parquet'))   # T0 = tautomer twin of K0
        ns = dict(os=os, pd=pd, hashlib=hashlib, POOL_DIR=pool, STAGE=os.path.join(t, 'stage'),
                  TEST_ADDUCTS=['[M+H]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+', '[M+Na]+', '[M+K]+', '[M-H]-', '[M-H2O-H]-', '[M+CH2O2-H]-', '[M+Cl]-'])
        exec(compile(src[a:b], 'mix', 'exec'), ns)
        n, q = ns['build_validation_mix'](comp, out, 'L', 6, 0, 3)
        te = pd.read_parquet(os.path.join(out, 'test.parquet')); tr = pd.read_parquet(os.path.join(out, 'train.parquet'))
        lab = pd.read_csv(os.path.join(ns['STAGE'], 'val_labels.csv')); samp = pd.read_csv(os.path.join(out, 'sample_submission.csv'))
    assert (n, q) == (6, 18) and len(te) == 18 and set(samp.molecule_id) == set(lab.molecule_id) == set(te.molecule_id)
    assert sorted(lab.val_class.value_counts().to_dict().items()) == [('all', 3), ('lib', 3)]
    assert (te.base_peak_intensity >= 1000).all() and te.groupby('molecule_id').size().eq(3).all()
    left = tr.groupby('inchikey14').size().to_dict(); held = dict(zip(lab.inchikey14, lab.val_class))
    for k, c in held.items():
        if c == 'all':
            assert k not in left                          # every row of the structure is gone, in every library
        else:
            assert left[k] == 6 - 3 + (2 if k == 'K0' else 0)   # only its three test spectra are gone (5 strong + 1 weak, K0: + 2 in library M)
    assert ('T0' in left) == (held.get('K0') != 'all')    # the tautomer twin is purged with a class-'all' structure only
    assert left['Z9'] == 4 and all(left[k] == 6 + (2 if k == 'K0' else 0) for k in ('K%d' % i for i in range(8)) if k not in held)
    assert list(lab.smiles) == ['S_' + k for k in lab.inchikey14]


if __name__ == '__main__':
    test_defaults(); test_policy_fail(); test_policy_e006(); test_policy_ours_and_final(); test_mode_e006_and_validation(); test_build_mode_e006()
    test_patches_8_9(); test_run_script_monitor(); test_mrr_table(); test_validation_mix()
    print("E006_BASE fallbacks OK: policy 'fail' / 'e006' / 'ours', no stand-in ever submitted, STRICT, mode 'e006', validation rehearsal, build, patches 8 / 9, "
          'run_script monitor, mrr_table, mixed validation set')
