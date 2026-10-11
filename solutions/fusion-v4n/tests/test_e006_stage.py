"""E006_FUSE (second pipeline + fusion of the two submission tables): off by default and off == the reference; the rules; the script patches.

Checks, without competition data:
  * the flag is off by default; with it off, cell 09 touches nothing but `E006_STATS` and `sub.head()`, the built notebook carries no cell of
    the public notebook, and cells 01 to 08 of a build with the flag are the cells of a build without it (so the first pipeline is v59a's);
  * a molecule without a second list (stage failed, placeholder row) keeps our row byte for byte, duplicates and all;
  * 'rrf' equals a direct reimplementation of 1/(k + r) + 1/(k + r) on the key, 'fill' keeps our first p-1 rows and the order of all our rows
    and inserts exactly their unseen rows in their order, 'agree' leads with the rows both lists hold in their first five; every fused list has
    at most 25 rows and one row per key;
  * the script made from the snapshot's cells parses, differs from the cells only by the three patches, and leaves no path outside its folder;
  * the fusion key is the snapshot's copy of the official metric function, in this process and through the worker process;
  * engine-2 lists are shared only when our engine-2 knobs are the reference's;
  * the presets v65a / v65b / v65c are v59a plus the option (v65c: the second list at weight 0.3, a post hoc variant).
usage: python tests/test_e006_stage.py   (or pytest)
"""
import ast, json, os, random, subprocess, sys, tempfile

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed')); sys.path.insert(0, os.path.join(SOL, 'tools'))
import e006_stage as ES
import build_notebook as BN

KEY = lambda s: None if s.startswith('bad') else s.split('~')[0]      # 'k3~b' and 'k3' share the key 'k3'; 'bad..' has none


def load_cfg():
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    return ns['CFG']


def rand_lists(rng):
    pool = ['k%d' % i for i in range(60)]
    def one():
        n = rng.randint(0, 32); out = []
        for q in rng.sample(pool, min(n, len(pool))):
            r = rng.random()
            out.append(q + '~x' if r < 0.15 else ('bad%d' % rng.randint(0, 9) if r < 0.2 else q))
            if rng.random() < 0.08:
                out.append(q + '~dup')
        return out
    return one(), one()


def test_off_by_default_and_cell_is_inert():
    cfg = load_cfg()
    assert cfg['E006_FUSE'] is None and cfg['E006_RRF_W'] == 1.0 and cfg['E006_SHARE_ENG'] is True and cfg['E006_DEADLINE_S'] <= 7.75 * 3600        # the fusion after it must end before 8 h

    class Sub:
        calls = 0
        def head(self):
            Sub.calls += 1
            return 'head'
    ns = dict(CFG=dict(cfg), sub=Sub())
    exec(compile(open(os.path.join(SOL, 'nb', '09_e006.py'), encoding='utf-8').read(), '09_e006.py', 'exec'), ns)
    assert set(ns) - {'__builtins__'} == {'CFG', 'sub', 'E006_STATS'} and ns['E006_STATS'] is None and Sub.calls == 1


def build(cfg_name, out):
    r = subprocess.run([sys.executable, os.path.join(SOL, 'tools', 'build_notebook.py'), '--user', 'u', '--slug', 's', '--config',
                        os.path.join(SOL, 'configs', cfg_name), '--out', out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    nb = json.load(open(os.path.join(out, 's.ipynb'), encoding='utf-8')); meta = json.load(open(os.path.join(out, 'kernel-metadata.json')))
    cells = {c['id']: ''.join(c['source']) for c in nb['cells']}
    ns = {}
    exec(cells['embed'], ns)
    return cells, ns['EMBED'], meta


def test_build_without_the_flag_carries_nothing_and_first_pipeline_is_unchanged():
    with tempfile.TemporaryDirectory() as t:
        c0, e0, m0 = build('v59a-v33b-ccby.json', os.path.join(t, 'a'))
        c1, e1, m1 = build('v65a-v59a-e006-rrf.json', os.path.join(t, 'b'))
    assert not any(k.startswith('e006/') for k in e0) and set(e1) - set(e0) == {'e006/cells.json', 'e006/NOTICE.md'}
    assert all(e1[k] == e0[k] for k in e0)
    same = [k for k in c0 if k not in ('embed', '00-config')]
    assert len(same) == 9 and all(c0[k] == c1[k] for k in same)
    assert c0['00-config'].split('\nCFG.update(')[0] == c1['00-config'].split('\nCFG.update(')[0]
    assert m0['dataset_sources'] == m1['dataset_sources'] and m0['docker_image'] == m1['docker_image'] and m1['is_private'] is True
    cells = json.loads(e1['e006/cells.json'])
    src = json.load(open(BN.E006_NB, encoding='utf-8'))
    assert cells == [''.join(c['source']) for c in src['cells'] if c['cell_type'] == 'code'] and len(cells) == 30
    assert 'Apache' in e1['e006/NOTICE.md'] and 'canqiang' in e1['e006/NOTICE.md']


def test_no_second_list_keeps_ours_byte_for_byte():
    rng = random.Random(1)
    for _ in range(200):
        o, e = rand_lists(rng)
        ours = dict(m1=o, m2=['CCO'], m3=o)
        for rule in ES.RULES:
            out, st = ES.fuse_tables(ours, {}, KEY, rule)
            assert out == ours and st['fused'] == 0 and st['theirs_missing'] == 3 and st['changed'] == 0
            out, st = ES.fuse_tables(ours, dict(m1=['CCO'], m2=[], m3=['bad1', 'bad2']), KEY, rule)
            assert out == ours and st['fused'] == 0, (rule, out, ours)


def ref_rrf(o, e, k, w=1.0):
    do, de = ES.dedup(o, KEY)[:25], ES.dedup(e, KEY)[:25]
    sc, rep, ro, re_ = {}, {}, {}, {}
    for r, (q, s) in enumerate(do, 1):
        sc[q] = 1.0 / (k + r); rep[q] = s; ro[q] = r
    for r, (q, s) in enumerate(de, 1):
        sc[q] = sc.get(q, 0.0) + w / (k + r); rep.setdefault(q, s); re_[q] = r
    order = sorted(sc, key=lambda q: (-sc[q], ro.get(q, 10 ** 6), re_.get(q, 10 ** 6)))
    return [rep[q] for q in order[:25]]


def test_rules():
    rng = random.Random(7); n_new = 0
    for _ in range(600):
        o, e = rand_lists(rng)
        do = [s for _, s in ES.dedup(o, KEY)[:25]]; de = [s for _, s in ES.dedup(e, KEY)[:25]]
        ko = [KEY(s) for s in do]; ke = [KEY(s) for s in de]
        for rule, kw in (('rrf', dict(rrf_k=1.0)), ('rrf', dict(rrf_k=3.0)), ('rrf', dict(rrf_k=10.0)), ('fill', dict(fill_pos=3)), ('fill', dict(fill_pos=6)),
                         ('fill', dict(fill_pos=11)), ('agree', {})):
            f = ES.fuse_lists(o, e, KEY, rule, **kw); kf = [KEY(s) for s in f]
            assert len(f) <= 25 and None not in kf and len(set(kf)) == len(kf)
            assert set(kf) <= set(ko) | set(ke) and len(f) == min(25, len(set(ko) | set(ke)))
            assert all(s == (do[ko.index(q)] if q in ko else de[ke.index(q)]) for q, s in zip(kf, f))       # representative: ours first
            if rule == 'rrf':
                assert f == ref_rrf(o, e, kw['rrf_k']) == ES.fuse_lists(o, e, KEY, rule, rrf_w=1.0, **kw)      # weight 1 is the default
                assert ES.fuse_lists(o, e, KEY, rule, rrf_w=0.3, **kw) == ref_rrf(o, e, kw['rrf_k'], 0.3)
                assert ES.fuse_lists(o, e, KEY, rule, rrf_w=0.0, **kw)[:len(do)] == do                          # weight 0: ours first, unchanged
            if rule == 'fill':
                p = kw['fill_pos']; new = [q for q in ke if q not in ko]
                assert kf[:min(p - 1, len(ko))] == ko[:p - 1]                                           # our first p-1 rows stay
                assert [q for q in kf if q in ko] == ko[:len([q for q in kf if q in ko])]                   # our rows keep their order
                assert [q for q in kf if q not in ko] == new[:len([q for q in kf if q not in ko])]          # theirs in their order
                if len(ko) >= p - 1:
                    assert kf[p - 1:p - 1 + len(new)] == new[:max(0, 25 - (p - 1))][:len(kf[p - 1:p - 1 + len(new)])]
                n_new += len(new) > 0
            if rule == 'agree':
                both = [q for q in ko[:5] if q in ke[:5]]
                assert set(kf[:len(both)]) == set(both)
                rest = [q for q in ref_rrf(o, e, 3.0) if KEY(q) not in both]
                assert f[len(both):] == rest[:25 - len(both)]
                sc = [1.0 / (ko.index(q) + 1) + 1.0 / (ke.index(q) + 1) for q in kf[:len(both)]]
                assert sc == sorted(sc, reverse=True)
    assert n_new > 300
    assert ES.fuse_lists(['A', 'B', 'C'], ['B', 'X'], lambda s: s, 'fill', fill_pos=2) == ['A', 'X', 'B', 'C']
    assert ES.fuse_lists(['A', 'B', 'C'], ['B', 'X'], lambda s: s, 'rrf', rrf_k=3.0) == ['B', 'A', 'X', 'C']
    assert ES.fuse_lists(['A', 'B', 'C'], ['C', 'A'], lambda s: s, 'agree') == ['A', 'C', 'B']


def test_tables():
    ours = dict(a=['k1', 'k2', 'k2~dup', 'k3'], b=['CCO'], c=['k1'])
    theirs = dict(a=['k9', 'k2~x'], b=['k5', 'k6'], c=['CCO'])
    out, st = ES.fuse_tables(ours, theirs, KEY, 'fill', fill_pos=2)
    assert out == dict(a=['k1', 'k9', 'k2', 'k3'], b=['k5', 'k6'], c=['k1'])
    assert st['fused'] == 2 and st['theirs_missing'] == 1 and st['new_rows'] == 3 and st['changed_top1'] == 1


def test_script_is_the_notebook_plus_three_patches():
    cells = json.loads(BN.e006_embed()['e006/cells.json']); work = '/kaggle/working/e006'
    s = ES.script_from_cells(cells, work)
    ast.parse(s)
    assert s.startswith(ES.PREAMBLE) and s.count('_ours_engine2(') == 2 and s.count("os.getenv('OURS_E006_FULL')") == 1
    body = s[len(ES.PREAMBLE):]
    assert body.count('/kaggle/working') == body.count(work) >= 15                      # no path outside the stage's folder
    back = body.replace(work, ES.WORK_LITERAL).replace(ES.ENG_CALL_NEW, ES.ENG_CALL).replace(ES.RERUN_LINE_NEW, ES.RERUN_LINE)
    assert back == '\n\n'.join('# ---- cell %d\n%s' % (i, c.rstrip('\n')) for i, c in enumerate(cells)) + '\n'
    for bad in (cells[:10] + cells[11:], [c.replace('KAGGLE_IS_COMPETITION_RERUN', 'X') for c in cells]):   # a changed source must not pass silently
        try:
            ES.script_from_cells(bad, work)
        except AssertionError:
            continue
        raise AssertionError('patch applied to a changed source')


def test_metric_key_is_the_notebooks_copy_of_the_official_function():
    """e006_stage.metric_key == canonicalize_smiles_to_ik14 of the snapshot (the metric's own code), also through the worker process."""
    try:
        import rdkit  # noqa: F401
    except ImportError:
        print('rdkit not installed: metric-key check skipped'); return
    cells = json.loads(BN.e006_embed()['e006/cells.json'])
    ns = {}
    exec([c for c in cells if 'def canonicalize_smiles_to_ik14' in c][0], ns)
    smis = ['CCO', 'C[C@H](N)C(=O)O', 'C[C@@H](N)C(=O)O', 'CC(N)C(=O)O', 'OC1=CC=CC=N1', 'O=C1C=CC=CN1', 'C/C=C/C(=O)O', 'C/C=C' + chr(92) + 'C(=O)O', 'not a smiles', '',
            'CC(=O)CC(=O)C', 'CC(O)=CC(=O)C', 'O=c1[nH]cnc2[nH]cnc12', 'Oc1ncnc2nc[nH]c12', 'C[C@H]1CC[C@@H](C(C)C)C(=O)C1', 'c1ccccc1', 'C[N+](C)(C)CC([O-])=O']
    ref = [ns['canonicalize_smiles_to_ik14'](x) or None for x in smis]      # '' (the empty molecule's key) counts as no key, as in the notebook's metric_identity
    assert [ES.metric_key(x) for x in smis] == ref and ES.metric_key(None) is None
    assert ref[1] == ref[2] == ref[3] and ref[4] == ref[5] and ref[6] == ref[7] and ref[8] is None                 # stereo and tautomers collapse
    with tempfile.TemporaryDirectory() as t:
        K = ES.metric_keys(smis * 3, workdir=t, workers=2, min_rows=1)
        assert os.path.exists(os.path.join(t, 'keys_out.json'))                                                     # the worker process ran
    assert K == dict(zip(smis, ref)) == ES.metric_keys(smis)


def test_engine_sharing_gate():
    cfg = load_cfg()
    assert ES.engine_shareable(cfg)
    for k, v in (('ENG_REPS', 2), ('ENG_SEEDS', [0, 1, 2, 3]), ('ENG_N_ANALOG', 250), ('ENG_EXTRA_FPNET', 'np-ours'), ('USE_ENG', False), ('ENG_W_PV', 0.8)):
        assert not ES.engine_shareable(dict(cfg, **{k: v})), k


def test_presets():
    j = lambda n: json.load(open(os.path.join(SOL, 'configs', n), encoding='utf-8'))
    v59 = j('v59a-v33b-ccby.json'); cfg = load_cfg()
    for name, extra in (('v65a-v59a-e006-rrf.json', dict(E006_FUSE='rrf', E006_RRF_K=3.0)), ('v65b-v59a-e006-fill.json', dict(E006_FUSE='fill', E006_FILL_POS=6)),
                        ('v65c-v59a-e006-rrf-w03.json', dict(E006_FUSE='rrf', E006_RRF_K=3.0, E006_RRF_W=0.3))):
        p = j(name)
        assert {k: v for k, v in p.items() if k not in extra and k != 'VERSION'} == {k: v for k, v in v59.items() if k != 'VERSION'}
        assert all(p[k] == v for k, v in extra.items()) and set(p) <= set(cfg) and p['E006_FUSE'] in ES.RULES


if __name__ == '__main__':
    for f in (test_off_by_default_and_cell_is_inert, test_build_without_the_flag_carries_nothing_and_first_pipeline_is_unchanged,
              test_no_second_list_keeps_ours_byte_for_byte, test_rules, test_tables, test_script_is_the_notebook_plus_three_patches,
              test_metric_key_is_the_notebooks_copy_of_the_official_function, test_engine_sharing_gate, test_presets):
        f()
    print('E006_FUSE OK: off by default and inert, first pipeline unchanged, rules, script patches, metric key, engine sharing gate, presets')
