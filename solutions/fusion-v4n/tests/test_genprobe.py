"""Tests of the genprobe cells (nb/embed/genprobe.py: capture of engine 1's raw list, frozen generated-candidate rule, G-current / G-first /
G-floor3 transforms, placeholders); no competition data. Run with pytest, or by hand: python solutions/fusion-v4n/tests/test_genprobe.py"""
import hashlib, os, sys, tempfile, types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed'))
import genprobe as G  # noqa: E402

KEY = lambda s: s.lower() if isinstance(s, str) and s.strip() else None


def _row(s, rr, gen, sim=0.9):
    return dict(smiles=s, key=s, raw=1.0 / rr, pid=-1 if gen else rr, gen=gen, raw_rank=rr, sim=sim if gen else 0.0, formula='F')


def _hac(smiles):
    k = G.gp_metric_identity(smiles)
    return k[1] if k else None


def test_selftest_passes_and_switch_default_off():
    assert G.gp_selftest() == 'ok'
    assert G.GENPROBE == dict(variant='off')
    assert (G.GP_LIB_MAX, G.GP_SIM_MIN, G.GP_TOPK, G.GP_FLOOR) == (0.5, 0.6, 3, 3)   # PREREG constants


def test_rule_selection():
    cap = dict(lib_max=0.49, rows=[_row('A', 1, False), _row('G', 2, True, 0.6), _row('H', 3, True, 0.99), _row('B', 4, False)])
    s, why = G.gp_select(cap, ['A', 'B', 'C', 'D'], KEY)
    assert why == 'frozen' and s['smiles'] == 'G' and s['final_rank'] is None and s['raw_rank'] == 2   # best raw rank; absent counts
    assert G.gp_select(dict(cap, lib_max=0.5), ['A'], KEY)[1] == 'lib'                                 # strict < 0.5
    low = dict(cap, rows=[_row('G', 1, True, 0.5999), _row('H', 2, True, 0.99)])
    assert G.gp_select(low, ['A'], KEY) == (None, 'sim')                                                 # no fall-through to H
    assert G.gp_select(cap, ['A', 'B', 'g'], KEY)[1] == 'in_top3'                                        # rank 3 by identity
    assert G.gp_select(cap, ['A', 'B', 'C', 'g'], KEY)[0]['final_rank'] == 4
    deep = dict(cap, rows=[_row('A', 1, False), _row('B', 2, False), _row('C', 3, False), _row('G', 4, True)])
    assert G.gp_select(deep, ['A'], KEY) == (None, 'no_gen_top3')
    assert G.gp_select(None, ['A'], KEY) == (None, 'no_capture')
    assert G.gp_select(dict(cap, rows=[_row('G', 1, True, None)]), ['A'], KEY) == (None, 'sim')


def test_rule_selection_metric_identity():
    """Stereo and tautomer variants of the candidate count as the same identity (final rank found through them)."""
    ident = G.gp_metric_identity
    cand = 'CC(=O)CC(=O)C'                                       # acetylacetone; enol form below has the same metric identity
    assert ident(cand) == ident('CC(O)=CC(=O)C')
    cap = dict(lib_max=0.1, rows=[_row(cand, 1, True)])
    toks = ['CCCCCCCCCCCCCCCC(=O)O', 'OCC(O)CO', 'CC(=O)OC', 'CC(O)=CC(=O)C']
    s, why = G.gp_select(cap, toks, ident)
    assert why == 'frozen' and s['final_rank'] == 4


def test_transforms():
    ph = G.GP_PLACEHOLDERS
    T = [f't{i}' for i in range(1, 26)]; T[9] = 'G'
    cap = dict(lib_max=0.2, rows=[_row('G', 1, True)])
    s, _ = G.gp_select(cap, T, KEY)
    assert s['final_rank'] == 10
    cur = G.gp_current(T, s)
    assert len(cur) == 25 and cur[9] == 'G' and [x for i, x in enumerate(cur) if i != 9] == [x for i, x in enumerate(ph) if i != 9]
    assert G.gp_first(T, s) == ['G'] + ph[:24]
    fl = G.gp_floor3(T, s, KEY)
    assert fl[:3] == ['t1', 't2', 'G'] and fl[3:] == ['t3', 't4', 't5', 't6', 't7', 't8', 't9'] + T[10:] and len(fl) == 25
    absent = [f't{i}' for i in range(1, 26)]
    s2, _ = G.gp_select(cap, absent, KEY)
    assert G.gp_current(absent, s2) == ph and G.gp_first(absent, s2)[0] == 'G'
    fl2 = G.gp_floor3(absent, s2, KEY)
    assert fl2 == ['t1', 't2', 'G'] + [f't{i}' for i in range(3, 25)]                                     # rank 25 drops
    assert G.gp_floor3(absent[:10], s2, KEY) == ['t1', 't2', 'G'] + [f't{i}' for i in range(3, 11)]     # short list: nothing drops
    assert G.gp_current(T, None) == ph and G.gp_first(T, None) == ph and G.gp_floor3(T, None, KEY) == T
    for v in G.GP_VARIANTS:
        assert G.gp_transform(v, T, None, KEY) == (T if v in ('off', 'floor3') else ph)
    try:
        G.gp_transform('bogus', T, s, KEY)
        raise AssertionError('unknown variant accepted')
    except ValueError:
        pass


def test_placeholders_rdkit_valid_distinct_and_keep_positions():
    import rdkit
    assert rdkit.__version__ == '2026.03.3', rdkit.__version__
    ks = G.gp_check_placeholders(G.gp_metric_identity)
    assert len({k[0] for k in ks}) == 25
    # with a real candidate first, grader-style dedup (first occurrence of each InChIKey14) keeps every position
    lst = ['CCCCCCCCCC(=O)OCC(O)CO'] + G.GP_PLACEHOLDERS[:24]
    ik = [G.gp_metric_identity(s)[0] for s in lst]
    first = {}
    for i, k in enumerate(ik, 1):
        first.setdefault(k, i)
    assert sorted(first.values()) == list(range(1, 26))
    assert all(G.gp_metric_identity(s)[0] not in {k[0] for k in ks} for s in ['CCCCCCCCCC(=O)OCC(O)CO'])


def _cell20(scope):
    """Faithful replica of stack1 cell 20's loop body (raw order, dedup by key, top-N, popularity re-rank) for the capture tests."""
    BASE = {}
    for mid in scope['mids']:
        scope['mid'] = mid
        smis, keys, lib_max, scs, forms, pids_ = [], [], 0.0, [], [], []
        C, X, F, names, info = scope['V'].run([], 0.0)
        lib_max = float(info.get('lib_max', 0.0))
        if len(C.pid):
            score = scope['rank_score'](np.concatenate([X, F], 1), list(scope['FEATURES']) + list(names))
            order = np.argsort(-score, kind='stable'); seen = set()
            for i in order:
                k = C.key[i]
                if k in seen:
                    continue
                seen.add(k); smis.append(C.smiles[i]); keys.append(k); scs.append(float(score[i])); forms.append(C.formula[i]); pids_.append(int(C.pid[i]))
                if len(smis) >= scope['TOPN']:
                    break
        if scope['POOLPOP_MU'] > 0 and len(scs) > 1:
            s_ = np.asarray(scs, np.float64); z_ = (s_ - s_.mean()) / (s_.std() + 1e-9)
            pid_ = np.asarray(pids_); isg_ = pid_ < 0
            f_ = z_ + scope['POOLPOP_MU'] * np.where(isg_, 0.0, scope['POOL_POP'][np.maximum(pid_, 0)])
            pool_i = np.where(~isg_)[0]; o_ = np.empty(len(pid_), np.int64)
            o_[isg_] = np.where(isg_)[0]; o_[~isg_] = pool_i[np.argsort(-f_[pool_i], kind='stable')]
            smis = [smis[i] for i in o_]; keys = [keys[i] for i in o_]; scs = [float(f_[i]) for i in o_]
            forms = [forms[i] + ('|gen' if isg_[i] else '') for i in o_]
        BASE[mid] = (smis, keys, lib_max, scs, forms)
    scope['BASE'] = BASE


def _fake_scope(seed=0, n_mol=6):
    rng = np.random.default_rng(seed)
    mids = [f'm{i}' for i in range(n_mol)]
    cands = {}
    for j, m in enumerate(mids):
        n_pool, n_gen = 12, j % 3
        pid = np.asarray(list(rng.choice(np.arange(30), n_pool, replace=False)) + [-1] * n_gen)
        key = [f'k{p}' for p in pid[:n_pool]] + [f'g{j}_{i}' for i in range(n_gen)]
        key[5] = key[1]                                                       # duplicate key
        sim = np.asarray([0.0] * n_pool + list(rng.uniform(0.3, 0.9, n_gen)), np.float32)
        cands[m] = types.SimpleNamespace(pid=pid, key=key, smiles=[f'S_{k}_{i}' for i, k in enumerate(key)], formula=['C10H12O'] * len(pid), gen_sim=sim)
    if n_mol > 1:
        cands[mids[-1]] = types.SimpleNamespace(pid=np.zeros(0, np.int64), key=[], smiles=[], formula=[], gen_sim=np.zeros(0))
    scores = {m: rng.normal(0, 1, len(cands[m].pid)) for m in mids}
    sc = dict(mids=mids, TOPN=8, FEATURES=['f0', 'gen_parent_sim'], POOLPOP_MU=0.15, POOL_POP=rng.uniform(0, 4, 30), GENPROBE=dict(variant='off'))

    class V:
        def run(self, spectra, target):
            C = cands[sc['mid']]
            X = np.column_stack([np.zeros(len(C.pid)), C.gen_sim]) if len(C.pid) else np.zeros((0, 2))
            return C, X, np.zeros((len(C.pid), 1)), ['h'], dict(lib_max=0.3)
    sc['V'] = V()
    sc['rank_score'] = lambda X, names: scores[sc['mid']].copy()
    return sc


def test_capture_wrappers_transparent_and_rebuild_exact():
    ref = _fake_scope(); _cell20(ref)
    sc = _fake_scope()
    G.gp_install(sc); _cell20(sc); G.gp_check_capture(sc)
    assert repr(ref['BASE']) == repr(sc['BASE'])                              # cell 20's output is unchanged by the wrappers
    assert not sc['GP_CAPFAIL'] and set(sc['GP_CAP']) == set(sc['mids'])
    assert 'run' not in vars(sc['V'])                                         # wrappers removed after cell 20
    assert sc['GP_STAT']['sim_c_vs_x_maxdiff'] == 0.0 and sc['GP_STAT']['sim_source'] == {'C.gen_sim': 5}
    for m in sc['mids']:
        cap = sc['GP_CAP'][m]; b = sc['BASE'][m]
        assert [r['smiles'] for r in sorted(cap['rows'], key=lambda r: r['pos'])] == b[0]
        assert [r['raw_rank'] for r in cap['rows']] == list(range(1, len(cap['rows']) + 1))
        assert all(r['pos'] == r['raw_rank'] for r in cap['rows'] if r['gen'])  # generated rows keep their slots
    bad = _fake_scope(); G.gp_install(bad); _cell20(bad)
    bad['BASE']['m1'] = (bad['BASE']['m1'][0][::-1],) + tuple(bad['BASE']['m1'][1:])
    G.gp_check_capture(bad)
    assert set(bad['GP_CAPFAIL']) == {'m1'}


def _apply_scope(variant, tmp):
    sc = _fake_scope(1)
    G.gp_install(sc); _cell20(sc); G.gp_check_capture(sc)
    real = ['CCCCCCCCCC(=O)OCC(O)CO', 'CCCCCCCCCCCC(=O)OC(CO)CO', 'CCCCCCCCCCC(=O)OCC(O)C', 'CCCCCCCCCCCCC(=O)OCCO', 'CCCCCCCCCC(O)C(=O)OCC(C)O']
    # give m1 a generated rank-1 row with parent similarity 0.8 and a real SMILES, absent from its final list; m2 the same but listed at rank 5
    for m, rank in (('m1', None), ('m2', 5)):
        cap = sc['GP_CAP'][m]
        cap['rows'][0] = dict(cap['rows'][0], smiles='CCCCCCCCC(C)C(=O)OCC(O)CO', gen=True, pid=-1, sim=0.8, raw_rank=1)
        cap['lib_max'] = 0.2
    finals = {m: ';'.join(real) for m in sc['mids']}
    finals['m2'] = ';'.join(real[:4] + ['CCCCCCCCC(C)C(=O)OCC(O)CO'] + real[4:])
    rows = [(m, finals[m]) for m in sc['mids']] + [('m_inactive', 'CCO')]
    p = os.path.join(tmp, 'submission.csv')
    pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(p, index=False)
    sc.update(GENPROBE=dict(variant=variant), metric_identity=G.gp_metric_identity, heavy_atom_composition=_hac,
              te=pd.DataFrame(dict(molecule_id=sc['mids'])), ENG={m: dict(smiles=real[:2]) for m in sc['mids']})
    return sc, p, real


def test_apply_off_identity_and_variants():
    with tempfile.TemporaryDirectory() as tmp:
        sc, p, real = _apply_scope('off', tmp)
        raw0 = open(p, 'rb').read()
        st = G.gp_apply(sc, workdir=tmp)
        assert open(p, 'rb').read() == raw0 and st['submission_sha256'] == hashlib.sha256(raw0).hexdigest() == st['base_sha256']
        assert st['frozen'] == 2 and st['fallback'] == 0 and st['status'].startswith('off') and st['frozen_final_ranks'] == [0, 5]
        out = {}
        for v in ('current', 'first', 'floor3'):
            open(p, 'wb').write(raw0)
            sc['GENPROBE'] = dict(variant=v)
            st = G.gp_apply(sc, workdir=tmp)
            assert st['status'] == 'applied' and st['frozen'] == 2, st
            A = pd.read_csv(p, dtype=str, keep_default_na=False)
            out[v] = dict(zip(A.molecule_id, A.smiles.str.split(';')))
            assert out[v]['m_inactive'] == ['CCO']
        g = 'CCCCCCCCC(C)C(=O)OCC(O)CO'
        ph = G.GP_PLACEHOLDERS
        assert out['current']['m1'] == ph and out['current']['m0'] == ph
        assert out['current']['m2'] == ph[:4] + [g] + ph[5:]
        assert out['first']['m1'] == [g] + ph[:24] and out['first']['m2'] == [g] + ph[:24] and out['first']['m3'] == ph
        assert out['floor3']['m1'] == real[:2] + [g] + real[2:] and out['floor3']['m2'] == real[:2] + [g] + real[2:]
        assert out['floor3']['m0'] == real


def test_apply_on_raises_without_capture():
    with tempfile.TemporaryDirectory() as tmp:
        sc, p, _ = _apply_scope('first', tmp)
        raw0 = open(p, 'rb').read()
        sc.pop('GP_CAP')
        try:
            G.gp_apply(sc, workdir=tmp)
            raise AssertionError('no raise')
        except RuntimeError as e:
            assert 'capture missing' in str(e)
        assert open(p, 'rb').read() == raw0                                    # the file is restored before raising
        sc['GENPROBE'] = dict(variant='off')
        st = G.gp_apply(sc, workdir=tmp)                                        # off never raises
        assert st['status'].startswith('FAILED') and open(p, 'rb').read() == raw0


if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_') and callable(f):
            f(); print('ok', n)
