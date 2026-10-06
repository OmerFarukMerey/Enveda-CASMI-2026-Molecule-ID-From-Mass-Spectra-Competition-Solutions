"""The fold-safe second pass (RANKER 'fs' / 'fs_avg', nb/embed/fs_pass.py): pure helpers, and cell nb/06 executed on a stub engine.

No competition data and no public package: the cell's source is executed with stand-ins for the engine, the families, the rankers
and the PubChem lists. What is pinned down:
  * flag off == v24a bit for bit: with the v24a preset the cell's BASE has the hash that the cell of commit b34236e (before the
    fold-safe pass existed) produces on the same stub inputs (GOLDEN; `python tests/test_fs_pass.py --golden <old cell file>` prints it);
  * flag on, every molecule at or above the gate: BASE is identical to v24a's (scoring rerun) although the pass is configured;
  * flag on: molecules below the gate carry the second pass's list, the others the first pass's; the engine objects are restored;
  * commit runs make the pass on every molecule and still use it only below the gate; a failure raises there;
  * scoring rerun: a failing pass keeps the first-pass list of that molecule and appends to FAILS;
  * rank_average: half weight to each chain, keys of one list only, score profile non-increasing.
usage: python tests/test_fs_pass.py   (or pytest)
"""
import hashlib, json, os, sys, tempfile, time, types

SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed'))
import numpy as np
import pandas as pd
import fs_pass
import pc_join
import ranker_infer

GOLDEN = '28d6241e6352faff5e96d6f1341d7d6215431affcb456bda6c7271d0f939d7fa'      # sha256 of BASE from the cell of commit b34236e on the stub inputs
FEATURES = ['fz', 'lib_max', 'ap', 'el_sum', 'fr_int', 'sf_int', 'mass_ppm', 'fz_rank', 'lib_max_rank', 'ap_rank', 'el_rank', 'fr_rank', 'sf_rank',
            'mass_ppm_rank', 'other']
NAMES_F = ['fam__a', 'fam__b']
V24A = dict(POP_MU=0.25, FRAG_LAM=0.5, FRAG_MODE='gap', ICE_LAM_LIB=0.0, GL_LAM_LIB=0.0, PC_JOIN_N=100, PC_JOIN_MAX_LIB=0.7, ICE_LAM=0.75, GL_LAM=0.75)
LIB = {'m0': 0.95, 'm1': 0.2, 'm2': 0.0, 'm3': 0.65, 'm4': 0.8, 'm5': 0.4}      # library evidence of the six stub molecules


def load_cfg(**over):
    ns = {}
    exec(open(os.path.join(SOL, 'nb', '00_config.py'), encoding='utf-8').read(), ns)
    c = ns['CFG']; c.update(V24A); c.update(over); c['DUMP_STAGES'] = False; c['USE_PC'] = False
    return c


class StubV:
    """V1FE stand-in: candidates and features are a deterministic function of the molecule, of the injected rows and of the bank."""

    def __init__(self, E):
        self.E = E; self.banks = ['fe_A', 'fe_B', 'fe_D']; self.mods = {}; self.calls = []; self.fail_fs = set()

    def model_views(self, prep):
        z = np.zeros((len(self.banks), 1), np.float32)
        return z, z, z, z, np.zeros(1, np.float32)

    def run(self, spectra, target):
        E = self.E; mid = E.by_target[round(target, 4)]; fs = E.bank == 'FS'
        self.calls.append((mid, E.bank, tuple(self.banks), len(self.model_views(None)[0])))
        if fs and mid in self.fail_fs:
            raise RuntimeError('stub failure in the fold-safe pass')
        rs = np.random.RandomState(abs(hash_int(mid)) % (2 ** 31) + (7 if fs else 0))
        n_pool = 8 + int(target) % 5; inj = list(E._pc_inject); E._pc_last = inj; n = n_pool + len(inj)
        X = rs.rand(n, len(FEATURES)).astype(np.float32); X[:, 0] = np.round(X[:, 0], 1)            # ties in f.z
        F = rs.rand(n, len(NAMES_F)).astype(np.float32)
        keys = [f'{mid}K{i % (n_pool - 1)}' for i in range(n_pool)] + [r['key'] for r in inj]       # one duplicated key among the pool rows
        C = types.SimpleNamespace(pid=np.r_[np.arange(n_pool), -np.ones(len(inj), int)], key=keys, smiles=[f'S{k}' for k in keys],
                                  formula=[f'C{i % 3}H' for i in range(n)])
        return C, X, F, list(NAMES_F), dict(lib_max=LIB[mid])


def hash_int(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


class StubRanker:
    features = ['x__fz', 'x__fz_rank', 'fam__a', 'j__inj', 'j__pcrank', 'j__fz_rel_pool', 'j__inj_above_pool', 'j__n_pool']
    boosters = [types.SimpleNamespace(predict=lambda M, num_threads=1: M[:, 0] - 0.5 * M[:, 1] + 0.3 * M[:, 2] - 0.2 * M[:, 3] + 0.01 * M[:, 5])]
    matrix = ranker_infer.OurRanker.matrix; score = ranker_infer.OurRanker.score


def run_cell(src, cfg, rerun=True, fail_fs=()):
    """Execute the source of nb/06 on six stub molecules; returns (BASE, FAILS, V, E, log lines)."""
    tmp = tempfile.mkdtemp(); mids = list(LIB)
    te = pd.DataFrame(dict(molecule_id=[m for m in mids for _ in range(2)], precursor_mz=[200.0 + 10 * i + 1.007 for i in range(len(mids)) for _ in range(2)],
                           adduct='[M+H]+', collision_energy_ev=[np.array([20.0, 40.0]), None] * len(mids), ionization_mode='positive',
                           ms2_mzs=[np.array([50.0, 60.0])] * (2 * len(mids)), ms2_normalized_intensities=[np.array([1.0, 0.5])] * (2 * len(mids))))
    te.to_parquet(os.path.join(tmp, 'test.parquet'))
    E = types.SimpleNamespace(bank='PUB', _pc_inject=[], _pc_last=[], by_target={round(200.0 + 10 * i, 4): m for i, m in enumerate(mids)})
    V = StubV(E); V.fail_fs = set(fail_fs)
    pj = types.ModuleType('pc_join_stub'); pj.__dict__.update({k: v for k, v in pc_join.__dict__.items() if not k.startswith('__')})
    pj.library_max = lambda E_, spectra, target: LIB[E.by_target[round(target, 4)]]
    pj.build_records = lambda smiles, keys, chem, fragmod, bits, fz=None: [dict(smiles=s, key=k, pc_smiles='P' + s, pc_rank=i, pc_fz=float(fz[i]), mass=0.0) for i, (s, k) in enumerate(zip(smiles, keys))]
    pj.injected_mask = lambda E_, C: np.r_[np.zeros(len(C.pid) - len(E._pc_last), bool), np.ones(len(E._pc_last), bool)]
    pj.injected_info = lambda E_: list(E._pc_last)
    pj.set_injection = lambda E_, recs: setattr(E, '_pc_inject', [dict(r, mass=0.0) for r in (recs or [])])
    PC_DEEP = {m: dict(pc=[f'{m}pc{i}' for i in range(5)], pc_keys=[f'{m}PK{i}' for i in range(5)], pc_fz=[5.0 - i for i in range(5)]) for m in mids if m != 'm2'}
    logs, fails = [], []
    chem = types.SimpleNamespace(neutral_mass=lambda mz, ad: np.asarray(mz, float) - 1.007)
    # the cell filters records by |mass - target| <= 30 ppm; the stub records carry mass 0 -> give them the target instead
    pj_build = pj.build_records
    g = dict(os=os, sys=sys, np=np, pd=pd, time=time, json=json, CFG=cfg, COMP=tmp, chem=chem, E=E, V=V, FEATURES=list(FEATURES), pc_join=pj, fs_pass=fs_pass,
             conf_gate=None, PC_DEEP=PC_DEEP, PC={}, IS_RERUN=rerun, FAILS=fails, log=lambda *a: logs.append(' '.join(map(str, a))), dump=lambda *a: None,
             rank_score=lambda X, names: X[:, names.index('fz')] * 2.0 + X[:, names.index('fam__a')] - X[:, names.index('fz_rank')],
             OUR_RK=StubRanker(), FS_BANK='FS', FS_VIEWS=['fs_view'], fragmod=None, P=types.SimpleNamespace(bits=None))
    targets = {m: 200.0 + 10 * i for i, m in enumerate(mids)}
    pj.build_records = lambda smiles, keys, chem_, fragmod, bits, fz=None: [dict(r, mass=targets[r['smiles'][:2]]) for r in pj_build(smiles, keys, chem_, fragmod, bits, fz)]
    pj.set_injection = lambda E_, recs: setattr(E, '_pc_inject', list(recs or []))
    exec(compile(src, 'nb/06_base_lists.py', 'exec'), g)
    return g['BASE'], fails, V, E, logs


def cell():
    return open(os.path.join(SOL, 'nb', '06_base_lists.py'), encoding='utf-8').read()


def digest(base):
    return hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()


def test_flag_off_is_v24a_bit_for_bit():
    base, fails, V, E, _ = run_cell(cell(), load_cfg())
    assert digest(base) == GOLDEN, digest(base)
    assert not fails and all(c[1] == 'PUB' for c in V.calls)


def test_flag_on_changes_only_molecules_below_the_gate():
    ref = run_cell(cell(), load_cfg())[0]
    for mode in fs_pass.MODES:
        for gate in (0.7, 0.5):
            base, fails, V, E, logs = run_cell(cell(), load_cfg(RANKER=mode, RANKER_LIB_GATE=gate))
            assert not fails
            for m, lib in LIB.items():
                if lib >= gate:
                    assert base[m] == ref[m], (mode, gate, m)
                else:
                    assert base[m][2] == ref[m][2] and set(base[m][1]) == set(ref[m][1]) or len(ref[m][1]) >= 60
            assert any(base[m] != ref[m] for m in LIB if LIB[m] < gate), (mode, gate)
            fs_calls = [c for c in V.calls if c[1] == 'FS']
            assert sorted(c[0] for c in fs_calls) == sorted(m for m in LIB if LIB[m] < gate)             # one extra pass per molecule below the gate
            assert all(c[2] == ('fs_view',) and c[3] == 3 for c in fs_calls)                              # one model, three view slots
            assert E.bank == 'PUB' and V.banks == ['fe_A', 'fe_B', 'fe_D'] and 'model_views' not in V.__dict__
            assert any('fold-safe pass' in l for l in logs)
    base = run_cell(cell(), load_cfg(RANKER='fs', RANKER_LIB_GATE=0.0))[0]                                # nobody below the gate
    assert digest(base) == GOLDEN


def test_joined_rows_and_pool_ids_survive_the_second_pass():
    base = run_cell(cell(), load_cfg(RANKER='fs', RANKER_LIB_GATE=0.7))[0]
    smis, keys, lib, scs, forms, pids = base['m1']
    assert len(smis) == len(keys) == len(scs) == len(forms) == len(pids) and len(set(keys)) == len(keys)
    assert scs == sorted(scs, reverse=True)
    j = [i for i, p in enumerate(pids) if p == -2]
    assert len(j) == 5 and all(smis[i].startswith('P') for i in j)                                        # joined rows keep the PubChem-tier SMILES
    assert base['m2'][5] and -2 not in base['m2'][5]                                                      # no PubChem list: second pass without joined rows


def test_commit_run_exercises_every_molecule_and_raises_on_a_bug():
    ref = run_cell(cell(), load_cfg(PC_JOIN_MAX_LIB=0.7), rerun=False)[0]
    base, fails, V, E, logs = run_cell(cell(), load_cfg(RANKER='fs_avg', RANKER_LIB_GATE=0.7), rerun=False)
    assert sorted(c[0] for c in V.calls if c[1] == 'FS') == sorted(LIB)
    assert all(base[m] == ref[m] for m in LIB if LIB[m] >= 0.7)
    try:
        run_cell(cell(), load_cfg(RANKER='fs', RANKER_LIB_GATE=0.7), rerun=False, fail_fs={'m4'})
    except (RuntimeError, AssertionError):                  # the cell's outer handler counts it and its final assert stops the commit run
        pass
    else:
        raise AssertionError('a failing fold-safe pass must fail the commit run')


def test_rerun_failure_falls_back_to_the_first_pass_list():
    ref = run_cell(cell(), load_cfg())[0]
    base, fails, V, E, _ = run_cell(cell(), load_cfg(RANKER='fs', RANKER_LIB_GATE=0.7), fail_fs={'m1'})
    assert base['m1'] == ref['m1'] and base['m5'] != ref['m5']
    assert len(fails) == 1 and 'm1' in fails[0] and E.bank == 'PUB' and V.banks == ['fe_A', 'fe_B', 'fe_D']


def test_rank_average():
    pub = [('a', 'sa', 'F', 1, 3.0), ('b', 'sb', 'F', 2, 2.0), ('c', 'sc', 'F', 3, 1.0), ('p', 'sp', 'F', 9, 0.5)]
    fs = [('c', 'tc', 'G', 3, 9.0), ('a', 'ta', 'G', 1, 8.0), ('b', 'tb', 'G', 2, 1.0), ('q', 'tq', 'G', -2, 0.0)]
    out = fs_pass.rank_average(pub, fs)
    assert [e[0] for e in out] == ['a', 'c', 'b', 'q', 'p']                # a: (1+2)/2, c: (3+1)/2 -> tie broken by the fold-safe rank? a 1.5 < c 2.0
    assert out[0][1] == 'ta' and out[4][1] == 'sp' and out[3][3] == -2     # rows of the fold-safe pass; a public-only key keeps its row
    s = [e[4] for e in out]; assert s == sorted(s, reverse=True)
    assert [e[0] for e in fs_pass.rank_average(pub, fs, w=1.0)][:4] == ['c', 'a', 'b', 'q']
    assert [e[0] for e in fs_pass.rank_average(pub, fs, w=0.0)][:4] == ['a', 'b', 'c', 'p']
    assert fs_pass.rank_average([], fs) == [(e[0], e[1], e[2], e[3], out_s) for e, out_s in zip(fs, [x[4] for x in fs_pass.rank_average([], fs)])]
    assert fs_pass.rank_average(pub, []) and fs_pass.rank_average([], []) == []
    tie = fs_pass.rank_average([('x', 1, 1, 1, 2.0), ('y', 1, 1, 1, 1.0)], [('y', 2, 2, 2, 2.0), ('x', 2, 2, 2, 1.0)])
    assert [e[0] for e in tie] == ['y', 'x']                               # equal average rank: the fold-safe chain decides


def test_join_features_match_the_training_definition():
    X = np.array([[3.0], [5.0], [4.0], [6.0], [1.0]], np.float32); inj = np.array([0, 0, 1, 1, 1], bool); pcr = np.array([-1, -1, 0, 1, 2])
    J = ranker_infer.join_features(X, ['fz'], inj, pcr)
    assert J['inj'].tolist() == [0, 0, 1, 1, 1] and np.allclose(J['pcrank'], [-1, -1, 0, 0.02, 0.04])
    assert np.allclose(J['fz_rel_pool'], [-2, 0, -1, 1, -4]) and np.allclose(J['inj_above_pool'], 1 / 3) and np.allclose(J['n_pool'], np.log1p(2))
    J0 = ranker_infer.join_features(X, ['fz'])
    assert not J0['inj'].any() and np.allclose(J0['pcrank'], -1) and np.allclose(J0['fz_rel_pool'], X[:, 0] - 6) and np.allclose(J0['n_pool'], np.log1p(5))


def test_entries_and_swapped():
    rows = fs_pass.entries(['a', 'b', 'a', 'c'], ['1', '2', '3', '4'], ['F'] * 4, [0, 1, 2, -2], [1.0, 3.0, 2.0, 3.0])
    assert [(r[0], r[1]) for r in rows] == [('b', '2'), ('c', '4'), ('a', '3')]
    assert fs_pass.to_base(rows, 2) == (['2', '4'], ['b', 'c'], [3.0, 3.0], ['F', 'F'], [1, -2])
    E = types.SimpleNamespace(bank='PUB'); V = StubV(E)
    try:
        with fs_pass.swapped(E, V, 'FS', ['v']):
            assert E.bank == 'FS' and V.banks == ['v'] and len(V.model_views(None)[0]) == 3
            raise KeyError('x')
    except KeyError:
        pass
    assert E.bank == 'PUB' and V.banks == ['fe_A', 'fe_B', 'fe_D'] and len(V.model_views(None)[0]) == 3 and 'model_views' not in V.__dict__


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--golden':
        print(digest(run_cell(open(sys.argv[2], encoding='utf-8').read(), load_cfg())[0])); sys.exit(0)
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
    print('fs_pass OK: flag off == v24a bit for bit, the pass acts below the gate only, fallback and commit-run behaviour, rank average, join features')
