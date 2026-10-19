"""Tests of the three q1 variant cells (nb/embed/q1v_pe.py PE promotion, q1v_f1k.py F1K identity-keyed forward-score lookup, q1v_e2x.py
E2_METRIC_EXPORT); no competition data. Synthetic lists: rule selection, transforms, switch-off identity, RDKit stereo / tautomer cases, and an
end-to-end run of an engine-2 runner (the export section of q1's eng_runner.py on a fake casmi_engine) with the switch off and on.
Run with pytest, or by hand: python solutions/fusion-v4n/tests/test_q1v.py   (RDKit 2026.03.3 for the RDKit cases)"""
import hashlib, io, json, os, subprocess, sys, tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); SOL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOL, 'nb', 'embed'))
import q1v_e2x as X  # noqa: E402
import q1v_f1k as K  # noqa: E402
import q1v_pe as P  # noqa: E402

TOY = lambda s: s.lower() if isinstance(s, str) and s.strip() and s.strip() != 'CCO' else None
A, B = 'C=C(C)C1CCC2CCC(=O)C=C2C1', 'C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1'      # stereo twin: canon_key differs, metric identity equal
KETO, ENOL = 'CC(=O)CC(=O)C', 'CC(O)=CC(=O)C'                                # tautomers: one canon_key, one metric identity


def ident():
    return X.e2x_identity()


def mix_identity():
    """q1's metric identity for SMILES, a lower-case toy identity for tokens RDKit cannot parse ('E', 'R1', 'J', ...)."""
    I = ident()
    return lambda s: I(s) if I(s) is not None else (s.lower() if isinstance(s, str) and s.strip() else None)


def real_key():
    I = ident(); ph = I('CCO')
    return lambda s: None if not isinstance(s, str) or not s.strip() or s.strip() == 'CCO' or I(s) in (None, ph) else I(s)


def test_rdkit_version_and_identity_source():
    import rdkit
    assert rdkit.__version__ == '2026.03.3', rdkit.__version__
    I = ident()
    assert I(A) == I(B) and I(KETO) == I(ENOL) and I('not a smiles') is None
    assert X.e2x_ref_canon(A) != X.e2x_ref_canon(B) and X.e2x_ref_canon(KETO) == X.e2x_ref_canon(ENOL)
    assert 'def metric_identity(smiles):' in X.E2X_IDENT_SRC and 'Chem.RemoveStereochemistry(mol)' in X.E2X_IDENT_SRC


def test_switches_default_off():
    assert P.PE == dict(promote=False) and K.F1K == dict(complete=False) and X.E2X == dict(E2_METRIC_EXPORT=False)


# ----------------------------------------------------------------------------------------------------- q1p: PE promotion
def test_pe_selftest_toy_and_rdkit():
    assert P.pe_selftest() == 'ok' and P.pe_selftest(real_key()) == 'ok'


def test_pe_rule_selection():
    on = dict(promote=True)
    F = ['R1', 'E', 'R3']
    assert P.pe_row(F, ['e', 'X'], ['E', 'Y'], 0.3, 0.9, TOY, on) == (['E', 'R1', 'R3'], 'promoted', '2')   # E2R2's rank-2 row -> rank 1
    assert P.pe_row(F, ['X', 'e'], ['E'], 0.3, 0.9, TOY, on)[1] == 'disagree'                              # only the channel's FIRST row
    assert P.pe_row(F, ['e'], ['Y', 'E'], 0.3, 0.9, TOY, on)[1] == 'disagree'                              # only engine 2's FIRST row
    assert P.pe_row(F, ['r1'], ['R1'], 0.3, 0.9, TOY, on)[1] == 'already_top'
    assert P.pe_row(F, ['e'], ['E'], 0.9, 0.9, TOY, on)[1] == 'lib_strong'                                 # channel not merged
    assert P.pe_row(F, ['e'], ['E'], 0.8999, 0.9, TOY, on)[1] == 'promoted'
    k = real_key()
    G = ['Oc1ccccc1', B, 'NCC(=O)O']
    assert P.pe_row(G, [A], [B], 0.5, 0.9, k, on) == ([B, 'Oc1ccccc1', 'NCC(=O)O'], 'promoted', '2')     # stereo twin channel / engine
    assert P.pe_row(G + [ENOL], [KETO], [ENOL], 0.5, 0.9, k, on) == ([ENOL] + G, 'promoted', '4')        # tautomer
    assert P.pe_row([A, B, KETO], [B], [A], 0.5, 0.9, k, on)[1] == 'already_top'


def _pe_scope(tmp, rows, base, pc, eng, promote=True):
    path = os.path.join(tmp, 'submission.csv')
    pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(path, index=False)
    return dict(PE=dict(promote=promote), metric_identity=mix_identity(), LIB_TAU=0.9,
                BASE=base, PC=pc, ENG=eng), path


def test_pe_main_off_on_and_fallback():
    rows = [('m1', 'R1;E;R3'), ('m2', 'R1;E'), ('m3', 'CCO'), ('m4', 'Z;W'), ('m5', 'Q;P')]
    base = {'m1': (['x'], ['x'], 0.5), 'm2': (['x'], ['x'], 0.95), 'm4': (['x'], ['x'], 0.1), 'm5': 7}
    pc = {'m1': dict(pc=['e']), 'm2': dict(pc=['e']), 'm4': dict(pc=['w']), 'm5': dict(pc=['p'])}
    eng = {'m1': dict(smiles=['E']), 'm2': dict(smiles=['E']), 'm4': dict(smiles=['V']), 'm5': dict(smiles=['P'])}
    with tempfile.TemporaryDirectory() as tmp:
        sc, path = _pe_scope(tmp, rows, base, pc, eng, promote=False)
        raw0 = open(path, 'rb').read()
        st = P._pe_main(sc, tmp)
        assert st['status'] == 'off' and open(path, 'rb').read() == raw0                       # switch off: q1's bytes
        sc['PE'] = dict(promote=True)
        st = P._pe_main(sc, tmp)
        assert st['status'] == 'applied' and st['selftest'] == 'ok' and st['base_sha256'] == hashlib.sha256(raw0).hexdigest()
        assert (st['promoted'], st['lib_strong'], st['skipped'], st['disagree'], st['fallback']) == (1, 1, 1, 1, 1), st
        assert st['moved_from'] == {'2': 1} and st['changed'] == 1 and st['changed_top1'] == 1
        out = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(out.smiles) == ['E;R1;R3', 'R1;E', 'CCO', 'Z;W', 'Q;P']
        assert st['submission_sha256'] == hashlib.sha256(open(path, 'rb').read()).hexdigest()


def test_pe_main_restores_file_on_error():
    with tempfile.TemporaryDirectory() as tmp:
        sc, path = _pe_scope(tmp, [('m1', 'R1;E')], {}, {}, {})                                 # empty BASE -> file-level error
        raw0 = open(path, 'rb').read()
        st = P._pe_main(sc, tmp)
        assert st['status'].startswith('FAILED') and open(path, 'rb').read() == raw0


# ----------------------------------------------------------------------------------------------------- q1k: F1K
def test_f1k_selftest_toy_and_rdkit():
    assert K.f1k_selftest() == 'ok' and K.f1k_selftest(ident()) == 'ok'


def test_f1k_stereo_tautomer_and_lookup():
    I = ident()
    gl = {A: 2.0, KETO: -1.0, 'Oc1ccccc1': None}
    out, n = K.f1k_complete(gl, [B, ENOL, 'Oc1ccccc1', A], I)
    assert n == 2 and out[B] == 2.0 and out[ENOL] == -1.0 and out['Oc1ccccc1'] is None and out[A] == 2.0
    # read back like gl_fuse._getter: own SMILES first, then the score key (never present in the dicts)
    of = lambda d, s: d.get(s) if d.get(s) is not None else d.get('score_key:' + s)
    assert of(gl, B) is None and of(out, B) == 2.0


def _f1k_scope(complete):
    return dict(F1K=dict(complete=complete), metric_identity=mix_identity(),
                ICE_SCORES={'m1': {'J': 1.0, 'g': None}, 'm2': {'Q': None}, 'm3': {}},
                GL_SCORES={'m1': {'G': 3.0}, 'm2': {'q': 5.0}},
                PREFUSION_LISTS={'m1': ['J', 'CCO', ''], 'm2': ['q'], 'm3': ['Z'], 'm4': ['CCO']},
                ENG={'m1': dict(smiles=['j', 'g'], keys=['j', 'g']), 'm2': dict(smiles=['Q'], keys=['q']), 'm4': dict(smiles=[], keys=[])})


def test_f1k_main_off_on():
    import copy
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, 'submission.csv'), 'wb').write(b'molecule_id,smiles\nm1,J\n')
        sc = _f1k_scope(False); ref = copy.deepcopy(sc)
        st = K._f1k_main(sc, tmp)
        assert st['status'] == 'off' and sc == ref                                              # switch off: nothing touched
        sc = _f1k_scope(True); ice0, gl0 = sc['ICE_SCORES'], sc['GL_SCORES']
        st = K._f1k_main(sc, tmp)
        assert st['status'] == 'applied' and st['selftest'] == 'ok' and st['fallback'] == 0
        assert sc['ICE_SCORES'] is ice0 and sc['GL_SCORES'] is gl0                              # same top-level dicts, entries replaced
        assert sc['ICE_SCORES']['m1'] == {'J': 1.0, 'g': None, 'j': 1.0}                        # 'g' has no scored twin in ICE
        assert sc['GL_SCORES']['m1'] == {'G': 3.0, 'g': 3.0} and sc['GL_SCORES']['m2'] == {'q': 5.0, 'Q': 5.0}
        assert sc['ICE_SCORES']['m2'] == {'Q': None} and sc['ICE_SCORES']['m3'] == {}            # all-None / empty: unchanged
        assert (st['ice_added'], st['gl_added'], st['ice_mols'], st['gl_mols']) == (1, 2, 1, 2), st
        assert st['base_sha256'] == hashlib.sha256(b'molecule_id,smiles\nm1,J\n').hexdigest()
        # truthiness of every per-molecule dict (cells 30 / 33 / 34 branch gates) is unchanged
        for D, D0 in ((sc['ICE_SCORES'], _f1k_scope(True)['ICE_SCORES']), (sc['GL_SCORES'], _f1k_scope(True)['GL_SCORES'])):
            assert {m: (bool(d), any(v is not None for v in d.values())) for m, d in D.items()} == \
                   {m: (bool(d), any(v is not None for v in d.values())) for m, d in D0.items()}


def test_f1k_per_molecule_fallback():
    sc = _f1k_scope(True)
    sc['ICE_SCORES']['m2'] = {None: 1.0, 'Q': None}                                              # this identity raises on None
    mix = mix_identity(); sc['metric_identity'] = lambda s: mix(s.strip())
    with tempfile.TemporaryDirectory() as tmp:
        st = K._f1k_main(sc, tmp)
    assert st['status'] == 'applied' and st['fallback'] == 1, st
    assert sc['GL_SCORES']['m2'] == {'q': 5.0} and sc['ICE_SCORES']['m2'] == {None: 1.0, 'Q': None}   # m2: q1's dicts (both)
    assert sc['GL_SCORES']['m1'] == {'G': 3.0, 'g': 3.0}


# ----------------------------------------------------------------------------------------------------- q1e: E2_METRIC_EXPORT
def test_e2x_selftest_and_select():
    assert X.e2x_selftest() == 'ok'
    canon, I = X.e2x_ref_canon, ident()
    ranked = [A, B, KETO, ENOL] + ['C' * i + 'O' for i in range(2, 12)]
    ls, lk = X.e2x_legacy(ranked, canon, n=5)
    assert ls == [A, B, KETO, 'CCO', 'CCCO']
    s, k, info = X.e2x_select(ranked, canon, I, ls, lk, n=5)
    assert s == [A, KETO, 'CCO', 'CCCO', 'CCCCO'] and k == [canon(x) for x in s] and info == dict(twins=1, added=1, changed=True)
    try:
        X.e2x_select(ranked, canon, I, ls[::-1], lk, n=5)
        raise AssertionError('legacy mismatch accepted')
    except AssertionError as e:
        assert 'legacy replay' in str(e)


def test_e2x_hook_off_identity_and_on_patch():
    src = RUNNER
    X.E2X.update(E2_METRIC_EXPORT=False)
    assert X.E2X_HOOK(src) == src                                                               # off: byte-identical runner
    X.E2X.update(E2_METRIC_EXPORT=True)
    try:
        new = X.E2X_HOOK(src)
        assert new != src and new == X.e2x_patch_runner(src) and X._E2X_HOOK_STATE['status'] == 'armed'
        assert X.E2X_HOOK(src.replace('    res = {}\n', '    res = dict()\n')) == src.replace('    res = {}\n', '    res = dict()\n')
        assert X._E2X_HOOK_STATE['status'].startswith('off (')                                   # missing anchor: legacy runner kept
    finally:
        X.E2X.update(E2_METRIC_EXPORT=False)


def test_e2x_glue_fallback():
    ns = dict(json=json)
    exec(X.E2X_SEL_SRC + X.E2X_GLUE_SRC, ns)
    ns['e2x_select'] = lambda *a, **k: (_ for _ in ()).throw(ValueError('boom'))
    assert ns['_e2x_apply']('m', ['x'], lambda s: s, lambda s: s, ['L'], ['l']) == (['L'], ['l'])
    assert ns['_E2X_STATS']['fallback'] == 1
    assert ns['_e2x_apply']('m', ['x'], lambda s: s, None, ['L'], ['l']) == (['L'], ['l'])      # ident None (self-test failed)


# Engine-2 runner: the export section of q1's eng_runner.py (cell 13; from the public notebook, Apache 2.0), engine replaced by a fake.
RUNNER = '''"""Our engine (two-ranker, BIO + AFIX) -> {molecule_id: {smiles: [...], keys: [...]}} (blend order, top 40)."""
import os, sys, json, time
import numpy as np
if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    test, train, sample, out = sys.argv[1:5]
    import casmi_engine as E
    T0 = time.time()
    subs, recs, S = E.main(test, train, sample, None, None, [], workers=1, w_pv=0.88)
    bl = E.blend_scores(S['pv'], S['ours'], 0.88)
    res = {}
    for r in recs:
        mid = r['mid']
        if mid not in bl:
            continue
        order = np.argsort(-bl[mid], kind='mergesort')[:80]
        smis, keys, seen = [], [], set()
        for c in r['cand'][order]:
            s = E.POOL['smiles'][c]; k = E.canon_key(s)
            if k is None or k in seen:
                continue
            seen.add(k); smis.append(s); keys.append(k)
            if len(smis) >= 40:
                break
        res[str(mid)] = dict(smiles=smis, keys=keys)
    json.dump(res, open(out, 'w'))
    print('engine lists', len(res), f'{time.time()-T0:.0f}s', flush=True)
'''

FAKE_ENGINE = '''"""fake casmi_engine for the q1v runner test: a fixed pool, three molecules, the real canon_key code."""
import numpy as np
SM = @@SM@@
POOL = dict(smiles=np.asarray(SM, dtype=object))


def canon_key(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if "te" not in _g: _g["te"] = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(_g["te"].Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


_g = {}
N = len(SM)


class _Knobs:                         # E.RANK / E.CFG: the real runner sets a few attributes on them
    pass


RANK, CFG = _Knobs(), _Knobs()


def find(n):
    return '/nonexistent/' + n


def main(*a, **k):
    c2 = np.r_[2, 0, 4, 6, 1, 3, 10:N]                     # stereo twin B before A
    recs = [dict(mid='m1', cand=np.arange(N)), dict(mid='m2', cand=c2), dict(mid='m3', cand=np.arange(5, 20)), dict(mid='m4', cand=np.arange(N))]
    return None, recs, dict(pv=None, ours=None)


def blend_scores(pv, ours, w):
    return {r: np.linspace(1.0, 0.0, n) for r, n in (('m1', N), ('m2', N - 4), ('m3', 15))}   # m4 has no scores (skipped)
'''


def fake_pool():
    """Pool with stereo twins / tautomers inside the first 40 legacy-unique rows and > 40 distinct structures in total."""
    alc = ['C' * i + 'O' for i in range(3, 41)] + ['C' * i + 'N' for i in range(3, 41)]
    sm = [A, 'Oc1ccccc1', B, 'c1ccccc1O', KETO, 'not a smiles', ENOL, 'C[C@H](N)C(=O)O', 'C[C@@H](N)C(=O)O', 'NCC(=O)O']
    return sm + alc                                                                             # 86 rows


def run_runner(runner_src, pool, tmp, name):
    d = os.path.join(tmp, name); os.makedirs(d)
    open(os.path.join(d, 'eng_runner.py'), 'w', encoding='utf-8', newline='\n').write(runner_src)
    open(os.path.join(d, 'casmi_engine.py'), 'w', encoding='utf-8', newline='\n').write(FAKE_ENGINE.replace('@@SM@@', repr(pool)))
    out = os.path.join(d, 'eng_lists.json')
    r = subprocess.run([sys.executable, os.path.join(d, 'eng_runner.py'), 't', 'tr', 's', out], capture_output=True, text=True, cwd=d)
    assert r.returncode == 0, r.stderr[-2000:]
    return open(out, 'rb').read(), r.stdout


def check_runner(runner_src, pool=None):
    """Legacy run vs E2X run of `runner_src` on the fake engine; returns (E2X_PATCH stats, legacy lists, new lists)."""
    pool = pool or fake_pool()
    canon, I = X.e2x_ref_canon, ident()
    with tempfile.TemporaryDirectory() as tmp:
        leg_b, _ = run_runner(runner_src, pool, tmp, 'legacy')
        X.E2X.update(E2_METRIC_EXPORT=False)
        assert X.E2X_HOOK(runner_src) == runner_src
        X.E2X.update(E2_METRIC_EXPORT=True)
        try:
            new_b, out = run_runner(X.E2X_HOOK(runner_src), pool, tmp, 'e2x')
        finally:
            X.E2X.update(E2_METRIC_EXPORT=False)
    leg, new = json.loads(leg_b), json.loads(new_b)
    line = [l for l in out.splitlines() if l.startswith('E2X_PATCH ')]
    assert len(line) == 1, out[-2000:]
    st = json.loads(line[0][10:])
    assert st['legacy_sha256'] == hashlib.sha256(json.dumps(leg).encode()).hexdigest() == hashlib.sha256(leg_b).hexdigest()
    assert st['export_sha256'] == hashlib.sha256(new_b).hexdigest()
    assert set(new) == set(leg)
    for m in leg:
        ls, ns_ = leg[m]['smiles'], new[m]['smiles']
        assert ns_[:1] == ls[:1] and new[m]['keys'] == [canon(s) for s in ns_]                  # first proposal, legacy keys
        seen, uniq = set(), []
        for s in ls:
            if I(s) not in seen:
                seen.add(I(s)); uniq.append(s)
        assert ns_[:len(uniq)] == uniq and len({I(s) for s in ns_}) == len(ns_) <= 40
    return st, leg, new


def test_e2x_runner_end_to_end():
    st, leg, new = check_runner(RUNNER)
    assert st['status'] == 'applied' and st['selftest'] == 'ok' and st['fallback'] == 0 and st['first_changed'] == 0
    assert (st['molecules'], st['changed'], st['twins'], st['added'], st['short']) == (3, 2, 2, 2, 1), st
    assert st['changed_mids'] == ['m1', 'm2'] and set(leg) == {'m1', 'm2', 'm3'}
    assert B not in new['m1']['smiles'] and B in leg['m1']['smiles'] and len(new['m1']['smiles']) == 40
    assert new['m1']['smiles'] == [s for s in leg['m1']['smiles'] if s != B] + ['C' * 37 + 'O']           # next legacy row (rank 41)
    assert new['m2']['smiles'][:3] == [B, KETO, 'Oc1ccccc1'] and A in leg['m2']['smiles'][:2]
    assert new['m3'] == leg['m3']


if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_') and callable(f):
            f(); print('ok', n)
