"""q1e hook cell (E2_METRIC_EXPORT), inserted verbatim right before q1's engine-2 cell (cell 13) by scratch/q1v/build_q1v.py; in cell 13 the one
line `open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write('<runner>')` becomes `...write(E2X_HOOK('<runner>'))`.

Rule (scratch/q1v/PREREG.md; Codex round 3): engine 2's runner exports per molecule the first 40 of its top-80 ranked candidates deduplicated
by E.canon_key (tautomer-canonical InChIKey-14 computed WITH stereo, so stereo/tautomer twins of one metric identity can use several of the 40
slots). New export: the same scan in the same order; a candidate is kept iff the legacy rule keeps it (canon_key not None, not seen by the
legacy scan) AND its exact metric identity (q1 cell 28's code, embedded verbatim: stereo stripped, tautomer canonical, InChIKey-14 +
heavy-atom composition) is not yet exported; stop at 40. Exported keys stay the legacy canon_key (downstream lookups in cells 22 / 26).
Asserted per molecule: row 1 unchanged, the legacy list deduplicated by metric identity is a prefix of the new list, new rows come from legacy
ranks 41-80. Any failure for a molecule -> its legacy list. Switch E2X (E2_METRIC_EXPORT False = the hook returns the runner source
unchanged: q1 byte for byte). The hook self-tests on RDKit stereo / tautomer cases (fails -> off); the runner repeats the self-test with the
engine's own canon_key (fails -> legacy export). Logs: E2X_HOOK {...} (this cell), E2X_PATCH {...} (runner, printed by cell 13).
"""
E2X = dict(E2_METRIC_EXPORT=False)

# q1 cell 28, lines 1-61 verbatim (scratch/q1v/build_q1v.py asserts it is a prefix of cell 28): the competition metric identity.
E2X_IDENT_SRC = r'''from functools import lru_cache
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
_tautomer_canonicalizer = rdMolStandardize.TautomerEnumerator()
def canonicalize_smiles_to_ik14(smiles: str) -> str:
    """Canonicalize SMILES to InChIKey14, or None if parsing fails.

    Stereochemistry is removed *before* the tautomer canonicalization, not left to fall out of
    the InChIKey14 truncation afterwards. Truncating to InChIKey14 discards the stereo block, so
    it looks like it should already make stereochemistry free -- but the canonicalization runs
    first, and `TautomerEnumerator` will not move a double bond whose position a neighbouring
    stereocentre constrains. The same skeleton drawn with and without that stereocentre can
    therefore land on two different canonical tautomers, and different tautomers are different
    connectivity, which is the part of the InChIKey14 that survives truncation. Stripping first
    makes the canonical form depend only on the skeleton, which is what the competition promises.

    Both the answer key and the submission go through this function, so the two sides can never
    be treated asymmetrically.
    """
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        canonical_mol = _tautomer_canonicalizer.Canonicalize(mol)
        inchikey = Chem.MolToInchiKey(canonical_mol)
        return inchikey[:14]
    # RDKit's MolSanitizeException family subclasses ValueError; RuntimeError covers the C++
    # layer and TypeError covers Boost.Python argument errors. One participant's malformed
    # SMILES must never take the whole scoring run down, so a failure here is just a non-match.
    except (ValueError, RuntimeError, TypeError):
        return None

def heavy_atom_composition(smiles: str):
    """Heavy-atom element counts (excluding H, ignoring charge) as a hashable key, or None
    if parsing fails.

    Cheap (no tautomer enumeration) -- used to gate the expensive canonicalization. Anything
    that shares an InChIKey14 with an accepted structure has the same heavy-atom skeleton, so
    a heavy-atom-composition mismatch lets us skip canonicalization with no change to any
    score. We deliberately do NOT gate on full molecular formula: protomers / charge variants
    (e.g. a zwitterion C(=O)[O-]...[N+] vs its neutral-acid protomer C(=O)O...[N+]) share an
    InChIKey14 but differ in H count and charge, so a formula gate would wrongly reject them.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != 'H':
            counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))

@lru_cache(maxsize=150000)
def metric_identity(smiles):
    key = canonicalize_smiles_to_ik14(smiles)
    return (key, heavy_atom_composition(smiles)) if key else None
'''

# pure selection code; embedded verbatim into eng_runner.py when the switch is on
E2X_SEL_SRC = r'''
def e2x_select(ranked, canon, ident, legacy_smis, legacy_keys, n=40):
    """ranked SMILES (engine 2's top-80, best first) -> (smiles, keys, info). Legacy scan state is replayed exactly; a row is exported iff
    the legacy rule keeps it and its metric identity is new. Checks the replayed legacy list against the runner's own legacy result."""
    lseen, iseen, leg_all, out_s, out_k = set(), set(), [], [], []
    for s in ranked:
        if len(out_s) >= n:
            break
        k = canon(s)
        if k is None or k in lseen:
            continue
        lseen.add(k); leg_all.append((s, k))
        mk = ident(s)
        ik = ('ck', k) if mk is None else mk
        if ik in iseen:
            continue
        iseen.add(ik); out_s.append(s); out_k.append(k)
    leg_s = [s for s, _ in leg_all[:n]]
    assert leg_s == list(legacy_smis) and [k for _, k in leg_all[:n]] == list(legacy_keys), 'e2x legacy replay differs'
    seen, uniq = set(), []
    for s, k in leg_all[:n]:
        mk = ident(s)
        ik = ('ck', k) if mk is None else mk
        if ik not in seen:
            seen.add(ik); uniq.append(s)
    assert out_s[:len(uniq)] == uniq, 'e2x unique prefix changed'
    assert not leg_s or out_s[:1] == leg_s[:1], 'e2x first proposal changed'
    assert set(out_s[len(uniq):]) <= {s for s, _ in leg_all[n:]}, 'e2x new rows not from legacy ranks > n'
    assert len(out_s) <= n and len(set(out_s)) == len(out_s)
    return out_s, out_k, dict(twins=len(leg_s) - len(uniq), added=len(out_s) - len(uniq), changed=out_s != leg_s)


def e2x_legacy(ranked, canon, n=40):
    """The runner's legacy export loop on SMILES (reference for tests and the self-test)."""
    smis, keys, seen = [], [], set()
    for s in ranked:
        k = canon(s)
        if k is None or k in seen:
            continue
        seen.add(k); smis.append(s); keys.append(k)
        if len(smis) >= n:
            break
    return smis, keys


def e2x_selftest_core(ident, canon):
    """RDKit cases: a stereo twin with a different canon_key and the same metric identity is collapsed; tautomers, stereoisomers with
    one canon_key and duplicates behave as in legacy; unparseable rows are skipped as in legacy."""
    A, B = 'C=C(C)C1CCC2CCC(=O)C=C2C1', 'C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1'   # the metric's own regression pair
    assert canon(A) != canon(B) and ident(A) == ident(B) and ident(A) is not None
    ranked = [A, 'Oc1ccccc1', B, 'c1ccccc1O', 'CC(=O)CC(=O)C', 'not a smiles', 'CC(O)=CC(=O)C', 'C[C@H](N)C(=O)O',
              'C[C@@H](N)C(=O)O', 'NCC(=O)O']
    ls, lk = e2x_legacy(ranked, canon, n=4)
    assert ls == [A, 'Oc1ccccc1', B, 'CC(=O)CC(=O)C'], ls
    s, k, info = e2x_select(ranked, canon, ident, ls, lk, n=4)
    assert s == [A, 'Oc1ccccc1', 'CC(=O)CC(=O)C', 'C[C@H](N)C(=O)O'] and k == [canon(x) for x in s], s
    assert info == dict(twins=1, added=1, changed=True), info
    ls, lk = e2x_legacy(ranked, canon, n=40)
    s, k, info = e2x_select(ranked, canon, ident, ls, lk, n=40)
    assert ls == [A, 'Oc1ccccc1', B, 'CC(=O)CC(=O)C', 'C[C@H](N)C(=O)O', 'NCC(=O)O'] and s == [x for x in ls if x != B], (ls, s)
    assert info == dict(twins=1, added=0, changed=True), info
    plain = ['Oc1ccccc1', 'CC(=O)CC(=O)C', 'NCC(=O)O']
    ls, lk = e2x_legacy(plain, canon)
    assert e2x_select(plain, canon, ident, ls, lk) == (ls, lk, dict(twins=0, added=0, changed=False))
    assert e2x_select([], canon, ident, [], []) == ([], [], dict(twins=0, added=0, changed=False))
    return 'ok'
'''
exec(compile(E2X_SEL_SRC, 'q1v_e2x:sel', 'exec'), globals())

# runner-side glue (embedded verbatim into eng_runner.py when the switch is on)
E2X_GLUE_SRC = r'''
_E2X_STATS = dict(selftest='not run', status='off', molecules=0, changed=0, twins=0, added=0, short=0, first_changed=0, fallback=0)


def _e2x_setup(E):
    """Metric identity (q1 cell 28 code) + self-test with the engine's own canon_key; None -> legacy export."""
    try:
        import rdkit
        ns = {}
        exec(compile(_E2X_IDENT_SRC, 'e2x:ident', 'exec'), ns)
        ident = ns['metric_identity']
        _E2X_STATS['rdkit'] = rdkit.__version__
        _E2X_STATS['selftest'] = e2x_selftest_core(ident, E.canon_key)
        _E2X_STATS['status'] = 'applied'
        return ident
    except Exception as exc:
        _E2X_STATS['selftest'] = 'FAILED ' + repr(exc)[:200]; _E2X_STATS['status'] = 'off (self-test)'
        print('E2X_SELFTEST_FAILED legacy export kept', repr(exc)[:300], flush=True)
        return None


def _e2x_apply(mid, ranked, canon, ident, smis, keys):
    if ident is None:
        return smis, keys
    _E2X_STATS['molecules'] += 1
    try:
        s2, k2, info = e2x_select(ranked, canon, ident, smis, keys)
    except Exception as exc:
        _E2X_STATS['fallback'] += 1
        print('E2X_FALLBACK', mid, repr(exc)[:200], flush=True)
        return smis, keys
    _E2X_STATS['changed'] += int(info['changed']); _E2X_STATS['twins'] += info['twins']; _E2X_STATS['added'] += info['added']
    _E2X_STATS['short'] += int(len(s2) < 40); _E2X_STATS['first_changed'] += int(s2[:1] != list(smis[:1]))
    return s2, k2


def _e2x_report(res, legacy):
    try:
        import hashlib
        _E2X_STATS['legacy_sha256'] = hashlib.sha256(json.dumps(legacy).encode()).hexdigest()
        _E2X_STATS['export_sha256'] = hashlib.sha256(json.dumps(res).encode()).hexdigest()
        _E2X_STATS['changed_mids'] = sorted(m for m in res if res[m] != legacy.get(m))[:5]
    except Exception as exc:
        _E2X_STATS['report_error'] = repr(exc)[:200]
    print('E2X_PATCH', json.dumps(_E2X_STATS), flush=True)
'''

# anchors in the runner source (each must occur exactly once) and their replacements
E2X_ANCHORS = (
    ("if __name__ == '__main__':\n",
     None),                                                  # definitions are inserted before this line
    ("    res = {}\n    for r in recs:\n",
     "    res = {}\n    _E2X_IDENT, _E2X_LEGACY = _e2x_setup(E), {}\n    for r in recs:\n"),
    ("        res[str(mid)] = dict(smiles=smis, keys=keys)\n",
     "        _E2X_LEGACY[str(mid)] = dict(smiles=smis, keys=keys)\n"
     "        smis, keys = _e2x_apply(str(mid), [E.POOL['smiles'][c] for c in r['cand'][order]], E.canon_key, _E2X_IDENT, smis, keys)\n"
     "        res[str(mid)] = dict(smiles=smis, keys=keys)\n"),
    ("    print('engine lists', len(res), f'{time.time()-T0:.0f}s', flush=True)",
     "    print('engine lists', len(res), f'{time.time()-T0:.0f}s', flush=True)\n    _e2x_report(res, _E2X_LEGACY)"),
)


def e2x_patch_runner(src):
    """Runner source -> patched source (pure). Raises if an anchor is missing or not unique."""
    for old, _ in E2X_ANCHORS:
        assert src.count(old) == 1, ('e2x anchor', old[:60], src.count(old))
    head = ('# ---- E2X (E2_METRIC_EXPORT): export deduplicated by the exact metric identity (q1v) ----\n'
            '_E2X_IDENT_SRC = ' + repr(E2X_IDENT_SRC) + '\n' + E2X_SEL_SRC + E2X_GLUE_SRC + '# ---- /E2X ----\n\n\n')
    out = src.replace(E2X_ANCHORS[0][0], head + E2X_ANCHORS[0][0])
    for old, new in E2X_ANCHORS[1:]:
        out = out.replace(old, new)
    compile(out, 'eng_runner.py', 'exec')
    return out


def e2x_ref_canon(smi):
    """= casmi_engine.canon_key (tautomer-canonical InChIKey-14 without stripping stereo); only for the notebook-side self-test."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if '_te' not in _E2X_REF:
        _E2X_REF['_te'] = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(_E2X_REF['_te'].Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


_E2X_REF = {}


def e2x_identity():
    """The embedded metric identity (q1 cell 28 code) in a private namespace."""
    if 'ident' not in _E2X_REF:
        ns = {}
        exec(compile(E2X_IDENT_SRC, 'q1v_e2x:ident', 'exec'), ns)
        _E2X_REF['ident'] = ns['metric_identity']
    return _E2X_REF['ident']


def e2x_selftest():
    return e2x_selftest_core(e2x_identity(), e2x_ref_canon)


_E2X_HOOK_STATE = {}


def E2X_HOOK(src):
    """Cell 13 writes eng_runner.py through this hook. Switch off (or a failed self-test / anchor) -> src unchanged."""
    import json
    sw = dict(globals().get('E2X') or {})
    st = dict(sw, status='off', selftest='not run')
    if sw.get('E2_METRIC_EXPORT'):
        try:
            st['selftest'] = e2x_selftest()
            new = e2x_patch_runner(src)
            st.update(status='armed', runner_bytes=[len(src), len(new)])
            src = new
        except Exception as exc:
            st['status'] = 'off (' + repr(exc)[:200] + ')'
            print('E2X_HOOK_FAILED legacy runner kept', repr(exc)[:300], flush=True)
    _E2X_HOOK_STATE.update(st)
    print('E2X_HOOK', json.dumps(st), flush=True)
    return src
