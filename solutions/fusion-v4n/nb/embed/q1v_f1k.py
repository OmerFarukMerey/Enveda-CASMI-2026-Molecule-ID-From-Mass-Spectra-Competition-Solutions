"""q1k cell (F1K: identity-keyed forward-score lookup), inserted verbatim right before q1's metric-fusion cell (cell 30, before the final
re-ranks of cells 30 and 34) by scratch/q1v/build_q1v.py.

Bug (solutions/fusion-v4n/reviews/audit-2026-10-17.md section 3, F1K): ICEBERG / GLACIER scores are stored per SMILES string sent to the
simulators (ICE_SCORES[m], GL_SCORES[m], cell 22) and read back by the candidate's own SMILES (then chem.score_key, which the dicts never
hold). An engine-2 row whose structure was sent under the base list's SMILES is read as missing: z = 0 instead of its score.
Fix (scratch/q1v/PREREG.md, = scratch/audit1017/au_replay.py F1K): per molecule, every candidate string of PREFUSION_LISTS[m] + ENG[m]['smiles']
whose own entry is missing or None gets the value of the first non-None entry with the same metric identity. Own scores are never replaced;
dict truthiness (and so every branch of cells 30 / 33 / 34) is unchanged. Switch F1K (complete False = nothing touched: q1 byte for byte).
Self-test first (fails -> off); per-molecule error -> q1's dicts for that molecule. Log: F1K_PATCH {...}.
"""
F1K = dict(complete=False)


def f1k_complete(d, candidates, ident):
    """Score dict {smiles: score | None} -> (completed copy, number of entries added). Pure; `d` is not modified."""
    if not isinstance(d, dict) or not d:
        return d, 0
    by = {}
    for s, v in d.items():
        if v is not None:
            k = ident(s)
            if k is not None and k not in by:
                by[k] = v
    out, n = dict(d), 0
    for s in candidates:
        if not isinstance(s, str) or out.get(s) is not None:
            continue
        k = ident(s)
        if k is not None and k in by:
            out[s] = by[k]; n += 1
    return out, n


def f1k_candidates(prefusion, eng):
    """The strings cells 30 / 34 can look up for one molecule: pre-fusion list (no blanks, no 'CCO') + engine 2's list."""
    first = [s for s in (prefusion or []) if s and s != 'CCO']
    second = list(eng.get('smiles') or []) if isinstance(eng, dict) else []
    return first + second


def f1k_selftest(ident=None):
    """Toy cases; with `ident` (q1's metric_identity) also RDKit stereo / tautomer twins."""
    key = lambda s: s.lower() if isinstance(s, str) and s.strip() else None
    d = {'A': 1.0, 'b': 2.0, 'c': None, 'C': 3.0, 'E': None}
    out, n = f1k_complete(d, ['a', 'B', 'c', 'D', 'e', 'A', None], key)
    assert out == {'A': 1.0, 'b': 2.0, 'c': 3.0, 'C': 3.0, 'E': None, 'a': 1.0, 'B': 2.0} and n == 3, (out, n)
    assert d == {'A': 1.0, 'b': 2.0, 'c': None, 'C': 3.0, 'E': None}                     # input untouched
    assert f1k_complete({'A': 5.0, 'a': 7.0}, ['a'], key) == ({'A': 5.0, 'a': 7.0}, 0)  # own score never replaced
    assert f1k_complete({'A': 5.0, 'a': None}, ['x', 'a'], key)[0]['a'] == 5.0          # None entry is filled
    assert f1k_complete({}, ['a'], key) == ({}, 0) and f1k_complete(None, ['a'], key) == (None, 0)
    allnone = {'A': None}
    assert f1k_complete(allnone, ['a'], key) == (allnone, 0)                            # no non-None value -> nothing added
    assert f1k_candidates(['X', '', 'CCO', 'Y'], dict(smiles=['Z'], keys=['z'])) == ['X', 'Y', 'Z']
    assert f1k_candidates(None, None) == []
    if ident is not None:
        A, B = 'C=C(C)C1CCC2CCC(=O)C=C2C1', 'C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1'   # stereo twin (metric regression pair)
        K, T = 'CC(=O)CC(=O)C', 'CC(O)=CC(=O)C'                                   # tautomers
        out, n = f1k_complete({A: 1.5, T: -2.0, 'Oc1ccccc1': 0.3}, [B, K, 'NCC(=O)O', 'not a smiles'], ident)
        assert n == 2 and out[B] == 1.5 and out[K] == -2.0 and 'NCC(=O)O' not in out and 'not a smiles' not in out, out
    return 'ok'


def _f1k_main(scope, workdir='.'):
    import hashlib, json
    from pathlib import Path
    sw = dict(scope.get('F1K') or {})
    stat = dict(sw, selftest='not run', status='off', molecules=0, ice_present=0, gl_present=0, ice_mols=0, gl_mols=0, ice_added=0,
                gl_added=0, fallback=0)
    if not sw.get('complete'):
        print('F1K_PATCH', json.dumps(stat), flush=True); return stat
    try:
        stat['selftest'] = f1k_selftest(scope['metric_identity'])
    except Exception as exc:
        stat['selftest'] = 'FAILED ' + repr(exc)[:200]; stat['status'] = 'off (self-test)'
        print('F1K_SELFTEST_FAILED patch switched off', repr(exc)[:300], flush=True)
        print('F1K_PATCH', json.dumps(stat), flush=True); return stat
    try:
        sub = Path(workdir) / 'submission.csv'
        stat['base_sha256'] = hashlib.sha256(sub.read_bytes()).hexdigest() if sub.is_file() else None
        ident = scope['metric_identity']
        ICE, GL, PRE, ENG = scope['ICE_SCORES'], scope['GL_SCORES'], scope['PREFUSION_LISTS'], scope['ENG']
        assert isinstance(PRE, dict) and isinstance(ENG, dict), 'f1k inputs'
        examples = []
        for mid in sorted(set(map(str, PRE)) | set(map(str, ENG))):
            stat['molecules'] += 1
            try:
                cands = f1k_candidates(PRE.get(mid), ENG.get(mid))
                upd = {}
                for name, D in (('ice', ICE), ('gl', GL)):
                    if not isinstance(D, dict) or not isinstance(D.get(mid), dict):
                        continue
                    stat[name + '_present'] += 1
                    new, n = f1k_complete(D[mid], cands, ident)
                    if n:
                        before = any(v is not None for v in D[mid].values())
                        assert before and all(new[s] == v for s, v in D[mid].items() if v is not None), 'f1k replaced a score'
                        upd[name] = (D, new, n)
                for name, (D, new, n) in upd.items():   # commit only after both dicts passed
                    D[mid] = new; stat[name + '_mols'] += 1; stat[name + '_added'] += n
                if upd and len(examples) < 3:
                    examples.append(dict(molecule_id=mid, **{k + '_added': v[2] for k, v in upd.items()}))
            except Exception as exc:
                stat['fallback'] += 1; print('F1K_FALLBACK', mid, repr(exc)[:200], flush=True)
        stat['status'] = 'applied'; stat['examples'] = examples
    except Exception as exc:
        stat['status'] = 'FAILED ' + repr(exc)[:200]
        print('F1K_FAILED', repr(exc)[:300], '| q1 score dicts kept for the molecules not yet completed', flush=True)
    print('F1K_PATCH', json.dumps(stat), flush=True)
    return stat


if __name__ == '__main__':
    F1K_STATS = _f1k_main(globals())
