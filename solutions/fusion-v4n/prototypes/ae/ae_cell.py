# ── Overlay (team Arsokan): spectral analog edits at rank 2 ──
# Idea from heliosli's r3p overlay (public notebook casmi26-0.481, whose library and site model are private); this
# implementation, its library and its gate are ours. A query whose spectrum matches a training spectrum after shifting by
# one of 19 natural-product modifications (methyl, hydroxyl, acetyl, hexose, ...; modified cosine, 0.01 Da) is probably
# that training structure with the modification at one site. Every single-site edit of the analog is enumerated, edits
# that are themselves training structures are dropped (those would be library matches), and the sites are ranked by
# fragment evidence: matched peaks that moved by the modification mass should come from the analog's 1-2-cut fragments
# that hold the site, unmoved peaks from fragments without it. A LightGBM gate trained by us on 7,858 held-out enveda-180
# structures of train.parquet predicts P(the top edit is the true structure); out of fold it is calibrated, and at
# p >= 0.25 it fires on 7% of held-out structures with precision 0.47.
# Rule, in lists whose library match is below AE_LIB_TAU: when the gate fires, the top edit goes to rank 2 (rank 1 and
# the order of the rest unchanged, the list stays at 25); if it is already in the list below rank 2 it moves up to rank 2.
# The analog library is built here from train.parquet. Gate file: dataset __AE_DATASET__ (ae_gate_v1.txt, SHA-256
# pinned). Any error restores the file the final layer wrote.
AE_ON = True
AE_TAU = __AE_TAU__
AE_LIB_TAU = __AE_LIB_TAU__
AE_BUDGET_S = 2400
AE_INPUT = '/kaggle/input'
AE_CORE_SRC = __AE_CORE_SRC__
AE_GATE_SHA256 = __AE_GATE_SHA__


def _ae_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _ae(lib_tau=AE_LIB_TAU, dry=False, state={}):
    """dry: examine every list (lib_tau above 1 reaches library lists too) into ae_dry_run.csv; submission.csv untouched."""
    import types, time, glob
    import numpy as np
    import lightgbm as lgb
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    t0 = time.time()
    rep = dict(status='start', tau=AE_TAU, lib_tau=lib_tau, dry=dry)
    if 'A' not in state:
        A = types.ModuleType('_ae_core'); exec(compile(AE_CORE_SRC, '_ae_core', 'exec'), A.__dict__)
        hits = [p for p in glob.glob(AE_INPUT + '/**/ae_gate_v1.txt', recursive=True) if _ae_sha(p) == AE_GATE_SHA256]
        assert hits, 'pinned ae_gate_v1.txt not found'
        booster = lgb.Booster(model_file=hits[0])
        assert booster.num_feature() == len(A.FEATS), (booster.num_feature(), len(A.FEATS))
        comp = sorted({Path(p).parent for p in glob.glob(AE_INPUT + '/**/test.parquet', recursive=True)},
                      key=lambda p: ('mass-spectra' not in str(p).lower(), len(str(p))))[0]
        Z, S, train_keys = A.build_library(comp / 'train.parquet', log=lambda *a: None)
        state.update(A=A, booster=booster, comp=comp, L=A.Library(Z, S), train_keys=train_keys, lib_seconds=round(time.time() - t0, 1))
        print('AE library', len(Z['sidx']), 'spectra of', len(S), 'structures in', state['lib_seconds'], 's', flush=True)
    A, booster, comp, L, train_keys = (state[k] for k in ('A', 'booster', 'comp', 'L', 'train_keys'))
    rep['library_seconds'] = state['lib_seconds']
    te = pd.read_parquet(comp / 'test.parquet', columns=['molecule_id', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'])
    te['molecule_id'] = te.molecule_id.astype(str)
    queries = {mid: A.test_queries(zip(g.adduct, g.precursor_mz, g.ms2_mzs, g.ms2_normalized_intensities))
               for mid, g in te.groupby('molecule_id', sort=False)}
    ev = pd.read_parquet(ROOT / 'frozen_candidate_evidence.parquet', columns=['molecule_id', 'library_max', 'processed'])
    ev['molecule_id'] = ev.molecule_id.astype(str)
    lib = ev.groupby('molecule_id').library_max.first(); proc = ev.groupby('molecule_id').processed.first()
    sub_path = ROOT / 'submission.csv'
    sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    if not dry: shutil.copy2(sub_path, ROOT / 'submission_pre_ae.csv')
    c = dict(molecules=len(sub), unprocessed=0, library=0, examined=0, with_candidates=0, fired=0, inserted=0, moved=0,
             already_top2=0, failed=0, out_of_time=0)
    fired, out = [], []

    def k14(s):
        try:
            m = Chem.MolFromSmiles(s)
            return Chem.MolToInchiKey(m)[:14] if m is not None else None
        except Exception:
            return None

    for mid, smiles in zip(sub.molecule_id.astype(str), sub.smiles):
        lst = smiles.split(';'); new = list(lst)
        try:
            if time.time() - t0 > AE_BUDGET_S:
                c['out_of_time'] += 1
            elif not bool(proc.get(mid, False)) or not queries.get(mid):
                c['unprocessed'] += 1
            elif float(lib.get(mid, 0.)) >= lib_tau:
                c['library'] += 1
            else:
                c['examined'] += 1
                rows = A.molecule_candidates(queries[mid], L, drop_keys=train_keys)
                if rows:
                    c['with_candidates'] += 1
                    p = booster.predict(pd.DataFrame(rows)[A.FEATS].astype(float))
                    b = int(np.argmax(p)); r = rows[b]
                    if p[b] >= AE_TAU:
                        c['fired'] += 1
                        keys = [k14(s) for s in lst]
                        if r['pick'] in keys[:2]:
                            action = 'already_top2'
                        elif r['pick'] in keys:
                            j = keys.index(r['pick']); new = [lst[0], lst[j]] + [x for i, x in enumerate(lst[1:], 1) if i != j]
                            action = 'moved'
                        else:
                            new = [lst[0], r['pick_smiles']] + lst[1:24]
                            action = 'inserted'
                        c[action] += 1
                        fired.append(dict(molecule_id=mid, p=round(float(p[b]), 3), kind=r['kind'], direction=r['direction'],
                                          score=round(r['score'], 3), analog=r['analog'], pick=r['pick'], action=action,
                                          lib=round(float(lib.get(mid, 0.)), 3), list_len=len(lst)))
            if not (len(set(new)) == len(new) and 1 <= len(new) <= 25 and new[0] == lst[0] and all(new)):
                raise ValueError('edited list malformed')
        except Exception as exc:  # noqa: BLE001 - one molecule never breaks the file
            c['failed'] += 1; new = list(lst)
            if c['failed'] <= 3: print('AE molecule failed', mid, type(exc).__name__, str(exc)[:200])
        out.append(';'.join(new))
    res = sub.copy(); res['smiles'] = out
    assert res.molecule_id.tolist() == sub.molecule_id.tolist()
    rep.update(c, fired_list=fired[:80], seconds=round(time.time() - t0, 1))
    if dry:
        res.to_csv(ROOT / 'ae_dry_run.csv', index=False)
        rep['status'] = 'dry_run'; return rep
    res.to_csv(ROOT / 'submission.ae.tmp.csv', index=False); os.replace(ROOT / 'submission.ae.tmp.csv', sub_path)
    rep.update(status='applied', submission_sha256=_ae_sha(sub_path), pre_sha256=_ae_sha(ROOT / 'submission_pre_ae.csv'))
    return rep


import time
_elapsed_h = (time.time() - globals().get('NB_T0', time.time())) / 3600
if AE_ON and _elapsed_h > 8.0:
    AE_REPORT = dict(status='skipped_time_guard', elapsed_hours=round(_elapsed_h, 2))  # the 9-hour limit comes first
    print('AE OVERLAY skipped by the time guard', round(_elapsed_h, 2), 'h')
elif AE_ON:
    try:
        AE_REPORT = _ae()
    except Exception as exc:  # noqa: BLE001
        import traceback; traceback.print_exc()
        if (ROOT / 'submission_pre_ae.csv').is_file():
            shutil.copy2(ROOT / 'submission_pre_ae.csv', ROOT / 'submission.csv')
        AE_REPORT = dict(status='restored_after_error', error=repr(exc)[:400])
    if not globals().get('IS_RERUN', True):
        # Commit run only: the visible molecules are library hits, so also exercise the overlay on every list (side file).
        try:
            AE_REPORT['commit_dry_run'] = _ae(lib_tau=1.01, dry=True)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc(); AE_REPORT['commit_dry_run'] = dict(status='error', error=repr(exc)[:400])
    (ROOT / 'ae_overlay_report.json').write_text(json.dumps(AE_REPORT, indent=1, default=str))
    print('AE OVERLAY', json.dumps({k: v for k, v in AE_REPORT.items() if k != 'fired_list'}, default=str)[:1500])
final = pd.read_csv(ROOT / 'submission.csv', dtype=str, keep_default_na=False)
assert list(final.columns) == ['molecule_id', 'smiles'] and final.molecule_id.tolist() == SAMPLE_IDS
assert final.smiles.str.split(';').map(len).between(1, 25).all()
assert final.smiles.str.split(';').map(lambda a: all(bool(s) for s in a) and len(set(a)) == len(a)).all()
