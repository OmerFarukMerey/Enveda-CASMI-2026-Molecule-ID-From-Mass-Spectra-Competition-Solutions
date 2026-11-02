# ── Overlay (team Arsokan): same-formula isomers re-ranked by a learned fragment-explanation model ──
# For every candidate, our own code enumerates the neutral masses left after cutting 1 or 2 bonds and measures how much
# of each query spectrum they explain (several tolerances, intensity scalings and cut depths, minus the chance level
# from peaks shifted by +-5 / +-9 Da). A LightGBM ranker trained by us on 71,891 enveda-180 molecules of train.parquet
# (true structure against up to 15 same-formula isomers; scaffold-held-out top-1 0.751 vs 0.663 for the untrained
# fragment check's own score) turns these into one score per candidate.
# Rule, in lists whose library match is below FRR_LIB_TAU: with FRR_FREEZE_TRAIN training-set structures keep their slots
# (heliosli's FREEZE idea); ranks 1-2 where the fragment check moved the leader always do. Within each molecular formula, the other members
# are re-ordered among their own slots by  score - FRR_LAMBDA * ln(list rank).  FRR_MODE 'top' keeps every slot but
# rank 1 as the merge wrote it: only a rank-1 change is allowed (the winner takes rank 1, the old leader rank 2).
# Model file: private dataset omerfarukmerey/casmi26-om-frr (frr_model_v2.txt, SHA-256 pinned). Any error restores the file the
# merge wrote.
FRR_ON = True
FRR_MODE = __FRR_MODE__
FRR_LAMBDA = __FRR_LAMBDA__
FRR_FREEZE_TRAIN = __FRR_FREEZE__
FRR_LIB_TAU = 0.87
FRR_MAX_SPEC = 6
FRR_FEAT_SRC = __FRR_FEAT_SRC__
FRR_MODEL_SHA256 = __FRR_MODEL_SHA__


def _frr_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _frr(tau=FRR_LIB_TAU, dry=False):
    """dry: examine every list (tau above 1 reaches library lists too) into frr_dry_run.csv; submission.csv untouched."""
    import types, time, glob
    import numpy as np
    import lightgbm as lgb
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog('rdApp.*')
    t0 = time.time()
    rep = dict(status='start', mode=FRR_MODE, lam=FRR_LAMBDA, lib_tau=tau, dry=dry)
    F = types.ModuleType('_frr_feat'); exec(compile(FRR_FEAT_SRC, '_frr_feat', 'exec'), F.__dict__)
    hits = [p for p in glob.glob('/kaggle/input/**/frr_model_v2.txt', recursive=True) if _frr_sha(p) == FRR_MODEL_SHA256]
    assert hits, 'pinned frr_model_v2.txt not found'
    booster = lgb.Booster(model_file=hits[0])
    assert booster.num_feature() == len(F.MODEL_FEATURES), (booster.num_feature(), len(F.MODEL_FEATURES))
    comp = sorted({Path(p).parent for p in glob.glob('/kaggle/input/**/test.parquet', recursive=True)},
                  key=lambda p: ('mass-spectra' not in str(p).lower(), len(str(p))))[0]
    te = pd.read_parquet(comp / 'test.parquet', columns=['molecule_id', 'adduct', 'ionization_mode', 'ms2_mzs', 'ms2_normalized_intensities'])
    te['molecule_id'] = te.molecule_id.astype(str)
    spectra = {}
    for mid, g in te.groupby('molecule_id', sort=False):
        g = g[g.ms2_mzs.map(len) >= 3]
        if len(g) > FRR_MAX_SPEC:
            g = g.sample(FRR_MAX_SPEC, random_state=int(hashlib.sha256(mid.encode()).hexdigest()[:8], 16))
        spectra[mid] = [dict(mz=np.asarray(r.ms2_mzs, float), it=np.asarray(r.ms2_normalized_intensities, float), adduct=str(r.adduct),
                             mode=str(r.ionization_mode)) for r in g.itertuples(index=False)]
    train_keys = set(pd.read_parquet(comp / 'train.parquet', columns=['inchikey14']).inchikey14.astype(str))
    ev = pd.read_parquet(ROOT / 'frozen_candidate_evidence.parquet', columns=['molecule_id', 'library_max', 'processed'])
    ev['molecule_id'] = ev.molecule_id.astype(str)
    lib = ev.groupby('molecule_id').library_max.first(); proc = ev.groupby('molecule_id').processed.first()
    sub_path = ROOT / 'submission.csv'
    sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    if not dry: shutil.copy2(sub_path, ROOT / 'submission_pre_frr.csv')
    moved = set((globals().get('MERGE_REPORT') or {}).get('moved_ids') or [])
    info, levels = {}, {}
    c = dict(molecules=len(sub), unprocessed=0, library=0, examined=0, groups=0, changed=0, top1_changed=0, failed=0,
             frozen_train=0, kept_leader_move=0, leader_train=0)
    fired, out = [], []

    def mol_info(s):
        if s not in info:
            try:
                m = Chem.MolFromSmiles(s)
                info[s] = (CalcMolFormula(m), Chem.MolToInchiKey(m)[:14]) if m is not None else None
            except Exception:
                info[s] = None
        return info[s]

    def lv(s):
        if s not in levels:
            try:
                levels[s] = F.frag_levels(s)
            except Exception:
                levels[s] = None
        return levels[s]

    for mid, smiles in zip(sub.molecule_id.astype(str), sub.smiles):
        lst = smiles.split(';'); new = list(lst)
        try:
            if not bool(proc.get(mid, False)) or not spectra.get(mid):
                c['unprocessed'] += 1
            elif float(lib.get(mid, 0.)) >= tau:
                c['library'] += 1
            else:
                c['examined'] += 1
                meta = [mol_info(s) for s in lst]
                fixed = 2 if mid in moved else 0; c['kept_leader_move'] += int(fixed == 2)
                in_train = {i for i, m in enumerate(meta) if m is not None and m[1] in train_keys}
                frozen = {i for i, m in enumerate(meta) if m is None} | (in_train if FRR_FREEZE_TRAIN else set())
                c['frozen_train'] += len(in_train) if FRR_FREEZE_TRAIN else 0; c['leader_train'] += int(0 in in_train)
                groups = {}
                for i, m in enumerate(meta):
                    if m is not None:
                        groups.setdefault(m[0], []).append(i)
                for form, idx in groups.items():
                    if FRR_MODE == 'top' and 0 not in idx:
                        continue
                    movable = [i for i in idx if i >= fixed and i not in frozen]
                    if len(movable) < 2:
                        continue
                    X = F.molecule_features(spectra[mid], [lv(lst[i]) for i in idx])
                    ok = np.isfinite(X).all(1)
                    if ok.sum() < 2:
                        continue
                    s = np.full(len(idx), np.nan)
                    s[ok] = booster.predict(F.model_matrix(X[ok]))
                    sc = {i: v for i, v in zip(idx, s)}
                    movable = [i for i in movable if np.isfinite(sc[i])]
                    if len(movable) < 2:
                        continue
                    c['groups'] += 1
                    b = np.array([sc[i] - FRR_LAMBDA * np.log(i + 1.0) for i in movable])
                    order = [movable[j] for j in np.argsort(-b, kind='stable')]
                    if FRR_MODE == 'top':
                        if movable[0] == 0 and order[0] != 0:
                            w = order[0]
                            new = [lst[w]] + [x for k, x in enumerate(lst) if k != w]
                            fired.append(dict(molecule_id=mid, rank=w + 1, margin=round(float(sc[w] - sc[0]), 3), lib=round(float(lib.get(mid, 0.)), 3),
                                              train_leader=0 in in_train, train_winner=w in in_train))
                    else:
                        for slot, src in zip(movable, order):
                            new[slot] = lst[src]
                        if new[0] != lst[0]:
                            fired.append(dict(molecule_id=mid, rank=lst.index(new[0]) + 1, margin=round(float(sc[lst.index(new[0])] - sc[0]), 3),
                                              lib=round(float(lib.get(mid, 0.)), 3)))
        except Exception as exc:  # noqa: BLE001 - one molecule never breaks the file
            c['failed'] += 1; new = list(lst)
            if c['failed'] <= 3: print('FRR molecule failed', mid, type(exc).__name__, str(exc)[:200])
        assert sorted(new) == sorted(lst) and len(set(new)) == len(new), mid
        if new != lst:
            c['changed'] += 1; c['top1_changed'] += int(new[0] != lst[0])
        out.append(';'.join(new))
    res = sub.copy(); res['smiles'] = out
    assert res.molecule_id.tolist() == sub.molecule_id.tolist()
    if dry:
        res.to_csv(ROOT / 'frr_dry_run.csv', index=False)
        rep.update(c, status='dry_run', fired=fired[:40], seconds=round(time.time() - t0, 1)); return rep
    res.to_csv(ROOT / 'submission.frr.tmp.csv', index=False); os.replace(ROOT / 'submission.frr.tmp.csv', sub_path)
    rep.update(c, status='applied', fired=fired[:60], seconds=round(time.time() - t0, 1), submission_sha256=_frr_sha(sub_path),
               pre_sha256=_frr_sha(ROOT / 'submission_pre_frr.csv'))
    return rep


import time
_elapsed_h = (time.time() - globals().get('NB_T0', time.time())) / 3600
if FRR_ON and _elapsed_h > 8.0:
    FRR_REPORT = dict(status='skipped_time_guard', elapsed_hours=round(_elapsed_h, 2))  # the 9-hour limit comes first
    print('FRR OVERLAY skipped by the time guard', round(_elapsed_h, 2), 'h')
elif FRR_ON:
    try:
        FRR_REPORT = _frr()
    except Exception as exc:  # noqa: BLE001
        import traceback; traceback.print_exc()
        if (ROOT / 'submission_pre_frr.csv').is_file():
            shutil.copy2(ROOT / 'submission_pre_frr.csv', ROOT / 'submission.csv')
        FRR_REPORT = dict(status='restored_after_error', error=repr(exc)[:400])
    if not globals().get('IS_RERUN', True):
        # Commit run only: the visible molecules are library hits, so also exercise the scorer on every list (side file).
        try:
            FRR_REPORT['commit_dry_run'] = _frr(tau=1.01, dry=True)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc(); FRR_REPORT['commit_dry_run'] = dict(status='error', error=repr(exc)[:400])
    (ROOT / 'frr_overlay_report.json').write_text(json.dumps(FRR_REPORT, indent=1, default=str))
    print('FRR OVERLAY', json.dumps(FRR_REPORT, default=str)[:1500])
