# all molecules -> base lists: BASE[mid] = [smiles, keys, lib_max, ranker scores, formulas, pool ids] (top TOPN, key-deduped;
# pool id < 0 = generated candidate)
te = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
te['nm'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
mols = list(te.groupby('molecule_id', sort=False))
BASE, n_err, n_rk_err, n_rk_diff, n_join, n_join_mol, n_join_err = {}, 0, 0, 0, 0, 0, 0
for gi, (mid, sub) in enumerate(mols):
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    smis, keys, lib_max, scs, forms, pids = [], [], 0.0, [], [], []
    if len(nms):
        target = float(np.median(nms))
        spectra = []
        for r in sub.itertuples():
            ce = r.collision_energy_ev
            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct,
                                prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,
                                ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
        try:
            recs, inj, _out = [], None, {}
            if CFG['PC_JOIN_N'] > 0:
                try:
                    _p = PC_DEEP.get(mid) or PC_DEEP.get(str(mid))
                    # smoke commit runs join every molecule (the smoke molecules are library hits): a bug must fail the commit
                    _gate = CFG['PC_JOIN_MAX_LIB'] if (IS_RERUN or CFG['VALIDATION']) else None
                    if _p and (_gate is None or pc_join.library_max(E, spectra, target) < _gate):
                        _ix = pc_join.select_injection(_p, CFG['PC_JOIN_N'], min_rel=CFG['PC_JOIN_MIN_REL'])
                        recs = pc_join.build_records([_p['pc'][i] for i in _ix], [_p['pc_keys'][i] for i in _ix], chem, fragmod,
                                                     P.bits, fz=[_p['pc_fz'][i] for i in _ix])
                        recs = [r for r in recs if abs(r['mass'] - target) <= target * 30e-6]   # never the 'CCO' fallback
                except Exception as e:                                # scoring rerun: never lose the molecule
                    if not IS_RERUN:
                        raise
                    print('PC JOIN FAILED -> no injection', mid, repr(e)); recs = []; n_join_err += 1
                pc_join.set_injection(E, recs)
            try:
                C, X, F, names, info = V.run(spectra, target)
            except Exception as e:
                if not recs or not IS_RERUN:
                    raise
                print('JOINED RUN FAILED -> run without injection', mid, repr(e)); n_join_err += 1
                pc_join.set_injection(E, []); C, X, F, names, info = V.run(spectra, target)
            lib_max = float(info.get('lib_max', 0.0))
            if CFG['PC_JOIN_N'] > 0:
                inj = pc_join.injected_mask(E, C); _ji = pc_join.injected_info(E)
                n_join += len(_ji); n_join_mol += int(len(_ji) > 0)
                _out = {len(C.pid) - len(_ji) + t: d['pc_smiles'] for t, d in enumerate(_ji)}   # tier SMILES: one forward-model job per structure
                if (gi + 1) % 50 == 0:                                # per-SMILES caches of pid < 0 rows (33 MB per molecule at N 50)
                    for _fm, _c in (('fragnet', '_GEN'), ('frag2', '_GF'), ('frag2', '_GC')):
                        if _fm in V.mods and hasattr(V.mods[_fm], _c):
                            getattr(V.mods[_fm], _c).clear()
            if len(C.pid):
                score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))
                if CFG['RANKER'] != 'public':
                    _a = (CFG['RANKER'], OUR_RK, C.formula, X, F, list(FEATURES), list(names), lib_max, rank_score)
                    if not IS_RERUN:      # commit runs: exercise the ungated branch on every molecule (smoke molecules
                        _dry = pc_join.choose_score(*_a, None, inj)       # are library hits); a bug must fail the commit
                        n_rk_diff += int(np.argmax(_dry) != np.argmax(score))
                    try:
                        score = pc_join.choose_score(*_a, CFG['RANKER_LIB_GATE'], inj)   # == ranker_infer.choose_score without injected rows
                    except Exception as e:                    # scoring rerun: never lose the molecule
                        if not IS_RERUN:
                            raise
                        print('RANKER FAILED -> public ranker', mid, repr(e)); n_rk_err += 1
                order = np.argsort(-score, kind='stable')
                seen = set()
                for i in order:
                    k = C.key[i]
                    if k in seen: continue
                    seen.add(k); smis.append(_out.get(int(i), C.smiles[i])); keys.append(k); scs.append(float(score[i])); forms.append(C.formula[i]); pids.append(int(C.pid[i]))
                    if len(smis) >= CFG['TOPN']: break
        except Exception as e:
            print('ERROR', mid, repr(e)); n_err += 1
    BASE[mid] = [smis, keys, lib_max, scs, forms, pids]
    if (gi + 1) % 25 == 0 or gi == len(mols) - 1:
        log(f'  {gi+1}/{len(mols)} molecules')
log('base lists done, errors', n_err, '| RANKER', CFG['RANKER'], 'gate', CFG['RANKER_LIB_GATE'], 'fallbacks', n_rk_err,
    '| ungated top-1 differs from the public ranker (commit runs):', n_rk_diff,
    '| PC join: injected rows', n_join, 'in', n_join_mol, 'molecules, failures', n_join_err)
assert IS_RERUN or (CFG['RANKER'] == 'public' and CFG['PC_JOIN_N'] == 0) or n_err == 0, \
    'base-list errors with RANKER != public or PC_JOIN_N > 0: fix before submitting'
dump('base', BASE)
