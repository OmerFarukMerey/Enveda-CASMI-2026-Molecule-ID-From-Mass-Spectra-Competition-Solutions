# all molecules -> base lists: BASE[mid] = [smiles, keys, lib_max, ranker scores, formulas, pool ids] (top TOPN, key-deduped;
# pool id < 0 = generated candidate)
GEN_Z, GEN_S1 = {}, {}                       # GEN_ANALOG: FPNet logits and the fold-safe ranker's top score per molecule (empty when off)
if CFG['GEN_ANALOG']:
    import gen_analog
    from casmi.spectra import merge_spectra as _gen_merge
    assert CFG['GEN_ANALOG'] in gen_analog.MODES, CFG['GEN_ANALOG']
    assert CFG['GEN_ANALOG'] != 'gate' or CFG['RANKER'] in fs_pass.MODES, "GEN_ANALOG 'gate' needs the fold-safe pass (RANKER 'fs' / 'fs_avg')"
te = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
te['nm'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
mols = list(te.groupby('molecule_id', sort=False))
BASE, n_err, n_rk_err, n_rk_diff, n_join, n_join_mol, n_join_err = {}, 0, 0, 0, 0, 0, 0
n_conf = n_conf_join = 0; CONF_P = {}
n_fs = n_fs_used = n_fs_diff = n_fs_err = 0; t_fs = 0.0; FS_USED = {}
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
            recs, inj, _out, _r0 = [], None, {}, None
            if CFG['PC_JOIN_N'] > 0:
                try:
                    _p = PC_DEEP.get(mid) or PC_DEEP.get(str(mid))
                    # smoke commit runs join every molecule (the smoke molecules are library hits): a bug must fail the commit
                    _live = IS_RERUN or CFG['VALIDATION']
                    _gate = CFG['PC_JOIN_MAX_LIB'] if _live else None
                    _n = CFG['PC_JOIN_N'] if (_p and (_gate is None or pc_join.library_max(E, spectra, target) < _gate)) else 0
                    if _n and CFG['PC_JOIN_CONF'] is not None:        # confidence pass: the engine without injected rows
                        pc_join.set_injection(E, []); _r0 = V.run(spectra, target); _pc = 0.0
                        if len(_r0[0].pid):
                            _s0 = rank_score(np.concatenate([_r0[1], _r0[2]], 1), list(FEATURES) + list(_r0[3]))
                            _pc = conf_gate.p_present(conf_gate.list_features(_s0, _r0[1], list(FEATURES), _r0[2], list(_r0[3]), _r0[0].formula))
                        n_conf += 1; CONF_P[mid] = round(float(_pc), 4)
                        _n = conf_gate.join_depth(_pc, 0.0, CFG['PC_JOIN_CONF'], _n, None, CFG['PC_JOIN_CONF_DEEP'], CFG['PC_JOIN_N_DEEP']) \
                            if _live else max(_n, CFG['PC_JOIN_N_DEEP'])   # commit runs: the model is exercised, every molecule is joined
                        n_conf_join += int(_n > 0)
                    if _n:
                        _ix = pc_join.select_injection(_p, _n, min_rel=CFG['PC_JOIN_MIN_REL'])
                        recs = pc_join.build_records([_p['pc'][i] for i in _ix], [_p['pc_keys'][i] for i in _ix], chem, fragmod,
                                                     P.bits, fz=[_p['pc_fz'][i] for i in _ix])
                        recs = [r for r in recs if abs(r['mass'] - target) <= target * 30e-6]   # never the 'CCO' fallback
                except Exception as e:                                # scoring rerun: never lose the molecule
                    if not IS_RERUN:
                        raise
                    print('PC JOIN FAILED -> no injection', mid, repr(e)); recs = []; n_join_err += 1
                pc_join.set_injection(E, recs)
            try:
                C, X, F, names, info = _r0 if (_r0 is not None and not recs) else V.run(spectra, target)
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
                if CFG['RANKER'] != 'public' and CFG['RANKER'] not in fs_pass.MODES:
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
                    seen.add(k); smis.append(_out.get(int(i), C.smiles[i])); keys.append(k); scs.append(float(score[i])); forms.append(C.formula[i])
                    pids.append(-2 if (inj is not None and inj[i]) else int(C.pid[i]))   # -2 = joined PubChem row, -1 = generated
                    if len(smis) >= CFG['TOPN']: break
                # fold-safe chain (RANKER 'fs' / 'fs_avg'): a second engine pass with fpnet_fold0 and the ranker fitted to it, used
                # below RANKER_LIB_GATE. Commit runs make the pass on every molecule (the smoke molecules are library hits): a bug
                # must fail the commit. Scoring rerun: a failure keeps the first pass's list for that molecule.
                if CFG['RANKER'] in fs_pass.MODES and (lib_max < CFG['RANKER_LIB_GATE'] or not (IS_RERUN or CFG['VALIDATION'])):
                    _t2 = time.time()
                    try:
                        with fs_pass.swapped(E, V, FS_BANK, FS_VIEWS):
                            C2, X2, F2, names2, _i2 = V.run(spectra, target)
                        _n2 = len(C2.pid); rows2 = []
                        if _n2:
                            _ji2 = pc_join.injected_info(E) if CFG['PC_JOIN_N'] > 0 else []
                            inj2 = np.zeros(_n2, bool); pcr2 = np.full(_n2, -1, np.int64)
                            if _ji2:
                                inj2[_n2 - len(_ji2):] = True; pcr2[_n2 - len(_ji2):] = [d['pc_rank'] for d in _ji2]
                            _out2 = {_n2 - len(_ji2) + t: d['pc_smiles'] for t, d in enumerate(_ji2)}
                            s2 = fs_pass.fs_score(OUR_RK, C2.formula, X2, F2, list(FEATURES), list(names2), inj2, pcr2)
                            assert np.all(np.isfinite(s2)), 'fold-safe ranker returned a non-finite score'
                            if CFG['GEN_ANALOG']:
                                GEN_S1[mid] = float(np.max(s2))
                            rows2 = fs_pass.entries(C2.key, [_out2.get(i, C2.smiles[i]) for i in range(_n2)], C2.formula,
                                                    [-2 if inj2[i] else int(C2.pid[i]) for i in range(_n2)], s2)
                            if CFG['RANKER'] == 'fs_avg':        # half weight to each chain (RANKER_FS_W / RANKER_FS_FADE: another weight)
                                _wfs = fs_pass.weight(lib_max, CFG['RANKER_FS_W'], CFG['RANKER_FS_FADE'])
                                rows2 = fs_pass.rank_average(fs_pass.entries(
                                    C.key, [_out.get(i, C.smiles[i]) for i in range(len(C.pid))], C.formula,
                                    [-2 if (inj is not None and inj[i]) else int(C.pid[i]) for i in range(len(C.pid))], score), rows2, w=_wfs)
                                if _wfs <= 0:                    # faded out: the first pass's list stays as it is
                                    rows2 = []
                        n_fs += 1; _d = int(bool(rows2) and rows2[0][0] != keys[0]); n_fs_diff += _d
                        if rows2 and lib_max < CFG['RANKER_LIB_GATE']:
                            smis, keys, scs, forms, pids = fs_pass.to_base(rows2, CFG['TOPN']); n_fs_used += 1; FS_USED[mid] = _d
                    except Exception as e:
                        if not IS_RERUN:
                            raise
                        print('FS PASS FAILED -> first-pass list', mid, repr(e)); n_fs_err += 1
                        FAILS.append(f'fold-safe pass failed for molecule {mid}: {e!r}'[:300])
                    t_fs += time.time() - _t2
        except Exception as e:
            print('ERROR', mid, repr(e)); n_err += 1
    BASE[mid] = [smis, keys, lib_max, scs, forms, pids]
    if CFG['GEN_ANALOG'] and smis:               # logits for the generator's parent ranking (the fold-safe model when it is loaded)
        try:
            GEN_Z[mid] = gen_analog.spectrum_logits(spectra, FS_BANK if FS_BANK is not None else bank, fpnet, chem, _gen_merge)
        except Exception as e:
            if not IS_RERUN:
                raise
            print('GEN_ANALOG LOGITS FAILED -> no generated rows', mid, repr(e))
    if (gi + 1) % 25 == 0 or gi == len(mols) - 1:
        try:                                  # resident memory: the join's per-SMILES caches grow with PC_JOIN_N
            import psutil; _rss = psutil.Process().memory_info().rss / 2 ** 30
        except Exception:
            _rss = float('nan')
        log(f'  {gi+1}/{len(mols)} molecules | RSS {_rss:.1f} GB | errors {n_err} | joined rows {n_join}, join failures {n_join_err}')
log('base lists done, errors', n_err, '| RANKER', CFG['RANKER'], 'gate', CFG['RANKER_LIB_GATE'], 'fallbacks', n_rk_err,
    '| ungated top-1 differs from the public ranker (commit runs):', n_rk_diff,
    '| PC join: injected rows', n_join, 'in', n_join_mol, 'molecules, failures', n_join_err,
    '| confidence gate: lists scored', n_conf, 'joined', n_conf_join)
if CFG['RANKER'] in fs_pass.MODES:
    log(f"fold-safe pass ({CFG['RANKER']}, gate {CFG['RANKER_LIB_GATE']}): run on {n_fs} molecules, list used for {n_fs_used}, first place differs "
        f"from the first pass in {n_fs_diff}, failures {n_fs_err}, {t_fs:.0f}s ({t_fs / max(1, n_fs + n_fs_err):.2f}s per pass)")
    dump('fs_used', FS_USED)
    if IS_RERUN and n_fs_used == 0:
        FAILS.append('the fold-safe pass was active but no molecule used its list')
if CFG['ADDUCT_TIE'] is not None:
    log('ADDUCT_TIE', CFG['ADDUCT_TIE'], ': merged-adduct ties broken by the rule (one per polarity group and engine pass):', determ.counts())
if CONF_P:
    _cp = np.array(list(CONF_P.values()))
    log('confidence p_present quantiles 10/25/50/75/90:', np.round(np.quantile(_cp, [.1, .25, .5, .75, .9]), 3).tolist())
    dump('conf_p', CONF_P)
if (n_err + n_join_err + n_rk_err) > max(2, 0.02 * len(mols)):      # a couple of odd molecules are not a stage failure
    FAILS.append(f'base lists: {n_err} molecules failed, {n_join_err} join fallbacks, {n_rk_err} ranker fallbacks')
if CFG['USE_PC'] and len(PC) < 0.9 * len(mols):
    FAILS.append(f'pubchem lists for {len(PC)} of {len(mols)} molecules')
if CFG['PC_JOIN_N'] > 0 and (IS_RERUN or CFG['VALIDATION']) and n_join_mol == 0:
    FAILS.append('the join was active but no molecule received joined rows')
assert IS_RERUN or (CFG['RANKER'] == 'public' and CFG['PC_JOIN_N'] == 0) or n_err == 0, \
    'base-list errors with RANKER != public or PC_JOIN_N > 0: fix before submitting'
dump('base', BASE)
