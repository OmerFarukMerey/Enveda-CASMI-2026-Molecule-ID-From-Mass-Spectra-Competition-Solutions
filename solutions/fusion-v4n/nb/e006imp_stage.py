# "e006 improved" (E006_BASE; this cell is put after cell 06 by tools/build_notebook.py ONLY when the option is set). The public notebook
# "CASMI26 E006 Window 40 ICE Fallback True" (canqiang, Apache-2.0; snapshot and notice in reference/e006/) runs as a script in
# /kaggle/working/e006. E006_BASE 'ours': with OUR base lists in place of its own engine pass (nb/embed/e006_stage.py, patch 4); its PubChem
# channel with the popularity prior, slots, promotion, forward models, engine fusion, tail fill, key vote and overlay are its own cells.
# E006_BASE 'e006': the script as it is (no hand-over, its own engine 2): the baseline profile; our cells 03 to 09 are then not in the notebook.
# Its table becomes the submission (nb/e006imp_final.py). Stage ok -> cells 07 / 08 skip the forward models (their table is only a stand-in).
# A failure, a bad table or a run past the deadline is handled by E006_BASE_ON_FAIL and ALWAYS logged as 'E006IMP STAGE FAILED' with the
# policy taken (reviews/diag67-2026-10-12.md): 'fail' raises here; 'e006' runs the script a second time without our base lists and takes
# that table (raises if it fails too); 'ours' leaves cells 07 / 08 in full, our own table is the submission.
E6I = dict(status='off')
if CFG['E006_BASE']:
    import gc, shutil, traceback
    import e006_stage
    _e6mode, _e6fail = CFG['E006_BASE'], CFG['E006_BASE_ON_FAIL']
    assert _e6mode in ('ours', 'e006'), _e6mode
    assert _e6fail in ('fail', 'e006', 'ours') and not (_e6mode == 'e006' and _e6fail == 'ours'), _e6fail
    assert not CFG['E006_FUSE'] and not CFG['OUT_TOPK'] and not CFG['GEN_ANALOG'], 'E006_BASE: not with these options'
    assert not CFG['VALIDATION'] or CFG['VAL_AS_RERUN'], 'E006_BASE in a validation run needs VAL_AS_RERUN (a rehearsal of the scoring run)'
    _ew = os.path.join(WORK, 'e006'); os.makedirs(_ew, exist_ok=True)
    resources('before the e006 stage')
    _t6 = time.time(); E6I = dict(status='not run', mode=_e6mode, on_fail=_e6fail, table=None, secs=0, shared_engine2=False, attempts=[])
    _e6val = bool(CFG['VALIDATION']) and not IS_RERUN
    _sid = list(pd.read_csv(os.path.join(COMP, 'sample_submission.csv')).molecule_id)
    _e6full = IS_RERUN or _e6val or bool(CFG['FULL_ON_COMMIT'])
    _e6active = list(_sid) if _e6full else sorted(_sid)[:12]          # the molecules the script processes (its own smoke rule: the first 12)

    def _e6_attempt(tag, base_file, n_handover, base_ids, deadline_s, env_more):
        """One run of the script. -> dict(tag, status 'ok' | 'failed: ...', secs, ...); never raises."""
        a = dict(tag=tag, status='not run', secs=0, started_at=round(time.time() - T0))
        try:
            _script = os.path.join(_ew, 'e006_main.py')
            open(_script, 'w').write(e006_stage.script_improved(json.load(open(os.path.join(OURS, 'e006', 'cells.json'))), _ew))
            env = dict(os.environ, **env_more)
            if base_file:
                env['OURS_E006_BASE'] = base_file
                env.update(e006_stage.improved_env(CFG))
            if _e6val:                                # validation: the held-out test and the purged library, every molecule
                env['OURS_E006_COMP'] = COMP; env['OURS_E006_FULL'] = '1'
            elif CFG['FULL_ON_COMMIT'] and not IS_RERUN:
                env['OURS_E006_FULL'] = '1'
            for _f in ('submission.csv',):            # a table of an earlier attempt must never be read as this one's
                if os.path.exists(os.path.join(_ew, _f)):
                    os.replace(os.path.join(_ew, _f), os.path.join(_ew, _f + '.prev'))
            gc.collect()
            if 'torch' in globals():
                torch.cuda.empty_cache()
            left = deadline_s - (time.time() - T0)
            if CFG['E006_DIAG_KILL_AFTER_S'] and not IS_RERUN and not E6I['attempts']:      # diagnostic: a kill in mid-run, first run only
                left = min(left, float(CFG['E006_DIAG_KILL_AFTER_S'])); print('E006IMP DIAGNOSTIC: the first run will be stopped after', left, 's', flush=True)
            log('E006IMP attempt', tag, '| base lists handed over', n_handover, '| env', {k: v for k, v in env.items() if k.startswith('OURS_E006_')},
                f'| notebook time {time.time() - T0:.0f}s | deadline {deadline_s:.0f}s | time left {left:.0f}s')
            if left < 1800:
                raise RuntimeError(f'only {left:.0f}s left before the deadline ({deadline_s:.0f}s of notebook time): not started')
            lp = os.path.join(_ew, 'e006_%s.log' % tag)
            rc, secs, st = e006_stage.run_script(_script, _ew, env, left, lp)
            a.update(status=st, secs=round(secs), returncode=rc, resources=dict(e006_stage.LAST_MONITOR))
            elog = open(lp, errors='replace').read()
            shutil.copyfile(lp, os.path.join(_ew, 'e006.log'))
            try:
                a['phases'] = e006_stage.phase_times(open(lp + '.timed', errors='replace').read())
            except Exception as e:
                a['phases'] = repr(e)[:100]
            print(elog[-40000:])
            log('E006IMP attempt', tag, 'ended:', st, f'after {secs:.0f}s, return code {rc} | phases (s on the stage clock)', a['phases'], '| resources', a['resources'])
            if st != 'ok':
                raise RuntimeError(f'the script {st} after {secs:.0f}s (return code {rc})')
            tab = e006_stage.read_table(os.path.join(_ew, 'submission.csv'))
            a['real_rows'] = e006_stage.check_table(tab, _sid, base_ids)
            if base_file:
                used = [l for l in elog.split('\n') if l.startswith('E006_STAGE base lists used:')]
                assert used and int(used[-1].split(':')[1].split('of')[0]) >= 0.98 * n_handover, used     # the hand-over really took place
                a['handover'] = used[-1]
            for bad in ('ICE FAILED', 'GL FAILED', 'PUBCHEM CHANNEL FAILED', 'ENGINE FAILED'):                # a stage of ITS pipeline fell back
                if bad in elog:
                    raise RuntimeError('stage fallback inside the script: ' + bad)
            shutil.copyfile(os.path.join(_ew, 'submission.csv'), os.path.join(WORK, 'submission_e006_%s.csv' % tag))
            a['status'] = 'ok'; a['table'] = os.path.join(WORK, 'submission_e006_%s.csv' % tag)
        except Exception as e:
            a['status'] = 'failed: ' + repr(e)[:200]; a['where'] = traceback.format_exc()[-600:]
            if os.path.exists(os.path.join(_ew, 'submission.csv')):        # kept for inspection only, never submitted
                shutil.copyfile(os.path.join(_ew, 'submission.csv'), os.path.join(WORK, 'submission_e006_%s_REJECTED.csv' % tag))
        a['ended_at'] = round(time.time() - T0)
        E6I['attempts'].append(a)
        return a

    _hard = float(CFG['E006_HARD_DEADLINE_S'])
    _dl1 = {'ours': float(CFG['E006_BASE_DEADLINE_S']), 'fail': _hard, 'e006': _hard - float(CFG['E006_FALLBACK_S'])}[_e6fail]
    if _e6mode == 'e006':                                 # the public notebook itself
        _a1 = _e6_attempt('own', None, 0, _e6active, _hard, {})
    else:
        _hb, _njp, _jst, _envs = {}, 0, None, {}
        try:
            _jp = None
            if CFG['E006_JOIN_POP_MU'] > 0:      # popularity of the joined rows (the table POP_MU reads; nb/embed/pc_pop_side.py)
                import pc_pop_side
                _tp = pc_pop_side.TierPop(PC_DIR, os.path.dirname(find('casmi26-pubchem-popularity-prior/**/pc_lsid.npy')))
                _jp, _jst = pc_pop_side.join_pop_table(BASE, {m: p.get('target') for m, p in PC.items()}, _tp)
                assert not _jst['rows'] or _jst['found'] >= 0.9 * _jst['rows'], _jst
            _hb, _njp = e006_stage.base_for_handover(BASE, _jp, CFG['E006_JOIN_POP_MU'], fusion_core.popularity_reorder_join)
            assert len(_hb) >= 0.98 * len(BASE) - 1, (len(_hb), len(BASE))
            json.dump(_hb, open(os.path.join(_ew, 'ours_base.json'), 'w'))
            E6I['shared_engine2'] = bool(CFG['E006_SHARE_ENG']) and e006_stage.engine_shareable(CFG) and bool(ENG) and \
                not any(f.startswith('engine-2') for f in FAILS)
            if E6I['shared_engine2']:
                _envs['OURS_E006_ENG'] = os.path.join(WORK, 'eng_lists.json')
            log('E006IMP stage: base lists', len(_hb), 'of', len(BASE), '| joined rows', sum(sum(1 for p in v[5] if p == -2) for v in _hb.values()), 'in',
                sum(1 for v in _hb.values() if any(p == -2 for p in v[5])), 'lists | lists longer than 25:', sum(1 for v in _hb.values() if len(v[0]) > 25),
                '| below lib_max 0.7 / 0.9:', sum(1 for v in _hb.values() if v[2] < 0.7), '/', sum(1 for v in _hb.values() if v[2] < 0.9),
                '| joined-row popularity lists', _njp, _jst, '| shared engine-2 lists', E6I['shared_engine2'])
            _a1 = _e6_attempt('oursbase', os.path.join(_ew, 'ours_base.json'), len(_hb), [m for m, v in BASE.items() if v[0]], _dl1, _envs)
        except Exception as e:                    # the hand-over itself failed
            _a1 = dict(tag='oursbase', status='failed: ' + repr(e)[:200], where=traceback.format_exc()[-600:]); E6I['attempts'].append(_a1)
    E6I.update(status=_a1['status'], secs=_a1.get('secs', 0), returncode=_a1.get('returncode'), real_rows=_a1.get('real_rows'), handover=_a1.get('handover'))
    if _a1['status'] == 'ok':
        E6I['table'] = _a1['table']; E6I['submission'] = 'e006 improved' if _e6mode == 'ours' else "e006's own table (E006_BASE 'e006')"
    else:
        FAILS.append('e006imp stage: ' + _a1['status'][:220])
        print('=' * 110 + '\nE006IMP STAGE FAILED:', _a1['status'], '\n', _a1.get('where', ''), '\npolicy E006_BASE_ON_FAIL =', _e6fail, '\n' + '=' * 110, flush=True)
        if _e6mode == 'ours' and _e6fail == 'e006':                       # the public notebook untouched, as a second run
            _pc1 = os.path.join(_ew, 'pc_lists.json'); _envf = dict(_envs)
            try:
                if len(json.load(open(_pc1))) >= 0.9 * len(_e6active):
                    shutil.copyfile(_pc1, os.path.join(_ew, 'pc_lists_first_run.json')); _envf['OURS_E006_PC'] = os.path.join(_ew, 'pc_lists_first_run.json')
            except Exception as e:
                print('E006IMP fallback run: no PubChem lists of the first run to reuse:', repr(e)[:150])
            for _d in ('ice_work', 'gl_work'):                             # scores of the first run must not be read by the second
                shutil.rmtree(os.path.join(_ew, _d), ignore_errors=True)
            _a2 = _e6_attempt('fallback', None, 0, _e6active, _hard, _envf)
            if _a2['status'] == 'ok':
                E6I.update(status='ok', table=_a2['table'], fallback='e006 own table', submission="E006'S OWN UNTOUCHED TABLE (fallback: the run with our base lists failed)")
            else:
                FAILS.append('e006imp fallback run: ' + _a2['status'][:220]); E6I['status'] = 'failed twice: ' + _a1['status'][:90] + ' || ' + _a2['status'][:90]
        if E6I['status'] != 'ok' and _e6fail == 'ours':
            E6I['submission'] = 'OUR OWN TABLE (cells 07 / 08 in full; fallback: the stage failed)'
            for _d in ('ice_site', 'work', 'numba_cache', 'eng', 'ice_work', 'gl_work'):      # room on the work disk for our own forward stage
                shutil.rmtree(os.path.join(_ew, _d), ignore_errors=True)
    E6I['stage_secs'] = round(time.time() - _t6)
    resources('after the e006 stage')
    if E6I['status'] == 'ok' and _e6mode == 'ours':   # cells 07 / 08: no forward models, their table is a stand-in that nb/e006imp_final.py replaces
        E6I['cfg_full'] = dict(USE_ICE=CFG['USE_ICE'], USE_GL=CFG['USE_GL'], FRAG_LAM=CFG['FRAG_LAM'])
        CFG.update(USE_ICE=False, USE_GL=False, FRAG_LAM=0.0)
    log('E006IMP stage', {k: v for k, v in E6I.items() if k != 'attempts'})
    dump('e006imp_stage', E6I)
    if E6I['status'] != 'ok' and _e6fail != 'ours':
        log('E006IMP SUBMISSION = NONE: the notebook stops here (E006_BASE_ON_FAIL = %r); no degraded table is written' % _e6fail)
        if os.path.exists(os.path.join(WORK, 'submission.csv')):
            os.remove(os.path.join(WORK, 'submission.csv'))
        raise RuntimeError('E006_BASE stage failed (E006_BASE_ON_FAIL=%s): %s' % (_e6fail, ' | '.join(FAILS)))
