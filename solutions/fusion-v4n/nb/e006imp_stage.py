# "e006 improved" (E006_BASE = 'ours'; this cell is put after cell 06 by tools/build_notebook.py ONLY when the option is set). The public notebook
# "CASMI26 E006 Window 40 ICE Fallback True" (canqiang, Apache-2.0; snapshot and notice in reference/e006/) runs as a script in
# /kaggle/working/e006 with OUR base lists in place of its own engine pass (nb/embed/e006_stage.py, patch 4): its PubChem channel with the
# popularity prior, slots, promotion, forward models, engine fusion, tail fill, key vote and overlay are its own cells. Its table becomes the
# submission (nb/e006imp_final.py). Stage ok -> cells 07 / 08 skip the forward models (their table is only a stand-in). Any failure, a bad
# table or a run past E006_BASE_DEADLINE_S -> nothing is switched off and cells 07 / 08 produce our own table in full.
E6I = dict(status='off')
if CFG['E006_BASE']:
    import gc
    import e006_stage
    assert CFG['E006_BASE'] == 'ours', CFG['E006_BASE']
    assert not CFG['E006_FUSE'] and not CFG['VALIDATION'] and not CFG['OUT_TOPK'] and not CFG['GEN_ANALOG'], 'E006_BASE: not with these options'
    _ew = os.path.join(WORK, 'e006'); os.makedirs(_ew, exist_ok=True)
    _t6 = time.time(); E6I = dict(status='not run', secs=0, shared_engine2=False)
    try:
        _jp, _jst = None, None
        if CFG['E006_JOIN_POP_MU'] > 0:      # popularity of the joined rows (the table POP_MU reads; nb/embed/pc_pop_side.py)
            import pc_pop_side
            _tp = pc_pop_side.TierPop(PC_DIR, os.path.dirname(find('casmi26-pubchem-popularity-prior/**/pc_lsid.npy')))
            _jp, _jst = pc_pop_side.join_pop_table(BASE, {m: p.get('target') for m, p in PC.items()}, _tp)
            assert not _jst['rows'] or _jst['found'] >= 0.9 * _jst['rows'], _jst
        _hb, _njp = e006_stage.base_for_handover(BASE, _jp, CFG['E006_JOIN_POP_MU'], fusion_core.popularity_reorder_join)
        assert len(_hb) >= 0.98 * len(BASE) - 1, (len(_hb), len(BASE))
        json.dump(_hb, open(os.path.join(_ew, 'ours_base.json'), 'w'))
        _cells = json.load(open(os.path.join(OURS, 'e006', 'cells.json')))
        _script = os.path.join(_ew, 'e006_main.py')
        open(_script, 'w').write(e006_stage.script_improved(_cells, _ew))
        _env = dict(os.environ, OURS_E006_BASE=os.path.join(_ew, 'ours_base.json'), **e006_stage.improved_env(CFG))
        E6I['shared_engine2'] = bool(CFG['E006_SHARE_ENG']) and e006_stage.engine_shareable(CFG) and bool(ENG) and \
            not any(f.startswith('engine-2') for f in FAILS)
        if E6I['shared_engine2']:
            _env['OURS_E006_ENG'] = os.path.join(WORK, 'eng_lists.json')
        if CFG['FULL_ON_COMMIT'] and not IS_RERUN:
            _env['OURS_E006_FULL'] = '1'
        gc.collect(); torch.cuda.empty_cache()
        _left = CFG['E006_BASE_DEADLINE_S'] - (time.time() - T0)
        log('E006IMP stage: base lists handed over', len(_hb), 'of', len(BASE), '| joined-row popularity lists', _njp, _jst,
            '| options', e006_stage.improved_env(CFG), '| shared engine-2 lists', E6I['shared_engine2'], f'| time left before the deadline {_left:.0f}s')
        if _left < 1800:
            raise RuntimeError(f'only {_left:.0f}s left before E006_BASE_DEADLINE_S: not started')
        _rc, _secs, _st = e006_stage.run_script(_script, _ew, _env, _left, os.path.join(_ew, 'e006.log'))
        E6I.update(status=_st, secs=round(_secs), returncode=_rc)
        _elog = open(os.path.join(_ew, 'e006.log'), errors='replace').read()
        print(_elog[-40000:])
        if _st != 'ok':
            raise RuntimeError(f'the script {_st} after {_secs:.0f}s (return code {_rc})')
        _e6t = e006_stage.read_table(os.path.join(_ew, 'submission.csv'))
        _sid = list(pd.read_csv(os.path.join(COMP, 'sample_submission.csv')).molecule_id)
        E6I['real_rows'] = e006_stage.check_table(_e6t, _sid, [m for m, v in BASE.items() if v[0]])
        _used = [l for l in _elog.split('\n') if l.startswith('E006_STAGE base lists used:')]
        assert _used and int(_used[-1].split(':')[1].split('of')[0]) >= 0.98 * len(_hb), _used     # the hand-over really took place
        E6I['handover'] = _used[-1]
        for _bad in ('ICE FAILED', 'GL FAILED', 'PUBCHEM CHANNEL FAILED', 'ENGINE FAILED'):                 # a stage of ITS pipeline fell back
            if _bad in _elog:
                raise RuntimeError('stage fallback inside the script: ' + _bad)
    except Exception as e:
        import traceback; E6I['status'] = 'failed: ' + repr(e)[:200]; E6I['where'] = traceback.format_exc()[-600:]
        print('E006IMP STAGE FAILED -> cells 07 / 08 run in full, our own table is the submission:', repr(e)); FAILS.append('e006imp stage: ' + repr(e)[:200])
    E6I['stage_secs'] = round(time.time() - _t6)
    if E6I['status'] == 'ok':                 # cells 07 / 08: no forward models, their table is a stand-in that nb/e006imp_final.py replaces
        E6I['cfg_full'] = dict(USE_ICE=CFG['USE_ICE'], USE_GL=CFG['USE_GL'], FRAG_LAM=CFG['FRAG_LAM'])
        CFG.update(USE_ICE=False, USE_GL=False, FRAG_LAM=0.0)
    log('E006IMP stage', {k: v for k, v in E6I.items()})
