# second pipeline (E006_FUSE; off by default: this cell then does nothing). The public notebook "CASMI26 E006 Window 40 ICE Fallback True"
# (canqiang, Apache-2.0; snapshot and notice in reference/e006/) runs as a script in /kaggle/working/e006 AFTER our own submission.csv exists;
# the two tables are fused on the metric key (nb/embed/e006_stage.py). Any failure, or a run past E006_DEADLINE_S, keeps our own table.
E006_STATS = None
if CFG['E006_FUSE']:
    import shutil
    import e006_stage
    assert CFG['E006_FUSE'] in e006_stage.RULES, CFG['E006_FUSE']
    assert not CFG['VALIDATION'] and not CFG['OUT_TOPK'], 'E006_FUSE is not meant for validation or probe runs'
    _ew = os.path.join(WORK, 'e006'); os.makedirs(_ew, exist_ok=True)
    _t9 = time.time()
    _ours = {str(m): [x for x in s.split(';') if x] for m, s in zip(sub.molecule_id, sub.smiles)}
    sub.to_csv(os.path.join(WORK, 'submission_ours.csv'), index=False)       # the first pipeline's table, as cell 08 wrote it
    _theirs, _e6 = {}, dict(status='not run', secs=0, shared_engine2=False)
    try:
        _cells = json.load(open(os.path.join(OURS, 'e006', 'cells.json')))
        _script = os.path.join(_ew, 'e006_main.py')
        open(_script, 'w').write(e006_stage.script_from_cells(_cells, _ew))
        _env = dict(os.environ)
        _e6['shared_engine2'] = bool(CFG['E006_SHARE_ENG']) and e006_stage.engine_shareable(CFG) and bool(ENG) and \
            not any(f.startswith('engine-2') for f in FAILS)
        if _e6['shared_engine2']:
            _env['OURS_E006_ENG'] = os.path.join(WORK, 'eng_lists.json')
        if CFG['FULL_ON_COMMIT'] and not IS_RERUN:
            _env['OURS_E006_FULL'] = '1'
        gc.collect(); torch.cuda.empty_cache()
        _left = CFG['E006_DEADLINE_S'] - (time.time() - T0)
        log('E006 stage: rule', CFG['E006_FUSE'], '| shared engine-2 lists', _e6['shared_engine2'], f'| time left before the deadline {_left:.0f}s')
        if _left < 900:
            raise RuntimeError(f'only {_left:.0f}s left before E006_DEADLINE_S: not started')
        _rc, _secs, _st = e006_stage.run_script(_script, _ew, _env, _left, os.path.join(_ew, 'e006.log'))
        _e6.update(status=_st, secs=round(_secs), returncode=_rc)
        print(open(os.path.join(_ew, 'e006.log'), errors='replace').read()[-30000:])
        if _st != 'ok':
            raise RuntimeError(f'second pipeline {_st} after {_secs:.0f}s (return code {_rc})')
        _theirs = e006_stage.read_table(os.path.join(_ew, 'submission.csv'))
        assert set(_theirs) == set(_ours), (len(_theirs), len(_ours))
        shutil.copyfile(os.path.join(_ew, 'submission.csv'), os.path.join(WORK, 'submission_e006.csv'))
    except Exception as e:
        _theirs = {}
        print('E006 STAGE FAILED -> our own table kept:', repr(e)); FAILS.append('e006 stage: ' + repr(e)[:200])
    _use = [m for m in _ours if not e006_stage.is_placeholder(_theirs.get(m) or [])]          # molecules with a second list
    _t9k = time.time()
    _K = e006_stage.metric_keys([s for m in _use for s in _ours[m] + _theirs[m]], workdir=os.path.join(_ew, 'keys'), workers=4)
    _key = lambda s: _K[s] if s in _K else e006_stage.metric_key(s)                           # the official metric's key, from the SMILES
    log('E006 fusion: metric keys for', len(_K), 'rows of', len(_use), f'molecules in {time.time() - _t9k:.0f}s')
    _fused, E006_STATS = e006_stage.fuse_tables(_ours, _theirs, _key, CFG['E006_FUSE'], rrf_k=CFG['E006_RRF_K'], fill_pos=CFG['E006_FILL_POS'],
                                                rrf_w=CFG['E006_RRF_W'])
    E006_STATS.update(_e6, fuse_secs=round(time.time() - _t9 - _e6['secs']))
    _fs = sub.copy(); _fs['smiles'] = [';'.join(_fused[str(m)]) for m in _fs.molecule_id]
    fusion_core.validate(_fs, list(samp.molecule_id))
    assert _theirs or (_fs.smiles.values == sub.smiles.values).all()         # no second table: ours byte for byte
    sub = _fs
    sub.to_csv(os.path.join(WORK, 'submission.csv'), index=False)
    for _d in ('ice_site', 'work', 'numba_cache', 'smoke_comp', 'eng'):      # large or many-file folders of the second pipeline
        shutil.rmtree(os.path.join(_ew, _d), ignore_errors=True)
    _man = json.load(open(os.path.join(WORK, 'run_manifest.json'))); _man.update(e006=E006_STATS, secs=round(time.time() - T0))
    json.dump(_man, open(os.path.join(WORK, 'run_manifest.json'), 'w'), indent=1)
    dump('e006', E006_STATS)
    log('E006 fusion', E006_STATS)
    log('stage fallbacks:', FAILS if FAILS else 'none')
    if CFG['STRICT'] and FAILS:
        raise RuntimeError('STRICT: ' + ' | '.join(FAILS))
sub.head()
