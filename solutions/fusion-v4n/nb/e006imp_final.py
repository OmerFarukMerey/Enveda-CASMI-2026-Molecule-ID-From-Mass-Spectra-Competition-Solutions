# "e006 improved", last cell (appended by tools/build_notebook.py ONLY when E006_BASE is set): the table of the stage (nb/e006imp_stage.py)
# becomes submission.csv. Stage not ok and E006_BASE_ON_FAIL 'ours' -> cells 07 / 08 ran in full and their table stays (logged as a fallback).
# Whatever happens, one line 'E006IMP SUBMISSION = ...' names the table that submission.csv holds; a stand-in table is never left as the
# submission (the cell raises instead). Validation runs score every table they hold against the labels on the metric key.
if CFG['E006_BASE']:
    import shutil
    if 'samp' not in globals():                 # E006_BASE 'e006': our cells 03 to 09 are not in the notebook
        samp = pd.read_csv(os.path.join(COMP, 'sample_submission.csv')); sub = None
    _e6tabs = {}
    if sub is not None:                         # cell 08's table: the stand-in without forward models (stage ok) or our own full table
        _e6tabs['cell08_standin' if E6I.get('status') == 'ok' else 'cell08_own'] = {str(m): [x for x in s.split(';') if x] for m, s in zip(sub.molecule_id, sub.smiles)}
        sub.to_csv(os.path.join(WORK, 'submission_standin.csv' if E6I.get('status') == 'ok' else 'submission_ours.csv'), index=False)
    if E6I.get('status') == 'ok':
        try:
            _e6t = e006_stage.read_table(E6I['table'])
            _fs = samp[['molecule_id']].copy(); _fs['smiles'] = [';'.join(_e6t[str(m)]) for m in _fs.molecule_id]
            fusion_core.validate(_fs, list(samp.molecule_id))
            shutil.copyfile(E6I['table'], os.path.join(WORK, 'submission_e006imp.csv'))
            sub = _fs
            sub.to_csv(os.path.join(WORK, 'submission.csv'), index=False)
            E6I['final'] = 'e006imp table written'
        except Exception as e:                  # cannot happen after the stage's own checks; a stand-in must not be submitted in its place
            E6I['final'] = 'failed: ' + repr(e)[:200]; FAILS.append('e006imp final: ' + repr(e)[:200])
            E6I['submission'] = 'NONE (the final cell could not write the stage table)'
            if os.path.exists(os.path.join(WORK, 'submission.csv')):
                os.remove(os.path.join(WORK, 'submission.csv'))
    for _a in E6I.get('attempts', []):          # every table the stage produced, also a rejected one (validation: scored below)
        for _f, _n in ((_a.get('table'), 'stage_' + _a['tag']), (os.path.join(WORK, 'submission_e006_%s_REJECTED.csv' % _a['tag']), 'stage_' + _a['tag'] + '_REJECTED')):
            if _f and os.path.exists(_f):
                _e6tabs[_n] = e006_stage.read_table(_f)
    if os.path.exists(os.path.join(WORK, 'submission.csv')):
        _e6tabs['FINAL_submission'] = e006_stage.read_table(os.path.join(WORK, 'submission.csv'))
    resources('last cell (after cells 07 / 08)')
    log('E006IMP SUBMISSION =', E6I.get('submission'), '| stage status', E6I.get('status'), '| final', E6I.get('final'), '| policy', E6I.get('on_fail'),
        '| attempts', [(a['tag'], a['status'][:80], a.get('secs')) for a in E6I.get('attempts', [])])
    if os.path.exists(os.path.join(STAGE, 'val_labels.csv')):          # validation: every table against the labels, on the metric key
        try:
            _lab = pd.read_csv(os.path.join(STAGE, 'val_labels.csv'))
            _labd = dict(zip(_lab.molecule_id.astype(str), _lab.smiles))
            _K = e006_stage.metric_keys([s for t in _e6tabs.values() for v in t.values() for s in v] + list(_labd.values()),
                                        workdir=os.path.join(WORK, 'e006', 'keys'), workers=4)
            _kf = lambda s: _K[s] if s in _K else e006_stage.metric_key(s)
            _rrs = {}
            for _n, _t in _e6tabs.items():
                _st, _rrs[_n] = e006_stage.mrr_table(_t, _labd, _kf)
                log(f'VALIDATION (metric key) {_n}: MRR@25 = {_st["mrr"]:.4f} | top1 {_st["top1"]:.3f} | hit@25 {_st["hit"]:.3f} | n {_st["n"]}')
            _lm = {str(m): round(float(v[2]), 4) for m, v in BASE.items()} if 'BASE' in globals() else {}
            json.dump(dict(rr=_rrs, lib_max=_lm, val_class=dict(zip(_lab.molecule_id.astype(str), _lab.val_class)) if 'val_class' in _lab else {}),
                      open(os.path.join(STAGE, 'val_rr.json'), 'w'))
        except Exception as e:
            print('E006IMP validation scoring failed:', repr(e))
    for _d in ('ice_site', 'work', 'numba_cache', 'smoke_comp', 'eng', 'ice_work', 'gl_work'):          # large or many-file folders of the script
        shutil.rmtree(os.path.join(WORK, 'e006', _d), ignore_errors=True)
    if CFG['VALIDATION'] and not IS_RERUN:      # validation runs: keep the version's output small (the purged library copy is 3 GB)
        for _d in ('val_comp/train.parquet', 'work', 'numba_cache', 'ice_work', 'gl_work'):
            _p = os.path.join(WORK, _d)
            shutil.rmtree(_p, ignore_errors=True) if os.path.isdir(_p) else (os.remove(_p) if os.path.exists(_p) else None)
    _mp = os.path.join(WORK, 'run_manifest.json')
    _man = json.load(open(_mp)) if os.path.exists(_mp) else dict(version=CFG['VERSION'], cfg=CFG)
    _man.update(e006imp={k: v for k, v in E6I.items()}, secs=round(time.time() - T0))
    json.dump(_man, open(_mp, 'w'), indent=1)
    dump('e006imp', E6I)
    log('E006IMP', E6I.get('status'), '|', E6I.get('final'), '| submission =', E6I.get('submission'))
    log('stage fallbacks:', FAILS if FAILS else 'none')
    if E6I.get('status') == 'ok' and E6I.get('final') != 'e006imp table written':
        raise RuntimeError('E006_BASE: the stage table could not be written; no stand-in is submitted: ' + str(E6I.get('final')))
    if CFG['STRICT'] and FAILS:
        raise RuntimeError('STRICT: ' + ' | '.join(FAILS))
    sub.head()
