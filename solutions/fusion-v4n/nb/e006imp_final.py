# "e006 improved", last cell (appended by tools/build_notebook.py ONLY when E006_BASE is set): the table of the stage (nb/e006imp_stage.py)
# becomes submission.csv. Stage not ok -> nothing happens here: cells 07 / 08 ran in full and their table stays.
if CFG['E006_BASE'] and E6I.get('status') == 'ok':
    import shutil
    try:
        _e6t = e006_stage.read_table(os.path.join(WORK, 'e006', 'submission.csv'))
        _fs = samp[['molecule_id']].copy(); _fs['smiles'] = [';'.join(_e6t[str(m)]) for m in _fs.molecule_id]
        fusion_core.validate(_fs, list(samp.molecule_id))
        sub.to_csv(os.path.join(WORK, 'submission_standin.csv'), index=False)      # cell 08's table without forward models (not the submission)
        shutil.copyfile(os.path.join(WORK, 'e006', 'submission.csv'), os.path.join(WORK, 'submission_e006imp.csv'))
        sub = _fs
        sub.to_csv(os.path.join(WORK, 'submission.csv'), index=False)
        E6I['final'] = 'e006imp table written'
    except Exception as e:                      # cannot happen after the stage's own checks; the stand-in table then stays (valid, weaker)
        E6I['final'] = 'failed: ' + repr(e)[:200]
        print('E006IMP FINAL FAILED -> the stand-in table of cell 08 stays:', repr(e)); FAILS.append('e006imp final: ' + repr(e)[:200])
    for _d in ('ice_site', 'work', 'numba_cache', 'smoke_comp', 'eng'):          # large or many-file folders of the script
        shutil.rmtree(os.path.join(WORK, 'e006', _d), ignore_errors=True)
    _man = json.load(open(os.path.join(WORK, 'run_manifest.json'))); _man.update(e006imp={k: v for k, v in E6I.items()}, secs=round(time.time() - T0))
    json.dump(_man, open(os.path.join(WORK, 'run_manifest.json'), 'w'), indent=1)
    dump('e006imp', E6I)
elif CFG['E006_BASE']:
    dump('e006imp', E6I)
if CFG['E006_BASE']:
    log('E006IMP', E6I.get('status'), '|', E6I.get('final'), '| submission =', 'e006 improved' if E6I.get('final') == 'e006imp table written' else 'OUR OWN TABLE (cells 07 / 08)')
    log('stage fallbacks:', FAILS if FAILS else 'none')
    if CFG['STRICT'] and FAILS:
        raise RuntimeError('STRICT: ' + ' | '.join(FAILS))
sub.head()
