# ===== VH: output hygiene - keep vh/, small top-level files and audits; delete work and temp folders so the output lists in one page =====
try:
    import shutil as _cl_sh
    from pathlib import Path as _CP
    _keep_dirs = {'vh'}
    _w = _CP('/kaggle/working')
    _removed = []
    for _p in sorted(_w.iterdir()):
        if _p.is_dir() and _p.name not in _keep_dirs:
            _cl_sh.rmtree(_p, ignore_errors=True); _removed.append(_p.name + '/')
        elif _p.is_file() and (_p.stat().st_size > 50 * 1024 * 1024 or _p.suffix not in ('.csv', '.json', '.parquet', '.log', '.txt')):
            _p.unlink(); _removed.append(_p.name)
    _cl_sh.rmtree('/tmp/enveda-casmi26-vh2-mass-spectra', ignore_errors=True)
    _left = [p for p in _w.rglob('*') if p.is_file()]
    print('[vh] cleanup: removed', len(_removed), 'entries; files left', len(_left), flush=True)
except Exception as _cl_e:
    print('[vh] cleanup failed:', repr(_cl_e)[:300], flush=True)
