# engine-2 (prvsiyan two-ranker + BIO + AFIX) in a subprocess -> ENG {mid: {smiles, keys}}
ENG = {}
if CFG['USE_ENG']:
    _extra = []
    if CFG['ENG_EXTRA_FPNET']:   # an FPNet averaged into engine-2's fingerprint logits. Resolved outside the try: a commit
        _slug = {'np-ours': 'casmi26-fp-np-ours'}.get(str(CFG['ENG_EXTRA_FPNET']), str(CFG['ENG_EXTRA_FPNET']).split('/')[-1])   # run without the dataset must fail, not fall back
        _extra = sorted(glob.glob(f'/kaggle/input/**/{_slug}/**/fp_*.pt', recursive=True))
        log('engine-2 extra FPNet', CFG['ENG_EXTRA_FPNET'], [(p, sha256(p)[:16]) for p in _extra])   # npx4 soup: 4f4c090aefd4c697
        if not _extra:
            assert IS_RERUN, f'ENG_EXTRA_FPNET: no fp_*.pt under a folder named {_slug} in /kaggle/input'
            FAILS.append('engine-2 extra FPNet not found: ' + _slug)
    try:
        _t0 = time.time()
        _env = dict(os.environ, CASMI_ENG_CFG=json.dumps(dict(
            W_A=CFG['ENG_W_A'], SEEDS=CFG['ENG_SEEDS'], N_ANALOG=CFG['ENG_N_ANALOG'], W_PV=CFG['ENG_W_PV'],
            ORDER_K=CFG['ENG_ORDER_K'], TOPK=CFG['ENG_TOPK'], EXTRA_FP=_extra, REPS=CFG['ENG_REPS'])))
        rr = subprocess.run([sys.executable, os.path.join(OURS, 'eng', 'eng_runner.py'), os.path.join(COMP, 'test.parquet'),
                             os.path.join(COMP, 'train.parquet'), os.path.join(COMP, 'sample_submission.csv'),
                             os.path.join(WORK, 'eng_lists.json')], capture_output=True, text=True,
                            timeout=CFG['STAGE_TIMEOUT_S'], cwd=os.path.join(OURS, 'eng'), env=_env)
        print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
        ENG = json.load(open(os.path.join(WORK, 'eng_lists.json')))
        if _extra and not all(os.path.basename(p) in rr.stdout for p in _extra):     # the runner prints the models it loads
            FAILS.append('engine-2 ran without the extra FPNet')
        log('engine-2 lists', len(ENG), f'{time.time()-_t0:.0f}s')
    except Exception as e:
        print('ENGINE-2 FAILED -> v4 lists kept:', repr(e)); FAILS.append('engine-2: ' + repr(e)[:200])
dump('eng', ENG)
