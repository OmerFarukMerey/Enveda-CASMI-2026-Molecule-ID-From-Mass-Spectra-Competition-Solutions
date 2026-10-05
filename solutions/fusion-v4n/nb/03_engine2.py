# engine-2 (prvsiyan two-ranker + BIO + AFIX) in a subprocess -> ENG {mid: {smiles, keys}}
ENG = {}
if CFG['USE_ENG']:
    try:
        _t0 = time.time()
        _env = dict(os.environ, CASMI_ENG_CFG=json.dumps(dict(
            W_A=CFG['ENG_W_A'], SEEDS=CFG['ENG_SEEDS'], N_ANALOG=CFG['ENG_N_ANALOG'], W_PV=CFG['ENG_W_PV'],
            ORDER_K=CFG['ENG_ORDER_K'], TOPK=CFG['ENG_TOPK'])))
        rr = subprocess.run([sys.executable, os.path.join(OURS, 'eng', 'eng_runner.py'), os.path.join(COMP, 'test.parquet'),
                             os.path.join(COMP, 'train.parquet'), os.path.join(COMP, 'sample_submission.csv'),
                             os.path.join(WORK, 'eng_lists.json')], capture_output=True, text=True,
                            timeout=CFG['STAGE_TIMEOUT_S'], cwd=os.path.join(OURS, 'eng'), env=_env)
        print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
        ENG = json.load(open(os.path.join(WORK, 'eng_lists.json')))
        log('engine-2 lists', len(ENG), f'{time.time()-_t0:.0f}s')
    except Exception as e:
        print('ENGINE-2 FAILED -> v4 lists kept:', repr(e)); FAILS.append('engine-2: ' + repr(e)[:200])
dump('eng', ENG)
