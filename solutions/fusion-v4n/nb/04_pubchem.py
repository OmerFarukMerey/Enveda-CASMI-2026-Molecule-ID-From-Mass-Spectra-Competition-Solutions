# PubChem-only channel in its own process (S1 code snapshot + FPNet A+B of casmi26-v3-models)
PC = {}
_pc_env = dict(os.environ, CASMI_PC_N1=str(CFG['PC_N1']))
if CFG['PC_POP_LAM'] > 0:                    # popularity prior inside the PubChem channel (wangpenghua v29)
    try:
        _pc_env.update(CASMI_POP_DIR=os.path.dirname(find('casmi26-pubchem-popularity-prior/**/pc_lsid.npy')),
                       CASMI_POP_LAM=str(CFG['PC_POP_LAM']), CASMI_POP_UNION=str(CFG['PC_POP_UNION']))
        log('PubChem-channel popularity prior', CFG['PC_POP_LAM'], CFG['PC_POP_UNION'], _pc_env['CASMI_POP_DIR'])
    except Exception as e:
        print('PC POPULARITY UNAVAILABLE -> off:', repr(e))
if CFG['USE_PC']:
    try:
        rr = subprocess.run([sys.executable, os.path.join(OURS, 'pc', 'pc_runner.py'), V3CODE, V3MOD, POOL_DIR, PC_DIR,
                             os.path.join(COMP, 'test.parquet'), os.path.join(WORK, 'pc_lists.json'), str(CFG['PC_WORKERS'])],
                            capture_output=True, text=True, timeout=CFG['STAGE_TIMEOUT_S'],
                            env=_pc_env)
        print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
        PC = json.load(open(os.path.join(WORK, 'pc_lists.json')))
    except Exception as e:
        print('PUBCHEM CHANNEL FAILED -> base lists only:', repr(e))
log('pubchem lists', len(PC))
dump('pc', PC)
PC_ALT = {}                                  # VAL_EXTRAS: the same channel with the popularity prior, for local replay
if CFG['VALIDATION'] and CFG['VAL_EXTRAS'] and CFG['USE_PC'] and not CFG['PC_POP_LAM'] > 0 and not IS_RERUN:
    try:
        _env2 = dict(_pc_env, CASMI_POP_DIR=os.path.dirname(find('casmi26-pubchem-popularity-prior/**/pc_lsid.npy')),
                     CASMI_POP_LAM='0.25', CASMI_POP_UNION='200')
        rr = subprocess.run([sys.executable, os.path.join(OURS, 'pc', 'pc_runner.py'), V3CODE, V3MOD, POOL_DIR, PC_DIR,
                             os.path.join(COMP, 'test.parquet'), os.path.join(WORK, 'pc_lists_pop.json'), str(CFG['PC_WORKERS'])],
                            capture_output=True, text=True, timeout=CFG['STAGE_TIMEOUT_S'], env=_env2)
        print(rr.stdout[-800:]); print(rr.stderr[-800:])
        PC_ALT = json.load(open(os.path.join(WORK, 'pc_lists_pop.json')))
    except Exception as e:
        print('PC ALT (popularity) FAILED:', repr(e))
    log('pubchem lists with popularity (extras)', len(PC_ALT))
    dump('pc_pop', PC_ALT)
