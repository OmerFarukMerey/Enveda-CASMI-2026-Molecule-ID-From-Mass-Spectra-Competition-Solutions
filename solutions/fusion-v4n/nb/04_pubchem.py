# PubChem-only channel in its own process (S1 code snapshot + FPNet A+B of casmi26-v3-models)
PC = {}
_pc_env = dict(os.environ, CASMI_PC_N1=str(CFG['PC_N1']), CASMI_PC_K=str(max(25, CFG['PC_JOIN_N'], CFG['PC_JOIN_N_DEEP'])),
               CASMI_PC_PPM=str(CFG['PC_PPM']), CASMI_MASS_OFFSET=json.dumps(CFG['MASS_OFFSET_BY_ADDUCT']))
if CFG['PC_POP_LAM'] > 0:                    # popularity prior inside the PubChem channel (wangpenghua v29)
    try:
        _pc_env.update(CASMI_POP_DIR=os.path.dirname(find('casmi26-pubchem-popularity-prior/**/pc_lsid.npy')),
                       CASMI_POP_LAM=str(CFG['PC_POP_LAM']), CASMI_POP_UNION=str(CFG['PC_POP_UNION']))
        log('PubChem-channel popularity prior', CFG['PC_POP_LAM'], CFG['PC_POP_UNION'], _pc_env['CASMI_POP_DIR'])
    except Exception as e:
        print('PC POPULARITY UNAVAILABLE -> off:', repr(e))
PCMOD = V3MOD
if CFG['USE_PC']:
    try:
        rr = subprocess.run([sys.executable, os.path.join(OURS, 'pc', 'pc_runner.py'), V3CODE, PCMOD, POOL_DIR, PC_DIR,
                             os.path.join(COMP, 'test.parquet'), os.path.join(WORK, 'pc_lists.json'), str(CFG['PC_WORKERS'])],
                            capture_output=True, text=True, timeout=CFG['STAGE_TIMEOUT_S'],
                            env=_pc_env)
        print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
        PC = json.load(open(os.path.join(WORK, 'pc_lists.json')))
    except Exception as e:
        print('PUBCHEM CHANNEL FAILED -> base lists only:', repr(e)); FAILS.append('pubchem channel: ' + repr(e)[:200])
PC_DEEP = {}
if CFG['PC_JOIN_N'] > 0:                     # the join (nb/06) reads the deep list; the fusion sees the first 25 as before
    for _m, _p in PC.items():
        PC_DEEP[_m] = dict(pc=list(_p['pc']), pc_fz=list(_p['pc_fz']), pc_keys=list(_p['pc_keys']), best_pool_fz=_p.get('best_pool_fz'))
        for _k in ('pc', 'pc_fz', 'pc_keys'):
            _p[_k] = _p[_k][:25]
    if CFG['PC_MODELS'] == 'hn' and CFG['USE_PC']:   # the join's candidates from the fine-tuned model (own logit scale:
        try:                                         #   only the order of its list is used)
            _hn = os.path.dirname(find('casmi26-fpnet-pc-hn/**/fpnet_0.pt'))
            rr = subprocess.run([sys.executable, os.path.join(OURS, 'pc', 'pc_runner.py'), V3CODE, _hn, POOL_DIR, PC_DIR,
                                 os.path.join(COMP, 'test.parquet'), os.path.join(WORK, 'pc_lists_hn.json'), str(CFG['PC_WORKERS'])],
                                capture_output=True, text=True, timeout=CFG['STAGE_TIMEOUT_S'], env=dict(_pc_env, CASMI_PC_MODELS_ANY='1'))
            print(rr.stdout[-1500:]); print(rr.stderr[-1500:])
            _hnl = json.load(open(os.path.join(WORK, 'pc_lists_hn.json')))
            assert len(_hnl) >= 0.9 * len(PC_DEEP), (len(_hnl), len(PC_DEEP))
            _n_hn = 0
            for _m, _p in _hnl.items():
                if _m in PC_DEEP and _p.get('pc'):
                    PC_DEEP[_m].update(pc=list(_p['pc']), pc_fz=list(_p['pc_fz']), pc_keys=list(_p['pc_keys'])); _n_hn += 1
            log('PC_MODELS hn: join candidates replaced for', _n_hn, 'of', len(PC_DEEP), 'molecules', _hn, sorted(os.listdir(_hn)))
        except Exception as e:
            print('PC_MODELS hn FAILED -> join candidates of A+B kept:', repr(e)); FAILS.append('pc hn pass: ' + repr(e)[:200])
    dump('pc_deep', PC_DEEP)
log('pubchem lists', len(PC), '| deep lists for the join', len(PC_DEEP))
dump('pc', PC)
PC_POP = {}                                  # PC_PROMOTE: the popularity-ordered channel of the public line next to ours (own file)
if CFG['PC_PROMOTE'] is not None and CFG['USE_PC']:
    try:
        _env3 = dict(_pc_env, CASMI_POP_DIR=os.path.dirname(find('casmi26-pubchem-popularity-prior/**/pc_lsid.npy')),
                     CASMI_POP_LAM='0.25', CASMI_POP_UNION='200', CASMI_PC_K='25')
        rr = subprocess.run([sys.executable, os.path.join(OURS, 'pc', 'pc_runner.py'), V3CODE, PCMOD, POOL_DIR, PC_DIR,
                             os.path.join(COMP, 'test.parquet'), os.path.join(WORK, 'pc_lists_promote.json'), str(CFG['PC_WORKERS'])],
                            capture_output=True, text=True, timeout=CFG['STAGE_TIMEOUT_S'], env=_env3)
        print(rr.stdout[-1500:]); print(rr.stderr[-1500:])
        PC_POP = json.load(open(os.path.join(WORK, 'pc_lists_promote.json')))
        assert len(PC_POP) >= 0.9 * len(PC), (len(PC_POP), len(PC))
    except Exception as e:
        PC_POP = {}
        print('PC PROMOTE CHANNEL FAILED -> no promotion:', repr(e)); FAILS.append('pc promote pass: ' + repr(e)[:200])
    log('PC_PROMOTE: popularity-ordered lists', len(PC_POP))
    dump('pc_promote', PC_POP)
PC_ALT = {}                                  # VAL_EXTRAS: the same channel with the popularity prior, for local replay
if CFG['VALIDATION'] and CFG['VAL_EXTRAS'] and CFG['USE_PC'] and not CFG['PC_POP_LAM'] > 0 and not IS_RERUN:
    try:
        _env2 = dict(_pc_env, CASMI_POP_DIR=os.path.dirname(find('casmi26-pubchem-popularity-prior/**/pc_lsid.npy')),
                     CASMI_POP_LAM='0.25', CASMI_POP_UNION='200')
        rr = subprocess.run([sys.executable, os.path.join(OURS, 'pc', 'pc_runner.py'), V3CODE, PCMOD, POOL_DIR, PC_DIR,
                             os.path.join(COMP, 'test.parquet'), os.path.join(WORK, 'pc_lists_pop.json'), str(CFG['PC_WORKERS'])],
                            capture_output=True, text=True, timeout=CFG['STAGE_TIMEOUT_S'], env=_env2)
        print(rr.stdout[-800:]); print(rr.stderr[-800:])
        PC_ALT = json.load(open(os.path.join(WORK, 'pc_lists_pop.json')))
    except Exception as e:
        print('PC ALT (popularity) FAILED:', repr(e))
    log('pubchem lists with popularity (extras)', len(PC_ALT))
    dump('pc_pop', PC_ALT)
if PC_POP and not PC_ALT:                    # PC_PROMOTE: cell 07 sends the rows of PC_ALT to the forward models (its VAL_EXTRAS loop)
    PC_ALT = PC_POP
