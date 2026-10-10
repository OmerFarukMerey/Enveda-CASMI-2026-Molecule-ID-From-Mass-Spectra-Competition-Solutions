# fusion_core: ICE/GL re-rank -> gated PubChem merge -> engine-2 fusion -> submission.csv
samp = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
MOL_ORDER = list(te.molecule_id.unique())
dump('meta', dict(mol_order=MOL_ORDER, sample_ids=list(samp.molecule_id), is_rerun=IS_RERUN, cfg=CFG))
assert not (CFG['NP_MU'] or CFG['NP_JOIN_MU']), 'NP_MU / NP_JOIN_MU: no NP-likeness scorer is shipped (reviews/nplike-2026-10-08.md)'
POOL_POP = None
if CFG['POP_MU'] > 0 or (CFG['POP_GROUP_MU'] or 0) > 0:
    try:                                  # log1p(PubChem SIDs) + log1p(PubMed ids) per v2-pool row (pid order)
        _pp = os.path.dirname(find('casmi26-pubchem-popularity-prior/**/pool_lsid.npy'))
        POOL_POP = np.load(os.path.join(_pp, 'pool_lsid.npy')).astype(np.float64) + \
            np.load(os.path.join(_pp, 'pool_lpmid.npy')).astype(np.float64)
        log('popularity prior loaded', _pp, POOL_POP.shape)
    except Exception as e:
        print('POPULARITY PRIOR UNAVAILABLE -> off:', repr(e))
    if CFG['POP_SPLIT'] and POOL_POP is not None:
        try:                              # substance count without spectral-library deposits on the rows of casmi26-pop-split
            import pop_split
            _ps = os.path.dirname(find('casmi26-pop-split/**/pool_split.parquet'))
            POOL_POP, _n = pop_split.pool_pop(np.load(os.path.join(_pp, 'pool_lsid.npy')), np.load(os.path.join(_pp, 'pool_lpmid.npy')), _ps)
            log('popularity prior: library deposits removed on', _n, 'pool rows', _ps)
        except Exception as e:
            print('POP_SPLIT UNAVAILABLE -> standard prior:', repr(e)); FAILS.append('pop split: ' + repr(e)[:200])
ADD_SCORE = None
if CFG['ADD_JOIN_MU'] > 0:
    try:                                  # adduct-evidence scorer: models of casmi26-ours-adduct-prior + the adducts each molecule was acquired with
        import adduct_prior
        _ad = os.path.dirname(find('casmi26-ours-adduct-prior/**/adduct_H.txt'))
        ADD_SCORE = adduct_prior.load(_ad, te.groupby('molecule_id').adduct.agg(lambda x: sorted(set(map(str, x)))).to_dict())
        log('adduct-evidence scorer loaded', _ad, len(ADD_SCORE.y), 'molecules')
    except Exception as e:
        print('ADDUCT-EVIDENCE SCORER UNAVAILABLE -> off:', repr(e)); FAILS.append('adduct-evidence scorer: ' + repr(e)[:200])
sub, FSTATS = fusion_core.build_submission(
    BASE, PC, ENG, ICE_SCORES, GL_SCORES, CFG, MOL_ORDER, list(samp.molecule_id),
    ice_fuse=ice_fuse, gl_fuse=gl_fuse, score_key=chem.score_key, formula=formula_of, log=print, pool_pop=POOL_POP,
    frag_scores=FRAG, add_score=ADD_SCORE,
    nostereo_key=fusion_core.make_nostereo_key(chem.score_key) if (CFG['KEY_NOSTEREO'] or CFG['FUSE_KEY_VOTE']) else None)
fusion_core.validate(sub, list(samp.molecule_id))
if CFG['GEN_ANALOG']:                      # analog generator channel: new structures placed into the final lists (slots or gated merge)
    try:
        _gp = gen_analog.PoolIndex(P.smiles, P.key, P.formula, P.fp_packed)
        _gk = {}                           # SMILES -> metric key where a stage already computed it
        for _b in BASE.values():
            _gk.update(zip(_b[0], _b[1]))
        for _p in list(PC.values()) + list(PC_DEEP.values()):
            _gk.update(zip(_p.get('pc') or [], _p.get('pc_keys') or []))
        for _e in ENG.values():
            _gk.update(zip(_e.get('smiles') or [], _e.get('keys') or []))
        _gl, GEN_STATS = gen_analog.run_channel(
            list(sub.molecule_id), {m: [x for x in s.split(';') if x] for m, s in zip(sub.molecule_id, sub.smiles)}, BASE, GEN_Z, GEN_S1, _gp,
            CFG['GEN_ANALOG'], pos=CFG['GEN_ANALOG_POS'], k=CFG['GEN_ANALOG_K'], fp_fn=lambda s: (lambda f: None if f is None else f[P.bits])(chem.raw_fingerprint(s)),
            key_fn=chem.score_key, known_keys=_gk, strict=not IS_RERUN, force_gate=not (IS_RERUN or CFG['VALIDATION']), log=print)
        _gs = sub.copy(); _gs['smiles'] = [';'.join(_gl[m]) for m in _gs.molecule_id]
        fusion_core.validate(_gs, list(samp.molecule_id)); sub = _gs
        log('GEN_ANALOG', GEN_STATS); FSTATS['gen_analog'] = GEN_STATS; dump('gen_analog', GEN_STATS)
        if IS_RERUN and (GEN_STATS['failed'] > max(2, 0.02 * len(sub)) or (CFG['GEN_ANALOG'] == 'slots' and GEN_STATS['changed'] == 0)):
            FAILS.append(f'analog generator: {GEN_STATS}'[:300])
    except Exception as e:                 # scoring rerun: the fused lists stay as they are
        if not IS_RERUN:
            raise
        print('GEN_ANALOG CHANNEL FAILED -> lists unchanged', repr(e)); FAILS.append('analog generator channel: ' + repr(e)[:200])
if CFG['OUT_TOPK']:                        # diagnostic probe: keep only the first k rows of every list
    sub['smiles'] = [';'.join(x.split(';')[:int(CFG['OUT_TOPK'])]) for x in sub.smiles]
    log('OUT_TOPK: lists cut to', CFG['OUT_TOPK'])
sub.to_csv(os.path.join(WORK, 'submission.csv'), index=False)
json.dump(dict(version=CFG['VERSION'], secs=round(time.time() - T0), stats=FSTATS, n_errors=n_err, families=MAN['families'],
               gpu=torch.cuda.get_device_name(0) if dev == 'cuda' else None, cfg=CFG),
          open(os.path.join(WORK, 'run_manifest.json'), 'w'), indent=1)
log('fusion stats', FSTATS)
log('done', sub.shape)
if os.path.exists(os.path.join(STAGE, 'val_labels.csv')):   # validation mode: score with the metric's key
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _taut = rdMolStandardize.TautomerEnumerator()
    def _ik14(s):
        try:
            return _Chem.MolToInchiKey(_taut.Canonicalize(_Chem.MolFromSmiles(s)))[:14]
        except Exception:
            return None
    _lab = pd.read_csv(os.path.join(STAGE, 'val_labels.csv')).set_index('molecule_id')
    _rr = []
    for _m, _s in zip(sub.molecule_id, sub.smiles):
        _t = _ik14(_lab.smiles[_m]); _r = 0.0
        for _i, _g in enumerate(_s.split(';'), 1):
            if _ik14(_g) == _t:
                _r = 1.0 / _i; break
        _rr.append(_r)
    log(f'VALIDATION MRR@25 = {np.mean(_rr):.4f} | top1 {np.mean(np.array(_rr) == 1):.3f} | hit@25 {np.mean(np.array(_rr) > 0):.3f} | n {len(_rr)}')
# the ICE/GLACIER wheel site holds thousands of files: drop it so the version's output stays small to list/download
import shutil; shutil.rmtree(os.path.join(WORK, 'ice_site'), ignore_errors=True)
shutil.rmtree(os.path.join(WORK, 'ice_pkg_ft'), ignore_errors=True)
log('stage fallbacks:', FAILS if FAILS else 'none')
if CFG['STRICT'] and FAILS:
    raise RuntimeError('STRICT: ' + ' | '.join(FAILS))
sub.head()
