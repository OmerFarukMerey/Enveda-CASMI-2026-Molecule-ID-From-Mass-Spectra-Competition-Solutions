# fusion_core: ICE/GL re-rank -> gated PubChem merge -> engine-2 fusion -> submission.csv
samp = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
MOL_ORDER = list(te.molecule_id.unique())
dump('meta', dict(mol_order=MOL_ORDER, sample_ids=list(samp.molecule_id), is_rerun=IS_RERUN, cfg=CFG))
POOL_POP = None
if CFG['POP_MU'] > 0:
    try:                                  # log1p(PubChem SIDs) + log1p(PubMed ids) per v2-pool row (pid order)
        _pp = os.path.dirname(find('casmi26-pubchem-popularity-prior/**/pool_lsid.npy'))
        POOL_POP = np.load(os.path.join(_pp, 'pool_lsid.npy')).astype(np.float64) + \
            np.load(os.path.join(_pp, 'pool_lpmid.npy')).astype(np.float64)
        log('popularity prior loaded', _pp, POOL_POP.shape)
    except Exception as e:
        print('POPULARITY PRIOR UNAVAILABLE -> off:', repr(e))
sub, FSTATS = fusion_core.build_submission(
    BASE, PC, ENG, ICE_SCORES, GL_SCORES, CFG, MOL_ORDER, list(samp.molecule_id),
    ice_fuse=ice_fuse, gl_fuse=gl_fuse, score_key=chem.score_key, formula=formula_of, log=print, pool_pop=POOL_POP,
    frag_scores=FRAG)
fusion_core.validate(sub, list(samp.molecule_id))
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
sub.head()
