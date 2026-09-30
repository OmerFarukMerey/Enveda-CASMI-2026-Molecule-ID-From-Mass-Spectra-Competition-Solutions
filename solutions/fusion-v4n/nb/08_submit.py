# fusion_core: ICE/GL re-rank -> gated PubChem merge -> engine-2 fusion -> submission.csv
samp = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
MOL_ORDER = list(te.molecule_id.unique())
dump('meta', dict(mol_order=MOL_ORDER, sample_ids=list(samp.molecule_id), is_rerun=IS_RERUN, cfg=CFG))
sub, FSTATS = fusion_core.build_submission(
    BASE, PC, ENG, ICE_SCORES, GL_SCORES, CFG, MOL_ORDER, list(samp.molecule_id),
    ice_fuse=ice_fuse, gl_fuse=gl_fuse, score_key=chem.score_key, formula=formula_of, log=print)
fusion_core.validate(sub, list(samp.molecule_id))
sub.to_csv(os.path.join(WORK, 'submission.csv'), index=False)
json.dump(dict(version=CFG['VERSION'], secs=round(time.time() - T0), stats=FSTATS, n_errors=n_err, families=MAN['families'],
               gpu=torch.cuda.get_device_name(0) if dev == 'cuda' else None, cfg=CFG),
          open(os.path.join(WORK, 'run_manifest.json'), 'w'), indent=1)
log('fusion stats', FSTATS)
log('done', sub.shape)
# the ICE/GLACIER wheel site holds thousands of files: drop it so the version's output stays small to list/download
import shutil; shutil.rmtree(os.path.join(WORK, 'ice_site'), ignore_errors=True)
sub.head()
