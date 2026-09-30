# SMOKE switch: the commit run (visible test = train examples) scores SMOKE_N molecules unless FULL_ON_COMMIT;
# the scoring rerun (hidden test) always runs everything.
import pandas as pd
_te = pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=['molecule_id', 'spectrum_id'])
_sig = hashlib.md5(','.join(sorted(_te.spectrum_id.astype(str))).encode()).hexdigest()
IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'
ICE_BUDGET = CFG['ICE_BUDGET']
if not IS_RERUN and not CFG['FULL_ON_COMMIT']:
    SM = os.path.join(WORK, 'smoke_comp'); os.makedirs(SM, exist_ok=True)
    _full = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
    _keep = sorted(_full.molecule_id.unique())[:CFG['SMOKE_N']]
    _full[_full.molecule_id.isin(_keep)].to_parquet(os.path.join(SM, 'test.parquet'))
    for _f in ['train.parquet', 'sample_submission.csv']:
        if not os.path.exists(os.path.join(SM, _f)):
            os.symlink(os.path.join(COMP, _f), os.path.join(SM, _f))
    COMP = SM
    ICE_BUDGET = CFG['SMOKE_ICE_BUDGET']
log('RERUN', IS_RERUN, '| sig', _sig, '| COMP', COMP, '| ICE_BUDGET', ICE_BUDGET, '| FULL_ON_COMMIT', CFG['FULL_ON_COMMIT'])
