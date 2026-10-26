"""Paths of the ownneural study (our spectrum <-> structure model for ranks 2-25): data under $CASMI_DATA, run folders, eval dumps.

Nothing here is in git except this file. The spectrum cache and pool tables are those of the joint study (train/joint/jpaths.py).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(SOLUTION))
CLEAN = os.path.join(SOLUTION, 'clean')
DATA = os.environ.get('CASMI_DATA', os.path.join(REPO, 'data')).replace('\\', '/')
WORK = DATA + '/directions/run/work'          # spectrum cache, pool tables, split.parquet (identical to clean/specdata.build_cache output)
JIDX = DATA + '/joint/index'                  # joint study index: per-spectrum metadata arrays, pool formula groups, PubChem map
PCNEG = DATA + '/hardneg/w10'                 # PubChem window negatives (train/mine_pc_negatives.py)
TIER = DATA + '/pubchem_study/tier'
ON = DATA + '/ownneural'                      # this study: split, negative tables, run folders
RUNS = ON + '/runs'
CW = os.environ.get('CASMI_WORK', 'C:/Users/hemre/casmi-work').replace('\\', '/')
DUMPS = {'s1val': CW + '/s1val/dump/s1val', 's1b': CW + '/rerank25/dumpb/s1val'}   # real final lists of s1 (submission.csv, labels, val_test.parquet)
SCAFFOLDS = CW + '/clean/scaffolds.parquet'   # Murcko scaffold of every training structure (scratch/clean/h_scaffolds.py; read only)
CLEAN_STATE = os.path.join(REPO, 'scratch', 'clean', 'STATE.md')
GPU_LOCK = 'C:/Users/hemre/casmi-work/GPU_LOCK'
