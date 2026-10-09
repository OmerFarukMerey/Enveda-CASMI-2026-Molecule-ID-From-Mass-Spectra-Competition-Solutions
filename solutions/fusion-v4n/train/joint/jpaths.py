"""Paths of the joint spectrum-structure model work: everything under $CASMI_DATA (default: repo data/), nothing in git."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(SOLUTION))
DATA = os.environ.get('CASMI_DATA', os.path.join(REPO, 'data')).replace('\\', '/')
WORK = DATA + '/directions/run/work'          # spectrum cache, pool tables, split.parquet (train/README.md)
CODE = DATA + '/directions/code'              # the public `casmi` package (fingerprint definition)
JOINT = DATA + '/joint'                       # this work's indices, checkpoints and score files
BENCH = JOINT + '/bench'                      # the fixed benchmark (bench_build.py, written once)
TABLE = DATA + '/rankbank4/data'              # feature table of the fsranker3 study (165 ranker inputs per candidate row)
PCNEG = DATA + '/hardneg/w10'                 # PubChem window negatives of train/mine_pc_negatives.py
TIER = DATA + '/pubchem_study/tier'           # PubChem tier (pc_mass / pc_off / pc_smiles), for the SMILES of those negatives
