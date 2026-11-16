import sys, shutil, os, pandas as pd, numpy as np
exec(open('/root/push/casmi-t0-stage/stage.py').read())
W='/root/lab/ta_local'; os.makedirs(W,exist_ok=True); shutil.copy('/root/lab/harn/submission.csv',W+'/submission.csv')
_ta_run(sub_path=W+'/submission.csv', test_path='/root/lab/harn/harness/test.parquet', train_path='/root/data/train.parquet')
d=pd.read_parquet('/root/lab/an_feat3.parquet'); 
from rdkit import Chem
a=pd.read_csv('/root/lab/harn/submission.csv',dtype=str,keep_default_na=False); b=pd.read_csv(W+'/submission.csv',dtype=str,keep_default_na=False)
Y={(r.m,r.rank):r.y for r in d.itertuples()}; s0=s1=0
for ra,rb in zip(a.itertuples(),b.itertuples()):
    la=ra.smiles.split(';'); lb=rb.smiles.split(';'); t=[c for i,c in enumerate(la) if Y.get((ra.molecule_id,i+1))]
    if t: s0+=1/(la.index(t[0])+1); s1+=1/(lb.index(t[0])+1)
print('MRR before',round(s0/len(a),4),'after',round(s1/len(a),4))
