import pandas as pd, numpy as np, pyarrow.dataset as pads, pickle, time
from rdkit import Chem, DataStructs, RDLogger; from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
H='/root/lab/harn/'; tr=pd.read_parquet(H+'harness/truth.parquet'); print(tr.columns.tolist(), tr.head(2).T)
sub=pd.read_csv(H+'submission.csv',dtype=str,keep_default_na=False)
ds=pads.dataset('/root/data/train.parquet')
st=ds.to_table(columns=['inchikey14','normalized_smiles']).to_pandas().drop_duplicates('inchikey14'); print('train structures',len(st))
tk=set(tr.key) if 'key' in tr.columns else set()
st=st[~st.inchikey14.isin(tk)].reset_index(drop=True); print('after removing harness truths',len(st))
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
t0=time.time(); fps=[]; keep=[]
for i,s in enumerate(st.normalized_smiles):
    m=Chem.MolFromSmiles(s)
    if m is None: continue
    fps.append(gen.GetFingerprint(m)); keep.append(i)
st=st.iloc[keep].reset_index(drop=True); print('fps',len(fps),round(time.time()-t0))
cands=sorted({x for s in sub.smiles for x in s.split(';') if x}); print('unique candidates',len(cands))
NB={}; t0=time.time()
for n,c in enumerate(cands):
    m=Chem.MolFromSmiles(c)
    if m is None: continue
    sims=np.array(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m),fps)); o=np.argsort(-sims)[:8]
    NB[c]=[(st.inchikey14[i],float(sims[i])) for i in o if sims[i]>=0.4]
    if n%2000==0: print(n,round(time.time()-t0),flush=True)
pickle.dump(NB,open('/root/lab/an_nb.pkl','wb'))
need={k for v in NB.values() for k,_ in v}; print('train keys needed',len(need)); pd.Series(sorted(need)).to_csv('/root/lab/an_need.csv',index=False)
