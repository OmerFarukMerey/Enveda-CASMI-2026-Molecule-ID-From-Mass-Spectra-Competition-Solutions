import pandas as pd, numpy as np, pyarrow.dataset as pads, pyarrow as pa, pickle, warnings; warnings.filterwarnings('ignore')
D='/root/lab/pcrank7/'; c=pd.read_parquet(D+'pccand.parquet'); c=c[c.rk<200].copy()
c['sanp']=c.sanp.replace(-np.inf,np.nan); c['sanp']=c.sanp.fillna(c.groupby('key').sanp.transform('min'))
g=c.groupby('key'); c['zf']=(c.fz-g.fz.transform('mean'))/(g.fz.transform('std')+1e-6); c['zs']=(c.sanp-g.sanp.transform('mean'))/(g.sanp.transform('std')+1e-6)
c['b']=c.zf+2*c.zs; c=c.sort_values(['key','b'],ascending=[True,False]); c['p']=c.groupby('key').cumcount()+1
t=c[(c.p<=25)]; print('missing smiles in top25:',(t.smiles=='').mean())
t=t[t.smiles!='']; L={k:(g.smiles.tolist(),g.y.tolist(),g.b.tolist()) for k,g in t.groupby('key')}
keys=sorted(L); print(len(keys))
ds=pads.dataset('/root/data/train.parquet'); rows=[]
flt=(pads.field('ingest_lib')=='enveda-180')&pads.field('inchikey14').isin(pa.array(keys))
for b in ds.scanner(columns=['inchikey14','adduct','precursor_mz','ms2_mzs','ms2_normalized_intensities'],filter=flt,batch_size=4096,batch_readahead=0,fragment_readahead=0,use_threads=False).to_batches():
    if b.num_rows: rows.append(b.to_pandas())
sp=pd.concat(rows); print(len(sp), sp.inchikey14.nunique())
pickle.dump((L,sp),open('/root/lab/net_bench.pkl','wb'))
