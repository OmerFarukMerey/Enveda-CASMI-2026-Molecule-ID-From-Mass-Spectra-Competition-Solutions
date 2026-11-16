import pandas as pd, numpy as np, lightgbm as lgb, re
c=pd.read_parquet('/root/lab/pcrank5/pccand.parquet').drop(columns='smiles')
c=c[c.rk<200].copy()   # truths outside the top-200 are counted as misses
keys=pd.read_csv('/root/lab/pcrank5/pcrank.csv').key.unique(); N=len(keys)
g=c.groupby('key')
c['gap']=g.fz.transform('max')-c.fz
c['z_top']=g.z.transform('max')
c['z_sd20']=g.z.transform(lambda v: v.head(20).std())
c['ffrac']=c.fcount/c.nwin
t20=c[c.rk<20].groupby(['key','form']).size().rename('f20').reset_index(); c=c.merge(t20,how='left',on=['key','form']); c['f20']=c.f20.fillna(0)
fm=c.groupby(['key','form']).agg(f_max=('fz','max'),f_n=('fz','size'),f_mean5=('fz',lambda v:v.head(5).mean())).reset_index(); c=c.merge(fm,on=['key','form'])
c['f_gap']=c.groupby('key').f_max.transform('max')-c.f_max
c['rk_in_form']=c.groupby(['key','form']).rk.rank()
c['pop']=c.lsid+c.lpmid
c['pop_rank']=c.groupby('key')['pop'].rank(ascending=False)
for el in ['C','H','N','O','S','F','Cl','Br','P']:
    c['n_'+el]=c.form.str.extract(el+r'(?![a-z])(\d*)')[0].replace('',1).fillna(0).astype(float)
    c.loc[~c.form.str.contains(el+r'(?![a-z])'),'n_'+el]=0
c['hetero']=c.n_N+c.n_O+c.n_S+c.n_F+c.n_Cl+c.n_Br+c.n_P
c['dbe']=c.n_C-(c.n_H+c.n_F+c.n_Cl+c.n_Br)/2+(c.n_N+c.n_P)/2+1
F=['fz','z','rk','gap','z_top','z_sd20','lsid','lpmid','pop','pop_rank','fcount','ffrac','nwin','f20','f_max','f_n','f_mean5','f_gap','rk_in_form','n_C','n_H','n_N','n_O','n_S','n_F','n_Cl','n_Br','n_P','hetero','dbe']
def mrr(df,col):
    r=df.assign(s=df[col]).sort_values(['key','s'],ascending=[True,False]); r['p']=r.groupby('key').cumcount()+1
    h=r[r.y==1].p; return dict(mrr=round(float(np.where(h<=25,1/h,0).sum()/N),4), top1=round(float((h==1).sum()/N),3), top5=round(float((h<=5).sum()/N),3), in25=round(float((h<=25).sum()/N),3))
print('fz only      ',mrr(c,'fz'))
for lam in (0.05,0.15,0.3): c['p']=c.z+lam*c['pop']; print('z+%.2f pop   '%lam,mrr(c,'p'))
rng=np.random.default_rng(0); fold=dict(zip(keys,rng.integers(0,5,N))); c['fold']=c.key.map(fold); c['pred']=0.0
for sets,name in ((F,'all'),([f for f in F if f not in('lsid','lpmid','pop','pop_rank')],'nopop'),(['fz','z','rk','gap','lsid','lpmid'],'small')):
    imp=np.zeros(len(sets))
    for k in range(5):
        tr=c[c.fold!=k].sort_values('key'); te=c[c.fold==k]
        m=lgb.LGBMRanker(objective='lambdarank',n_estimators=400,learning_rate=0.05,num_leaves=31,min_child_samples=30,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,lambdarank_truncation_level=30)
        m.fit(tr[sets],tr.y,group=tr.groupby('key',sort=False).size().values)
        c.loc[c.fold==k,'pred']=m.predict(te[sets]); imp+=m.booster_.feature_importance('gain')
    print('LGBM',name,mrr(c,'pred'))
    if name=='all':
        print(sorted(zip((imp/imp.sum()).round(3),sets),reverse=True)[:12]); c['pred_all']=c.pred
c[['key','rk','y','fz','z','pred_all']].to_parquet('/root/lab/pcrank5/pred.parquet')
