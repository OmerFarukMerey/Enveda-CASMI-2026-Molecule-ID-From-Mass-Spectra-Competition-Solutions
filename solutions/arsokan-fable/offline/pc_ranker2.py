import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
D='/root/lab/pcrank7/'
c=pd.read_parquet(D+'pccand.parquet').drop(columns='smiles'); c=c[c.rk<200].copy()
keys=pd.read_csv(D+'pcrank.csv').key.unique(); N=len(keys)
c['sanp']=c.sanp.replace(-np.inf,np.nan); c['sanp']=c.sanp.fillna(c.groupby('key').sanp.transform('min'))
g=c.groupby('key')
c['zf']=(c.fz-g.fz.transform('mean'))/(g.fz.transform('std')+1e-6)       # z within the top-200
c['zs']=(c.sanp-g.sanp.transform('mean'))/(g.sanp.transform('std')+1e-6)
c['rs']=g.sanp.rank(ascending=False); c['gap']=g.fz.transform('max')-c.fz; c['sgap']=g.sanp.transform('max')-c.sanp
c['pop']=c.lsid+c.lpmid; c['ffrac']=c.fcount/c.nwin
fm=c.groupby(['key','form']).agg(f_max=('fz','max'),f_n=('fz','size'),fs_max=('sanp','max')).reset_index(); c=c.merge(fm,on=['key','form'])
c['f_gap']=c.groupby('key').f_max.transform('max')-c.f_max; c['fs_gap']=c.groupby('key').fs_max.transform('max')-c.fs_max
def mrr(df,col,K=200):
    r=df[df.rk<K].assign(s=lambda x:x[col]).sort_values(['key','s'],ascending=[True,False]); r['p']=r.groupby('key').cumcount()+1
    h=r[r.y==1].p; return dict(mrr=round(float(np.where(h<=25,1/h,0).sum()/N),4), top1=round(float((h==1).sum()/N),3), top5=round(float((h<=5).sum()/N),3), in25=round(float((h<=25).sum()/N),3))
print('N',N); print('fz only        ',mrr(c,'fz')); print('scorer only top200',mrr(c,'sanp')); print('scorer only top50 ',mrr(c,'sanp',50)); print('scorer only top25 ',mrr(c,'sanp',25))
for w in (0.25,0.5,1,2,4):
    c['b']=c.zf+w*c.zs; print('zf+%.2f*zs'%w, mrr(c,'b'), '| top50', mrr(c,'b',50)['mrr'])
F=['fz','z','zf','rk','gap','sanp','zs','rs','sgap','lsid','lpmid','pop','fcount','ffrac','nwin','f_max','f_n','f_gap','fs_max','fs_gap']
rng=np.random.default_rng(0); fold=dict(zip(keys,rng.integers(0,5,N))); c['fold']=c.key.map(fold); c['pred']=0.0
imp=np.zeros(len(F))
for k in range(5):
    tr=c[c.fold!=k].sort_values('key'); te=c[c.fold==k]
    m=lgb.LGBMRanker(objective='lambdarank',n_estimators=400,learning_rate=0.05,num_leaves=31,min_child_samples=30,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,lambdarank_truncation_level=30)
    m.fit(tr[F],tr.y,group=tr.groupby('key',sort=False).size().values); c.loc[c.fold==k,'pred']=m.predict(te[F]); imp+=m.booster_.feature_importance('gain')
print('LGBM all       ',mrr(c,'pred')); print(sorted(zip((imp/imp.sum()).round(3),F),reverse=True)[:8])
# confidence of the new top-1
c['b']=c.zf+1*c.zs
r=c.sort_values(['key','pred'],ascending=[True,False]); t=r.groupby('key').head(2).copy(); t['p']=t.groupby('key').cumcount()
a=t[t.p==0].set_index('key'); b=t[t.p==1].set_index('key'); a['m']=a.pred-b.pred
print(a.groupby(pd.qcut(a.m,8)).y.agg(['size','mean']).round(3).T.to_string())
c[['key','rk','y','fz','sanp','pred']].to_parquet(D+'pred.parquet')
