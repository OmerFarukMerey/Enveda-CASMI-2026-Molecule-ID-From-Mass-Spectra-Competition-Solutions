import pandas as pd, numpy as np, pickle, sys
sys.path.insert(0,'/root/pipe'); from net_bench import build_sp, mcos
H='/root/lab/harn/'; sub=pd.read_csv(H+'submission.csv',dtype=str,keep_default_na=False)
te=pd.read_parquet(H+'harness/test.parquet'); te['_m']=te.molecule_id.astype(str)
NB=pickle.load(open('/root/lab/an_nb.pkl','rb')); SPT=pickle.load(open('/root/lab/an_sp.pkl','rb')); SPQ=build_sp(te,set(te._m))
d0=pd.read_parquet('/root/lab/an_feat.parquet'); Y={(r.m,r.rank):r.y for r in d0.itertuples()}
R=[]
for r in sub.itertuples():
    lst=[x for x in r.smiles.split(';') if x]; q=SPQ.get(r.molecule_id,{})
    for i,c in enumerate(lst):
        y=Y[(r.molecule_id,i+1)]; ssum=0.0; mx=0.0; selfc=-1.0; intrain=0; nn=0
        for t,sim in NB.get(c,[]):
            for ad in set(q)&set(SPT.get(t,{})):
                s,n=mcos(q[ad],SPT[t][ad])
                if sim>=0.999 and not y: intrain=1; selfc=max(selfc,s if n>=4 else 0.0)
                if sim>=0.85 or n<4: continue
                ssum+=sim*s; mx=max(mx,sim*s); nn+=1
        R.append((r.molecule_id,i+1,y,ssum,mx,nn,intrain,selfc,len(q)>0))
d=pd.DataFrame(R,columns=['m','rank','y','asum','amax','an','intrain','selfc','hasq']); d.to_parquet('/root/lab/an_feat3.parquet')
z=lambda v:(v-v.mean())/(v.std()+1e-9); N=d.m.nunique()
r1=d[d['rank']==1].set_index('m')
print('rank1 in train:',r1.intrain.mean(),'| rank1 correct overall',r1.y.mean(),'| rank1 selfc>=0.6:',(r1.selfc>=0.6).mean(), '| has query spectrum',r1.hasq.mean())
print(r1.groupby(pd.cut(r1.selfc,[-2,-0.5,0.3,0.6,0.8,1.1])).y.agg(['size','mean']))
def ev(col,lam,gate=None,keep1=False):
    s=0;t1=0;ch=0
    for m,g in d.groupby('m',sort=False):
        sc=z(-np.log(g['rank'].values))+lam*z(g[col].values)
        if gate is not None and g.selfc.values[0]>=gate: sc=-g['rank'].values.astype(float)
        if keep1: sc[0]=1e9
        o=np.argsort(-sc,kind='stable'); h=np.flatnonzero(g.y.values[o]); ch+=o[0]!=0
        if len(h): s+=1/(h[0]+1); t1+=h[0]==0
    return round(float(s/N),4),round(float(t1/N),3),ch
print('baseline',ev('asum',0))
for col in ('asum','amax'):
    for gate in (None,0.6,0.4):
        print(col,'gate',gate,[ev(col,l,gate) for l in (0.35,0.5,0.75,1.0)])
print('asum keep1',[ev('asum',l,None,True) for l in (0.5,1.0)])
# bootstrap of the gain at lam 0.75
rng=np.random.default_rng(0); ms=d.m.unique(); per={}
for m,g in d.groupby('m',sort=False):
    def rr(sc):
        o=np.argsort(-sc,kind='stable'); h=np.flatnonzero(g.y.values[o]); return 1/(h[0]+1) if len(h) else 0
    per[m]=rr(z(-np.log(g['rank'].values))+0.75*z(g.asum.values))-rr(-g['rank'].values.astype(float))
v=np.array([per[m] for m in ms]); bs=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(2000)]
print('gain',round(v.mean(),4),'95% CI',np.round(np.percentile(bs,[2.5,97.5]),4),'improved',int((v>0).sum()),'worsened',int((v<0).sum()))
