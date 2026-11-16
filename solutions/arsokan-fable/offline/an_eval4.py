import pandas as pd, numpy as np, pickle, sys
sys.path.insert(0,'/root/pipe'); from net_bench import mcos
H='/root/lab/harn/'; sub=pd.read_csv(H+'submission.csv',dtype=str,keep_default_na=False)
te=pd.read_parquet(H+'harness/test.parquet'); te['_m']=te.molecule_id.astype(str)
NB=pickle.load(open('/root/lab/an_nb.pkl','rb')); SPT=pickle.load(open('/root/lab/an_sp.pkl','rb'))
d0=pd.read_parquet('/root/lab/an_feat.parquet'); Y={(r.m,r.rank):r.y for r in d0.itertuples()}
def build_all(df):
    SP={}
    for (m,ad),g in df.groupby(['_m','adduct']):
        pr=float(np.median(pd.to_numeric(g.precursor_mz,errors='coerce')))
        mz=np.concatenate([np.asarray(x,float) for x in g.ms2_mzs]); it=np.concatenate([np.asarray(x,float)/max(float(np.max(x)) if len(x) else 1.0,1e-9) for x in g.ms2_normalized_intensities])
        k=mz<pr-1.5; mz,it=mz[k],it[k]
        if not len(mz): continue
        u,inv=np.unique(np.round(mz,2),return_inverse=True); w=np.zeros(len(u)); np.maximum.at(w,inv,it)
        o=np.argsort(-w)[:120]; u,w=u[o],np.sqrt(w[o]); oo=np.argsort(u)
        SP.setdefault(m,{})[str(ad)]=(u[oo],(w/max(np.linalg.norm(w),1e-12))[oo],pr)
    return SP
SPQ=build_all(te); print('queries with any spectrum',len(SPQ))
def pairs(q,tt,mode):
    out=[]
    for ad,qs in q.items():
        if ad in tt: out.append((qs,tt[ad]))
        elif mode=='fallback':
            ref='[M+H]+' if ad.endswith('+') else '[M-H]-'
            if ref in tt and ad not in ('[M+H]+','[M-H]-'): out.append((qs,tt[ref]))
    return out
z=lambda v:(v-v.mean())/(v.std()+1e-9); N=len(sub)
def feats(mode,nmin=4,HI=0.85,LO=0.4,pw=1.0):
    F={}
    for r in sub.itertuples():
        lst=[x for x in r.smiles.split(';') if x]; q=SPQ.get(r.molecule_id,{}); ev=np.zeros(len(lst)); selfc=-1.0
        for i,c in enumerate(lst):
            y=Y[(r.molecule_id,i+1)]
            for t,sim in NB.get(c,[]):
                tt=SPT.get(t)
                if tt is None: continue
                for qs,ts in pairs(q,tt,mode):
                    s,n=mcos(qs,ts)
                    if sim>=0.999 and not y:
                        if i==0: selfc=max(selfc,s if n>=4 else 0.0)
                        continue
                    if sim>=HI or sim<LO or n<nmin: continue
                    ev[i]+=sim**pw*s
        F[r.molecule_id]=(ev,selfc,np.array([Y[(r.molecule_id,i+1)] for i in range(len(lst))]))
    return F
def ev(F,lam,gate=0.6):
    s=0;t1=0;used=0
    for m,(e,selfc,y) in F.items():
        sc=-np.arange(len(e)).astype(float)
        if selfc<gate and e.max()>0: sc=z(-np.log(np.arange(1,len(e)+1)))+lam*z(e); used+=1
        o=np.argsort(-sc,kind='stable'); h=np.flatnonzero(y[o])
        if len(h): s+=1/(h[0]+1); t1+=h[0]==0
    return round(float(s/N),4),round(float(t1/N),3),used
for name,kw in (('strict',dict(mode='strict')),('fallback',dict(mode='fallback')),('fallback nmin3',dict(mode='fallback',nmin=3)),('fallback HI0.9',dict(mode='fallback',HI=0.9)),('fallback HI0.75',dict(mode='fallback',HI=0.75)),('fallback LO0.5',dict(mode='fallback',LO=0.5)),('fallback pw2',dict(mode='fallback',pw=2.0))):
    F=feats(**kw); print(f'{name:18s}',[ev(F,l) for l in (0.5,0.75,1.0,1.5)],flush=True)
