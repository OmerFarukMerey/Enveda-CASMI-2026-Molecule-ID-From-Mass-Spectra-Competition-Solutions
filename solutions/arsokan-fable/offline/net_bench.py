import pandas as pd, numpy as np, pickle, sys, time
from rdkit import Chem, DataStructs, RDLogger; from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048); fpc={}
def fp(s):
    if s not in fpc:
        m=Chem.MolFromSmiles(s); fpc[s]=gen.GetFingerprint(m) if m is not None else None
    return fpc[s]
def build_sp(te, ids):
    SP={}
    for (m,ad),g in te.groupby(['_m','adduct']):
        if str(ad) not in ('[M+H]+','[M-H]-') or m not in ids: continue
        pr=float(np.median(pd.to_numeric(g.precursor_mz,errors='coerce')))
        mz=np.concatenate([np.asarray(x,float) for x in g.ms2_mzs]); it=np.concatenate([np.asarray(x,float)/max(float(np.max(x)) if len(x) else 1.0,1e-9) for x in g.ms2_normalized_intensities])
        k=mz<pr-1.5; mz,it=mz[k],it[k]
        if not len(mz): continue
        kb=np.round(mz,2); u,inv=np.unique(kb,return_inverse=True); w=np.zeros(len(u)); np.maximum.at(w,inv,it)
        o=np.argsort(-w)[:120]; u,w=u[o],np.sqrt(w[o]); oo=np.argsort(u)
        SP.setdefault(m,{})[str(ad)]=(u[oo],(w/max(np.linalg.norm(w),1e-12))[oo],pr)
    return SP
def mcos(a,b):
    (ma,ia_,pa),(mb,ib,pb)=a,b; sh=pa-pb; cand=[]
    for off in ((0.0,sh) if abs(sh)>0.02 else (0.0,)):
        lo=np.searchsorted(mb,ma-off-0.0101); hi=np.searchsorted(mb,ma-off+0.0101,'right')
        for i in np.flatnonzero(hi>lo):
            for j in range(lo[i],hi[i]): cand.append((ia_[i]*ib[j],i,j))
    cand.sort(reverse=True); ui,uj,s,n=set(),set(),0.0,0
    for v,i,j in cand:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s+=v; n+=1
    return s,n
def edges(SP,tau=0.0,nmin=6):
    K=sorted(SP); E=[]
    for x in range(len(K)):
        for y in range(x+1,len(K)):
            best=None
            for ad in set(SP[K[x]])&set(SP[K[y]]):
                # cheap prefilter: precursor difference under 400 Da
                s,n=mcos(SP[K[x]][ad],SP[K[y]][ad])
                if n>=nmin and s>=tau and (best is None or s>best): best=s
            if best is not None: E.append((K[x],K[y],best))
    return E
if __name__=='__main__':
    L,sp=pickle.load(open('/root/lab/net_bench.pkl','rb')); sp['_m']=sp.inchikey14
    t0=time.time(); SP=build_sp(sp,set(L)); E=edges(SP,tau=0.4); print('bench mols',len(SP),'edges>=0.4',len(E),'>=0.6',sum(e[2]>=0.6 for e in E),round(time.time()-t0),'s')
    te=pd.read_parquet('/root/data/test.parquet'); te['_m']=te.molecule_id.astype(str)
    SPt=build_sp(te,set(te._m)); Et=edges(SPt,tau=0.4); print('visible test mols',len(SPt),'edges>=0.4',len(Et),'>=0.6',sum(e[2]>=0.6 for e in Et), 'mols with nb>=0.6',len({x for e in Et if e[2]>=0.6 for x in e[:2]}))
    print('bench mols with nb>=0.6',len({x for e in E if e[2]>=0.6 for x in e[:2]}))
    pickle.dump((E,Et),open('/root/lab/net_edges.pkl','wb'))
