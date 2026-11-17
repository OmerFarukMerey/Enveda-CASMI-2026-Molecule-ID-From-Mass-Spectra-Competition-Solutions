"""vh step 1: eligible pool (cross-lab + leak-free enveda-only), their would-be query spectra, and the in-test network graph."""
import pandas as pd, numpy as np, pyarrow as pa, pyarrow.dataset as pads, pickle, time, scipy.sparse as sps
R='/home/user/Enveda-CASMI-2026-Molecule-ID-From-Mass-Spectra-Competition-Solutions/'
t0=time.time(); m=pd.read_parquet('train_meta.parquet'); sp=pd.read_parquet('ext/split.parquet'); mo=pd.read_parquet('ext/molecules.parquet')
hold=set(sp.inchikey14.astype(str)); val=set(mo[mo.split=='validation'].key.astype(str).str[:14])
QADD=['[M+H]+','[M-H]-','[M+CH2O2-H]-','[M+Na]+','[M+NH4]+','[M+K]+','[M+Cl]-']; ADD=('[M+H]+','[M-H]-')
libs=m.groupby('inchikey14').ingest_lib.agg(lambda x: tuple(sorted(set(x))))
X=set(libs[libs.map(lambda t:'enveda-180' in t and len(t)>1)].index); E=set(libs[libs.map(lambda t:t==('enveda-180',))].index)&hold&val
en=m[(m.ingest_lib=='enveda-180')&m.inchikey14.isin(X|E)]
q=en[en.adduct.isin(QADD)]; hasadd=set(q[q.adduct.isin(ADD)].inchikey14)
pool=sorted((X|E)&hasadd); print('pool',len(pool),'X',len(X&hasadd),'E',len(E&hasadd),flush=True)
q=q[q.inchikey14.isin(pool)]; rows=set(q.row.tolist())
ds=pads.dataset(R+'data/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet')
import os
parts=[]; base=0
import pyarrow.parquet as pq
pf=pq.ParquetFile(R+'data/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet'); want=np.array(sorted(rows))
for g in range(pf.num_row_groups):
    n=pf.metadata.row_group(g).num_rows; a,b=np.searchsorted(want,[base,base+n])
    if b>a:
        t=pf.read_row_group(g,columns=['inchikey14','normalized_smiles','adduct','precursor_mz','ms2_mzs','ms2_normalized_intensities','collision_energy_orig']); d=t.take(pa.array(want[a:b]-base)).to_pandas(); d['row']=want[a:b]; parts.append(d)
    base+=n
Q=pd.concat(parts); Q.to_parquet('pool_query_rows.parquet'); print('query rows',len(Q),round(time.time()-t0),flush=True)
# n1c-style merged spectra ([M+H]+/[M-H]- only), exactly the cell's preprocessing
SP={}
for (k,ad),g in Q[Q.adduct.isin(ADD)].groupby(['inchikey14','adduct']):
    pr=float(np.median(pd.to_numeric(g.precursor_mz,errors='coerce')))
    mz=np.concatenate([np.asarray(x,float) for x in g.ms2_mzs]); it=np.concatenate([np.asarray(x,float)/max(float(np.max(x)) if len(x) else 1.0,1e-9) for x in g.ms2_normalized_intensities])
    kk=mz<pr-1.5; mz,it=mz[kk],it[kk]
    if not len(mz): continue
    u,inv=np.unique(np.round(mz,2),return_inverse=True); w=np.zeros(len(u)); np.maximum.at(w,inv,it)
    o=np.argsort(-w)[:120]; u,w=u[o],np.sqrt(w[o]); oo=np.argsort(u); SP.setdefault(k,{})[ad]=(u[oo],(w/max(np.linalg.norm(w),1e-12))[oo],pr)
def mcos(a,b):
    (ma,ia_,pa_),(mb,ib,pb)=a,b; sh=pa_-pb; cand=[]
    for i in range(len(ma)):
        for off in (0.0,sh):
            lo,hi=np.searchsorted(mb,ma[i]-off-0.0101),np.searchsorted(mb,ma[i]-off+0.0101,'right')
            for j in range(lo,hi): cand.append((ia_[i]*ib[j],i,j))
    cand.sort(reverse=True); ui,uj,s,n=set(),set(),0.0,0
    for v,i,j in cand:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s+=v; n+=1
    return s,n
# prefilter: binned plain cosine on fragments and on neutral losses (covers the shifted matches), then the exact rule
keys=sorted(SP); idx={k:i for i,k in enumerate(keys)}; EDG={}
for ad in ADD:
    ks=[k for k in keys if ad in SP[k]]; r=[];c=[];v=[];r2=[];c2=[];v2=[]
    r0=[];c0=[];v0=[];r3=[];c3=[];v3=[]
    for i,k in enumerate(ks):
        u,w,pr=SP[k][ad]; b=np.round(u*50).astype(int); nl=np.round((pr-u)*50).astype(int)+200000
        r0+=[i]*len(b); c0+=list(b); v0+=list(w); r3+=[i]*len(b); c3+=list(nl); v3+=list(w)
        for d in (-1,0,1): r+=[i]*len(b); c+=list(b+d); v+=list(w); r2+=[i]*len(b); c2+=list(nl+d); v2+=list(w)
    W=max(max(c),max(c2))+2
    A=sps.csr_matrix((v,(r,c)),shape=(len(ks),W)); A0=sps.csr_matrix((v0,(r0,c0)),shape=(len(ks),W)); B=sps.csr_matrix((v2,(r2,c2)),shape=(len(ks),W)); B0=sps.csr_matrix((v3,(r3,c3)),shape=(len(ks),W))
    S=(A0@A.T)+(B0@B.T); S=S.tocoo(); cand=[(i,j) for i,j,x in zip(S.row,S.col,S.data) if i!=j and x>=0.55]; cand=sorted({(min(i,j),max(i,j)) for i,j in cand})
    print(ad,'molecules',len(ks),'candidate pairs',len(cand),round(time.time()-t0),flush=True)
    for i,j in cand:
        s,n=mcos(SP[ks[i]][ad],SP[ks[j]][ad])
        if n>=6 and s>=0.6:
            a,b=ks[i],ks[j]; EDG[(a,b)]=max(EDG.get((a,b),0),s)
print('edges in pool',len(EDG),'molecules with an edge',len({k for e in EDG for k in e}),round(time.time()-t0))
pickle.dump(dict(pool=pool,X=sorted(X&hasadd),E=sorted(E&hasadd),EDG=EDG,hold=hold&set(pool),val=val&set(pool)),open('pool.pkl','wb'))
