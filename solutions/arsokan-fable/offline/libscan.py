# best library match per test molecule vs extlib2 (all structures, not only best)
import numpy as np, pandas as pd, json, sys
D='/root/ext/lib2/nk/extlib2/'
meta=pd.read_parquet(D+'ext2_meta.parquet'); off=np.load(D+'ext2_offsets.npy'); MZ=np.load(D+'ext2_mz_flat.npy',mmap_mode='r'); IN=np.load(D+'ext2_in_flat.npy',mmap_mode='r')
print(off.shape, MZ.shape, MZ.dtype)
te=pd.read_parquet('/root/data/test.parquet'); print(te.columns.tolist())
mode=np.where(meta.ionization_mode.str.contains('pos'),1,-1); prec=meta.precursor_mz.values
o=np.argsort(prec,kind='stable'); ps=prec[o]
def cos(qm,qi,rm,ri,tol=0.015):
    qa=np.sqrt(qi); ra=np.sqrt(ri); qa/=np.linalg.norm(qa)+1e-12; ra/=np.linalg.norm(ra)+1e-12
    j0=np.searchsorted(rm,qm-tol,'left'); j1=np.searchsorted(rm,qm+tol,'right')
    pairs=[(qa[i]*ra[j],i,j) for i in range(len(qm)) for j in range(j0[i],j1[i])]
    pairs.sort(reverse=True); ui=set(); uj=set(); s=0; n=0
    for p,i,j in pairs:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s+=p; n+=1
    return s,n
out=[]
for mid,g in te.groupby('molecule_id',sort=False):
    best={}
    ncand=0
    for r in g.itertuples(index=False):
        m=np.asarray(r.ms2_mzs,float); it=np.asarray(r.ms2_normalized_intensities,float); pz=float(r.precursor_mz)
        ok=(m>0)&(it>0)&(m<=pz+0.5); m,it=m[ok],it[ok]; q=np.argsort(m); m,it=m[q],it[q]
        md=1 if 'pos' in str(r.ionization_mode).lower() else -1
        tol=pz*10e-6+0.002
        for x in range(np.searchsorted(ps,pz-tol,'left'),np.searchsorted(ps,pz+tol,'right')):
            i=o[x]
            if mode[i]!=md: continue
            a,b=int(off[i]),int(off[i+1]); rm=np.asarray(MZ[a:b],float); ri=np.asarray(IN[a:b],float)
            k=(rm>0)&(ri>0); rm,ri=rm[k],ri[k]
            if len(rm)<3: continue
            q2=np.argsort(rm); c,n=cos(m,it,rm[q2],ri[q2])
            key=meta.inchikey14.values[i]
            if n>=4 and c>best.get(key,(0,))[0]: best[key]=(c,n,meta.normalized_smiles.values[i],meta.source.values[i],str(meta.name.values[i])[:40])
            elif key not in best: best[key]=(0.0,n,meta.normalized_smiles.values[i],meta.source.values[i],str(meta.name.values[i])[:40])
    rk=sorted(best.items(),key=lambda kv:-kv[1][0])[:5]
    out.append(dict(molecule_id=mid,nstruct=len(best),adduct=str(g.adduct.iloc[0]),top=[(k,)+tuple(v) for k,v in rk]))
json.dump(out,open('/root/lab/libscan_ext2.json','w'),default=float)
c1=[x['top'][0][1] if x['top'] else -1 for x in out]
import collections
print('molecules',len(out),'with any precursor-matching structure',sum(x['nstruct']>0 for x in out))
for t in (0.9,0.8,0.72,0.65,0.5,0.3,0.0001): print('best cos >=',t,sum(c>=t for c in c1))
