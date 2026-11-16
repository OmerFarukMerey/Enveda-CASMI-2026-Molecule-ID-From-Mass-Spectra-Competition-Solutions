import numpy as np, pandas as pd, json, pyarrow.parquet as pq, pyarrow.compute as pc, random
D='/root/ext/lib2/nk/extlib2/'
meta=pd.read_parquet(D+'ext2_meta.parquet',columns=['ionization_mode','precursor_mz','inchikey14']); off=np.load(D+'ext2_offsets.npy'); MZ=np.load(D+'ext2_mz_flat.npy',mmap_mode='r'); IN=np.load(D+'ext2_in_flat.npy',mmap_mode='r')
mode=np.where(meta.ionization_mode.str.contains('pos'),1,-1); prec=meta.precursor_mz.values; KEY=meta.inchikey14.values
o=np.argsort(prec,kind='stable'); ps=prec[o]; libkeys=set(KEY)
ml=pd.read_parquet('/root/lab/train_meta_light.parquet'); ml=ml[ml.ingest_lib=='enveda-180']
keys=sorted(set(ml.inchikey14)); random.seed(7); samp=set(random.sample(keys,4000)); print('enveda mols',len(keys),'in lib frac',np.mean([k in libkeys for k in keys]))
def cos(qm,qi,rm,ri,tol=0.015):
    qa=np.sqrt(qi); ra=np.sqrt(ri); qa/=np.linalg.norm(qa)+1e-12; ra/=np.linalg.norm(ra)+1e-12
    j0=np.searchsorted(rm,qm-tol,'left'); j1=np.searchsorted(rm,qm+tol,'right')
    pairs=[(qa[i]*ra[j],i,j) for i in range(len(qm)) for j in range(j0[i],j1[i])]
    pairs.sort(reverse=True); ui=set(); uj=set(); s=0; n=0
    for p,i,j in pairs:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s+=p; n+=1
    return s,n
import pyarrow.dataset as pads, pyarrow as pa
ds=pads.dataset('/root/data/train.parquet'); rows=[]
flt=(pads.field('ingest_lib')=='enveda-180')&pads.field('inchikey14').isin(pa.array(sorted(samp)))
for b in ds.scanner(columns=['inchikey14','ionization_mode','adduct','precursor_mz','ms2_mzs','ms2_normalized_intensities'],filter=flt,batch_size=4096,batch_readahead=0,fragment_readahead=0,use_threads=False).to_batches():
    if b.num_rows: rows.append(b.to_pandas())
tr=pd.concat(rows); print(len(tr),tr.inchikey14.nunique()); del rows
out=[]
for mid,g in tr.groupby('inchikey14',sort=False):
    best={}
    for r in g.itertuples(index=False):
        m=np.asarray(r.ms2_mzs,float); it=np.asarray(r.ms2_normalized_intensities,float); pz=float(r.precursor_mz)
        ok=(m>0)&(it>0)&(m<=pz+0.5); m,it=m[ok],it[ok]; q=np.argsort(m); m,it=m[q],it[q]
        md=1 if 'pos' in str(r.ionization_mode).lower() else -1; tol=pz*10e-6+0.002
        for x in range(np.searchsorted(ps,pz-tol,'left'),np.searchsorted(ps,pz+tol,'right')):
            i=o[x]
            if mode[i]!=md: continue
            a,b_=int(off[i]),int(off[i+1]); rm=np.asarray(MZ[a:b_],float); ri=np.asarray(IN[a:b_],float)
            k=(rm>0)&(ri>0); rm,ri=rm[k],ri[k]
            if len(rm)<3: continue
            q2=np.argsort(rm); c,n=cos(m,it,rm[q2],ri[q2]); key=KEY[i]
            if n>=4 and c>best.get(key,(0,0))[0]: best[key]=(c,n)
            elif key not in best: best[key]=(0.0,n)
    rk=sorted(best.items(),key=lambda kv:-kv[1][0])[:3]
    out.append(dict(key=mid,inlib=mid in libkeys,adduct=g.adduct.iloc[0],nspec=len(g),top=[(k,)+tuple(v) for k,v in rk]))
json.dump(out,open('/root/lab/libval_ext2.json','w'),default=float)
