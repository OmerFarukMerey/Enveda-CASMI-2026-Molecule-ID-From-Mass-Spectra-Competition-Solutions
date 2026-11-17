"""vh Task 11: per-candidate own-spectrum agreement for the top-10 of every n0 final list; separability of class-1 errors."""
import pandas as pd, numpy as np, pickle, pyarrow as pa, pyarrow.dataset as pads, time, itertools
exec(open('audit.py').read().split("ST=[")[0])
from rdkit.Chem import rdMolDescriptors
R='/home/user/Enveda-CASMI-2026-Molecule-ID-From-Mass-Spectra-Competition-Solutions/'
P=pickle.load(open(O+'stages.pkl','rb')); F=P['LL']['extlib overlay (final)']; cls=tr.cls.values; mids=list(tr.molecule_id); t0=time.time()
def k14(s):
    try: m=Chem.MolFromSmiles(s); return Chem.MolToInchiKey(m)[:14] if m is not None else None
    except Exception: return None
def form(s):
    m=Chem.MolFromSmiles(s); return rdMolDescriptors.CalcMolFormula(m) if m else None
te=pd.read_parquet(O+'vh/test.parquet'); TOPK=10
C={m:[(c,k14(c),form(c),key(c)) for c in F[m][:TOPK]] for m in mids}
want=sorted({k for v in C.values() for _,k,_,_ in v if k}); ds=pads.dataset(R+'scratch/vh/local/pool/train.parquet')
rows=[]
for b in ds.scanner(columns=['inchikey14','adduct','precursor_mz','ms2_mzs','ms2_normalized_intensities','ingest_lib','collision_energy_orig'],filter=pads.field('inchikey14').isin(pa.array(want)),batch_size=4096,batch_readahead=0,fragment_readahead=0,use_threads=False).to_batches():
    if b.num_rows: rows.append(b.to_pandas())
T=pd.concat(rows); print('own rows',len(T),'structures with rows',T.inchikey14.nunique(),'of',len(want),round(time.time()-t0),flush=True)
def prep(mz,it,pr):
    mz=np.asarray(mz,float); it=np.asarray(it,float); pr=float(pr)
    if not len(mz): return None
    it=it/max(float(it.max()),1e-9); k=mz<pr-1.5; mz,it=mz[k],it[k]
    if len(mz)<3: return None
    o=np.argsort(mz); w=np.sqrt(it[o]); return (mz[o],w/max(np.linalg.norm(w),1e-12),pr)
def merged(specs):
    pr=float(np.median([s[2] for s in specs])); mz=np.concatenate([s[0] for s in specs]); it=np.concatenate([s[1]**2/np.max(s[1]**2) for s in specs])
    u,inv=np.unique(np.round(mz,2),return_inverse=True); w=np.zeros(len(u)); np.maximum.at(w,inv,it); o=np.argsort(-w)[:120]; u,w=u[o],np.sqrt(w[o]); oo=np.argsort(u); return (u[oo],(w/max(np.linalg.norm(w),1e-12))[oo],pr)
def cosn(a,b,shift=False):
    ma,ia_,pa_=a; mb,ib,pb=b; cand=[]
    for off in ((0.0,pa_-pb) if (shift and abs(pa_-pb)>0.02) else (0.0,)):
        lo=np.searchsorted(mb,ma-off-0.0101); hi=np.searchsorted(mb,ma-off+0.0101,'right')
        for i in np.flatnonzero(hi>lo):
            for j in range(lo[i],hi[i]): cand.append((ia_[i]*ib[j],i,j))
    cand.sort(reverse=True); ui,uj,s,n=set(),set(),0.0,0
    for v,i,j in cand:
        if i in ui or j in uj: continue
        ui.add(i); uj.add(j); s+=v; n+=1
    return s,n
Q={}
for m,ad,pr,mz,it in zip(te.molecule_id,te.adduct.astype(str),te.precursor_mz,te.ms2_mzs,te.ms2_normalized_intensities):
    s=prep(mz,it,pr)
    if s is not None: Q.setdefault(m,{}).setdefault(ad,[]).append(s)
OWN={}
for (k,ad),g in T.groupby(['inchikey14','adduct']):
    sp=[prep(a,b,c) for a,b,c in zip(g.ms2_mzs,g.ms2_normalized_intensities,g.precursor_mz)]; sp=[s for s in sp if s is not None]
    if sp: OWN.setdefault(k,{})[str(ad)]=(sp,sorted(set(g.ingest_lib)))
rec=[]
for m,c_ in zip(mids,cls):
    q=Q.get(m,{}); tk=dict(zip(tr.molecule_id,tr.mk))[m]; tform=None
    for rank,(smi,k,f,mk) in enumerate(C[m],1):
        o=OWN.get(k,{}) if k else {}; ads=set(q)&set(o); bp=bpn=mg=mgn=np.nan; nref=sum(len(o[a][0]) for a in o); nsh=sum(len(o[a][0]) for a in ads); libs=';'.join(sorted({l for a in o for l in o[a][1]}))
        if ads:
            pairs=[cosn(a,b) for ad in ads for a in q[ad] for b in o[ad][0][:60]]; bp,bpn=max(pairs)
            mm=[cosn(merged(q[ad]),merged(o[ad][0][:60])) for ad in ads]; mg,mgn=max(mm)
        rec.append(dict(m=m,cls=c_,rank=rank,is_truth=mk==tk,formula=f,key=k,nref=nref,nref_shared=nsh,libs=libs,bp=bp,bp_n=bpn,mg=mg,mg_n=mgn))
D=pd.DataFrame(rec); tf=D[D.is_truth].set_index('m').formula.to_dict(); D['same_formula']=[f==tf.get(m) for m,f in zip(D.m,D.formula)]; lf=D[D['rank']==1].set_index('m').formula.to_dict(); D['same_formula_as_leader']=[f==lf.get(m) for m,f in zip(D.m,D.formula)]
D.to_parquet(O+'own_evidence.parquet'); print('done',round(time.time()-t0),'candidates',len(D),'with own shared-adduct spectra',int(D.bp.notna().sum()))
