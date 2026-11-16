import pandas as pd, numpy as np, pickle, sys, os, pyarrow as pa, pyarrow.dataset as pads
sys.path.insert(0,'/root/pipe'); from net_bench import mcos, build_sp
H='/root/lab/harn/'; sub=pd.read_csv(H+'submission.csv',dtype=str,keep_default_na=False)
te=pd.read_parquet(H+'harness/test.parquet'); te['_m']=te.molecule_id.astype(str)
NB=pickle.load(open('/root/lab/an_nb30.pkl','rb')); need=sorted({k for v in NB.values() for k,_ in v})
if not os.path.exists('/root/lab/an_sp30.pkl'):
    ds=pads.dataset('/root/data/train.parquet'); SPT={}; buf=[]; nrow=0
    def flush():
        global buf
        if buf:
            d=pd.concat(buf); d['_m']=d.inchikey14; SPT.update(build_sp(d,set(d._m))); buf=[]
    cnt={}
    for b in ds.scanner(columns=['inchikey14','adduct','precursor_mz','ms2_mzs','ms2_normalized_intensities'],filter=pads.field('inchikey14').isin(pa.array(need))&pads.field('adduct').isin(pa.array(['[M+H]+','[M-H]-'])),batch_size=4096,batch_readahead=0,fragment_readahead=0,use_threads=False).to_batches():
        if not b.num_rows: continue
        buf.append(b.to_pandas().groupby(['inchikey14','adduct']).head(6)); nrow+=len(buf[-1])
        if nrow>60000: flush(); nrow=0
    flush(); pickle.dump(SPT,open('/root/lab/an_sp30.pkl','wb'))
SPT=pickle.load(open('/root/lab/an_sp30.pkl','rb')); SPQ=build_sp(te,set(te._m)); print('train sp',len(SPT),flush=True)
d0=pd.read_parquet('/root/lab/an_feat.parquet'); Y={(r.m,r.rank):r.y for r in d0.itertuples()}
z=lambda v:(v-v.mean())/(v.std()+1e-9); N=len(sub); RAW={}
for r in sub.itertuples():
    lst=[x for x in r.smiles.split(';') if x]; q=SPQ.get(r.molecule_id,{}); rows=[]; selfc=-1.0
    for i,c in enumerate(lst):
        y=Y[(r.molecule_id,i+1)]; rr=[]
        for j,(t,sim) in enumerate(NB.get(c,[])):
            tt=SPT.get(t)
            if tt is None: continue
            for ad in set(q)&set(tt):
                s,n=mcos(q[ad],tt[ad])
                if sim>=0.999 and not y:
                    if i==0: selfc=max(selfc,s if n>=4 else 0.0)
                    continue
                if n>=4: rr.append((j,sim,s))
        rows.append(rr)
    RAW[r.molecule_id]=(rows,selfc,np.array([Y[(r.molecule_id,i+1)] for i in range(len(lst))]))
pickle.dump(RAW,open('/root/lab/an_raw30.pkl','wb'))
def ev(K,LO,HI,lam,agg='sum',pw=1.0,gate=0.6):
    s=0;t1=0
    for m,(rows,selfc,y) in RAW.items():
        e=np.array([ (sum(sim**pw*sc for j,sim,sc in rr if j<K and LO<=sim<HI) if agg=='sum' else max([sim**pw*sc for j,sim,sc in rr if j<K and LO<=sim<HI],default=0.0)) for rr in rows])
        sco=-np.arange(len(e)).astype(float)
        if selfc<gate and e.max()>0: sco=z(-np.log(np.arange(1,len(e)+1)))+lam*z(e)
        o=np.argsort(-sco,kind='stable'); h=np.flatnonzero(y[o])
        if len(h): s+=1/(h[0]+1); t1+=h[0]==0
    return round(float(s/N),4)
print('K8 LO.4 (current)',[ev(8,0.4,0.85,l) for l in (0.5,0.75,1.0)])
for K in (8,15,30):
    for LO in (0.25,0.3,0.4):
        print('K',K,'LO',LO,'sum',[ev(K,LO,0.85,l) for l in (0.5,0.75,1.0,1.5)],'max',[ev(K,LO,0.85,l,'max') for l in (0.5,0.75,1.0)],flush=True)
print('K30 LO.25 pw2',[ev(30,0.25,0.85,l,pw=2.0) for l in (0.5,0.75,1.0,1.5)],'pw3',[ev(30,0.25,0.85,l,pw=3.0) for l in (0.5,0.75,1.0,1.5)])
