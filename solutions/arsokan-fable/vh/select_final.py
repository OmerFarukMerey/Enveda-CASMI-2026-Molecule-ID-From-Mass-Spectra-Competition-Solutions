"""vh step 2: stratified, network-enriched sample -> selection.json (full) and selection_smoke.json (16 molecules)."""
import pandas as pd, numpy as np, pickle, json, hashlib, collections, sys
P=pickle.load(open('pool.pkl','rb')); m=pd.read_parquet('train_meta.parquet'); Q=pd.read_parquet('pool_query_rows.parquet')
SEED='vh-20261117'; rng=np.random.default_rng(20261117); N=dict(C1x=110,C1s=60,C2=130)
X=set(P['X']); E=set(P['E']); hold=P['hold']; val=P['val']; EDG=P['EDG']
nb=collections.defaultdict(set)
for (a,b) in EDG: nb[a].add(b); nb[b].add(a)
QADD=['[M+H]+','[M-H]-','[M+CH2O2-H]-','[M+Na]+','[M+NH4]+','[M+K]+','[M+Cl]-']; ADD=('[M+H]+','[M-H]-')
mk=m[m.inchikey14.isin(X|E)]; byk={k:g for k,g in mk.groupby('inchikey14')}
QCE={'40','60','-40','-60'}
def plan(k,cls):
    g=byk[k]; en=g[g.ingest_lib=='enveda-180']; qf=en[en.adduct.isin(QADD)].sort_values('row')
    if cls=='C1s':
        q=qf[qf.collision_energy_orig.astype(str).isin(QCE)]; rest=qf[~qf.row.isin(q.row)]
        ok=any(ad in set(q.adduct) and ad in set(rest.adduct) for ad in ADD)
        if not ok: return None
        q=q.head(9); return dict(q=q.row.tolist(),d=q.row.tolist(),left=g[~g.row.isin(q.row)])
    q=qf.head(9)
    if not len(q[q.adduct.isin(ADD)]): return None
    d=g.row.tolist() if cls=='C2' else en.row.tolist(); return dict(q=q.row.tolist(),d=d,left=g[~g.row.isin(d)])
tier=lambda k: 0 if (k in hold and k in val) else (1 if k in hold else (2 if k in val else 3))
def run(p_nb, quotas):
    left=dict(quotas); chosen={}
    xs=sorted(X,key=lambda k:(tier(k),hashlib.sha1((SEED+k).encode()).hexdigest())); es=sorted(E,key=lambda k:hashlib.sha1((SEED+k).encode()).hexdigest())
    def take(k):
        if k in chosen: return False
        opts=['C1x'] if k in X else (['C1s','C2'] if rng.random()<left['C1s']/max(left['C1s']+left['C2'],1) else ['C2','C1s'])
        for c in opts:
            if left.get(c,0)<=0: continue
            pl=plan(k,c)
            if pl is None: continue
            chosen[k]=(c,pl); left[c]-=1; return True
        return False
    order=[]; ix=ie=0
    while sum(left.values())>0 and (ix<len(xs) or ie<len(es)):
        wantx=left['C1x']>0 and ix<len(xs)
        if wantx and (rng.random()<left['C1x']/sum(left.values()) or ie>=len(es)): k=xs[ix]; ix+=1
        elif ie<len(es): k=es[ie]; ie+=1
        else: k=xs[ix]; ix+=1
        if take(k) and rng.random()<p_nb:
            for j in list(rng.permutation(sorted(nb[k])))[:2]: take(j)
    ks=set(chosen); ne=sum(1 for (a,b) in EDG if a in ks and b in ks); return chosen,ne
best=None
for p in (0.0,0.1,0.2,0.3,0.4,0.5):
    rng=np.random.default_rng(20261117); ch,ne=run(p,N); print('p_nb',p,'molecules',len(ch),'pool-graph edges',ne,collections.Counter(c for c,_ in ch.values()))
    if best is None or abs(ne-82)<abs(best[2]-82): best=(p,ch,ne)
p,ch,ne=best; print('chosen p',p,'edges',ne)
smi=Q.drop_duplicates('inchikey14').set_index('inchikey14').normalized_smiles.to_dict(); mols=[]
for k,(c,pl) in sorted(ch.items()):
    L=pl['left']
    mols.append(dict(mid='V'+hashlib.sha1((SEED+'|mid|'+k).encode()).hexdigest()[:10],key=k,cls=c,smiles=smi[k],q=[int(x) for x in pl['q']],d=[int(x) for x in pl['d']],
                     libs_left=';'.join(sorted(set(L.ingest_lib))),n_left=int(len(L)),left_add=';'.join(sorted(set(L[L.adduct.isin(ADD)].adduct))),left_instr=';'.join(sorted(set(L.instrument_type.astype(str)))),
                     q_add=';'.join(sorted(set(m.loc[pl['q']].adduct))),fpnet_hold=bool(k in hold),sanpier_val=bool(k in val)))
mols.sort(key=lambda r:r['mid']); sel=dict(seed=SEED,n_train_rows=int(len(m)),mols=mols); json.dump(sel,open('selection.json','w'))
d=pd.DataFrame(mols); print(d.groupby('cls').agg(n=('mid','size'),qrows=('q',lambda x:np.mean([len(v) for v in x])),fpnet_hold=('fpnet_hold','mean'),sanpier_val=('sanpier_val','mean'),n_left=('n_left','mean')).round(2))
print('C1x libs left',d[d.cls=='C1x'].libs_left.value_counts().to_dict()); print('C1x with same-polarity [M+H]+/[M-H]- reference left',int((d[d.cls=='C1x'].left_add!='').sum()),'| instruments left',collections.Counter(i for s in d[d.cls=='C1x'].left_instr for i in s.split(';')).most_common(6))
print('query adducts',collections.Counter(a for s in d.q_add for a in s.split(';')).most_common(), 'total query rows',sum(len(v) for v in d.q), 'rows dropped',sum(len(v) for v in d.d))
# smoke: 16 molecules, all classes, including some linked pairs
ks={r['key']:r for r in mols}; pairs=[(a,b) for (a,b) in EDG if a in ks and b in ks][:3]; sm=[]
for a,b in pairs: sm+= [ks[a],ks[b]]
for c,n in (('C1x',5),('C1s',3),('C2',4)): sm+=[r for r in mols if r['cls']==c and r not in sm][:n]
sm={r['mid']:r for r in sm}; sm=sorted(sm.values(),key=lambda r:r['mid'])[:16]; json.dump(dict(seed=SEED,n_train_rows=int(len(m)),mols=sm),open('selection_smoke.json','w')); print('smoke',collections.Counter(r['cls'] for r in sm))
