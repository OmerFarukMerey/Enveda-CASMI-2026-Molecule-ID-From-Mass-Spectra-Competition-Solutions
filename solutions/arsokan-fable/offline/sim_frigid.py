import json,sys,pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
def k14(s):
    try:
        m=Chem.MolFromSmiles(s); return Chem.MolToInchiKey(m)[:14] if m else None
    except Exception: return None
R=[json.loads(l) for l in open('/root/lab/fr1/frigid_gen.jsonl')]
for st,ds,sub in (('v3','/root/valds3','/root/val/out3/submission.csv'),('v2','/root/valds2','/root/val/out2/submission.csv')):
    tr=pd.read_parquet(ds+'/val_truth.parquet'); 
    if st=='v3': print(tr.columns.tolist())
    scol=[c for c in tr.columns if 'smiles' in c.lower()][0]
    T={r.molecule_id:k14(getattr(r,scol)) for r in tr.itertuples()}
    S=pd.read_csv(sub).fillna('')
    L={r.molecule_id:[k14(x) for x in r.smiles.split(';') if x] for r in S.itertuples()}
    G={r['molecule_id']:r for r in R if r['set']==st}
    folds=sorted({r.get('fold') for r in G.values()})
    for fold in folds:
        ids=[m for m in G if G[m].get('fold')==fold]
        def rank(l,t): return l.index(t)+1 if t in l else 0
        base=[rank(L.get(m,[])[:25],T[m]) for m in ids]
        mrr=lambda rs: sum(1/r for r in rs if r)/len(rs)
        absent=[m for m,b in zip(ids,base) if b==0]
        ha=sum(1 for m in absent if G[m].get('hit',0)>0); ha1=sum(1 for m in absent if G[m].get('hit',0)==1)
        print(st,fold,'n',len(ids),'base',round(mrr(base),4),'absent',len(absent),'hit among absent',ha,'at gen1',ha1)
        for thr in (0,0.3,0.4,0.5,0.6,0.7):
          for pos in (2,3):
            new=[];ins=0;gain=0;loss=0
            for m,b in zip(ids,base):
                g=G[m]; l=list(L.get(m,[])[:25])
                if g.get('gen') and g['score'][0]>=thr and g['keys'][0] not in l:
                    l.insert(pos-1,g['keys'][0]); l=l[:25]; ins+=1
                r=rank(l,T[m]); new.append(r)
                a=(1/r if r else 0)-(1/b if b else 0)
                if a>0: gain+=1
                if a<0: loss+=1
            print('   thr',thr,'pos',pos,'ins',ins,'mrr',round(mrr(new),4),'delta',round(mrr(new)-mrr(base),4),'up',gain,'down',loss)
