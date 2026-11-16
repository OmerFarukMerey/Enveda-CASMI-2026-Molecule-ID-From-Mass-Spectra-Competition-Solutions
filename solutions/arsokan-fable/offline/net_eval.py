import pandas as pd, numpy as np, pickle, itertools
from rdkit import DataStructs
import sys; sys.path.insert(0,'/root/pipe'); from net_bench import fp
L,sp=pickle.load(open('/root/lab/net_bench.pkl','rb')); E,_=pickle.load(open('/root/lab/net_edges.pkl','rb'))
keys=sorted(L); N=len(keys)
z=lambda v:(v-v.mean())/(v.std()+1e-9)
def mrr(R):
    s=0;t1=0
    for k in keys:
        lst=R[k]; y=dict(zip(L[k][0],L[k][1]))
        for i,c in enumerate(lst[:25]):
            if y[c]: s+=1/(i+1); t1+=i==0; break
    return round(s/N,4), round(t1/N,4)
TC={}
def tani(a,b):
    k=(a,b) if a<b else (b,a)
    if k not in TC:
        fa,fb=fp(a),fp(b); TC[k]=DataStructs.TanimotoSimilarity(fa,fb) if fa is not None and fb is not None else 0.0
    return TC[k]
def run(LAM=0.35,TAU=0.6,PW=4,TOPN=5,iters=1,nbw='rr',agg='sum',base='logrank',maxnb=None,simpow=1.0,selfnorm=False):
    adj={}
    for a,b,w in E:
        if w>=TAU: adj.setdefault(a,[]).append((b,w)); adj.setdefault(b,[]).append((a,w))
    cur={k:list(L[k][0]) for k in keys}
    for _ in range(iters):
        new={}
        for k in keys:
            lst=cur[k]; nb=adj.get(k,[])
            if maxnb: nb=sorted(nb,key=lambda x:-x[1])[:maxnb]
            if not nb or len(lst)<2: new[k]=lst; continue
            if base=='logrank': b0=-np.log(np.arange(1,len(lst)+1))
            else:
                sc=dict(zip(L[k][0],L[k][2])); b0=np.array([sc[c] for c in lst])
            Nn=np.zeros(len(lst)); W=0
            for b,w in nb:
                lb=cur[b][:TOPN]
                if nbw=='rr': p=1/np.arange(1,len(lb)+1)
                elif nbw=='uni': p=np.ones(len(lb))
                else:
                    sc=dict(zip(L[b][0],L[b][2])); v=np.array([sc[c] for c in lb]); p=np.exp(v-v.max())
                p=p/p.sum()
                contrib=np.array([sum(pp*tani(c,j)**simpow for j,pp in zip(lb,p)) if agg=='sum' else max(pp*tani(c,j)**simpow for j,pp in zip(lb,p)) for c in lst])
                Nn+=w**PW*contrib; W+=w**PW
            if selfnorm: Nn/=W
            o=np.argsort(-(z(b0)+LAM*z(Nn)),kind='stable'); new[k]=[lst[i] for i in o]
        cur=new
    return cur
if __name__=='__main__':
    print('baseline',mrr({k:L[k][0] for k in keys}))
    for lam in (0.15,0.25,0.35,0.5,0.75,1.0,1.5): print('LAM',lam,mrr(run(LAM=lam)))
