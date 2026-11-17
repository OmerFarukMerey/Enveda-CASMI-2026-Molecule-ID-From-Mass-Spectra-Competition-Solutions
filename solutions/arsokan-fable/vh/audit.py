"""vh stage audit: every saved n0 stage snapshot, consecutive deltas per class with gain/loss sides."""
import pandas as pd, numpy as np, pickle, os, sys
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*'); _te=rdMolStandardize.TautomerEnumerator(); _kc={}
def key(s):
    if s not in _kc:
        try: m=Chem.MolFromSmiles(s); _kc[s]=Chem.MolToInchiKey(_te.Canonicalize(m))[:14] if m is not None else None
        except Exception: _kc[s]=None
    return _kc[s]
O='out_full/'; tr=pd.read_parquet(O+'vh/truth.parquet'); tr['mk']=[key(s) for s in tr.smiles]
def load(p):
    s=pd.read_csv(O+p,dtype=str,keep_default_na=False); L=dict(zip(s.molecule_id,s.smiles)); return {m:[x for x in L[m].split(';') if x][:25] for m in tr.molecule_id}
def rr(L): return np.array([next((1/(i+1) for i,x in enumerate(L[m]) if key(x)==k),0.0) for m,k in zip(tr.molecule_id,tr.mk)])
ST=[('pipeline reference (e004)','submission_reference.csv'),('e006 control','submission_e006_control.csv'),('frozen 0.465 final','submission_baseline0465.csv'),('V0 consensus, no neural','variants/submission_V0.csv'),
    ('V1 neural 0.25 (public 0.466)','submission_public0466.csv'),('V2 leader-safe (neural 0.5 on 2-25)','variants/submission_V2.csv'),('CUSTOM (neural 0.75 on 2-25)','variants/submission_CUSTOM.csv'),
    ('control_V2 file','submission_control_V2.csv'),('merge: fragment check + doc/lib/NP prior + mass prior','submission_before_mass_lead.csv'),('mass rank-1 rule','submission_before_rr8.csv'),
    ('pre N-1c','submission_pre_n1c.csv'),('N-1c LAM 0.35','submission_pre_apex_extlib.csv'),('extlib overlay (final)','submission.csv')]
LL={n:load(p) for n,p in ST}; RR={n:rr(L) for n,L in LL.items()}; cls=tr.cls.values; rows=[]
for (a,_),(b,_) in zip(ST[:-1],ST[1:]):
    d=RR[b]-RR[a]; same=sum(LL[a][m]==LL[b][m] for m in tr.molecule_id); row=dict(step=f'{a} -> {b}',identical_lists=same,MRR=round(RR[b].mean(),4),delta=round(d.mean(),4),win=int((d>0).sum()),loss=int((d<0).sum()),gain_side=round(d[d>0].sum()/len(d),4),loss_side=round(d[d<0].sum()/len(d),4),r1ok_demoted=int(((RR[a]==1)&(RR[b]<1)).sum()),lead_changed=int(sum(LL[a][m][0]!=LL[b][m][0] for m in tr.molecule_id)))
    for c in ('C1x','C1s','C2'):
        k=cls==c; row[c]=f"{d[k].mean():+.4f} w{int((d[k]>0).sum())}/l{int((d[k]<0).sum())} dem{int(((RR[a][k]==1)&(RR[b][k]<1)).sum())} g{d[k][d[k]>0].sum()/k.sum():+.4f} L{d[k][d[k]<0].sum()/k.sum():+.4f}"
    rows.append(row)
o=pd.DataFrame(rows); pd.set_option('display.width',400); pd.set_option('display.max_columns',40); pd.set_option('display.max_colwidth',70); print(o.to_string(index=False)); o.to_csv(O+'stage_audit.csv',index=False)
print({n:round(v.mean(),4) for n,v in RR.items()}); pickle.dump(dict(LL=LL,RR=RR,ST=ST),open(O+'stages.pkl','wb'))
