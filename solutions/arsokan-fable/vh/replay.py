"""vh acceptance replay: run the known stages (N-1c LAM curve, TA-1, vpd, w30c) offline on the harness kernel's saved lists.
argv: out_dir train_parquet. Stage code is executed verbatim from the notebooks; only paths are redirected."""
import json, os, sys, shutil, re, io, contextlib, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
R='/home/user/Enveda-CASMI-2026-Molecule-ID-From-Mass-Spectra-Competition-Solutions/'
OUT=sys.argv[1]; TRAIN=sys.argv[2]; W=os.path.join(OUT,'replay'); shutil.rmtree(W,ignore_errors=True); os.makedirs(W)
INP=os.path.join(W,'inp','mass-spectra'); os.makedirs(INP); shutil.copy(os.path.join(OUT,'vh/test.parquet'),INP+'/test.parquet'); os.symlink(os.path.abspath(TRAIN),INP+'/train.parquet')
cellsrc=lambda p,i: (lambda c: ''.join(c['source']) if isinstance(c['source'],list) else c['source'])(json.load(open(p))['cells'][i])
N1C=cellsrc(R+'submissions/n0-0.492/notebook.ipynb',21); N1C=N1C[:N1C.index('try:\n    _n1c()')]
TA=open(R+'solutions/arsokan-fable/stages/train_analog_rerank.py').read()
VPD=cellsrc(R+'scratch/om-vpd/notebook.ipynb',23); VPD=VPD[:VPD.index('try:\n    _v1()')]
W30=cellsrc(R+'scratch/om-w30c/notebook.ipynb',23); W30=W30[:W30.index('try:\n    _v1()')]
_te=rdMolStandardize.TautomerEnumerator(); _kc={}
def key(s):
    if s not in _kc:
        try:
            m=Chem.MolFromSmiles(s); _kc[s]=Chem.MolToInchiKey(_te.Canonicalize(m))[:14] if m is not None else None
        except Exception: _kc[s]=None
    return _kc[s]
tr=pd.read_parquet(os.path.join(OUT,'vh/truth.parquet')); tr['mk']=[key(s) for s in tr.smiles]; TK=dict(zip(tr.molecule_id,tr.mk))
rd=lambda p: pd.read_csv(p,dtype=str,keep_default_na=False)
def rrs(path):
    s=rd(path); L=dict(zip(s.molecule_id,s.smiles)); out=[]
    for m in tr.molecule_id:
        g=[x for x in L[m].split(';') if x][:25]; h=next((i+1 for i,x in enumerate(g) if key(x)==TK[m]),0); out.append(1.0/h if h else 0.0)
    return np.array(out)
pre_n1c=os.path.join(OUT,'submission_pre_n1c.csv'); pre_ov=os.path.join(OUT,'submission_pre_apex_extlib.csv'); final=os.path.join(OUT,'submission.csv')
# overlay decisions of the kernel run (LAM 0.35): molecules whose rank 1 it replaced, and with what
a=rd(pre_ov); b=rd(final); OV={}
for m,x,y in zip(a.molecule_id,a.smiles,b.smiles):
    if x!=y: OV[m]=y.split(';')[0]
print('overlay changed lists in the kernel run:',len(OV))
def apply_overlay(path):   # re-apply the kernel's overlay promotions to another pre-overlay list (same rule: promoted structure to rank 1)
    if not OV: return
    s=rd(path); out=[]
    for m,x in zip(s.molecule_id,s.smiles):
        if m in OV:
            l=[c for c in x.split(';') if c]; k=key(OV[m]); l=[OV[m]]+[c for c in l if key(c)!=k]; x=';'.join(l[:25])
        out.append(x)
    s['smiles']=out; s.to_csv(path,index=False)
def stage(name,src_csv,fn):
    d=os.path.join(W,name); os.makedirs(d,exist_ok=True); shutil.copy(src_csv,d+'/submission.csv'); shutil.copy(os.path.join(OUT,'frozen_candidate_evidence.parquet'),d+'/frozen_candidate_evidence.parquet')
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): fn(d)
    log=[l for l in buf.getvalue().splitlines() if l.startswith('[')]; return d+'/submission.csv',log
RES={}; LOG={}
RES['base: pipeline before N-1c']=rrs(pre_n1c); RES['n0 kernel output (LAM 0.35 + overlay)']=rrs(final)
def n1c(lam):
    def f(d):
        g={}; exec(N1C,g); g['_n1c'](work=d,inp=os.path.join(W,'inp'),LAM=lam,TAU=0.6)
    return f
N1OUT={}
for lam in (0.10,0.20,0.25,0.30,0.35,0.38,0.40,0.42,0.45,0.50):
    p,log=stage(f'n1c_{lam}',pre_n1c,n1c(lam)); N1OUT[lam]=p
    if lam==0.35:
        same=(rd(p).smiles==rd(pre_ov).smiles).mean(); print('offline N-1c 0.35 reproduces the kernel pre-overlay file on',round(float(same),4),'of lists'); LOG['n1c 0.35']=log
        shutil.copy(p,os.path.join(W,'n1c035_noov.csv'))
    apply_overlay(p); RES[f'N-1c LAM {lam:.2f} (+overlay)']=rrs(p)
BASE35=os.path.join(W,'n1c035_noov.csv')
def ta(d):
    g={}; exec(TA,g); g['_ta_run'](sub_path=d+'/submission.csv',test_path=INP+'/test.parquet',train_path=INP+'/train.parquet')
def cell(src):
    def f(d):
        g={}; exec(src,g); g['_v1'](work=d,inp=os.path.join(W,'inp'))
    return f
for name,fn in (('TA-1 (t0)',ta),('vpd veto',cell(VPD)),('w30c veto + confirmation',cell(W30))):
    p,log=stage(re.sub(r'\W+','_',name),BASE35,fn); LOG[name]=log; apply_overlay(p); RES[name]=rrs(p)
ref=RES['N-1c LAM 0.35 (+overlay)']; rng=np.random.default_rng(0); rows=[]; cls=tr.cls.values
for name,v in RES.items():
    d=v-ref; ci=np.percentile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[2.5,97.5]); row=dict(stage=name,MRR=round(v.mean(),4),delta_vs_LAM035=round(d.mean(),4),ci=f'[{ci[0]:+.4f},{ci[1]:+.4f}]',win=int((d>0).sum()),loss=int((d<0).sum()),r1ok_demoted=int(((ref==1)&(v<1)).sum()))
    for c in ('C1x','C1s','C2'):
        k=cls==c; row['MRR_'+c]=round(v[k].mean(),4); row['d_'+c]=round(d[k].mean(),4); row['wl_'+c]=f'{int((d[k]>0).sum())}/{int((d[k]<0).sum())}'
    c1=(cls!='C2');
    for s1 in (0.3,0.4,0.5): row[f'blend_c1={int(s1*100)}%']=round(s1*d[c1].mean()+0.45*d[cls=="C2"].mean(),4)
    rows.append(row)
o=pd.DataFrame(rows); o.to_csv(os.path.join(OUT,'acceptance.csv'),index=False); pd.set_option('display.width',330); pd.set_option('display.max_columns',40)
print(o.to_string(index=False))
for k,v in LOG.items(): print('LOG',k,'|',' || '.join(v))
print('n',len(tr),dict(pd.Series(cls).value_counts()),'| base top1',round(float((ref==1).mean()),4),'recall',round(float((ref>0).mean()),4), {c: (round(float((ref[cls==c]==1).mean()),3), round(float((ref[cls==c]>0).mean()),3)) for c in ('C1x','C1s','C2')})
