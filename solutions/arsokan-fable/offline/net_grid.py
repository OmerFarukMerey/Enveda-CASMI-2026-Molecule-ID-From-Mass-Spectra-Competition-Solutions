import sys; sys.path.insert(0,'/root/pipe'); from net_eval import *
def show(name,**kw):
    r=[mrr(run(LAM=l,**kw))[0] for l in (0.25,0.35,0.5,0.75)]; print(f'{name:34s}',r,flush=True)
show('default')
for t in (0.4,0.5,0.7,0.8): show(f'TAU {t}',TAU=t)
for p in (1,2,8): show(f'PW {p}',PW=p)
for n in (1,3,10,25): show(f'TOPN {n}',TOPN=n)
show('nbw uni',nbw='uni'); show('nbw score',nbw='score')
show('agg max',agg='max'); show('base score',base='score')
show('iters 2',iters=2); show('iters 3',iters=3)
for m in (1,3): show(f'maxnb {m}',maxnb=m)
for s in (2,3): show(f'simpow {s}',simpow=s)
show('selfnorm',selfnorm=True)
