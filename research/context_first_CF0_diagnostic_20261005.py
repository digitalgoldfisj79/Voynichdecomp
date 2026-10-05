#!/usr/bin/env python3
import urllib.request, json, numpy as np, collections
from sklearn.metrics import roc_auc_score
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cfmod"};exec(compile(urllib.request.urlopen(URL).read().decode(),URL,"exec"),m)
rows,_=m["synth_rows"](False)
types,dc,vc,tc=m["eligible_types"](rows);vmap,_=m["context_vocab"](rows)
base,glob=m["fit_baseline"](rows,vmap,(2,3));obs,exp=m["profiles"](rows,vmap,(2,3),set(types),base,glob)
E=m["embedding"](types,obs,exp);sim=E@E.T
val=[r for r in rows if r["fold"]==4]
lags=(-2,-1,1,2);li={z:i for i,z in enumerate(lags)}
def ll_for(r,mm,lagset):
    ll=0;n=0
    for lag in lagset:
        j=m["cat_of"](r[f"n{lag:+d}"],vmap)
        if j is None:continue
        p0=base.get(m["nuis_key"](r,lag),glob[lag]);q=p0*mm[li[lag]];q=q/q.sum()
        ll+=np.log2(max(float(q[j]),1e-15));n+=1
    return ll/n if n else None
def pair_panels(a,b):
    ma=m["multiplier"](obs[a],exp[a]);mb=m["multiplier"](obs[b],exp[b])
    mp=m["multiplier"](obs[a]+obs[b],exp[a]+exp[b])
    rr=[r for r in val if r["token"] in (a,b)]
    outs={}
    for name,ls in [("all",lags),("left",(-2,-1)),("right",(1,2)),("near",(-1,1)),("far",(-2,2))]:
        d=[]
        for r in rr:
            sep=ma if r["token"]==a else mb
            x=ll_for(r,sep,ls);y=ll_for(r,mp,ls)
            if x is not None and y is not None:d.append(y-x)
        outs[name]=float(np.mean(d)) if d else None
    # block stability across 4 sequential val chunks
    ds=collections.defaultdict(list)
    for r in rr:
        sep=ma if r["token"]==a else mb
        x=ll_for(r,sep,lags);y=ll_for(r,mp,lags)
        if x is not None and y is not None:ds[r["pos"]//500].append(y-x)
    outs["minblock"]=min((np.mean(v) for v in ds.values()),default=-999)
    outs["posblocks"]=sum(np.mean(v)>0 for v in ds.values());outs["nblocks"]=len(ds)
    return outs
records=[]
for i,a in enumerate(types):
    for j in range(i+1,len(types)):
        b=types[j]
        p=pair_panels(a,b)
        sa=int(a.split("_")[0][1:]);sb=int(b.split("_")[0][1:])
        records.append({"a":a,"b":b,"truth":sa==sb,"cos":float(sim[i,j]),**p})
print("N",len(records),"TRUE",sum(x["truth"] for x in records))
for feature in ["cos","all","left","right","near","far","minblock","posblocks"]:
    vals=np.array([x[feature] for x in records],float);y=np.array([x["truth"] for x in records],int)
    try:auc=roc_auc_score(y,vals)
    except:auc=None
    print("AUC",feature,auc)
rules=[
 ("lr",lambda x:x["all"]>0 and x["left"]>0 and x["right"]>0),
 ("lr_cos",lambda x:x["all"]>0 and x["left"]>0 and x["right"]>0 and x["cos"]>.25),
 ("4panel",lambda x:x["all"]>0 and x["left"]>0 and x["right"]>0 and x["near"]>0 and x["far"]>0),
 ("4panel_cos",lambda x:x["all"]>0 and x["left"]>0 and x["right"]>0 and x["near"]>0 and x["far"]>0 and x["cos"]>.25),
 ("block",lambda x:x["all"]>0 and x["left"]>0 and x["right"]>0 and x["posblocks"]>=3),
]
for name,fn in rules:
    rr=[x for x in records if fn(x)]
    print("RULE",name,"n",len(rr),"true",sum(x["truth"] for x in rr),"precision",sum(x["truth"] for x in rr)/len(rr) if rr else None,
          "top",[(x["a"],x["b"],x["truth"],round(x["all"],3),round(x["left"],3),round(x["right"],3),round(x["cos"],3)) for x in sorted(rr,key=lambda z:z["all"],reverse=True)[:15]])
print("TRUE_TOP",[(x["a"],x["b"],round(x["all"],3),round(x["left"],3),round(x["right"],3),round(x["cos"],3)) for x in sorted([x for x in records if x["truth"]],key=lambda z:z["all"],reverse=True)[:30]])
