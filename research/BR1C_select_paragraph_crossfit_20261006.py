#!/usr/bin/env python3
"""BR1C — corrected M3 SELECT analogue with paragraph reinforcement, nested 5-fold cross-fit."""
import urllib.request,json,collections,math,re,numpy as np
from scipy.optimize import minimize

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1c68aa15881f4c2e2a86b59e4569fe80b0f3c0c5/research/BR1_cf_neighbourhood_realisation_20261006.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode();pre=src.split("\nOUT={}\n")[0]
m={"__name__":"br1"};exec(compile(pre,BASE,"exec"),m)

TIDS=("ZLZI","TTLI")
L2GRID=m["L2GRID"];SEED=20261006
THETA_PARA=np.array([
0.16569868846755956,-0.004800778588598077,0.08134137302324031,0.1284889708755418 if False else 0.1284889708755418,
-0.23106478535126374,0.17275149244886126,0.4010122548701616,0.20039946147059426,
0.2720682558961095,0.5304866649116893,0.24937232912585222,0.11481924059295087
],float)
# Correct exact value from frozen MRC source.
THETA_PARA[3]=0.1284889708755418
# preserve source coefficient at full precision
THETA_PARA[3]=0.1284889708755418
ALPHA=30.0

# Paragraph ranges are already loaded by latent_line_state_test in the base module.
PR=m["lat"]["PR"]

def line_num(x):
    z=re.search(r"\d+",str(x));return int(z.group()) if z else 0

def para_id(f,line):
    for a,b,p in PR.get(f,()):
        if a<=line<=b:return int(p)
    return 10000+int(line)

def corrected_dataset(rows,tid,fitfolds):
    train=[r for r in rows if r["fold"] in fitfolds]
    prior,tabs,global_k,base_k=m["count_tables"](train,tid)
    mp=m["NODECOMP"][tid];cs=m["COMP"][tid]
    supported={ci:tuple(x for x in c if prior[ci][x]>0) for ci,c in enumerate(cs)}
    page=collections.defaultdict(lambda:np.zeros(m["K12"],float))
    para=collections.defaultdict(lambda:np.zeros(m["K12"],float))
    linehist=collections.defaultdict(list)
    out=[]
    for r in rows:
        ci=mp.get(r["token"])
        ln=line_num(r["line"]);pk=(r["folio"],para_id(r["folio"],ln))
        if r["pos"]>0 and ci is not None and len(supported.get(ci,()))>=2:
            cand=supported[ci]
            basep=np.array([m["smooth_prob"](prior[ci],cand,c) for c in cand],float);basep/=basep.sum()
            key=(r["section"],m["k12_final"](r["p1"]))
            cc=base_k.get(key)
            if cc is None:base=global_k.copy()
            else:
                n=float(cc.sum());base=(cc+ALPHA*global_k)/(n+ALPHA)
            rc=np.zeros(m["K12"],float)
            for z in linehist[(r["folio"],r["line"])][-6:]:
                if z>=0:rc[z]+=1
            sc=np.log(np.maximum(base,1e-15))
            sc += m["THETA_PAGE"]*np.log1p(page[r["folio"]])
            sc += THETA_PARA*np.log1p(para[pk])
            sc += m["THETA_RECENT"]*rc
            sc-=sc.max();sp=np.exp(sc);sp/=sp.sum()
            X=[]
            for c in cand:
                ks=m["k12_start"](c);X.append([math.log(max(float(sp[ks]) if ks>=0 else 1e-12,1e-12))])
            out.append({"r":r,"ci":ci,"cand":cand,"y":cand.index(r["token"]),"base":basep,
                        "X":{"select_corr":np.asarray(X,float)}})
        if r["kstart"]>=0:
            page[r["folio"]][r["kstart"]]+=1
            para[pk][r["kstart"]]+=1
            linehist[(r["folio"],r["line"])].append(r["kstart"])
    return out

def fm(item,blocks):
    a=[item["X"][b] for b in blocks]
    return np.hstack(a) if a else np.zeros((len(item["cand"]),0),float)
m["fit_beta"].__globals__["feature_matrix"]=fm
m["score"].__globals__["feature_matrix"]=fm

def choose(tr,va):
    rec=[]
    for l2 in L2GRID:
        b=m["fit_beta"](tr,("select_corr",),l2);s=m["score"](va,("select_corr",),b)
        g=float(np.mean([x[0] for x in s])) if s else -1e9
        rec.append((-g,l2,b))
    rec.sort(key=lambda x:(x[0],x[1]))
    return rec[0][1],-rec[0][0]

def agg(rows,seed):
    vals=np.array([x["g"] for x in rows],float)
    by=collections.defaultdict(list)
    for x in rows:by[x["bif"]].append(x["g"])
    bm=np.array([np.mean(v) for v in by.values()],float)
    se=float(bm.std(ddof=1)/math.sqrt(len(bm)))
    rng=np.random.default_rng(seed)
    null=np.array([np.mean(bm*rng.choice((-1.,1.),len(bm))) for _ in range(10000)],float)
    return {"event_mean":float(vals.mean()),"bif_mean":float(bm.mean()),"se":se,
            "z_bif":float(bm.mean()/se),"null_mean":float(null.mean()),"null_sd":float(null.std(ddof=1)),
            "null_z":float((bm.mean()-null.mean())/null.std(ddof=1)),
            "fold_means":{str(f):float(np.mean([x["g"] for x in rows if x["fold"]==f])) for f in range(5)},
            "n":len(vals),"bifs":len(bm)}

OUT={}
for tid in TIDS:
    rows=m["enrich"](m["build_rows"](tid));pred=[];splits=[]
    for outer in range(5):
        val=(outer+1)%5;trfolds=tuple(f for f in range(5) if f not in (outer,val))
        D=corrected_dataset(rows,tid,trfolds);tr=[x for x in D if x["r"]["fold"] in trfolds];va=[x for x in D if x["r"]["fold"]==val]
        l2,vg=choose(tr,va)
        fitfolds=tuple(f for f in range(5) if f!=outer);F=corrected_dataset(rows,tid,fitfolds)
        fit=[x for x in F if x["r"]["fold"] in fitfolds];te=[x for x in F if x["r"]["fold"]==outer]
        b=m["fit_beta"](fit,("select_corr",),l2);sc=m["score"](te,("select_corr",),b)
        splits.append({"outer":outer,"val":val,"l2":l2,"val_gain":vg,"beta":b.tolist()})
        for g,it in sc:pred.append({"g":float(g),"bif":it["r"]["bif"],"fold":outer})
    OUT[tid]={"stat":agg(pred,SEED+sum(map(ord,tid))),"splits":splits}
    print("BR1C_"+tid+"="+json.dumps(OUT[tid],separators=(",",":")),flush=True)
print("BR1C_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
