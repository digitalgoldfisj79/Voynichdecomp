#!/usr/bin/env python3
"""
BR1 — frozen CF-neighbourhood realisation bridge.
Preregistered 2026-10-06.

Question: conditional on an already-frozen CF neighbourhood, what prospective
information predicts which exact surface node is realised?

No CF rediscovery. No target/current ED shell. Future context appears only in
the explicitly non-causal M5 diagnostic.
"""
import collections, hashlib, json, math, os, urllib.request
import numpy as np
from scipy.optimize import minimize

SEED=20261006
TIDS=("ZLZI","ZLZB","TTLI")
L2GRID=(0.03,0.1,0.3,1.0,3.0,10.0,30.0)
N_SIGN=5000
ALPHA=.5

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
core={"__name__":"br1_core"};exec(compile(urllib.request.urlopen(CORE_URL,timeout=120).read().decode(),CORE_URL,"exec"),core)
build_rows=core["build_rows"]; safe_form=core["safe_form"]; fnum=core["fnum"]

OCC_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/research/hf_emergent_occupancy_fold.py"
occ={"__name__":"br1_occ"};exec(compile(urllib.request.urlopen(OCC_URL,timeout=120).read().decode(),OCC_URL,"exec"),occ)
davis_hand=occ["davis_hand"]

LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
lat={"__name__":"br1_lat"};exec(compile(urllib.request.urlopen(LAT_URL,timeout=120).read().decode(),LAT_URL,"exec"),lat)
segment=lat["segment"]; ST=lat["ST"]; K12=lat["K"]

THETA_PAGE=np.array([0.43629794996543747,0.36282605325926687,0.39092158411431266,0.4231810759705418,0.6492318060245758,0.7358884471975975,0.39822673906467176,1.3219154004567564,0.47484899899915256,0.9347775080001645,0.43338661396844674,1.4251068528287683])
THETA_RECENT=0.04840931314842397
SELECT_ALPHA=30.0

EDGES={
"ZLZI":[("aiiin","aiin"),("chcthy","cheey"),("cheey","kaiin"),("cheky","ykeey"),("cheody","opchedy"),("otain","qokar"),("otam","shy"),("qokaiin","qokar"),("qokaiin","qol")],
"ZLZB":[("aiiin","aiin"),("chcthy","cheey"),("cheey","kaiin"),("cheky","ykeey"),("cheody","opchedy"),("otain","qokar"),("otam","shy"),("qokaiin","qokar"),("qokaiin","qol")],
"TTLI":[("aiiin","aiin"),("aiiin","al"),("aiiin","ar"),("checkhy","cheedy"),("cheedy","keedy"),("cheor","dor"),("dair","dor"),("kar","odaiin"),("qokain","qokeedy")]
}

def comps(edges):
    adj=collections.defaultdict(set)
    for a,b in edges: adj[a].add(b);adj[b].add(a)
    seen=set();out=[]
    for s in sorted(adj):
        if s in seen:continue
        st=[s];seen.add(s);cc=[]
        while st:
            x=st.pop();cc.append(x)
            for y in adj[x]:
                if y not in seen:seen.add(y);st.append(y)
        out.append(tuple(sorted(cc)))
    return sorted(out,key=lambda x:(-len(x),x))

COMP={t:comps(EDGES[t]) for t in TIDS}
NODECOMP={t:{x:i for i,c in enumerate(COMP[t]) for x in c} for t in TIDS}

def lev(a,b,cap=4):
    if a is None:return cap
    if abs(len(a)-len(b))>=cap:return cap
    p=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        q=[i]
        for j,y in enumerate(b,1):q.append(min(q[-1]+1,p[j]+1,p[j-1]+(x!=y)))
        p=q
    return min(p[-1],cap)

def k12_start(t):
    try:return int(ST[segment(t)[0]])
    except Exception:return -1
def k12_final(t):
    try:return int(ST[segment(t)[-1]])
    except Exception:return -1

def enrich(rows):
    by=collections.defaultdict(list)
    for r in rows:
        r["hand"]=davis_hand(r["folio"],int(r["line"]))
        r["kstart"]=k12_start(r["token"]);r["kfinal"]=k12_final(r["token"])
        by[(r["folio"],r["line"])].append(r)
    for rs in by.values():
        rs.sort(key=lambda z:z["pos"])
        toks=[r["token"] for r in rs]
        for i,r in enumerate(rs):
            for lag in (1,2):
                r[f"p{lag}"]=rs[i-lag]["token"] if i-lag>=0 else None
                r[f"n{lag}"]=rs[i+lag]["token"] if i+lag<len(rs) else None
            r["prev6"]=tuple(toks[max(0,i-6):i])
            r["lp"]="EARLY" if i<=1 else ("FINAL" if i>=len(rs)-2 else "MID")
    return rows

def eligible_events(rows,tid,foldset):
    mp=NODECOMP[tid]
    return [r for r in rows if r["fold"] in foldset and r["pos"]>0 and r["token"] in mp]

def count_tables(train,tid):
    cs=COMP[tid];mp=NODECOMP[tid]
    prior=collections.defaultdict(collections.Counter)
    tabs={name:collections.defaultdict(collections.Counter) for name in
          ("hand","section","lp","p1s","p1f","p2s","p2f","n1s","n1f","n2s","n2f","compat")}
    global_k=np.ones(K12,float)*.5
    base_k=collections.defaultdict(lambda:np.ones(K12,float)*.5)
    for r in train:
        if r["token"] in mp:
            ci=mp[r["token"]];t=r["token"];prior[ci][t]+=1
            vals={
              "hand":r["hand"],"section":r["section"],"lp":r["lp"],
              "p1s":k12_start(r["p1"]),"p1f":k12_final(r["p1"]),
              "p2s":k12_start(r["p2"]),"p2f":k12_final(r["p2"]),
              "n1s":k12_start(r["n1"]),"n1f":k12_final(r["n1"]),
              "n2s":k12_start(r["n2"]),"n2f":k12_final(r["n2"]),
            }
            for name,v in vals.items():tabs[name][(ci,v)][t]+=1
        if r["pos"]>0 and r["kstart"]>=0:
            global_k[r["kstart"]]+=1
            base_k[(r["section"],k12_final(r["p1"]))][r["kstart"]]+=1
        if r["pos"]>0 and r["p1"] is not None:
            ci=mp.get(r["token"])
            if ci is not None:tabs["compat"][(ci,k12_final(r["p1"]))][r["token"]]+=1
    global_k/=global_k.sum()
    return prior,tabs,global_k,base_k

def smooth_prob(counter,cands,c,alpha=ALPHA):
    n=sum(counter.get(x,0) for x in cands)
    return (counter.get(c,0)+alpha)/(n+alpha*len(cands))

def lr_feature(tab,ci,key,c,cands,prior):
    q=tab.get((ci,key),{})
    return math.log(max(smooth_prob(q,cands,c),1e-12)/max(smooth_prob(prior[ci],cands,c),1e-12))

def select_prob_for_event(r,cands,global_k,base_k,page_counts,recent_classes):
    key=(r["section"],k12_final(r["p1"]))
    cc=base_k.get(key)
    if cc is None:base=global_k.copy()
    else:
        n=float(cc.sum())
        base=(cc+SELECT_ALPHA*global_k)/(n+SELECT_ALPHA)
    sc=np.log(np.maximum(base,1e-15))+THETA_PAGE*np.log1p(page_counts)+THETA_RECENT*recent_classes
    sc-=sc.max();p=np.exp(sc);p/=p.sum()
    return p

BLOCKS=("physical","renderer","select","past")

def make_dataset(rows,tid,fitfolds):
    train=[r for r in rows if r["fold"] in fitfolds]
    prior,tabs,global_k,base_k=count_tables(train,tid)
    mp=NODECOMP[tid];cs=COMP[tid]
    supported={}
    for ci,c in enumerate(cs):
        supported[ci]=tuple(x for x in c if prior[ci][x]>0)
    # prospective page/recent state is recomputed over the full observed sequence;
    # only PREVIOUS realised classes enter the current prediction.
    page=collections.defaultdict(lambda:np.zeros(K12,float))
    linehist=collections.defaultdict(list)
    allitems=[]
    for r in rows:
        ci=mp.get(r["token"])
        if r["pos"]>0 and ci is not None and len(supported.get(ci,()))>=2:
            cand=supported[ci]
            basep=np.array([smooth_prob(prior[ci],cand,c) for c in cand],float);basep/=basep.sum()
            rc=np.zeros(K12,float)
            for z in linehist[(r["folio"],r["line"])][-6:]:
                if z>=0:rc[z]+=1
            sp=select_prob_for_event(r,cand,global_k,base_k,page[r["folio"]],rc)
            X={b:[] for b in BLOCKS};X["future"]=[]
            for c in cand:
                # PHYSICAL: candidate propensities associated with hand/section only.
                X["physical"].append([
                    lr_feature(tabs["hand"],ci,r["hand"],c,cand,prior),
                    lr_feature(tabs["section"],ci,r["section"],c,cand,prior)
                ])
                # RENDERER: prospective ED geometry, line-position prior, near recurrence,
                # and previous-final -> candidate compatibility.
                d=lev(r["p1"],c,4)
                sf=safe_form(c); first8=(sf[1][0] if sf else -1)
                prevf=k12_final(r["p1"])
                X["renderer"].append([
                    float(d==0),float(d==1),float(d==2),float(d==3),float(d>=4),
                    float(c in r["prev6"]),
                    lr_feature(tabs["lp"],ci,r["lp"],c,cand,prior),
                    lr_feature(tabs["compat"],ci,prevf,c,cand,prior),
                    float(first8==(safe_form(r["p1"])[1][-1] if r["p1"] and safe_form(r["p1"]) else -99))
                ])
                ks=k12_start(c)
                X["select"].append([math.log(max(float(sp[ks]) if ks>=0 else 1e-12,1e-12))])
                # PAST CONTEXT: candidate propensity under coarse prior token classes.
                X["past"].append([
                    lr_feature(tabs["p1s"],ci,k12_start(r["p1"]),c,cand,prior),
                    lr_feature(tabs["p1f"],ci,k12_final(r["p1"]),c,cand,prior),
                    lr_feature(tabs["p2s"],ci,k12_start(r["p2"]),c,cand,prior),
                    lr_feature(tabs["p2f"],ci,k12_final(r["p2"]),c,cand,prior)
                ])
                X["future"].append([
                    lr_feature(tabs["n1s"],ci,k12_start(r["n1"]),c,cand,prior),
                    lr_feature(tabs["n1f"],ci,k12_final(r["n1"]),c,cand,prior),
                    lr_feature(tabs["n2s"],ci,k12_start(r["n2"]),c,cand,prior),
                    lr_feature(tabs["n2f"],ci,k12_final(r["n2"]),c,cand,prior)
                ])
            yi=cand.index(r["token"])
            allitems.append({"r":r,"ci":ci,"cand":cand,"y":yi,"base":basep,
                             "X":{k:np.asarray(v,float) for k,v in X.items()}})
        if r["kstart"]>=0:
            page[r["folio"]][r["kstart"]]+=1
            linehist[(r["folio"],r["line"])].append(r["kstart"])
    return allitems

def feature_matrix(item,blocks):
    arr=[]
    for b in blocks:
        arr.append(item["X"][b])
    return np.hstack(arr) if arr else np.zeros((len(item["cand"]),0),float)

def fit_beta(items,blocks,l2):
    d=sum(feature_matrix(items[0],blocks).shape[1] for _ in [0]) if items else 0
    if d==0:return np.zeros(0)
    def fg(beta):
        loss=.5*l2*float(beta@beta);g=l2*beta.copy()
        for it in items:
            F=feature_matrix(it,blocks);u=np.log(np.maximum(it["base"],1e-15))+F@beta
            u-=u.max();p=np.exp(u);p/=p.sum();y=it["y"]
            loss-=math.log(max(float(p[y]),1e-300))
            g+=F.T@p-F[y]
        return loss/max(1,len(items)),g/max(1,len(items))
    z=np.zeros(d,float)
    rr=minimize(lambda b:fg(b),z,jac=True,method="L-BFGS-B",options={"maxiter":250,"ftol":1e-10})
    return rr.x

def score(items,blocks,beta):
    out=[]
    for it in items:
        F=feature_matrix(it,blocks);u=np.log(np.maximum(it["base"],1e-15))
        if len(beta):u=u+F@beta
        u-=u.max();p=np.exp(u);p/=p.sum();y=it["y"]
        gain=math.log2(max(float(p[y]),1e-15))-math.log2(max(float(it["base"][y]),1e-15))
        out.append((gain,it))
    return out

def block_stat(sc,seed):
    vals=np.array([x[0] for x in sc],float)
    if not len(vals):return {"mean":None,"se":None,"z0":None,"null_sd":None,"null_z":None,"n":0,"blocks":0}
    by=collections.defaultdict(list)
    for g,it in sc:by[it["r"]["bif"]].append(float(g))
    keys=sorted(by);mu=float(vals.mean())
    sums=np.array([sum(by[k]) for k in keys]);ns=np.array([len(by[k]) for k in keys])
    if len(keys)>1:
        cen=sums-ns*mu
        se=math.sqrt((len(keys)/(len(keys)-1))*float(np.sum(cen*cen))/(len(vals)**2))
    else:se=None
    rng=np.random.default_rng(seed);null=[]
    for _ in range(N_SIGN):
        sg=rng.choice((-1.,1.),size=len(keys));null.append(float(np.sum(sg*sums)/len(vals)))
    null=np.asarray(null);sd=float(null.std(ddof=1));nm=float(null.mean())
    return {"mean":mu,"se":se,"z0":(mu/se if se else None),"null_mean":nm,"null_sd":sd,
            "null_z":((mu-nm)/sd if sd>0 else None),"n":len(vals),"blocks":len(keys)}

def fold_means(sc):
    o={}
    for f in (0,1):
        z=[g for g,it in sc if it["r"]["fold"]==f];o[str(f)]=float(np.mean(z)) if z else None
    return o

def choose_l2(tr,va,blocks):
    rec=[]
    for l2 in L2GRID:
        b=fit_beta(tr,blocks,l2);s=score(va,blocks,b)
        rec.append((-(np.mean([x[0] for x in s]) if s else -1e9),l2,b))
    rec.sort(key=lambda z:(z[0],z[1]))
    return rec[0][1],rec[0][2], [{"l2":x[1],"val_gain":-x[0]} for x in rec]

ARMS={
"M1_PHYSICAL":("physical",),
"M2_RENDERER":("renderer",),
"M3_SELECT_ANALOGUE":("select",),
"M4_PAST_CONTEXT":("past",),
"M5_BIDIRECTIONAL_DIAGNOSTIC":("past","future"),
"M6_FULL_PROSPECTIVE":("physical","renderer","select","past")
}

def compare_scores(a,b,seed):
    # a - b on identical events
    d=[(ga-gb,ita) for (ga,ita),(gb,itb) in zip(a,b)]
    return block_stat(d,seed)

def run_tid(tid):
    rows=enrich(build_rows(tid))
    # fit feature tables on discovery only for model selection
    D=make_dataset(rows,tid,(2,3))
    tr=[x for x in D if x["r"]["fold"] in (2,3)]
    va=[x for x in D if x["r"]["fold"]==4]
    # after hyperparameters chosen, rebuild all derived propensity tables on 2/3/4 for final
    choices={}
    for ai,(name,blocks) in enumerate(ARMS.items()):
        l2,_,grid=choose_l2(tr,va,blocks);choices[name]={"l2":l2,"grid":grid,"blocks":blocks}
    F=make_dataset(rows,tid,(2,3,4))
    fit=[x for x in F if x["r"]["fold"] in (2,3,4)]
    te=[x for x in F if x["r"]["fold"] in (0,1)]
    res={}
    scored={}
    for ai,(name,blocks) in enumerate(ARMS.items()):
        beta=fit_beta(fit,blocks,choices[name]["l2"]);sc=score(te,blocks,beta);scored[name]=sc
        res[name]={"selected_l2":choices[name]["l2"],"beta":beta.tolist(),
                   "stat":block_stat(sc,SEED+1000*ai+sum(map(ord,tid))),"folds":fold_means(sc),
                   "validation_grid":choices[name]["grid"]}
    full=scored["M6_FULL_PROSPECTIVE"]
    ab={}
    for j,b in enumerate(BLOCKS):
        blocks=tuple(x for x in BLOCKS if x!=b)
        key="M6_MINUS_"+b.upper()
        # tune ablation separately on validation, respecting freeze
        l2,_,grid=choose_l2(tr,va,blocks);beta=fit_beta(fit,blocks,l2);ss=score(te,blocks,beta)
        ab[key]={"selected_l2":l2,"stat":block_stat(ss,SEED+7000+j+sum(map(ord,tid))),
                 "unique_full_minus_ablation":compare_scores(full,ss,SEED+8000+j+sum(map(ord,tid)))}
    # future-context unique diagnostic relative to past only and full prospective
    bidi=scored["M5_BIDIRECTIONAL_DIAGNOSTIC"];past=scored["M4_PAST_CONTEXT"]
    diag={"bidi_minus_past":compare_scores(bidi,past,SEED+9001+sum(map(ord,tid))),
          "bidi_minus_full_prospective":compare_scores(bidi,full,SEED+9002+sum(map(ord,tid)))}
    return {"tid":tid,"components":[list(x) for x in COMP[tid]],
            "n":{"train":len(fit),"test":len(te),"fold0":sum(x["r"]["fold"]==0 for x in te),"fold1":sum(x["r"]["fold"]==1 for x in te)},
            "arms":res,"ablations":ab,"diagnostics":diag}

OUT={}
for tid in TIDS:
    r=run_tid(tid);OUT[tid]=r
    print("BR1_"+tid+"="+json.dumps(r,separators=(",",":")),flush=True)

def adjudicate(out):
    rows=[]
    for tid,r in out.items():
        z={k:v["stat"]["null_z"] for k,v in r["arms"].items()}
        g={k:v["stat"]["mean"] for k,v in r["arms"].items()}
        rows.append({"tid":tid,"gain":g,"z":z,"future_unique":r["diagnostics"]})
    return {"status":"BR1_STAGE1_COMPLETE","summary":rows,
            "note":"M3 is a SELECT analogue: frozen MRC functional form/coefficients with transcription-specific discovery-count base refit; it is not a literal replay of the ZLZI count table."}

FINAL=adjudicate(OUT)
print("BR1_FINAL="+json.dumps(FINAL,separators=(",",":")),flush=True)
