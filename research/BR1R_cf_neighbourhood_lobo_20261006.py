#!/usr/bin/env python3
"""BR1R — physical leave-one-bifolium-out robustness for BR1 Stage 1."""
import urllib.request, json, collections, math, numpy as np

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1c68aa15881f4c2e2a86b59e4569fe80b0f3c0c5/research/BR1_cf_neighbourhood_realisation_20261006.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode()
pre=src.split("\nOUT={}\n")[0]
ns={"__name__":"br1base"};exec(compile(pre,BASE,"exec"),ns)

TIDS=("ZLZI","TTLI")
ARMS={
"M1_PHYSICAL":("physical",),
"M2_RENDERER":("renderer",),
"M3_SELECT_ANALOGUE":("select",),
"M6_FULL_PROSPECTIVE":("physical","renderer","select","past")
}
L2GRID=ns["L2GRID"];SEED=20261006
COMP=ns["COMP"];NODECOMP=ns["NODECOMP"];build_rows=ns["build_rows"];enrich=ns["enrich"]
count_tables=ns["count_tables"];smooth_prob=ns["smooth_prob"];lr_feature=ns["lr_feature"]
k12_start=ns["k12_start"];k12_final=ns["k12_final"];safe_form=ns["safe_form"];lev=ns["lev"]
select_prob_for_event=ns["select_prob_for_event"];fit_beta=ns["fit_beta"];score=ns["score"]

def make_custom(rows,tid,fit_bifs):
    train=[r for r in rows if r["bif"] in fit_bifs]
    prior,tabs,global_k,base_k=count_tables(train,tid)
    mp=NODECOMP[tid];cs=COMP[tid]
    supported={ci:tuple(x for x in c if prior[ci][x]>0) for ci,c in enumerate(cs)}
    page=collections.defaultdict(lambda:np.zeros(12,float));linehist=collections.defaultdict(list)
    out=[]
    for r in rows:
        ci=mp.get(r["token"])
        if r["pos"]>0 and ci is not None and len(supported.get(ci,()))>=2:
            cand=supported[ci]
            basep=np.array([smooth_prob(prior[ci],cand,c) for c in cand],float);basep/=basep.sum()
            rc=np.zeros(12,float)
            for z in linehist[(r["folio"],r["line"])][-6:]:
                if z>=0:rc[z]+=1
            sp=select_prob_for_event(r,cand,global_k,base_k,page[r["folio"]],rc)
            X={b:[] for b in ("physical","renderer","select","past","future")}
            for c in cand:
                X["physical"].append([
                    lr_feature(tabs["hand"],ci,r["hand"],c,cand,prior),
                    lr_feature(tabs["section"],ci,r["section"],c,cand,prior)])
                d=lev(r["p1"],c,4);sf=safe_form(c);first8=(sf[1][0] if sf else -1);prevf=k12_final(r["p1"])
                X["renderer"].append([
                    float(d==0),float(d==1),float(d==2),float(d==3),float(d>=4),
                    float(c in r["prev6"]),
                    lr_feature(tabs["lp"],ci,r["lp"],c,cand,prior),
                    lr_feature(tabs["compat"],ci,prevf,c,cand,prior),
                    float(first8==(safe_form(r["p1"])[1][-1] if r["p1"] and safe_form(r["p1"]) else -99))])
                ks=k12_start(c);X["select"].append([math.log(max(float(sp[ks]) if ks>=0 else 1e-12,1e-12))])
                X["past"].append([
                    lr_feature(tabs["p1s"],ci,k12_start(r["p1"]),c,cand,prior),
                    lr_feature(tabs["p1f"],ci,k12_final(r["p1"]),c,cand,prior),
                    lr_feature(tabs["p2s"],ci,k12_start(r["p2"]),c,cand,prior),
                    lr_feature(tabs["p2f"],ci,k12_final(r["p2"]),c,cand,prior)])
                X["future"].append([])
            out.append({"r":r,"ci":ci,"cand":cand,"y":cand.index(r["token"]),"base":basep,
                        "X":{k:np.asarray(v,float) for k,v in X.items()}})
        if r["kstart"]>=0:
            page[r["folio"]][r["kstart"]]+=1;linehist[(r["folio"],r["line"])].append(r["kstart"])
    return out

# fit_beta/score resolve feature_matrix from their original global namespace; replace it
def feature_matrix(item,blocks):
    arr=[item["X"][b] for b in blocks]
    return np.hstack(arr) if arr else np.zeros((len(item["cand"]),0),float)
ns["feature_matrix"]=feature_matrix
fit_beta.__globals__["feature_matrix"]=feature_matrix
score.__globals__["feature_matrix"]=feature_matrix

def choose(tr,va,blocks):
    best=None
    for l2 in L2GRID:
        b=fit_beta(tr,blocks,l2);sc=score(va,blocks,b)
        g=float(np.mean([x[0] for x in sc])) if sc else -1e9
        z=(-g,l2,b)
        if best is None or z[0:2]<best[0:2]:best=z
    return best[1]

def agg(rows):
    vals=np.array([x["gain"] for x in rows],float);by=collections.defaultdict(list)
    for x in rows:by[x["bif"]].append(x["gain"])
    bm=np.array([np.mean(v) for v in by.values()],float);mu=float(vals.mean())
    se=float(bm.std(ddof=1)/math.sqrt(len(bm))) if len(bm)>1 else None
    rng=np.random.default_rng(SEED+len(rows));null=[]
    for _ in range(10000):
        null.append(float(np.mean(bm*rng.choice((-1.,1.),len(bm)))))
    null=np.array(null);sd=float(null.std(ddof=1));nm=float(null.mean())
    fm={str(f):float(np.mean([x["gain"] for x in rows if x["fold"]==f])) for f in range(5) if any(x["fold"]==f for x in rows)}
    return {"mean":mu,"bif_mean":float(bm.mean()),"se_bif_mean":se,
            "z_bif":float(bm.mean()/se) if se else None,
            "null_mean":nm,"null_sd":sd,"null_z":float((bm.mean()-nm)/sd) if sd else None,
            "n":len(vals),"bifs":len(bm),"outer_fold_means":fm}

OUT={}
for tid in TIDS:
    rows=enrich(build_rows(tid));bifs=sorted(set(r["bif"] for r in rows))
    byarm={a:[] for a in ARMS}
    chosen=collections.defaultdict(list)
    for oi,bif in enumerate(bifs):
        outer_fold=next(r["fold"] for r in rows if r["bif"]==bif)
        vfold=4 if outer_fold!=4 else 3
        vbifs={r["bif"] for r in rows if r["fold"]==vfold and r["bif"]!=bif}
        inner_train=set(bifs)-{bif}-vbifs
        D=make_custom(rows,tid,inner_train)
        tr=[x for x in D if x["r"]["bif"] in inner_train]
        va=[x for x in D if x["r"]["bif"] in vbifs]
        final_train=set(bifs)-{bif}
        F=make_custom(rows,tid,final_train)
        fit=[x for x in F if x["r"]["bif"] in final_train];te=[x for x in F if x["r"]["bif"]==bif]
        for name,blocks in ARMS.items():
            l2=choose(tr,va,blocks);chosen[name].append(l2);beta=fit_beta(fit,blocks,l2);sc=score(te,blocks,beta)
            for g,it in sc:byarm[name].append({"gain":float(g),"bif":bif,"fold":it["r"]["fold"]})
        if (oi+1)%10==0:print("BR1R_PROGRESS",tid,oi+1,len(bifs),flush=True)
    rr={a:{"stat":agg(z),"l2_counts":dict(collections.Counter(chosen[a]))} for a,z in byarm.items()}
    OUT[tid]=rr;print("BR1R_"+tid+"="+json.dumps(rr,separators=(",",":")),flush=True)
print("BR1R_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
