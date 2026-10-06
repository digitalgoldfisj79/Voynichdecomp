#!/usr/bin/env python3
"""
CF-ID1 — exact identity residual test inside frozen CF4 neighbourhoods.
2026-10-06.

Primary question:
Once CF-neighbourhood identity is known, does the exact surface node retain
information at distances never used to discover CF?

Two independent endpoints:
A) DISTANT CONTEXT: heldout gain from exact-node-specific context distributions
   at lags +/-3..10. CF discovery used only +/-1,2.
B) RECURRENCE IDENTITY: when the same CF component recurs later in a paragraph,
   does the same exact surface node recur more often than its component/section
   frequency predicts?

Both are evaluated on folds 0/1 after training/tuning on folds 2/3/4 and
compared to topology + frequency-bin + dominant-section matched Voynich
networks. No current-token morphology, FORM, ED, DINO, or NeuroDecipher
representation enters either endpoint.
"""
import collections, json, math, re, urllib.request, os, multiprocessing as mp
import numpy as np

SEED=20261006
TIDS=("ZLZI","ZLZB","TTLI")
ALPHAS=(3.,10.,30.,100.,300.)
TOPV=512
NNULL=500
NSIGN=5000
LAGS=tuple(list(range(-10,-2))+list(range(3,11)))

CF4_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/4d4d9ba742b674065f76b99cc894de26f3e09fea/research/context_first_CF4_topology_audit_20261005.py"
src=urllib.request.urlopen(CF4_URL,timeout=120).read().decode()
cf={"__name__":"cf4base"};exec(compile(src.split("\nOUT={}\n")[0],CF4_URL,"exec"),cf)
ns=cf["ns"]; OBJ=ns["c"]["OBJ"]; BIF=ns["c"]["BIF"]; folds=ns["c"]["folds"]
section=ns["c"]["section"]; fnum=ns["c"]["fnum"]
EDGES=cf["EDGES"]

PARA_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/de1d843a04f27862036e3551be291083ed96a1af/research/data/paragraph_ranges_20261004.tsv"
PR=collections.defaultdict(list)
for ln in urllib.request.urlopen(PARA_URL,timeout=120).read().decode().splitlines():
    if not ln.strip():continue
    f,p,a,b=ln.split("\t");PR[f].append((int(a),int(b),int(p)))

def line_num(s):
    m=re.match(r"(\d+)",str(s))
    return int(m.group(1)) if m else 0
def para_id(f,line):
    n=line_num(line)
    for a,b,p in PR.get(f,()):
        if a<=n<=b:return int(p)
    return 100000+n

def linepos(pos,n):
    if pos<=1:return 0
    if pos>=n-2:return 2
    return 1

def components(edges):
    adj=collections.defaultdict(set)
    for a,b in edges:adj[a].add(b);adj[b].add(a)
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

def build_rows(tid):
    rows=[]; paras=collections.defaultdict(list)
    for fol,ld in OBJ["pages"].items():
        n=fnum(fol)
        if n not in BIF or BIF[n] not in folds:continue
        bif=BIF[n];sec=section(fol)
        for lo,(ls,rec) in enumerate(ld.items()):
            if str(rec.get("u",""))!="+P0":continue
            txt=rec.get("t",{}).get(tid,"")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower())]
            if not toks:continue
            pk=(fol,para_id(fol,ls))
            for pos,t in enumerate(toks):
                full={"token":t,"folio":fol,"line":str(ls),"line_ord":lo,"pos":pos,
                      "line_len":len(toks),"bif":bif,"fold":int(folds[bif]),
                      "section":sec,"lp":linepos(pos,len(toks)),"para":pk}
                paras[pk].append(full)
                if pos==0:continue
                r=dict(full)
                ctx=[]
                for lag in LAGS:
                    j=pos+lag
                    if 0<=j<len(toks):ctx.append((lag,toks[j]))
                r["dctx"]=ctx
                rows.append(r)
    # paragraph order index for all tokens; target rows inherit rank.
    rank={}
    for pk,z in paras.items():
        z.sort(key=lambda r:(r["line_ord"],r["pos"]))
        for i,r in enumerate(z):rank[(r["folio"],r["line"],r["pos"])]=i
    for r in rows:r["prank"]=rank[(r["folio"],r["line"],r["pos"])]
    return rows,paras

def eligible_types(rows):
    return ns["eligibility"](rows)[0]

def top_vocab(rows):
    c=collections.Counter()
    for r in rows:
        if r["fold"] not in (2,3):continue
        for lag,t in r["dctx"]:c[t]+=1
    return {t:i for i,(t,_) in enumerate(c.most_common(TOPV))}

def ctxcat(t,V):return t if t in V else "__OTHER__"

def fit_context(rows, comps, V, folds_fit):
    node=collections.defaultdict(collections.Counter)
    comp=collections.defaultdict(collections.Counter)
    ntot=collections.Counter();ctot=collections.Counter()
    mp={x:i for i,c in enumerate(comps) for x in c}
    for r in rows:
        if r["fold"] not in folds_fit or r["token"] not in mp:continue
        ci=mp[r["token"]];y=r["token"]
        for lag,t in r["dctx"]:
            k=(lag,ctxcat(t,V))
            node[y][k]+=1;ntot[y]+=1;comp[ci][k]+=1;ctot[ci]+=1
    return mp,node,comp,ntot,ctot

def context_scores(rows,comps,V,folds_fit,folds_eval,alpha):
    mp,node,comp,ntot,ctot=fit_context(rows,comps,V,folds_fit)
    gains=[]
    for r in rows:
        if r["fold"] not in folds_eval or r["token"] not in mp:continue
        ci=mp[r["token"]];y=r["token"];C=comp[ci];denC=ctot[ci]
        if denC<=0 or ntot[y]<=0:continue
        # component distribution is lag-specific through key counts; add tiny uniform floor.
        # Node model shrinks to the component model with alpha pseudo-observations.
        bylag=collections.Counter(k[0] for k in C)
        nodelag=collections.Counter(k[0] for k in node[y])
        for lag,t in r["dctx"]:
            k=(lag,ctxcat(t,V));K=TOPV+1
            denlag=bylag[lag]
            p0=(C[k]+.25)/(denlag+.25*K) if denlag>0 else 1.0/K
            q=(node[y][k]+alpha*p0)/(nodelag[lag]+alpha)
            gains.append((math.log2(max(q,1e-15))-math.log2(max(p0,1e-15)),r))
    return gains

def block_stat(vals,seed):
    if not vals:return {"mean":None,"bif_mean":None,"se":None,"z0":None,"null_mean":None,"null_sd":None,"null_z":None,"n":0,"bifs":0,"fold0":None,"fold1":None}
    ev=np.array([g for g,_ in vals],float);by=collections.defaultdict(list)
    for g,r in vals:by[r["bif"]].append(float(g))
    keys=sorted(by);bm=np.array([np.mean(by[k]) for k in keys]);mu=float(ev.mean())
    se=float(bm.std(ddof=1)/math.sqrt(len(bm))) if len(bm)>1 else None
    rng=np.random.default_rng(seed);null=[]
    for _ in range(NSIGN):
        null.append(float(np.mean(bm*rng.choice((-1.,1.),len(bm)))))
    a=np.asarray(null);sd=float(a.std(ddof=1));nm=float(a.mean())
    return {"mean":mu,"bif_mean":float(bm.mean()),"se":se,
            "z0":float(bm.mean()/se) if se and se>0 else None,
            "null_mean":nm,"null_sd":sd,"null_z":float((bm.mean()-nm)/sd) if sd>0 else None,
            "n":len(ev),"bifs":len(keys),
            "fold0":float(np.mean([g for g,r in vals if r["fold"]==0])) if any(r["fold"]==0 for _,r in vals) else None,
            "fold1":float(np.mean([g for g,r in vals if r["fold"]==1])) if any(r["fold"]==1 for _,r in vals) else None}

def tune_alpha(rows,comps,V):
    rec=[]
    for a in ALPHAS:
        z=context_scores(rows,comps,V,(2,3),(4,),a)
        g=float(np.mean([x[0] for x in z])) if z else -1e9
        rec.append((g,a))
    rec.sort(reverse=True)
    return rec[0][1],[{"alpha":a,"val_gain":g} for g,a in rec]

def recurrence_scores(rows,paras,comps,folds_fit=(2,3,4),folds_eval=(0,1)):
    mp={x:i for i,c in enumerate(comps) for x in c}
    # nuisance baseline: component x section exact-node distribution.
    cnt=collections.defaultdict(collections.Counter)
    for r in rows:
        if r["fold"] in folds_fit and r["token"] in mp:
            cnt[(mp[r["token"]],r["section"])][r["token"]]+=1
    # target occurrences only, ordered within paragraph; line starts are excluded.
    byp=collections.defaultdict(list)
    for r in rows:
        if r["token"] in mp:byp[r["para"]].append(r)
    vals=[]
    for pk,z in byp.items():
        z.sort(key=lambda r:r["prank"])
        last={}
        for r in z:
            ci=mp[r["token"]]
            if ci in last and r["fold"] in folds_eval:
                p=last[ci]
                # require truly non-local recurrence: >2 token positions within paragraph.
                if r["prank"]-p["prank"]>2:
                    C=cnt[(ci,r["section"])]
                    cands=comps[ci];den=sum(C.get(x,0) for x in cands);K=len(cands)
                    p_same=(C.get(p["token"],0)+.5)/(den+.5*K) if den>0 else 1.0/K
                    vals.append(((1.0 if r["token"]==p["token"] else 0.0)-p_same,r))
            last[ci]=r
    return vals

def bif_mean_only(vals):
    if not vals:return None
    by=collections.defaultdict(list)
    for g,r in vals:by[r["bif"]].append(float(g))
    return float(np.mean([np.mean(v) for v in by.values()]))

def topology_maps(edges,rows,types,tid,n=NNULL):
    rng=np.random.default_rng(SEED+sum(map(ord,tid))+771)
    out=[]
    for _ in range(n):
        e=cf["mapped_edges"](edges,rows,types,rng)
        if e is not None:out.append(e)
    return out

_G={}

def _null_worker(e):
    cp=components(e)
    cv=context_scores(_G["rows"],cp,_G["V"],(2,3,4),(0,1),_G["alpha"])
    rv=recurrence_scores(_G["rows"],_G["paras"],cp)
    return bif_mean_only(cv),bif_mean_only(rv)

def topo_stat(obs,arr):
    a=np.asarray(arr,float);sd=float(a.std(ddof=1));nm=float(a.mean())
    return {"obs":float(obs),"null_mean":nm,"null_sd":sd,"z":float((obs-nm)/sd) if sd>0 else None,"n":len(a)}

def run(tid):
    rows,paras=build_rows(tid);types=eligible_types(rows);V=top_vocab(rows)
    real_edges=EDGES[tid];real_comps=components(real_edges)
    alpha,agrid=tune_alpha(rows,real_comps,V)
    cvals=context_scores(rows,real_comps,V,(2,3,4),(0,1),alpha)
    rvals=recurrence_scores(rows,paras,real_comps)
    C=block_stat(cvals,SEED+11+sum(map(ord,tid)))
    R=block_stat(rvals,SEED+22+sum(map(ord,tid)))
    maps=topology_maps(real_edges,rows,types,tid)
    nc=[];nr=[];failed=0
    _G.clear();_G.update({"rows":rows,"paras":paras,"V":V,"alpha":alpha})
    ctx=mp.get_context("fork")
    with ctx.Pool(processes=min(16,os.cpu_count() or 4)) as pool:
        for a,b in pool.imap_unordered(_null_worker,maps,chunksize=4):
            if a is not None:nc.append(a)
            else:failed+=1
            if b is not None:nr.append(b)
    out={"tid":tid,"components":[list(x) for x in real_comps],"alpha":alpha,"validation":agrid,
         "distant_context":{"stat":C,"topology_matched":topo_stat(C["bif_mean"],nc) if nc else None},
         "recurrence_identity":{"stat":R,"topology_matched":topo_stat(R["bif_mean"],nr) if nr else None},
         "null_maps":len(maps),"failed":failed,
         "interpretation_key":{"positive_distant":"exact nodes retain distinct long-range contextual identity",
             "near_zero_distant":"component captures most long-range context",
             "positive_recurrence":"exact identity persists when component recurs",
             "near_zero_recurrence":"component recurs without stable exact-node identity"}}
    return out

OUT={}
for tid in TIDS:
    print("START_"+tid,flush=True)
    OUT[tid]=run(tid)
    print("CFID1_"+tid+"="+json.dumps(OUT[tid],separators=(",",":")),flush=True)
print("CFID1_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
