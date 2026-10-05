#!/usr/bin/env python3
import urllib.request, json, collections, itertools, pickle, hashlib
import numpy as np

SRC="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/ca2d01d98b091a945d9c2cf3530ae76d64dcd5d1/research/context_first_CF1_CF3_real_20261005.py"
src=urllib.request.urlopen(SRC,timeout=120).read().decode()
ns={"__name__":"cfbase"}
exec(compile(src.split("\nOUT={}\n")[0],SRC,"exec"),ns)

SEED=20261005
EDGES={
"ZLZI":[("aiiin","aiin"),("chcthy","cheey"),("cheey","kaiin"),("cheky","ykeey"),("cheody","opchedy"),("otain","qokar"),("otam","shy"),("qokaiin","qokar"),("qokaiin","qol")],
"ZLZB":[("aiiin","aiin"),("chcthy","cheey"),("cheey","kaiin"),("cheky","ykeey"),("cheody","opchedy"),("otain","qokar"),("otam","shy"),("qokaiin","qokar"),("qokaiin","qol")],
"TTLI":[("aiiin","aiin"),("aiiin","al"),("aiiin","ar"),("checkhy","cheedy"),("cheedy","keedy"),("cheor","dor"),("dair","dor"),("kar","odaiin"),("qokain","qokeedy")]
}

def buckets(rows,types):
    cnt,dom=ns["secprop"](rows,(2,3,4))
    return {t:(ns["fbin"](cnt[t]),dom.get(t,"UNK")) for t in types}

def mapped_edges(template, rows, types, rng):
    b=buckets(rows,types)
    pools=collections.defaultdict(list)
    for t in types: pools[b[t]].append(t)
    deg=collections.Counter(x for e in template for x in e)
    verts=sorted(deg,key=lambda v:(len(pools[b[v]]),-deg[v],v))
    for _ in range(200):
        mp={}; used=set()
        for v in verts:
            cand=[x for x in pools[b[v]] if x not in used]
            if not cand: break
            x=str(rng.choice(cand)); mp[v]=x; used.add(x)
        if len(mp)==len(verts):
            out=[tuple(sorted((mp[a],mp[b_]))) for a,b_ in template]
            if len(set(out))==len(out): return out
    return None

def eval_net(rows,V,types,edges):
    cv=ns["final_cross"](rows,V,edges,types)
    pg=ns["pool_gain_final"](rows,V,edges,types)
    return float(np.mean(cv)), pg

def null(template,rows,V,types,tid,tag,n=1000):
    rng=np.random.default_rng(SEED+sum(map(ord,tid+tag)))
    cx=[]; pg=[]
    for _ in range(n):
        e=mapped_edges(template,rows,types,rng)
        if e is None: continue
        c,p=eval_net(rows,V,types,e)
        cx.append(c); pg.append(p["mean"])
    oc,op=eval_net(rows,V,types,template)
    def stat(obs,a):
        a=np.asarray(a,float); sd=float(a.std(ddof=1))
        return {"obs":obs,"null_mean":float(a.mean()),"null_sd":sd,"z":float((obs-a.mean())/sd),"n":len(a)}
    return {"cross":stat(oc,cx),"pool":stat(op["mean"],pg),
            "pool_internal":{"mean":op["mean"],"se":op.get("se"),"z":op.get("z"),"n":op.get("n"),
                             "blocks":op.get("blocks"),"fold0":op.get("fold0"),"fold1":op.get("fold1")}}

def max_matchings(edges):
    best=[]
    for r in range(1,len(edges)+1):
        valid=[]
        for c in itertools.combinations(edges,r):
            flat=[x for e in c for x in e]
            if len(flat)==len(set(flat)): valid.append(list(c))
        if valid: best=valid
        else: break
    return best

OUT={}
for tid in ("ZLZI","ZLZB","TTLI"):
    rows=ns["build_rows"](tid)
    types,_,_,_=ns["eligibility"](rows)
    V=ns["vocab"](rows)
    edges=EDGES[tid]
    full=null(edges,rows,V,types,tid,"full",1000)
    mats=max_matchings(edges)
    mb=[]
    for i,m in enumerate(mats):
        z=null(m,rows,V,types,tid,"m"+str(i),500)
        mb.append({"edges":m,"cross":z["cross"],"pool":z["pool"],"pool_internal":z["pool_internal"]})
    deg=dict(collections.Counter(x for e in edges for x in e))
    out={"tid":tid,"edges":edges,"degree":deg,"topology_matched":full,
         "max_matching_size":len(mats[0]) if mats else 0,"n_max_matchings":len(mats),
         "matching_bounds":{
           "cross_z_min":min(x["cross"]["z"] for x in mb) if mb else None,
           "cross_z_max":max(x["cross"]["z"] for x in mb) if mb else None,
           "pool_z_min":min(x["pool"]["z"] for x in mb) if mb else None,
           "pool_z_max":max(x["pool"]["z"] for x in mb) if mb else None},
         "matchings":mb}
    OUT[tid]=out
    p="/tmp/cf4_"+tid+".pkl"
    with open(p,"wb") as f: pickle.dump(out,f,pickle.HIGHEST_PROTOCOL)
    print("CF4_"+tid+"="+json.dumps(out,separators=(",",":")),flush=True)
    print("PICKLE_SHA_"+tid+"="+hashlib.sha256(open(p,"rb").read()).hexdigest(),flush=True)
print("CF4_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
