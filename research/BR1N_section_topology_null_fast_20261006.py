#!/usr/bin/env python3
"""BR1N-fast — topology/frequency/dominant-section matched null for section-conditioned node choice."""
import urllib.request,json,collections,math,numpy as np

CF1="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/ca2d01d98b091a945d9c2cf3530ae76d64dcd5d1/research/context_first_CF1_CF3_real_20261005.py"
src=urllib.request.urlopen(CF1,timeout=120).read().decode();ns={"__name__":"cf1"};exec(compile(src.split("\nOUT={}\n")[0],CF1,"exec"),ns)

EDGES={
"ZLZI":[("aiiin","aiin"),("chcthy","cheey"),("cheey","kaiin"),("cheky","ykeey"),("cheody","opchedy"),("otain","qokar"),("otam","shy"),("qokaiin","qokar"),("qokaiin","qol")],
"ZLZB":[("aiiin","aiin"),("chcthy","cheey"),("cheey","kaiin"),("cheky","ykeey"),("cheody","opchedy"),("otain","qokar"),("otam","shy"),("qokaiin","qokar"),("qokaiin","qol")],
"TTLI":[("aiiin","aiin"),("aiiin","al"),("aiiin","ar"),("checkhy","cheedy"),("cheedy","keedy"),("cheor","dor"),("dair","dor"),("kar","odaiin"),("qokain","qokeedy")]
}
ALPHAS=(1.,3.,10.,30.,100.)
NNULL=1000
SEED=20261006

def components(edges):
    a=collections.defaultdict(set)
    for x,y in edges:a[x].add(y);a[y].add(x)
    seen=set();out=[]
    for s in sorted(a):
        if s in seen:continue
        q=[s];seen.add(s);cc=[]
        while q:
            x=q.pop();cc.append(x)
            for y in a[x]:
                if y not in seen:seen.add(y);q.append(y)
        out.append(tuple(sorted(cc)))
    return out

def prep(rows,types):
    # Aggregate once: token -> [(fold,bif,section,count), ...]
    c=collections.Counter((r["token"],r["fold"],r["bif"],r["section"]) for r in rows)
    bytok=collections.defaultdict(list)
    for (t,f,b,s),n in c.items():bytok[t].append((int(f),b,s,int(n)))
    cnt,dom=ns["secprop"](rows,(2,3,4))
    bucket={t:(ns["fbin"](cnt[t]),dom.get(t,"UNK")) for t in types}
    pools=collections.defaultdict(list)
    for t in types:pools[bucket[t]].append(t)
    return bytok,bucket,pools

def mapped_edges(template,bucket,pools,rng):
    deg=collections.Counter(x for e in template for x in e)
    verts=sorted(deg,key=lambda v:(len(pools[bucket[v]]),-deg[v],v))
    for _ in range(300):
        mp={};used=set()
        for v in verts:
            cand=[x for x in pools[bucket[v]] if x not in used]
            if not cand:break
            x=str(rng.choice(cand));mp[v]=x;used.add(x)
        if len(mp)==len(verts):
            out=[tuple(sorted((mp[a],mp[b]))) for a,b in template]
            if len(set(out))==len(out):return out
    return None

def build_counts(comps,bytok,trainfolds):
    glob=[collections.Counter() for _ in comps]
    sec=collections.defaultdict(collections.Counter)
    for ci,comp in enumerate(comps):
        for t in comp:
            for f,b,s,n in bytok.get(t,()):
                if f in trainfolds:
                    glob[ci][t]+=n;sec[(ci,s)][t]+=n
    return glob,sec

def evaluate(comps,bytok,trainfolds,evalfolds,alpha):
    glob,sec=build_counts(comps,bytok,trainfolds)
    tsum=0.0;tn=0;bs=collections.defaultdict(lambda:[0.0,0])
    for ci,comp in enumerate(comps):
        g=glob[ci];ng=sum(g.values());k=len(comp)
        q={t:(g[t]+.5)/(ng+.5*k) for t in comp}
        for t in comp:
            for f,b,s,n in bytok.get(t,()):
                if f not in evalfolds:continue
                ss=sec.get((ci,s),{});nn=sum(ss.values())
                p=(ss.get(t,0)+alpha*q[t])/(nn+alpha)
                d=math.log2(max(p,1e-15))-math.log2(max(q[t],1e-15))
                tsum+=n*d;tn+=n;bs[b][0]+=n*d;bs[b][1]+=n
    if not tn:return None
    return {"sum":tsum,"n":tn,"event_mean":tsum/tn,
            "bifs":{b:(z[0],z[1]) for b,z in bs.items()}}

def crossfit(bytok,edges):
    comps=components(edges);chosen=[];all_sum=0.0;all_n=0;bs=collections.defaultdict(lambda:[0.0,0]);foldm={}
    for outer in range(5):
        val=(outer+1)%5;tr=tuple(f for f in range(5) if f not in (outer,val))
        best=None
        for a in ALPHAS:
            z=evaluate(comps,bytok,tr,(val,),a);g=z["event_mean"] if z else -1e99
            cand=(-g,a)
            if best is None or cand<best:best=cand
        alpha=best[1];chosen.append(alpha)
        ft=tuple(f for f in range(5) if f!=outer)
        z=evaluate(comps,bytok,ft,(outer,),alpha)
        if z:
            all_sum+=z["sum"];all_n+=z["n"];foldm[str(outer)]=z["event_mean"]
            for b,(sm,n) in z["bifs"].items():bs[b][0]+=sm;bs[b][1]+=n
    bm=[sm/n for sm,n in bs.values() if n]
    return {"event_mean":all_sum/all_n,"bif_mean":float(np.mean(bm)),"n":all_n,"bifs":len(bm),
            "alphas":chosen,"fold_means":foldm}

OUT={}
for tid in ("ZLZI","ZLZB","TTLI"):
    rows=ns["build_rows"](tid);types,_,_,_=ns["eligibility"](rows)
    bytok,bucket,pools=prep(rows,types);edges=EDGES[tid]
    obs=crossfit(bytok,edges);rng=np.random.default_rng(SEED+sum(map(ord,tid)))
    bv=[];ev=[];fail=0
    for i in range(NNULL):
        e=mapped_edges(edges,bucket,pools,rng)
        if e is None:fail+=1;continue
        z=crossfit(bytok,e);bv.append(z["bif_mean"]);ev.append(z["event_mean"])
    B=np.asarray(bv,float);E=np.asarray(ev,float)
    out={"tid":tid,"obs":obs,
         "topology_matched":{
           "bif_null_mean":float(B.mean()),"bif_null_sd":float(B.std(ddof=1)),
           "bif_z":float((obs["bif_mean"]-B.mean())/B.std(ddof=1)),
           "event_null_mean":float(E.mean()),"event_null_sd":float(E.std(ddof=1)),
           "event_z":float((obs["event_mean"]-E.mean())/E.std(ddof=1)),
           "n":len(B),"failed_maps":fail}}
    OUT[tid]=out;print("BR1N_"+tid+"="+json.dumps(out,separators=(",",":")),flush=True)
print("BR1N_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
