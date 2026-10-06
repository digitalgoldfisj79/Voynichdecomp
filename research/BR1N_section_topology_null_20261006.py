#!/usr/bin/env python3
"""BR1N — topology/frequency/dominant-section matched null for section-conditioned node choice."""
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

def comps(edges):
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

def buckets(rows,types):
 cnt,dom=ns["secprop"](rows,(2,3,4))
 return {t:(ns["fbin"](cnt[t]),dom.get(t,"UNK")) for t in types}

def mapped_edges(template,rows,types,rng):
 b=buckets(rows,types);pools=collections.defaultdict(list)
 for t in types:pools[b[t]].append(t)
 deg=collections.Counter(x for e in template for x in e)
 verts=sorted(deg,key=lambda v:(len(pools[b[v]]),-deg[v],v))
 for _ in range(300):
  mp={};used=set()
  for v in verts:
   cand=[x for x in pools[b[v]] if x not in used]
   if not cand:break
   x=str(rng.choice(cand));mp[v]=x;used.add(x)
  if len(mp)==len(verts):
   out=[tuple(sorted((mp[a],mp[b_]))) for a,b_ in template]
   if len(set(out))==len(out):return out
 return None

def fit_counts(rows,components,foldset):
 mp={t:i for i,c in enumerate(components) for t in c}
 glob=collections.defaultdict(collections.Counter)
 sec=collections.defaultdict(collections.Counter)
 for r in rows:
  if r["fold"] not in foldset:continue
  ci=mp.get(r["token"])
  if ci is None:continue
  glob[ci][r["token"]]+=1;sec[(ci,r["section"])][r["token"]]+=1
 return mp,glob,sec

def eval_gain(rows,components,trainfolds,evalfolds,alpha,details=False):
 mp,glob,sec=fit_counts(rows,components,trainfolds)
 gains=[];bg=collections.defaultdict(list)
 for r in rows:
  if r["fold"] not in evalfolds:continue
  ci=mp.get(r["token"])
  if ci is None:continue
  cand=components[ci];g=glob[ci];ng=sum(g.values())
  q={t:(g[t]+.5)/(ng+.5*len(cand)) for t in cand}
  sc=sec.get((ci,r["section"]),{});n=sum(sc.values())
  p=(sc.get(r["token"],0)+alpha*q[r["token"]])/(n+alpha)
  d=math.log2(max(p,1e-15))-math.log2(max(q[r["token"]],1e-15))
  gains.append(d);bg[r["bif"]].append(d)
 if not gains:return None
 bm=[float(np.mean(v)) for v in bg.values()]
 return {"event_mean":float(np.mean(gains)),"bif_mean":float(np.mean(bm)),"n":len(gains),"bifs":len(bm)}

def crossfit(rows,edges):
 components=comps(edges);outer=[]
 alphas=[]
 for f in range(5):
  val=(f+1)%5;tr=tuple(x for x in range(5) if x not in (f,val))
  best=None
  for a in ALPHAS:
   z=eval_gain(rows,components,tr,(val,),a)
   g=z["event_mean"] if z else -1e9
   if best is None or (-g,a)<best[0]:best=((-g,a),a)
  alpha=best[1];alphas.append(alpha)
  ft=tuple(x for x in range(5) if x!=f);z=eval_gain(rows,components,ft,(f,),alpha)
  if z:outer.append((f,z))
 # Aggregate over all outer events by re-running foldwise chosen models and collecting event/bif gains.
 allg=[];byb=collections.defaultdict(list);foldmeans={}
 for f,alpha in enumerate(alphas):
  components2=components;mp,glob,sec=fit_counts(rows,components2,tuple(x for x in range(5) if x!=f))
  fg=[]
  for r in rows:
   if r["fold"]!=f:continue
   ci=mp.get(r["token"])
   if ci is None:continue
   cand=components2[ci];g=glob[ci];ng=sum(g.values());q={t:(g[t]+.5)/(ng+.5*len(cand)) for t in cand}
   sc=sec.get((ci,r["section"]),{});n=sum(sc.values());p=(sc.get(r["token"],0)+alpha*q[r["token"]])/(n+alpha)
   d=math.log2(max(p,1e-15))-math.log2(max(q[r["token"]],1e-15))
   allg.append(d);fg.append(d);byb[r["bif"]].append(d)
  foldmeans[str(f)]=float(np.mean(fg)) if fg else None
 return {"event_mean":float(np.mean(allg)),"bif_mean":float(np.mean([np.mean(v) for v in byb.values()])),
         "n":len(allg),"bifs":len(byb),"alphas":alphas,"fold_means":foldmeans}

OUT={}
for tid in ("ZLZI","ZLZB","TTLI"):
 rows=ns["build_rows"](tid);types,_,_,_=ns["eligibility"](rows);edges=EDGES[tid]
 obs=crossfit(rows,edges)
 rng=np.random.default_rng(SEED+sum(map(ord,tid)))
 vals=[];ev=[];nfail=0
 for i in range(NNULL):
  e=mapped_edges(edges,rows,types,rng)
  if e is None:nfail+=1;continue
  z=crossfit(rows,e);vals.append(z["bif_mean"]);ev.append(z["event_mean"])
 A=np.array(vals);E=np.array(ev)
 out={"tid":tid,"obs":obs,
      "topology_matched":{"bif_null_mean":float(A.mean()),"bif_null_sd":float(A.std(ddof=1)),
                          "bif_z":float((obs["bif_mean"]-A.mean())/A.std(ddof=1)),
                          "event_null_mean":float(E.mean()),"event_null_sd":float(E.std(ddof=1)),
                          "event_z":float((obs["event_mean"]-E.mean())/E.std(ddof=1)),
                          "n":len(A),"failed_maps":nfail}}
 OUT[tid]=out;print("BR1N_"+tid+"="+json.dumps(out,separators=(",",":")),flush=True)
print("BR1N_FINAL="+json.dumps(OUT,separators=(",",":")),flush=True)
