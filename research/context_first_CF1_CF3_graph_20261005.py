#!/usr/bin/env python3
"""
Context-first lexical equivalence — calibrated graph assay on real Voynich.
Method frozen after CF0 only, before any real-Voynich outcome inspection.

Discovery: folds 2/3, current-token morphology completely hidden.
Candidate graph: edges above q99.5 of frequency+dominant-section matched random pair
cross-fold external-context similarity.
Validation: fold4 aggregate graph similarity vs degree-preserving matched relabel null.
Final: folds0/1 reciprocal cross-fold graph similarity vs same null.
"""
import collections, json, math, re, urllib.request, numpy as np

SEED=20261005; TOPCTX=256; MIN_DISC=20; MIN_VAL=3; MIN_TEST=3; BETA=5.; NNULL=2000
CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
c={"__name__":"core"};exec(compile(urllib.request.urlopen(CORE_URL,timeout=120).read().decode(),CORE_URL,"exec"),c)
OBJ=c["OBJ"];BIF=c["BIF"];folds=c["folds"];section=c["section"];fnum=c["fnum"]

def lp(pos,n):
    if pos<=1:return 0
    if pos>=n-2:return 2
    return 1

def build_rows(tid):
    rows=[]
    for fol,ld in OBJ["pages"].items():
        n=fnum(fol)
        if n not in BIF:continue
        bif=BIF[n]
        if bif not in folds:continue
        for lo,(ls,rec) in enumerate(ld.items()):
            if str(rec.get("u",""))!="+P0":continue
            txt=rec.get("t",{}).get(tid,"")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower())]
            if len(toks)<2:continue
            for pos,t in enumerate(toks):
                if pos==0:continue
                r=dict(token=t,folio=fol,line=str(ls),pos=pos,line_len=len(toks),bif=bif,
                       fold=int(folds[bif]),section=section(fol),lp=lp(pos,len(toks)))
                for lag in (-2,-1,1,2):
                    j=pos+lag;r[f"n{lag:+d}"]=toks[j] if 0<=j<len(toks) else None
                rows.append(r)
    return rows

def eligible(rows):
    dc=collections.Counter(r["token"] for r in rows if r["fold"] in (2,3))
    vc=collections.Counter(r["token"] for r in rows if r["fold"]==4)
    tc=collections.Counter(r["token"] for r in rows if r["fold"] in (0,1))
    ts=sorted(t for t,n in dc.items() if n>=MIN_DISC and vc[t]>=MIN_VAL and tc[t]>=MIN_TEST)
    return ts,dc,vc,tc

def ctx_vocab(rows):
    z=collections.Counter()
    for r in rows:
        if r["fold"] not in (2,3):continue
        for lag in (-2,-1,1,2):
            x=r[f"n{lag:+d}"]
            if x:z[x]+=1
    top=[x for x,_ in z.most_common(TOPCTX)]
    return {x:i for i,x in enumerate(top)}

def cat(x,vmap):
    if x is None:return None
    return vmap.get(x,len(vmap))

def nkey(r,lag):return (lag,r["section"],r["lp"])

def baseline(rows,vmap):
    K=len(vmap)+1
    glob={l:np.ones(K)*.25 for l in (-2,-1,1,2)}
    by=collections.defaultdict(lambda:np.zeros(K))
    for r in rows:
        if r["fold"] not in (2,3):continue
        for l in (-2,-1,1,2):
            j=cat(r[f"n{l:+d}"],vmap)
            if j is None:continue
            glob[l][j]+=1;by[nkey(r,l)][j]+=1
    for l in glob:glob[l]/=glob[l].sum()
    out={}
    for k,a in by.items():
        v=a+20*glob[k[0]];out[k]=v/v.sum()
    return out,glob

def prof(rows,vmap,types,fold,base,glob):
    K=len(vmap)+1;lags=(-2,-1,1,2);li={z:i for i,z in enumerate(lags)}
    o=collections.defaultdict(lambda:np.zeros((4,K)));e=collections.defaultdict(lambda:np.zeros((4,K)))
    for r in rows:
        if r["fold"]!=fold or r["token"] not in types:continue
        for l in lags:
            j=cat(r[f"n{l:+d}"],vmap)
            if j is None:continue
            q=base.get(nkey(r,l),glob[l]);o[r["token"]][li[l],j]+=1;e[r["token"]][li[l]]+=q
    z={}
    for t in types:
        x=np.log(np.maximum((o[t]+BETA)/(e[t]+BETA),1e-9)).ravel()
        x-=x.mean();nn=np.linalg.norm(x);z[t]=x/nn if nn else x
    return z

def fbin(n):
    if n<30:return 0
    if n<60:return 1
    if n<120:return 2
    if n<240:return 3
    return 4

def domsec(rows):
    by=collections.defaultdict(collections.Counter)
    for r in rows:
        if r["fold"] in (2,3):by[r["token"]][r["section"]]+=1
    return {t:c.most_common(1)[0][0] for t,c in by.items() if c}

def sim_disc(a,b,E2,E3):
    return float(.25*(np.dot(E2[a],E2[b])+np.dot(E3[a],E3[b])+np.dot(E2[a],E3[b])+np.dot(E2[b],E3[a])))
def sim_same(a,b,E):return float(np.dot(E[a],E[b]))
def sim_cross(a,b,E0,E1):return float(.5*(np.dot(E0[a],E1[b])+np.dot(E0[b],E1[a])))

def matched_random_pairs(types,dc,ds,rng,n):
    buckets=collections.defaultdict(list)
    for t in types:buckets[(fbin(dc[t]),ds.get(t,"UNK"))].append(t)
    keys=[k for k,v in buckets.items() if len(v)>=2]
    out=[];seen=set()
    for _ in range(n*20):
        if len(out)>=n:break
        k=keys[int(rng.integers(len(keys)))];a,b=rng.choice(buckets[k],2,replace=False)
        p=tuple(sorted((str(a),str(b))))
        if p not in seen:seen.add(p);out.append(p)
    return out

def components(edges):
    adj=collections.defaultdict(set)
    for a,b in edges:adj[a].add(b);adj[b].add(a)
    seen=set();out=[]
    for x in adj:
        if x in seen:continue
        st=[x];seen.add(x);cc=[]
        while st:
            u=st.pop();cc.append(u)
            for v in adj[u]:
                if v not in seen:seen.add(v);st.append(v)
        out.append(sorted(cc))
    return sorted(out,key=lambda z:(-len(z),z))

def relabel_graph_null(edges,types,dc,ds,Ea,Eb,cross,rng,nrep=NNULL):
    nodes=sorted(set(x for e in edges for x in e))
    # node strata; preserve topology and approximate endpoint frequency/section
    buckets=collections.defaultdict(list)
    for t in types:buckets[(fbin(dc[t]),ds.get(t,"UNK"))].append(t)
    strata={u:(fbin(dc[u]),ds.get(u,"UNK")) for u in nodes}
    vals=[]
    for _ in range(nrep):
        mp={};used=set();ok=True
        # most constrained first
        for u in sorted(nodes,key=lambda x:len(buckets[strata[x]])):
            cand=[x for x in buckets[strata[u]] if x not in used]
            if not cand:ok=False;break
            v=str(rng.choice(cand));mp[u]=v;used.add(v)
        if not ok:continue
        ss=[]
        for a,b in edges:
            aa,bb=mp[a],mp[b]
            ss.append(sim_cross(aa,bb,Ea,Eb) if cross else sim_same(aa,bb,Ea))
        vals.append(float(np.mean(ss)))
    return np.array(vals)

def self_control(types,E0,E1,dc,ds,rng):
    obs=np.array([float(np.dot(E0[t],E1[t])) for t in types])
    rand=[]
    for _ in range(2000):
        ps=matched_random_pairs(types,dc,ds,rng,len(types))
        if not ps:continue
        rand.append(float(np.mean([sim_cross(a,b,E0,E1) for a,b in ps])))
    return {"mean_self":float(obs.mean()),"null_mean":float(np.mean(rand)),"null_sd":float(np.std(rand,ddof=1)),
            "z":float((obs.mean()-np.mean(rand))/np.std(rand,ddof=1))}

def run(tid):
    rows=build_rows(tid);types,dc,vc,tc=eligible(rows);vmap=ctx_vocab(rows);base,glob=baseline(rows,vmap)
    E2=prof(rows,vmap,set(types),2,base,glob);E3=prof(rows,vmap,set(types),3,base,glob)
    E4=prof(rows,vmap,set(types),4,base,glob);E0=prof(rows,vmap,set(types),0,base,glob);E1=prof(rows,vmap,set(types),1,base,glob)
    ds=domsec(rows);rng=np.random.default_rng(SEED+sum(map(ord,tid)))
    # candidate threshold from matched pair null
    rp=matched_random_pairs(types,dc,ds,rng,6000)
    rv=np.array([sim_disc(a,b,E2,E3) for a,b in rp])
    q995=float(np.quantile(rv,.995))
    edges=[]
    for i,a in enumerate(types):
        for b in types[i+1:]:
            s=sim_disc(a,b,E2,E3)
            if s>q995:edges.append((a,b))
    comp=components(edges)
    # validation graph
    vobs=float(np.mean([sim_same(a,b,E4) for a,b in edges])) if edges else None
    vn=relabel_graph_null(edges,types,dc,ds,E4,E4,False,rng) if edges else np.array([])
    vz=float((vobs-vn.mean())/vn.std(ddof=1)) if len(vn)>1 else None
    # final reciprocal crossfold
    tobs=float(np.mean([sim_cross(a,b,E0,E1) for a,b in edges])) if edges else None
    tn=relabel_graph_null(edges,types,dc,ds,E0,E1,True,rng) if edges else np.array([])
    tz=float((tobs-tn.mean())/tn.std(ddof=1)) if len(tn)>1 else None
    # fold directional within-fold similarity diagnostics
    f0=float(np.mean([sim_same(a,b,E0) for a,b in edges])) if edges else None
    f1=float(np.mean([sim_same(a,b,E1) for a,b in edges])) if edges else None
    pc=self_control(types,E0,E1,dc,ds,rng)
    gate=bool(edges and vz is not None and vz>2 and tz is not None and tz>2 and f0 is not None and f0>0 and f1 is not None and f1>0 and pc["z"]>2)
    return {"tid":tid,"population_rows":len(rows),"eligible_types":len(types),"q995":q995,
            "edges":len(edges),"components":comp,"component_sizes":[len(x) for x in comp],
            "edge_list":[[a,b,sim_disc(a,b,E2,E3),sim_same(a,b,E4),sim_cross(a,b,E0,E1)] for a,b in edges],
            "validation":{"observed":vobs,"null_mean":float(vn.mean()) if len(vn) else None,
                          "null_sd":float(vn.std(ddof=1)) if len(vn)>1 else None,"z":vz,"nnull":len(vn)},
            "final":{"observed":tobs,"null_mean":float(tn.mean()) if len(tn) else None,
                     "null_sd":float(tn.std(ddof=1)) if len(tn)>1 else None,"z":tz,"nnull":len(tn),
                     "fold0_within":f0,"fold1_within":f1},
            "exact_token_self_positive_control":pc,"gate":gate}

OUT={}
for tid in ("ZLZI","ZLZB","TTLI"):
    r=run(tid);OUT[tid]=r;print("CF3_"+tid+"="+json.dumps(r,separators=(",",":")),flush=True)
primary=bool(OUT["ZLZI"]["gate"] and OUT["ZLZB"]["final"]["observed"] is not None and OUT["ZLZB"]["final"]["observed"]>0)
print("FINAL_RESULT="+json.dumps({"phase":"CF1_CF3_GRAPH","status":"COMPLETE","results":OUT,
                                  "primary_gate":primary,"cf4_licensed":primary},separators=(",",":")),flush=True)
