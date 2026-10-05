#!/usr/bin/env python3
"""
CF1-CF3 real Voynich context-first lexical-equivalence test.
Frozen from CF0d:
- current-token morphology forbidden
- discovery split-half context fingerprint
- threshold = q=.995 of within-folio/line-position label-shuffle null
- validation conditional-indistinguishability gain <= 0
- final folds 0/1 untouched
"""
import collections, json, math, re, urllib.request
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

SEED=20261005
TOPCTX=256
MIN_DISC=20
MIN_VAL=3
MIN_TEST=3
BETA=5.0
Q=.995
NSHUFF=60
NRAND=1000
CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
c={"__name__":"core"};exec(compile(urllib.request.urlopen(CORE_URL,timeout=120).read().decode(),CORE_URL,"exec"),c)
OBJ=c["OBJ"];BIF=c["BIF"];folds=c["folds"];section=c["section"];fnum=c["fnum"]

def linepos(pos,n):
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
        for line_ord,(ls,rec) in enumerate(ld.items()):
            if str(rec.get("u",""))!="+P0":continue
            txt=rec.get("t",{}).get(tid,"")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower())]
            if len(toks)<2:continue
            for pos,t in enumerate(toks):
                if pos==0:continue
                r={"token":t,"folio":fol,"line":str(ls),"line_ord":line_ord,"pos":pos,"line_len":len(toks),
                   "bif":bif,"fold":int(folds[bif]),"section":section(fol),"lp":linepos(pos,len(toks))}
                for lag in (-2,-1,1,2):
                    j=pos+lag;r[f"n{lag:+d}"]=toks[j] if 0<=j<len(toks) else None
                rows.append(r)
    return rows

def eligibility(rows):
    dc=collections.Counter(r["token"] for r in rows if r["fold"] in (2,3))
    vc=collections.Counter(r["token"] for r in rows if r["fold"]==4)
    tc=collections.Counter(r["token"] for r in rows if r["fold"] in (0,1))
    types=sorted(t for t,n in dc.items() if n>=MIN_DISC and vc[t]>=MIN_VAL and tc[t]>=MIN_TEST)
    return types,dc,vc,tc

def vocab(rows):
    z=collections.Counter()
    for r in rows:
        if r["fold"] not in (2,3):continue
        for lag in (-2,-1,1,2):
            x=r[f"n{lag:+d}"]
            if x:z[x]+=1
    vv=[x for x,_ in z.most_common(TOPCTX)]
    return {x:i for i,x in enumerate(vv)}

def cat(x,V):return None if x is None else V.get(x,len(V))
def nk(r,lag):return (lag,r["section"],r["lp"])

def baseline(rows,V,foldset):
    K=len(V)+1
    glob={l:np.ones(K)*.25 for l in (-2,-1,1,2)}
    by=collections.defaultdict(lambda:np.zeros(K))
    for r in rows:
        if r["fold"] not in foldset:continue
        for l in (-2,-1,1,2):
            j=cat(r[f"n{l:+d}"],V)
            if j is None:continue
            glob[l][j]+=1;by[nk(r,l)][j]+=1
    for l in glob:glob[l]/=glob[l].sum()
    out={}
    for k,a in by.items():
        l=k[0];q=a+20*glob[l];out[k]=q/q.sum()
    return out,glob

def profiles(rows,V,foldset,types,base,glob,token_override=None):
    K=len(V)+1;obs=collections.defaultdict(lambda:np.zeros((4,K)));exp=collections.defaultdict(lambda:np.zeros((4,K)))
    li={l:i for i,l in enumerate((-2,-1,1,2))}
    for ii,r in enumerate(rows):
        if r["fold"] not in foldset:continue
        t=token_override[ii] if token_override is not None else r["token"]
        if t not in types:continue
        for l in (-2,-1,1,2):
            j=cat(r[f"n{l:+d}"],V)
            if j is None:continue
            q=base.get(nk(r,l),glob[l]);obs[t][li[l],j]+=1;exp[t][li[l]]+=q
    return obs,exp

def emb(types,obs,exp):
    out={}
    for t in types:
        x=np.log(np.maximum((obs[t]+BETA)/(exp[t]+BETA),1e-9)).ravel()
        x-=x.mean();n=np.linalg.norm(x);out[t]=x/n if n else x
    return out

def cross_pairs(types,Ea,Eb):
    out=[]
    for i,a in enumerate(types):
        for b in types[i+1:]:
            z=.5*(float(Ea[a]@Eb[b])+float(Ea[b]@Eb[a]))
            out.append((a,b,z))
    return out

def shuffled_tokens(rows,types,rng):
    labels=[r["token"] for r in rows]
    idx=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if r["fold"] in (2,3) and r["token"] in types:
            idx[(r["fold"],r["folio"],r["lp"])].append(i)
    out=list(labels)
    for ids in idx.values():
        vals=[out[i] for i in ids];rng.shuffle(vals)
        for i,v in zip(ids,vals):out[i]=v
    return out

def feats(r):
    d={"sec="+r["section"]:1.,"lp="+str(r["lp"]):1.}
    for l in (-2,-1,1,2):d[f"L{l}="+str(r[f'n{l:+d}'])]=1.
    d["near="+str(r["n-1"])+"|"+str(r["n+1"])]=1.
    return d

def discrim_gain(rows,a,b,trainfolds,evalfolds,C=.1):
    tr=[r for r in rows if r["fold"] in trainfolds and r["token"] in (a,b)]
    ev=[r for r in rows if r["fold"] in evalfolds and r["token"] in (a,b)]
    if min(sum(r["token"]==a for r in tr),sum(r["token"]==b for r in tr),
           sum(r["token"]==a for r in ev),sum(r["token"]==b for r in ev))<2:return None
    v=DictVectorizer();X=v.fit_transform([feats(r) for r in tr]).tocsr();Y=v.transform([feats(r) for r in ev]).tocsr()
    X.indices=X.indices.astype(np.int32);X.indptr=X.indptr.astype(np.int32)
    Y.indices=Y.indices.astype(np.int32);Y.indptr=Y.indptr.astype(np.int32)
    yt=np.array([r["token"]==b for r in tr],int);ye=np.array([r["token"]==b for r in ev],int)
    prior=(yt.sum()+.5)/(len(yt)+1)
    bll=np.mean(np.log2(np.maximum(np.where(ye==1,prior,1-prior),1e-12)))
    md=LogisticRegression(C=C,max_iter=500,solver="liblinear").fit(X,yt)
    p=md.predict_proba(Y)[:,1]
    ll=np.mean(np.log2(np.maximum(np.where(ye==1,p,1-p),1e-12)))
    return float(ll-bll)

def likelihood_model(rows,V,types,trainfolds):
    base,glob=baseline(rows,V,trainfolds)
    obs,exp=profiles(rows,V,trainfolds,set(types),base,glob)
    return base,glob,obs,exp

def ll_occ(r,V,base,glob,mult):
    li={l:i for i,l in enumerate((-2,-1,1,2))}
    z=0;n=0
    for l in (-2,-1,1,2):
        j=cat(r[f"n{l:+d}"],V)
        if j is None:continue
        p0=base.get(nk(r,l),glob[l]);q=p0*mult[li[l]];q=q/q.sum()
        z+=math.log2(max(float(q[j]),1e-15));n+=1
    return z/n if n else None

def pool_gain_final(rows,V,pairs,types):
    base,glob,obs,exp=likelihood_model(rows,V,types,(2,3,4))
    gains=[];blocks=[];foldvals={0:[],1:[]}
    for a,b in pairs:
        ma=(obs[a]+BETA)/(exp[a]+BETA);mb=(obs[b]+BETA)/(exp[b]+BETA)
        mp=(obs[a]+obs[b]+BETA)/(exp[a]+exp[b]+BETA)
        for r in rows:
            if r["fold"] not in (0,1) or r["token"] not in (a,b):continue
            sep=ma if r["token"]==a else mb
            x=ll_occ(r,V,base,glob,sep);y=ll_occ(r,V,base,glob,mp)
            if x is None or y is None:continue
            d=y-x;gains.append(d);blocks.append(r["bif"]);foldvals[r["fold"]].append(d)
    if not gains:return {"mean":None,"z":None,"n":0}
    mu=float(np.mean(gains));by=collections.defaultdict(list)
    for d,b in zip(gains,blocks):by[b].append(d)
    B=len(by);sums=np.array([sum(v)-len(v)*mu for v in by.values()])
    se=math.sqrt((B/(B-1))*float(np.sum(sums*sums))/(len(gains)**2)) if B>1 else None
    return {"mean":mu,"se":se,"z":mu/se if se and se>0 else None,"n":len(gains),"blocks":B,
            "fold0":float(np.mean(foldvals[0])) if foldvals[0] else None,
            "fold1":float(np.mean(foldvals[1])) if foldvals[1] else None}

def final_cross(rows,V,pairs,types):
    base,glob=baseline(rows,V,(2,3,4))
    o0,e0=profiles(rows,V,(0,),set(types),base,glob);o1,e1=profiles(rows,V,(1,),set(types),base,glob)
    E0=emb(types,o0,e0);E1=emb(types,o1,e1)
    vals=[]
    for a,b in pairs:vals.append(.5*(float(E0[a]@E1[b])+float(E0[b]@E1[a])))
    return vals

def secprop(rows,foldset):
    by=collections.defaultdict(collections.Counter);cnt=collections.Counter()
    for r in rows:
        if r["fold"] in foldset:
            by[r["token"]][r["section"]]+=1;cnt[r["token"]]+=1
    dom={t:(c.most_common(1)[0][0] if c else "UNK") for t,c in by.items()}
    return cnt,dom

def fbin(n):
    if n<30:return 0
    if n<60:return 1
    if n<120:return 2
    if n<240:return 3
    return 4

def matched_random_pairs(rows,types,npairs,rng):
    cnt,dom=secprop(rows,(2,3,4))
    buckets=collections.defaultdict(list)
    for t in types:buckets[(fbin(cnt[t]),dom.get(t,"UNK"))].append(t)
    keys=[k for k,v in buckets.items() if len(v)>=2]
    pairs=[];used=set()
    for _ in range(20000):
        if len(pairs)>=npairs:break
        k=keys[int(rng.integers(len(keys)))];a,b=rng.choice(buckets[k],2,replace=False);a,b=sorted((str(a),str(b)))
        if a in used or b in used:continue
        pairs.append((a,b));used|={a,b}
    return pairs

def run(tid):
    rows=build_rows(tid);types,dc,vc,tc=eligibility(rows);T=set(types);V=vocab(rows)
    base,glob=baseline(rows,V,(2,3))
    o2,e2=profiles(rows,V,(2,),T,base,glob);o3,e3=profiles(rows,V,(3,),T,base,glob)
    E2=emb(types,o2,e2);E3=emb(types,o3,e3)
    real=cross_pairs(types,E2,E3)
    rng=np.random.default_rng(SEED+sum(map(ord,tid)))
    null=[]
    for s in range(NSHUFF):
        ov=shuffled_tokens(rows,T,rng)
        so2,se2=profiles(rows,V,(2,),T,base,glob,ov);so3,se3=profiles(rows,V,(3,),T,base,glob,ov)
        S2=emb(types,so2,se2);S3=emb(types,so3,se3)
        vals=[z for a,b,z in cross_pairs(types,S2,S3)]
        null.extend(vals)
    th=float(np.quantile(np.array(null),Q))
    cand=[(a,b,z) for a,b,z in real if z>th]
    selected=[]
    for a,b,z in cand:
        g=discrim_gain(rows,a,b,(2,3),(4,))
        if g is not None and g<=0:selected.append((a,b,z,g))
    pairs=[(a,b) for a,b,z,g in selected]
    # Final fingerprint and pooling
    crossv=final_cross(rows,V,pairs,types) if pairs else []
    pool=pool_gain_final(rows,V,pairs,types) if pairs else {"mean":None,"z":None,"n":0}
    # matched random final null
    nullcross=[];nullpool=[]
    for k in range(NRAND):
        rp=matched_random_pairs(rows,types,len(pairs),rng)
        if len(rp)!=len(pairs) or not rp:continue
        cv=final_cross(rows,V,rp,types)
        if cv:nullcross.append(float(np.mean(cv)))
        pg=pool_gain_final(rows,V,rp,types)
        if pg["mean"] is not None:nullpool.append(pg["mean"])
    cm=float(np.mean(crossv)) if crossv else None
    cnullm=float(np.mean(nullcross)) if nullcross else None;cnulls=float(np.std(nullcross,ddof=1)) if len(nullcross)>1 else None
    cz=(cm-cnullm)/cnulls if cm is not None and cnulls and cnulls>0 else None
    pm=pool["mean"];pnm=float(np.mean(nullpool)) if nullpool else None;pns=float(np.std(nullpool,ddof=1)) if len(nullpool)>1 else None
    pz=(pm-pnm)/pns if pm is not None and pns and pns>0 else None
    out={"tid":tid,"eligible":len(types),"null_cross_q995":th,"candidates":len(cand),"selected_pairs":len(pairs),
         "pairs":[{"a":a,"b":b,"disc_cross":z,"val_discrim_gain":g} for a,b,z,g in selected],
         "final_cross":{"mean":cm,"random_mean":cnullm,"random_sd":cnulls,"z":cz,"nnull":len(nullcross)},
         "final_pool":{**pool,"random_mean":pnm,"random_sd":pns,"random_z":pz,"nnull":len(nullpool)},
         "gate":bool(pairs and cz is not None and cz>2 and pool.get("z") is not None and pool["z"]>0 and
                     pool.get("fold0") is not None and pool["fold0"]>=0 and pool.get("fold1") is not None and pool["fold1"]>=0)}
    return out

OUT={}
for tid in ("ZLZI","ZLZB","TTLI"):
    print("START",tid,flush=True)
    OUT[tid]=run(tid)
    print("RESULT_"+tid+"="+json.dumps(OUT[tid],separators=(",",":")),flush=True)
print("FINAL_RESULT="+json.dumps(OUT,separators=(",",":")),flush=True)
