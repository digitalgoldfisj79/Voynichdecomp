#!/usr/bin/env python3
"""
Context-first lexical-equivalence programme CF0-CF3.
Current-token morphology is firewalled from family discovery and inference.
"""
import collections, hashlib, json, math, pickle, re, urllib.request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SEED=20261005
TOPCTX=256
MIN_DISC=20
MIN_VAL=3
MIN_TEST=3
BETA=5.0
TOPPARTNER=20
NRAND=1200
NNULL=300

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
CI_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/67a73f80da2caefd6788de43c003701225833825/Paper/Cipher_paper/ci_corpus_parsed.pkl"
c={"__name__":"cfcore"}
exec(compile(urllib.request.urlopen(CORE_URL,timeout=120).read().decode(),CORE_URL,"exec"),c)
OBJ=c["OBJ"]; BIF=c["BIF"]; folds=c["folds"]; section=c["section"]; fnum=c["fnum"]

def linepos(pos,n):
    if pos<=1:return 0
    if pos>=n-2:return 2
    return 1

def build_rows(tid):
    rows=[]
    for fol,ld in OBJ["pages"].items():
        n=fnum(fol)
        if n not in BIF: continue
        bif=BIF[n]
        if bif not in folds: continue
        for line_ord,(ls,rec) in enumerate(ld.items()):
            if str(rec.get("u",""))!="+P0": continue
            txt=rec.get("t",{}).get(tid,"")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower())]
            if len(toks)<2: continue
            for pos,t in enumerate(toks):
                if pos==0: continue  # LINE_ENTRY firewall
                rr=dict(token=t,folio=fol,line=str(ls),line_ord=line_ord,pos=pos,line_len=len(toks),
                        bif=bif,fold=int(folds[bif]),section=section(fol),lp=linepos(pos,len(toks)))
                for lag in (-2,-1,1,2):
                    j=pos+lag
                    rr[f"n{lag:+d}"]=toks[j] if 0<=j<len(toks) else None
                rows.append(rr)
    return rows

def context_vocab(rows,disc_folds=(2,3),top=TOPCTX):
    z=collections.Counter()
    for r in rows:
        if r["fold"] not in disc_folds: continue
        for lag in (-2,-1,1,2):
            x=r[f"n{lag:+d}"]
            if x:z[x]+=1
    topv=[x for x,_ in z.most_common(top)]
    return {x:i for i,x in enumerate(topv)}, topv

def cat_of(x,vmap):
    if x is None:return None
    return vmap.get(x,len(vmap))

def eligible_types(rows):
    dc=collections.Counter(r["token"] for r in rows if r["fold"] in (2,3))
    vc=collections.Counter(r["token"] for r in rows if r["fold"]==4)
    tc=collections.Counter(r["token"] for r in rows if r["fold"] in (0,1))
    return sorted(t for t,n in dc.items() if n>=MIN_DISC and vc[t]>=MIN_VAL and tc[t]>=MIN_TEST),dc,vc,tc

def nuis_key(r,lag): return (lag,r["section"],r["lp"])
def fit_baseline(rows,vmap,foldset):
    K=len(vmap)+1
    glob={lag:np.ones(K,float)*.25 for lag in (-2,-1,1,2)}
    by=collections.defaultdict(lambda:np.zeros(K,float))
    for r in rows:
        if r["fold"] not in foldset:continue
        for lag in (-2,-1,1,2):
            j=cat_of(r[f"n{lag:+d}"],vmap)
            if j is None:continue
            glob[lag][j]+=1
            by[nuis_key(r,lag)][j]+=1
    for lag in glob:glob[lag]/=glob[lag].sum()
    out={}
    for k,a in by.items():
        lag=k[0];v=a+20.0*glob[lag];out[k]=v/v.sum()
    return out,glob

def baseprob(r,lag,j,base,glob):
    p=base.get(nuis_key(r,lag),glob[lag])
    return max(float(p[j]),1e-12)

def profiles(rows,vmap,foldset,typeset,base,glob):
    K=len(vmap)+1
    obs=collections.defaultdict(lambda:np.zeros((4,K),float))
    exp=collections.defaultdict(lambda:np.zeros((4,K),float))
    lags=(-2,-1,1,2); li={z:i for i,z in enumerate(lags)}
    for r in rows:
        if r["fold"] not in foldset or r["token"] not in typeset:continue
        t=r["token"]
        for lag in lags:
            j=cat_of(r[f"n{lag:+d}"],vmap)
            if j is None:continue
            q=base.get(nuis_key(r,lag),glob[lag])
            obs[t][li[lag],j]+=1; exp[t][li[lag]]+=q
    return obs,exp

def multiplier(o,e,beta=BETA):
    return (o+beta)/(e+beta)

def embedding(types,obs,exp):
    vv=[]
    for t in types:
        m=multiplier(obs[t],exp[t])
        x=np.log(np.maximum(m,1e-9)).ravel()
        x-=x.mean()
        n=np.linalg.norm(x)
        vv.append(x/n if n>0 else x)
    return np.stack(vv)

def score_rows(rows,vmap,types,obs,exp,base,glob,poolmap=None):
    """Return per-current-token context log2 likelihoods; poolmap maps token->family tuple."""
    type_set=set(types); lags=(-2,-1,1,2); li={z:i for i,z in enumerate(lags)}
    cache={}
    if poolmap is None:
        for t in types:cache[t]=multiplier(obs[t],exp[t])
    else:
        fams={}
        for t in types:fams.setdefault(poolmap.get(t,(t,)),[]).append(t)
        for fam,mem in fams.items():
            oo=sum((obs[t] for t in mem),np.zeros_like(obs[mem[0]]))
            ee=sum((exp[t] for t in mem),np.zeros_like(exp[mem[0]]))
            mm=multiplier(oo,ee)
            for t in mem:cache[t]=mm
    out=[]
    for r in rows:
        t=r["token"]
        if t not in type_set:continue
        ll=0.;n=0
        mm=cache[t]
        for lag in lags:
            j=cat_of(r[f"n{lag:+d}"],vmap)
            if j is None:continue
            p0=base.get(nuis_key(r,lag),glob[lag])
            q=p0*mm[li[lag]]
            den=q.sum()
            if den<=0:continue
            ll+=math.log2(max(float(q[j]/den),1e-15));n+=1
        if n:out.append((r,ll/n,n))
    return out

def pair_gain(pair,valrows,vmap,obs,exp,base,glob):
    a,b=pair
    sep=score_rows(valrows,vmap,[a,b],obs,exp,base,glob)
    pool={(a):(a,b),(b):(a,b)}
    poo=score_rows(valrows,vmap,[a,b],obs,exp,base,glob,pool)
    if not sep or len(sep)!=len(poo):return None
    # same iteration order
    d=[y[1]-x[1] for x,y in zip(sep,poo)]
    return float(np.mean(d)) if d else None

def domsec(rows,foldset):
    by=collections.defaultdict(collections.Counter)
    for r in rows:
        if r["fold"] in foldset:by[r["token"]][r["section"]]+=1
    return {t:(c.most_common(1)[0][0] if c else "UNK") for t,c in by.items()}

def fbin(n):
    if n<30:return 0
    if n<60:return 1
    if n<120:return 2
    if n<240:return 3
    return 4

def make_random_pairs(types,dc,ds,rng,n=NRAND):
    buckets=collections.defaultdict(list)
    for t in types:buckets[(fbin(dc[t]),ds.get(t,"UNK"))].append(t)
    pairs=[]
    for _ in range(n*4):
        if len(pairs)>=n:break
        k=list(buckets)[rng.integers(len(buckets))]
        xs=buckets[k]
        if len(xs)<2:continue
        a,b=rng.choice(xs,2,replace=False)
        p=tuple(sorted((str(a),str(b))))
        if p not in pairs:pairs.append(p)
    return pairs

def discover_pairs(rows,tid):
    types,dc,vc,tc=eligible_types(rows)
    vmap,vlist=context_vocab(rows)
    base,glob=fit_baseline(rows,vmap,(2,3))
    obs,exp=profiles(rows,vmap,(2,3),set(types),base,glob)
    E=embedding(types,obs,exp);sim=E@E.T
    cand=set()
    for i,t in enumerate(types):
        ix=np.argsort(-sim[i])
        for j in ix[1:TOPPARTNER+1]:
            cand.add(tuple(sorted((t,types[int(j)]))))
    val=[r for r in rows if r["fold"]==4]
    rng=np.random.default_rng(SEED+sum(map(ord,tid)))
    ds=domsec(rows,(2,3))
    rp=make_random_pairs(types,dc,ds,rng)
    rg=[pair_gain(p,val,vmap,obs,exp,base,glob) for p in rp]
    rg=np.array([x for x in rg if x is not None],float)
    q99=float(np.quantile(rg,.99)) if len(rg)>20 else float("inf")
    accepted=[]
    for p in cand:
        g=pair_gain(p,val,vmap,obs,exp,base,glob)
        if g is not None and g>0 and g>q99:accepted.append((g,p))
    accepted.sort(reverse=True)
    used=set();pairs=[]
    for g,p in accepted:
        if p[0] in used or p[1] in used:continue
        pairs.append((p[0],p[1],g));used.update(p)
    return dict(types=types,dc=dc,tc=tc,vmap=vmap,base=base,glob=glob,obs=obs,exp=exp,
                q99=q99,random_val_gain=rg,pairs=pairs,n_candidates=len(cand),n_accepted_raw=len(accepted))

def block_stats(delta,blocks):
    if len(delta)==0:return {"mean":None,"se":None,"z":None,"n":0,"blocks":0}
    mu=float(np.mean(delta));by=collections.defaultdict(list)
    for x,b in zip(delta,blocks):by[b].append(float(x))
    B=len(by)
    if B<2:return {"mean":mu,"se":None,"z":None,"n":len(delta),"blocks":B}
    sums=np.array([sum(v)-len(v)*mu for v in by.values()])
    se=math.sqrt((B/(B-1))*float(np.sum(sums*sums))/(len(delta)**2))
    return {"mean":mu,"se":se,"z":mu/se if se>0 else None,"n":len(delta),"blocks":B}

def evaluate_real(rows,tid,D):
    types=D["types"];vmap=D["vmap"]
    # refit baseline + profiles on 2,3,4 after frozen pair selection
    base,glob=fit_baseline(rows,vmap,(2,3,4))
    obs,exp=profiles(rows,vmap,(2,3,4),set(types),base,glob)
    pairmap={}
    for a,b,g in D["pairs"]:
        pairmap[a]=(a,b);pairmap[b]=(a,b)
    test=[r for r in rows if r["fold"] in (0,1)]
    merged=set(pairmap)
    sep=score_rows(test,vmap,sorted(merged),obs,exp,base,glob)
    poo=score_rows(test,vmap,sorted(merged),obs,exp,base,glob,pairmap)
    delta=np.array([y[1]-x[1] for x,y in zip(sep,poo)],float)
    blocks=[x[0]["bif"] for x in sep]
    bs=block_stats(delta,blocks)
    folds={}
    for f in (0,1):
        ix=[i for i,x in enumerate(sep) if x[0]["fold"]==f]
        folds[str(f)]=float(np.mean(delta[ix])) if ix else None
    # exact-token positive control against nuisance baseline, on all eligible types
    exact=score_rows(test,vmap,types,obs,exp,base,glob)
    eg=[];eb=[]
    for r,ll,n in exact:
        bll=0.;nn=0
        for lag in (-2,-1,1,2):
            j=cat_of(r[f"n{lag:+d}"],vmap)
            if j is None:continue
            bll+=math.log2(baseprob(r,lag,j,base,glob));nn+=1
        if nn:eg.append(ll-bll/nn);eb.append(r["bif"])
    exact_bs=block_stats(np.array(eg),eb)
    # matched random partitions with same number pairs, drawn from eligibility buckets
    rng=np.random.default_rng(SEED+7000+sum(map(ord,tid)))
    ds=domsec(rows,(2,3,4));dc=collections.Counter(r["token"] for r in rows if r["fold"] in (2,3,4))
    null=[]
    for _ in range(NNULL):
        rp=make_random_pairs(types,dc,ds,rng,n=max(len(D["pairs"])*5,100))
        rng.shuffle(rp);used=set();chosen=[]
        for a,b in rp:
            if a in used or b in used:continue
            chosen.append((a,b));used|={a,b}
            if len(chosen)>=len(D["pairs"]):break
        if len(chosen)<len(D["pairs"]) or not chosen:continue
        pm={}
        for a,b in chosen:pm[a]=(a,b);pm[b]=(a,b)
        ss=score_rows(test,vmap,sorted(pm),obs,exp,base,glob)
        pp=score_rows(test,vmap,sorted(pm),obs,exp,base,glob,pm)
        if ss and len(ss)==len(pp):null.append(float(np.mean([y[1]-x[1] for x,y in zip(ss,pp)])))
    nm=float(np.mean(null)) if null else None;ns=float(np.std(null,ddof=1)) if len(null)>1 else None
    nz=(bs["mean"]-nm)/ns if ns and ns>0 else None
    return {"tid":tid,"eligible_types":len(types),"validation_q99_random":D["q99"],
            "candidate_pairs":D["n_candidates"],"raw_pass_pairs":D["n_accepted_raw"],
            "frozen_pairs":[{"a":a,"b":b,"val_gain_bits":g} for a,b,g in D["pairs"]],
            "n_frozen_pairs":len(D["pairs"]),"pooled_vs_exact":bs,"fold_gain":folds,
            "random_partition_null":{"n":len(null),"mean":nm,"sd":ns,"z":nz},
            "exact_token_positive_control":exact_bs,
            "gate":bool(len(D["pairs"])>0 and bs["z"] is not None and bs["z"]>2 and
                        folds["0"] is not None and folds["0"]>0 and folds["1"] is not None and folds["1"]>0 and
                        nz is not None and nz>2)}

# ---------- CF0 synthetic calibration ----------
def ci_words():
    ci=pickle.loads(urllib.request.urlopen(CI_URL,timeout=120).read())
    return [str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1]

def synth_rows(shuffle=False):
    W=ci_words()[:24000]
    vocab=[w for w,_ in collections.Counter(W[:16000]).most_common(24)]
    vid={w:i for i,w in enumerate(vocab)};OTHER=24
    src=np.array([vid.get(w,OTHER) for w in W],int)
    if shuffle:
        rng=np.random.default_rng(SEED+999);src=rng.permutation(src)
    rng=np.random.default_rng(SEED+123+(1 if shuffle else 0))
    variants={s:[f"V{s:02d}_{j}" for j in range(4)] for s in range(25)}
    surf=[variants[int(s)][int(rng.integers(4))] for s in src]
    # one artificial line; use split labels as folds 2/3,4,0/1
    rows=[]
    for i,t in enumerate(surf):
        if i<8000:f=2
        elif i<16000:f=3
        elif i<20000:f=4
        elif i<22000:f=0
        else:f=1
        r=dict(token=t,source=int(src[i]),folio="SYN",line="L",line_ord=0,pos=i,line_len=len(surf),
               bif=f"S{f}_{i//400:03d}",fold=f,section="SYN",lp=1)
        for lag in (-2,-1,1,2):
            j=i+lag;r[f"n{lag:+d}"]=surf[j] if 0<=j<len(surf) else None
        rows.append(r)
    return rows,variants

def generic_discover_synth(rows):
    # override thresholds naturally; same machinery, section fixed.
    return discover_pairs(rows,"SYN")

def cf0():
    out={}
    for label,sh in (("ordered",False),("shuffled",True)):
        rows,var= synth_rows(sh)
        D=generic_discover_synth(rows)
        ev=evaluate_real(rows,"SYN"+label,D)
        truth=[]
        for p in ev["frozen_pairs"]:
            sa=int(p["a"].split("_")[0][1:]);sb=int(p["b"].split("_")[0][1:])
            truth.append(sa==sb)
        ev["same_source_fraction_selected"]=float(np.mean(truth)) if truth else None
        ev["same_source_selected"]=int(sum(truth));ev["selected_pairs"]=len(truth)
        out[label]=ev
    # calibration gate: ordered finds >=5 pairs, >=80% same-source, gate true;
    # shuffled must not pass with comparable true pair recovery.
    o=out["ordered"];s=out["shuffled"]
    out["calibration_pass"]=bool(o["selected_pairs"]>=5 and (o["same_source_fraction_selected"] or 0)>=.8 and o["gate"] and
                                 not (s["gate"] and (s["same_source_fraction_selected"] or 0)>=.8))
    return out

def main():
    cf0res=cf0()
    print("CF0_RESULT="+json.dumps(cf0res,separators=(",",":")),flush=True)
    if not cf0res["calibration_pass"]:
        print("FINAL_RESULT="+json.dumps({"phase":"CF0","status":"ASSAY_FAIL","cf0":cf0res},separators=(",",":")),flush=True)
        return
    real={}
    for tid in ("ZLZI","ZLZB","TTLI"):
        rows=build_rows(tid)
        D=discover_pairs(rows,tid)
        ev=evaluate_real(rows,tid,D)
        real[tid]=ev
        print("CF3_"+tid+"="+json.dumps(ev,separators=(",",":")),flush=True)
    gate=bool(real["ZLZI"]["gate"] and real["ZLZB"]["pooled_vs_exact"]["mean"] is not None and
              real["ZLZB"]["pooled_vs_exact"]["mean"]>0)
    print("FINAL_RESULT="+json.dumps({"phase":"CF0_CF3","status":"COMPLETE","cf0":cf0res,"real":real,
                                      "cf3_primary_gate":gate,
                                      "cf4_licensed":gate},separators=(",",":")),flush=True)

if __name__=="__main__":main()
