#!/usr/bin/env python3
# ABA-XTRANS1 — preregistered 2026-10-08.
import collections,json,math,os,re,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
CORPUS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
lm={"__name__":"latent"}
exec(compile(urllib.request.urlopen(LAT_URL,timeout=120).read().decode(),LAT_URL,"exec"),lm)

segment=lm["segment"]; ST=lm["ST"]; K=lm["K"]; fit_struct=lm["fit_struct"]; para_id=lm["para_id"]; folds=lm["folds"]; ALPHA=lm["ALPHA"]
OBJ=json.loads(urllib.request.urlopen(CORPUS_URL,timeout=120).read())
TIDS=("TTLI","VDRB")
COHORTS=("HERBAL_A","HERBAL_B","BALNEO_FULL","RECIPES_FULL")
SEEDS=list(range(202610093000,202610093400))
L2=10.0
PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),
(25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),
(43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),
(71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),
(94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112),(108,111)]
BIF={n:f"B{a:03d}_{b:03d}" for a,b in PAIRS for n in (a,b)}

def fnum(f):
    z=re.match(r"f(\d+)",str(f)); return int(z.group(1)) if z else -1
def section(f):
    n=fnum(f)
    if n<=66:return "HERBAL"
    if n<=73:return "ASTRO"
    if 75<=n<=84:return "BIO"
    if 85<=n<=102:return "PHARMA"
    if 103<=n<=116:return "RECIPES"
    return "UNK"
def cohort(f):
    n=fnum(f)
    if 1<=n<=25:return "HERBAL_A"
    if 26<=n<=56:return "HERBAL_B"
    if 75<=n<=84:return "BALNEO_FULL"
    if 103<=n<=116:return "RECIPES_FULL"
    return None

def build_rows_tid(tid):
    rr=[]
    for fol,ld in OBJ["pages"].items():
        n=fnum(fol)
        if n not in BIF:continue
        bif=BIF[n]
        if bif not in folds:continue
        for ls,rec in ld.items():
            if str(rec.get("u",""))!="+P0":continue
            toks=[]
            for t in rec.get("t",{}).get(tid,"").split():
                t=t.lower()
                if not re.fullmatch(r"[a-z]+",t):continue
                try:ps=segment(t)
                except Exception:continue
                toks.append((t,ps))
            for pos,(t,ps) in enumerate(toks):
                rr.append({"folio":fol,"line":line_raw,"line_num":line_num,"pos":pos,"line_len":len(toks),"bif":bif,"fold":int(folds[bif]),
                           "section":section(fol),"token":t,"start":int(ST[ps[0]]),"final_piece":ps[-1]})
    return rr

def make_lines(rr):
    page=collections.defaultdict(lambda:np.zeros(K,float)); para=collections.defaultdict(lambda:np.zeros(K,float)); hist=collections.defaultdict(list)
    by=collections.OrderedDict()
    for r in rr:
        fol=r["folio"];ln=r["line"];lk=(fol,ln);pk=fol;pq=(pk,para_id(pk,r["line_num"]));y=r["start"]
        rec=by.setdefault(lk,{"folio":fol,"line":ln,"bif":r["bif"],"fold":r["fold"],"events":[]})
        if r["pos"]>0:
            prev=hist[lk][-1][1];rc=np.zeros(K,float)
            for yy,pp in hist[lk][-6:]:rc[int(yy)]+=1
            rec["events"].append({"y":y,"section":r["section"],"piece":prev,"page":page[pk].copy(),"para":para[pq].copy(),"recent":rc})
        page[pk][y]+=1;para[pq][y]+=1;hist[lk].append((y,r["final_piece"]))
    return [v for v in by.values() if len(v["events"])>=3]

ROWS={tid:build_rows_tid(tid) for tid in TIDS}
LINES={tid:make_lines(ROWS[tid]) for tid in TIDS}
MODELS={}
for tid in TIDS:
    MODELS[tid]={}
    for ff in range(5):
        MODELS[tid][ff]=fit_struct([l for l in LINES[tid] if l["fold"]!=ff])

def prob(model,sec,piece,pagev,parav,recent):
    M,G,th=model;ng=sum(G.values());cc=M.get((sec,piece),{});n=sum(cc.values())
    b=np.array([(cc.get(c,0)+ALPHA*((G[c]+.5)/(ng+.5*K)))/(n+ALPHA) for c in range(K)],float)
    sc=np.log(np.maximum(b,1e-15))+np.log1p(pagev)*th[:K]+np.log1p(parav)*th[K:2*K]+recent*th[-1]
    sc-=sc.max();q=np.exp(sc);q/=q.sum();return q

def generate_tid(tid,observed=False,seed=None):
    rng=np.random.default_rng(seed if seed is not None else 12345)
    page=collections.defaultdict(lambda:np.zeros(K,float));para=collections.defaultdict(lambda:np.zeros(K,float));hist=collections.defaultdict(list)
    by={c:collections.OrderedDict() for c in COHORTS}
    for r in ROWS[tid]:
        fol=r["folio"];ln=r["line"];lk=(fol,ln);pk=fol;pq=(pk,para_id(pk,r["line_num"]));yobs=r["start"];co=cohort(fol)
        if r["pos"]==0:y=yobs
        else:
            prevpiece=hist[lk][-1][1];rc=np.zeros(K,float)
            for yy,pp in hist[lk][-6:]:rc[int(yy)]+=1
            p=prob(MODELS[tid][r["fold"]],r["section"],prevpiece,page[pk],para[pq],rc)
            y=yobs if observed else int(rng.choice(K,p=p))
            if co is not None:by[co].setdefault(lk,[]).append({"p":p,"y":y,"folio":fol})
        page[pk][y]+=1;para[pq][y]+=1;hist[lk].append((y,r["final_piece"]))
    return {c:list(by[c].values()) for c in COHORTS}

def role(i,n):
    if i==0:return "FIRST"
    if i==1:return "SECOND"
    if i==n-1:return "FINAL"
    if i==n-2:return "PENULT"
    rel=(i-2)/max(1,n-5)
    return "EARLY" if rel<1/3 else ("MIDDLE" if rel<2/3 else "LATE")
def lbin(n):return "L7" if n<=7 else ("L10" if n<=10 else ("L14" if n<=14 else "L15P"))
def fit_tilt(es):
    if not es:return np.zeros(K,float)
    P=np.stack([e["p"] for e in es]);Y=np.array([e["y"] for e in es],int);b=np.zeros(K,float);I=np.eye(K)
    def obj(bb):
        sc=np.log(np.maximum(P,1e-15))+bb[None,:];mx=sc.max(1,keepdims=True);z=mx[:,0]+np.log(np.exp(sc-mx).sum(1))
        return -float(np.sum(sc[np.arange(len(Y)),Y]-z))+.5*L2*float(bb@bb)
    old=obj(b)
    for _ in range(20):
        sc=np.log(np.maximum(P,1e-15))+b[None,:];mx=sc.max(1,keepdims=True);Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True)
        D=Q.copy();D[np.arange(len(Y)),Y]-=1.;g=D.sum(0)+L2*b;g-=g.mean()
        if np.max(np.abs(g))<1e-9:break
        H=L2*I
        for q in Q:H+=np.diag(q)-np.outer(q,q)
        step=np.linalg.solve(H+I*1e-10,g);step-=step.mean();tt=1.
        while tt>1e-6:
            cc=b-tt*step;cc-=cc.mean();nv=obj(cc)
            if nv<=old+1e-12:b,old=cc,nv;break
            tt*=.5
        if tt<=1e-6:break
    return b
def tilt_p(p,b):
    sc=np.log(np.maximum(p,1e-15))+b;sc-=sc.max();q=np.exp(sc);return q/q.sum()
def position_adjust(lines):
    mods={}
    for par in (0,1):
        allv=[];byrole=collections.defaultdict(list);bycell=collections.defaultdict(list)
        for seq in lines:
            if not seq or fnum(seq[0]["folio"])%2!=par:continue
            n=len(seq)
            for i,e in enumerate(seq):
                ro=role(i,n);allv.append(e);byrole[ro].append(e);bycell[(ro,lbin(n))].append(e)
        gb=fit_tilt(allv);rb={r:fit_tilt(v) for r,v in byrole.items() if len(v)>=40};cb={k:fit_tilt(v) for k,v in bycell.items() if len(v)>=20}
        mods[par]=(gb,rb,cb)
    out=[]
    for seq in lines:
        if not seq:continue
        tr=1-fnum(seq[0]["folio"])%2;gb,rb,cb=mods[tr];n=len(seq);zz=[]
        for i,e in enumerate(seq):
            ro=role(i,n);b=cb.get((ro,lbin(n)),rb.get(ro,gb));x=dict(e);x["p"]=tilt_p(e["p"],b);zz.append(x)
        out.append(zz)
    return out

def r2(lines):
    z=[]
    for seq in lines:
        for t in range(2,len(seq)):
            a=int(seq[t-2]["y"]);e=seq[t];z.append((1. if int(e["y"])==a else 0.)-float(e["p"][a]))
    return float(np.mean(z)) if z else 0.

def analyze_tid(tid,observed=False,seed=None):
    D=generate_tid(tid,observed,seed);out={}
    for c in COHORTS:
        pa=position_adjust(D[c]);out[c]={"R2":r2(pa),"n_events":sum(len(x) for x in pa),"n_lines":len(pa)}
    return out

REAL={tid:analyze_tid(tid,True,None) for tid in TIDS}
print("REAL",json.dumps(REAL,separators=(",",":")),flush=True)

def task(seed):
    return seed,{
        "TTLI":analyze_tid("TTLI",False,seed),
        "VDRB":analyze_tid("VDRB",False,seed+10000000)
    }

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:rr=list(ex.map(task,SEEDS,chunksize=2))
    mp={s:v for s,v in rr}
    cells={};calz={};blindz={};realz={}
    for tid in TIDS:
        for c in COHORTS:
            arr=np.array([mp[s][tid][c]["R2"] for s in SEEDS],float)
            scale=arr[:200];cal=arr[200:300];blind=arr[300:400]
            mu=float(scale.mean());sd=max(float(scale.std(ddof=1)),1e-12)
            cz=(cal-mu)/sd;bz=(blind-mu)/sd;rz=(REAL[tid][c]["R2"]-mu)/sd
            key=(tid,c);calz[key]=cz;blindz[key]=bz;realz[key]=float(rz)
            cells[f"{tid}:{c}"]={"real_R2":REAL[tid][c]["R2"],"real_Z":float(rz),"null_mean":mu,"null_sd":sd,
                                "own_q95_Z":float(np.quantile(cz,.95)),"n_events":REAL[tid][c]["n_events"],"n_lines":REAL[tid][c]["n_lines"]}
    cal_min_tid={tid:np.min(np.vstack([calz[(tid,c)] for c in COHORTS]),axis=0) for tid in TIDS}
    blind_min_tid={tid:np.min(np.vstack([blindz[(tid,c)] for c in COHORTS]),axis=0) for tid in TIDS}
    real_min_tid={tid:min(realz[(tid,c)] for c in COHORTS) for tid in TIDS}
    cal_min8=np.minimum(cal_min_tid["TTLI"],cal_min_tid["VDRB"])
    blind_min8=np.minimum(blind_min_tid["TTLI"],blind_min_tid["VDRB"])
    real_min8=min(real_min_tid.values())
    q95=float(np.quantile(cal_min8,.95));blind_acc=float(np.mean(blind_min8<=q95))
    p=float((1+np.sum(np.r_[cal_min8,blind_min8]>=real_min8))/(len(cal_min8)+len(blind_min8)+1))
    tq={tid:float(np.quantile(cal_min_tid[tid],.95)) for tid in TIDS}
    passed=bool(blind_acc>=.90 and real_min8>q95 and p<=.05 and all(real_min_tid[tid]>tq[tid] for tid in TIDS))
    out={"programme":"ABA-XTRANS1","status":"complete",
         "primary":{"real_MIN8":real_min8,"cal_q95_MIN8":q95,"blind_acceptance":blind_acc,"add_one_p":p,
                    "real_MIN_by_transcription":real_min_tid,"q95_MIN_by_transcription":tq,
                    "decision":"PORTABLE_LAG2_RETURN_REPLICATES" if passed else "PORTABLE_LAG2_RETURN_NOT_REPLICATED"},
         "cells":cells}
    print("ABA_XTRANS1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
