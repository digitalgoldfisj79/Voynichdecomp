#!/usr/bin/env python3
# STARS-INNOV2 — preregistered 2026-10-08.
import collections,json,math,os,re,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/7ea7f89ac447571e8340da0e75c2ae3e8fd73689/research/latent_line_state_test_20261004.py"
m={"__name__":"latent_line_state_module"}
src=urllib.request.urlopen(LAT_URL,timeout=120).read().decode()
exec(compile(src,LAT_URL,"exec"),m)
rows=m["rows"];LINES=m["LINES"];fit_struct=m["fit_struct"];para_id=m["para_id"];K=m["K"];ALPHA=m["ALPHA"]
TR=[l for l in LINES if l["fold"] in (2,3,4)]
M,G,TH=fit_struct(TR);NG=sum(G.values())
STAR_FOL=set(f"f{n}{s}" for n in list(range(103,109))+list(range(111,115)) for s in ("r","v"))
FIT_SEEDS=list(range(202610085000,202610085300))
CAL_SEEDS=list(range(202610085300,202610085400))
BLIND_SEEDS=list(range(202610085400,202610085500))
LAGS=(1,2,3,5,10);L2=10.0

def fnum(f):return int(re.search(r"\d+",str(f)).group())

def base_prob(section,piece,pagev,parav,recent):
    cc=M.get((section,piece),{});n=sum(cc.values())
    b=np.array([(cc.get(c,0)+ALPHA*((G[c]+.5)/(NG+.5*K)))/(n+ALPHA) for c in range(K)],float)
    sc=np.log(np.maximum(b,1e-15))+np.log1p(pagev)*TH[:K]+np.log1p(parav)*TH[K:2*K]+recent*TH[-1]
    sc-=sc.max();p=np.exp(sc);p/=p.sum();return p

def generate(observed=False,seed=None):
    rng=np.random.default_rng(seed if seed is not None else 12345)
    page=collections.defaultdict(lambda:np.zeros(K,float));para=collections.defaultdict(lambda:np.zeros(K,float));hist=collections.defaultdict(list)
    by=collections.OrderedDict()
    for r in rows:
        fol=r["folio"];ln=int(r["line"]);lk=(fol,ln);pk=fol;pid=para_id(pk,ln);q=(pk,pid);yobs=int(r["start"]);pos=int(r["pos"])
        if pos==0:y=yobs
        else:
            prev_piece=hist[lk][-1][1];rc=np.zeros(K,float)
            for yy,pp in hist[lk][-6:]:rc[int(yy)]+=1
            p=base_prob(r["section"],prev_piece,page[pk],para[q],rc)
            prev=int(hist[lk][-1][0]);y=yobs if observed else int(rng.choice(K,p=p))
            if fol in STAR_FOL:
                e={"p":p,"y":y,"prev":prev,"folio":fol,"line":lk,"bif":r["bifolium"]}
                by.setdefault(lk,[]).append(e)
        page[pk][y]+=1;para[q][y]+=1;hist[lk].append((y,r["final_piece"]))
    return list(by.values())

def role(i,n):
    if i==0:return "FIRST"
    if i==1:return "SECOND"
    if i==n-1:return "FINAL"
    if i==n-2:return "PENULT"
    rel=(i-2)/max(1,n-5)
    return "EARLY" if rel<1/3 else ("MIDDLE" if rel<2/3 else "LATE")
def lbin(n):return "L7" if n<=7 else ("L10" if n<=10 else ("L14" if n<=14 else "L15P"))

def fit_tilt(es):
    P=np.stack([e["p"] for e in es]);Y=np.array([e["y"] for e in es],int);b=np.zeros(K,float);I=np.eye(K)
    def obj(bb):
        sc=np.log(np.maximum(P,1e-15))+bb[None,:];mx=sc.max(1,keepdims=True)
        z=mx[:,0]+np.log(np.exp(sc-mx).sum(1))
        return -float(np.sum(sc[np.arange(len(Y)),Y]-z))+.5*L2*float(np.dot(bb,bb))
    old=obj(b)
    for _ in range(20):
        sc=np.log(np.maximum(P,1e-15))+b[None,:];mx=sc.max(1,keepdims=True)
        Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True)
        D=Q.copy();D[np.arange(len(Y)),Y]-=1.;g=D.sum(0)+L2*b;g-=g.mean()
        if np.max(np.abs(g))<1e-9:break
        H=L2*I
        for q in Q:H+=np.diag(q)-np.outer(q,q)
        step=np.linalg.solve(H+I*1e-10,g);step-=step.mean();t=1.
        while t>1e-6:
            c=b-t*step;c-=c.mean();nv=obj(c)
            if nv<=old+1e-12:b,old=c,nv;break
            t*=.5
        if t<=1e-6:break
    return b

def fit_pos(lines,parity):
    allv=[];byrole=collections.defaultdict(list);bycell=collections.defaultdict(list)
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        n=len(seq)
        for i,e in enumerate(seq):
            ro=role(i,n);allv.append(e);byrole[ro].append(e);bycell[(ro,lbin(n))].append(e)
    globalb=fit_tilt(allv)
    rb={ro:fit_tilt(es) for ro,es in byrole.items() if len(es)>=40}
    cb={k:fit_tilt(es) for k,es in bycell.items() if len(es)>=20}
    return globalb,rb,cb

def tilt_p(p,b):
    sc=np.log(np.maximum(p,1e-15))+b;sc-=sc.max();q=np.exp(sc);return q/q.sum()

def position_adjust(lines):
    mods={0:fit_pos(lines,0),1:fit_pos(lines,1)};out=[]
    for seq in lines:
        if not seq:continue
        tr=1-fnum(seq[0]["folio"])%2;gb,rb,cb=mods[tr];n=len(seq);zz=[]
        for i,e in enumerate(seq):
            ro=role(i,n);b=cb.get((ro,lbin(n)),rb.get(ro,gb));q=tilt_p(e["p"],b)
            x=dict(e);x["p"]=q;zz.append(x)
        out.append(zz)
    return out

def entropy(p):
    q=np.asarray(p,float);q=q[q>0];return float(-(q*np.log2(q)).sum())
def safe_corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3:return 0.
    x=x-x.mean();y=y-y.mean();d=math.sqrt(float(np.dot(x,x)*np.dot(y,y)))
    return float(np.dot(x,y)/d) if d>1e-12 else 0.

def metric_vector(lines):
    events=[e for z in lines for e in z];vals=[];names=[]
    for lag in LAGS:
        z=[]
        for seq in lines:
            for i in range(lag,len(seq)):
                e=seq[i];prior=seq[i-lag]["y"];z.append((1. if e["y"]==prior else 0.)-float(e["p"][prior]))
        vals.append(float(np.mean(z)) if z else 0.);names.append(f"repeat_resid_{lag}")
    S=np.zeros((K,K));N=np.zeros(K)
    for e in events:
        a=e["prev"];v=-e["p"].copy();v[e["y"]]+=1.;S[a]+=v;N[a]+=1
    R=np.zeros_like(S)
    for a in range(K):
        if N[a]>0:R[a]=S[a]/N[a]
    vals.append(float(np.sqrt(np.mean(R*R))));names.append("transition_resid_norm")
    sb=[];alls=[]
    for seq in lines:
        z=[]
        for e in seq:
            s=-math.log2(max(float(e["p"][e["y"]]),1e-15))-entropy(e["p"]);z.append(s);alls.append(s)
        sb.append(z)
    for lag in LAGS:
        a=[];b=[]
        for z in sb:
            if len(z)>lag:a.extend(z[:-lag]);b.extend(z[lag:])
        vals.append(safe_corr(a,b));names.append(f"surprise_ac_{lag}")
    sv=np.linalg.svd(R,compute_uv=False);en=float(np.sum(sv*sv))
    vals.append(float(sv[0]**2/en) if en>1e-15 else 0.);names.append("sv1_energy")
    vals.append(float(np.sum(sv[:2]**2)/en) if en>1e-15 else 0.);names.append("sv12_energy")
    vals.append(float(np.mean(alls)));names.append("mean_excess_surprise")
    pe=[]
    for e in events:
        p=e["p"];r=-p.copy();r[e["y"]]+=1.;r/=np.sqrt(np.maximum(p*(1-p),1e-9));pe.append(float(np.mean(r*r)))
    vals.append(float(np.mean(pe)));names.append("pearson_energy")
    return np.array(vals),names,len(events)

def statistic(observed=False,seed=None):
    lines=position_adjust(generate(observed,seed));v,names,n=metric_vector(lines)
    return {"vector":v,"names":names,"n_events":n,"n_lines":len(lines)}
def task(seed):
    x=statistic(False,seed);return seed,x["vector"].tolist()

REAL=statistic(True,None)
print("REAL",json.dumps({"n_events":REAL["n_events"],"n_lines":REAL["n_lines"],"metrics":dict(zip(REAL["names"],REAL["vector"].tolist()))},separators=(",",":")),flush=True)

def setup(A):
    mu=A.mean(0);S=np.cov(A,rowvar=False);C=.75*S+.25*np.diag(np.diag(S))+np.eye(A.shape[1])*1e-9
    return mu,np.linalg.pinv(C)
def adjud(F,C,B,r):
    mu,iv=setup(F)
    def d(x):z=x-mu;return float(z@iv@z)
    cd=np.array([d(x) for x in C]);bd=np.array([d(x) for x in B]);rd=d(r);q=float(np.quantile(cd,.99));ref=np.r_[cd,bd]
    return {"real_D2":rd,"cal_q99":q,"blind_acceptance":float(np.mean(bd<=q)),"add_one_p":float((1+np.sum(ref>=rd))/(len(ref)+1))}

if __name__=="__main__":
    seeds=FIT_SEEDS+CAL_SEEDS+BLIND_SEEDS
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:rr=list(ex.map(task,seeds,chunksize=2))
    mp={s:np.array(v) for s,v in rr};F=np.vstack([mp[s] for s in FIT_SEEDS]);C=np.vstack([mp[s] for s in CAL_SEEDS]);B=np.vstack([mp[s] for s in BLIND_SEEDS])
    primary=adjud(F,C,B,REAL["vector"])
    if primary["blind_acceptance"]<.90:decision="THE_METRIC_DOES_NOT_RESOLVE_CALIBRATION_FAIL"
    elif primary["real_D2"]<=primary["cal_q99"]:decision="LINE_POSITION_LENGTH_ADJUSTMENT_ABSORBS_INNOV1"
    elif primary["add_one_p"]>.01:decision="THE_METRIC_DOES_NOT_RESOLVE_PVALUE_FAIL"
    else:decision="POSITION_ADJUSTED_RESIDUAL_ORDERED_INNOVATION_PRESENT"
    names=REAL["names"]
    groups={
      "without_repeat":[i for i,n in enumerate(names) if not n.startswith("repeat_resid_")],
      "without_transition":[i for i,n in enumerate(names) if n not in ("transition_resid_norm","sv1_energy","sv12_energy")],
      "without_surprise":[i for i,n in enumerate(names) if not(n.startswith("surprise_ac_") or n in ("mean_excess_surprise","pearson_energy"))]
    }
    abl={g:adjud(F[:,ix],C[:,ix],B[:,ix],REAL["vector"][ix]) for g,ix in groups.items()}
    out={"programme":"STARS-INNOV2","status":"complete","real":{"n_events":REAL["n_events"],"n_lines":REAL["n_lines"],"vector":REAL["vector"].tolist()},
         "metric_names":names,"primary":primary,"decision":decision,"ablations":abl}
    print("STARS_INNOV2_JSON="+json.dumps(out,separators=(",",":")),flush=True)
