#!/usr/bin/env python3
# STARS-INNOV1 — preregistered 2026-10-08.
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
FIT_SEEDS=list(range(202610082000,202610082300))
CAL_SEEDS=list(range(202610082300,202610082400))
BLIND_SEEDS=list(range(202610082400,202610082500))
LAGS=(1,2,3,5,10)

def base_prob(section,piece,pagev,parav,recent):
    cc=M.get((section,piece),{});n=sum(cc.values())
    b=np.array([(cc.get(c,0)+ALPHA*((G[c]+.5)/(NG+.5*K)))/(n+ALPHA) for c in range(K)],float)
    sc=np.log(np.maximum(b,1e-15))+np.log1p(pagev)*TH[:K]+np.log1p(parav)*TH[K:2*K]+recent*TH[-1]
    sc-=sc.max();p=np.exp(sc);p/=p.sum();return p

def entropy(p):
    q=np.asarray(p,float);q=q[q>0]
    return float(-(q*np.log2(q)).sum())

def generate(observed=False,seed=None):
    rng=np.random.default_rng(seed if seed is not None else 12345)
    page=collections.defaultdict(lambda:np.zeros(K,float))
    para=collections.defaultdict(lambda:np.zeros(K,float))
    linehist=collections.defaultdict(list)
    byline=collections.OrderedDict()
    for r in rows:
        fol=r["folio"];ln=r["line"];lk=(fol,ln);pk=fol;q=(pk,para_id(pk,ln))
        yobs=int(r["start"]);pos=int(r["pos"])
        if pos==0:
            y=yobs
        else:
            prev_piece=linehist[lk][-1][1]
            rc=np.zeros(K,float)
            for yy,pp in linehist[lk][-6:]:rc[int(yy)]+=1
            p=base_prob(r["section"],prev_piece,page[pk],para[q],rc)
            prev=int(linehist[lk][-1][0])
            y=yobs if observed else int(rng.choice(K,p=p))
            if fol in STAR_FOL:
                e={"p":p,"y":y,"prev":prev,"folio":fol,"line":lk,"bif":r["bifolium"]}
                byline.setdefault(lk,[]).append(e)
        page[pk][y]+=1;para[q][y]+=1;linehist[lk].append((y,r["final_piece"]))
    lines=[z for z in byline.values() if len(z)>=3]
    events=[e for z in lines for e in z]
    return events,lines

def safe_corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3:return 0.
    x=x-x.mean();y=y-y.mean();d=math.sqrt(float(np.dot(x,x)*np.dot(y,y)))
    return float(np.dot(x,y)/d) if d>1e-12 else 0.

def metric_vector(events,lines):
    vals=[];names=[]
    for lag in LAGS:
        z=[]
        for seq in lines:
            for i in range(lag,len(seq)):
                e=seq[i];prior=seq[i-lag]["y"]
                z.append((1. if e["y"]==prior else 0.)-float(e["p"][prior]))
        vals.append(float(np.mean(z)) if z else 0.);names.append(f"repeat_resid_{lag}")
    S=np.zeros((K,K),float);N=np.zeros(K,float)
    for e in events:
        a=e["prev"]
        if a<0:continue
        v=-e["p"].copy();v[e["y"]]+=1
        S[a]+=v;N[a]+=1
    R=np.zeros_like(S)
    for a in range(K):
        if N[a]>0:R[a]=S[a]/N[a]
    vals.append(float(np.sqrt(np.mean(R*R))));names.append("transition_resid_norm")
    s_by=[];all_s=[]
    for seq in lines:
        zz=[]
        for e in seq:
            x=-math.log2(max(float(e["p"][e["y"]]),1e-15))-entropy(e["p"])
            zz.append(x);all_s.append(x)
        s_by.append(zz)
    for lag in LAGS:
        a=[];b=[]
        for z in s_by:
            if len(z)>lag:a.extend(z[:-lag]);b.extend(z[lag:])
        vals.append(safe_corr(a,b));names.append(f"surprise_ac_{lag}")
    sv=np.linalg.svd(R,compute_uv=False);en=float(np.sum(sv*sv))
    vals.append(float((sv[0]**2)/en) if en>1e-15 else 0.);names.append("sv1_energy")
    vals.append(float(np.sum(sv[:2]**2)/en) if en>1e-15 else 0.);names.append("sv12_energy")
    vals.append(float(np.mean(all_s)) if all_s else 0.);names.append("mean_excess_surprise")
    pe=[]
    for e in events:
        p=e["p"];y=e["y"];r=-p.copy();r[y]+=1.;den=np.sqrt(np.maximum(p*(1-p),1e-9));r/=den;pe.append(float(np.mean(r*r)))
    vals.append(float(np.mean(pe)) if pe else 0.);names.append("pearson_energy")
    return np.array(vals,float),names

REAL_EVENTS,REAL_LINES=generate(True,None)
REAL,NAMES=metric_vector(REAL_EVENTS,REAL_LINES)
print("REAL",json.dumps({"n_events":len(REAL_EVENTS),"n_lines":len(REAL_LINES),"metrics":dict(zip(NAMES,REAL.tolist()))},separators=(",",":")),flush=True)

def task(seed):
    ev,ln=generate(False,seed);v,_=metric_vector(ev,ln);return seed,v.tolist()

def setup(A):
    mu=A.mean(0);S=np.cov(A,rowvar=False);D=np.diag(np.diag(S));C=.75*S+.25*D+np.eye(A.shape[1])*1e-9
    return mu,np.linalg.pinv(C)
def dist(v,mu,inv):
    d=v-mu;return float(d@inv@d)

def adjudicate(F,C,B,real):
    mu,inv=setup(F);cd=np.array([dist(x,mu,inv) for x in C]);bd=np.array([dist(x,mu,inv) for x in B]);rd=dist(real,mu,inv)
    q99=float(np.quantile(cd,.99));accept=float(np.mean(bd<=q99));ref=np.r_[cd,bd]
    p=float((1+np.sum(ref>=rd))/(len(ref)+1))
    return {"real_D2":rd,"cal_q99":q99,"blind_acceptance":accept,"add_one_p":p,
            "fit_null_median_D2":float(np.median([dist(x,mu,inv) for x in F])),
            "cal_median_D2":float(np.median(cd)),"blind_median_D2":float(np.median(bd))}

if __name__=="__main__":
    seeds=FIT_SEEDS+CAL_SEEDS+BLIND_SEEDS
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(task,seeds,chunksize=2))
    mp={s:np.array(v,float) for s,v in rr}
    F=np.vstack([mp[s] for s in FIT_SEEDS]);C=np.vstack([mp[s] for s in CAL_SEEDS]);B=np.vstack([mp[s] for s in BLIND_SEEDS])
    primary=adjudicate(F,C,B,REAL)
    if primary["blind_acceptance"]<.90:decision="THE_METRIC_DOES_NOT_RESOLVE_CALIBRATION_FAIL"
    elif primary["real_D2"]<=primary["cal_q99"]:decision="KNOWN_COMPACT_SELECT_MODEL_SUFFICIENT_FOR_INNOVATION_PANEL"
    elif primary["add_one_p"]>.01:decision="THE_METRIC_DOES_NOT_RESOLVE_PVALUE_FAIL"
    else:decision="RESIDUAL_ORDERED_INNOVATION_PRESENT"
    groups={
      "without_repeat":[i for i,n in enumerate(NAMES) if not n.startswith("repeat_resid_")],
      "without_transition":[i for i,n in enumerate(NAMES) if n not in ("transition_resid_norm","sv1_energy","sv12_energy")],
      "without_surprise":[i for i,n in enumerate(NAMES) if not (n.startswith("surprise_ac_") or n in ("mean_excess_surprise","pearson_energy"))]
    }
    abl={}
    for name,keep in groups.items():
        abl[name]=adjudicate(F[:,keep],C[:,keep],B[:,keep],REAL[keep])
    out={"programme":"STARS-INNOV1","status":"complete","metric_names":NAMES,
         "real":{"n_events":len(REAL_EVENTS),"n_lines":len(REAL_LINES),"vector":REAL.tolist()},
         "primary":primary,"decision":decision,"ablations":abl}
    print("STARS_INNOV1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
