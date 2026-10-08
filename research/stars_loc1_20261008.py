#!/usr/bin/env python3
# STARS-LOC1 — preregistered 2026-10-08.
import collections,json,math,os,re,urllib.request
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/7ea7f89ac447571e8340da0e75c2ae3e8fd73689/research/latent_line_state_test_20261004.py"
OCC_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/research/hf_emergent_occupancy_fold.py"
m={"__name__":"latent_line_state_module"}
src=urllib.request.urlopen(LAT_URL,timeout=120).read().decode();exec(compile(src,LAT_URL,"exec"),m)
occ={"__name__":"occ"};osrc=urllib.request.urlopen(OCC_URL,timeout=120).read().decode();exec(compile(osrc,OCC_URL,"exec"),occ)

rows=m["rows"];LINES=m["LINES"];fit_struct=m["fit_struct"];para_id=m["para_id"];K=m["K"];ALPHA=m["ALPHA"]
TR=[l for l in LINES if l["fold"] in (2,3,4)]
M,G,TH=fit_struct(TR);NG=sum(G.values())
STAR_FOL=set(f"f{n}{s}" for n in list(range(103,109))+list(range(111,115)) for s in ("r","v"))
PAIRS=occ["PAIRS"];MATE={a:b for a,b in PAIRS};MATE.update({b:a for a,b in PAIRS})
CAL_SEEDS=list(range(202610084000,202610084200));BLIND_SEEDS=list(range(202610084200,202610084300))
L2=10.0;SCALES=("L","P","F","B")

def fnum(f):return int(re.search(r"\d+",str(f)).group())

def base_prob(section,piece,pagev,parav,recent):
    cc=M.get((section,piece),{});n=sum(cc.values())
    b=np.array([(cc.get(c,0)+ALPHA*((G[c]+.5)/(NG+.5*K)))/(n+ALPHA) for c in range(K)],float)
    sc=np.log(np.maximum(b,1e-15))+np.log1p(pagev)*TH[:K]+np.log1p(parav)*TH[K:2*K]+recent*TH[-1]
    sc-=sc.max();p=np.exp(sc);p/=p.sum();return p

def generate(observed=False,seed=None):
    rng=np.random.default_rng(seed if seed is not None else 12345)
    page=collections.defaultdict(lambda:np.zeros(K,float));para=collections.defaultdict(lambda:np.zeros(K,float));hist=collections.defaultdict(list)
    events=[]
    for order,r in enumerate(rows):
        fol=r["folio"];ln=int(r["line"]);lk=(fol,ln);pk=fol;pid=para_id(pk,ln);q=(pk,pid);yobs=int(r["start"]);pos=int(r["pos"])
        if pos==0:y=yobs
        else:
            prev_piece=hist[lk][-1][1];rc=np.zeros(K,float)
            for yy,pp in hist[lk][-6:]:rc[int(yy)]+=1
            p=base_prob(r["section"],prev_piece,page[pk],para[q],rc)
            y=yobs if observed else int(rng.choice(K,p=p))
            if fol in STAR_FOL:
                events.append({"p":p,"y":y,"folio":fol,"num":fnum(fol),"line":ln,"para":int(pid),
                               "bif":r["bifolium"],"order":order})
        page[pk][y]+=1;para[q][y]+=1;hist[lk].append((y,r["final_piece"]))
    return events

def fit_tilt(donor):
    P=np.stack([e["p"] for e in donor]);Y=np.array([e["y"] for e in donor],int)
    def fg(b):
        bb=b-b.mean();sc=np.log(np.maximum(P,1e-15))+bb[None,:];mx=sc.max(1,keepdims=True)
        Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True)
        loss=-float(np.sum(np.log(np.maximum(Q[np.arange(len(Y)),Y],1e-300))))+.5*L2*float(np.dot(bb,bb))
        D=Q.copy();D[np.arange(len(Y)),Y]-=1.;g=D.sum(0)+L2*bb;g-=g.mean()
        return loss,g
    z=minimize(lambda b:fg(b),np.zeros(K),jac=True,method="L-BFGS-B",options={"maxiter":80,"ftol":1e-10})
    return z.x-z.x.mean()

def score_target(target,b):
    base=tilt=0.
    for e in target:
        p=e["p"];y=e["y"];base+=-math.log2(max(float(p[y]),1e-300))
        sc=np.log(np.maximum(p,1e-15))+b;sc-=sc.max();q=np.exp(sc);q/=q.sum()
        tilt+=-math.log2(max(float(q[y]),1e-300))
    return base,tilt,len(target)

def evaluate_units(units):
    B=T=N=0.;ug=[]
    for donor,target in units:
        if not donor or not target:continue
        b=fit_tilt(donor);bb,tt,nn=score_target(target,b);B+=bb;T+=tt;N+=nn;ug.append((bb-tt)/nn)
    return {"gain":float((B-T)/N) if N else 0.,"n_target":N,"n_units":len(ug),
            "units_positive":int(sum(x>0 for x in ug)),"unit_fraction_positive":float(np.mean(np.array(ug)>0)) if ug else None}

def units_line(ev):
    g=collections.OrderedDict()
    for e in ev:g.setdefault((e["folio"],e["line"]),[]).append(e)
    return [(z[:3],z[3:]) for z in g.values() if len(z)>=7]

def units_para(ev):
    g=collections.OrderedDict()
    for e in ev:g.setdefault((e["folio"],e["para"]),[]).append(e)
    out=[]
    for z in g.values():
        lines=len(set(e["line"] for e in z))
        if len(z)>=12 and lines>=2:
            q=len(z)//2;out.append((z[:q],z[q:]))
    return out

def units_folio(ev):
    g=collections.OrderedDict()
    for e in ev:g.setdefault(e["folio"],collections.OrderedDict()).setdefault(e["line"],[]).append(e)
    out=[]
    for fol,ld in g.items():
        lines=list(ld.values())
        n=sum(map(len,lines))
        if n>=40 and len(lines)>=4:
            q=len(lines)//2;don=[e for z in lines[:q] for e in z];tar=[e for z in lines[q:] for e in z]
            out.append((don,tar))
    return out

def units_bif(ev):
    bynum=collections.defaultdict(list)
    for e in ev:bynum[e["num"]].append(e)
    out=[]
    for n,tar in sorted(bynum.items()):
        mate=MATE.get(n)
        if mate in bynum:
            out.append((bynum[mate],tar))
    return out

def statistic(observed=False,seed=None):
    ev=generate(observed,seed)
    res={"L":evaluate_units(units_line(ev)),"P":evaluate_units(units_para(ev)),
         "F":evaluate_units(units_folio(ev)),"B":evaluate_units(units_bif(ev))}
    return {"n_events":len(ev),"scales":res}

def task(seed):
    x=statistic(False,seed);return seed,[x["scales"][s]["gain"] for s in SCALES]

REAL=statistic(True,None)
print("REAL",json.dumps(REAL,separators=(",",":")),flush=True)

if __name__=="__main__":
    seeds=CAL_SEEDS+BLIND_SEEDS
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(task,seeds,chunksize=2))
    mp={s:np.array(v,float) for s,v in rr}
    C=np.vstack([mp[s] for s in CAL_SEEDS]);B=np.vstack([mp[s] for s in BLIND_SEEDS])
    mu=C.mean(0);sd=C.std(0,ddof=1);sd=np.maximum(sd,1e-12)
    Cz=(C-mu)/sd;Bz=(B-mu)/sd;rv=np.array([REAL["scales"][s]["gain"] for s in SCALES]);rz=(rv-mu)/sd
    cmax=Cz.max(1);bmax=Bz.max(1);rmax=float(rz.max());q99=float(np.quantile(cmax,.99));accept=float(np.mean(bmax<=q99))
    pooled=np.r_[cmax,bmax];p=float((1+np.sum(pooled>=rmax))/(len(pooled)+1))
    perq=np.quantile(Cz,.99,axis=0)
    if accept<.90:decision="THE_METRIC_DOES_NOT_RESOLVE_CALIBRATION_FAIL"
    elif rmax<=q99:decision="NO_REGIME_SCALE_PERSISTENCE_RESOLVED"
    elif p>.01:decision="THE_METRIC_DOES_NOT_RESOLVE_PVALUE_FAIL"
    else:decision="RESIDUAL_REGIME_STATE_LOCALIZED_AT_ONE_OR_MORE_SCALES"
    resolved={s:bool(rz[i]>perq[i]) for i,s in enumerate(SCALES)} if decision=="RESIDUAL_REGIME_STATE_LOCALIZED_AT_ONE_OR_MORE_SCALES" else {s:False for s in SCALES}
    out={"programme":"STARS-LOC1","status":"complete","real":REAL,
         "calibration":{"n":len(C),"mean_gain":{s:float(mu[i]) for i,s in enumerate(SCALES)},
                        "sd_gain":{s:float(sd[i]) for i,s in enumerate(SCALES)},
                        "maxZ_q99":q99,"per_scale_z_q99":{s:float(perq[i]) for i,s in enumerate(SCALES)}},
         "blind":{"n":len(B),"maxZ_self_acceptance":accept},
         "real_Z":{s:float(rz[i]) for i,s in enumerate(SCALES)},"real_maxZ":rmax,
         "add_one_p":p,"decision":decision,"resolved_scales":resolved}
    print("STARS_LOC1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
