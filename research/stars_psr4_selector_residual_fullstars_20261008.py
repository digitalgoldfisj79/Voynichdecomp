#!/usr/bin/env python3
# STARS-PSR4 full-Stars reconstruction — preregistered 2026-10-08.
import collections,json,math,os,re,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/7ea7f89ac447571e8340da0e75c2ae3e8fd73689/research/latent_line_state_test_20261004.py"
m={"__name__":"latent_line_state_module"}
src=urllib.request.urlopen(LAT_URL,timeout=120).read().decode()
exec(compile(src,LAT_URL,"exec"),m)

rows=m["rows"]; LINES=m["LINES"]; fit_struct=m["fit_struct"]; para_id=m["para_id"]; K=m["K"]; ALPHA=m["ALPHA"]
TR=[l for l in LINES if l["fold"] in (2,3,4)]
MODEL=fit_struct(TR)
M,G,TH=MODEL; NG=sum(G.values())
STAR_FOL=set(f"f{n}{s}" for n in list(range(103,109))+list(range(111,115)) for s in ("r","v"))
CAL_SEEDS=list(range(202610081000,202610081400))
BLIND_SEEDS=list(range(202610081400,202610081500))

def fnum(f):
    return int(re.search(r"\d+",str(f)).group())

def base_prob(section,piece,pagev,parav,recent):
    cc=M.get((section,piece),{});n=sum(cc.values())
    b=np.array([(cc.get(c,0)+ALPHA*((G[c]+.5)/(NG+.5*K)))/(n+ALPHA) for c in range(K)],float)
    sc=np.log(np.maximum(b,1e-15))+np.log1p(pagev)*TH[:K]+np.log1p(parav)*TH[K:2*K]+recent*TH[-1]
    sc-=sc.max();p=np.exp(sc);p/=p.sum();return p

def make_lines(observed=False,seed=None):
    rng=np.random.default_rng(seed if seed is not None else 12345)
    page=collections.defaultdict(lambda:np.zeros(K,float))
    para=collections.defaultdict(lambda:np.zeros(K,float))
    linehist=collections.defaultdict(list)
    by=collections.OrderedDict()
    ll=0.;nn=0
    for r in rows:
        fol=r["folio"];ln=r["line"];lk=(fol,ln);pk=fol;q=(pk,para_id(pk,ln))
        yobs=int(r["start"]); pos=int(r["pos"])
        if pos==0:
            y=yobs
        else:
            prev_piece=linehist[lk][-1][1]
            rc=np.zeros(K,float)
            for yy,pp in linehist[lk][-6:]:rc[int(yy)]+=1
            p=base_prob(r["section"],prev_piece,page[pk],para[q],rc)
            y=yobs if observed else int(rng.choice(K,p=p))
            if observed:
                ll += -math.log2(max(float(p[y]),1e-300)); nn+=1
            if fol in STAR_FOL:
                v=-p.copy();v[y]+=1.
                by.setdefault(lk,[]).append(v)
        page[pk][y]+=1;para[q][y]+=1
        # final piece is an observed FORM nuisance in the frozen selector.
        linehist[lk].append((y,r["final_piece"]))
    out=[]
    for (fol,ln),vs in by.items():
        if len(vs)>=7:out.append((fol,np.stack(vs)))
    return out,(ll/nn if nn else None),nn

def role(i,n):
    if i==0:return "START"
    if i==1:return "SECOND"
    if i==n-1:return "FINAL"
    if i==n-2:return "PENULT"
    rel=(i-2)/max(1,n-5)
    return "EARLY" if rel<1/3 else ("MIDDLE" if rel<2/3 else "LATE")
def lbin(n):
    return "L7" if n<=7 else ("L10" if n<=10 else ("L14" if n<=14 else "L15P"))

def fit_means(lines,parity):
    cell=collections.defaultdict(list);rr=collections.defaultdict(list);allv=[]
    for fol,A in lines:
        if fnum(fol)%2!=parity:continue
        n=len(A)
        for i,v in enumerate(A):
            ro=role(i,n);cell[(ro,lbin(n))].append(v);rr[ro].append(v);allv.append(v)
    return ({k:np.mean(v,0) for k,v in cell.items()},
            {k:np.mean(v,0) for k,v in rr.items()},
            np.mean(allv,0))

def residualize(lines):
    mods={0:fit_means(lines,0),1:fit_means(lines,1)};out=[]
    for fol,A in lines:
        cell,rr,g=mods[1-fnum(fol)%2];n=len(A);B=[]
        for i,v in enumerate(A):
            ro=role(i,n);mu=cell.get((ro,lbin(n)),rr.get(ro,g));B.append(v-mu)
        out.append((fol,np.asarray(B)))
    return out

def crossop(lines):
    P=[];F=[]
    for fol,A in lines:
        for t in range(3,len(A)-2):
            P.append(A[t-3:t].reshape(-1));F.append(A[t:t+3].reshape(-1))
    P=np.asarray(P,float);F=np.asarray(F,float)
    P-=P.mean(0);F-=F.mean(0);C=P.T@F/len(P)
    U,s,Vt=np.linalg.svd(C,full_matrices=False)
    return C,U,s,Vt.T

def cv_scores(lines):
    odd=[x for x in lines if fnum(x[0])%2==1];even=[x for x in lines if fnum(x[0])%2==0]
    Co,Uo,so,Vo=crossop(odd);Ce,Ue,se,Ve=crossop(even)
    z=[]
    for i in range(min(6,len(so),len(se))):
        z.append(.5*(float(Uo[:,i].T@Ce@Vo[:,i])+float(Ue[:,i].T@Co@Ve[:,i])))
    return z

def statistic(observed=False,seed=None):
    lines,bits,n=make_lines(observed,seed)
    res=residualize(lines)
    return {"scores":cv_scores(res),"lines":len(lines),"events":sum(len(a) for _,a in lines),
            "global_observed_selector_bits":bits,"global_scored_events":n}

def task(seed):
    x=statistic(False,seed);return seed,x["scores"]

REAL=statistic(True,None)
print("REAL",json.dumps(REAL,separators=(",",":")),flush=True)

if __name__=="__main__":
    seeds=CAL_SEEDS+BLIND_SEEDS
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rowsout=list(ex.map(task,seeds,chunksize=2))
    mp={s:z for s,z in rowsout}
    cal=np.array([mp[s][0] for s in CAL_SEEDS],float)
    blind=np.array([mp[s][0] for s in BLIND_SEEDS],float)
    q99=float(np.quantile(cal,.99));blind_accept=float(np.mean(blind<=q99))
    real=float(REAL["scores"][0]);pooled=np.r_[cal,blind]
    p_add=float((1+np.sum(pooled>=real))/(len(pooled)+1))
    z=float((real-cal.mean())/cal.std(ddof=1))
    if blind_accept<.90:decision="THE_METRIC_DOES_NOT_RESOLVE_CALIBRATION_FAIL"
    elif real<=q99:decision="KNOWN_COMPACT_SELECT_MECHANICS_ABSORB_LEADING_DIRECTION"
    elif p_add>.01:decision="THE_METRIC_DOES_NOT_RESOLVE_PVALUE_FAIL"
    else:decision="SURVIVES_KNOWN_SELECT_MECHANICS"
    out={"programme":"STARS-PSR4","status":"complete","real":REAL,
         "calibration":{"n":len(cal),"q99_c1":q99,"mean":float(cal.mean()),"sd":float(cal.std(ddof=1))},
         "blind":{"n":len(blind),"self_acceptance":blind_accept,"mean":float(blind.mean()),"sd":float(blind.std(ddof=1))},
         "real_minus_cal_mean_over_sd":z,"pooled_add_one_p":p_add,"decision":decision}
    print("STARS_PSR4_JSON="+json.dumps(out,separators=(",",":")),flush=True)
