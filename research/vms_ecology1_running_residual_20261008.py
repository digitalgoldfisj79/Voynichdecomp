#!/usr/bin/env python3
# VMS-ECOLOGY-1 — preregistered 2026-10-08.
import collections,json,math,os,re,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
m={"__name__":"latent"}
exec(compile(urllib.request.urlopen(LAT_URL,timeout=120).read().decode(),LAT_URL,"exec"),m)

rows=m["rows"]; LINES=m["LINES"]; fit_struct=m["fit_struct"]; para_id=m["para_id"]; K=m["K"]; ALPHA=m["ALPHA"]; folds=m["folds"]
SEEDS=list(range(202610092000,202610092500))
LAGS=(1,2,3,5,10); L2=10.0
COHORTS=("HERBAL_A","HERBAL_B","BALNEO_FULL","RECIPES_FULL")

def fnum(f):
    z=re.match(r"f(\d+)",str(f)); return int(z.group(1)) if z else -1
def cohort(f):
    n=fnum(f)
    if 1<=n<=25:return "HERBAL_A"
    if 26<=n<=56:return "HERBAL_B"
    if 75<=n<=84:return "BALNEO_FULL"
    if 103<=n<=116:return "RECIPES_FULL"
    return None

# Five fold-excluded compact SELECT models.
MODELS={}
for ff in range(5):
    tr=[l for l in LINES if l["fold"]!=ff]
    MODELS[ff]=fit_struct(tr)

def prob(model,section,piece,pagev,parav,recent):
    M,G,th=model; ng=sum(G.values()); cc=M.get((section,piece),{}); n=sum(cc.values())
    b=np.array([(cc.get(c,0)+ALPHA*((G[c]+.5)/(ng+.5*K)))/(n+ALPHA) for c in range(K)],float)
    sc=np.log(np.maximum(b,1e-15))+np.log1p(pagev)*th[:K]+np.log1p(parav)*th[K:2*K]+recent*th[-1]
    sc-=sc.max(); q=np.exp(sc); q/=q.sum(); return q

def generate(observed=False,seed=None):
    rng=np.random.default_rng(seed if seed is not None else 12345)
    page=collections.defaultdict(lambda:np.zeros(K,float))
    para=collections.defaultdict(lambda:np.zeros(K,float))
    hist=collections.defaultdict(list)
    by={c:collections.OrderedDict() for c in COHORTS}
    for r in rows:
        fol=r["folio"]; ln=int(r["line"]); lk=(fol,ln); pk=fol; pid=para_id(pk,ln); pq=(pk,pid)
        yobs=int(r["start"]); pos=int(r["pos"]); ff=int(r["fold"]); co=cohort(fol)
        if pos==0:
            y=yobs
        else:
            prev_piece=hist[lk][-1][1]
            rc=np.zeros(K,float)
            for yy,pp in hist[lk][-6:]: rc[int(yy)]+=1
            p=prob(MODELS[ff],r["section"],prev_piece,page[pk],para[pq],rc)
            prev=int(hist[lk][-1][0])
            y=yobs if observed else int(rng.choice(K,p=p))
            if co is not None:
                by[co].setdefault(lk,[]).append({"p":p,"y":y,"prev":prev,"folio":fol,"line":lk,"fold":ff})
        page[pk][y]+=1; para[pq][y]+=1; hist[lk].append((y,r["final_piece"]))
    return {c:list(by[c].values()) for c in COHORTS}

def role(i,n):
    if i==0:return "FIRST"
    if i==1:return "SECOND"
    if i==n-1:return "FINAL"
    if i==n-2:return "PENULT"
    rel=(i-2)/max(1,n-5)
    return "EARLY" if rel<1/3 else ("MIDDLE" if rel<2/3 else "LATE")
def lbin(n): return "L7" if n<=7 else ("L10" if n<=10 else ("L14" if n<=14 else "L15P"))

def fit_tilt(es):
    if not es:return np.zeros(K,float)
    P=np.stack([e["p"] for e in es]); Y=np.array([e["y"] for e in es],int); b=np.zeros(K,float); I=np.eye(K)
    def obj(bb):
        sc=np.log(np.maximum(P,1e-15))+bb[None,:]; mx=sc.max(1,keepdims=True)
        z=mx[:,0]+np.log(np.exp(sc-mx).sum(1))
        return -float(np.sum(sc[np.arange(len(Y)),Y]-z))+.5*L2*float(np.dot(bb,bb))
    old=obj(b)
    for _ in range(20):
        sc=np.log(np.maximum(P,1e-15))+b[None,:]; mx=sc.max(1,keepdims=True)
        Q=np.exp(sc-mx); Q/=Q.sum(1,keepdims=True)
        D=Q.copy(); D[np.arange(len(Y)),Y]-=1.
        g=D.sum(0)+L2*b; g-=g.mean()
        if np.max(np.abs(g))<1e-9:break
        H=L2*I
        for q in Q:H+=np.diag(q)-np.outer(q,q)
        step=np.linalg.solve(H+I*1e-10,g); step-=step.mean(); t=1.
        while t>1e-6:
            cc=b-t*step; cc-=cc.mean(); nv=obj(cc)
            if nv<=old+1e-12: b,old=cc,nv; break
            t*=.5
        if t<=1e-6:break
    return b

def fit_pos(lines,parity):
    allv=[]; byrole=collections.defaultdict(list); bycell=collections.defaultdict(list)
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        n=len(seq)
        for i,e in enumerate(seq):
            ro=role(i,n); allv.append(e); byrole[ro].append(e); bycell[(ro,lbin(n))].append(e)
    gb=fit_tilt(allv)
    rb={ro:fit_tilt(es) for ro,es in byrole.items() if len(es)>=40}
    cb={k:fit_tilt(es) for k,es in bycell.items() if len(es)>=20}
    return gb,rb,cb

def tilt_p(p,b):
    sc=np.log(np.maximum(p,1e-15))+b; sc-=sc.max(); q=np.exp(sc); return q/q.sum()

def position_adjust(lines):
    mods={0:fit_pos(lines,0),1:fit_pos(lines,1)}; out=[]
    for seq in lines:
        if not seq:continue
        tr=1-fnum(seq[0]["folio"])%2; gb,rb,cb=mods[tr]; n=len(seq); zz=[]
        for i,e in enumerate(seq):
            ro=role(i,n); b=cb.get((ro,lbin(n)),rb.get(ro,gb)); x=dict(e); x["p"]=tilt_p(e["p"],b); zz.append(x)
        out.append(zz)
    return out

def entropy(p):
    q=np.asarray(p,float); q=q[q>0]; return float(-(q*np.log2(q)).sum())
def safe_corr(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3:return 0.
    x=x-x.mean(); y=y-y.mean(); d=math.sqrt(float(np.dot(x,x)*np.dot(y,y)))
    return float(np.dot(x,y)/d) if d>1e-12 else 0.

def metric_vector(lines):
    events=[e for z in lines for e in z]; vals=[]; names=[]
    for lag in LAGS:
        z=[]
        for seq in lines:
            for i in range(lag,len(seq)):
                e=seq[i]; prior=seq[i-lag]["y"]; z.append((1. if e["y"]==prior else 0.)-float(e["p"][prior]))
        vals.append(float(np.mean(z)) if z else 0.); names.append(f"repeat_resid_{lag}")
    S=np.zeros((K,K)); N=np.zeros(K)
    for e in events:
        a=e["prev"]; v=-e["p"].copy(); v[e["y"]]+=1.; S[a]+=v; N[a]+=1
    R=np.zeros_like(S)
    for a in range(K):
        if N[a]>0:R[a]=S[a]/N[a]
    vals.append(float(np.sqrt(np.mean(R*R)))); names.append("transition_resid_norm")
    sb=[]; alls=[]
    for seq in lines:
        z=[]
        for e in seq:
            s=-math.log2(max(float(e["p"][e["y"]]),1e-15))-entropy(e["p"]); z.append(s); alls.append(s)
        sb.append(z)
    for lag in LAGS:
        a=[]; b=[]
        for z in sb:
            if len(z)>lag:a.extend(z[:-lag]); b.extend(z[lag:])
        vals.append(safe_corr(a,b)); names.append(f"surprise_ac_{lag}")
    sv=np.linalg.svd(R,compute_uv=False); en=float(np.sum(sv*sv))
    vals.append(float(sv[0]**2/en) if en>1e-15 else 0.); names.append("sv1_energy")
    vals.append(float(np.sum(sv[:2]**2)/en) if en>1e-15 else 0.); names.append("sv12_energy")
    vals.append(float(np.mean(alls)) if alls else 0.); names.append("mean_excess_surprise")
    pe=[]
    for e in events:
        p=e["p"]; r=-p.copy(); r[e["y"]]+=1.; r/=np.sqrt(np.maximum(p*(1-p),1e-9)); pe.append(float(np.mean(r*r)))
    vals.append(float(np.mean(pe)) if pe else 0.); names.append("pearson_energy")
    return np.array(vals,float),names,len(events)

def analyze(observed=False,seed=None):
    D=generate(observed,seed); out={}
    for c in COHORTS:
        pa=position_adjust(D[c]); v,names,n=metric_vector(pa)
        out[c]={"vector":v,"names":names,"n_events":n,"n_lines":len(pa),"lines":pa}
    return out

REAL=analyze(True,None)
NAMES=REAL["HERBAL_A"]["names"]
print("REAL_COUNTS",json.dumps({c:{"n_events":REAL[c]["n_events"],"n_lines":REAL[c]["n_lines"]} for c in COHORTS},separators=(",",":")),flush=True)

def task(seed):
    x=analyze(False,seed)
    return seed,{c:x[c]["vector"].tolist() for c in COHORTS}

def setup(A):
    mu=A.mean(0); S=np.cov(A,rowvar=False); C=.75*S+.25*np.diag(np.diag(S))+np.eye(A.shape[1])*1e-9
    return mu,np.linalg.pinv(C)
def d2(x,mu,iv):
    z=x-mu; return float(z@iv@z)

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(task,SEEDS,chunksize=2))
    mp={s:v for s,v in rr}
    details={}; calZ={}; blindZ={}; realZ={}
    for c in COHORTS:
        V=np.vstack([np.asarray(mp[s][c],float) for s in SEEDS])
        F,SCALE,CAL,BLIND=V[:200],V[200:300],V[300:400],V[400:500]
        mu,iv=setup(F)
        sd=np.array([d2(x,mu,iv) for x in SCALE]); cm=np.array([d2(x,mu,iv) for x in CAL]); bd=np.array([d2(x,mu,iv) for x in BLIND])
        sm=float(sd.mean()); ss=max(float(sd.std(ddof=1)),1e-12)
        cz=(cm-sm)/ss; bz=(bd-sm)/ss; rz=(d2(REAL[c]["vector"],mu,iv)-sm)/ss
        calZ[c]=cz; blindZ[c]=bz; realZ[c]=float(rz)
        details[c]={"real_D2":d2(REAL[c]["vector"],mu,iv),"real_Z":float(rz),
                    "own_q99_Z":float(np.quantile(cz,.99)),
                    "n_events":REAL[c]["n_events"],"n_lines":REAL[c]["n_lines"],
                    "real_vector":REAL[c]["vector"].tolist()}
        if c=="RECIPES_FULL":
            sens_lines=[z for z in REAL[c]["lines"] if z and z[0]["folio"]!="f115r"]
            sv,_,sn=metric_vector(sens_lines); sz=(d2(sv,mu,iv)-sm)/ss
            details[c]["sensitivity_exclude_f115r"]={"Z":float(sz),"D2":d2(sv,mu,iv),"n_events":sn,"vector":sv.tolist()}
    cmax=np.max(np.vstack([calZ[c] for c in COHORTS]),axis=0)
    bmax=np.max(np.vstack([blindZ[c] for c in COHORTS]),axis=0)
    rmax=max(realZ.values()); fq=float(np.quantile(cmax,.99))
    blind=float(np.mean(bmax<=fq)); p=float((1+np.sum(np.r_[cmax,bmax]>=rmax))/(len(cmax)+len(bmax)+1))
    globalpass=bool(blind>=.90 and rmax>fq and p<=.01)
    resolved={}
    for c in COHORTS:
        resolved[c]=bool(globalpass and realZ[c]>details[c]["own_q99_Z"])
        details[c]["resolved"]=resolved[c]
    if not globalpass: decision="NO_RUNNING_TEXT_RESIDUAL_RESOLVED"
    else:
        rr=[c for c in COHORTS if resolved[c]]
        decision="RESIDUAL_ORDERED_INNOVATION_PRESENT:"+(",".join(rr) if rr else "GLOBAL_ONLY")
    out={"programme":"VMS-ECOLOGY-1","status":"complete","metric_names":NAMES,
         "familywise":{"real_max_Z":rmax,"q99_max_Z":fq,"blind_acceptance":blind,"add_one_p":p,"global_pass":globalpass,"decision":decision},
         "cohorts":details,
         "fold_models":{"n":5,"rule":"exclude own physical bifolium fold"}}
    print("VMS_ECOLOGY1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
