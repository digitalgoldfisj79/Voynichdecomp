#!/usr/bin/env python3
import collections,json,math,re,urllib.request
import numpy as np
from sklearn.covariance import LedoitWolf

SEED=20261008
K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
COREURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=120).read().decode(),K0URL,"exec"),k0)
core={"__name__":"core"};exec(compile(urllib.request.urlopen(COREURL,timeout=120).read().decode(),COREURL,"exec"),core)
segment=k0["segment"];ST=k0["ST"];PIECES=k0["PIECES"];controls=k0["controls"];gen_token=k0["gen_token"]

STAR_FOLIOS=set("""f103r f103v f104r f104v f105r f105v f106r f106v f107r f107v f108r f108v f111r f111v f112r f112v f113r f113v f114r f114v""".split())
RANKS=(1,2,3,4,6,8,12,16,24,32);GEOMS=("dense","sparse","sticky","block");STRENGTHS=(.75,1.5)
CLOCKS={"C1_1to1":(0.,0.),"C2_expand":(0.,.25),"C3_delete":(.25,0.),"C4_mixed":(.15,.15)}
KMAX=33;POOLN=350

def fnum(f):return int(re.search(r"\d+",f).group())

def real_records():
    rows=[r for r in core["build_rows"]("ZLZI") if r["folio"] in STAR_FOLIOS];by=collections.defaultdict(list)
    for r in rows:by[(r["folio"],r["line_ord"])].append(r)
    out=[]
    for key in sorted(by,key=lambda x:(fnum(x[0]),0 if x[0].endswith("r") else 1,x[1])):
        line=[]
        for r in sorted(by[key],key=lambda z:z["pos"]):
            try:z=segment(r["token"])
            except Exception:continue
            if z:line.append((r["token"],z))
        if line:out.append((key[0],line))
    return out
REALREC=real_records();LINE_META=[(f,len(x)) for f,x in REALREC]
print("REAL_GEOM",len(REALREC),sum(n for _,n in LINE_META),flush=True)

def feat(tok,route):
    cls=[int(ST[p]) for p in route];v=np.zeros(29,float);v[cls[0]]=1;v[12+cls[-1]]=1;v[24+min(len(route),5)-1]=1;return v
def feature_lines(records):
    return [(fol,np.array([feat(t,z) for t,z in line],float)) for fol,line in records]
REALFL=feature_lines(REALREC)

def crossop(fl):
    P=[];F=[]
    for fol,A in fl:
        if len(A)<6:continue
        for t in range(3,len(A)-3+1):
            P.append(A[t-3:t].reshape(-1));F.append(A[t:t+3].reshape(-1))
    P=np.array(P,float);F=np.array(F,float)
    if len(P)<10:return np.zeros((87,87)),np.zeros((87,12)),np.zeros(12),np.zeros((87,12))
    P-=P.mean(0);F-=F.mean(0);C=(P.T@F)/len(P);U,s,Vt=np.linalg.svd(C,full_matrices=False)
    return C,U[:,:12],s[:12],Vt.T[:,:12]

def cv_summary(fl):
    odd=[x for x in fl if fnum(x[0])%2==1];even=[x for x in fl if fnum(x[0])%2==0]
    Co,Uo,so,Vo=crossop(odd);Ce,Ue,se,Ve=crossop(even)
    raw=[];rat=[]
    for i in range(12):
        a=float(Uo[:,i].T@Ce@Vo[:,i]);b=float(Ue[:,i].T@Co@Ve[:,i]);c=.5*(a+b)
        rr=.5*(a/max(float(so[i]),1e-8)+b/max(float(se[i]),1e-8))
        raw.append(c);rat.append(rr)
    pos=np.maximum(np.array(raw),0);den=max(float(pos.max()**2),1e-16)
    stable=float(np.sum(pos*pos)/den) if pos.max()>0 else 0.
    vec=np.array(raw+rat+[float(np.sum(pos>0)),float(np.sum(pos*pos)),stable],float)
    return vec,{"raw":raw,"ratio":rat,"positive_count":int(np.sum(pos>0)),"replicated_stable_rank":stable,
                "odd_singular":so.tolist(),"even_singular":se.tolist()}
REAL_SUM,REAL_DETAIL=cv_summary(REALFL)

def stationary(A):
    p=np.ones(len(A))/len(A)
    for _ in range(5000):
        q=p@A
        if np.max(np.abs(q-p))<1e-13:break
        p=q
    p=np.maximum(p,0);return p/p.sum()
def make_A(R,geom,seed):
    if R==1:return np.ones((1,1))
    for att in range(100):
        rng=np.random.default_rng(seed+att*991)
        if geom=="dense":A=rng.dirichlet(np.ones(R),size=R)
        elif geom=="sparse":
            A=np.zeros((R,R));d=min(3,R)
            for i in range(R):
                idx=rng.choice(R,size=d,replace=False);A[i,idx]=rng.dirichlet(np.ones(d))
            A=.97*A+.03*rng.dirichlet(np.ones(R),size=R);A/=A.sum(1,keepdims=True)
        elif geom=="sticky":
            A=.65*np.eye(R)+.35*rng.dirichlet(np.ones(R),size=R);A/=A.sum(1,keepdims=True)
        elif geom=="block":
            cut=max(1,R//2);A=np.zeros((R,R))
            for i in range(R):
                own=np.arange(0,cut) if i<cut else np.arange(cut,R);other=np.arange(cut,R) if i<cut else np.arange(0,cut)
                if len(other)==0:other=own
                A[i,own]+=0.85*rng.dirichlet(np.ones(len(own)));A[i,other]+=0.15*rng.dirichlet(np.ones(len(other)))
            A/=A.sum(1,keepdims=True)
        if np.linalg.matrix_rank(A,tol=1e-10)==R:return A
    raise RuntimeError("rank draw")
POOLS={}
for strength in STRENGTHS:
    rng=np.random.default_rng(SEED+int(strength*1000));U,Ve,Vr=controls(KMAX,2,strength,rng,"F1");pp=[]
    for x in range(KMAX):
        pool=[]
        while len(pool)<POOLN:
            z=gen_token(x,U,Ve,Vr,rng)
            if z is not None:
                route=[PIECES[int(i)] for i in z];pool.append(("".join(route),route))
        pp.append(pool)
    POOLS[strength]=pp
print("POOLS_READY",flush=True)

def render_line(A,pi,L,strength,clock,seed):
    rng=np.random.default_rng(seed);x=int(rng.choice(len(A),p=pi));p0,p2=CLOCKS[clock];surf=[];used=False
    while len(surf)<L:
        if used:x=int(rng.choice(len(A),p=A[x]))
        used=True;u=rng.random();nout=0 if u<p0 else (2 if u>1-p2 else 1)
        for _ in range(nout):
            if len(surf)>=L:break
            pool=POOLS[strength][x];surf.append(pool[int(rng.integers(len(pool)))])
    return surf
def simulate_case(rep,R,geom,strength,clock):
    base=SEED+rep*10_000_000+R*100_000+GEOMS.index(geom)*10_000+int(strength*1000)*10+list(CLOCKS).index(clock)
    A=make_A(R,geom,base);pi=stationary(A);rec=[]
    for j,(fol,L) in enumerate(LINE_META):
        rec.append((fol,render_line(A,pi,L,strength,clock,base+1000+j*173)))
    vec,det=cv_summary(feature_lines(rec))
    return {"rep":rep,"R":R,"geom":geom,"strength":strength,"clock":clock,"summary":vec.tolist(),"raw":det["raw"],"ratio":det["ratio"]}
def run_rep(rep):
    out=[]
    for R in RANKS:
      for g in GEOMS:
       for st in STRENGTHS:
        for c in CLOCKS:out.append(simulate_case(rep,R,g,st,c))
    return rep,out
SIM=[]
if __name__=="__main__":
    import multiprocessing as mp
    with mp.get_context("fork").Pool(4) as pool:
        for rep,out in pool.imap_unordered(run_rep,range(4)):
            SIM.extend(out);print("SIM_DONE",rep,len(out),flush=True)

MODELS={}
for R in RANKS:
    tr=[x for x in SIM if x["R"]==R and x["rep"] in (0,1)];X=np.array([x["summary"] for x in tr]);lw=LedoitWolf().fit(X);P=lw.precision_
    def dd(v):
        D=X-v;return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",D,P,D),0))
    cal=[x for x in SIM if x["R"]==R and x["rep"]==2];q99=float(np.quantile([np.min(dd(np.array(x["summary"]))) for x in cal],.99))
    bl=[x for x in SIM if x["R"]==R and x["rep"]==3];accept=float(np.mean([np.min(dd(np.array(x["summary"])))<=q99 for x in bl]))
    rd=float(np.min(dd(REAL_SUM)));MODELS[R]={"X":X,"P":P,"q99":q99,"blind_accept":accept,"eligible":bool(accept>=.90),"real_distance":rd,"real_reject":bool(accept>=.90 and rd>q99)}
def dist(R,v):
    M=MODELS[R];D=M["X"]-v;return float(np.min(np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",D,M["P"],D),0))))
pairs=hit=0;errs=[]
for t in [x for x in SIM if x["rep"]==3]:
    v=np.array(t["summary"]);true=t["R"];pred=min(RANKS,key=lambda R:dist(R,v)/max(MODELS[R]["q99"],1e-9));errs.append(abs(math.log2(pred/true)))
    for R in RANKS:
        if R<=true/2 and MODELS[R]["eligible"]:
            pairs+=1;hit+=int(dist(R,v)>MODELS[R]["q99"])
power=float(hit/pairs) if pairs else 0.;mederr=float(np.median(errs));GLOBAL=bool(power>=.70 and mederr<=1.0)
rej=[R for R in RANKS if MODELS[R]["real_reject"]];cons=[]
for R in RANKS:
    if R in rej:cons.append(R)
    else:break
hard=max(cons) if cons and GLOBAL else None
compat=[R for R in RANKS if MODELS[R]["eligible"] and not MODELS[R]["real_reject"]]

# Secondary R=1 replicated-component thresholds.
r1=[x for x in SIM if x["R"]==1 and x["rep"]==3];thr=[];repl=[]
for i in range(12):
    q=float(np.quantile([x["raw"][i] for x in r1],.99));thr.append(q);repl.append(bool(REAL_DETAIL["raw"][i]>q))
consec=0
for z in repl:
    if z:consec+=1
    else:break

OUT={"programme":"STARS-PSR2","status":"complete","seed":SEED,
     "real":{"N_tokens":sum(n for _,n in LINE_META),"summary":REAL_SUM.tolist(),"detail":REAL_DETAIL},
     "models":{str(R):{k:v for k,v in M.items() if k not in ("X","P")} for R,M in MODELS.items()},
     "power_reject_half_rank":power,"median_abs_log2_rank_error":mederr,"global_power":GLOBAL,
     "rejected_ranks":rej,"hard_exclude_through":hard,"compatible_ranks":compat,
     "r1_component_q99":thr,"real_component_exceeds_r1_q99":repl,"consecutive_replicated_components":consec}
print("STARS_PSR2_JSON="+json.dumps(OUT,separators=(",",":")),flush=True)
