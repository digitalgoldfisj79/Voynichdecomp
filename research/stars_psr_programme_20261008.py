#!/usr/bin/env python3
# STARS-PSR programme, preregistered 2026-10-08.
import collections, json, math, pickle, re, urllib.request, warnings
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score

warnings.filterwarnings("ignore")
SEED=20261008
K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
COREURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
k0={"__name__":"k0"}; exec(compile(urllib.request.urlopen(K0URL,timeout=120).read().decode(),K0URL,"exec"),k0)
core={"__name__":"core"}; exec(compile(urllib.request.urlopen(COREURL,timeout=120).read().decode(),COREURL,"exec"),core)
segment=k0["segment"]; ST=k0["ST"]; PIECES=k0["PIECES"]; controls=k0["controls"]; gen_token=k0["gen_token"]

STARS=set("""f103r f103v f104r f104v f105r f105v f106r f106v f107r f107v f108r f108v f111r f111v f112r f112v f113r f113v f114r f114v""".split())
RGRID=(1,2,3,4,6,8,12,16,24,32)
GEOMS=("dense","sparse","sticky","block")
STRENGTHS=(.75,1.5)
CLOCKS={"C1_1to1":(0.,0.),"C2_expand":(0.,.25),"C3_delete":(.25,0.),"C4_mixed":(.15,.15)}
KMAX=32; POOLN=320

def fnum(f):
    m=re.search(r"(\d+)",f); return int(m.group(1)) if m else -1

def route_feat(route):
    v=np.zeros(29,dtype=np.float32)
    v[int(ST[route[0]])]=1.
    v[12+int(ST[route[-1]])]=1.
    v[24+min(len(route),5)-1]=1.
    return v

def real_lines():
    rows=[r for r in core["build_rows"]("ZLZI") if r["folio"] in STARS]
    by=collections.defaultdict(list)
    for r in rows: by[(r["folio"],r["line_ord"])].append(r)
    out=[]
    for key in sorted(by,key=lambda x:(fnum(x[0]),x[0],x[1])):
        feats=[]
        for r in sorted(by[key],key=lambda z:z["pos"]):
            try: z=segment(r["token"])
            except Exception: continue
            if z: feats.append(route_feat(z))
        if len(feats)>=7: out.append({"folio":key[0],"feat":np.stack(feats),"L":len(feats)})
    return out

REAL_LINES=real_lines()
REAL_N=sum(x["L"] for x in REAL_LINES)
ODD=[x for x in REAL_LINES if fnum(x["folio"])%2==1]
EVEN=[x for x in REAL_LINES if fnum(x["folio"])%2==0]
print("REAL",{"lines":len(REAL_LINES),"N":REAL_N,"odd_lines":len(ODD),"even_lines":len(EVEN)},flush=True)

POOLS={}
for strength in STRENGTHS:
    rng=np.random.default_rng(SEED+int(strength*1000))
    U,Ve,Vr=controls(KMAX,2,strength,rng,"F1")
    pp=[]
    for x in range(KMAX):
        pool=[]
        while len(pool)<POOLN:
            z=gen_token(x,U,Ve,Vr,rng)
            if z is None: continue
            route=[PIECES[int(i)] for i in z]
            pool.append(route_feat(route))
        pp.append(np.stack(pool))
    POOLS[strength]=pp
print("POOLS_READY",flush=True)

def stationary(A):
    p=np.ones(len(A))/len(A)
    for _ in range(10000):
        q=p@A
        if np.max(np.abs(q-p))<1e-13: break
        p=q
    p=np.maximum(p,0); p/=p.sum(); return p

def make_A(R,geom,seed):
    if R==1: return np.ones((1,1),float)
    rng=np.random.default_rng(seed)
    for _ in range(100):
        if geom=="dense":
            W=rng.gamma(1.5,1.,size=(R,R))
        elif geom=="sparse":
            W=np.full((R,R),1e-3)
            d=min(3,R)
            for i in range(R):
                js=rng.choice(R,size=d,replace=False); W[i,js]+=rng.gamma(1.5,1.,size=d)
        elif geom=="sticky":
            W=rng.gamma(1.,1.,size=(R,R)); W/=W.sum(1,keepdims=True)
            A=.25*W+.75*np.eye(R)
            if np.linalg.matrix_rank(A,tol=1e-10)==R: return A/A.sum(1,keepdims=True)
            continue
        elif geom=="block":
            groups=np.arange(R)%2
            mult=np.where(groups[:,None]==groups[None,:],5.,.35)
            W=rng.gamma(1.5,1.,size=(R,R))*mult
        A=W/W.sum(1,keepdims=True)
        if np.linalg.matrix_rank(A,tol=1e-10)==R: return A
    raise RuntimeError(("rank_fail",R,geom))

def render(A,line_geom,strength,clock,seed):
    rng=np.random.default_rng(seed); pi=stationary(A); p0,p2=CLOCKS[clock]; R=len(A)
    surf=[]; src=[]
    for gi,g in enumerate(line_geom):
        L=g["L"]; x=int(rng.choice(R,p=pi)); feats=[]; states=[]; guard=0
        while len(feats)<L:
            guard+=1
            if guard>100000: raise RuntimeError("clock_guard")
            if states: x=int(rng.choice(R,p=A[x]))
            states.append(x)
            u=rng.random(); nout=0 if u<p0 else (2 if u>1-p2 else 1)
            for _ in range(nout):
                if len(feats)>=L: break
                pool=POOLS[strength][x]
                feats.append(pool[int(rng.integers(len(pool)))])
        surf.append({"folio":g["folio"],"feat":np.stack(feats),"L":L})
        src.append(np.asarray(states,dtype=np.int16))
    return surf,src

def xy_surface(lines):
    X=[];Y=[];meta=[]
    for line in lines:
        F=line["feat"]
        for i in range(3,len(F)-2):
            X.append(F[i-3:i].reshape(-1)); Y.append(F[i:i+3].reshape(-1)); meta.append(line["folio"])
    if not X: return np.empty((0,87)),np.empty((0,87)),[]
    return np.asarray(X,float),np.asarray(Y,float),meta

def xy_source(src_lines,R):
    X=[];Y=[]
    eye=np.eye(R,dtype=float)
    for z in src_lines:
        if len(z)<7: continue
        F=eye[z]
        for i in range(3,len(F)-2):
            X.append(F[i-3:i].reshape(-1));Y.append(F[i:i+3].reshape(-1))
    if not X:return np.empty((0,3*R)),np.empty((0,3*R))
    return np.asarray(X),np.asarray(Y)

def op_from_xy(X,Y):
    if len(X)<20: raise RuntimeError(("too_few",len(X)))
    mx=X.mean(0); my=Y.mean(0); Xc=X-mx;Yc=Y-my
    C=(Xc.T@Yc)/len(Xc)
    U,s,Vt=np.linalg.svd(C,full_matrices=False)
    return {"C":C,"U":U,"s":s,"V":Vt.T,"mx":mx,"my":my,"Xc":Xc,"Yc":Yc}

def rank_summary(op):
    s=op["s"]; top=np.zeros(12); top[:min(12,len(s))]=s[:12]
    logs=np.log10(top+1e-8)
    sq=s*s; tot=float(sq.sum()); s1=float(s[0]) if len(s) else 0.
    stable=float(tot/(s1*s1+1e-30))
    if s.sum()>0:
        p=s/s.sum(); er=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum()))
    else: er=0.
    cums=[float(sq[:k].sum()/(tot+1e-30)) for k in (1,2,4,8)]
    frob=float(np.sqrt(tot))
    return np.r_[logs,stable,er,cums,frob]

REAL_OP=op_from_xy(*xy_surface(REAL_LINES)[:2]); REAL_SUM=rank_summary(REAL_OP)
print("REAL_SPECTRUM",json.dumps({"s":REAL_OP["s"][:12].tolist(),"summary":REAL_SUM.tolist()}),flush=True)

def simulate_case(rep,R,geom,strength,clock):
    base=SEED+rep*10_000_000+R*100_000+GEOMS.index(geom)*10_000+int(strength*100)*10+list(CLOCKS).index(clock)
    A=make_A(R,geom,base+1)
    surf,src=render(A,REAL_LINES,strength,clock,base+7)
    sop=op_from_xy(*xy_surface(surf)[:2]); summ=rank_summary(sop)
    xsr,ysr=xy_source(src,R); srcop=op_from_xy(xsr,ysr); srcsum=rank_summary(srcop)
    surf_norm=float(summ[-1]/87.0); src_norm=float(srcsum[-1]/max(3*R,1))
    return {"rep":rep,"R":R,"geom":geom,"strength":strength,"clock":clock,
            "summary":summ.tolist(),
            "surf_stable":float(summ[12]),"surf_eff":float(summ[13]),"surf_frob":float(summ[-1]),
            "src_stable":float(srcsum[12]),"src_eff":float(srcsum[13]),"src_frob":float(srcsum[-1]),
            "energy_ratio":(float(surf_norm/src_norm) if src_norm>1e-15 else None)}

def run_rep(rep):
    out=[]
    for R in RGRID:
      for geom in GEOMS:
       for strength in STRENGTHS:
        for clock in CLOCKS:
            out.append(simulate_case(rep,R,geom,strength,clock))
    return rep,out

SIM=[]
if __name__=="__main__":
    import multiprocessing as mp
    with mp.get_context("fork").Pool(4) as pool:
        for rep,out in pool.imap_unordered(run_rep,range(4)):
            SIM.extend(out); print("SIM_DONE",rep,len(out),flush=True)

def fit_rank(R):
    rows=[r for r in SIM if r["R"]==R and r["rep"] in (0,1)]
    X=np.array([r["summary"] for r in rows]); mu=X.mean(0); lw=LedoitWolf().fit(X)
    return rows,mu,lw.precision_

MODELS={R:fit_rank(R) for R in RGRID}
def distance(R,v):
    rows,mu,P=MODELS[R]; d=v-mu
    return float(np.sqrt(max(0.,d@P@d)))

CAL={}
for R in RGRID:
    ds=[distance(R,np.array(r["summary"])) for r in SIM if r["R"]==R and r["rep"]==2]
    q99=float(np.quantile(ds,.99))
    blind=[distance(R,np.array(r["summary"])) for r in SIM if r["R"]==R and r["rep"]==3]
    acc=float(np.mean(np.array(blind)<=q99))
    CAL[R]={"q99":q99,"blind_acceptance":acc,"eligible":bool(acc>=.90),
            "cal_dist":ds,"blind_dist":blind}

pairs=rejects=0
rank_errors=[]
for r in [z for z in SIM if z["rep"]==3]:
    v=np.array(r["summary"]); trueR=r["R"]
    scores=[]
    for R in RGRID:
        d=distance(R,v); q=CAL[R]["q99"]; scores.append((d/max(q,1e-12),R))
        if CAL[R]["eligible"] and R<=trueR/2:
            pairs+=1; rejects+=int(d>q)
    pred=min(scores)[1]; rank_errors.append(abs(math.log2(pred/trueR)))
POWER=float(rejects/pairs) if pairs else 0.
MEDERR=float(np.median(rank_errors))
GLOBAL_POWER=bool(POWER>=.70 and MEDERR<=1.0)

REAL_CLASS={}
for R in RGRID:
    d=distance(R,REAL_SUM); q=CAL[R]["q99"]; eligible=CAL[R]["eligible"]
    REAL_CLASS[R]={"distance":d,"q99":q,"eligible":eligible,"rejected":bool(eligible and d>q),"ratio":float(d/max(q,1e-12))}
eligible_nonrej=[R for R in RGRID if REAL_CLASS[R]["eligible"] and not REAL_CLASS[R]["rejected"]]
min_nonrej=min(eligible_nonrej) if eligible_nonrej else None
prefix=[]
for R in RGRID:
    if REAL_CLASS[R]["eligible"] and REAL_CLASS[R]["rejected"]: prefix.append(R)
    else: break
hard_excluded_through=(max(prefix) if (GLOBAL_POWER and prefix) else None)

STAGE_A={"calibration":{str(k):{kk:vv for kk,vv in v.items() if kk not in ("cal_dist","blind_dist")} for k,v in CAL.items()},
         "blind_low_rank_rejection_power":POWER,"nearest_log2_rank_error_median":MEDERR,"global_power_gate":GLOBAL_POWER,
         "real_classes":{str(k):v for k,v in REAL_CLASS.items()},"min_nonrejected_eligible_rank":min_nonrej,
         "hard_excluded_through":hard_excluded_through}
print("STAGE_A",json.dumps(STAGE_A,separators=(",",":")),flush=True)

# Stage C always descriptive
STAGE_C={}
for R in RGRID:
    rows=[r for r in SIM if r["R"]==R]
    def mean(key):
        a=[r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        return float(np.mean(a)) if a else None
    STAGE_C[str(R)]={"src_stable_mean":mean("src_stable"),"surf_stable_mean":mean("surf_stable"),
                     "src_eff_mean":mean("src_eff"),"surf_eff_mean":mean("surf_eff"),
                     "energy_ratio_mean":mean("energy_ratio")}
for key in ("stable","eff"):
    xs=np.array([r["src_"+key] for r in SIM],float);ys=np.array([r["surf_"+key] for r in SIM],float)
    STAGE_C["corr_"+key]=float(np.corrcoef(xs,ys)[0,1])

def overlap(op1,op2,k):
    kk=min(k,op1["U"].shape[1],op2["U"].shape[1])
    pu=float(np.linalg.norm(op1["U"][:,:kk].T@op2["U"][:,:kk],"fro")**2/kk)
    pv=float(np.linalg.norm(op1["V"][:,:kk].T@op2["V"][:,:kk],"fro")**2/kk)
    return .5*(pu+pv),pu,pv

STAGE_B={"opened":False}
stageB_gate=False;k_embed=None;REAL_ODD_OP=REAL_EVEN_OP=None
if GLOBAL_POWER and min_nonrej is not None:
    STAGE_B["opened"]=True
    REAL_ODD_OP=op_from_xy(*xy_surface(ODD)[:2]); REAL_EVEN_OP=op_from_xy(*xy_surface(EVEN)[:2])
    same={k:[] for k in (2,4,8,12)}; null={k:[] for k in (2,4,8,12)}
    Rt=min_nonrej
    for rep in range(8):
      for geom in GEOMS:
       for strength in STRENGTHS:
        for clock in CLOCKS:
            base=SEED+90_000_000+rep*1_000_000+Rt*10000+GEOMS.index(geom)*1000+int(strength*100)+list(CLOCKS).index(clock)
            A=make_A(Rt,geom,base+1); A2=make_A(Rt,geom,base+333)
            so,_=render(A,ODD,strength,clock,base+11); se,_=render(A,EVEN,strength,clock,base+22); ne,_=render(A2,EVEN,strength,clock,base+44)
            opo=op_from_xy(*xy_surface(so)[:2]); ope=op_from_xy(*xy_surface(se)[:2]); opn=op_from_xy(*xy_surface(ne)[:2])
            for k in same:
                same[k].append(overlap(opo,ope,k)[0]); null[k].append(overlap(opo,opn,k)[0])
    out={}
    for k in same:
        rv,rp,rf=overlap(REAL_ODD_OP,REAL_EVEN_OP,k); nm=float(np.mean(null[k])); ns=float(np.std(null[k],ddof=1))
        lo,hi=np.quantile(same[k],[.025,.975]); z=float((rv-nm)/ns) if ns>0 else None
        resolved=bool(z is not None and z>=2 and lo<=rv<=hi)
        out[str(k)]={"real":rv,"past":rp,"future":rf,"null_mean":nm,"null_sd":ns,"z":z,
                     "same_source_q025":float(lo),"same_source_q975":float(hi),"resolved":resolved}
    choices=[2,4,8,12]; k_embed=next((k for k in choices if k>=Rt),12)
    stageB_gate=bool(out[str(k_embed)]["resolved"])
    STAGE_B.update({"rank_used":Rt,"overlaps":out,"embedding_k":k_embed,"gate_for_stageD":stageB_gate})
print("STAGE_B",json.dumps(STAGE_B,separators=(",",":")),flush=True)

# Stage D anonymous state reconstruction
STAGE_D={"opened":False}
if GLOBAL_POWER and hard_excluded_through is not None and hard_excluded_through>=2 and stageB_gate:
    STAGE_D["opened"]=True
    Xall,Yall,meta=xy_surface(REAL_LINES); opall=op_from_xy(Xall,Yall)
    Z=opall["Xc"]@opall["U"][:,:k_embed]
    oddmask=np.array([fnum(f)%2==1 for f in meta]); evenmask=~oddmask
    Ks=(2,3,4,6,8,12)
    def fit_direction(trainmask,testmask,K,seed,nperm=100):
        gm=GaussianMixture(K,covariance_type="full",random_state=seed,n_init=5,reg_covar=1e-5).fit(Z[trainmask])
        lt=gm.predict(Z[trainmask]); lv=gm.predict(Z[testmask])
        probs=[]
        for c in range(K):
            yy=Yall[trainmask][lt==c]
            p=np.clip(yy.mean(0) if len(yy) else np.full(Yall.shape[1],.5),.02,.98);probs.append(p)
        probs=np.array(probs)
        yv=Yall[testmask]; p=probs[lv]
        real=float(np.mean(np.sum(yv*np.log(p)+(1-yv)*np.log(1-p),axis=1)))
        rng=np.random.default_rng(seed+999);null=[]
        yt=Yall[trainmask].copy(); ytest=Yall[testmask].copy()
        for _ in range(nperm):
            ytp=yt[rng.permutation(len(yt))]; yvp=ytest[rng.permutation(len(ytest))]
            pp=[]
            for c in range(K):
                yy=ytp[lt==c]; pp.append(np.clip(yy.mean(0) if len(yy) else np.full(Yall.shape[1],.5),.02,.98))
            pp=np.array(pp)[lv]
            null.append(float(np.mean(np.sum(yvp*np.log(pp)+(1-yvp)*np.log(1-pp),axis=1))))
        ns=float(np.std(null,ddof=1));z=float((real-np.mean(null))/ns) if ns>0 else None
        occ=[float(np.mean(lt==c)) for c in range(K)]
        return gm,{"real_ll":real,"null_mean":float(np.mean(null)),"null_sd":ns,"z":z,"min_train_occupancy":min(occ)}
    candidates={}
    fitted={}
    for K in Ks:
        go,a=fit_direction(oddmask,evenmask,K,SEED+K*10+1); ge,b=fit_direction(evenmask,oddmask,K,SEED+K*10+2)
        candidates[K]={"odd_to_even":a,"even_to_odd":b,"mean_ll":.5*(a["real_ll"]+b["real_ll"])}
        fitted[K]=(go,ge)
    bestK=max(Ks,key=lambda k:candidates[k]["mean_ll"])
    go,ge=fitted[bestK]
    co=go.means_;ce=ge.means_
    lo=np.argmin(((Z[:,None,:]-co[None,:,:])**2).sum(2),axis=1)
    le=np.argmin(((Z[:,None,:]-ce[None,:,:])**2).sum(2),axis=1)
    ami=float(adjusted_mutual_info_score(lo,le))
    occo=float(min(np.mean(lo==c) for c in range(bestK))); occe=float(min(np.mean(le==c) for c in range(bestK)))
    ca=candidates[bestK]
    promoted=bool(ca["odd_to_even"]["z"]>=2 and ca["even_to_odd"]["z"]>=2 and ami>=.35 and occo>=.02 and occe>=.02)
    STAGE_D.update({"embedding_k":k_embed,"candidates":{str(k):v for k,v in candidates.items()},"selected_K":bestK,
                    "AMI_transferred":ami,"min_occupancy_odd_model":occo,"min_occupancy_even_model":occe,"promoted":promoted})
print("STAGE_D",json.dumps(STAGE_D,separators=(",",":")),flush=True)

OUT={"programme":"STARS-PSR-v01","status":"complete","seed":SEED,
     "real":{"N":REAL_N,"lines":len(REAL_LINES),"singular_values":REAL_OP["s"][:20].tolist(),"summary":REAL_SUM.tolist()},
     "stageA":STAGE_A,"stageB":STAGE_B,"stageC":STAGE_C,"stageD":STAGE_D}
print("STARS_PSR_JSON="+json.dumps(OUT,separators=(",",":")),flush=True)
