#!/usr/bin/env python3
import collections,json,math,pickle,re,urllib.request
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score

SEED=20261008
K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
COREURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=120).read().decode(),K0URL,"exec"),k0)
core={"__name__":"core"};exec(compile(urllib.request.urlopen(COREURL,timeout=120).read().decode(),COREURL,"exec"),core)
segment=k0["segment"];ST=k0["ST"];PIECES=k0["PIECES"];controls=k0["controls"];gen_token=k0["gen_token"]

STAR_FOLIOS=set("""f103r f103v f104r f104v f105r f105v f106r f106v f107r f107v f108r f108v f111r f111v f112r f112v f113r f113v f114r f114v""".split())
RANKS=(1,2,3,4,6,8,12,16,24,32)
GEOMS=("dense","sparse","sticky","block")
STRENGTHS=(.75,1.5)
CLOCKS={"C1_1to1":(0.,0.),"C2_expand":(0.,.25),"C3_delete":(.25,0.),"C4_mixed":(.15,.15)}
KMAX=33
POOLN=350

def fnum(f):
    return int(re.search(r"\d+",f).group())

def real_line_records():
    rows=[r for r in core["build_rows"]("ZLZI") if r["folio"] in STAR_FOLIOS]
    by=collections.defaultdict(list)
    for r in rows:by[(r["folio"],r["line_ord"])].append(r)
    out=[]
    for key in sorted(by,key=lambda x:(fnum(x[0]),0 if x[0].endswith("r") else 1,x[1])):
        line=[]
        for r in sorted(by[key],key=lambda z:z["pos"]):
            try:z=segment(r["token"])
            except Exception:continue
            if z:line.append((r["token"],z))
        if line:out.append({"folio":key[0],"line_ord":key[1],"tokens":line})
    return out

REALREC=real_line_records()
LINE_META=[(r["folio"],len(r["tokens"])) for r in REALREC]
print("REAL_GEOM",len(REALREC),sum(n for _,n in LINE_META),len(set(f for f,_ in LINE_META)),flush=True)

def tokfeat(tok,route):
    cls=[int(ST[p]) for p in route]
    v=np.zeros(29,float);v[cls[0]]=1.;v[12+cls[-1]]=1.;v[24+min(len(route),5)-1]=1.
    return v

def surface_feature_lines(records_or_lines,records=True):
    out=[]
    if records:
        for r in records_or_lines:out.append((r["folio"],np.array([tokfeat(t,z) for t,z in r["tokens"]],float)))
    else:
        for fol,line in records_or_lines:out.append((fol,np.array([tokfeat(t,z) for t,z in line],float)))
    return out

REALFL=surface_feature_lines(REALREC,True)

def windows(feature_lines,L=3):
    P=[];F=[]
    for fol,A in feature_lines:
        if len(A)<2*L:continue
        for t in range(L,len(A)-L+1):
            P.append(A[t-L:t].reshape(-1));F.append(A[t:t+L].reshape(-1))
    return np.array(P,float),np.array(F,float)

def op_svd(feature_lines,maxsv=20):
    P,F=windows(feature_lines,3)
    if len(P)<10:return np.zeros(maxsv),np.zeros((87,maxsv)),np.zeros((87,maxsv)),np.zeros(87),np.zeros(87),0.
    pm=P.mean(0);fm=F.mean(0);Pc=P-pm;Fc=F-fm;C=(Pc.T@Fc)/len(P)
    U,s,Vt=np.linalg.svd(C,full_matrices=False)
    z=np.zeros(maxsv);z[:min(maxsv,len(s))]=s[:maxsv]
    return z,U[:,:maxsv],Vt.T[:,:maxsv],pm,fm,float(np.linalg.norm(C,"fro")**2/(C.shape[0]*C.shape[1]))

def rank_summary(feature_lines):
    s,U,V,pm,fm,normE=op_svd(feature_lines,20)
    eps=1e-8
    ss=s[s>0]
    s1=max(s[0],eps);stable=float(np.sum(s*s)/(s1*s1))
    p=s/max(s.sum(),eps);eff=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0]))))
    e=s*s;den=max(e.sum(),eps)
    cum=[float(e[:k].sum()/den) for k in (1,2,4,8)]
    vec=np.array([math.log10(float(s[i])+eps) for i in range(12)]+[stable,eff]+cum+[normE],float)
    return vec,{"singular_values":s.tolist(),"stable_rank":stable,"effective_rank":eff,"cum_energy":cum,"norm_frob_energy":normE,"U":U,"V":V,"pm":pm,"fm":fm}

REAL_SUM,REAL_SPEC=rank_summary(REALFL)

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
        if geom=="dense":
            A=rng.dirichlet(np.ones(R),size=R)
        elif geom=="sparse":
            A=np.zeros((R,R));d=min(3,R)
            for i in range(R):
                idx=rng.choice(R,size=d,replace=False);w=rng.dirichlet(np.ones(d));A[i,idx]=w
            A=.97*A+.03*rng.dirichlet(np.ones(R),size=R);A/=A.sum(1,keepdims=True)
        elif geom=="sticky":
            B=rng.dirichlet(np.ones(R),size=R);A=.65*np.eye(R)+.35*B;A/=A.sum(1,keepdims=True)
        elif geom=="block":
            cut=max(1,R//2);A=np.zeros((R,R))
            for i in range(R):
                own=np.arange(0,cut) if i<cut else np.arange(cut,R)
                other=np.arange(cut,R) if i<cut else np.arange(0,cut)
                if len(other)==0:other=own
                wo=rng.dirichlet(np.ones(len(own)));wx=rng.dirichlet(np.ones(len(other)))
                A[i,own]+=0.85*wo;A[i,other]+=0.15*wx
            A/=A.sum(1,keepdims=True)
        else:raise ValueError(geom)
        if np.linalg.matrix_rank(A,tol=1e-10)==R:return A
    raise RuntimeError(("rank_draw_fail",R,geom))

# Frozen renderer pools.
POOLS={}
for strength in STRENGTHS:
    rng=np.random.default_rng(SEED+int(strength*1000))
    U,Ve,Vr=controls(KMAX,2,strength,rng,"F1");pp=[]
    for x in range(KMAX):
        pool=[]
        while len(pool)<POOLN:
            z=gen_token(x,U,Ve,Vr,rng)
            if z is not None:
                route=[PIECES[int(i)] for i in z];pool.append(("".join(route),route))
        pp.append(pool)
    POOLS[strength]=pp
print("POOLS_READY",flush=True)

def render_line(A,L,strength,clock,seed):
    rng=np.random.default_rng(seed);pi=stationary(A);x=int(rng.choice(len(A),p=pi))
    p0,p2=CLOCKS[clock];surf=[];src=[]
    guard=0
    while len(surf)<L:
        guard+=1
        if guard>100000:raise RuntimeError("clock")
        if src:x=int(rng.choice(len(A),p=A[x]))
        src.append(x)
        u=rng.random();nout=0 if u<p0 else (2 if u>1-p2 else 1)
        for _ in range(nout):
            if len(surf)>=L:break
            pool=POOLS[strength][x];surf.append(pool[int(rng.integers(len(pool)))])
    return surf,src

def source_feature_lines(src_records,R):
    out=[]
    for fol,seq in src_records:
        A=np.zeros((len(seq),R),float)
        if len(seq):A[np.arange(len(seq)),np.array(seq,int)]=1.
        out.append((fol,A))
    return out

def overlap(U1,V1,U2,V2,k):
    k=min(k,U1.shape[1],U2.shape[1],V1.shape[1],V2.shape[1])
    if k<=0:return 0.
    a=float(np.linalg.norm(U1[:,:k].T@U2[:,:k],"fro")**2/k)
    b=float(np.linalg.norm(V1[:,:k].T@V2[:,:k],"fro")**2/k)
    return (a+b)/2

def simulate_case(rep,R,geom,strength,clock):
    base=SEED+rep*10_000_000+R*100_000+GEOMS.index(geom)*10_000+int(strength*1000)*10+list(CLOCKS).index(clock)
    A=make_A(R,geom,base)
    surfrec=[];srcrec=[]
    for j,(fol,L) in enumerate(LINE_META):
        surf,src=render_line(A,L,strength,clock,base+1000+j*173)
        surfrec.append((fol,surf));srcrec.append((fol,src))
    fl=surface_feature_lines(surfrec,False);vec,spec=rank_summary(fl)
    sf=source_feature_lines(srcrec,R);svec,sspec=rank_summary_general(sf)
    # odd/even predictive subspaces
    odd=[x for x in fl if fnum(x[0])%2==1];even=[x for x in fl if fnum(x[0])%2==0]
    _,ospec=rank_summary(odd);_,espec=rank_summary(even)
    return {"rep":rep,"R":R,"geom":geom,"strength":strength,"clock":clock,"summary":vec.tolist(),
            "surface_stable":spec["stable_rank"],"surface_eff":spec["effective_rank"],"surface_energy":spec["norm_frob_energy"],
            "source_stable":sspec["stable_rank"],"source_eff":sspec["effective_rank"],"source_energy":sspec["norm_frob_energy"],
            "Uo":ospec["U"].tolist(),"Vo":ospec["V"].tolist(),"Ue":espec["U"].tolist(),"Ve":espec["V"].tolist()}

def op_svd_general(feature_lines,maxsv=20):
    # same 3-step windows, arbitrary per-token feature dimension.
    P=[];F=[]
    for fol,A in feature_lines:
        if len(A)<6:continue
        for t in range(3,len(A)-3+1):
            P.append(A[t-3:t].reshape(-1));F.append(A[t:t+3].reshape(-1))
    if not P:
        d=3*feature_lines[0][1].shape[1];return np.zeros(maxsv),np.zeros((d,maxsv)),np.zeros((d,maxsv)),0.
    P=np.array(P,float);F=np.array(F,float);P-=P.mean(0);F-=F.mean(0);C=(P.T@F)/len(P)
    U,s,Vt=np.linalg.svd(C,full_matrices=False);z=np.zeros(maxsv);z[:min(maxsv,len(s))]=s[:maxsv]
    return z,U[:,:maxsv],Vt.T[:,:maxsv],float(np.linalg.norm(C,"fro")**2/(C.shape[0]*C.shape[1]))

def rank_summary_general(feature_lines):
    s,U,V,normE=op_svd_general(feature_lines,20);eps=1e-8;s1=max(s[0],eps)
    stable=float(np.sum(s*s)/(s1*s1));p=s/max(s.sum(),eps);eff=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0]))))
    e=s*s;den=max(e.sum(),eps);cum=[float(e[:k].sum()/den) for k in (1,2,4,8)]
    vec=np.array([math.log10(float(s[i])+eps) for i in range(12)]+[stable,eff]+cum+[normE],float)
    return vec,{"stable_rank":stable,"effective_rank":eff,"norm_frob_energy":normE}

def run_rep(rep):
    out=[]
    for R in RANKS:
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
            SIM.extend(out);print("SIM_DONE",rep,len(out),flush=True)

# Stage A class models.
MODELS={}
for R in RANKS:
    train=[r for r in SIM if r["R"]==R and r["rep"] in (0,1)]
    X=np.array([r["summary"] for r in train]);lw=LedoitWolf().fit(X);P=lw.precision_
    cal=[r for r in SIM if r["R"]==R and r["rep"]==2]
    def dist(v):
        D=X-v;return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",D,P,D),0))
    cald=[float(np.min(dist(np.array(r["summary"])))) for r in cal]
    q99=float(np.quantile(cald,.99))
    blind=[r for r in SIM if r["R"]==R and r["rep"]==3]
    blindd=[float(np.min(dist(np.array(r["summary"])))) for r in blind]
    accept=float(np.mean(np.array(blindd)<=q99))
    reald=float(np.min(dist(REAL_SUM)))
    MODELS[R]={"X":X,"P":P,"q99":q99,"blind_accept":accept,"eligible":bool(accept>=.90),"real_distance":reald,"real_reject":bool(accept>=.90 and reald>q99)}

def class_distance(R,v):
    M=MODELS[R];D=M["X"]-v;dd=np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",D,M["P"],D),0));return float(np.min(dd))

# Power gate.
pairs=hit=0;rankerrs=[]
for t in [r for r in SIM if r["rep"]==3]:
    v=np.array(t["summary"]);true=t["R"]
    pred=min(RANKS,key=lambda R:class_distance(R,v)/max(MODELS[R]["q99"],1e-9))
    rankerrs.append(abs(math.log2(pred/true)))
    for R in RANKS:
        if R<=true/2 and MODELS[R]["eligible"]:
            pairs+=1;hit+=int(class_distance(R,v)>MODELS[R]["q99"])
power=float(hit/pairs) if pairs else 0.
mederr=float(np.median(rankerrs))
GLOBAL_POWER=bool(power>=.70 and mederr<=1.0)

eligible_rej=[R for R in RANKS if MODELS[R]["real_reject"]]
consecutive=[]
for R in RANKS:
    if R in eligible_rej:consecutive.append(R)
    else:break
hard_exclude_through=max(consecutive) if consecutive and GLOBAL_POWER else None
compatible=[R for R in RANKS if MODELS[R]["eligible"] and not MODELS[R]["real_reject"]]
min_compatible=min(compatible) if compatible else None

# Stage B.
def real_split_spec(parity):
    fl=[x for x in REALFL if fnum(x[0])%2==parity]
    return rank_summary(fl)[1]
RO=real_split_spec(1);RE=real_split_spec(0)
STAGEB={"opened":GLOBAL_POWER,"results":{},"resolved_any":False}
if GLOBAL_POWER:
    # index cases
    ix={(r["rep"],r["R"],r["geom"],r["strength"],r["clock"]):r for r in SIM}
    for k in (2,4,8,12):
        realov=overlap(np.array(RO["U"]),np.array(RO["V"]),np.array(RE["U"]),np.array(RE["V"]),k)
        same=[];null=[]
        for r in [q for q in SIM if q["rep"]==3]:
            same.append(overlap(np.array(r["Uo"]),np.array(r["Vo"]),np.array(r["Ue"]),np.array(r["Ve"]),k))
            mate=ix[(2,r["R"],r["geom"],r["strength"],r["clock"])]
            null.append(overlap(np.array(r["Uo"]),np.array(r["Vo"]),np.array(mate["Ue"]),np.array(mate["Ve"]),k))
        mu=float(np.mean(null));sd=float(np.std(null,ddof=1));z=(realov-mu)/sd if sd>0 else None
        lo,hi=np.quantile(same,[.025,.975]);resolved=bool(z is not None and z>=2 and lo<=realov<=hi)
        STAGEB["results"][str(k)]={"real_overlap":realov,"null_mean":mu,"null_sd":sd,"z":z,"same_q025":float(lo),"same_q975":float(hi),"resolved":resolved}
        STAGEB["resolved_any"]=STAGEB["resolved_any"] or resolved

# Stage C.
STAGEC={}
for R in RANKS:
    rr=[x for x in SIM if x["R"]==R]
    a=np.array([x["source_eff"] for x in rr]);b=np.array([x["surface_eff"] for x in rr])
    c=np.array([x["source_stable"] for x in rr]);d=np.array([x["surface_stable"] for x in rr])
    se=np.array([x["source_energy"] for x in rr]);oe=np.array([x["surface_energy"] for x in rr])
    STAGEC[str(R)]={"source_eff_median":float(np.median(a)),"surface_eff_median":float(np.median(b)),
                    "source_stable_median":float(np.median(c)),"surface_stable_median":float(np.median(d)),
                    "norm_energy_ratio_median":float(np.median(oe/np.maximum(se,1e-15))),
                    "eff_corr":float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None,
                    "stable_corr":float(np.corrcoef(c,d)[0,1]) if np.std(c)>0 and np.std(d)>0 else None}

# Stage D anonymous state reconstruction.
STAGED={"opened":bool(GLOBAL_POWER and hard_exclude_through is not None and hard_exclude_through>=2 and STAGEB["resolved_any"]),"result":None}
if STAGED["opened"]:
    # choose embedding dimension: smallest {2,4,8,12} >= min compatible; fallback 12
    k=next((x for x in (2,4,8,12) if min_compatible is not None and x>=min_compatible),12)
    P,F=windows(REALFL,3);pm=P.mean(0);fm=F.mean(0);Pc=P-pm;Fc=F-fm;C=(Pc.T@Fc)/len(P);U,s,Vt=np.linalg.svd(C,full_matrices=False);V=Vt.T
    Zp=Pc@U[:,:k];Zf=Fc@V[:,:k]
    # reconstruct folio parity per window in same order
    pars=[]
    for fol,A in REALFL:
        if len(A)<6:continue
        pars += [fnum(fol)%2]*(len(A)-6+1)
    pars=np.array(pars)
    dirs=[(1,0),(0,1)];cand=(2,3,4,6,8,12);dirres=[];models=[]
    rng=np.random.default_rng(SEED+9090)
    for trpar,tepar in dirs:
        tr=np.where(pars==trpar)[0];te=np.where(pars==tepar)[0]
        scores=[]
        for Kstate in cand:
            gm=GaussianMixture(Kstate,covariance_type="full",reg_covar=1e-4,random_state=SEED+Kstate+trpar*100,max_iter=300,n_init=3).fit(Zp[tr])
            resp=gm.predict_proba(Zp[tr]);means=[];vars=[]
            for z in range(Kstate):
                w=resp[:,z]+1e-9;w/=w.sum();m=(w[:,None]*Zf[tr]).sum(0);var=(w[:,None]*(Zf[tr]-m)**2).sum(0)+1e-4;means.append(m);vars.append(var)
            means=np.array(means);vars=np.array(vars);rte=gm.predict_proba(Zp[te])
            logcomp=[]
            for z in range(Kstate):
                lp=-.5*(np.sum(np.log(2*np.pi*vars[z]))+np.sum((Zf[te]-means[z])**2/vars[z],axis=1))
                logcomp.append(np.log(rte[:,z]+1e-300)+lp)
            L=np.stack(logcomp,1);mx=L.max(1);ll=float(np.mean(mx+np.log(np.exp(L-mx[:,None]).sum(1))))
            scores.append((ll,Kstate,gm,means,vars))
        scores.sort(reverse=True,key=lambda x:x[0]);best=scores[0];ll,Kstate,gm,means,vars=best
        # rank-matched shuffled-history null: keep GMM/past clusters, shuffle future in training and refit conditional future model.
        null=[]
        resp=gm.predict_proba(Zp[tr]);rte=gm.predict_proba(Zp[te])
        for b in range(100):
            perm=rng.permutation(len(tr));m2=[];v2=[]
            for z in range(Kstate):
                w=resp[:,z]+1e-9;w/=w.sum();Y=Zf[tr][perm];m=(w[:,None]*Y).sum(0);var=(w[:,None]*(Y-m)**2).sum(0)+1e-4;m2.append(m);v2.append(var)
            m2=np.array(m2);v2=np.array(v2);lc=[]
            for z in range(Kstate):
                lp=-.5*(np.sum(np.log(2*np.pi*v2[z]))+np.sum((Zf[te]-m2[z])**2/v2[z],axis=1));lc.append(np.log(rte[:,z]+1e-300)+lp)
            L2=np.stack(lc,1);mx=L2.max(1);null.append(float(np.mean(mx+np.log(np.exp(L2-mx[:,None]).sum(1)))))
        mu=float(np.mean(null));sd=float(np.std(null,ddof=1));z=(ll-mu)/sd if sd>0 else None
        occ=np.bincount(gm.predict(Zp),minlength=Kstate)/len(Zp)
        dirres.append({"train_parity":trpar,"test_parity":tepar,"Kstate":Kstate,"heldout_ll":ll,"null_mean":mu,"null_sd":sd,"z":z,"min_occupancy":float(occ.min())})
        models.append(gm)
    labels0=models[0].predict(Zp);labels1=models[1].predict(Zp);ami=float(adjusted_mutual_info_score(labels0,labels1))
    promoted=bool(all(x["z"] is not None and x["z"]>=2 and x["min_occupancy"]>=.02 for x in dirres) and ami>=.35)
    STAGED["result"]={"embedding_k":k,"directions":dirres,"crossfit_AMI":ami,"promoted":promoted}

OUT={"programme":"STARS-PSR-v1","status":"complete","seed":SEED,
     "real":{"N_tokens":sum(n for _,n in LINE_META),"N_lines":len(LINE_META),
             "summary":REAL_SUM.tolist(),"singular_values":REAL_SPEC["singular_values"],
             "stable_rank":REAL_SPEC["stable_rank"],"effective_rank":REAL_SPEC["effective_rank"],
             "cum_energy":REAL_SPEC["cum_energy"],"norm_frob_energy":REAL_SPEC["norm_frob_energy"]},
     "stageA":{"models":{str(R):{k:v for k,v in M.items() if k not in ("X","P")} for R,M in MODELS.items()},
               "power_reject_half_rank":power,"median_abs_log2_rank_error":mederr,"global_power":GLOBAL_POWER,
               "rejected_ranks":eligible_rej,"hard_exclude_through":hard_exclude_through,
               "compatible_ranks":compatible,"min_compatible_rank":min_compatible},
     "stageB":STAGEB,"stageC":STAGEC,"stageD":STAGED}
print("STARS_PSR_V1_JSON="+json.dumps(OUT,separators=(",",":")),flush=True)
