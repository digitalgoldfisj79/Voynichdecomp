#!/usr/bin/env python3
import collections,json,math,re,urllib.request,warnings
import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
SEED=20261008
K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
COREURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=120).read().decode(),K0URL,"exec"),k0)
core={"__name__":"core"};exec(compile(urllib.request.urlopen(COREURL,timeout=120).read().decode(),COREURL,"exec"),core)
segment=k0["segment"];ST=k0["ST"];PIECES=k0["PIECES"];controls=k0["controls"];gen_token=k0["gen_token"];source_graph=k0["source_graph"]
FOL=set("""f103r f103v f104r f104v f105r f105v f106r f106v f107r f107v f108r f108v f111r f111v f112r f112v f113r f113v f114r f114v""".split())
MODES=12
def fnum(f):return int(re.search(r"\d+",f).group())
def records():
    rows=[r for r in core["build_rows"]("ZLZI") if r["folio"] in FOL];by=collections.defaultdict(list)
    for r in rows:by[(r["folio"],r["line_ord"])].append(r)
    out=[]
    for key in sorted(by,key=lambda x:(fnum(x[0]),0 if x[0].endswith("r") else 1,x[1])):
        line=[]
        for r in sorted(by[key],key=lambda z:z["pos"]):
            try:z=segment(r["token"])
            except Exception:continue
            if z:line.append(z)
        if line:out.append((key[0],line))
    return out
REALREC=records()
def feat_route(route):
    c=[int(ST[p]) for p in route];v=np.zeros(29,float);v[c[0]]=1;v[12+c[-1]]=1;v[24+min(len(route),5)-1]=1;return v
REAL=[(f,np.array([feat_route(z) for z in line],float)) for f,line in REALREC]
META=[(f,len(line)) for f,line in REALREC]
def role(i,n):
    if i==0:return "START"
    if i==1:return "SECOND"
    if i==n-1:return "FINAL"
    if i==n-2:return "PENULT"
    rel=(i-2)/max(1,(n-5))
    return "EARLY" if rel<1/3 else ("MIDDLE" if rel<2/3 else "LATE")
def lbin(n):return "L7" if n<=7 else ("L10" if n<=10 else ("L14" if n<=14 else "L15P"))
def fit_means(fl,parity):
    cell=collections.defaultdict(list);rr=collections.defaultdict(list);allv=[]
    for f,A in fl:
        if fnum(f)%2!=parity:continue
        n=len(A)
        for i,v in enumerate(A):
            ro=role(i,n);cell[(ro,lbin(n))].append(v);rr[ro].append(v);allv.append(v)
    return ({k:np.mean(v,0) for k,v in cell.items()},{k:np.mean(v,0) for k,v in rr.items()},np.mean(allv,0))
def residualize(fl):
    mods={0:fit_means(fl,0),1:fit_means(fl,1)};out=[]
    for f,A in fl:
        p=fnum(f)%2;cell,rr,g=mods[1-p];n=len(A);B=[]
        for i,v in enumerate(A):
            ro=role(i,n);m=cell.get((ro,lbin(n)),rr.get(ro,g));B.append(v-m)
        out.append((f,np.array(B,float)))
    return out
REALRES=residualize(REAL)
def windows(fl,past=3,future=3,required_past=None):
    X=[];Y=[];fol=[]
    rp=past if required_past is None else required_past
    for f,A in fl:
        n=len(A)
        for t in range(rp,n-future+1):
            X.append(A[t-past:t].reshape(-1));Y.append(A[t:t+future].reshape(-1));fol.append(f)
    return np.array(X,float),np.array(Y,float),np.array(fol,object)
def sym_whitener(cov):
    w,V=np.linalg.eigh((cov+cov.T)*.5);w=np.maximum(w,1e-8);return (V*(1/np.sqrt(w)))@V.T
def fit_cca(X,Y,modes=MODES):
    mx=X.mean(0);my=Y.mean(0);Xc=X-mx;Yc=Y-my
    sx=LedoitWolf().fit(Xc).covariance_;sy=LedoitWolf().fit(Yc).covariance_
    Wx=sym_whitener(sx);Wy=sym_whitener(sy);C=Xc.T@Yc/len(Xc)
    U,s,Vt=np.linalg.svd(Wx@C@Wy,full_matrices=False);m=min(modes,len(s))
    return mx,my,Wx@U[:,:m],Wy@Vt.T[:,:m],s[:m]
def eval_corr(model,X,Y):
    mx,my,A,B,_=model;xp=(X-mx)@A;yp=(Y-my)@B;out=[]
    for i in range(xp.shape[1]):
        x=xp[:,i];y=yp[:,i]
        if np.std(x)<1e-12 or np.std(y)<1e-12:out.append(0.);continue
        out.append(float(np.corrcoef(x,y)[0,1]))
    return np.array(out,float)
def crossfit_from_arrays(X,Y,fol,modes=MODES):
    odd=np.array([fnum(str(f))%2==1 for f in fol]);even=~odd
    mo=fit_cca(X[odd],Y[odd],modes);me=fit_cca(X[even],Y[even],modes)
    oe=eval_corr(mo,X[even],Y[even]);eo=eval_corr(me,X[odd],Y[odd])
    m=min(len(oe),len(eo));avg=.5*(oe[:m]+eo[:m]);rho=np.maximum(avg,0.)
    return {"rho":rho,"odd_to_even":oe[:m],"even_to_odd":eo[:m]}
def permute_future_within_folio(Y,fol,rng):
    Z=Y.copy()
    for f in np.unique(fol):
        ix=np.where(fol==f)[0];Z[ix]=Y[rng.permutation(ix)]
    return Z
def null_threshold(X,Y,fol,nrep,seed):
    rng=np.random.default_rng(seed);mx=[];allrho=[]
    for b in range(nrep):
        Z=permute_future_within_folio(Y,fol,rng);cf=crossfit_from_arrays(X,Z,fol)
        r=cf["rho"];mx.append(float(r.max()) if len(r) else 0.);allrho.append(r.tolist())
    return float(np.quantile(mx,.99)),mx,allrho
def prd_curve(rhos,maxR=8.,step=.05):
    rhos=np.array([r for r in rhos if r>0],float)
    nB=int(round(maxR/step))
    if len(rhos)==0:return {"I_inf":0.,"budgets":[0.],"curve":[0.],"R50":None,"R80":None,"R90":None}
    vals=[]
    grid=np.arange(nB+1)*step
    for rho in rhos:
        q=1-np.power(2.,-2*grid)
        vals.append(-.5*np.log2(np.maximum(1-rho*rho*q,1e-15)))
    dp=np.full(nB+1,-np.inf);dp[0]=0.
    for v in vals:
        nd=np.full_like(dp,-np.inf)
        for b in range(nB+1):
            arr=dp[:b+1]+v[b::-1]
            nd[b]=np.max(arr)
        dp=nd
    # Allow <= budget rather than exactly budget.
    dp=np.maximum.accumulate(dp)
    Iinf=float(np.sum(-.5*np.log2(np.maximum(1-rhos*rhos,1e-15))))
    def rr(frac):
        if Iinf<=0:return None
        ix=np.where(dp>=frac*Iinf)[0]
        return float(ix[0]*step) if len(ix) else None
    return {"I_inf":Iinf,"budgets":grid.tolist(),"curve":dp.tolist(),"R50":rr(.5),"R80":rr(.8),"R90":rr(.9)}
def atR(prd,R):
    b=np.array(prd["budgets"]);c=np.array(prd["curve"]);return float(c[np.argmin(abs(b-R))])
# Primary real
X,Y,fol=windows(REALRES,3,3)
REALCF=crossfit_from_arrays(X,Y,fol)
Q99,NULLMAX,_=null_threshold(X,Y,fol,256,SEED+501)
validated=[float(r) for r in REALCF["rho"] if r>Q99]
PRIMARY_GATE=len(validated)>0
REALPRD=prd_curve(validated) if PRIMARY_GATE else prd_curve([])
PRIMARY={"n":len(X),"rho":REALCF["rho"].tolist(),"fold_oe":REALCF["odd_to_even"].tolist(),"fold_eo":REALCF["even_to_odd"].tolist(),
         "null_max_q99":Q99,"validated_rho":validated,"validated_modes":len(validated),"gate":PRIMARY_GATE,
         "I_inf":REALPRD["I_inf"],"I_at":{str(r):atR(REALPRD,r) for r in (.5,1,2,4,8)},
         "R50":REALPRD["R50"],"R80":REALPRD["R80"],"R90":REALPRD["R90"]}
print("PRIMARY",json.dumps(PRIMARY,separators=(",",":")),flush=True)
# Horizon audit
HORIZ={}
for L in (1,2,3,5):
    x,y,f=windows(REALRES,L,3,required_past=5);cf=crossfit_from_arrays(x,y,f);q,_,_=null_threshold(x,y,f,128,SEED+700+L)
    vr=[float(r) for r in cf["rho"] if r>q];p=prd_curve(vr)
    HORIZ[str(L)]={"n":len(x),"q99":q,"validated_modes":len(vr),"validated_rho":vr,"I_inf":p["I_inf"],"R80":p["R80"],
                   "I1":atR(p,1),"I2":atR(p,2),"I4":atR(p,4)}
    print("HORIZ",L,json.dumps(HORIZ[str(L)],separators=(",",":")),flush=True)
# Stage D source family simulations
GENS=("M1","VAR2","RENEW","MOTIF");STRENGTHS=(.75,1.5);CLOCKS={"C1":(0.,0.),"C2":(0.,.25),"C3":(.25,0.),"C4":(.15,.15)}
KMAX=32;POOLN=180;POOLS={}
for st in STRENGTHS:
    rng=np.random.default_rng(SEED+int(st*1000));U,Ve,Vr=controls(KMAX,2,st,rng,"F1");pp=[]
    for x in range(KMAX):
        pool=[]
        while len(pool)<POOLN:
            z=gen_token(x,U,Ve,Vr,rng)
            if z is not None:pool.append(feat_route([PIECES[int(i)] for i in z]))
        pp.append(np.array(pool,float))
    POOLS[st]=pp
print("POOLS_READY",flush=True)
def make_params(fam,K,level,seed):
    rs=np.random.default_rng(seed)
    if fam=="M1":return {"fam":fam,"K":K,"A":source_graph(K,2 if level==0 else min(8,K),rs)[0]}
    if fam=="VAR2":return {"fam":fam,"K":K,"A":source_graph(K,min(4,K),rs)[0],"eta":(.25,.55)[level]}
    if fam=="RENEW":
        A=source_graph(K,min(4,K),rs)[0].copy();np.fill_diagonal(A,0.)
        for i in range(K):
            if A[i].sum()==0:A[i]=1.;A[i,i]=0.
            A[i]/=A[i].sum()
        return {"fam":fam,"K":K,"A":A,"mean":(2.,5.)[level]}
    motifs=[rs.integers(0,K,size=int(rs.integers(3,7)),dtype=np.int16) for _ in range(4)]
    return {"fam":fam,"K":K,"noise":(.25,.10)[level],"persist":(8.,20.)[level],"motifs":motifs}
def gen_source(par,n,seed):
    rq=np.random.default_rng(seed);fam=par["fam"];K=par["K"]
    if fam=="M1":
        A=par["A"];z=np.empty(n,np.int16);z[0]=rq.integers(K)
        for t in range(1,n):z[t]=rq.choice(K,p=A[z[t-1]])
        return z
    if fam=="VAR2":
        A=par["A"];eta=par["eta"];z=np.empty(n,np.int16);z[0]=rq.integers(K);z[1]=rq.choice(K,p=A[z[0]])
        for t in range(2,n):z[t]=z[t-2] if rq.random()<eta else rq.choice(K,p=A[z[t-1]])
        return z
    if fam=="RENEW":
        A=par["A"];mean=par["mean"];o=[];x=int(rq.integers(K))
        while len(o)<n:
            d=1+int(rq.poisson(max(.01,mean-1)));o.extend([x]*d);x=int(rq.choice(K,p=A[x]))
        return np.array(o[:n],np.int16)
    noise=par["noise"];persist=par["persist"];motifs=par["motifs"];o=np.empty(n,np.int16);m=int(rq.integers(4));ph=0
    for t in range(n):
        if t>0 and rq.random()<1/persist:m=int(rq.integers(4));ph=0
        o[t]=int(rq.integers(K)) if rq.random()<noise else int(motifs[m][ph%len(motifs[m])]);ph+=1
    return o
def render_lines(par,st,clock,rep,scode):
    p0,p2=CLOCKS[clock];surf=[];srcout=[]
    for j,(f,L) in enumerate(META):
        seed=SEED+rep*1000000+scode*200000+j*97+par["K"]*13
        z=gen_source(par,max(40,int(L*3)+20),seed);rng=np.random.default_rng(seed+555);i=0;line=[];used=[]
        while len(line)<L:
            x=int(z[i]);i+=1;used.append(x);u=rng.random();nout=0 if u<p0 else (2 if u>1-p2 else 1)
            for _ in range(nout):
                if len(line)>=L:break
                pool=POOLS[st][x];line.append(pool[int(rng.integers(len(pool)))])
        surf.append((f,np.array(line,float)));srcout.append((f,np.eye(par["K"])[np.array(used,dtype=int)]))
    return surf,srcout
def simple_prd(fl,resid=False):
    if resid:fl=residualize(fl)
    x,y,f=windows(fl,3,3);cf=crossfit_from_arrays(x,y,f);rr=[float(r) for r in cf["rho"] if r>0];p=prd_curve(rr)
    return {"I_inf":p["I_inf"],"R80":p["R80"],"rho1":float(cf["rho"][0]) if len(cf["rho"]) else 0.,"modes":len(rr)}
SIM=[]
for rep in (0,1):
  for K in (8,16,32):
   for fam in GENS:
    for level in (0,1):
     seed=SEED+rep*5_000_000+K*10000+level*100+GENS.index(fam)
     par=make_params(fam,K,level,seed)
     for st in STRENGTHS:
      for clock in CLOCKS:
       surf,src=render_lines(par,st,clock,rep,3)
       s= simple_prd(surf,True);q=simple_prd(src,False)
       SIM.append({"rep":rep,"K":K,"fam":fam,"level":level,"strength":st,"clock":clock,"surface":s,"source":q})
  print("SIMREP_DONE",rep,flush=True)
FAM={}
for fam in GENS:
    rr=[r for r in SIM if r["fam"]==fam]
    def stats(side,key):
        a=np.array([r[side][key] for r in rr if r[side][key] is not None and np.isfinite(r[side][key])],float)
        return {"median":float(np.median(a)),"q10":float(np.quantile(a,.1)),"q90":float(np.quantile(a,.9))}
    FAM[fam]={"surface_I_inf":stats("surface","I_inf"),"surface_R80":stats("surface","R80"),
              "source_I_inf":stats("source","I_inf"),"source_R80":stats("source","R80")}
xs=np.array([r["source"]["I_inf"] for r in SIM],float);ys=np.array([r["surface"]["I_inf"] for r in SIM],float)
sp=float(spearmanr(xs,ys).statistic)
STAGED={"families":FAM,"spearman_source_surface_I_inf":sp,
        "real_vs_envelopes":{fam:{"I_inf_inside_q10_q90":bool(FAM[fam]["surface_I_inf"]["q10"]<=REALPRD["I_inf"]<=FAM[fam]["surface_I_inf"]["q90"]),
                                         "R80_inside_q10_q90":(False if REALPRD["R80"] is None else bool(FAM[fam]["surface_R80"]["q10"]<=REALPRD["R80"]<=FAM[fam]["surface_R80"]["q90"]))} for fam in GENS}}
OUT={"programme":"STARS-PRD1","status":"complete","seed":SEED,"primary":PRIMARY,"horizon":HORIZ,"stageD":STAGED}
print("STARS_PRD1_JSON="+json.dumps(OUT,separators=(",",":")),flush=True)
