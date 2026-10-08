#!/usr/bin/env python3
import collections,json,re,urllib.request
import numpy as np

SEED=20261008
K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
COREURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=120).read().decode(),K0URL,"exec"),k0)
core={"__name__":"core"};exec(compile(urllib.request.urlopen(COREURL,timeout=120).read().decode(),COREURL,"exec"),core)
segment=k0["segment"];ST=k0["ST"];PIECES=k0["PIECES"];controls=k0["controls"];gen_token=k0["gen_token"]
FOL=set("""f103r f103v f104r f104v f105r f105v f106r f106v f107r f107v f108r f108v f111r f111v f112r f112v f113r f113v f114r f114v""".split())
GEOMS=("dense","sparse","sticky","block");STRENGTHS=(.75,1.5);CLOCKS={"C1":(0.,0.),"C2":(0.,.25),"C3":(.25,0.),"C4":(.15,.15)}
RAW_FROZEN=0.05730764303793941

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
            if z:line.append((r["token"],z))
        if line:out.append((key[0],line))
    return out
REALREC=records();META=[(f,len(x)) for f,x in REALREC]
def feat(tok,route):
    c=[int(ST[p]) for p in route];v=np.zeros(29);v[c[0]]=1;v[12+c[-1]]=1;v[24+min(len(route),5)-1]=1;return v
def flines(rec):return [(f,np.array([feat(t,z) for t,z in line],float)) for f,line in rec]
REAL=flines(REALREC)

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
        out.append((f,np.array(B)))
    return out
def crossop(fl):
    P=[];F=[]
    for f,A in fl:
        if len(A)<6:continue
        for t in range(3,len(A)-3+1):P.append(A[t-3:t].reshape(-1));F.append(A[t:t+3].reshape(-1))
    P=np.array(P);F=np.array(F);P-=P.mean(0);F-=F.mean(0);C=P.T@F/len(P);U,s,Vt=np.linalg.svd(C,full_matrices=False);return C,U[:,:12],s[:12],Vt.T[:,:12]
def cvraw(fl):
    o=[x for x in fl if fnum(x[0])%2==1];e=[x for x in fl if fnum(x[0])%2==0];Co,Uo,so,Vo=crossop(o);Ce,Ue,se,Ve=crossop(e);raw=[]
    for i in range(12):raw.append(.5*(float(Uo[:,i].T@Ce@Vo[:,i])+float(Ue[:,i].T@Co@Ve[:,i])))
    return raw
REAL_RAW=cvraw(REAL);REAL_RES=cvraw(residualize(REAL))

# R1 renderer controls.
POOLS={}
for st in STRENGTHS:
    rng=np.random.default_rng(SEED+int(st*1000));U,Ve,Vr=controls(2,2,st,rng,"F1");pool=[]
    while len(pool)<500:
        z=gen_token(0,U,Ve,Vr,rng)
        if z is not None:
            route=[PIECES[int(i)] for i in z];pool.append(("".join(route),route))
    POOLS[st]=pool
def sim(rep,g,st,c):
    base=SEED+rep*10_000_000+GEOMS.index(g)*10_000+int(st*1000)*10+list(CLOCKS).index(c);rng=np.random.default_rng(base);p0,p2=CLOCKS[c];rec=[]
    for j,(fol,L) in enumerate(META):
        rr=np.random.default_rng(base+1000+j*173);line=[]
        while len(line)<L:
            u=rr.random();nout=0 if u<p0 else (2 if u>1-p2 else 1)
            for _ in range(nout):
                if len(line)>=L:break
                line.append(POOLS[st][int(rr.integers(len(POOLS[st])))])
        rec.append((fol,line))
    return {"rep":rep,"geom":g,"strength":st,"clock":c,"raw":cvraw(residualize(flines(rec)))}
SIM=[sim(rep,g,st,c) for rep in range(4) for g in GEOMS for st in STRENGTHS for c in CLOCKS]
cal=[x["raw"][0] for x in SIM if x["rep"] in (0,1,2)];q99=float(np.quantile(cal,.99))
blind=[x["raw"][0] for x in SIM if x["rep"]==3];accept=float(np.mean(np.array(blind)<=q99))
ret=float(REAL_RES[0]/RAW_FROZEN)
if accept<.90:decision="UNRESOLVED_CALIBRATION_FAIL"
elif REAL_RES[0]<=q99:decision="ABSORBED_BY_LINE_POSITION_NUISANCE"
elif ret<.50:decision="MIXED_RESIDUAL_SURVIVES_BUT_MAJORITY_POSITION_LINKED"
else:decision="SURVIVES_LINE_POSITION_NUISANCE"
OUT={"programme":"STARS-PSR3","status":"complete","real_raw_recomputed":REAL_RAW,"real_residual":REAL_RES,
     "frozen_raw_c1":RAW_FROZEN,"residual_c1":REAL_RES[0],"retention":ret,
     "r1_cal_q99":q99,"r1_blind_accept":accept,"decision":decision,
     "c1_to_c6_raw_to_residual":[{"i":i+1,"raw":REAL_RAW[i],"residual":REAL_RES[i],"ratio":(REAL_RES[i]/REAL_RAW[i] if abs(REAL_RAW[i])>1e-12 else None)} for i in range(6)]}
print("STARS_PSR3_JSON="+json.dumps(OUT,separators=(",",":")),flush=True)
