#!/usr/bin/env python3
# VMS-RESIDREC1 — preregistered 2026-10-08.
import json,math,os,urllib.request,warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.covariance import LedoitWolf
warnings.filterwarnings("ignore")

ECO_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0677011497ff31d84ebb93c2c65a67be57ae39af/research/vms_ecology1_running_residual_20261008.py"
m={"__name__":"eco_module"}
exec(compile(urllib.request.urlopen(ECO_URL,timeout=120).read().decode(),ECO_URL,"exec"),m)

COHORTS=tuple(m["COHORTS"])
REAL=m["REAL"]
analyze=m["analyze"]
SEEDS=list(range(202610095000,202610095320))

# Fixed orthonormal basis of the K12 sum-zero subspace.
A=np.vstack([np.eye(11),-np.ones((1,11))])
Q,_=np.linalg.qr(A)
HB=Q.T # 11 x 12

def residual_features(lines):
    out=[]
    for seq in lines:
        z=[]
        for e in seq:
            r=-np.asarray(e["p"],float).copy()
            r[int(e["y"])]+=1.0
            z.append(HB@r)
        if len(z)>=6:
            out.append(np.asarray(z,float))
    return out

def windows(lines):
    X=[];Y=[]
    for Z in residual_features(lines):
        n=len(Z)
        for t in range(3,n-2):
            X.append(Z[t-3:t].reshape(-1))
            Y.append(Z[t:t+3].reshape(-1))
    if not X:return np.empty((0,33)),np.empty((0,33))
    return np.asarray(X,float),np.asarray(Y,float)

def invsqrt(S):
    w,V=np.linalg.eigh((S+S.T)*.5)
    w=np.maximum(w,1e-8)
    return (V*(1/np.sqrt(w)))@V.T

def fit_cca(X,Y):
    mx=X.mean(0);my=Y.mean(0)
    Xc=X-mx;Yc=Y-my
    sx=LedoitWolf().fit(Xc).covariance_
    sy=LedoitWolf().fit(Yc).covariance_
    Wx=invsqrt(sx);Wy=invsqrt(sy)
    C=Xc.T@Yc/len(Xc)
    U,s,Vt=np.linalg.svd(Wx@C@Wy,full_matrices=False)
    return {"mx":mx,"my":my,"A":Wx@U,"B":Wy@Vt.T,"s":s}

def eval_rho(op,X,Y):
    xp=(X-op["mx"])@op["A"];yp=(Y-op["my"])@op["B"]
    rr=[]
    for j in range(xp.shape[1]):
        x=xp[:,j];y=yp[:,j]
        sx=float(np.std(x));sy=float(np.std(y))
        if sx<1e-12 or sy<1e-12:rr.append(0.0)
        else:rr.append(float(np.corrcoef(x,y)[0,1]))
    return np.asarray(rr,float)

def pool_xy(dataset,heldout):
    xs=[];ys=[]
    for c in COHORTS:
        if c==heldout:continue
        X,Y=windows(dataset[c]["lines"]);xs.append(X);ys.append(Y)
    return np.vstack(xs),np.vstack(ys)

# Real leave-one-ecology-out operators, frozen before nulls.
REALOPS={};REALRHO={};REALN={}
for h in COHORTS:
    X,Y=pool_xy(REAL,h)
    op=fit_cca(X,Y);REALOPS[h]=op
    Xh,Yh=windows(REAL[h]["lines"])
    REALRHO[h]=eval_rho(op,Xh,Yh)
    REALN[h]={"train_windows":len(X),"heldout_windows":len(Xh)}
    print("REAL_OP",h,json.dumps({"train_n":len(X),"heldout_n":len(Xh),
          "s":op["s"][:12].tolist(),"heldout_rho":REALRHO[h][:12].tolist()},separators=(",",":")),flush=True)

def task(seed):
    D=analyze(False,seed)
    out={}
    for h in COHORTS:
        X,Y=pool_xy(D,h)
        opn=fit_cca(X,Y)
        Xh,Yh=windows(D[h]["lines"])
        rho=eval_rho(REALOPS[h],Xh,Yh)
        out[h]={"smax":float(opn["s"][0]),"rho":rho.tolist()}
    return seed,out

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(task,SEEDS,chunksize=1))
    mp={s:v for s,v in rr}
    details={};calZ={};blindZ={};realZ={}
    select_ok=True;transfer_ok=True;all_modes=True

    for h in COHORTS:
        smax=np.array([mp[s][h]["smax"] for s in SEEDS],float)
        qsel=float(np.quantile(smax[:80],.99))
        sel_blind=float(np.mean(smax[80:160]<=qsel))
        idx=np.where(REALOPS[h]["s"]>qsel)[0]
        all_modes=all_modes and len(idx)>0
        select_ok=select_ok and sel_blind>=.90

        def T_from_rho(r):
            if len(idx)==0:return 0.0
            z=np.asarray(r,float)[idx]
            z=np.maximum(z,0.0)
            return float(np.sum(z*z))

        realT=T_from_rho(REALRHO[h])
        tcal=np.array([T_from_rho(mp[s][h]["rho"]) for s in SEEDS[160:240]],float)
        tblind=np.array([T_from_rho(mp[s][h]["rho"]) for s in SEEDS[240:320]],float)
        mu=float(tcal.mean());sd=max(float(tcal.std(ddof=1)),1e-12)
        cz=(tcal-mu)/sd;bz=(tblind-mu)/sd;rz=float((realT-mu)/sd)
        ownq=float(np.quantile(tcal,.99));bacc=float(np.mean(tblind<=ownq))
        transfer_ok=transfer_ok and bacc>=.90
        calZ[h]=cz;blindZ[h]=bz;realZ[h]=rz
        details[h]={
            "selection_q99_smax":qsel,
            "selection_blind_acceptance":sel_blind,
            "retained_dimension":int(len(idx)),
            "retained_indices":idx.tolist(),
            "train_singular_values":REALOPS[h]["s"][:12].tolist(),
            "heldout_rho":REALRHO[h][:12].tolist(),
            "real_T":realT,
            "transfer_null_mean":mu,
            "transfer_null_sd":sd,
            "transfer_real_Z":rz,
            "transfer_own_q99_T":ownq,
            "transfer_blind_acceptance":bacc,
            **REALN[h]
        }

    cmin=np.min(np.vstack([calZ[h] for h in COHORTS]),axis=0)
    bmin=np.min(np.vstack([blindZ[h] for h in COHORTS]),axis=0)
    rmin=min(realZ.values())
    q95=float(np.quantile(cmin,.95))
    bacc_global=float(np.mean(bmin<=q95))
    p=float((1+np.sum(np.r_[cmin,bmin]>=rmin))/(len(cmin)+len(bmin)+1))
    passed=bool(select_ok and all_modes and transfer_ok and bacc_global>=.90 and rmin>q95 and p<=.05)
    decision="TRANSFERABLE_RESIDUAL_PREDICTIVE_STATE_RECOVERED" if passed else "NO_UNIVERSAL_RESIDUAL_STATE_RECOVERED"

    out={"programme":"VMS-RESIDREC1","status":"complete",
         "global":{"selection_calibrated":select_ok,"all_rotations_have_modes":all_modes,
                   "transfer_calibrated":transfer_ok,"real_MIN_Z":rmin,"cal_q95_MIN_Z":q95,
                   "blind_acceptance":bacc_global,"add_one_p":p,"decision":decision},
         "rotations":details}
    print("VMS_RESIDREC1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
