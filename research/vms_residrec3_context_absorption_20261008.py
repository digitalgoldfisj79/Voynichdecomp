#!/usr/bin/env python3
# VMS-RESIDREC3 — preregistered 2026-10-08.
import json,math,os,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

PHASEA_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/9ea2a0d87af8b8a9dfb57acbef77acaee0a3dfd1/research/vms_residrec2_context_vs_hmm_phaseA_20261008.py"
src=urllib.request.urlopen(PHASEA_URL,timeout=120).read().decode()
prefix=src.split("\nimport os\n")[0]
pa={"__name__":"residrec2_defs"}
exec(compile(prefix,PHASEA_URL,"exec"),pa)
base=pa["ns"]

COHORTS=pa["COHORTS"]; REAL=pa["REAL"]
fit_ctx=pa["fit_ctx"]; ctx_prob=pa["ctx_prob"]; pool_prob=pa["pool_prob"]
metric_vector=base["metric_vector"]; analyze=base["analyze"]

HPS={
"HERBAL_A":{0:(2,32.0,.25),1:(1,8.0,.25),2:(5,32.0,.25),3:(1,32.0,.25),4:(5,32.0,.25)},
"HERBAL_B":{0:(2,32.0,.25),1:(5,32.0,.25),2:(2,32.0,.25),3:(1,2.0,.25),4:(5,32.0,.25)},
"BALNEO_FULL":{0:(2,32.0,.25),1:(1,32.0,.25),2:(1,32.0,.25),3:(1,32.0,.25),4:(2,32.0,.25)},
"RECIPES_FULL":{0:(2,32.0,.25),1:(5,32.0,.25),2:(1,32.0,.25),3:(2,32.0,.25),4:(3,32.0,.25)}
}
SEEDS=list(range(202610097000,202610097300))
EPS=1e-15

def ffold(seq): return int(seq[0]["fold"])

def apply_m1(lines,cohort):
    folds={j:[s for s in lines if ffold(s)==j] for j in range(5)}
    out=[]; b0=b1=0.; nall=0
    for j in range(5):
        v=(j+1)%5
        tr=[s for k in range(5) if k not in (j,v) for s in folds[k]]
        te=folds[j]
        model=fit_ctx(tr,5); L,a,lam=HPS[cohort][j]
        for seq in te:
            if not seq: continue
            hist=[int(seq[0]["prev"])]; zz=[]
            for e in seq:
                pc=ctx_prob(hist,model[0],model[1],L,a)
                p1=pool_prob(np.asarray(e["p"],float),pc,lam)
                y=int(e["y"])
                b0 += -math.log2(max(float(e["p"][y]),EPS))
                b1 += -math.log2(max(float(p1[y]),EPS))
                x=dict(e); x["p"]=p1; zz.append(x)
                hist.append(y); nall+=1
            out.append(zz)
    return out,(b0-b1)/max(nall,1),nall

REALOUT={}
for c in COHORTS:
    q1,g,n=apply_m1(REAL[c]["lines"],c)
    vec,names,ne=metric_vector(q1)
    REALOUT[c]={"vector":vec,"gain":g,"n":n,"names":names}
print("REAL_M1",json.dumps({c:{"gain":REALOUT[c]["gain"],"n":REALOUT[c]["n"],
      "vector":REALOUT[c]["vector"].tolist()} for c in COHORTS},separators=(",",":")),flush=True)

def task(seed):
    D=analyze(False,seed)
    o={}
    for c in COHORTS:
        q1,g,n=apply_m1(D[c]["lines"],c)
        v,_,_=metric_vector(q1)
        o[c]={"vector":v.tolist(),"gain":g}
    return seed,o

def setup(A):
    mu=A.mean(0); S=np.cov(A,rowvar=False)
    C=.75*S+.25*np.diag(np.diag(S))+np.eye(A.shape[1])*1e-9
    return mu,np.linalg.pinv(C)
def d2(x,mu,iv):
    z=x-mu; return float(z@iv@z)

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(task,SEEDS,chunksize=2))
    mp={s:v for s,v in rr}
    details={}; calZ={}; blindZ={}; gainCalZ={}; gainBlindZ={}; realZ={}; realGZ={}
    for c in COHORTS:
        V=np.vstack([np.asarray(mp[s][c]["vector"],float) for s in SEEDS])
        G=np.array([float(mp[s][c]["gain"]) for s in SEEDS],float)
        FIT=V[:120]; SCALE=V[120:180]; CAL=V[180:240]; BLIND=V[240:300]
        mu,iv=setup(FIT)
        sd=np.array([d2(x,mu,iv) for x in SCALE])
        cd=np.array([d2(x,mu,iv) for x in CAL]); bd=np.array([d2(x,mu,iv) for x in BLIND])
        sm=float(sd.mean()); ss=max(float(sd.std(ddof=1)),1e-12)
        cz=(cd-sm)/ss; bz=(bd-sm)/ss
        rd=d2(REALOUT[c]["vector"],mu,iv); rz=(rd-sm)/ss
        calZ[c]=cz; blindZ[c]=bz; realZ[c]=float(rz)

        # gain clarification: 0:120 scale, 120:180 calibration, 240:300 blind
        gm=float(G[:120].mean()); gs=max(float(G[:120].std(ddof=1)),1e-12)
        gcz=(G[120:180]-gm)/gs; gbz=(G[240:300]-gm)/gs
        rgz=(REALOUT[c]["gain"]-gm)/gs
        gainCalZ[c]=gcz; gainBlindZ[c]=gbz; realGZ[c]=float(rgz)

        ownq=float(np.quantile(cz,.99))
        details[c]={"real_D2":rd,"real_Z_D2":float(rz),"own_q99_Z_D2":ownq,
                    "absorption":"CONTEXT_ABSORBS" if rz<=ownq else "RESIDUAL_SURVIVES_CONTEXT",
                    "real_gain":REALOUT[c]["gain"],"gain_null_mean":gm,"gain_null_sd":gs,
                    "real_Z_gain":float(rgz),"gain_own_q99_Z":float(np.quantile(gcz,.99)),
                    "real_vector":REALOUT[c]["vector"].tolist()}

    cmax=np.max(np.vstack([calZ[c] for c in COHORTS]),axis=0)
    bmax=np.max(np.vstack([blindZ[c] for c in COHORTS]),axis=0)
    rmax=max(realZ.values()); q99=float(np.quantile(cmax,.99))
    blind=float(np.mean(bmax<=q99))
    global_abs=bool(blind>=.90 and rmax<=q99)

    gcmax=np.max(np.vstack([gainCalZ[c] for c in COHORTS]),axis=0)
    gbmax=np.max(np.vstack([gainBlindZ[c] for c in COHORTS]),axis=0)
    rgmax=max(realGZ.values()); gq99=float(np.quantile(gcmax,.99))
    gblind=float(np.mean(gbmax<=gq99))
    gp=float((1+np.sum(np.r_[gcmax,gbmax]>=rgmax))/(len(gcmax)+len(gbmax)+1))
    gainpass=bool(gblind>=.90 and rgmax>gq99 and gp<=.01)

    out={"programme":"VMS-RESIDREC3","status":"complete",
         "residual_familywise":{"real_max_Z":rmax,"q99_max_Z":q99,"blind_acceptance":blind,
             "decision":"GLOBAL_CONTEXT_ABSORPTION" if global_abs else "RESIDUAL_SURVIVES_CONTEXT_GLOBALLY"},
         "gain_familywise":{"real_max_Z":rgmax,"q99_max_Z":gq99,"blind_acceptance":gblind,
             "add_one_p":gp,"qualified":gainpass},
         "ecologies":details}
    print("VMS_RESIDREC3_JSON="+json.dumps(out,separators=(",",":")),flush=True)
