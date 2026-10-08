#!/usr/bin/env python3
# STARS-CLOSE1 — preregistered 2026-10-08. No new model.
import json,os,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/01d60a0b8d058ee6f40c5a859d0215bb4d29e9a9/research/stars_temp1_calibration_20261008.py"
src=urllib.request.urlopen(URL,timeout=120).read().decode()
m={"__name__":"temp1_module"}
exec(compile(src,URL,"exec"),m)

REAL=m["REAL"];analyze=m["analyze"];generate_temp=m["generate_temp"]
SEEDS=list(range(202610089900,202610090200))

def task(seed):
    x=analyze(generate_temp(seed));return seed,x["full"].tolist()

def setup(A):
    mu=A.mean(0);S=np.cov(A,rowvar=False)
    C=.75*S+.25*np.diag(np.diag(S))+np.eye(A.shape[1])*1e-9
    return mu,np.linalg.pinv(C)

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(task,SEEDS,chunksize=2))
    mp={s:np.asarray(v,float) for s,v in rr}
    F=np.vstack([mp[s] for s in SEEDS[:150]])
    C=np.vstack([mp[s] for s in SEEDS[150:225]])
    B=np.vstack([mp[s] for s in SEEDS[225:]])
    mu,iv=setup(F)
    def d(x):
        z=x-mu;return float(z@iv@z)
    cd=np.array([d(x) for x in C]);bd=np.array([d(x) for x in B])
    rd=d(REAL["full"]);q=float(np.quantile(cd,.99));ba=float(np.mean(bd<=q))
    ref=np.r_[cd,bd];p=float((1+np.sum(ref>=rd))/(len(ref)+1))
    if ba<.90:dec="FULL_RESIDUAL_CALIBRATION_FAIL"
    elif rd<=q:dec="FULL_RESIDUAL_CLOSED_AT_E3_TEMP1_RESOLUTION"
    elif p<=.01:dec="FULL_RESIDUAL_SURVIVES_E3_TEMP1"
    else:dec="FULL_RESIDUAL_UNRESOLVED_PVALUE"
    out={"programme":"STARS-CLOSE1","status":"complete","metric_names":REAL["full_names"],
         "real_vector":REAL["full"].tolist(),"real_D2":rd,"cal_q99":q,
         "blind_acceptance":ba,"add_one_p":p,"decision":dec}
    print("STARS_CLOSE1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
