#!/usr/bin/env python3
import json,urllib.request,numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5d46189f00e3a3fa4db3eeb9a1331b48890c8ae7/research/inverse_renderer_piece_oracle_diagnostic_20261004.py"
m={"__name__":"piece"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
m["BASE_MIX"]=.02;m["BIAS_BOUND"]=5.0
A,pi,U,V,z,routes=m["generate"](16,4,2,5.0,4000,20261004);X=m["flatten"](routes)
P0=m["P0"];ctxn=X.sum(2,keepdims=True);R=X-ctxn*P0[None,:,:];W=R/np.sqrt(P0[None,:,:]+.01)
F=W.reshape(len(X),-1)
out=[]
for nc in (8,16,32):
    Z=TruncatedSVD(n_components=nc,random_state=20261004).fit_transform(F)
    Z=StandardScaler().fit_transform(Z)
    for seed in range(12):
        h=GaussianHMM(n_components=16,covariance_type="diag",n_iter=150,tol=1e-3,
                      random_state=20261004+seed,min_covar=1e-3,params="stmc",init_params="stmc")
        try:
            h.fit(Z);pred=h.predict(Z);score=float(h.score(Z))
            rec={"components":nc,"seed":seed,"score_per_token":score/len(Z),
                 "nmi":m["nmi"](z,pred),"ari":m["ari"](z,pred),"converged":bool(h.monitor_.converged)}
        except Exception as e:
            rec={"components":nc,"seed":seed,"error":str(e)}
        out.append(rec);print("HMM_INIT_JSON="+json.dumps(rec,separators=(",",":")),flush=True)
valid=[x for x in out if "nmi" in x]
best_score=sorted(valid,key=lambda x:x["score_per_token"],reverse=True)[:10]
best_nmi=sorted(valid,key=lambda x:x["nmi"],reverse=True)[:10]
print("HMM_INIT_DIAGNOSTIC_JSON="+json.dumps({"best_by_score":best_score,"best_by_nmi":best_nmi},separators=(",",":")),flush=True)
