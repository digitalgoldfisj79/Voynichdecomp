#!/usr/bin/env python3
import json,urllib.request,numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5d46189f00e3a3fa4db3eeb9a1331b48890c8ae7/research/inverse_renderer_piece_oracle_diagnostic_20261004.py"
m={"__name__":"piece"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
m["BASE_MIX"]=.02;m["BIAS_BOUND"]=5.0
A,pi,U,V,z,routes=m["generate"](16,4,2,5.0,4000,20261004);X=m["flatten"](routes)
P0=m["P0"];ctxn=X.sum(2,keepdims=True);R=X-ctxn*P0[None,:,:]
W=(R/np.sqrt(P0[None,:,:]+.01)).reshape(len(X),-1).astype(np.float64)
W-=W.mean(0,keepdims=True)
# Lagged cross-covariance. Independent emission noise does not contribute systematically across time.
C=(W[:-1].T@W[1:])/(len(W)-1)
Ucv,S,Vt=randomized_svd(C,n_components=24,n_iter=7,random_state=20261004)
out=[]
for r in (2,4,8,16,24):
    Z=np.concatenate([W@Ucv[:,:r],W@Vt[:r].T],axis=1)
    Z=StandardScaler().fit_transform(Z)
    lab=KMeans(16,n_init=30,random_state=20261004,max_iter=500).fit_predict(Z)
    rec={"method":"crosscov_kmeans","rank":r,"nmi":m["nmi"](z,lab),"ari":m["ari"](z,lab)}
    out.append(rec);print("SPECTRAL_INIT_JSON="+json.dumps(rec,separators=(",",":")),flush=True)
    # Sequence-aware HMM on the predictive spectral coordinates.
    for seed in range(4):
        h=GaussianHMM(16,covariance_type="diag",n_iter=180,tol=1e-3,random_state=20261004+seed,min_covar=1e-3)
        try:
            h.fit(Z);pred=h.predict(Z)
            rr={"method":"crosscov_hmm","rank":r,"seed":seed,"score_per_token":float(h.score(Z)/len(Z)),
                "nmi":m["nmi"](z,pred),"ari":m["ari"](z,pred)}
        except Exception as e:rr={"method":"crosscov_hmm","rank":r,"seed":seed,"error":str(e)}
        out.append(rr);print("SPECTRAL_INIT_JSON="+json.dumps(rr,separators=(",",":")),flush=True)
valid=[x for x in out if "nmi" in x]
print("SPECTRAL_INIT_DIAGNOSTIC_JSON="+json.dumps({"singular_values":S.tolist(),"best":sorted(valid,key=lambda x:x["nmi"],reverse=True)[:12]},separators=(",",":")),flush=True)
