#!/usr/bin/env python3
import json,urllib.request,numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5d46189f00e3a3fa4db3eeb9a1331b48890c8ae7/research/inverse_renderer_piece_oracle_diagnostic_20261004.py"
m={"__name__":"piece"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
m["BASE_MIX"]=.02;m["BIAS_BOUND"]=5.0
A,pi,U,V,z,routes=m["generate"](16,4,2,5.0,4000,20261004);X=m["flatten"](routes)
P0=m["P0"];N=len(X)
ctxn=X.sum(2,keepdims=True)
R=X-ctxn*P0[None,:,:]
W=R/np.sqrt(P0[None,:,:]+.01)
families={"raw":X.reshape(N,-1),"resid":R.reshape(N,-1),"white":W.reshape(N,-1)}

def roll(Z,w):
    if w<=1:return Z
    h=w//2;cs=np.vstack([np.zeros((1,Z.shape[1])),np.cumsum(Z,axis=0)])
    out=np.empty_like(Z)
    for i in range(len(Z)):
        a=max(0,i-h);b=min(len(Z),i+h+1);out[i]=(cs[b]-cs[a])/(b-a)
    return out

def cluster(Z):
    Z=StandardScaler().fit_transform(Z)
    return KMeans(n_clusters=16,n_init=20,random_state=20261004,max_iter=400).fit_predict(Z)

out=[]
for name,F in families.items():
    for nc in (2,4,8,16,32):
        Z=TruncatedSVD(n_components=nc,random_state=20261004).fit_transform(F)
        for w in (1,3,5):
            L=cluster(roll(Z,w))
            rec={"family":name,"components":nc,"window":w,"nmi":m["nmi"](z,L),"ari":m["ari"](z,L)}
            out.append(rec);print("INIT_JSON="+json.dumps(rec,separators=(",",":")),flush=True)

Z=TruncatedSVD(n_components=8,random_state=20261004).fit_transform(families["white"])
for radius in (1,2,3):
    parts=[]
    for off in range(-radius,radius+1):
        idx=np.clip(np.arange(N)+off,0,N-1);parts.append(Z[idx])
    L=cluster(np.concatenate(parts,axis=1))
    rec={"family":"white_neighbor_concat","radius":radius,"nmi":m["nmi"](z,L),"ari":m["ari"](z,L)}
    out.append(rec);print("INIT_JSON="+json.dumps(rec,separators=(",",":")),flush=True)

print("INIT_DIAGNOSTIC_JSON="+json.dumps({"best":sorted(out,key=lambda x:x["nmi"],reverse=True)[:10]},separators=(",",":")),flush=True)
