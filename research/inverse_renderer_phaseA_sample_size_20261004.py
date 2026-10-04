#!/usr/bin/env python3
# Phase A sample-size diagnostic: same K8 source shape, increasing N.
import json,urllib.request,torch,numpy as np
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/274791e3e0a5642a9344333e3987e17d654c84c0/research/inverse_renderer_recoverability_phaseA_v4_20261004.py"
m={"__name__":"v4"}
exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
dev=torch.device("cpu");K=8;D=4;R=2;S=3.0;SEED=20261004
out=[]
for N in (4000,12000,34087):
    dat=m["generate"](K,D,R,S,N,SEED);X=m["flatten_decisions"](dat["routes"]);orc=m["oracle"](dat,X,dev)
    runs=[]
    for r in range(3):
        vi=m["variational_init"](X,K,R,SEED+1000*r,dev,350,.045);p=vi["q"].argmax(1)
        z={"restart":r,"score":vi["score"],"nmi":m["nmi"](dat["z"],p),"ari":m["ari"](dat["z"],p),"seconds":vi["seconds"]}
        runs.append(z);print("NSCALE_RESTART_JSON="+json.dumps({"N":N,**z},separators=(",",":")),flush=True)
    q={"N":N,"oracle":orc,"median_var_nmi":float(np.median([x["nmi"] for x in runs])),
       "best_var_nmi":max(x["nmi"] for x in runs),"runs":runs}
    out.append(q);print("NSCALE_JSON="+json.dumps(q,separators=(",",":")),flush=True)
print("INVERSE_NSCALE_JSON="+json.dumps(out,separators=(",",":")),flush=True)
