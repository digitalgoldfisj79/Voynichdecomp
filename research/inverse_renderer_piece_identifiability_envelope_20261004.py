#!/usr/bin/env python3
import json,urllib.request,numpy as np
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/5d46189f00e3a3fa4db3eeb9a1331b48890c8ae7/research/inverse_renderer_piece_oracle_diagnostic_20261004.py"
m={"__name__":"piece"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)

def one(K,d,rank,strength,mix,bound,seeds=(20261004,20261005)):
    m["BASE_MIX"]=mix;m["BIAS_BOUND"]=bound
    reps=[]
    for seed in seeds:
        A,pi,U,V,z,routes=m["generate"](K,d,rank,strength,4000,seed)
        X=m["flatten"](routes);E=m["emission"](X,U,V);ll,g,_=m["m"]["fb_numpy"](E,A,pi);p=g.argmax(1)
        reps.append({"seed":seed,"nmi":m["nmi"](z,p),"ari":m["ari"](z,p),
                     "mean_len":float(np.mean(list(map(len,routes)))),"max_len":int(max(map(len,routes)))})
    return {"K":K,"d":d,"rank":rank,"strength":strength,"mix":mix,"bound":bound,
            "median_nmi":float(np.median([r["nmi"] for r in reps])),
            "min_nmi":float(min(r["nmi"] for r in reps)),"reps":reps}

out=[]
# State-count envelope under conservative channel.
for K,d in ((4,4),(8,4),(16,4)):
    x=one(K,d,2,3.0,.10,3.0);print("ENVELOPE_JSON="+json.dumps(x,separators=(",",":")),flush=True);out.append(x)
# Stronger but still renderer-constrained K16 channel.
for mix,bound,strength in ((.05,4.,4.),(.02,5.,5.),(.00,5.,5.),(.02,8.,8.),(.00,8.,8.)):
    x=one(16,4,2,strength,mix,bound);print("ENVELOPE_JSON="+json.dumps(x,separators=(",",":")),flush=True);out.append(x)
print("PIECE_IDENTIFIABILITY_ENVELOPE_JSON="+json.dumps(out,separators=(",",":")),flush=True)
