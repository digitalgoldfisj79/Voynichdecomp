#!/usr/bin/env python3
import json,urllib.request,torch
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/274791e3e0a5642a9344333e3987e17d654c84c0/research/inverse_renderer_recoverability_phaseA_v4_20261004.py"
m={"__name__":"v4"}
exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
dev=torch.device("cpu")
for K in (8,16):
  for s in (2.0,2.5,3.0):
    d=m["generate"](K,2,2,s,4000,20261004+K);X=m["flatten_decisions"](d["routes"]);o=m["oracle"](d,X,dev)
    print("SPARSE_ORACLE_JSON="+json.dumps({"K":K,"d":2,"strength":s,**o},separators=(",",":")),flush=True)
