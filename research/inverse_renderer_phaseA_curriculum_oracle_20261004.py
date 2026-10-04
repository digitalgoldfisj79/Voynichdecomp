#!/usr/bin/env python3
import json,urllib.request,torch
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/a149e5372d892423e1354eb2ab6062bd4a933a20/research/inverse_renderer_recoverability_phaseA_v3_20261004.py"
m={"__name__":"v3"}
exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
dev=torch.device("cpu")
for K in (4,8,16):
    for s in (1.5,2.0,2.5,3.0):
        d=m["generate"](K,min(4,K),2,s,4000,20261004+K)
        X=m["flatten_decisions"](d["routes"]);o=m["oracle"](d,X,dev)
        print("CURRICULUM_ORACLE_JSON="+json.dumps({"K":K,"strength":s,**o},separators=(",",":")),flush=True)
