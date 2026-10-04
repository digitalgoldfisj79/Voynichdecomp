#!/usr/bin/env python3
import json,urllib.request,torch
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/a149e5372d892423e1354eb2ab6062bd4a933a20/research/inverse_renderer_recoverability_phaseA_v3_20261004.py"
m={"__name__":"v3"}
exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
dev=torch.device("cpu")
for s in (3.0,4.0,5.0):
    d=m["generate"](16,4,2,s,4000,20261004)
    X=m["flatten_decisions"](d["routes"])
    o=m["oracle"](d,X,dev)
    print("ORACLE_ONLY_JSON="+json.dumps({"strength":s,**o},separators=(",",":")),flush=True)
