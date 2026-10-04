#!/usr/bin/env python3
import json,urllib.request
import numpy as np, torch
from sklearn.metrics import adjusted_mutual_info_score
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/83d96e02bf2149946bf849697e22217a410f42e4/research/inverse_renderer_recoverability_phaseA_20261004.py"
b={"__name__":"base"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),b)
out=[]
for rep in range(20):
    seed=20264000+rep*1009
    dat=b["generate_corpus"](16,4,2,3.5,4000,seed);X=b["flatten_decisions"](dat["routes"])
    ntr=3200
    for split,name in ((slice(0,ntr),"train"),(slice(ntr,None),"test")):
        U=torch.tensor(dat["U"],dtype=torch.float32);V=torch.tensor(dat["V"],dtype=torch.float32)
        with torch.no_grad():E=b["emission_loglik"](X[split],U,V,torch.device("cpu")).numpy().astype(np.float64)
        ll,g,_=b["fb_numpy"](E,dat["A"],dat["pi"]);pred=g.argmax(1)
        ami=float(adjusted_mutual_info_score(dat["z"][split],pred))
        if name=="train":tr=ami
        else:te=ami
    occ=np.bincount(dat["z"],minlength=16)/4000
    rec={"replicate":rep,"seed":seed,"oracle_train_ami":tr,"oracle_test_ami":te,
         "min_state_share":float(occ.min()),"max_state_share":float(occ.max())}
    print("PHASEA_ORACLE_PANEL_JSON="+json.dumps(rec,separators=(",",":")),flush=True);out.append(rec)
print("PHASEA_ORACLE_SUMMARY_JSON="+json.dumps({
 "median_test_ami":float(np.median([x["oracle_test_ami"] for x in out])),
 "n_test_ami_ge70":int(sum(x["oracle_test_ami"]>=.70 for x in out)),
 "min_test_ami":float(min(x["oracle_test_ami"] for x in out)),
 "max_test_ami":float(max(x["oracle_test_ami"] for x in out))
},separators=(",",":")),flush=True)
