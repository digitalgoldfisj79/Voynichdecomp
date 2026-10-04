#!/usr/bin/env python3
import concurrent.futures,json,os,urllib.request
import numpy as np,torch
from sklearn.metrics import adjusted_mutual_info_score
BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/83d96e02bf2149946bf849697e22217a410f42e4/research/inverse_renderer_recoverability_phaseA_20261004.py"
V2_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1973ce55b572753fcb57fc020d423fb5ee111ad2/research/inverse_renderer_phaseA_v2_topology_explore_20261004.py"
b={"__name__":"base"};exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),b)
v={"__name__":"v2"};exec(compile(urllib.request.urlopen(V2_URL,timeout=60).read().decode(),V2_URL,"exec"),v)
REPS=(2,6,9,18);NR=32
DATA={}
for rep in REPS:
    seed=20264000+rep*1009
    dat=b["generate_corpus"](16,4,2,3.5,4000,seed);X=b["flatten_decisions"](dat["routes"])
    DATA[rep]=(seed,X[:3200],X[3200:],dat["z"][:3200],dat["z"][3200:])

def task(arg):
    rep,j=arg;torch.set_num_threads(1)
    seed,Xtr,Xte,ztr,zte=DATA[rep]
    f=v["fit_v2"](Xtr,16,4,2,seed+100000+j*8191,torch.device("cpu"),55,20,.06,12,.03)
    ptr=f["gamma"].argmax(1);tr=float(adjusted_mutual_info_score(ztr,ptr))
    with torch.no_grad():
        E=b["emission_loglik"](Xte,torch.tensor(f["U"],dtype=torch.float32),
           torch.tensor(f["V"],dtype=torch.float32),torch.device("cpu")).numpy().astype(np.float64)
    llte,g,_=b["fb_numpy"](E,f["A"],f["pi"]);pte=g.argmax(1);te=float(adjusted_mutual_info_score(zte,pte))
    return {"replicate":rep,"restart":j,"train_ll":float(f["ll"]),"test_ll":float(llte),
            "train_ami":tr,"test_ami":te}

if __name__=="__main__":
    jobs=[(r,j) for r in REPS for j in range(NR)];allr=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(28,os.cpu_count() or 8)) as ex:
        futs=[ex.submit(task,x) for x in jobs]
        for fut in concurrent.futures.as_completed(futs):
            z=fut.result();allr.append(z);print("RESTART_SCALE_ITEM_JSON="+json.dumps(z,separators=(",",":")),flush=True)
    summaries=[]
    for rep in REPS:
        rr=[x for x in allr if x["replicate"]==rep]
        best=max(rr,key=lambda x:x["train_ll"]);oraclebest=max(rr,key=lambda x:x["test_ami"])
        s={"replicate":rep,"n":len(rr),"selected_by_train_ll":best,
           "max_test_ami_diagnostic":oraclebest["test_ami"],
           "n_test_ami_ge70":sum(x["test_ami"]>=.70 for x in rr)}
        summaries.append(s);print("RESTART_SCALE_REP_JSON="+json.dumps(s,separators=(",",":")),flush=True)
    print("RESTART_SCALE_SUMMARY_JSON="+json.dumps(summaries,separators=(",",":")),flush=True)
