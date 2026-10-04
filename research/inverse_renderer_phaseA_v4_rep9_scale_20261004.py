#!/usr/bin/env python3
import concurrent.futures,json,os,urllib.request
import numpy as np,torch
from sklearn.metrics import adjusted_mutual_info_score
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/4adc3a655310165a6450a1a8a127ff93b1591708/research/inverse_renderer_phaseA_v4_dense_relax_20261004.py"
m={"__name__":"v4"};exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)
rep=9;seed=20264000+rep*1009
dat=m["b"]["generate_corpus"](16,4,2,3.5,4000,seed);X=m["b"]["flatten_decisions"](dat["routes"])
Xtr,Xte=X[:3200],X[3200:];ztr,zte=dat["z"][:3200],dat["z"][3200:]
def task(j):
    torch.set_num_threads(1)
    f=m["fit_dense_then_sparse"](Xtr,16,4,2,seed+220000+j*8191,torch.device("cpu"))
    ptr=f["gamma"].argmax(1);tr=float(adjusted_mutual_info_score(ztr,ptr))
    with torch.no_grad():
        E=m["b"]["emission_loglik"](Xte,torch.tensor(f["U"],dtype=torch.float32),
          torch.tensor(f["V"],dtype=torch.float32),torch.device("cpu")).numpy().astype(np.float64)
    llte,g,_=m["b"]["fb_numpy"](E,f["A"],f["pi"]);te=float(adjusted_mutual_info_score(zte,g.argmax(1)))
    return {"restart":j,"train_ll":f["ll"],"test_ll":float(llte),"train_ami":tr,"test_ami":te}
if __name__=="__main__":
    rr=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(28,os.cpu_count() or 8)) as ex:
        for z in ex.map(task,range(32)):
            rr.append(z);print("V4_SCALE_ITEM_JSON="+json.dumps(z,separators=(",",":")),flush=True)
    best=max(rr,key=lambda x:x["train_ll"]);mx=max(rr,key=lambda x:x["test_ami"])
    print("V4_SCALE_SUMMARY_JSON="+json.dumps({"selected_by_train_ll":best,"max_test_diagnostic":mx,
      "n_ge70":sum(x["test_ami"]>=.70 for x in rr)},separators=(",",":")),flush=True)
