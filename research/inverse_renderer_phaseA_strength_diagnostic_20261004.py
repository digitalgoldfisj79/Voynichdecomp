#!/usr/bin/env python3
import json,urllib.request,types,time
import numpy as np, torch
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/664cd58a01a0d26bf66619fafc92945bee1ffe75/research/inverse_renderer_recoverability_phaseA_20261004.py"
m={"__name__":"phaseA"}
exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)

def oracle(dat,X,device):
    U=torch.tensor(dat["U"],dtype=torch.float32,device=device)
    V=torch.tensor(dat["V"],dtype=torch.float32,device=device)
    with torch.no_grad():
        E=m["emission_loglik"](X,U,V,device).cpu().numpy().astype(np.float64)
    ll,g,_=m["fb_numpy"](E,dat["A"],dat["pi"])
    p=g.argmax(1)
    return {"ll":float(ll),"nmi":m["nmi"](dat["z"],p),"ari":m["ari"](dat["z"],p)}

device=torch.device("cpu")
out=[]
for strength in (.5,.85,1.25,1.75,2.5,3.5):
    dat=m["generate_corpus"](16,4,2,strength,4000,20261004)
    X=m["flatten_decisions"](dat["routes"])
    orc=oracle(dat,X,device)
    fits=[]
    for r in range(2):
        f=m["fit_one"](X,16,4,2,20262000+int(strength*100)+r*991,device,45,20,.06)
        pred=f["gamma"].argmax(1)
        fits.append({"ll":f["ll"],"nmi":m["nmi"](dat["z"],pred),"ari":m["ari"](dat["z"],pred),
                     "epochs":f["epochs"],"seconds":f["seconds"]})
    z={"strength":strength,"oracle":orc,"fits":fits,"best_nmi":max(x["nmi"] for x in fits)}
    print("STRENGTH_DIAGNOSTIC_JSON="+json.dumps(z,separators=(",",":")),flush=True);out.append(z)
print("PHASEA_DIAGNOSTIC_JSON="+json.dumps(out,separators=(",",":")),flush=True)
