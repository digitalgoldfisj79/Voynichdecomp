#!/usr/bin/env python3
# Phase A v5: successive-halving variational restarts + exact-likelihood selection + consensus.
# Engineering canary K=8, strength=3.0. NO P70.
import json,urllib.request
import numpy as np, torch
from scipy.optimize import linear_sum_assignment

URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/274791e3e0a5642a9344333e3987e17d654c84c0/research/inverse_renderer_recoverability_phaseA_v4_20261004.py"
m={"__name__":"v4base"}
exec(compile(urllib.request.urlopen(URL,timeout=60).read().decode(),URL,"exec"),m)

K=8;D=4;R=2;S=3.0;N=4000;SEED=20261004
N_SCREEN=24;N_REFINE=6
device=torch.device("cpu")
dat=m["generate"](K,D,R,S,N,SEED);X=m["flatten_decisions"](dat["routes"]);orc=m["oracle"](dat,X,device)

screen=[]
for i in range(N_SCREEN):
    vi=m["variational_init"](X,K,R,SEED+1000*i,device,350,.045)
    pred=vi["q"].argmax(1)
    z={"i":i,"score":float(vi["score"]),"nmi":m["nmi"](dat["z"],pred),"ari":m["ari"](dat["z"],pred),
       "seconds":vi["seconds"],"vi":vi}
    screen.append(z)
    print("V5_SCREEN_JSON="+json.dumps({k:v for k,v in z.items() if k!="vi"},separators=(",",":")),flush=True)

screen.sort(key=lambda x:x["score"],reverse=True)
refined=[]
for rank,z in enumerate(screen[:N_REFINE]):
    vi=z["vi"]
    fit=m["refine"](X,vi["A"],vi["pi"],vi["U"],vi["V"],D,device,65,18,.035,18)
    ll,A,pi,U,V,g,ep,sec=fit;pred=g.argmax(1)
    q={"screen_rank":rank+1,"screen_i":z["i"],"screen_score":z["score"],"screen_nmi":z["nmi"],
       "ll":float(ll),"nmi":m["nmi"](dat["z"],pred),"ari":m["ari"](dat["z"],pred),
       "epochs":ep,"seconds":sec,"g":g}
    refined.append(q)
    print("V5_REFINE_JSON="+json.dumps({k:v for k,v in q.items() if k!="g"},separators=(",",":")),flush=True)

refined.sort(key=lambda x:x["ll"],reverse=True)
ref=refined[0]["g"];aligned=[]
stabs=[]
for q in refined:
    g=q["g"];over=ref.T@g
    rr,cc=linear_sum_assignment(-over)
    ga=np.zeros_like(g)
    for r,c in zip(rr,cc):ga[:,r]=g[:,c]
    aligned.append(ga)
    stabs.append(m["nmi"](ref.argmax(1),ga.argmax(1)))
cons=np.mean(aligned[:min(3,len(aligned))],axis=0);cp=cons.argmax(1)
best=refined[0]
out={
 "phase":"A_v5_successive_halving","K":K,"d":D,"rank":R,"strength":S,"N":N,
 "oracle":orc,
 "screen_n":N_SCREEN,"refine_n":N_REFINE,
 "screen_top":[{k:v for k,v in z.items() if k!="vi"} for z in screen[:6]],
 "refined":[{k:v for k,v in q.items() if k!="g"} for q in refined],
 "best_likelihood":{"ll":best["ll"],"nmi":best["nmi"],"ari":best["ari"]},
 "restart_stability_to_best":stabs,
 "top3_consensus":{"nmi":m["nmi"](dat["z"],cp),"ari":m["ari"](dat["z"],cp)},
 "gate70_best":bool(best["nmi"]>=.70),
 "gate70_consensus":bool(m["nmi"](dat["z"],cp)>=.70)
}
print("INVERSE_PHASEA_V5_JSON="+json.dumps(out,separators=(",",":")),flush=True)
