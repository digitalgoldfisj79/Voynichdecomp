#!/usr/bin/env python3
# VMS-RESIDREC4B — preregistered 2026-10-08.
import argparse,copy,json,math,os,random,urllib.request
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch

PHASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0d17701bc711cf1aa0314564fc69e231b7ea9e5b/research/vms_residrec4_gru_phaseA_20261008.py"
BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0677011497ff31d84ebb93c2c65a67be57ae39af/research/vms_ecology1_running_residual_20261008.py"

# Import frozen Phase-A definitions without executing its cohort loop.
psrc=urllib.request.urlopen(PHASE_URL,timeout=120).read().decode()
pfx=psrc.split("\nOUT={}")[0]
pa={"__name__":"r4_defs"}
exec(compile(pfx,PHASE_URL,"exec"),pa)

# Import ECOLOGY-1 definitions without null block.
bsrc=urllib.request.urlopen(BASE_URL,timeout=120).read().decode()
bpfx=bsrc.split('if __name__=="__main__":')[0]
ba={"__name__":"eco_defs"}
exec(compile(bpfx,BASE_URL,"exec"),ba)

CO="RECIPES_FULL"
K=pa["K"]; EPS=1e-15
HPS=pa["HPS"]; fit_ctx=pa["fit_ctx"]; pa_apply_context=pa["apply_context"]
OffsetGRU=pa["OffsetGRU"]; batches=pa["batches"]; seq_fold=pa["seq_fold"]
metric_vector=ba["metric_vector"]
torch.set_num_threads(min(8,os.cpu_count() or 4))

ARCH={
0:{"H":8,"wd":1e-3,"epochs":44},
1:{"H":8,"wd":1e-4,"epochs":1},
2:{"H":16,"wd":1e-3,"epochs":32},
3:{"H":16,"wd":1e-3,"epochs":14},
4:{"H":8,"wd":1e-4,"epochs":14},
}
PHASE_BPE={
0:2.6330057200900403,
1:2.7984482402651873,
2:2.662014158438322,
3:2.885568577037255,
4:2.8356481163991476,
}
SEEDBASE=202610081900 + pa["COHORTS"].index(CO)*100000

def q1_split(lines,j):
    folds={k:[s for s in lines if seq_fold(s)==k] for k in range(5)}
    v=(j+1)%5; trks=[k for k in range(5) if k not in (j,v)]
    hp=HPS[CO][j]
    tr=[]
    for k in trks:
        source=[s for kk in trks if kk!=k for s in folds[kk]]
        model=fit_ctx(source,5)
        tr.extend(pa_apply_context(folds[k],model,hp))
    model_all=fit_ctx([s for k in trks for s in folds[k]],5)
    va=pa_apply_context(folds[v],model_all,hp)
    te=pa_apply_context(folds[j],model_all,hp)
    return tr,va,te

def train_fixed(tr,j):
    a=ARCH[j]; h=a["H"]; wd=a["wd"]; epochs=a["epochs"]
    wi=0 if wd==1e-4 else 1
    seed=SEEDBASE+j*10000+h*100+wi
    torch.manual_seed(seed); np.random.seed(seed%(2**32-1)); random.seed(seed)
    model=OffsetGRU(h)
    opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=wd)
    for epoch in range(epochs):
        model.train()
        for x,lq,y,m in batches(tr,64,True,seed+epoch*1009):
            opt.zero_grad()
            logits=model(x,lq)
            lp=torch.log_softmax(logits,dim=-1)
            loss=(-lp.gather(-1,y.unsqueeze(-1)).squeeze(-1)[m]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
    return model

def predict(model,lines):
    model.eval(); out=[]; bits=0.; n=0
    with torch.no_grad():
        for seq in lines:
            T=len(seq)
            x=torch.zeros((1,T,K),dtype=torch.float32)
            lq=torch.zeros((1,T,K),dtype=torch.float32)
            prev=int(seq[0]["prev"])
            for t,e in enumerate(seq):
                x[0,t,prev]=1.0
                p=np.asarray(e["p"],float)
                lq[0,t]=torch.from_numpy(np.log(np.maximum(p,EPS)).astype(np.float32))
                prev=int(e["y"])
            logits=model(x,lq)
            pp=torch.softmax(logits,dim=-1)[0].cpu().numpy()
            zz=[]
            for t,e in enumerate(seq):
                y=int(e["y"]); p=pp[t].astype(float)
                bits += -math.log2(max(float(p[y]),EPS)); n+=1
                ee=dict(e); ee["p"]=p; zz.append(ee)
            out.append(zz)
    return out,bits,n

def bits_q1(lines):
    z=0.;n=0
    for s in lines:
        for e in s:
            z += -math.log2(max(float(e["p"][int(e["y"])]),EPS)); n+=1
    return z,n

def analyze_lines(lines,check_real=False):
    q1bits=q3bits=0.; N=0; allq3=[]; fold=[]
    for j in range(5):
        tr,va,te=q1_split(lines,j)
        model=train_fixed(tr,j)
        q3,b3,n=predict(model,te)
        b1,n1=bits_q1(te); assert n==n1
        bpe3=b3/n
        if check_real:
            diff=abs(bpe3-PHASE_BPE[j])
            if diff>2e-5:
                raise RuntimeError(("REAL_RECONSTRUCTION_FAIL",j,bpe3,PHASE_BPE[j],diff))
        q1bits+=b1; q3bits+=b3; N+=n; allq3.extend(q3)
        fold.append({"fold":j,"n":n,"q1_bpe":b1/n,"q3_bpe":bpe3,"gain":(b1-b3)/n,
                     "H":ARCH[j]["H"],"wd":ARCH[j]["wd"],"epochs":ARCH[j]["epochs"]})
    vec,names,nm=metric_vector(allq3)
    assert nm==N
    return {"gain":(q1bits-q3bits)/N,"q1_bpe":q1bits/N,"q3_bpe":q3bits/N,
            "vector":vec.tolist(),"metric_names":names,"n":N,"folds":fold}

def real_lines():
    return pa["REAL"][CO]["lines"]

def null_lines(seed):
    # Dynamic compact-q0 null; then realization-specific position correction.
    D=ba["generate"](False,seed)
    return ba["position_adjust"](D[CO])

ap=argparse.ArgumentParser()
ap.add_argument("--mode",choices=["real","null"],required=True)
ap.add_argument("--start",type=int,default=0)
ap.add_argument("--stop",type=int,default=0)\nap.add_argument("--workers",type=int,default=1)
args=ap.parse_args()

if args.mode=="real":
    r=analyze_lines(real_lines(),True)
    print("R4B_REAL="+json.dumps(r,separators=(",",":")),flush=True)
else:
    def null_task(idx):
        torch.set_num_threads(max(1,8//max(1,args.workers)))
        seed=202610098000+idx
        r=analyze_lines(null_lines(seed),False)
        return {"idx":idx,"seed":seed,"gain":r["gain"],"vector":r["vector"],"n":r["n"]}
    indices=list(range(args.start,args.stop))
    if args.workers<=1:
        out=[]
        for idx in indices:
            print("NULL_START",idx,202610098000+idx,flush=True)
            z=null_task(idx); out.append(z)
            print("NULL_DONE",idx,json.dumps({"gain":z["gain"]},separators=(",",":")),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            out=[]
            for z in ex.map(null_task,indices,chunksize=1):
                out.append(z)
                print("NULL_DONE",z["idx"],json.dumps({"gain":z["gain"]},separators=(",",":")),flush=True)
    out=sorted(out,key=lambda z:z["idx"])
    print("R4B_NULLS="+json.dumps(out,separators=(",",":")),flush=True)
