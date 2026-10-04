#!/usr/bin/env python3
# Phase K3a: asymmetric F1 oracle/length screen.
# Strong ENTRY, weak token-constant ROUTE. Synthetic-only. NO P70.
import json,urllib.request
import numpy as np
K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=60).read().decode(),K0URL,"exec"),k0)

K=8;D=4;RANK=2;ENTRY=5.;N=4000
real_lens=np.array([len(x) for x in k0["routes"]],float)
real={"mean":float(real_lens.mean()),"median":float(np.median(real_lens)),
      "q90":float(np.quantile(real_lens,.9)),"q95":float(np.quantile(real_lens,.95))}

def generate(route_strength,seed):
    rng=np.random.default_rng(seed)
    A,pi=k0["source_graph"](K,D,rng)
    U,Ve0,Vr0=k0["controls"](K,RANK,1.0,rng,"F1")
    Ve=Ve0*ENTRY;Vr=Vr0*route_strength
    z=k0["hidden_seq"](A,pi,N,rng);obs=[]
    for x in z:
        for _ in range(100):
            t=k0["gen_token"](int(x),U,Ve,Vr,rng)
            if t is not None:obs.append(t);break
        else:raise RuntimeError("nontermination")
    return A,pi,U,Ve,Vr,z,obs

if __name__=="__main__":
    outs=[]
    print("K3_REAL_LENGTH_JSON="+json.dumps(real,separators=(",",":")),flush=True)
    for rs in (0.,.125,.25,.5,.75,1.0):
        reps=[]
        for seed in (20261004,20261005,20261006):
            A,pi,U,Ve,Vr,z,obs=generate(rs,seed)
            E=k0["emission"](obs,U,Ve,Vr);ll,g=k0["fb"](E,A,pi);pred=g.argmax(1)
            lens=np.array([len(x) for x in obs],float)
            q={"seed":seed,"nmi":k0["nmi"](z,pred),"ari":k0["ari"](z,pred),
               "mean_len":float(lens.mean()),"median_len":float(np.median(lens)),
               "q90":float(np.quantile(lens,.9)),"q95":float(np.quantile(lens,.95))}
            reps.append(q)
        means=np.array([x["mean_len"] for x in reps])
        rec={"entry_strength":ENTRY,"route_strength":rs,
             "median_nmi":float(np.median([x["nmi"] for x in reps])),
             "min_nmi":float(min(x["nmi"] for x in reps)),
             "max_mean_length_ratio":float(max(np.maximum(means/real["mean"],real["mean"]/means))),
             "all_mean_within_20pct":bool(np.all((means>=.8*real["mean"])&(means<=1.2*real["mean"]))),
             "reps":reps}
        print("K3_ASYM_ORACLE_JSON="+json.dumps(rec,separators=(",",":")),flush=True);outs.append(rec)
    print("SELECT_FORM_PHASEK3A_JSON="+json.dumps({"real_length":real,"panel":outs},separators=(",",":")),flush=True)
