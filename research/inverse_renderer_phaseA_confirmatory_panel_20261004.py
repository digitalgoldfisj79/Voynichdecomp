#!/usr/bin/env python3
import concurrent.futures,json,math,os,urllib.request
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score

BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/83d96e02bf2149946bf849697e22217a410f42e4/research/inverse_renderer_recoverability_phaseA_20261004.py"
V2_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1973ce55b572753fcb57fc020d423fb5ee111ad2/research/inverse_renderer_phaseA_v2_topology_explore_20261004.py"
b={"__name__":"base"};exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),b)
v={"__name__":"v2"};exec(compile(urllib.request.urlopen(V2_URL,timeout=60).read().decode(),V2_URL,"exec"),v)

K=16;D=4;R=2;STRENGTH=3.5;N=4000;NREP=20;RESTARTS=4

def align_map(true,pred,K):
    C=np.zeros((K,K),dtype=np.int64)
    np.add.at(C,(np.asarray(true,int),np.asarray(pred,int)),1)
    rr,cc=linear_sum_assignment(-C)
    # predicted -> true
    mp={int(c):int(r) for r,c in zip(rr,cc)}
    return mp

def edge_f1_aligned(Atrue,Ahat,mp,d):
    K=Atrue.shape[0]
    inv={tr:pr for pr,tr in mp.items()}
    Ah=np.zeros_like(Ahat)
    for ti in range(K):
        pi=inv.get(ti,ti)
        for tj in range(K):
            pj=inv.get(tj,tj)
            Ah[ti,tj]=Ahat[pi,pj]
    tp=fp=fn=0
    for i in range(K):
        t=set(np.argsort(Atrue[i])[-d:])
        h=set(np.argsort(Ah[i])[-d:])
        tp+=len(t&h);fp+=len(h-t);fn+=len(t-h)
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    return float(2*p*r/max(p+r,1e-15))

def state_emissions(U,V):
    # actual normalized FORM decision distributions by source state/context.
    bias=np.einsum("kr,rco->kco",U,V)
    L=b["LOGP0"][None,:,:]+bias
    L[:,0,b["END"]]=-1e30
    m=L.max(2,keepdims=True);q=np.exp(L-m);q/=q.sum(2,keepdims=True)
    return q

def emission_cosine(datU,datV,hatU,hatV,mp):
    qt=state_emissions(datU,datV).reshape(K,-1)
    qh=state_emissions(hatU,hatV).reshape(K,-1)
    vals=[]
    for pr,tr in mp.items():
        a=qt[tr];c=qh[pr]
        vals.append(float(np.dot(a,c)/(np.linalg.norm(a)*np.linalg.norm(c)+1e-15)))
    return float(np.mean(vals))

def one_rep(rep):
    torch.set_num_threads(1)
    seed=20264000+rep*1009
    dat=b["generate_corpus"](K,D,R,STRENGTH,N,seed)
    X=b["flatten_decisions"](dat["routes"])
    ntr=int(.8*N);Xtr=X[:ntr];Xte=X[ntr:];ztr=dat["z"][:ntr];zte=dat["z"][ntr:]
    runs=[]
    for j in range(RESTARTS):
        f=v["fit_v2"](Xtr,K,D,R,seed+50000+j*8191,torch.device("cpu"),55,20,.06,12,.03)
        ptr=f["gamma"].argmax(1)
        runs.append({"fit":f,"ptr":ptr,"train_ami":float(adjusted_mutual_info_score(ztr,ptr))})
    runs.sort(key=lambda x:x["fit"]["ll"],reverse=True)
    best=runs[0];second=runs[1]
    f=best["fit"]
    with torch.no_grad():
        U=torch.tensor(f["U"],dtype=torch.float32)
        V=torch.tensor(f["V"],dtype=torch.float32)
        Ete=b["emission_loglik"](Xte,U,V,torch.device("cpu")).numpy().astype(np.float64)
    llte,gte,_=b["fb_numpy"](Ete,f["A"],f["pi"])
    pte=gte.argmax(1)
    test_ami=float(adjusted_mutual_info_score(zte,pte))
    pair_ami=float(adjusted_mutual_info_score(best["ptr"],second["ptr"]))
    mp=align_map(ztr,best["ptr"],K)
    ef=edge_f1_aligned(dat["A"],f["A"],mp,D)
    ec=emission_cosine(dat["U"],dat["V"],f["U"],f["V"],mp)
    rec={
      "replicate":rep,"seed":seed,
      "best_train_ll":float(f["ll"]),"test_ll":float(llte),
      "best_train_ami":best["train_ami"],"test_ami":test_ami,
      "top2_pairwise_train_ami":pair_ami,
      "aligned_edge_f1":ef,"aligned_emission_cosine":ec,
      "pass":bool(test_ami>=.70 and pair_ami>=.60),
      "restart_train_ami":[float(x["train_ami"]) for x in runs],
      "restart_train_ll":[float(x["fit"]["ll"]) for x in runs]
    }
    return rec

if __name__=="__main__":
    out=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(NREP,os.cpu_count() or 8)) as ex:
        futs={ex.submit(one_rep,i):i for i in range(NREP)}
        for fut in concurrent.futures.as_completed(futs):
            rec=fut.result();out.append(rec)
            print("PHASEA_PANEL_REPLICATE_JSON="+json.dumps(rec,separators=(",",":")),flush=True)
    out.sort(key=lambda x:x["replicate"])
    passes=sum(x["pass"] for x in out)
    summary={
      "architecture":{"K":K,"d":D,"rank":R,"strength":STRENGTH,"N":N,"replicates":NREP,"restarts":RESTARTS},
      "passes":passes,"pass_fraction":passes/NREP,
      "median_test_ami":float(np.median([x["test_ami"] for x in out])),
      "median_top2_pairwise_ami":float(np.median([x["top2_pairwise_train_ami"] for x in out])),
      "median_edge_f1":float(np.median([x["aligned_edge_f1"] for x in out])),
      "median_emission_cosine":float(np.median([x["aligned_emission_cosine"] for x in out])),
      "gate_pass":bool(passes>=18 and np.median([x["test_ami"] for x in out])>=.70),
      "replicates_detail":out
    }
    print("PHASEA_CONFIRMATORY_PANEL_JSON="+json.dumps(summary,separators=(",",":")),flush=True)
