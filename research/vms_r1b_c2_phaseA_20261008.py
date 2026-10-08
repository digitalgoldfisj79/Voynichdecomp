#!/usr/bin/env python3
# VMS R1B Phase A — C2 real-data test, frozen 2026-10-08.
import argparse,copy,json,math,os,random,urllib.request
import numpy as np
import torch
import torch.nn as nn

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/b02193969cb42745ee9cf3cea4985fa7664602b2/research/vms_r1_channel_core_20261008.py"
CAL_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/36740cd684ed22dba02acf33f3bb1109c19bfc54/research/data/residrec4b_residual15_calibration_20261008.json"

src=urllib.request.urlopen(CORE_URL,timeout=120).read().decode()
core={"__name__":"r1core"}
exec(compile(src,CORE_URL,"exec"),core)
r4=core["r4"]
K=core["K"]; EPS=1e-15
CAL=json.loads(urllib.request.urlopen(CAL_URL,timeout=120).read().decode())
MU=np.asarray(CAL["mean"],float); IV=np.asarray(CAL["inv_cov"],float)
SM=float(CAL["scale_mean"]); SS=float(CAL["scale_sd"])
PARENT_D2=161.34337398354887

ARCH={
0:{"H":8,"wd":1e-3,"epochs":44},
1:{"H":8,"wd":1e-4,"epochs":1},
2:{"H":16,"wd":1e-3,"epochs":32},
3:{"H":16,"wd":1e-3,"epochs":14},
4:{"H":8,"wd":1e-4,"epochs":14},
}
SEEDBASE=202610081900 + r4["pa"]["COHORTS"].index("RECIPES_FULL")*100000 if "pa" in r4 else 202610381900
# exact known Recipes seed base from RESIDREC4B:
SEEDBASE=202610081900 + 3*100000

torch.set_num_threads(min(8,os.cpu_count() or 4))

def d2(vec):
    z=np.asarray(vec,float)-MU
    return float(z@IV@z)

def z_d2(v):
    return (d2(v)-SM)/SS

def bits_probs(P,Y):
    return float(-np.log2(np.maximum(P[np.arange(len(Y)),Y],EPS)).sum())

def role(t,T):
    if t==0:return 0
    if t==T-1:return 2
    return 1

def c1c2(e):
    c1=np.zeros(12,float); c1[int(e["prev_final_class"])]=1.
    c2=np.zeros(8,float)
    c2[int(e["prev_piece_bin"])]=1.
    c2[4+int(e["prev_char_bin"])]=1.
    return c1,c2

def shuffle_c2(lines,seed):
    out=copy.deepcopy(lines)
    rng=np.random.default_rng(seed)
    buckets={}
    loc=[]
    for si,s in enumerate(out):
        T=len(s)
        for t,e in enumerate(s):
            key=(int(e["prev"]),role(t,T))
            buckets.setdefault(key,[]).append((si,t))
    for key,ix in buckets.items():
        vals=[(out[si][t]["prev_piece_bin"],out[si][t]["prev_char_bin"]) for si,t in ix]
        perm=rng.permutation(len(vals))
        for dest,src in enumerate(perm):
            si,t=ix[dest]; pb,cb=vals[int(src)]
            out[si][t]["prev_piece_bin"]=int(pb);out[si][t]["prev_char_bin"]=int(cb)
    return out

def flatten(lines):
    ev=[e for s in lines for e in s]
    P=np.vstack([np.asarray(e["p"],float) for e in ev])
    Y=np.array([int(e["y"]) for e in ev],int)
    return ev,P,Y

def design(lines,mode):
    ev,_,_=flatten(lines)
    X=[]
    for e in ev:
        a,b=c1c2(e)
        if mode=="c1":
            X.append(np.r_[a,np.zeros(8)])
        else:
            X.append(np.r_[a,b])
    return np.asarray(X,float)

def fit_scaler(X):
    mu=X.mean(0); sd=X.std(0)
    sd=np.where(sd<1e-8,1.0,sd)
    return mu,sd

def zx(X,mu,sd): return (X-mu)/sd

def fit_ridge(P,Y,X,lam):
    R=-P.copy();R[np.arange(len(Y)),Y]+=1.
    X1=np.c_[np.ones(len(X)),X]
    pen=np.eye(X1.shape[1])*lam;pen[0,0]=1e-9
    B=np.linalg.solve(X1.T@X1+pen,X1.T@R)
    B-=B.mean(1,keepdims=True)
    return B

def apply_ridge(P,X,B,gamma):
    X1=np.c_[np.ones(len(X)),X]
    z=np.log(np.maximum(P,EPS))+gamma*(X1@B)
    z-=z.max(1,keepdims=True);q=np.exp(z);q/=q.sum(1,keepdims=True)
    return q

def choose_ridge(tr,va,mode):
    evtr,Ptr,Ytr=flatten(tr); evv,Pv,Yv=flatten(va)
    Xtr=design(tr,mode);Xv=design(va,mode)
    mu,sd=fit_scaler(Xtr);Xtr=zx(Xtr,mu,sd);Xv=zx(Xv,mu,sd)
    best=None
    for lam in (1.,10.,100.,1000.):
        B=fit_ridge(Ptr,Ytr,Xtr,lam)
        for gam in (.25,.5,1.,2.):
            Q=apply_ridge(Pv,Xv,B,gam);b=bits_probs(Q,Yv)/len(Yv)
            key=(b,lam,gam)
            if best is None or key<best[0]:best=(key,B,mu,sd)
    return best

def predict_ridge(te,fit,mode):
    (b,lam,gam),B,mu,sd=fit
    ev,P,Y=flatten(te);X=zx(design(te,mode),mu,sd);Q=apply_ridge(P,X,B,gam)
    out=[];k=0
    for s in te:
        zz=[]
        for e in s:
            x=dict(e);x["p"]=Q[k];zz.append(x);k+=1
        out.append(zz)
    return out,bits_probs(Q,Y),len(Y),{"lambda":lam,"gamma":gam,"val_bpe":b}

def metric(lines):
    v,names,n=r4["metric_vector"](lines)
    return np.asarray(v,float),names,n

def parent_q3_by_fold():
    q3,meta,n=core["build"]()
    by={j:[] for j in range(5)}
    for s in q3: by[int(s[0]["fold"])].append(s)
    return by,meta,n

class ExpandedGRU(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.gru=nn.GRU(K+12+8,h,batch_first=True)
        self.out=nn.Linear(h,K)
    def forward(self,x,logq):
        z,_=self.gru(x);tilt=self.out(z);tilt-=tilt.mean(dim=-1,keepdim=True)
        return logq+tilt

def batches(lines,batch_size,shuffle,seed,mode):
    idx=np.arange(len(lines))
    if shuffle:
        rng=np.random.default_rng(seed);rng.shuffle(idx)
    for st in range(0,len(idx),batch_size):
        ss=[lines[i] for i in idx[st:st+batch_size]]
        B=len(ss);T=max(len(s) for s in ss)
        x=torch.zeros((B,T,K+20),dtype=torch.float32)
        lq=torch.zeros((B,T,K),dtype=torch.float32)
        y=torch.zeros((B,T),dtype=torch.long);m=torch.zeros((B,T),dtype=torch.bool)
        for b,s in enumerate(ss):
            for t,e in enumerate(s):
                x[b,t,int(e["prev"])]=1.
                a,c=c1c2(e);x[b,t,K:K+12]=torch.from_numpy(a.astype(np.float32))
                if mode!="c1":
                    x[b,t,K+12:]=torch.from_numpy(c.astype(np.float32))
                p=np.asarray(e["p"],float)
                lq[b,t]=torch.from_numpy(np.log(np.maximum(p,EPS)).astype(np.float32))
                y[b,t]=int(e["y"]);m[b,t]=True
        yield x,lq,y,m

def train_gru(tr,j,mode):
    a=ARCH[j];h=a["H"];wd=a["wd"];epochs=a["epochs"]
    wi=0 if wd==1e-4 else 1
    seed=SEEDBASE+j*10000+h*100+wi
    torch.manual_seed(seed);np.random.seed(seed%(2**32-1));random.seed(seed)
    model=ExpandedGRU(h);opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=wd)
    for ep in range(epochs):
        model.train()
        for x,lq,y,m in batches(tr,64,True,seed+ep*1009,mode):
            opt.zero_grad();logits=model(x,lq);lp=torch.log_softmax(logits,dim=-1)
            loss=(-lp.gather(-1,y.unsqueeze(-1)).squeeze(-1)[m]).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
    return model

def predict_gru(model,lines,mode):
    model.eval();out=[];bits=0.;n=0
    with torch.no_grad():
        for x,lq,y,m in batches(lines,128,False,0,mode):
            logits=model(x,lq);pp=torch.softmax(logits,dim=-1).cpu().numpy()
            # batches() may contain many lines, easier re-run linewise below
            break
        for s in lines:
            B=list(batches([s],1,False,0,mode))[0]
            x,lq,y,m=B;logits=model(x,lq);p=torch.softmax(logits,dim=-1)[0].cpu().numpy()
            zz=[]
            for t,e in enumerate(s):
                yy=int(e["y"]);bits+=-math.log2(max(float(p[t,yy]),EPS));n+=1
                ee=dict(e);ee["p"]=p[t].astype(float);zz.append(ee)
            out.append(zz)
    return out,bits,n

def run_ridge():
    by,meta,N=parent_q3_by_fold()
    pooled={"parent":0.,"c1":0.,"real":0.,"shuffle":0.};nn=0;folds=[];allreal=[]
    for j in range(5):
        v=(j+1)%5;trks=[k for k in range(5) if k not in (j,v)]
        tr=[s for k in trks for s in by[k]];va=by[v];te=by[j]
        shtr=shuffle_c2(tr,202610082100+j*100+1);shva=shuffle_c2(va,202610082100+j*100+2);shte=shuffle_c2(te,202610082100+j*100+3)
        f1=choose_ridge(tr,va,"c1");fr=choose_ridge(tr,va,"real");fs=choose_ridge(shtr,shva,"real")
        o1,b1,n1,h1=predict_ridge(te,f1,"c1")
        ore,br,nr,hr=predict_ridge(te,fr,"real")
        osh,bs,ns,hs=predict_ridge(shte,fs,"real")
        _,P,Y=flatten(te);bp=bits_probs(P,Y);n=len(Y)
        assert n==n1==nr==ns
        for key,val in [("parent",bp),("c1",b1),("real",br),("shuffle",bs)]:pooled[key]+=val
        nn+=n;allreal.extend(ore)
        folds.append({"fold":j,"n":n,"parent_bpe":bp/n,"c1_bpe":b1/n,"real_bpe":br/n,"shuffle_bpe":bs/n,
          "gain_vs_parent":(bp-br)/n,"incremental_vs_c1":(b1-br)/n,"real_minus_shuffle":(bs-br)/n,
          "real_hp":hr,"c1_hp":h1,"shuffle_hp":hs})
    vec,names,nm=metric(allreal);D=d2(vec)
    out={k:v/nn for k,v in pooled.items()}
    return {"model":"RIDGE","n":nn,"bpe":out,
      "gain_vs_parent":out["parent"]-out["real"],
      "incremental_vs_c1":out["c1"]-out["real"],
      "real_minus_shuffle":out["shuffle"]-out["real"],
      "positive_folds":sum(x["gain_vs_parent"]>0 for x in folds),
      "vector":vec.tolist(),"metric_names":names,"D2":D,"Z_D2":z_d2(vec),
      "D2_reduction_fraction":1-D/PARENT_D2,
      "phaseA_pass":bool(out["real"]<out["parent"] and sum(x["gain_vs_parent"]>0 for x in folds)>=4 and out["real"]<out["shuffle"] and (1-D/PARENT_D2)>=.20),
      "folds":folds}

def q1_enriched_split(j):
    real=r4["real_lines"]()
    tr,va,te=r4["q1_split"](real,j)
    return core["enrich_q3_lines"](tr)[0],core["enrich_q3_lines"](va)[0],core["enrich_q3_lines"](te)[0]

def run_gru():
    # exact parent q3 fold bpe from R4B reconstruction
    by,meta,N=parent_q3_by_fold()
    pooled={"parent":0.,"c1":0.,"real":0.,"shuffle":0.};nn=0;folds=[];allreal=[]
    for j in range(5):
        tr,va,te=q1_enriched_split(j)
        shtr=shuffle_c2(tr,202610082100+j*100+1);shte=shuffle_c2(te,202610082100+j*100+3)
        m1=train_gru(tr,j,"c1");mr=train_gru(tr,j,"real");ms=train_gru(shtr,j,"real")
        o1,b1,n1=predict_gru(m1,te,"c1")
        ore,br,nr=predict_gru(mr,te,"real")
        osh,bs,ns=predict_gru(ms,shte,"real")
        pev,PP,YY=flatten(by[j]);bp=bits_probs(PP,YY);n=len(YY)
        assert n==n1==nr==ns
        for key,val in [("parent",bp),("c1",b1),("real",br),("shuffle",bs)]:pooled[key]+=val
        nn+=n;allreal.extend(ore)
        folds.append({"fold":j,"n":n,"parent_bpe":bp/n,"c1_bpe":b1/n,"real_bpe":br/n,"shuffle_bpe":bs/n,
          "gain_vs_parent":(bp-br)/n,"incremental_vs_c1":(b1-br)/n,"real_minus_shuffle":(bs-br)/n,
          "H":ARCH[j]["H"],"wd":ARCH[j]["wd"],"epochs":ARCH[j]["epochs"]})
    vec,names,nm=metric(allreal);D=d2(vec);out={k:v/nn for k,v in pooled.items()}
    return {"model":"GRU","n":nn,"bpe":out,
      "gain_vs_parent":out["parent"]-out["real"],
      "incremental_vs_c1":out["c1"]-out["real"],
      "real_minus_shuffle":out["shuffle"]-out["real"],
      "positive_folds":sum(x["gain_vs_parent"]>0 for x in folds),
      "vector":vec.tolist(),"metric_names":names,"D2":D,"Z_D2":z_d2(vec),
      "D2_reduction_fraction":1-D/PARENT_D2,
      "phaseA_pass":bool(out["real"]<out["parent"] and sum(x["gain_vs_parent"]>0 for x in folds)>=4 and out["real"]<out["shuffle"] and (1-D/PARENT_D2)>=.20),
      "folds":folds}

ap=argparse.ArgumentParser()
ap.add_argument("--model",choices=["ridge","gru"],required=True)
args=ap.parse_args()
res=run_ridge() if args.model=="ridge" else run_gru()
print("R1B_PHASEA="+json.dumps(res,separators=(",",":")),flush=True)
