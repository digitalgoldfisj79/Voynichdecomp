#!/usr/bin/env python3
# VMS-RESIDREC4 Phase A — preregistered 2026-10-08.
import copy,json,math,os,random,urllib.request
import numpy as np
import torch
import torch.nn as nn

BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/9ea2a0d87af8b8a9dfb57acbef77acaee0a3dfd1/research/vms_residrec2_context_vs_hmm_phaseA_20261008.py"
src=urllib.request.urlopen(BASE_URL,timeout=120).read().decode()
prefix=src.split("\nimport os\n")[0]
pa={"__name__":"residrec2_defs"}
exec(compile(prefix,BASE_URL,"exec"),pa)

COHORTS=pa["COHORTS"]; K=pa["K"]; REAL=pa["REAL"]
fit_ctx=pa["fit_ctx"]; ctx_prob=pa["ctx_prob"]; pool_prob=pa["pool_prob"]
baseline_bits=pa["baseline_bits"]
EPS=1e-15
torch.set_num_threads(min(8,os.cpu_count() or 4))

HPS={
"HERBAL_A":{0:(2,32.0,.25),1:(1,8.0,.25),2:(5,32.0,.25),3:(1,32.0,.25),4:(5,32.0,.25)},
"HERBAL_B":{0:(2,32.0,.25),1:(5,32.0,.25),2:(2,32.0,.25),3:(1,2.0,.25),4:(5,32.0,.25)},
"BALNEO_FULL":{0:(2,32.0,.25),1:(1,32.0,.25),2:(1,32.0,.25),3:(1,32.0,.25),4:(2,32.0,.25)},
"RECIPES_FULL":{0:(2,32.0,.25),1:(5,32.0,.25),2:(1,32.0,.25),3:(2,32.0,.25),4:(3,32.0,.25)}
}
HGRID=(4,8,16); WDGRID=(1e-4,1e-3)

def seq_fold(seq): return int(seq[0]["fold"])

def apply_context(lines,model,hp):
    cnt0,cnt=model; L,a,lam=hp; out=[]
    for seq in lines:
        if not seq:continue
        hist=[int(seq[0]["prev"])]; zz=[]
        for e in seq:
            pc=ctx_prob(hist,cnt0,cnt,L,a)
            p1=pool_prob(np.asarray(e["p"],float),pc,lam)
            x=dict(e); x["p"]=p1; zz.append(x)
            hist.append(int(e["y"]))
        out.append(zz)
    return out

def q1_split(lines,co,j):
    folds={k:[s for s in lines if seq_fold(s)==k] for k in range(5)}
    v=(j+1)%5; trks=[k for k in range(5) if k not in (j,v)]
    hp=HPS[co][j]
    tr=[]
    # Each training fold receives a q1 offset fit on the other two TRAIN folds.
    for k in trks:
        source=[s for kk in trks if kk!=k for s in folds[kk]]
        model=fit_ctx(source,5)
        tr.extend(apply_context(folds[k],model,hp))
    model_all=fit_ctx([s for k in trks for s in folds[k]],5)
    va=apply_context(folds[v],model_all,hp)
    te=apply_context(folds[j],model_all,hp)
    return tr,va,te,hp

def bits(lines):
    z=0.;n=0
    for seq in lines:
        for e in seq:
            z += -math.log2(max(float(e["p"][int(e["y"])]),EPS)); n+=1
    return z,n

class OffsetGRU(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.gru=nn.GRU(K,h,batch_first=True)
        self.out=nn.Linear(h,K)
    def forward(self,x,logq):
        z,_=self.gru(x)
        tilt=self.out(z)
        tilt=tilt-tilt.mean(dim=-1,keepdim=True)
        return logq+tilt

def batches(lines,batch_size,shuffle,seed):
    idx=np.arange(len(lines))
    if shuffle:
        rng=np.random.default_rng(seed); rng.shuffle(idx)
    for start in range(0,len(idx),batch_size):
        xs=[lines[i] for i in idx[start:start+batch_size]]
        B=len(xs); T=max(len(s) for s in xs)
        x=torch.zeros((B,T,K),dtype=torch.float32)
        logq=torch.zeros((B,T,K),dtype=torch.float32)
        y=torch.zeros((B,T),dtype=torch.long)
        mask=torch.zeros((B,T),dtype=torch.bool)
        for b,seq in enumerate(xs):
            prev=int(seq[0]["prev"])
            for t,e in enumerate(seq):
                x[b,t,prev]=1.0
                p=np.asarray(e["p"],float)
                logq[b,t]=torch.from_numpy(np.log(np.maximum(p,EPS)).astype(np.float32))
                yy=int(e["y"]); y[b,t]=yy; mask[b,t]=True; prev=yy
        yield x,logq,y,mask

def eval_model(model,lines):
    model.eval(); loss=0.;n=0
    with torch.no_grad():
        for x,lq,y,m in batches(lines,128,False,0):
            logits=model(x,lq)
            lp=torch.log_softmax(logits,dim=-1)
            vals=-lp.gather(-1,y.unsqueeze(-1)).squeeze(-1)
            loss += float(vals[m].sum().item())/math.log(2.0)
            n += int(m.sum().item())
    return loss,n

def train_candidate(tr,va,h,wd,seed):
    torch.manual_seed(seed); np.random.seed(seed%(2**32-1)); random.seed(seed)
    model=OffsetGRU(h)
    opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=wd)
    best_bits=float("inf"); best_state=None; best_epoch=0; patience=0
    for epoch in range(60):
        model.train()
        for bi,(x,lq,y,m) in enumerate(batches(tr,64,True,seed+epoch*1009)):
            opt.zero_grad()
            logits=model(x,lq)
            lp=torch.log_softmax(logits,dim=-1)
            loss=(-lp.gather(-1,y.unsqueeze(-1)).squeeze(-1)[m]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
        vb,vn=eval_model(model,va); bpe=vb/max(vn,1)
        if bpe < best_bits-1e-6:
            best_bits=bpe; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1; patience=0
        else:
            patience+=1
            if patience>=8: break
    model.load_state_dict(best_state)
    return model,best_bits,best_epoch

def outer(co,lines,j):
    tr,va,te,hp=q1_split(lines,co,j)
    best=None
    ci=COHORTS.index(co)
    for h in HGRID:
        for wi,wd in enumerate(WDGRID):
            seed=202610081900 + ci*100000 + j*10000 + h*100 + wi
            model,vb,ep=train_candidate(tr,va,h,wd,seed)
            key=(vb,h,wd)
            if best is None or key<best[0]:
                best=(key,copy.deepcopy(model.state_dict()),ep)
    (_,h,wd),state,ep=best
    model=OffsetGRU(h); model.load_state_dict(state)
    q1b,n1=bits(te); q3b,n3=eval_model(model,te); assert n1==n3
    # q0 for exact same outer test lines
    q0te=[s for s in lines if seq_fold(s)==j]
    q0b,n0=baseline_bits(q0te); assert n0==n1
    return {"test_fold":j,"validation_fold":(j+1)%5,"n":n1,
            "H":h,"weight_decay":wd,"best_epoch":ep,
            "m0_bpe":q0b/n0,"m1_bpe":q1b/n1,"m3_bpe":q3b/n3,
            "gain_m1_vs_m0":(q0b-q1b)/n1,
            "gain_m3_vs_m1":(q1b-q3b)/n1,
            "gain_m3_vs_m0":(q0b-q3b)/n1}

def run_cohort(co):
    lines=REAL[co]["lines"]; ff=[]
    for j in range(5):
        print("FOLD_START",co,j,flush=True)
        r=outer(co,lines,j); ff.append(r)
        print("FOLD_RESULT",co,j,json.dumps(r,separators=(",",":")),flush=True)
    N=sum(x["n"] for x in ff)
    m0=sum(x["m0_bpe"]*x["n"] for x in ff)/N
    m1=sum(x["m1_bpe"]*x["n"] for x in ff)/N
    m3=sum(x["m3_bpe"]*x["n"] for x in ff)/N
    g31=m1-m3; pos=sum(x["gain_m3_vs_m1"]>0 for x in ff)
    cand=bool(g31>.0005 and pos>=4 and m3<m0)
    return {"pooled_bpe":{"M0":m0,"M1":m1,"M3":m3},
            "gain_M3_vs_M1":g31,"gain_M3_vs_M0":m0-m3,
            "positive_folds_M3_vs_M1":pos,
            "decision":"M3_RECOVERY_CANDIDATE" if cand else "RECURRENT_STATE_NOT_QUALIFIED",
            "folds":ff}

OUT={}
for co in COHORTS:
    print("COHORT_START",co,flush=True)
    OUT[co]=run_cohort(co)
    print("COHORT_RESULT",co,json.dumps(OUT[co],separators=(",",":")),flush=True)
licensed=[c for c,v in OUT.items() if v["decision"]=="M3_RECOVERY_CANDIDATE"]
print("VMS_RESIDREC4_PHASEA_JSON="+json.dumps({"programme":"VMS-RESIDREC4","phase":"A","status":"complete",
      "residrec4b_licensed":bool(licensed),"licensed_ecologies":licensed,"ecologies":OUT},separators=(",",":")),flush=True)
