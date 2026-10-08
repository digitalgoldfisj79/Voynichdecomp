#!/usr/bin/env python3
# VMS-R2A observed-timescale recoverability instrument.
import argparse,json,math,urllib.request
import numpy as np

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/a1f996f5c3e1ceaf7c561821993f6194b7fecb60/research/vms_r2_timescale_core_20261008.py"
src=urllib.request.urlopen(CORE_URL,timeout=120).read().decode()
core={"__name__":"r2core"}
exec(compile(src.split('\nif __name__=="__main__":')[0],CORE_URL,"exec"),core)

K=core["K"]; EPS=1e-15
LAMBDAS=(1.0,10.0,100.0,1000.0)
GAMMAS=(0.25,0.5,1.0,2.0)
TARGET_KL_BITS=0.008
N_CAL=20;N_BLIND=20;N_PLANT=20

ap=argparse.ArgumentParser()
ap.add_argument("--level",type=int,choices=[1,2],required=True)
args=ap.parse_args()
LEVEL=args.level

lines,fold_meta,nmapped=core["build_parent"]()
lines=core["attach_timescale_features"](lines)
events=[e for s in lines for e in s]
PALL=np.vstack([np.asarray(e["p"],float) for e in events])
YREAL=np.array([int(e["y"]) for e in events],int)
FOLD=np.array([int(e["fold"]) for e in events],int)
if len(events)!=9616 or nmapped!=9616:
    raise RuntimeError(("R2_EVENT_COUNT",len(events),nmapped))

def raw_feature(e,level):
    if level==0:
        return np.zeros(0,float)
    if level==1:
        return np.r_[np.asarray(e["t1_cum_k12"],float),
                     np.log1p(float(e["t1_prior_lines"])),
                     float(e["t1_prior_lines"]>0)]
    if level==2:
        return np.r_[raw_feature(e,1),
                     float(e["t2_has_prev"]),
                     np.asarray(e["t2_prev_k12"],float),
                     np.asarray(e["t2_prev_final"],float),
                     np.asarray(e["t2_prev_char"],float),
                     np.asarray(e["t2_prev_piece"],float),
                     np.log1p(float(e["t2_prev_length"])),
                     float(e["t2_prev_surprise"]),
                     float(e["t2_prev_pearson"])]
    raise ValueError(level)

def X_for(evs,level):
    if level==0:return np.ones((len(evs),1),float)
    R=np.vstack([raw_feature(e,level) for e in evs])
    return R

def standardize(Xtr,Xva,Xte):
    mu=Xtr.mean(0);sd=Xtr.std(0)
    sd=np.where(sd<1e-8,1.,sd)
    return (Xtr-mu)/sd,(Xva-mu)/sd,(Xte-mu)/sd

def softmax_logits(logp,delta):
    z=logp+delta
    z-=z.max(1,keepdims=True)
    q=np.exp(z);q/=q.sum(1,keepdims=True)
    return q

def bits(P,Y):
    return float(-np.log2(np.maximum(P[np.arange(len(Y)),Y],EPS)).sum())

def fit_ridge(P,Y,X,lam):
    X1=np.c_[np.ones(len(X)),X]
    R=-P.copy();R[np.arange(len(Y)),Y]+=1.
    pen=np.eye(X1.shape[1])*lam;pen[0,0]=1e-9
    B=np.linalg.solve(X1.T@X1+pen,X1.T@R)
    B-=B.mean(1,keepdims=True)
    return B

def apply(P,X,B,gam):
    X1=np.c_[np.ones(len(X)),X]
    return softmax_logits(np.log(np.maximum(P,EPS)),gam*(X1@B))

def select_eval(Y,level):
    prev=max(0,level-1)
    total_prev=0.;total_cand=0.;totaln=0;fold_gains=[]
    for j in range(5):
        v=(j+1)%5
        trix=np.where((FOLD!=j)&(FOLD!=v))[0]
        vaix=np.where(FOLD==v)[0]
        teix=np.where(FOLD==j)[0]
        etr=[events[i] for i in trix];eva=[events[i] for i in vaix];ete=[events[i] for i in teix]
        Xtrc=X_for(etr,level);Xvac=X_for(eva,level);Xtec=X_for(ete,level)
        Xtrp=X_for(etr,prev);Xvap=X_for(eva,prev);Xtep=X_for(ete,prev)
        if level>0:Xtrc,Xvac,Xtec=standardize(Xtrc,Xvac,Xtec)
        if prev>0:Xtrp,Xvap,Xtep=standardize(Xtrp,Xvap,Xtep)
        elif prev==0:
            Xtrp=np.zeros((len(trix),0));Xvap=np.zeros((len(vaix),0));Xtep=np.zeros((len(teix),0))

        def choose(Xtr,Xva):
            best=None
            for lam in LAMBDAS:
                B=fit_ridge(PALL[trix],Y[trix],Xtr,lam)
                for gam in GAMMAS:
                    Q=apply(PALL[vaix],Xva,B,gam)
                    b=bits(Q,Y[vaix])/len(vaix)
                    z=(b,lam,gam,B)
                    if best is None or (z[0],z[1],z[2])<(best[0],best[1],best[2]):best=z
            return best
        bp=choose(Xtrp,Xvap);bc=choose(Xtrc,Xvac)
        Qp=apply(PALL[teix],Xtep,bp[3],bp[2]);Qc=apply(PALL[teix],Xtec,bc[3],bc[2])
        pbits=bits(Qp,Y[teix]);cbits=bits(Qc,Y[teix]);n=len(teix)
        total_prev+=pbits;total_cand+=cbits;totaln+=n
        fold_gains.append((pbits-cbits)/n)
    return {"gain":(total_prev-total_cand)/totaln,
            "positive_folds":int(sum(g>0 for g in fold_gains)),
            "fold_gains":fold_gains}

def delta_features(level):
    full=np.vstack([raw_feature(e,level) for e in events])
    if level==1:return full
    prev=np.vstack([raw_feature(e,level-1) for e in events])
    return full[:,prev.shape[1]:]

def plant_probs(level):
    X=delta_features(level)
    mu=X.mean(0);sd=np.where(X.std(0)<1e-8,1.,X.std(0));X=(X-mu)/sd
    rng=np.random.default_rng(202610082500+level)
    W=rng.normal(0,1/math.sqrt(max(1,X.shape[1])),size=(X.shape[1],K))
    W-=W.mean(1,keepdims=True)
    D=X@W;logp=np.log(np.maximum(PALL,EPS))
    def klbits(s):
        Q=softmax_logits(logp,s*D)
        return float(np.mean(np.sum(Q*(np.log(np.maximum(Q,EPS))-logp),axis=1))/math.log(2))
    lo,hi=0.,1.
    while klbits(hi)<TARGET_KL_BITS and hi<128:hi*=2
    for _ in range(50):
        mid=(lo+hi)/2
        if klbits(mid)<TARGET_KL_BITS:lo=mid
        else:hi=mid
    s=(lo+hi)/2
    return softmax_logits(logp,s*D),s,klbits(s)

def sample_labels(P,seed):
    rng=np.random.default_rng(seed);out=[]
    for p in P:
        q=np.maximum(np.asarray(p,float),0.);q/=q.sum()
        out.append(rng.choice(K,p=q))
    return np.array(out,int)

PPLANT,pscale,pkl=plant_probs(LEVEL)
null=[]
for i in range(N_CAL+N_BLIND):
    Y=sample_labels(PALL,202610082600+LEVEL*1000+i)
    z=select_eval(Y,LEVEL);null.append(z)
    print("R2A_NULL",LEVEL,i,json.dumps({"gain":z["gain"],"pos":z["positive_folds"]},separators=(",",":")),flush=True)
cal=np.array([x["gain"] for x in null[:N_CAL]],float)
blind=np.array([x["gain"] for x in null[N_CAL:]],float)
thr=float(np.quantile(cal,.95))

plant=[]
for i in range(N_PLANT):
    Y=sample_labels(PPLANT,202610082900+LEVEL*1000+i)
    z=select_eval(Y,LEVEL);plant.append(z)
    print("R2A_PLANT",LEVEL,i,json.dumps({"gain":z["gain"],"pos":z["positive_folds"]},separators=(",",":")),flush=True)
pg=np.array([x["gain"] for x in plant],float)
fp=float(np.mean(blind>thr));det=float(np.mean(pg>thr));pos=float(np.mean(pg>0))
passed=bool(fp<=.05 and det>=.90 and float(np.median(pg))>0 and pos>=.80)
out={
 "programme":"VMS-R2A","level":LEVEL,"status":"complete","n_events":len(events),
 "plant_target_kl_bits":TARGET_KL_BITS,"plant_realized_kl_bits":pkl,"plant_scale":pscale,
 "null_cal_q95_gain":thr,"blind_false_positive_rate":fp,
 "planted_detection_rate":det,"planted_positive_fraction":pos,
 "planted_median_gain":float(np.median(pg)),
 "planted_gain_q10":float(np.quantile(pg,.10)),"planted_gain_q90":float(np.quantile(pg,.90)),
 "decision":"INSTRUMENT_QUALIFIED" if passed else "INSTRUMENT_UNDERPOWERED"
}
print("R2A_RESULT="+json.dumps(out,separators=(",",":")),flush=True)
