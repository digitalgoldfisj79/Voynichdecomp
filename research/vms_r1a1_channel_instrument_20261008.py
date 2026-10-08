#!/usr/bin/env python3
# VMS-R1A1 observable-channel recoverability instrument.
# Preregistered in voynich_vms_r1_observable_channel_preregister_20261008 v2.
import argparse,json,math,urllib.request
import numpy as np

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/b02193969cb42745ee9cf3cea4985fa7664602b2/research/vms_r1_channel_core_20261008.py"
src=urllib.request.urlopen(CORE_URL,timeout=120).read().decode()
core={"__name__":"r1core"}
exec(compile(src,CORE_URL,"exec"),core)

K=core["K"]; EPS=1e-15
LAMBDAS=(1.0,10.0,100.0,1000.0)
GAMMAS=(0.25,0.5,1.0,2.0)
TARGET_KL_BITS=0.008
N_CAL=20; N_BLIND=20; N_PLANT=20
TOP_FAMILIES=64; PCA_RANK=8

ap=argparse.ArgumentParser()
ap.add_argument("--channel",type=int,required=True,choices=[1,2,3,4,5])
args=ap.parse_args()
CH=args.channel

lines,fold_meta,nmapped=core["build"]()
events=[e for s in lines for e in s]
PALL=np.vstack([np.asarray(e["p"],float) for e in events])
FOLD=np.array([int(e["fold"]) for e in events],int)
N=len(events)
if N!=nmapped:
    raise RuntimeError(("mapped_count_mismatch",N,nmapped))

class Featurizer:
    def __init__(self,train_events,max_channel):
        self.max_channel=max_channel
        fam_counts={}
        for e in train_events:
            fam_counts[e["prev_family"]]=fam_counts.get(e["prev_family"],0)+1
        top=sorted(fam_counts,key=lambda x:(-fam_counts[x],x))[:TOP_FAMILIES]
        self.fam_map={x:i for i,x in enumerate(top)}
        self.fam_other=len(top)
        self.joint_mean=None; self.joint_basis=None
        if max_channel>=5:
            J=np.vstack([np.asarray(e["recent_joint_counts"],float) for e in train_events])
            self.joint_mean=J.mean(0)
            Z=J-self.joint_mean
            # deterministic training-only SVD
            _,_,vt=np.linalg.svd(Z,full_matrices=False)
            self.joint_basis=vt[:min(PCA_RANK,vt.shape[0])].T
        # raw cumulative block dimensions
        self.dims=[0]
        if max_channel>=1:self.dims.append(self.dims[-1]+12)
        if max_channel>=2:self.dims.append(self.dims[-1]+8)
        if max_channel>=3:self.dims.append(self.dims[-1]+len(top)+1)
        if max_channel>=4:self.dims.append(self.dims[-1]+18)
        if max_channel>=5:self.dims.append(self.dims[-1]+self.joint_basis.shape[1])
        # pad dims to channel index for convenience
        while len(self.dims)<=max_channel:self.dims.append(self.dims[-1])
        R=self._raw(train_events,max_channel)
        self.mu=R.mean(0) if R.shape[1] else np.zeros(0)
        self.sd=R.std(0,ddof=0) if R.shape[1] else np.ones(0)
        self.sd=np.where(self.sd<1e-8,1.0,self.sd)

    def _raw(self,evs,ch):
        xs=[]
        for e in evs:
            a=[]
            if ch>=1:
                z=np.zeros(12); z[int(e["prev_final_class"])]=1.; a.extend(z)
            if ch>=2:
                z=np.zeros(4); z[int(e["prev_piece_bin"])]=1.; a.extend(z)
                z=np.zeros(4); z[int(e["prev_char_bin"])]=1.; a.extend(z)
            if ch>=3:
                z=np.zeros(len(self.fam_map)+1)
                z[self.fam_map.get(e["prev_family"],self.fam_other)]=1.; a.extend(z)
            if ch>=4:
                a.extend(np.asarray(e["recent_final_counts"],float)/6.0)
                a.extend(np.asarray(e["recent_char_counts"],float)/6.0)
                a.append(float(e["recent_prev_family_count"])/6.0)
                a.append(float(e["recent_unique_families"])/6.0)
            if ch>=5:
                j=np.asarray(e["recent_joint_counts"],float)
                a.extend((j-self.joint_mean)@self.joint_basis)
            xs.append(a)
        return np.asarray(xs,float)

    def transform(self,evs,ch):
        if ch==0:
            return np.ones((len(evs),1),float) # common intercept only
        R=self._raw(evs,ch)
        # standardization parameters correspond to full max_channel prefix;
        # channel blocks are cumulative prefixes.
        d=R.shape[1]
        Z=(R-self.mu[:d])/self.sd[:d]
        return np.c_[np.ones(len(evs)),Z]

def softmax_logits(logp,delta):
    z=logp+delta
    z-=z.max(1,keepdims=True)
    q=np.exp(z);q/=q.sum(1,keepdims=True)
    return q

def fit_ridge(P,Y,X,lam):
    R=-P.copy(); R[np.arange(len(Y)),Y]+=1.0
    pen=np.eye(X.shape[1])*lam; pen[0,0]=1e-9
    B=np.linalg.solve(X.T@X+pen,X.T@R)
    # offset should not create arbitrary common logit shifts
    B-=B.mean(1,keepdims=True)
    return B

def bits(P,Y):
    return float(-np.log2(np.maximum(P[np.arange(len(Y)),Y],EPS)).sum())

def select_and_eval(Y,channel):
    total_prev=total_cand=0.; totaln=0; fold_gains=[]
    for j in range(5):
        v=(j+1)%5
        trix=np.where((FOLD!=j)&(FOLD!=v))[0]
        vaix=np.where(FOLD==v)[0]
        teix=np.where(FOLD==j)[0]
        tr_events=[events[i] for i in trix]; va_events=[events[i] for i in vaix]; te_events=[events[i] for i in teix]
        feat=Featurizer(tr_events,channel)
        # candidate and previous simpler use same candidate-fitted feature universe;
        # previous is exact cumulative prefix.
        Xtr_c=feat.transform(tr_events,channel)
        Xva_c=feat.transform(va_events,channel)
        Xte_c=feat.transform(te_events,channel)
        prev=channel-1
        Xtr_p=feat.transform(tr_events,prev)
        Xva_p=feat.transform(va_events,prev)
        Xte_p=feat.transform(te_events,prev)

        def choose(Xtr,Xva):
            best=None
            for lam in LAMBDAS:
                B=fit_ridge(PALL[trix],Y[trix],Xtr,lam)
                for gam in GAMMAS:
                    Q=softmax_logits(np.log(np.maximum(PALL[vaix],EPS)),gam*(Xva@B))
                    b=bits(Q,Y[vaix])/len(vaix)
                    z=(b,lam,gam,B)
                    if best is None or (z[0],z[1],z[2])<(best[0],best[1],best[2]):best=z
            return best
        bp=choose(Xtr_p,Xva_p); bc=choose(Xtr_c,Xva_c)
        Qp=softmax_logits(np.log(np.maximum(PALL[teix],EPS)),bp[2]*(Xte_p@bp[3]))
        Qc=softmax_logits(np.log(np.maximum(PALL[teix],EPS)),bc[2]*(Xte_c@bc[3]))
        pbits=bits(Qp,Y[teix]); cbits=bits(Qc,Y[teix]); nn=len(teix)
        total_prev+=pbits; total_cand+=cbits; totaln+=nn
        fold_gains.append((pbits-cbits)/nn)
    return {"gain":(total_prev-total_cand)/totaln,
            "positive_folds":int(sum(g>0 for g in fold_gains)),
            "fold_gains":fold_gains}

def global_delta_features(channel):
    # outcome-independent global transform only for planting.
    feat=Featurizer(events,channel)
    Xc=feat.transform(events,channel)[:,1:]
    if channel==1:
        start=0
    else:
        # previous cumulative raw dimension from the same featurizer
        start=feat.dims[channel-1]
    end=feat.dims[channel]
    return Xc[:,start:end]

def planted_probs(channel):
    X=global_delta_features(channel)
    if X.shape[1]==0: raise RuntimeError(("empty_delta_block",channel))
    rng=np.random.default_rng(202610081000+channel)
    W=rng.normal(0,1/math.sqrt(max(1,X.shape[1])),size=(X.shape[1],K))
    W-=W.mean(1,keepdims=True)
    D=X@W
    logp=np.log(np.maximum(PALL,EPS))
    def klbits(s):
        Q=softmax_logits(logp,s*D)
        return float(np.mean(np.sum(Q*(np.log(np.maximum(Q,EPS))-logp),axis=1))/math.log(2))
    lo,hi=0.,1.
    while klbits(hi)<TARGET_KL_BITS and hi<128: hi*=2
    for _ in range(50):
        mid=(lo+hi)/2
        if klbits(mid)<TARGET_KL_BITS:lo=mid
        else:hi=mid
    s=(lo+hi)/2
    Q=softmax_logits(logp,s*D)
    return Q,s,klbits(s)

def sample_labels(P,seed):
    rng=np.random.default_rng(seed)
    out=[]
    for p in P:
        q=np.asarray(p,float).copy()
        q=np.maximum(q,0.0)
        s=float(q.sum())
        if not np.isfinite(s) or s<=0:
            raise RuntimeError(("invalid_sampling_probability_row",s,q.tolist()))
        q/=s
        out.append(rng.choice(K,p=q))
    return np.array(out,int)

PPLANT,plant_scale,plant_kl=planted_probs(CH)
null=[]
for i in range(N_CAL+N_BLIND):
    Y=sample_labels(PALL,202610081100+CH*1000+i)
    r=select_and_eval(Y,CH)
    null.append(r)
    print("R1A_NULL",CH,i,json.dumps({"gain":r["gain"],"pos":r["positive_folds"]},separators=(",",":")),flush=True)

cal=np.array([x["gain"] for x in null[:N_CAL]],float)
blind=np.array([x["gain"] for x in null[N_CAL:]],float)
thr=float(np.quantile(cal,.95))

plant=[]
for i in range(N_PLANT):
    Y=sample_labels(PPLANT,202610081500+CH*1000+i)
    r=select_and_eval(Y,CH)
    plant.append(r)
    print("R1A_PLANT",CH,i,json.dumps({"gain":r["gain"],"pos":r["positive_folds"]},separators=(",",":")),flush=True)

pg=np.array([x["gain"] for x in plant],float)
fp=float(np.mean(blind>thr))
det=float(np.mean(pg>thr))
family=float(np.mean(pg>0))
passed=bool(fp<=.05 and det>=.90 and float(np.median(pg))>0 and family>=.80)
out={
 "programme":"VMS-R1A1",
 "channel":CH,
 "status":"complete",
 "parent_fold_meta":fold_meta,
 "n_events":N,
 "plant_target_kl_bits":TARGET_KL_BITS,
 "plant_realized_kl_bits":plant_kl,
 "plant_scale":plant_scale,
 "null_cal_q95_gain":thr,
 "blind_false_positive_rate":fp,
 "planted_detection_rate":det,
 "planted_positive_fraction":family,
 "planted_median_gain":float(np.median(pg)),
 "planted_gain_q10":float(np.quantile(pg,.10)),
 "planted_gain_q90":float(np.quantile(pg,.90)),
 "decision":"INSTRUMENT_QUALIFIED" if passed else "INSTRUMENT_UNDERPOWERED"
}
print("R1A_RESULT="+json.dumps(out,separators=(",",":")),flush=True)
