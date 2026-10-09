#!/usr/bin/env python3
# VMS-R4A: finite spectral predictive-state subspace recoverability instrument.
# Preregistered 2026-10-09 before any R4A synthetic outcome.
import argparse,collections,json,math,urllib.request
import numpy as np
from scipy.special import logsumexp

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/a1f996f5c3e1ceaf7c561821993f6194b7fecb60/research/vms_r2_timescale_core_20261008.py"
src=urllib.request.urlopen(CORE_URL,timeout=120).read().decode()
core={"__name__":"r2core"}
exec(compile(src.split('\nif __name__=="__main__":')[0],CORE_URL,"exec"),core)

K=core["K"]; EPS=1e-15
H_OPTIONS=(2,4,6); RANKS=(1,2,4,8); FUT=3
ALPHA=20.0; RIDGE=10.0; MIN_FULL_COUNT=3
TARGET_KL_BITS=.008
RESET_SET={0,2,4,6,8,10}

ap=argparse.ArgumentParser()
ap.add_argument("--mode",choices=["cal","blind","plant"],required=True)
ap.add_argument("--start",type=int,default=0)
ap.add_argument("--count",type=int,default=20)
ARGS=ap.parse_args(); MODE=ARGS.mode

lines,meta,n=core["build_parent"]()
lines,_=core["annotate"](lines)
if n!=9616: raise RuntimeError(("N",n))
# Canonical line container order is deterministic; within-line event order is already physical.
lines=sorted(lines,key=lambda s:(int(s[0]["fold"]),str(s[0]["folio"]),int(s[0]["line_no"])))

def flatten(ls):
    ev=[e for s in ls for e in s]
    P=np.vstack([np.asarray(e["p"],float) for e in ev])
    Y=np.array([int(e["y"]) for e in ev],int)
    return ev,P,Y

def bits(P,Y):
    return float(-np.log2(np.maximum(P[np.arange(len(Y)),Y],EPS)).sum())

def softmax_logits(z):
    z=z-z.max(axis=1,keepdims=True)
    q=np.exp(z); q/=q.sum(axis=1,keepdims=True)
    return q

def replace_y(ls,Y):
    out=[];k=0
    for s in ls:
        zz=[]
        for e in s:
            x=dict(e); x["y"]=int(Y[k]); zz.append(x); k+=1
        out.append(zz)
    if k!=len(Y): raise RuntimeError(("replace",k,len(Y)))
    return out

def hist_key(prev,H,padded=True):
    if H==0:return ()
    z=tuple(int(x) for x in prev[-H:])
    if padded and len(z)<H:z=(-1,)*(H-len(z))+z
    return z

def aggregate_profiles(ls,H):
    # Future innovation profiles are learned only from the supplied basis data.
    # Store all suffix lengths for deterministic backoff, but derive the spectral
    # basis from full-H histories only.
    sums=[collections.defaultdict(lambda:np.zeros(FUT*K,float)) for _ in range(H+1)]
    counts=[collections.Counter() for _ in range(H+1)]
    total=np.zeros(FUT*K,float); ntotal=0
    for s in ls:
        T=len(s); ys=[int(e["y"]) for e in s]
        for t in range(T):
            if t+FUT>=T: continue
            fvec=[]
            for h in range(1,FUT+1):
                e=s[t+h]; y=int(e["y"]); p=np.asarray(e["p"],float)
                r=-p.copy(); r[y]+=1.0; fvec.append(r)
            fv=np.concatenate(fvec)
            total+=fv; ntotal+=1
            prev=ys[:t]
            for L in range(H+1):
                key=hist_key(prev,L,padded=(L==H))
                sums[L][key]+=fv; counts[L][key]+=1
    if ntotal<1: raise RuntimeError("NO_FUTURE_PROFILES")
    gmean=total/ntotal
    return sums,counts,gmean,ntotal

def shrunk_profile(sumv,c,gmean):
    return (sumv+ALPHA*gmean)/(float(c)+ALPHA)

def fit_state_map(ls,H,maxrank=8):
    sums,counts,gmean,ntotal=aggregate_profiles(ls,H)
    full=[]
    weights=[]
    for key,c in counts[H].items():
        if c>=MIN_FULL_COUNT:
            full.append(shrunk_profile(sums[H][key],c,gmean))
            weights.append(float(c))
    if len(full)<2:
        center=gmean.copy()
        basis=np.zeros((FUT*K,1),float)
    else:
        M=np.vstack(full); w=np.asarray(weights,float)
        center=np.average(M,axis=0,weights=w)
        X=(M-center[None,:])*np.sqrt(w[:,None])
        try:
            _,_,Vt=np.linalg.svd(X,full_matrices=False)
        except np.linalg.LinAlgError:
            Vt=np.zeros((1,FUT*K),float)
        rr=max(1,min(maxrank,Vt.shape[0]))
        basis=Vt[:rr].T.copy()
    maps=[]
    for L in range(H+1):
        md={}
        for key,c in counts[L].items():
            prof=shrunk_profile(sums[L][key],c,gmean)
            md[key]=(prof-center)@basis
        maps.append(md)
    return {"H":H,"center":center,"basis":basis,"maps":maps,
            "n_profile_events":ntotal,"n_full_histories":len(full)}

def transform_state(ls,sm,H,r):
    rr=min(r,sm["basis"].shape[1])
    Z=[]
    for s in ls:
        prev=[]
        for e in s:
            z=None
            # Exact full-H padded history first, then suffix backoff.
            kfull=hist_key(prev,H,True)
            if kfull in sm["maps"][H]:
                z=sm["maps"][H][kfull][:rr]
            else:
                for L in range(H-1,-1,-1):
                    kk=hist_key(prev,L,False)
                    if kk in sm["maps"][L]:
                        z=sm["maps"][L][kk][:rr]; break
            if z is None:z=np.zeros(rr,float)
            if rr<r:z=np.r_[z,np.zeros(r-rr,float)]
            Z.append(np.asarray(z,float))
            prev.append(int(e["y"]))
    return np.vstack(Z) if Z else np.zeros((0,r),float)

def fit_residual(P,Y,Z):
    X=np.c_[np.ones(len(Z)),Z]
    R=-P.copy(); R[np.arange(len(Y)),Y]+=1.0
    pen=np.eye(X.shape[1])*RIDGE; pen[0,0]=1e-9
    B=np.linalg.solve(X.T@X+pen,X.T@R)
    B-=B.mean(axis=1,keepdims=True)
    return B

def predict_residual(P,Z,B):
    X=np.c_[np.ones(len(Z)),Z]
    return softmax_logits(np.log(np.maximum(P,EPS))+X@B)

def lines_by_fold(ls):
    return {j:[s for s in ls if int(s[0]["fold"])==j] for j in range(5)}

def outer_fold(ls,j):
    by=lines_by_fold(ls)
    v=(j+1)%5
    trks=[k for k in range(5) if k not in (j,v)]
    va=by[v]; te=by[j]
    _,Pva,Yva=flatten(va); _,Pte,Yte=flatten(te)
    candidates={}
    for H in H_OPTIONS:
        # Precompute each rotation's state map and transformed validation/test.
        rotations=[]
        for wi in trks:
            bks=[k for k in trks if k!=wi]
            basis_ls=[s for k in bks for s in by[k]]
            fit_ls=by[wi]
            sm=fit_state_map(basis_ls,H,maxrank=max(RANKS))
            _,Pfit,Yfit=flatten(fit_ls)
            Zfit_all=transform_state(fit_ls,sm,H,max(RANKS))
            Zva_all=transform_state(va,sm,H,max(RANKS))
            Zte_all=transform_state(te,sm,H,max(RANKS))
            rotations.append((Pfit,Yfit,Zfit_all,Zva_all,Zte_all,sm))
        for r in RANKS:
            pv=[];pt=[]
            for Pfit,Yfit,Zfit_all,Zva_all,Zte_all,sm in rotations:
                B=fit_residual(Pfit,Yfit,Zfit_all[:,:r])
                pv.append(predict_residual(Pva,Zva_all[:,:r],B))
                pt.append(predict_residual(Pte,Zte_all[:,:r],B))
            Qv=np.mean(np.stack(pv),axis=0); Qt=np.mean(np.stack(pt),axis=0)
            vb=bits(Qv,Yva)/len(Yva)
            candidates[(H,r)]={"val_bpe":vb,"test_Q":Qt}
    best=min(candidates,key=lambda hr:(candidates[hr]["val_bpe"],hr[0],hr[1]))
    Qt=candidates[best]["test_Q"]
    bp=bits(Pte,Yte); bm=bits(Qt,Yte); nte=len(Yte)
    return {"fold":j,"n":nte,"gain":(bp-bm)/nte,
            "parent_bpe":bp/nte,"model_bpe":bm/nte,
            "selected_H":best[0],"selected_rank":best[1],
            "val_bpe":candidates[best]["val_bpe"]}

def cv(Y):
    syn=replace_y(lines,Y)
    rows=[];num=0.;den=0
    for j in range(5):
        z=outer_fold(syn,j); rows.append(z)
        num+=z["gain"]*z["n"]; den+=z["n"]
    return {"gain":float(num/den),
            "positive_folds":int(sum(z["gain"]>0 for z in rows)),
            "folds":rows}

EV,PALL,Y0=flatten(lines)
rngw=np.random.default_rng(202610097000)
w=rngw.normal(size=K); w-=w.mean(); w/=np.linalg.norm(w)

def tilt_probs(P,b):
    z=np.log(np.maximum(P,EPS))+b[None,:]
    z-=logsumexp(z,axis=1)[:,None]
    return np.exp(z)

def expected_kl(scale):
    vals=[]
    for sign in (-1,1):
        Q=tilt_probs(PALL,sign*scale*w)
        vals.append(np.mean(np.sum(Q*(np.log(np.maximum(Q,EPS))-np.log(np.maximum(PALL,EPS))),axis=1))/math.log(2))
    return float(np.mean(vals))
lo,hi=0.,1.
while expected_kl(hi)<TARGET_KL_BITS:hi*=2
for _ in range(50):
    m=(lo+hi)/2
    if expected_kl(m)<TARGET_KL_BITS:lo=m
    else:hi=m
PLANT_SCALE=(lo+hi)/2
PLANT_B=np.vstack([-PLANT_SCALE*w,PLANT_SCALE*w])

def sample_null(seed):
    rng=np.random.default_rng(seed); Y=[]
    for s in lines:
        for e in s:
            p=np.asarray(e["p"],float)
            Y.append(int(rng.choice(K,p=p/p.sum())))
    return np.asarray(Y,int)

def sample_plant(seed):
    rng=np.random.default_rng(seed); Y=[]
    for s in lines:
        st=0
        for e in s:
            p=np.asarray(e["p"],float)
            q=tilt_probs(p[None,:],PLANT_B[st])[0]
            y=int(rng.choice(K,p=q/q.sum()))
            Y.append(y)
            st=0 if y in RESET_SET else 1-st
    return np.asarray(Y,int)

ALL_SEEDS={"cal":list(range(202610097100,202610097120)),
           "blind":list(range(202610097120,202610097140)),
           "plant":list(range(202610097200,202610097220))}[MODE]
if ARGS.start<0 or ARGS.count<1 or ARGS.start+ARGS.count>20:
    raise ValueError(("SHARD",ARGS.start,ARGS.count))
seeds=ALL_SEEDS[ARGS.start:ARGS.start+ARGS.count]
outs=[]
for i,seed in enumerate(seeds):
    Y=sample_plant(seed) if MODE=="plant" else sample_null(seed)
    z=cv(Y); z.update(rep=ARGS.start+i,seed=seed); outs.append(z)
    print("R4A_REP",MODE,i,json.dumps({"gain":z["gain"],"pos":z["positive_folds"],
          "selected":[[f["selected_H"],f["selected_rank"]] for f in z["folds"]]},
          separators=(",",":")),flush=True)
print("R4A_SHARD="+json.dumps({"programme":"VMS-R4A","mode":MODE,"status":"complete",
      "n_events":len(Y0),"n_lines":len(lines),"H_options":H_OPTIONS,"ranks":RANKS,
      "future_horizon":FUT,"alpha":ALPHA,"ridge":RIDGE,"shard_start":ARGS.start,
      "shard_count":ARGS.count,"plant_scale":PLANT_SCALE,
      "plant_expected_kl_bits":expected_kl(PLANT_SCALE),"results":outs},
      separators=(",",":")),flush=True)
