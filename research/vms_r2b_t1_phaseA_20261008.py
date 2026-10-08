#!/usr/bin/env python3
# VMS-R2B T1 real Phase A — observed prior-line cumulative history.
import copy,json,math,urllib.request
import numpy as np

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/a1f996f5c3e1ceaf7c561821993f6194b7fecb60/research/vms_r2_timescale_core_20261008.py"
CAL_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/36740cd684ed22dba02acf33f3bb1109c19bfc54/research/data/residrec4b_residual15_calibration_20261008.json"

src=urllib.request.urlopen(CORE_URL,timeout=120).read().decode()
core={"__name__":"r2core"}
exec(compile(src.split('\nif __name__=="__main__":')[0],CORE_URL,"exec"),core)

K=core["K"]; r4=core["r1"]["r4"]; EPS=1e-15
CAL=json.loads(urllib.request.urlopen(CAL_URL,timeout=120).read().decode())
MU=np.asarray(CAL["mean"],float); IV=np.asarray(CAL["inv_cov"],float)
SM=float(CAL["scale_mean"]); SS=float(CAL["scale_sd"]); PARENT_D2=161.34337398354887

LAMBDAS=(1.,10.,100.,1000.); GAMMAS=(.25,.5,1.,2.)

lines,fold_meta,nmapped=core["build_parent"]()
lines=core["attach_timescale_features"](lines)
if nmapped!=9616: raise RuntimeError(("MAP_COUNT",nmapped))

def fblock(e):
    return np.r_[np.asarray(e["t1_cum_k12"],float),
                 np.log1p(float(e["t1_prior_lines"])),
                 float(e["t1_prior_lines"]>0)]

def flatten(ls):
    ev=[e for s in ls for e in s]
    P=np.vstack([np.asarray(e["p"],float) for e in ev])
    Y=np.array([int(e["y"]) for e in ev],int)
    return ev,P,Y

def design(ls):
    ev,_,_=flatten(ls)
    return np.vstack([fblock(e) for e in ev])

def standardize(Xtr,Xva,Xte):
    mu=Xtr.mean(0);sd=Xtr.std(0);sd=np.where(sd<1e-8,1.,sd)
    return (Xtr-mu)/sd,(Xva-mu)/sd,(Xte-mu)/sd,mu,sd

def fit(P,Y,X,lam):
    X1=np.c_[np.ones(len(X)),X]
    R=-P.copy();R[np.arange(len(Y)),Y]+=1.
    pen=np.eye(X1.shape[1])*lam;pen[0,0]=1e-9
    B=np.linalg.solve(X1.T@X1+pen,X1.T@R);B-=B.mean(1,keepdims=True)
    return B

def apply(P,X,B,gam):
    X1=np.c_[np.ones(len(X)),X]
    z=np.log(np.maximum(P,EPS))+gam*(X1@B);z-=z.max(1,keepdims=True)
    q=np.exp(z);q/=q.sum(1,keepdims=True);return q

def bits(P,Y):
    return float(-np.log2(np.maximum(P[np.arange(len(Y)),Y],EPS)).sum())

def entry_len_bin(n):
    if n<=1:return 0
    if n<=2:return 1
    if n<=4:return 2
    return 3

def shuffled(ls,seed):
    out=copy.deepcopy(ls);rng=np.random.default_rng(seed)
    # shuffle T1 line-level feature vectors within preregistered physical strata
    groups={}
    for i,s in enumerate(out):
        if not s: continue
        e=s[0]
        key=(str(e["folio"]),int(e["entry_line_index"]),entry_len_bin(int(e["entry_line_count"])))
        groups.setdefault(key,[]).append(i)
    for key,ix in groups.items():
        vals=[(out[i][0]["t1_cum_k12"].copy(),float(out[i][0]["t1_prior_lines"])) for i in ix]
        perm=rng.permutation(len(vals))
        for di,si in enumerate(perm):
            k12,npl=vals[int(si)]
            for e in out[ix[di]]:
                e["t1_cum_k12"]=k12.copy();e["t1_prior_lines"]=npl
    return out

def choose(tr,va):
    etr,Ptr,Ytr=flatten(tr);eva,Pv,Yv=flatten(va)
    Xtr=design(tr);Xv=design(va)
    mu=Xtr.mean(0);sd=Xtr.std(0);sd=np.where(sd<1e-8,1.,sd)
    Xtr=(Xtr-mu)/sd;Xv=(Xv-mu)/sd
    best=None
    for lam in LAMBDAS:
        B=fit(Ptr,Ytr,Xtr,lam)
        for gam in GAMMAS:
            Q=apply(Pv,Xv,B,gam);b=bits(Q,Yv)/len(Yv)
            z=(b,lam,gam,B,mu,sd)
            if best is None or (z[0],z[1],z[2])<(best[0],best[1],best[2]):best=z
    return best

def predict(te,md):
    b,lam,gam,B,mu,sd=md
    ev,P,Y=flatten(te);X=(design(te)-mu)/sd;Q=apply(P,X,B,gam)
    out=[];k=0
    for s in te:
        zz=[]
        for e in s:
            x=dict(e);x["p"]=Q[k];zz.append(x);k+=1
        out.append(zz)
    return out,bits(Q,Y),len(Y),{"lambda":lam,"gamma":gam,"val_bpe":b}

def d2(vec):
    z=np.asarray(vec,float)-MU;return float(z@IV@z)

by={j:[s for s in lines if int(s[0]["fold"])==j] for j in range(5)}
pooled_parent=pooled_real=pooled_shuffle=0.;nn=0;folds=[];allreal=[];allparent=[]
nof115_parent=nof115_real=0.;nof115_n=0

for j in range(5):
    v=(j+1)%5;trks=[k for k in range(5) if k not in (j,v)]
    tr=[s for k in trks for s in by[k]];va=by[v];te=by[j]
    shtr=shuffled(tr,202610083100+j*100+1)
    shva=shuffled(va,202610083100+j*100+2)
    shte=shuffled(te,202610083100+j*100+3)
    mr=choose(tr,va);ms=choose(shtr,shva)
    ore,br,nr,hr=predict(te,mr);osh,bs,ns,hs=predict(shte,ms)
    _,PP,YY=flatten(te);bp=bits(PP,YY);n=len(YY)
    assert n==nr==ns
    pooled_parent+=bp;pooled_real+=br;pooled_shuffle+=bs;nn+=n
    allreal.extend(ore);allparent.extend(te)
    # f115r exclusion predictive sensitivity
    keep=[s for s in te if str(s[0]["folio"])!="f115r"]
    if keep:
        _,P0,Y0=flatten(keep);b0=bits(P0,Y0)
        ok,br0,n0,_=predict(keep,mr)
        nof115_parent+=b0;nof115_real+=br0;nof115_n+=n0
    folds.append({"fold":j,"n":n,"parent_bpe":bp/n,"real_bpe":br/n,"shuffle_bpe":bs/n,
                  "gain_vs_parent":(bp-br)/n,"real_minus_shuffle":(bs-br)/n,
                  "real_hp":hr,"shuffle_hp":hs})

vec,names,nm=r4["metric_vector"](allreal);D=d2(vec);ZD=(D-SM)/SS
pb=pooled_parent/nn;rb=pooled_real/nn;sb=pooled_shuffle/nn
pos=sum(x["gain_vs_parent"]>0 for x in folds)
red=1-D/PARENT_D2
passed=bool(rb<pb and pos>=4 and rb<sb and red>=.20)
out={
 "programme":"VMS-R2B-T1","status":"complete","n":nn,
 "parent_bpe":pb,"real_bpe":rb,"shuffle_bpe":sb,
 "gain_vs_parent":pb-rb,"real_minus_shuffle":sb-rb,
 "positive_folds":pos,
 "D2":D,"Z_D2":ZD,"D2_reduction_fraction":red,
 "vector":np.asarray(vec,float).tolist(),"metric_names":names,
 "f115r_exclusion":{"n":nof115_n,"parent_bpe":nof115_parent/max(nof115_n,1),
                    "real_bpe":nof115_real/max(nof115_n,1),
                    "gain_vs_parent":(nof115_parent-nof115_real)/max(nof115_n,1)},
 "phaseA_pass":passed,
 "decision":"OBSERVED_T1_PHASEA_CANDIDATE" if passed else "OBSERVED_T1_PHASEA_FAIL",
 "folds":folds
}
print("R2B_T1_RESULT="+json.dumps(out,separators=(",",":")),flush=True)
