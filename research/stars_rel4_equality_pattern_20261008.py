#!/usr/bin/env python3
# STARS-REL4 — preregistered 2026-10-08.
import collections,json,math,os,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/cfebbd657e31cc8a49e1513230e7e2c34514fbcc/research/stars_innov2_position_20261008.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode()
ns={"__name__":"innov2_module"}
exec(compile(src,BASE,"exec"),ns)

generate0=ns["generate"];position_adjust=ns["position_adjust"];fit_pos=ns["fit_pos"]
tilt_p=ns["tilt_p"];metric_vector=ns["metric_vector"];fnum=ns["fnum"]
rows=ns["rows"];para_id=ns["para_id"];base_prob=ns["base_prob"];STAR_FOL=ns["STAR_FOL"]
K=ns["K"];role=ns["role"];lbin=ns["lbin"];L2=10.0

A_SEEDS=list(range(202610088200,202610088500))
B_SEEDS=list(range(202610088500,202610088700))
C_SEEDS=list(range(202610088700,202610089000))

REAL_BASE=generate0(True,None)
REAL_Q0=position_adjust(REAL_BASE)
REAL_POSMOD={0:fit_pos(REAL_BASE,0),1:fit_pos(REAL_BASE,1)}
SC_LEN=collections.Counter()
for r in rows:
    if r["folio"] in STAR_FOL and int(r["pos"])>0:
        SC_LEN[(r["folio"],int(r["line"]))]+=1

def fit_A1(lines,parity):
    groups=[[] for _ in range(K)]
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        for t in range(2,len(seq)):
            groups[int(seq[t-1]["y"])].append({"p":seq[t]["p"],"y":int(seq[t]["y"])})
    A=np.zeros((K,K),float)
    for a,es in enumerate(groups):
        if es:A[a]=ns["fit_tilt"](es)
    return A

REAL_A={0:fit_A1(REAL_Q0,0),1:fit_A1(REAL_Q0,1)}

def relation_F(a,b,active):
    F=np.zeros((K,len(active)),float)
    amap={v:j for j,v in enumerate(active)}
    if a==b:
        if 0 in amap:F[b,amap[0]]=1.
    else:
        if 1 in amap:F[a,amap[1]]=1.
        if 2 in amap:F[b,amap[2]]=1.
    return F

def arrays_E(lines,parity,active):
    P=[];Y=[];FF=[]
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        for t in range(2,len(seq)):
            a=int(seq[t-2]["y"]);b=int(seq[t-1]["y"])
            P.append(seq[t]["p"]);Y.append(int(seq[t]["y"]));FF.append(relation_F(a,b,active))
    return np.asarray(P,float),np.asarray(Y,int),np.asarray(FF,float)

def nll_grad_hess(beta,P,Y,F):
    sc=np.log(np.maximum(P,1e-15))+np.tensordot(F,beta,axes=([2],[0]))
    mx=sc.max(1,keepdims=True);Q=np.exp(sc-mx);Q/=Q.sum(1,keepdims=True)
    loss=-float(np.sum(np.log(np.maximum(Q[np.arange(len(Y)),Y],1e-300))))+.5*L2*float(np.dot(beta,beta))
    Ef=np.einsum("nk,nkj->nj",Q,F);Fy=F[np.arange(len(Y)),Y,:]
    grad=(Ef-Fy).sum(0)+L2*beta
    E2=np.einsum("nk,nkj,nkl->jl",Q,F,F)
    H=np.eye(len(beta))*L2+E2-Ef.T@Ef
    return loss,grad,H

def fit_beta(P,Y,F):
    b=np.zeros(F.shape[2],float)
    for _ in range(25):
        loss,g,H=nll_grad_hess(b,P,Y,F)
        if np.max(np.abs(g))<1e-9:break
        step=np.linalg.solve(H+np.eye(len(b))*1e-10,g);t=1.
        while t>1e-6:
            c=b-t*step;nl,_,_=nll_grad_hess(c,P,Y,F)
            if nl<=loss+1e-12:b=c;break
            t*=.5
        if t<=1e-6:break
    return b

def fit_E(lines,parity,active=(0,1,2)):
    P,Y,F=arrays_E(lines,parity,active)
    return fit_beta(P,Y,F)

def apply_E_p(p,beta,a,b,active=(0,1,2)):
    F=relation_F(int(a),int(b),active)
    sc=np.log(np.maximum(np.asarray(p,float),1e-15))+F@np.asarray(beta,float)
    sc-=sc.max();q=np.exp(sc);return q/q.sum()

def score_E(lines,active=(0,1,2)):
    betas={tr:fit_E(lines,tr,active) for tr in (0,1)}
    B=R=N=0.0
    for seq in lines:
        if not seq:continue
        te=fnum(seq[0]["folio"])%2;tr=1-te;b=betas[tr]
        for t in range(2,len(seq)):
            y=int(seq[t]["y"]);p=np.asarray(seq[t]["p"],float)
            q=apply_E_p(p,b,int(seq[t-2]["y"]),int(seq[t-1]["y"]),active)
            B+=-math.log2(max(float(p[y]),1e-300));R+=-math.log2(max(float(q[y]),1e-300));N+=1
    return {"gain":float((B-R)/N),"n":int(N),"betas":{str(k):v.tolist() for k,v in betas.items()}}

REAL_FULL=score_E(REAL_Q0,(0,1,2))
REAL_SINGLE={name:score_E(REAL_Q0,(j,)) for j,name in enumerate(("RUN","RETURN","STAY"))}
REAL_E={0:fit_E(REAL_Q0,0,(0,1,2)),1:fit_E(REAL_Q0,1,(0,1,2))}
print("REAL",json.dumps({"full":REAL_FULL,"singles":REAL_SINGLE},separators=(",",":")),flush=True)

def pos_bias(mod,i,n):
    gb,rb,cb=mod;ro=role(i,n)
    return cb.get((ro,lbin(n)),rb.get(ro,gb))

def generate_param(seed,kind):
    rng=np.random.default_rng(seed)
    page=collections.defaultdict(lambda:np.zeros(K,float))
    para=collections.defaultdict(lambda:np.zeros(K,float))
    hist=collections.defaultdict(list);by=collections.OrderedDict()
    for r in rows:
        fol=r["folio"];ln=int(r["line"]);lk=(fol,ln);pk=fol;pid=para_id(pk,ln);pq=(pk,pid)
        yobs=int(r["start"]);pos=int(r["pos"])
        if pos==0:y=yobs
        else:
            prev_piece=hist[lk][-1][1];rc=np.zeros(K,float)
            for yy,pp in hist[lk][-6:]:rc[int(yy)]+=1
            pbase=base_prob(r["section"],prev_piece,page[pk],para[pq],rc);pgen=pbase
            if fol in STAR_FOL:
                seq=by.setdefault(lk,[]);i=len(seq);n=SC_LEN[lk];te=fnum(fol)%2;tr=1-te
                pgen=tilt_p(pgen,pos_bias(REAL_POSMOD[tr],i,n))
                if i>=2:
                    a=int(seq[i-2]["y"]);b=int(seq[i-1]["y"])
                    if kind=="T1":pgen=tilt_p(pgen,REAL_A[tr][b])
                    elif kind=="E3":pgen=apply_E_p(pgen,REAL_E[tr],a,b,(0,1,2))
                y=int(rng.choice(K,p=pgen))
                seq.append({"p":pbase,"y":y,"prev":int(hist[lk][-1][0]),"folio":fol,
                            "line":lk,"bif":r["bifolium"]})
            else:y=int(rng.choice(K,p=pbase))
        page[pk][y]+=1;para[pq][y]+=1;hist[lk].append((y,r["final_piece"]))
    return list(by.values())

def gain_task(args):
    seed,kind=args;base=generate_param(seed,kind);q0=position_adjust(base)
    full=score_E(q0,(0,1,2))["gain"]
    singles=[score_E(q0,(j,))["gain"] for j in range(3)]
    return seed,[full]+singles

def augment_E(lines):
    betas={tr:fit_E(lines,tr,(0,1,2)) for tr in (0,1)};out=[]
    for seq in lines:
        if not seq:continue
        te=fnum(seq[0]["folio"])%2;tr=1-te;b=betas[tr];zz=[]
        for t,e in enumerate(seq):
            x=dict(e);p=np.asarray(e["p"],float)
            if t>=2:p=apply_E_p(p,b,int(seq[t-2]["y"]),int(seq[t-1]["y"]),(0,1,2))
            x["p"]=p;zz.append(x)
        out.append(zz)
    return out

def analyze_panel(base):
    q0=position_adjust(base);aug=augment_E(q0);v,names,n=metric_vector(aug)
    return {"vector":v,"names":names,"n_events":n,"n_lines":len(aug)}

def panel_task(seed):
    x=analyze_panel(generate_param(seed,"E3"));return seed,x["vector"].tolist()

def setup(A):
    mu=A.mean(0);S=np.cov(A,rowvar=False)
    C=.75*S+.25*np.diag(np.diag(S))+np.eye(A.shape[1])*1e-9
    return mu,np.linalg.pinv(C)
def adjud(F,C,B,r):
    mu,iv=setup(F)
    def d(x):z=x-mu;return float(z@iv@z)
    cd=np.array([d(x) for x in C]);bd=np.array([d(x) for x in B]);rd=d(r)
    q=float(np.quantile(cd,.99));ref=np.r_[cd,bd]
    return {"real_D2":rd,"cal_q99":q,"blind_acceptance":float(np.mean(bd<=q)),
            "add_one_p":float((1+np.sum(ref>=rd))/(len(ref)+1))}

if __name__=="__main__":
    realvec=np.array([REAL_FULL["gain"]]+[REAL_SINGLE[n]["gain"] for n in ("RUN","RETURN","STAY")],float)
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(gain_task,[(s,"Q0") for s in A_SEEDS],chunksize=2))
    A=np.array([v for s,v in rr],float)
    fit=A[:100];cal=A[100:200];blind=A[200:]
    q=float(np.quantile(cal[:,0],.99));ba=float(np.mean(blind[:,0]<=q))
    ref=np.r_[cal[:,0],blind[:,0]];p=float((1+np.sum(ref>=realvec[0]))/(len(ref)+1))
    apass=bool(ba>=.90 and realvec[0]>q and p<=.01)
    secondary={"opened":False}
    if apass:
        mu=fit[:,1:].mean(0);sd=np.maximum(fit[:,1:].std(0,ddof=1),1e-12)
        cz=(cal[:,1:]-mu)/sd;bz=(blind[:,1:]-mu)/sd;rz=(realvec[1:]-mu)/sd
        cm=cz.max(1);bm=bz.max(1);mq=float(np.quantile(cm,.99));mba=float(np.mean(bm<=mq))
        mref=np.r_[cm,bm];mp=float((1+np.sum(mref>=float(rz.max())))/(len(mref)+1))
        perq=np.quantile(cz,.99,axis=0);glob=bool(mba>=.90 and float(rz.max())>mq and mp<=.01)
        names=("RUN","RETURN","STAY")
        secondary={"opened":True,"real_Z":{names[i]:float(rz[i]) for i in range(3)},
                   "per_motif_q99_Z":{names[i]:float(perq[i]) for i in range(3)},
                   "maxZ_q99":mq,"blind_acceptance":mba,"add_one_p":mp,
                   "global_pass":glob,
                   "resolved":{names[i]:bool(glob and rz[i]>perq[i]) for i in range(3)}}
    stageA={"real_gain":float(realvec[0]),"cal_q99":q,"blind_acceptance":ba,"add_one_p":p,
            "decision":("EQUALITY_PATTERN_MECHANISM_PRESENT" if apass else "NO_EQUALITY_PATTERN_MECHANISM_RESOLVED"),
            "secondary":secondary}
    print("STAGE_A",json.dumps(stageA,separators=(",",":")),flush=True)

    stageB={"opened":False};stageC={"opened":False}
    if apass:
        with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
            rr=list(ex.map(gain_task,[(s,"T1") for s in B_SEEDS],chunksize=2))
        V=np.array([v for s,v in rr],float);cal=V[:100,0];blind=V[100:,0]
        bq=float(np.quantile(cal,.99));bba=float(np.mean(blind<=bq))
        bref=np.r_[cal,blind];bp=float((1+np.sum(bref>=realvec[0]))/(len(bref)+1))
        bpass=bool(bba>=.90 and realvec[0]>bq and bp<=.01)
        stageB={"opened":True,"real_gain":float(realvec[0]),"cal_q99":bq,"blind_acceptance":bba,
                "add_one_p":bp,"decision":("EQUALITY_PATTERN_EXCEEDS_FIRST_ORDER_TRANSITION_NULL"
                if bpass else "EQUALITY_PATTERN_NOT_DISTINGUISHED_FROM_FIRST_ORDER")}
        print("STAGE_B",json.dumps(stageB,separators=(",",":")),flush=True)

        realC=analyze_panel(REAL_BASE)
        with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
            rr=list(ex.map(panel_task,C_SEEDS,chunksize=2))
        mp={s:np.asarray(v,float) for s,v in rr}
        F=np.vstack([mp[s] for s in C_SEEDS[:150]])
        C=np.vstack([mp[s] for s in C_SEEDS[150:225]])
        B=np.vstack([mp[s] for s in C_SEEDS[225:]])
        primary=adjud(F,C,B,realC["vector"])
        if primary["blind_acceptance"]<.90:cdec="E3_ABSORPTION_CALIBRATION_FAIL"
        elif primary["real_D2"]<=primary["cal_q99"]:cdec="E3_ABSORBS_INNOV2_RESIDUAL"
        elif primary["add_one_p"]<=.01:cdec="RESIDUAL_SURVIVES_E3"
        else:cdec="E3_ABSORPTION_UNRESOLVED_PVALUE"
        names=realC["names"]
        groups={
          "without_repeat":[i for i,n in enumerate(names) if not n.startswith("repeat_resid_")],
          "without_transition":[i for i,n in enumerate(names) if n not in ("transition_resid_norm","sv1_energy","sv12_energy")],
          "without_surprise":[i for i,n in enumerate(names) if not(n.startswith("surprise_ac_") or n in ("mean_excess_surprise","pearson_energy"))]
        }
        abl={g:adjud(F[:,ix],C[:,ix],B[:,ix],realC["vector"][ix]) for g,ix in groups.items()}
        stageC={"opened":True,"real":{"n_events":realC["n_events"],"n_lines":realC["n_lines"],
                                      "metric_names":names,"vector":realC["vector"].tolist()},
                "primary":primary,"decision":cdec,"ablations":abl}
        print("STAGE_C",json.dumps(stageC,separators=(",",":")),flush=True)

    out={"programme":"STARS-REL4","status":"complete","real":{"full":REAL_FULL,"singles":REAL_SINGLE},
         "stageA":stageA,"stageB":stageB,"stageC":stageC}
    print("STARS_REL4_JSON="+json.dumps(out,separators=(",",":")),flush=True)
