#!/usr/bin/env python3
# STARS-TEMP1 — preregistered 2026-10-08.
import collections,json,math,os,urllib.request
import numpy as np
from scipy.optimize import minimize_scalar
from concurrent.futures import ProcessPoolExecutor

REL4URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/87dff8d519d18bf2a080f28dc7aeb2cf25ea6ed7/research/stars_rel4_equality_pattern_20261008.py"
src=urllib.request.urlopen(REL4URL,timeout=120).read().decode()
r={"__name__":"rel4_module"}
exec(compile(src,REL4URL,"exec"),r)

REAL_BASE=r["REAL_BASE"];position_adjust=r["position_adjust"];augment_E=r["augment_E"]
generate_E3=r["generate_param"];fnum=r["fnum"];metric_vector=r["metric_vector"]
rows=r["rows"];para_id=r["para_id"];base_prob=r["base_prob"];STAR_FOL=r["STAR_FOL"]
K=r["K"];SC_LEN=r["SC_LEN"];REAL_POSMOD=r["REAL_POSMOD"];REAL_E=r["REAL_E"]
tilt_p=r["tilt_p"];pos_bias=r["pos_bias"]

A_SEEDS=list(range(202610089700,202610089900))
B_SEEDS=list(range(202610089900,202610090200))
THETA_L2_HALF=5.0

def temp_p(p,theta):
    lam=math.exp(float(theta));sc=lam*np.log(np.maximum(np.asarray(p,float),1e-15))
    sc-=sc.max();q=np.exp(sc);return q/q.sum()

def fit_theta(lines,parity):
    P=[];Y=[]
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        for e in seq:P.append(e["p"]);Y.append(int(e["y"]))
    P=np.asarray(P,float);Y=np.asarray(Y,int);LP=np.log(np.maximum(P,1e-15))
    def obj(th):
        lam=math.exp(float(th));sc=lam*LP;mx=sc.max(1,keepdims=True)
        z=mx[:,0]+np.log(np.exp(sc-mx).sum(1))
        return -float(np.sum(sc[np.arange(len(Y)),Y]-z))+THETA_L2_HALF*float(th*th)
    z=minimize_scalar(obj,bounds=(-3.,3.),method="bounded",
                      options={"xatol":1e-10,"maxiter":200})
    return float(z.x)

def temp_crossfit(lines):
    th={0:fit_theta(lines,0),1:fit_theta(lines,1)}
    out=[];B=R=N=0.0
    for seq in lines:
        if not seq:continue
        te=fnum(seq[0]["folio"])%2;tr=1-te;t=th[tr];zz=[]
        for e in seq:
            p=np.asarray(e["p"],float);q=temp_p(p,t);y=int(e["y"])
            B+=-math.log2(max(float(p[y]),1e-300));R+=-math.log2(max(float(q[y]),1e-300));N+=1
            x=dict(e);x["p"]=q;zz.append(x)
        out.append(zz)
    return out,th,float((B-R)/N),int(N)

def analyze(base):
    q0=position_adjust(base);e3=augment_E(q0);qt,th,g,n=temp_crossfit(e3)
    v,names,ne=metric_vector(qt)
    idx=[i for i,x in enumerate(names) if x.startswith("surprise_ac_") or x in ("mean_excess_surprise","pearson_energy")]
    return {"gain":g,"n":n,"theta":th,
            "lambda":{k:math.exp(vv) for k,vv in th.items()},
            "surprise":v[idx],"surprise_names":[names[i] for i in idx],
            "full":v,"full_names":names,"lines":qt}

REAL=analyze(REAL_BASE)
print("REAL_TEMP1="+json.dumps({"gain":REAL["gain"],"n":REAL["n"],"theta":REAL["theta"],
      "lambda":REAL["lambda"],"surprise":dict(zip(REAL["surprise_names"],REAL["surprise"].tolist()))},
      separators=(",",":")),flush=True)

def stageA_task(seed):
    x=analyze(generate_E3(seed,"E3"));return seed,x["gain"]

# Generate from fixed real E3 + fixed real cross-fitted TEMP1.
def generate_temp(seed):
    rng=np.random.default_rng(seed)
    page=collections.defaultdict(lambda:np.zeros(K,float))
    para=collections.defaultdict(lambda:np.zeros(K,float))
    hist=collections.defaultdict(list);by=collections.OrderedDict()
    # real temp fit is training parity -> opposite target parity.
    RTH=REAL["theta"]
    for rr in rows:
        fol=rr["folio"];ln=int(rr["line"]);lk=(fol,ln);pk=fol;pid=para_id(pk,ln);pq=(pk,pid)
        yobs=int(rr["start"]);pos=int(rr["pos"])
        if pos==0:y=yobs
        else:
            prev_piece=hist[lk][-1][1];rc=np.zeros(K,float)
            for yy,pp in hist[lk][-6:]:rc[int(yy)]+=1
            pbase=base_prob(rr["section"],prev_piece,page[pk],para[pq],rc);pgen=pbase
            if fol in STAR_FOL:
                seq=by.setdefault(lk,[]);i=len(seq);n=SC_LEN[lk];te=fnum(fol)%2;tr=1-te
                pgen=tilt_p(pgen,pos_bias(REAL_POSMOD[tr],i,n))
                if i>=2:
                    pgen=r["apply_E_p"](pgen,REAL_E[tr],int(seq[i-2]["y"]),int(seq[i-1]["y"]),(0,1,2))
                pgen=temp_p(pgen,RTH[tr])
                y=int(rng.choice(K,p=pgen))
                seq.append({"p":pbase,"y":y,"prev":int(hist[lk][-1][0]),"folio":fol,
                            "line":lk,"bif":rr["bifolium"]})
            else:y=int(rng.choice(K,p=pbase))
        page[pk][y]+=1;para[pq][y]+=1;hist[lk].append((y,rr["final_piece"]))
    return list(by.values())

def stageB_task(seed):
    x=analyze(generate_temp(seed));return seed,x["surprise"].tolist()

def setup(A):
    mu=A.mean(0);S=np.cov(A,rowvar=False)
    C=.75*S+.25*np.diag(np.diag(S))+np.eye(A.shape[1])*1e-9
    return mu,np.linalg.pinv(C)
def adjud(F,C,B,rvec):
    mu,iv=setup(F)
    def d(x):z=x-mu;return float(z@iv@z)
    cd=np.array([d(x) for x in C]);bd=np.array([d(x) for x in B]);rd=d(rvec)
    q=float(np.quantile(cd,.99));ref=np.r_[cd,bd]
    return {"real_D2":rd,"cal_q99":q,"blind_acceptance":float(np.mean(bd<=q)),
            "add_one_p":float((1+np.sum(ref>=rd))/(len(ref)+1))}

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(stageA_task,A_SEEDS,chunksize=2))
    av=np.asarray([v for s,v in rr],float);cal=av[:100];blind=av[100:]
    q=float(np.quantile(cal,.99));ba=float(np.mean(blind<=q))
    p=float((1+np.sum(av>=REAL["gain"]))/(len(av)+1))
    apass=bool(ba>=.90 and REAL["gain"]>q and p<=.01)
    stageA={"real_gain":REAL["gain"],"cal_q99":q,"blind_acceptance":ba,"add_one_p":p,
            "theta":{str(k):float(v) for k,v in REAL["theta"].items()},
            "lambda":{str(k):float(v) for k,v in REAL["lambda"].items()},
            "decision":("STATIC_TEMPERATURE_CALIBRATION_PRESENT" if apass
                        else "NO_STATIC_TEMPERATURE_MISCALE_RESOLVED")}
    print("STAGE_A",json.dumps(stageA,separators=(",",":")),flush=True)

    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(stageB_task,B_SEEDS,chunksize=2))
    mp={s:np.asarray(v,float) for s,v in rr}
    F=np.vstack([mp[s] for s in B_SEEDS[:150]])
    C=np.vstack([mp[s] for s in B_SEEDS[150:225]])
    B=np.vstack([mp[s] for s in B_SEEDS[225:]])
    primary=adjud(F,C,B,REAL["surprise"])
    if primary["blind_acceptance"]<.90:bdec="TEMP1_SURPRISE_CALIBRATION_FAIL"
    elif primary["real_D2"]<=primary["cal_q99"]:bdec="TEMP1_ABSORBS_SURPRISE_FAMILY"
    elif primary["add_one_p"]<=.01:bdec="SURPRISE_STRUCTURE_SURVIVES_TEMP1"
    else:bdec="TEMP1_SURPRISE_UNRESOLVED_PVALUE"

    # Stage C frozen component localization.
    mu=F.mean(0);sd=np.maximum(F.std(0,ddof=1),1e-12)
    rz=(REAL["surprise"]-mu)/sd;cz=np.abs((C-mu)/sd);bz=np.abs((B-mu)/sd)
    cm=cz.max(1);bm=bz.max(1);mq=float(np.quantile(cm,.99));mba=float(np.mean(bm<=mq))
    ref=np.r_[cm,bm];mpv=float((1+np.sum(ref>=float(np.abs(rz).max())))/(len(ref)+1))
    perq=np.quantile(cz,.99,axis=0);glob=bool(mba>=.90 and float(np.abs(rz).max())>mq and mpv<=.01)
    names=REAL["surprise_names"]
    stageC={"real_Z":{names[i]:float(rz[i]) for i in range(len(names))},
            "per_metric_q99_absZ":{names[i]:float(perq[i]) for i in range(len(names))},
            "maxAbsZ_q99":mq,"blind_acceptance":mba,"add_one_p":mpv,"global_pass":glob,
            "resolved":{names[i]:bool(glob and abs(rz[i])>perq[i]) for i in range(len(names))}}
    stageB={"primary":primary,"decision":bdec,
            "real_metrics":dict(zip(REAL["surprise_names"],REAL["surprise"].tolist())),
            "localization":stageC}
    print("STAGE_B",json.dumps(stageB,separators=(",",":")),flush=True)

    out={"programme":"STARS-TEMP1","status":"complete","stageA":stageA,"stageB":stageB}
    print("STARS_TEMP1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
