#!/usr/bin/env python3
# STARS-REL3 — preregistered 2026-10-08.
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
K=ns["K"];role=ns["role"];lbin=ns["lbin"]

A_SEEDS=list(range(202610087500,202610087700))
B_SEEDS=list(range(202610087700,202610087900))
C_SEEDS=list(range(202610087900,202610088200))

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
            a=int(seq[t-1]["y"])
            groups[a].append({"p":seq[t]["p"],"y":int(seq[t]["y"])})
    A=np.zeros((K,K),float)
    for a,es in enumerate(groups):
        if es:A[a]=ns["fit_tilt"](es)
    return A

def fit_B2(lines,parity):
    groups={}
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        for t in range(2,len(seq)):
            c=(int(seq[t-2]["y"]),int(seq[t-1]["y"]))
            groups.setdefault(c,[]).append({"p":seq[t]["p"],"y":int(seq[t]["y"])})
    B=np.zeros((K,K,K),float)
    for (a,b),es in groups.items():
        B[a,b]=ns["fit_tilt"](es)
    return B

def fit_models(lines):
    out={}
    for tr in (0,1):
        out[tr]={"A1":fit_A1(lines,tr),"B2":fit_B2(lines,tr)}
    return out

REAL_MODELS=fit_models(REAL_Q0)

def score_crossfit(lines):
    models=fit_models(lines);B0=B1=B2=N=0.0
    for seq in lines:
        if not seq:continue
        te=fnum(seq[0]["folio"])%2;tr=1-te
        A=models[tr]["A1"];C=models[tr]["B2"]
        for t in range(2,len(seq)):
            y=int(seq[t]["y"]);p0=np.asarray(seq[t]["p"],float)
            a=int(seq[t-2]["y"]);b=int(seq[t-1]["y"])
            p1=tilt_p(p0,A[b]);p2=tilt_p(p0,C[a,b])
            B0+=-math.log2(max(float(p0[y]),1e-300))
            B1+=-math.log2(max(float(p1[y]),1e-300))
            B2+=-math.log2(max(float(p2[y]),1e-300))
            N+=1
    return {"G1":float((B0-B1)/N),"G2":float((B0-B2)/N),
            "D21":float((B1-B2)/N),"n":int(N),"models":models}

REAL_SCORE=score_crossfit(REAL_Q0)
print("REAL_SCORE",json.dumps({k:v for k,v in REAL_SCORE.items() if k!="models"},separators=(",",":")),flush=True)

def pos_bias(mod,i,n):
    gb,rb,cb=mod;ro=role(i,n)
    return cb.get((ro,lbin(n)),rb.get(ro,gb))

def generate_param(seed,kind):
    # kind Q0, T1 or C2. Generative context parameters are fixed real cross-fit fits.
    rng=np.random.default_rng(seed)
    page=collections.defaultdict(lambda:np.zeros(K,float))
    para=collections.defaultdict(lambda:np.zeros(K,float))
    hist=collections.defaultdict(list)
    by=collections.OrderedDict()
    for r in rows:
        fol=r["folio"];ln=int(r["line"]);lk=(fol,ln);pk=fol;pid=para_id(pk,ln);pq=(pk,pid)
        yobs=int(r["start"]);pos=int(r["pos"])
        if pos==0:
            y=yobs
        else:
            prev_piece=hist[lk][-1][1];rc=np.zeros(K,float)
            for yy,pp in hist[lk][-6:]:rc[int(yy)]+=1
            pbase=base_prob(r["section"],prev_piece,page[pk],para[pq],rc)
            pgen=pbase
            if fol in STAR_FOL:
                seq=by.setdefault(lk,[])
                i=len(seq);n=SC_LEN[lk];te=fnum(fol)%2;tr=1-te
                pgen=tilt_p(pgen,pos_bias(REAL_POSMOD[tr],i,n))
                if i>=2:
                    a=int(seq[i-2]["y"]);b=int(seq[i-1]["y"])
                    if kind=="T1":
                        pgen=tilt_p(pgen,REAL_MODELS[tr]["A1"][b])
                    elif kind=="C2":
                        pgen=tilt_p(pgen,REAL_MODELS[tr]["B2"][a,b])
                y=int(rng.choice(K,p=pgen))
                seq.append({"p":pbase,"y":y,"prev":int(hist[lk][-1][0]),"folio":fol,
                            "line":lk,"bif":r["bifolium"]})
            else:
                y=int(rng.choice(K,p=pbase))
        page[pk][y]+=1;para[pq][y]+=1;hist[lk].append((y,r["final_piece"]))
    return list(by.values())

def score_task(args):
    seed,kind=args
    base=generate_param(seed,kind);q0=position_adjust(base);s=score_crossfit(q0)
    return seed,[s["G1"],s["G2"],s["D21"]]

def augment_C2(lines,models):
    out=[]
    for seq in lines:
        if not seq:continue
        te=fnum(seq[0]["folio"])%2;tr=1-te;C=models[tr]["B2"];zz=[]
        for t,e in enumerate(seq):
            x=dict(e);p=np.asarray(e["p"],float)
            if t>=2:p=tilt_p(p,C[int(seq[t-2]["y"]),int(seq[t-1]["y"])])
            x["p"]=p;zz.append(x)
        out.append(zz)
    return out

def analyze_C2_panel(base):
    q0=position_adjust(base);models=fit_models(q0);aug=augment_C2(q0,models)
    v,names,n=metric_vector(aug)
    return {"vector":v,"names":names,"n_events":n,"n_lines":len(aug)}

def panel_task(seed):
    base=generate_param(seed,"C2");x=analyze_C2_panel(base)
    return seed,x["vector"].tolist()

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
    # Stage A: C2 against q0
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(score_task,[(s,"Q0") for s in A_SEEDS],chunksize=2))
    Avals=np.array([v for s,v in rr],float)
    cal=Avals[:100,1];blind=Avals[100:,1]
    aq=float(np.quantile(cal,.99));aba=float(np.mean(blind<=aq))
    ap=float((1+np.sum(Avals[:,1]>=REAL_SCORE["G2"]))/(len(Avals)+1))
    apass=bool(aba>=.90 and REAL_SCORE["G2"]>aq and ap<=.01)
    stageA={"real_G2":REAL_SCORE["G2"],"real_G1":REAL_SCORE["G1"],"real_D21":REAL_SCORE["D21"],
            "cal_q99":aq,"blind_acceptance":aba,"add_one_p":ap,
            "decision":("SECOND_ORDER_CONTEXT_OVER_COMPACT_SELECT" if apass else "NO_SECOND_ORDER_CONTEXT_RESOLVED")}
    print("STAGE_A",json.dumps(stageA,separators=(",",":")),flush=True)

    stageB={"opened":False};stageC={"opened":False}
    if apass:
        # Stage B: C2 against T1 parametric null
        with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
            rr=list(ex.map(score_task,[(s,"T1") for s in B_SEEDS],chunksize=2))
        Bvals=np.array([v for s,v in rr],float)
        cal=Bvals[:100,2];blind=Bvals[100:,2]
        bq=float(np.quantile(cal,.99));bba=float(np.mean(blind<=bq))
        bp=float((1+np.sum(Bvals[:,2]>=REAL_SCORE["D21"]))/(len(Bvals)+1))
        bpass=bool(bba>=.90 and REAL_SCORE["D21"]>bq and bp<=.01)
        stageB={"opened":True,"real_D21":REAL_SCORE["D21"],"cal_q99":bq,
                "blind_acceptance":bba,"add_one_p":bp,
                "decision":("SECOND_ORDER_CONTEXT_PRESENT_BEYOND_FIRST_ORDER" if bpass
                            else "SECOND_ORDER_CONTEXT_NOT_DISTINGUISHED_FROM_FIRST_ORDER")}
        print("STAGE_B",json.dumps(stageB,separators=(",",":")),flush=True)

        # Stage C: C2 absorption
        realC=analyze_C2_panel(REAL_BASE)
        with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
            rr=list(ex.map(panel_task,C_SEEDS,chunksize=2))
        mp={s:np.asarray(v,float) for s,v in rr}
        F=np.vstack([mp[s] for s in C_SEEDS[:150]])
        C=np.vstack([mp[s] for s in C_SEEDS[150:225]])
        B=np.vstack([mp[s] for s in C_SEEDS[225:]])
        primary=adjud(F,C,B,realC["vector"])
        if primary["blind_acceptance"]<.90:cdec="C2_ABSORPTION_CALIBRATION_FAIL"
        elif primary["real_D2"]<=primary["cal_q99"]:cdec="C2_ABSORBS_INNOV2_RESIDUAL"
        elif primary["add_one_p"]<=.01:cdec="RESIDUAL_SURVIVES_C2"
        else:cdec="C2_ABSORPTION_UNRESOLVED_PVALUE"
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

    out={"programme":"STARS-REL3","status":"complete","real_score":{k:v for k,v in REAL_SCORE.items() if k!="models"},
         "stageA":stageA,"stageB":stageB,"stageC":stageC}
    print("STARS_REL3_JSON="+json.dumps(out,separators=(",",":")),flush=True)
