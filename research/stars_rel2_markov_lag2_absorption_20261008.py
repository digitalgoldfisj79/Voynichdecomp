#!/usr/bin/env python3
# STARS-REL2 — preregistered 2026-10-08.
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

A_SEEDS=list(range(202610087000,202610087200))
B_SEEDS=list(range(202610087200,202610087500))

# Canonical observed base and position-adjusted target.
REAL_BASE=generate0(True,None)
REAL_Q0=position_adjust(REAL_BASE)
REAL_POSMOD={0:fit_pos(REAL_BASE,0),1:fit_pos(REAL_BASE,1)}

# Scored-event line lengths, known from manuscript structure.
SC_LEN=collections.Counter()
for r in rows:
    if r["folio"] in STAR_FOL and int(r["pos"])>0:
        SC_LEN[(r["folio"],int(r["line"]))]+=1

def fit_A(lines,parity):
    groups=[[] for _ in range(K)]
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity: continue
        # preregistered common mechanism population t>=2
        for t in range(2,len(seq)):
            a=int(seq[t-1]["y"])
            groups[a].append({"p":seq[t]["p"],"y":int(seq[t]["y"])})
    A=np.zeros((K,K),float)
    for a,es in enumerate(groups):
        if es:
            A[a]=ns["fit_tilt"](es)
    return A

def lagtilt(p,beta,lagcls):
    sc=np.log(np.maximum(np.asarray(p,float),1e-15))
    sc[int(lagcls)]+=float(beta)
    sc-=sc.max();q=np.exp(sc);return q/q.sum()

def fit_beta2(lines,parity,A):
    ps=[];ys=[];ls=[]
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        for t in range(2,len(seq)):
            a=int(seq[t-1]["y"]);q=tilt_p(seq[t]["p"],A[a])
            ps.append(q);ys.append(int(seq[t]["y"]));ls.append(int(seq[t-2]["y"]))
    P=np.asarray(ps,float);Y=np.asarray(ys,int);L=np.asarray(ls,int)
    b=0.0
    for _ in range(30):
        qlag=P[np.arange(len(P)),L]
        eb=math.exp(max(-30.,min(30.,b)))
        den=1.0+qlag*(eb-1.0)
        prob=qlag*eb/den
        obs=(Y==L).astype(float)
        g=float(np.sum(prob-obs)+L2*b)
        h=float(np.sum(prob*(1-prob))+L2)
        step=g/max(h,1e-12)
        nb=b-step
        if abs(nb-b)<1e-10:b=nb;break
        b=nb
    return float(b)

def fit_models(lines):
    out={}
    for tr in (0,1):
        A=fit_A(lines,tr);b=fit_beta2(lines,tr,A)
        out[tr]={"A":A,"beta":b}
    return out

REAL_MODELS=fit_models(REAL_Q0)

def score_crossfit(lines):
    models=fit_models(lines)
    B=T=T2=N=0.0
    for seq in lines:
        if not seq:continue
        te=fnum(seq[0]["folio"])%2;tr=1-te;A=models[tr]["A"];b=models[tr]["beta"]
        for t in range(2,len(seq)):
            y=int(seq[t]["y"]);p0=np.asarray(seq[t]["p"],float)
            a=int(seq[t-1]["y"]);lag=int(seq[t-2]["y"])
            pt=tilt_p(p0,A[a]);pt2=lagtilt(pt,b,lag)
            B+=-math.log2(max(float(p0[y]),1e-300))
            T+=-math.log2(max(float(pt[y]),1e-300))
            T2+=-math.log2(max(float(pt2[y]),1e-300))
            N+=1
    return {"gain_T1":float((B-T)/N),"gain_L2_given_T1":float((T-T2)/N),
            "n":int(N),"models":models}

REAL_SCORE=score_crossfit(REAL_Q0)
print("REAL_STAGE_A",json.dumps({"gain_T1":REAL_SCORE["gain_T1"],
      "gain_L2_given_T1":REAL_SCORE["gain_L2_given_T1"],"n":REAL_SCORE["n"],
      "beta_train0":REAL_SCORE["models"][0]["beta"],"beta_train1":REAL_SCORE["models"][1]["beta"]},
      separators=(",",":")),flush=True)

def pos_bias(mod,i,n):
    gb,rb,cb=mod;ro=role(i,n)
    return cb.get((ro,lbin(n)),rb.get(ro,gb))

def generate_param(seed,kind):
    # kind: T1 or T12. Generative parameters are frozen real cross-fit models.
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
                    A=REAL_MODELS[tr]["A"];a=int(seq[i-1]["y"])
                    pgen=tilt_p(pgen,A[a])
                    if kind=="T12":
                        pgen=lagtilt(pgen,REAL_MODELS[tr]["beta"],int(seq[i-2]["y"]))
                y=int(rng.choice(K,p=pgen))
                e={"p":pbase,"y":y,"prev":int(hist[lk][-1][0]),"folio":fol,
                   "line":lk,"bif":r["bifolium"]}
                seq.append(e)
            else:
                y=int(rng.choice(K,p=pbase))
        page[pk][y]+=1;para[pq][y]+=1;hist[lk].append((y,r["final_piece"]))
    return list(by.values())

def stageA_task(seed):
    base=generate_param(seed,"T1");q0=position_adjust(base);s=score_crossfit(q0)
    return seed,float(s["gain_L2_given_T1"])

def augment_crossfit(lines,models,kind):
    out=[]
    for seq in lines:
        if not seq:continue
        te=fnum(seq[0]["folio"])%2;tr=1-te;A=models[tr]["A"];b=models[tr]["beta"];zz=[]
        for t,e in enumerate(seq):
            x=dict(e);p=np.asarray(e["p"],float)
            if t>=2:
                p=tilt_p(p,A[int(seq[t-1]["y"])])
                if kind=="T12":p=lagtilt(p,b,int(seq[t-2]["y"]))
            x["p"]=p;zz.append(x)
        out.append(zz)
    return out

def analyze_panel(base,kind):
    q0=position_adjust(base);models=fit_models(q0)
    aug=augment_crossfit(q0,models,kind)
    v,names,n=metric_vector(aug)
    return {"vector":v,"names":names,"n_events":n,"n_lines":len(aug),
            "betas":{str(k):float(models[k]["beta"]) for k in models}}

def stageB_task(args):
    seed,kind=args
    base=generate_param(seed,kind);x=analyze_panel(base,kind)
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
    # Stage A
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(stageA_task,A_SEEDS,chunksize=2))
    vals=np.array([v for s,v in rr],float)
    cal=vals[:100];blind=vals[100:];q99=float(np.quantile(cal,.99))
    ba=float(np.mean(blind<=q99));p=float((1+np.sum(vals>=REAL_SCORE["gain_L2_given_T1"]))/(len(vals)+1))
    if ba<.90:
        adecision="MARKOV_LAG2_METRIC_UNRESOLVED"
        selected=None
    elif REAL_SCORE["gain_L2_given_T1"]>q99 and p<=.01:
        adecision="GENUINE_LAG2_BEYOND_FIRST_ORDER_TRANSITION"
        selected="T12"
    else:
        adecision="LAG2_NOT_RESOLVED_BEYOND_FIRST_ORDER_TRANSITION"
        selected="T1"
    stageA={"real_gain_L2_given_T1":float(REAL_SCORE["gain_L2_given_T1"]),
            "real_gain_T1":float(REAL_SCORE["gain_T1"]),"cal_q99":q99,
            "blind_acceptance":ba,"add_one_p":p,"decision":adecision,
            "selected_model":selected}
    print("STAGE_A",json.dumps(stageA,separators=(",",":")),flush=True)

    stageB={"opened":False}
    if selected is not None:
        realB=analyze_panel(REAL_BASE,selected)
        args=[(s,selected) for s in B_SEEDS]
        with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
            rr=list(ex.map(stageB_task,args,chunksize=2))
        mp={s:np.asarray(v,float) for s,v in rr}
        F=np.vstack([mp[s] for s in B_SEEDS[:150]])
        C=np.vstack([mp[s] for s in B_SEEDS[150:225]])
        B=np.vstack([mp[s] for s in B_SEEDS[225:]])
        primary=adjud(F,C,B,realB["vector"])
        if primary["blind_acceptance"]<.90:bdecision="ABSORPTION_METRIC_CALIBRATION_FAIL"
        elif primary["real_D2"]<=primary["cal_q99"]:bdecision="SELECTED_RELATIONAL_MODEL_ABSORBS_INNOV2_RESIDUAL"
        elif primary["add_one_p"]<=.01:bdecision="RESIDUAL_STRUCTURE_SURVIVES_SELECTED_RELATIONAL_MODEL"
        else:bdecision="UNRESOLVED_PVALUE"
        names=realB["names"]
        groups={
          "without_repeat":[i for i,n in enumerate(names) if not n.startswith("repeat_resid_")],
          "without_transition":[i for i,n in enumerate(names) if n not in ("transition_resid_norm","sv1_energy","sv12_energy")],
          "without_surprise":[i for i,n in enumerate(names) if not(n.startswith("surprise_ac_") or n in ("mean_excess_surprise","pearson_energy"))]
        }
        abl={g:adjud(F[:,ix],C[:,ix],B[:,ix],realB["vector"][ix]) for g,ix in groups.items()}
        stageB={"opened":True,"selected_model":selected,
                "real":{"n_events":realB["n_events"],"n_lines":realB["n_lines"],
                        "metric_names":names,"vector":realB["vector"].tolist(),"betas":realB["betas"]},
                "primary":primary,"decision":bdecision,"ablations":abl}
        print("STAGE_B",json.dumps(stageB,separators=(",",":")),flush=True)

    out={"programme":"STARS-REL2","status":"complete","stageA":stageA,"stageB":stageB}
    print("STARS_REL2_JSON="+json.dumps(out,separators=(",",":")),flush=True)
