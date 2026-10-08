#!/usr/bin/env python3
# ABA-REP1 — preregistered 2026-10-08.
import collections,json,math,os,re,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/cfebbd657e31cc8a49e1513230e7e2c34514fbcc/research/stars_innov2_position_20261008.py"
src=urllib.request.urlopen(BASE,timeout=120).read().decode()
ns={"__name__":"innov2_module"}
exec(compile(src,BASE,"exec"),ns)

rows=ns["rows"];para_id=ns["para_id"];base_prob=ns["base_prob"];K=ns["K"]
fnum=ns["fnum"];role=ns["role"];lbin=ns["lbin"];fit_tilt=ns["fit_tilt"];tilt_p=ns["tilt_p"]
L2=10.0
BETA_STAR=0.1816323337305774

HA=set("""f10r f10v f11r f11v f13r f13v f14r f14v f15r f15v f16r f16v f17r f17v f18r f18v f19r f19v f1v f20r f20v f21r f21v f22r f22v f23r f23v f24r f24v f25r f25v f26r f26v f27r f27v f28r f28v f29r f29v f2r f2v f30r f30v f31r f31v f32r f32v f33r f33v f34r f34v f35r f35v f36r f36v f37r f37v f38r f38v f39r f39v f3r f3v f40r f40v f41r f41v f42r f42v f43r f43v f44r f44v f45r f45v f46r f46v f47r f47v f48r f48v f49r f49v f4r f4v f50r f50v f51r f51v f52r f52v f53r f53v f54r f54v f55r f55v f56r f56v f57r f5r f5v f65r f65v f66v f6r f6v f7r f7v f8r f8v f9r f9v""".split())
Q13=set("""f75r f75v f76v f77r f77v f78r f78v f79r f79v f80r f80v f81r f81v f82r f82v f83r f83v f84r f84v""".split())
TARGETS={"Herbal-A":HA,"Q13":Q13}
ALL_TARGET=HA|Q13

A_SEEDS=list(range(202610089000,202610089300))
B_SEEDS={"Herbal-A":list(range(202610089300,202610089500)),
         "Q13":list(range(202610089500,202610089700))}

# Ordinary scored-event counts per physical line.
SC_LEN=collections.Counter()
for r in rows:
    if r["folio"] in ALL_TARGET and int(r["pos"])>0:
        SC_LEN[(r["folio"],int(r["line"]))]+=1

def generate_all(observed=False,seed=None,gen_section=None,Agen=None,posmods=None):
    rng=np.random.default_rng(seed if seed is not None else 12345)
    page=collections.defaultdict(lambda:np.zeros(K,float))
    para=collections.defaultdict(lambda:np.zeros(K,float))
    hist=collections.defaultdict(list)
    by={s:collections.OrderedDict() for s in TARGETS}
    for r in rows:
        fol=r["folio"];ln=int(r["line"]);lk=(fol,ln);pk=fol;pid=para_id(pk,ln);pq=(pk,pid)
        yobs=int(r["start"]);pos=int(r["pos"])
        if pos==0:
            y=yobs
        else:
            prev_piece=hist[lk][-1][1];rc=np.zeros(K,float)
            for yy,pp in hist[lk][-6:]:rc[int(yy)]+=1
            pbase=base_prob(r["section"],prev_piece,page[pk],para[pq],rc);pgen=pbase
            section=None
            if fol in HA:section="Herbal-A"
            elif fol in Q13:section="Q13"
            if section is not None:
                seq=by[section].setdefault(lk,[])
                # Optional fixed-real T1 generative mechanism for Stage B.
                if gen_section==section and Agen is not None and posmods is not None:
                    i=len(seq);n=SC_LEN[lk];te=fnum(fol)%2;tr=1-te
                    pgen=tilt_p(pgen,pos_bias(posmods[tr],i,n))
                    if i>=2:
                        pgen=tilt_p(pgen,Agen[tr][int(seq[i-1]["y"])])
                y=yobs if observed else int(rng.choice(K,p=pgen))
                seq.append({"p":pbase,"y":y,"prev":int(hist[lk][-1][0]),
                            "folio":fol,"line":lk,"bif":r["bifolium"]})
            else:
                y=yobs if observed else int(rng.choice(K,p=pbase))
        page[pk][y]+=1;para[pq][y]+=1;hist[lk].append((y,r["final_piece"]))
    return {s:list(by[s].values()) for s in TARGETS}

def fit_pos_target(lines,parity):
    allv=[];byrole=collections.defaultdict(list);bycell=collections.defaultdict(list)
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        n=len(seq)
        for i,e in enumerate(seq):
            ro=role(i,n);allv.append(e);byrole[ro].append(e);bycell[(ro,lbin(n))].append(e)
    globalb=fit_tilt(allv)
    rb={ro:fit_tilt(es) for ro,es in byrole.items() if len(es)>=40}
    cb={k:fit_tilt(es) for k,es in bycell.items() if len(es)>=20}
    return globalb,rb,cb

def pos_bias(mod,i,n):
    gb,rb,cb=mod;ro=role(i,n)
    return cb.get((ro,lbin(n)),rb.get(ro,gb))

def position_adjust_target(lines):
    mods={0:fit_pos_target(lines,0),1:fit_pos_target(lines,1)}
    out=[]
    for seq in lines:
        if not seq:continue
        te=fnum(seq[0]["folio"])%2;tr=1-te;gb,rb,cb=mods[tr];n=len(seq);zz=[]
        for i,e in enumerate(seq):
            ro=role(i,n);b=cb.get((ro,lbin(n)),rb.get(ro,gb))
            x=dict(e);x["p"]=tilt_p(e["p"],b);zz.append(x)
        out.append(zz)
    return out,mods

def eligible_records(lines):
    rec=[]
    for seq in lines:
        for t in range(2,len(seq)):
            a=int(seq[t-2]["y"]);b=int(seq[t-1]["y"])
            if a==b:continue
            rec.append((seq[t],a,b))
    return rec

def aba_score(lines):
    rec=eligible_records(lines)
    z=[(1.0 if int(e["y"])==a else 0.0)-float(e["p"][a]) for e,a,b in rec]
    return float(np.mean(z)) if z else 0.0,len(z)

def fixed_beta_gain(lines,beta=BETA_STAR):
    rec=eligible_records(lines);B=R=0.0
    for e,a,b in rec:
        p=np.asarray(e["p"],float);q=p.copy()
        w=math.exp(beta);den=1.0+p[a]*(w-1.0);q/=den;q[a]*=w
        y=int(e["y"])
        B+=-math.log2(max(float(p[y]),1e-300))
        R+=-math.log2(max(float(q[y]),1e-300))
    return float((B-R)/len(rec)) if rec else 0.0

def fit_return_beta(lines,parity):
    P=[];Y=[];A=[]
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        for t in range(2,len(seq)):
            a=int(seq[t-2]["y"]);b=int(seq[t-1]["y"])
            if a==b:continue
            P.append(seq[t]["p"]);Y.append(int(seq[t]["y"]));A.append(a)
    P=np.asarray(P,float);Y=np.asarray(Y,int);A=np.asarray(A,int)
    b=0.0
    for _ in range(30):
        pa=P[np.arange(len(P)),A];eb=math.exp(max(-30.,min(30.,b)))
        den=1.0+pa*(eb-1.0);prob=pa*eb/den;obs=(Y==A).astype(float)
        g=float(np.sum(prob-obs)+L2*b);h=float(np.sum(prob*(1-prob))+L2)
        nb=b-g/max(h,1e-12)
        if abs(nb-b)<1e-10:b=nb;break
        b=nb
    return float(b)

def score_return_beta(lines,train_parity,beta):
    B=R=N=0.0
    te=1-train_parity
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=te:continue
        for t in range(2,len(seq)):
            a=int(seq[t-2]["y"]);b=int(seq[t-1]["y"])
            if a==b:continue
            p=np.asarray(seq[t]["p"],float);w=math.exp(beta);den=1.0+p[a]*(w-1.0)
            q=p/den;q=q.copy();q[a]*=w;y=int(seq[t]["y"])
            B+=-math.log2(max(float(p[y]),1e-300));R+=-math.log2(max(float(q[y]),1e-300));N+=1
    return B,R,N

def target_fit_transfer(lines):
    B=R=N=0.0;betas={}
    for tr in (0,1):
        b=fit_return_beta(lines,tr);bb,rr,nn=score_return_beta(lines,tr,b)
        B+=bb;R+=rr;N+=nn;betas[str(tr)]=b
    return {"gain":float((B-R)/N) if N else 0.0,"n":int(N),"betas":betas}

def fit_A1(lines,parity):
    groups=[[] for _ in range(K)]
    for seq in lines:
        if not seq or fnum(seq[0]["folio"])%2!=parity:continue
        for t in range(2,len(seq)):
            groups[int(seq[t-1]["y"])].append({"p":seq[t]["p"],"y":int(seq[t]["y"])})
    A=np.zeros((K,K),float)
    for a,es in enumerate(groups):
        if es:A[a]=fit_tilt(es)
    return A

def augment_T1(lines):
    A={0:fit_A1(lines,0),1:fit_A1(lines,1)};out=[]
    for seq in lines:
        if not seq:continue
        te=fnum(seq[0]["folio"])%2;tr=1-te;zz=[]
        for t,e in enumerate(seq):
            x=dict(e);p=np.asarray(e["p"],float)
            if t>=2:p=tilt_p(p,A[tr][int(seq[t-1]["y"])])
            x["p"]=p;zz.append(x)
        out.append(zz)
    return out,A

def section_stats(base_lines):
    q0,mods=position_adjust_target(base_lines)
    s,n=aba_score(q0)
    return {"score":s,"n":n,"fixed_beta_gain":fixed_beta_gain(q0),
            "target_fit":target_fit_transfer(q0),"q0":q0,"posmods":mods}

REAL_BASE=generate_all(True,None)
REAL={}
for s in TARGETS:
    st=section_stats(REAL_BASE[s]);REAL[s]={k:v for k,v in st.items() if k not in ("q0","posmods")}
    REAL[s]["_q0"]=st["q0"];REAL[s]["_posmods"]=st["posmods"]
print("REAL",json.dumps({s:{k:v for k,v in REAL[s].items() if not k.startswith("_")} for s in TARGETS},
                        separators=(",",":")),flush=True)

def stageA_task(seed):
    b=generate_all(False,seed);out=[]
    for s in ("Herbal-A","Q13"):
        q0,_=position_adjust_target(b[s]);sc,n=aba_score(q0)
        out.append([sc,fixed_beta_gain(q0),target_fit_transfer(q0)["gain"]])
    return seed,out

def stageB_task(args):
    seed,section,Agen,posmods=args
    b=generate_all(False,seed,section,Agen,posmods)[section]
    q0,_=position_adjust_target(b)
    q1,_=augment_T1(q0);sc,n=aba_score(q1)
    return seed,sc

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
        rr=list(ex.map(stageA_task,A_SEEDS,chunksize=2))
    V=np.asarray([v for seed,v in rr],float) # 300 x 2 x 3
    fit=V[:100,:,0];cal=V[100:200,:,0];blind=V[200:,:,0]
    mu=fit.mean(0);sd=np.maximum(fit.std(0,ddof=1),1e-12)
    cz=(cal-mu)/sd;bz=(blind-mu)/sd
    names=("Herbal-A","Q13")
    rz=np.array([REAL[s]["score"] for s in names]);rz=(rz-mu)/sd
    cm=cz.max(1);bm=bz.max(1);mq=float(np.quantile(cm,.99));ba=float(np.mean(bm<=mq))
    ref=np.r_[cm,bm];rp=float(rz.max());p=float((1+np.sum(ref>=rp))/(len(ref)+1))
    perq=np.quantile(cz,.99,axis=0)
    globalpass=bool(ba>=.90 and rp>mq and p<=.01)
    resolved={names[i]:bool(globalpass and rz[i]>perq[i]) for i in range(2)}
    nresolved=sum(resolved.values())
    if not globalpass:decision="NO_EXTERNAL_REPLICATION"
    elif nresolved==2:decision="TWO_SECTION_REPLICATION"
    elif nresolved==1:decision="ONE_SECTION_REPLICATION"
    else:decision="GLOBAL_MAX_PASS_NO_SECTION_OWN_Q99"
    stageA={"real_score":{names[i]:float(np.array([REAL[s]["score"] for s in names])[i]) for i in range(2)},
            "real_Z":{names[i]:float(rz[i]) for i in range(2)},
            "per_section_q99_Z":{names[i]:float(perq[i]) for i in range(2)},
            "maxZ_q99":mq,"blind_acceptance":ba,"add_one_p":p,
            "global_pass":globalpass,"resolved":resolved,"decision":decision,
            "secondary_fixed_beta_gain":{s:REAL[s]["fixed_beta_gain"] for s in names},
            "secondary_target_fit":{s:REAL[s]["target_fit"] for s in names}}
    print("STAGE_A",json.dumps(stageA,separators=(",",":")),flush=True)

    stageB={}
    for section in names:
        if not resolved[section]:continue
        q0=REAL[section]["_q0"];q1,Areal=augment_T1(q0)
        rscore,rn=aba_score(q1)
        seeds=B_SEEDS[section];posmods=REAL[section]["_posmods"]
        args=[(s,section,Areal,posmods) for s in seeds]
        with ProcessPoolExecutor(max_workers=min(24,os.cpu_count() or 8)) as ex:
            rr=list(ex.map(stageB_task,args,chunksize=2))
        vv=np.asarray([v for seed,v in rr],float);cal=vv[:100];blind=vv[100:]
        q=float(np.quantile(cal,.99));bacc=float(np.mean(blind<=q))
        pp=float((1+np.sum(vv>=rscore))/(len(vv)+1))
        dec=("ABA_EXCEEDS_FIRST_ORDER_NULL" if bacc>=.90 and rscore>q and pp<=.01
             else "ABA_COMPATIBLE_WITH_FIRST_ORDER_ORIGIN")
        stageB[section]={"real_score_given_T1":float(rscore),"n":int(rn),"cal_q99":q,
                         "blind_acceptance":bacc,"add_one_p":pp,"decision":dec}
        print("STAGE_B",section,json.dumps(stageB[section],separators=(",",":")),flush=True)

    out={"programme":"ABA-REP1","status":"complete",
         "real":{s:{k:v for k,v in REAL[s].items() if not k.startswith("_")} for s in names},
         "stageA":stageA,"stageB":stageB}
    print("ABA_REP1_JSON="+json.dumps(out,separators=(",",":")),flush=True)
