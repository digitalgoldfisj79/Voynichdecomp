import os,json,math,collections,urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

BASE_COMMIT="9ea05f51aef9b2ab00e443bdd0a0e0a295a42ba8"
ROOT=f"https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/{BASE_COMMIT}/research"
os.environ["TARGET_URL"]=ROOT+"/data/selector_target_bundle_20261004.gz.b64"
os.environ["CAUSAL_URL"]=ROOT+"/data/selector_causal_baseline_20261004.xz.b64"
os.environ["SOURCE_URL"]=ROOT+"/data/selector_source_controls_20261004.json"
src=urllib.request.urlopen(ROOT+"/selector_mechanism_tournament_renderer_20261004.py",timeout=60).read().decode()
_ns={"__name__":"selector_base"}
exec(compile(src,"selector_mechanism_tournament_renderer_20261004.py","exec"),_ns)

K=_ns["K"]; TARGET=_ns["TARGET"]; CAUSAL=_ns["CAUSAL"]; HALVES=_ns["HALVES"]
sample_source=_ns["sample_source"]; table_core=_ns["table"]; source_table_core=_ns["hybrid"]

ALPHA=30.0
THETA_PAGE=np.array([
0.3437162263574351,0.2537411498430932,0.3207633023520617,0.31227310588302554,
0.4907827653485044,0.5232580702790207,0.34612681856793276,0.9031121641334551,
0.32199281125042034,0.6190083392438619,0.3399427373676645,1.1119277161237104
],float)
THETA_PARA=np.array([
0.1688071877608053,0.04523238539311162,0.028524860013724937,0.1322435921987762,
-0.18705809522210926,0.2606257361488568,0.25561668314686503,0.2897810358341678,
0.29020526000134056,0.31500410181665006,0.21889503280717712,0.08568000296553302
],float)
THETA_RECENT=0.0530951464493163
G=np.array(CAUSAL["global"],float); PG=(G+.5)/(G.sum()+.5*K)
CM={(int(s),int(p)):np.array(c,float) for s,p,c in CAUSAL["counts"]}
RECORDS=CAUSAL["records"]
NORD=sum(int(r[5])>0 for r in RECORDS)
LAGS=[1,2,3,5,10]

def entropy(p):
    p=np.asarray(p,float); q=p[p>0]
    return float(-(q*np.log2(q)).sum())

# Identify target physical-half membership of each paragraph by exact observed ordinary sequence.
def build_half_map():
    obs=collections.OrderedDict()
    for fol,ln,pa,sec,piece,pos,y in RECORDS:
        if int(pos)>0: obs.setdefault((int(fol),int(pa)),[]).append(int(y))
    ctr={h:collections.Counter(tuple(x) for x in HALVES[h]) for h in ("0","1")}
    mp={}; amb=0;miss=0
    for pk,seq in obs.items():
        t=tuple(seq)
        c0=ctr["0"][t]; c1=ctr["1"][t]
        if c0 and c1: amb+=1
        if c0>=c1 and c0:
            mp[pk]=0;ctr["0"][t]-=1
        elif c1:
            mp[pk]=1;ctr["1"][t]-=1
        elif c0:
            mp[pk]=0;ctr["0"][t]-=1
        else:
            miss+=1; mp[pk]=(hash(pk)&1)
    return mp,amb,miss
HALF_MAP,HALF_AMBIG,HALF_MISS=build_half_map()

def baseline_prob(sec,piece,pagev,parav,linehist):
    c=CM.get((int(sec),int(piece)))
    base=(c+ALPHA*PG)/(c.sum()+ALPHA) if c is not None else PG
    rc=np.zeros(K,float)
    for q in linehist[-6:]:rc[int(q)]+=1
    sc=np.log(np.maximum(base,1e-15))
    sc += THETA_PAGE*np.log1p(pagev)
    sc += THETA_PARA*np.log1p(parav)
    sc += THETA_RECENT*rc
    sc-=sc.max();p=np.exp(sc);p/=p.sum()
    return p

def run_records(seed=None,latent=None,beta=0.0,observed=False):
    rng=np.random.default_rng(seed if seed is not None else 12345)
    flat=[x for s in latent for x in s] if latent is not None else []
    if latent is not None and len(flat)!=NORD: raise RuntimeError(("latent",len(flat),NORD))
    li=0
    page=collections.defaultdict(lambda:np.zeros(K,float))
    para=collections.defaultdict(lambda:np.zeros(K,float))
    line=collections.defaultdict(list)
    events=[]; byline=collections.OrderedDict()
    for fol,ln,pa,sec,piece,pos,yobs in RECORDS:
        fk=int(fol);lk=(fk,int(ln));pk=(fk,int(pa));pos=int(pos)
        if pos==0:
            y=int(yobs)
        else:
            p=baseline_prob(sec,piece,page[fk],para[pk],line[lk])
            prev=int(line[lk][-1]) if line[lk] else -1
            if observed:
                y=int(yobs)
            else:
                sc=np.log(np.maximum(p,1e-15))
                if latent is not None:
                    z=int(flat[li]);sc[z]+=float(beta)
                sc-=sc.max();q=np.exp(sc);q/=q.sum()
                y=int(rng.choice(K,p=q))
            if latent is not None: li+=1
            e={"p":p,"y":y,"prev":prev,"half":HALF_MAP.get(pk,0),"line":lk}
            events.append(e);byline.setdefault(lk,[]).append(e)
        page[fk][y]+=1;para[pk][y]+=1;line[lk].append(y)
    return events,list(byline.values())

def row_residual_matrix(events):
    S=np.zeros((K,K),float);N=np.zeros(K,float)
    for e in events:
        a=e["prev"]
        if a<0:continue
        v=-e["p"].copy();v[e["y"]]+=1
        S[a]+=v;N[a]+=1
    R=np.zeros_like(S)
    for a in range(K):
        if N[a]>0:R[a]=S[a]/N[a]
    return R,N

def safe_corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3:return 0.0
    x=x-x.mean();y=y-y.mean()
    d=math.sqrt(float(np.dot(x,x)*np.dot(y,y)))
    return float(np.dot(x,y)/d) if d>1e-12 else 0.0

def metric_vector(events,lines,include_halfcos=True):
    vals=[];names=[]
    # residual repeat innovations
    for lag in LAGS:
        z=[]
        for seq in lines:
            for i in range(lag,len(seq)):
                e=seq[i];prior=seq[i-lag]["y"]
                z.append((1.0 if e["y"]==prior else 0.0)-float(e["p"][prior]))
        vals.append(float(np.mean(z)) if z else 0.0);names.append(f"repeat_resid_{lag}")
    R,N=row_residual_matrix(events)
    vals.append(float(np.sqrt(np.mean(R*R))));names.append("transition_resid_norm")
    # half concordance
    if include_halfcos:
        A=[e for e in events if e["half"]==0];B=[e for e in events if e["half"]==1]
        RA,_=row_residual_matrix(A);RB,_=row_residual_matrix(B)
        x=RA.ravel();y=RB.ravel()
        vals.append(safe_corr(x,y));names.append("half_transition_concordance")
    # surprise innovations by within-line lag
    s_by_line=[]
    all_s=[]
    for seq in lines:
        z=[]
        for e in seq:
            p=e["p"];y=e["y"]
            s=-math.log2(max(float(p[y]),1e-15))-entropy(p)
            z.append(s);all_s.append(s)
        s_by_line.append(z)
    for lag in LAGS:
        a=[];b=[]
        for z in s_by_line:
            if len(z)>lag:
                a.extend(z[:-lag]);b.extend(z[lag:])
        vals.append(safe_corr(a,b));names.append(f"surprise_ac_{lag}")
    # low rank transition structure
    sv=np.linalg.svd(R,compute_uv=False);en=float(np.sum(sv*sv))
    vals.append(float((sv[0]**2)/en) if en>1e-15 else 0.0);names.append("sv1_energy")
    vals.append(float(np.sum(sv[:2]**2)/en) if en>1e-15 else 0.0);names.append("sv12_energy")
    vals.append(float(np.mean(all_s)));names.append("mean_excess_surprise")
    # Pearson innovation energy
    pe=[]
    for e in events:
        p=e["p"];y=e["y"]
        r=-p.copy();r[y]+=1
        den=np.sqrt(np.maximum(p*(1-p),1e-9));r=r/den
        pe.append(float(np.mean(r*r)))
    vals.append(float(np.mean(pe)));names.append("pearson_energy")
    return np.array(vals,float),names

TARGET_EVENTS,TARGET_LINES=run_records(observed=True)
TARGET_VEC,NAMES=metric_vector(TARGET_EVENTS,TARGET_LINES,True)

GROUPS={
 "repeat":[i for i,n in enumerate(NAMES) if n.startswith("repeat_")],
 "transition":[i for i,n in enumerate(NAMES) if n in ("transition_resid_norm","half_transition_concordance","sv1_energy","sv12_energy")],
 "surprise":[i for i,n in enumerate(NAMES) if n.startswith("surprise_") or n in ("mean_excess_surprise","pearson_energy")]
}

def half_vector(h,events,lines):
    ev=[e for e in events if e["half"]==h]
    ls=[[e for e in seq if e["half"]==h] for seq in lines]
    ls=[z for z in ls if z]
    return metric_vector(ev,ls,False)[0]

def half_task(seed):
    ev,ln=run_records(seed,None,0.0,False)
    return [half_vector(h,ev,ln).tolist() for h in (0,1)]

def sim_task(arg):
    kind,seed,beta=arg
    latent=None
    if kind=="SOURCE":latent=sample_source(seed)
    elif kind=="TABLE":latent=table_core(seed)
    elif kind=="SOURCE_TABLE":latent=source_table_core(seed)
    ev,ln=run_records(seed,latent,beta,False)
    v,_=metric_vector(ev,ln,True)
    return kind,float(beta),v.tolist()

def mahal_setup(null):
    mu=null.mean(0);S=np.cov(null,rowvar=False)
    D=np.diag(np.diag(S));C=.75*S+.25*D
    C+=np.eye(C.shape[0])*1e-9
    inv=np.linalg.pinv(C)
    return mu,inv
def md(v,mu,inv,keep=None):
    if keep is None:
        d=v-mu;return float(d@inv@d)
    Skeep=np.asarray(keep,int)
    # caller supplies recomputed inverse for ablations
    d=v[Skeep]-mu[Skeep]
    return d
def distance(v,mu,Cinv):
    d=v-mu;return float(d@Cinv@d)


def lag_examples(lines,L,h):
    out=[]
    for seq in lines:
        q=[e for e in seq if e["half"]==h]
        for i in range(L,len(q)):
            out.append((q[i],int(q[i-L]["y"])))
    return out

def fit_delta(examples,ridge=1.0):
    d=0.0
    for _ in range(60):
        g=-ridge*d; h=ridge
        ed=math.exp(max(-20,min(20,d)))
        for e,c in examples:
            pc=float(e["p"][c])
            qc=(pc*ed)/(1.0+pc*(ed-1.0))
            g += (1.0 if int(e["y"])==c else 0.0)-qc
            h += qc*(1.0-qc)
        step=g/max(h,1e-12)
        d += max(-1.0,min(1.0,step))
        if abs(step)<1e-9: break
    return float(d)

def eval_delta(examples,d):
    if not examples:return {"n":0,"gain_bits":0.0}
    s0=s1=0.0
    ed=math.exp(max(-20,min(20,d)))
    for e,c in examples:
        p=e["p"];y=int(e["y"])
        Z=1.0+float(p[c])*(ed-1.0)
        qy=float(p[y])*(ed if y==c else 1.0)/Z
        s0-=math.log2(max(float(p[y]),1e-15))
        s1-=math.log2(max(qy,1e-15))
    return {"n":len(examples),"gain_bits":float((s0-s1)/len(examples))}

def lag_transfer(events,lines,L):
    x0=lag_examples(lines,L,0);x1=lag_examples(lines,L,1)
    d0=fit_delta(x0,1.0);d1=fit_delta(x1,1.0)
    e01=eval_delta(x1,d0);e10=eval_delta(x0,d1)
    n=e01["n"]+e10["n"]
    g=(e01["gain_bits"]*e01["n"]+e10["gain_bits"]*e10["n"])/max(n,1)
    return {"lag":L,"delta_half0":d0,"delta_half1":d1,
            "gain_0_to_1":e01["gain_bits"],"gain_1_to_0":e10["gain_bits"],
            "n_eval_0_to_1":e01["n"],"n_eval_1_to_0":e10["n"],
            "symmetric_gain_bits":float(g)}

def null_lag_task(seed):
    ev,ln=run_records(seed,None,0.0,False)
    return {str(L):lag_transfer(ev,ln,L) for L in (1,2,3)}

if __name__=="__main__":
    target={str(L):lag_transfer(TARGET_EVENTS,TARGET_LINES,L) for L in (1,2,3)}
    with ProcessPoolExecutor(max_workers=min(32,os.cpu_count() or 8)) as ex:
        null=list(ex.map(null_lag_task,[202610170000+i for i in range(500)],chunksize=2))
    out={"target":target,"null":{}}
    for L in (1,2,3):
        a=np.array([x[str(L)]["symmetric_gain_bits"] for x in null],float)
        t=target[str(L)]["symmetric_gain_bits"]
        out["null"][str(L)]={"mean_gain":float(a.mean()),"sd_gain":float(a.std(ddof=1)),
                              "q99":float(np.quantile(a,.99)),
                              "p_add":float((1+np.sum(a>=t))/(len(a)+1))}
    t2=target["2"]
    out["promotion"]={
      "positive_deltas":bool(t2["delta_half0"]>0 and t2["delta_half1"]>0),
      "positive_both_transfer_gains":bool(t2["gain_0_to_1"]>0 and t2["gain_1_to_0"]>0),
      "p_lt_01":bool(out["null"]["2"]["p_add"]<.01),
      "beats_lag1_lag3":bool(t2["symmetric_gain_bits"]>target["1"]["symmetric_gain_bits"] and t2["symmetric_gain_bits"]>target["3"]["symmetric_gain_bits"])
    }
    out["promotion"]["pass"]=all(out["promotion"].values())
    print("LAG2_ECHO_RESULT_JSON="+json.dumps(out,separators=(",",":")),flush=True)
