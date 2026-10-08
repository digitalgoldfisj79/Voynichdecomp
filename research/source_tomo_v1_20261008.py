#!/usr/bin/env python3
# SOURCE-TOMO v1 — preregistered 2026-10-08.
# Simulator-calibrated identified regions for source-process invariants.
import collections, hashlib, json, math, pickle, re, urllib.request
import numpy as np
from sklearn.covariance import LedoitWolf

SEED=20261008
K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
COREURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
CIURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/67a73f80da2caefd6788de43c003701225833825/Paper/Cipher_paper/ci_corpus_parsed.pkl"

k0={"__name__":"k0"}; exec(compile(urllib.request.urlopen(K0URL,timeout=120).read().decode(),K0URL,"exec"),k0)
core={"__name__":"core"}; exec(compile(urllib.request.urlopen(COREURL,timeout=120).read().decode(),COREURL,"exec"),core)
segment=k0["segment"]; ST=k0["ST"]; PIECES=k0["PIECES"]; PID=k0["PID"]; PCLASS=k0["PCLASS"]
controls=k0["controls"]; gen_token=k0["gen_token"]; source_graph=k0["source_graph"]
ci=pickle.loads(urllib.request.urlopen(CIURL,timeout=120).read())
CIWORDS=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1]

HA=set("""f10r f10v f11r f11v f13r f13v f14r f14v f15r f15v f16r f16v f17r f17v f18r f18v f19r f19v f1v f20r f20v f21r f21v f22r f22v f23r f23v f24r f24v f25r f25v f26r f26v f27r f27v f28r f28v f29r f29v f2r f2v f30r f30v f31r f31v f32r f32v f33r f33v f34r f34v f35r f35v f36r f36v f37r f37v f38r f38v f39r f39v f3r f3v f40r f40v f41r f41v f42r f42v f43r f43v f44r f44v f45r f45v f46r f46v f47r f47v f48r f48v f49r f49v f4r f4v f50r f50v f51r f51v f52r f52v f53r f53v f54r f54v f55r f55v f56r f56v f57r f5r f5v f65r f65v f66v f6r f6v f7r f7v f8r f8v f9r f9v""".split())
Q13=set("""f75r f75v f76v f77r f77v f78r f78v f79r f79v f80r f80v f81r f81v f82r f82v f83r f83v f84r f84v""".split())

def real_lines(folios):
    rows=[r for r in core["build_rows"]("ZLZI") if r["folio"] in folios]
    by=collections.defaultdict(list)
    for r in rows: by[(r["folio"],r["line_ord"])].append(r)
    out=[]
    for key in sorted(by,key=lambda x:(core["fnum"](x[0]),x[1])):
        rs=sorted(by[key],key=lambda z:z["pos"])
        line=[]
        for r in rs:
            try:z=segment(r["token"])
            except Exception:continue
            if z: line.append((r["token"],z))
        if line:out.append(line)
    return out

REALLINES={"Herbal-A":real_lines(HA),"Q13":real_lines(Q13)}

def H_counts(vals):
    c=np.array(list(collections.Counter(vals).values()),float)
    if c.sum()==0:return 0.
    p=c/c.sum();return float(-(p*np.log2(p)).sum())

def condH(lines,order,keyfn):
    cnt=collections.Counter();ctx=collections.Counter()
    for line in lines:
        a=[keyfn(x) for x in line]
        for i in range(order,len(a)):
            h=tuple(a[i-order:i]); cnt[(h,a[i])]+=1;ctx[h]+=1
    n=sum(cnt.values())
    if n==0:return 0.
    s=0.
    for (h,y),v in cnt.items():
        p=v/n;q=v/ctx[h];s-=p*math.log2(max(q,1e-300))
    return float(s)

def lag_repeat(lines,lag,keyfn):
    same=tot=0; allv=[]
    for line in lines:
        a=[keyfn(x) for x in line];allv+=a
        for i in range(lag,len(a)):
            tot+=1;same+=int(a[i]==a[i-lag])
    if not tot or not allv:return 0.
    c=collections.Counter(allv);n=len(allv);base=sum((v/n)**2 for v in c.values())
    return float(same/tot-base)

def aba(lines,keyfn):
    num=den=0
    for line in lines:
        a=[keyfn(x) for x in line]
        for i in range(2,len(a)):
            den+=1;num+=int(a[i]==a[i-2] and a[i]!=a[i-1])
    return float(num/den) if den else 0.

def token_repr(tok,route):
    cls=[int(ST[p]) for p in route]
    sc=cls[0];fam=(cls[0],cls[-1],min(len(route),5))
    return tok,route,sc,fam

def summaries(lines):
    rr=[[token_repr(tok,z) for tok,z in line] for line in lines]
    sc=lambda x:x[2];fm=lambda x:x[3];ex=lambda x:x[0]
    flat=[x for L in rr for x in L]
    famvals=[fm(x) for x in flat]
    v=[
      H_counts([sc(x) for x in flat]),
      condH(rr,1,sc),condH(rr,2,sc),
      lag_repeat(rr,1,sc),lag_repeat(rr,2,sc),lag_repeat(rr,5,sc),aba(rr,sc),
      H_counts(famvals),condH(rr,1,fm),
      lag_repeat(rr,1,fm),lag_repeat(rr,2,fm),lag_repeat(rr,5,fm),lag_repeat(rr,10,fm),
      len(set(famvals))/max(1,len(famvals))
    ]
    # hostile / renderer audit panel
    lens=[len(x[1]) for x in flat]; exact=[ex(x) for x in flat]
    h={"exact_unique_frac":len(set(exact))/max(1,len(exact)),
       "exact_H1":H_counts(exact),"piece_len_mean":float(np.mean(lens)),"piece_len_var":float(np.var(lens)),
       "start_marginal":[collections.Counter(sc(x) for x in flat).get(i,0)/max(1,len(flat)) for i in range(12)]}
    return np.array(v,float),h

REAL={s:{"summary":summaries(L)[0],"audit":summaries(L)[1],"line_lengths":[len(x) for x in L],"N":sum(map(len,L))} for s,L in REALLINES.items()}

def stationary(A):
    p=np.ones(len(A))/len(A)
    for _ in range(10000):
        q=p@A
        if np.max(np.abs(q-p))<1e-13:break
        p=q
    p=np.maximum(p,0);p/=p.sum();return p

def source_truth(A):
    p=stationary(A)
    H=-float(np.sum(p*np.log2(np.maximum(p,1e-300))))
    hr=float(sum(p[i]*(-np.sum(A[i,A[i]>0]*np.log2(A[i,A[i]>0]))) for i in range(len(p))))
    rep=float(sum(p[i]*A[i,i] for i in range(len(p))))
    return {"log2_neff":H,"entropy_rate":hr,"memory_gap":H-hr,"repeat1":rep}

CLOCKS={
"C1_1to1":(0.,0.),
"C2_expand":(0.,.25),
"C3_delete":(.25,0.),
"C4_mixed":(.15,.15)
}

# Precompute F1 FORM emissions. Max source-state index 64; source controls are nuisance.
KMAX=65; POOLN=500; STRENGTHS=(.75,1.5,3.0)
POOLS={}
for strength in STRENGTHS:
    rng=np.random.default_rng(SEED+int(strength*1000))
    U,Ve,Vr=controls(KMAX,2,strength,rng,"F1")
    pp=[]
    for x in range(KMAX):
        pool=[]
        while len(pool)<POOLN:
            z=gen_token(x,U,Ve,Vr,rng)
            if z is not None:
                route=[PIECES[int(i)] for i in z]
                pool.append(("".join(route),route))
        pp.append(pool)
    POOLS[strength]=pp
print("POOLS_READY",flush=True)

def render_lines(A,line_lengths,strength,clock,seed,forced_source=None):
    rng=np.random.default_rng(seed);pi=stationary(A) if A is not None else None
    p0,p2=CLOCKS[clock];out=[];src_index=0
    for L in line_lengths:
        line=[];x=None if forced_source is None else 0
        if forced_source is None:x=int(rng.choice(len(pi),p=pi))
        guard=0
        while len(line)<L:
            guard+=1
            if guard>100000:raise RuntimeError("clock")
            if forced_source is not None:
                if src_index>=len(forced_source):src_index=0
                x=int(forced_source[src_index]);src_index+=1
            elif guard>1:
                x=int(rng.choice(len(pi),p=A[x]))
            u=rng.random()
            nout=0 if u<p0 else (2 if u>1-p2 else 1)
            for _ in range(nout):
                if len(line)>=L:break
                pool=POOLS[strength][x]
                tok,route=pool[int(rng.integers(len(pool)))]
                line.append((tok,route))
        out.append(line)
    return out

def build_sims(section,rep):
    lens=REAL[section]["line_lengths"];rows=[]
    for K in (8,16,32,64):
      for d in (2,4,8,16):
        if d>K:continue
        for strength in STRENGTHS:
          for clock in CLOCKS:
            seed=SEED+rep*100000+K*1000+d*100+int(strength*10)+list(CLOCKS).index(clock)
            A,_=source_graph(K,d,np.random.default_rng(seed))
            tr=source_truth(A)
            lines=render_lines(A,lens,strength,clock,seed+17)
            sm,au=summaries(lines)
            rows.append({"section":section,"rep":rep,"K":K,"d":d,"strength":strength,"clock":clock,
                         "truth":tr,"summary":sm.tolist(),"audit":au})
    return rows

SIM={s:[] for s in REAL}
for rep in range(4):
    for s in REAL:
        q=build_sims(s,rep);SIM[s]+=q
        print("SIM_DONE",s,rep,len(q),flush=True)

PROPS=("log2_neff","entropy_rate","memory_gap","repeat1")
def analyze(section):
    rows=SIM[section];train=[r for r in rows if r["rep"]<3];test=[r for r in rows if r["rep"]==3]
    X=np.array([r["summary"] for r in train]);mu=X.mean(0);lw=LedoitWolf().fit(X);P=lw.precision_
    def dist(v):
        D=X-v;return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",D,P,D),0))
    ranges={p:(min(r["truth"][p] for r in rows),max(r["truth"][p] for r in rows)) for p in PROPS}
    cal={p:{"cover":0,"widths":[]} for p in PROPS}; clock_ok=0; recalls=collections.Counter();den=collections.Counter();nnd=[]
    for r in test:
        dd=dist(np.array(r["summary"]));ix=np.argsort(dd)[:40];nn=[train[i] for i in ix];nnd.append(float(dd[ix[0]]))
        for p in PROPS:
            vals=np.array([z["truth"][p] for z in nn]);lo,hi=np.quantile(vals,[.05,.95]);truth=r["truth"][p]
            cal[p]["cover"]+=int(lo<=truth<=hi);cal[p]["widths"].append((hi-lo)/max(1e-12,ranges[p][1]-ranges[p][0]))
        pred=collections.Counter(z["clock"] for z in nn).most_common(1)[0][0]
        den[r["clock"]]+=1;recalls[(r["clock"],pred)]+=1;clock_ok+=int(pred==r["clock"])
    nt=len(test);qual={}
    for p in PROPS:
        cov=cal[p]["cover"]/nt;wid=float(np.median(cal[p]["widths"]))
        qual[p]={"coverage90":cov,"median_width_fraction":wid,"qualified":bool(.85<=cov<=.95 and wid<.60),"range":ranges[p]}
    crec={c:recalls[(c,c)]/den[c] for c in CLOCKS}
    cacc=clock_ok/nt;cqual=bool(cacc>=.70 and min(crec.values())>=.60)
    q99=float(np.quantile(nnd,.99))
    rv=REAL[section]["summary"];dd=dist(rv);ix=np.argsort(dd)[:40];nn=[train[i] for i in ix];real_nnd=float(dd[ix[0]])
    real={"nearest_distance":real_nnd,"q99_blind":q99,"ood":bool(real_nnd>q99),
          "nearest_configs":[{"K":z["K"],"d":z["d"],"strength":z["strength"],"clock":z["clock"],"distance":float(dd[i])} for i,z in zip(ix[:10],[train[j] for j in ix[:10]])],
          "clock_vote":dict(collections.Counter(z["clock"] for z in nn))}
    for p in PROPS:
        vals=np.array([z["truth"][p] for z in nn]);real[p]={"q05":float(np.quantile(vals,.05)),"median":float(np.median(vals)),"q95":float(np.quantile(vals,.95)),"qualified":qual[p]["qualified"]}
    return {"qualification":qual,"clock":{"accuracy":cacc,"recall":crec,"qualified":cqual},
            "blind_nearest_q99":q99,"real":real,"train_n":len(train),"blind_n":len(test)}

RESULT={s:analyze(s) for s in REAL}

# Historical Circa Instans control on Herbal-A line lengths.
def circa_control():
    vocab=[w for w,_ in collections.Counter(CIWORDS).most_common(64)];mp={w:i for i,w in enumerate(vocab)};OTHER=64
    z=np.array([mp.get(w,OTHER) for w in CIWORDS],int)
    C=np.ones((65,65),float)*.1
    for a,b in zip(z[:-1],z[1:]):C[a,b]+=1
    A=C/C.sum(1,keepdims=True)
    tr=source_truth(A)
    lens=REAL["Herbal-A"]["line_lengths"];need=sum(lens);zz=np.resize(z,need*2)
    lines=render_lines(None,lens,1.5,"C1_1to1",SEED+777,forced_source=zz)
    sm,au=summaries(lines)
    rows=[r for r in SIM["Herbal-A"] if r["rep"]<3];X=np.array([r["summary"] for r in rows]);lw=LedoitWolf().fit(X);P=lw.precision_
    D=X-sm;dd=np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",D,P,D),0));ix=np.argsort(dd)[:40];nn=[rows[i] for i in ix]
    out={"truth":tr,"nearest_distance":float(dd[ix[0]]),"intervals":{}}
    for p in PROPS:
        vals=np.array([q["truth"][p] for q in nn]);lo,hi=np.quantile(vals,[.05,.95]);out["intervals"][p]={"q05":float(lo),"q95":float(hi),"contains_truth":bool(lo<=tr[p]<=hi)}
    out["clock_vote"]=dict(collections.Counter(q["clock"] for q in nn));return out
RESULT["Circa_Control"]=circa_control()

OUT={"programme":"SOURCE-TOMO-v1","status":"complete","seed":SEED,
     "source_sensitive_metrics":["sc_H1","sc_Hcond1","sc_Hcond2","sc_rep1_ex","sc_rep2_ex","sc_rep5_ex","sc_ABA",
     "fam_H1","fam_Hcond1","fam_rep1_ex","fam_rep2_ex","fam_rep5_ex","fam_rep10_ex","fam_unique_frac"],
     "renderer":"K0 F1 ENTRY+ROUTE frozen socket","clock_regimes":CLOCKS,"results":RESULT,
     "real":{"Herbal-A":{"N":REAL["Herbal-A"]["N"],"summary":REAL["Herbal-A"]["summary"].tolist(),"audit":REAL["Herbal-A"]["audit"]},
             "Q13":{"N":REAL["Q13"]["N"],"summary":REAL["Q13"]["summary"].tolist(),"audit":REAL["Q13"]["audit"]}}}
print("SOURCE_TOMO_V1_JSON="+json.dumps(OUT,separators=(",",":")),flush=True)
