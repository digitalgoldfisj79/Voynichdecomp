#!/usr/bin/env python3
import collections,json,math,pickle,re,urllib.request
import numpy as np
from sklearn.covariance import LedoitWolf
SEED=20261008
K0URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f/research/select_form_socket_oracle_phaseK0_20261004.py"
COREURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/498a6bdade562bf3dad245677fe959771eacefcb/research/naibbe_kernel_NK3_rebuild_core_v2_20261005.py"
CIURL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/67a73f80da2caefd6788de43c003701225833825/Paper/Cipher_paper/ci_corpus_parsed.pkl"
k0={"__name__":"k0"};exec(compile(urllib.request.urlopen(K0URL,timeout=120).read().decode(),K0URL,"exec"),k0)
core={"__name__":"core"};exec(compile(urllib.request.urlopen(COREURL,timeout=120).read().decode(),COREURL,"exec"),core)
segment=k0["segment"];ST=k0["ST"];PIECES=k0["PIECES"];controls=k0["controls"];gen_token=k0["gen_token"];source_graph=k0["source_graph"]
ci=pickle.loads(urllib.request.urlopen(CIURL,timeout=120).read())
CIWORDS=[str(w).lower() for w in ci["all_words"] if re.fullmatch(r"[a-zA-Z]+",str(w)) and len(str(w))>1]
STARS=set("""f103r f103v f104r f104v f105r f105v f106r f106v f107r f107v f108r f108v f111r f111v f112r f112v f113r f113v f114r f114v""".split())
def real_lines(folios):
    rows=[r for r in core["build_rows"]("ZLZI") if r["folio"] in folios];by=collections.defaultdict(list)
    for r in rows:by[(r["folio"],r["line_ord"])].append(r)
    out=[]
    for key in sorted(by,key=lambda x:(core["fnum"](x[0]),x[1])):
        line=[]
        for r in sorted(by[key],key=lambda z:z["pos"]):
            try:z=segment(r["token"])
            except Exception:continue
            if z:line.append((r["token"],z))
        if line:out.append(line)
    return out
REALLINES={"Stars":real_lines(STARS)}
def H_counts(vals):
    c=np.array(list(collections.Counter(vals).values()),float)
    if c.sum()==0:return 0.
    p=c/c.sum();return float(-(p*np.log2(p)).sum())
def condH_seq(a,order):
    if len(a)<=order:return 0.
    cnt=collections.Counter();ctx=collections.Counter()
    for i in range(order,len(a)):
        h=tuple(a[i-order:i]);cnt[(h,a[i])]+=1;ctx[h]+=1
    n=sum(cnt.values())
    return float(-sum((v/n)*math.log2(v/ctx[h]) for (h,y),v in cnt.items())) if n else 0.
def condH_lines(lines,order,keyfn):
    cnt=collections.Counter();ctx=collections.Counter()
    for line in lines:
        a=[keyfn(x) for x in line]
        for i in range(order,len(a)):
            h=tuple(a[i-order:i]);cnt[(h,a[i])]+=1;ctx[h]+=1
    n=sum(cnt.values())
    return float(-sum((v/n)*math.log2(v/ctx[h]) for (h,y),v in cnt.items())) if n else 0.
def lag_repeat(lines,lag,keyfn,excess=True):
    same=tot=0;allv=[]
    for line in lines:
        a=[keyfn(x) for x in line];allv+=a
        for i in range(lag,len(a)):tot+=1;same+=int(a[i]==a[i-lag])
    if not tot:return 0.
    raw=same/tot
    if not excess:return float(raw)
    c=collections.Counter(allv);n=len(allv);base=sum((v/n)**2 for v in c.values())
    return float(raw-base)
def aba(lines,keyfn):
    num=den=0
    for line in lines:
        a=[keyfn(x) for x in line]
        for i in range(2,len(a)):
            den+=1;num+=int(a[i]==a[i-2] and a[i]!=a[i-1])
    return float(num/den) if den else 0.
def run_frac(lines,keyfn,threshold):
    nr=good=0
    for line in lines:
        a=[keyfn(x) for x in line]
        if not a:continue
        r=1
        for i in range(1,len(a)):
            if a[i]==a[i-1]:r+=1
            else:nr+=1;good+=int(r>=threshold);r=1
        nr+=1;good+=int(r>=threshold)
    return float(good/nr) if nr else 0.
def token_repr(tok,route):
    cls=[int(ST[p]) for p in route];return tok,route,cls[0],(cls[0],cls[-1],min(len(route),5))
def surface_panels(lines):
    rr=[[token_repr(tok,z) for tok,z in line] for line in lines];flat=[x for L in rr for x in L]
    sc=lambda x:x[2];fm=lambda x:x[3];famvals=[fm(x) for x in flat]
    source=np.array([H_counts([sc(x) for x in flat]),condH_lines(rr,1,sc),condH_lines(rr,2,sc),
      lag_repeat(rr,1,sc),lag_repeat(rr,2,sc),lag_repeat(rr,5,sc),aba(rr,sc),
      H_counts(famvals),condH_lines(rr,1,fm),lag_repeat(rr,1,fm),lag_repeat(rr,2,fm),
      lag_repeat(rr,5,fm),lag_repeat(rr,10,fm),len(set(famvals))/max(1,len(famvals))],float)
    clock=[]
    for key in (sc,fm):clock += [lag_repeat(rr,l,key) for l in range(1,9)]
    hs=H_counts([sc(x) for x in flat]);h1s=condH_lines(rr,1,sc);h2s=condH_lines(rr,2,sc)
    hf=H_counts(famvals);h1f=condH_lines(rr,1,fm);h2f=condH_lines(rr,2,fm)
    clock += [hs-h1s,h1s-h2s,hf-h1f,h1f-h2f,aba(rr,sc),aba(rr,fm),
      lag_repeat(rr,1,sc,False),lag_repeat(rr,1,fm,False),run_frac(rr,sc,2),run_frac(rr,sc,3),run_frac(rr,fm,2),run_frac(rr,fm,3)]
    return source,np.array(clock,float),{"N":len(flat),"exact_unique_frac":len(set(x[0] for x in flat))/max(1,len(flat)),"piece_len_mean":float(np.mean([len(x[1]) for x in flat]))}
REAL={}
for s,L in REALLINES.items():
    sp,cp,au=surface_panels(L);REAL[s]={"source":sp,"clock":cp,"audit":au,"line_lengths":[len(x) for x in L],"N":sum(map(len,L))}
def make_params(fam,K,level,struct_seed):
    rs=np.random.default_rng(struct_seed)
    if fam=="M1":
        return {"fam":fam,"K":K,"A":source_graph(K,2 if level==0 else min(8,K),rs)[0]}
    if fam=="VAR2":
        return {"fam":fam,"K":K,"A":source_graph(K,min(4,K),rs)[0],"eta":(.25,.55)[level]}
    if fam=="RENEW":
        A=source_graph(K,min(4,K),rs)[0].copy();np.fill_diagonal(A,0.)
        for i in range(K):
            if A[i].sum()==0:A[i]=1.;A[i,i]=0.
            A[i]/=A[i].sum()
        return {"fam":fam,"K":K,"A":A,"mean":(2.,5.)[level]}
    if fam=="MOTIF":
        motifs=[]
        for _ in range(4):motifs.append(rs.integers(0,K,size=int(rs.integers(3,7)),dtype=np.int16))
        return {"fam":fam,"K":K,"noise":(.25,.10)[level],"persist":(8.,20.)[level],"motifs":motifs}
    raise ValueError(fam)
def generate_source(par,n,seq_seed):
    rq=np.random.default_rng(seq_seed);fam=par["fam"];K=par["K"]
    if fam=="M1":
        A=par["A"];z=np.empty(n,np.int16);z[0]=rq.integers(K)
        for t in range(1,n):z[t]=rq.choice(K,p=A[z[t-1]])
        return z
    if fam=="VAR2":
        A=par["A"];eta=par["eta"];z=np.empty(n,np.int16);z[0]=rq.integers(K);z[1]=rq.choice(K,p=A[z[0]])
        for t in range(2,n):z[t]=z[t-2] if rq.random()<eta else rq.choice(K,p=A[z[t-1]])
        return z
    if fam=="RENEW":
        A=par["A"];mean=par["mean"];out=[];x=int(rq.integers(K))
        while len(out)<n:
            dur=1+int(rq.poisson(max(.01,mean-1.)));out.extend([x]*dur);x=int(rq.choice(K,p=A[x]))
        return np.array(out[:n],np.int16)
    if fam=="MOTIF":
        noise=par["noise"];persist=par["persist"];motifs=par["motifs"];out=np.empty(n,np.int16);m=int(rq.integers(4));phase=0
        for t in range(n):
            if t>0 and rq.random()<1/persist:m=int(rq.integers(4));phase=0
            out[t]=int(rq.integers(K)) if rq.random()<noise else int(motifs[m][phase%len(motifs[m])]);phase+=1
        return out
    raise ValueError(fam)
def source_truth(z):
    a=list(map(int,z));return {"H1":H_counts(a),"Hcond1":condH_seq(a,1),"Hcond2":condH_seq(a,2),
      "repeat1":float(np.mean(z[1:]==z[:-1])),"repeat2":float(np.mean(z[2:]==z[:-2]))}
CLOCKS={"C1_1to1":(0.,0.),"C2_expand":(0.,.25),"C3_delete":(.25,0.),"C4_mixed":(.15,.15)}
STRENGTHS=(.75,1.5);KMAX=65;POOLN=350;POOLS={}
for strength in STRENGTHS:
    rng=np.random.default_rng(SEED+int(strength*1000));U,Ve,Vr=controls(KMAX,2,strength,rng,"F1");pp=[]
    for x in range(KMAX):
        pool=[]
        while len(pool)<POOLN:
            z=gen_token(x,U,Ve,Vr,rng)
            if z is not None:
                route=[PIECES[int(i)] for i in z];pool.append(("".join(route),route))
        pp.append(pool)
    POOLS[strength]=pp
print("POOLS_READY",flush=True)
def apply_clock(source_seq,L,strength,clock,seed):
    rng=np.random.default_rng(seed);p0,p2=CLOCKS[clock];line=[];i=0
    while len(line)<L:
        if i>=len(source_seq):raise RuntimeError("source exhausted")
        x=int(source_seq[i]);i+=1;u=rng.random();nout=0 if u<p0 else (2 if u>1-p2 else 1)
        for _ in range(nout):
            if len(line)>=L:break
            pool=POOLS[strength][x];line.append(pool[int(rng.integers(len(pool)))])
    return line
def render_config(fam,K,level,strength,clock,line_lengths,rep,section_code):
    struct_seed=SEED+rep*1000000+K*10000+level*1000+{"M1":10,"VAR2":20,"RENEW":30,"MOTIF":40}[fam];par=make_params(fam,K,level,struct_seed);lines=[]
    for j,L in enumerate(line_lengths):
        z=generate_source(par,max(24,int(L*2.5)+12),SEED+rep*2000000+section_code*500000+j*97+K*13+level)
        lines.append(apply_clock(z,L,strength,clock,SEED+rep*3000000+section_code*700000+j*131+int(strength*100)+list(CLOCKS).index(clock)))
    return lines
GENS=("M1","VAR2","RENEW","MOTIF");PROPS=("H1","Hcond1","Hcond2","repeat1","repeat2");SIM={s:[] for s in REAL}
def run_rep(rep):
    local={s:[] for s in REAL}
    for K in (8,16,32,64):
      for fam in GENS:
       for level in (0,1):
        struct_seed=SEED+rep*1000000+K*10000+level*1000+{"M1":10,"VAR2":20,"RENEW":30,"MOTIF":40}[fam]
        truth=source_truth(generate_source(make_params(fam,K,level,struct_seed),50000,struct_seed+777))
        for strength in STRENGTHS:
         for clock in CLOCKS:
          cid=f"r{rep}|K{K}|{fam}|L{level}|S{strength}|{clock}"
          for section,scode in (("Stars",3),):
           lines=render_config(fam,K,level,strength,clock,REAL[section]["line_lengths"],rep,scode);sp,cp,au=surface_panels(lines)
           local[section].append({"config_id":cid,"rep":rep,"K":K,"gen":fam,"level":level,"strength":strength,"clock":clock,"truth":truth,"source":sp.tolist(),"clock_panel":cp.tolist(),"audit":au})
    return rep,local
if __name__=="__main__":
    import multiprocessing as mp
    with mp.get_context("fork").Pool(4) as pool:
        for rep,local in pool.imap_unordered(run_rep,range(4)):
            for section in SIM:SIM[section].extend(local[section])
            print("SIM_DONE",rep,{s:len(local[s]) for s in local},flush=True)
def build_model(section):
    train=[r for r in SIM[section] if r["rep"] in (0,1)]
    X=np.array([r["source"] for r in train]);lw=LedoitWolf().fit(X)
    ranges={p:(min(r["truth"][p] for r in SIM[section]),max(r["truth"][p] for r in SIM[section])) for p in PROPS}
    return train,X,lw.precision_,ranges
MODELS={s:build_model(s) for s in REAL}
def predict(section,v,k=30):
    train,X,P,ranges=MODELS[section];D=X-v;dd=np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",D,P,D),0));ix=np.argsort(dd)[:k];nn=[train[i] for i in ix]
    out={}
    for p in PROPS:
        a=np.array([r["truth"][p] for r in nn],float);q25,q75=np.quantile(a,[.25,.75]);rng=ranges[p][1]-ranges[p][0]
        out[p]={"m":float(np.median(a)),"s":float(max(q75-q25,.05*rng))}
    return float(dd[ix[0]]),out
def qhigher(a,alpha=.90):
    a=np.array(a,float);n=len(a);q=min(1.0,math.ceil(alpha*(n+1))/n)
    return float(np.quantile(a,q,method="higher"))
def calibrate(section):
    train,X,P,ranges=MODELS[section];cal=[r for r in SIM[section] if r["rep"]==2];test=[r for r in SIM[section] if r["rep"]==3]
    scores={p:{"two":[],"lower":[],"upper":[]} for p in PROPS};caldist=[]
    for r in cal:
        d,pr=predict(section,np.array(r["source"]));caldist.append(d)
        for p in PROPS:
            t=r["truth"][p];m=pr[p]["m"];s=pr[p]["s"];scores[p]["two"].append(abs(t-m)/s);scores[p]["lower"].append((m-t)/s);scores[p]["upper"].append((t-m)/s)
    qs={p:{k:qhigher(v) for k,v in scores[p].items()} for p in PROPS};q99=float(np.quantile(caldist,.99))
    stat={p:{"two_hit":0,"two_width":[],"low_hit":0,"low_exc":[],"up_hit":0,"up_exc":[],
             "by":{g:{"n":0,"two_hit":0,"two_width":[],"low_hit":0,"up_hit":0} for g in GENS}} for p in PROPS}
    for r in test:
        d,pr=predict(section,np.array(r["source"]))
        for p in PROPS:
            loR,hiR=ranges[p];R=hiR-loR;t=r["truth"][p];m=pr[p]["m"];s=pr[p]["s"]
            lo=max(loR,m-qs[p]["two"]*s);hi=min(hiR,m+qs[p]["two"]*s)
            lb=max(loR,m-qs[p]["lower"]*s);ub=min(hiR,m+qs[p]["upper"]*s)
            S=stat[p];S["two_hit"]+=int(lo<=t<=hi);S["two_width"].append((hi-lo)/R);S["low_hit"]+=int(t>=lb);S["low_exc"].append((lb-loR)/R);S["up_hit"]+=int(t<=ub);S["up_exc"].append((hiR-ub)/R)
            B=S["by"][r["gen"]];B["n"]+=1;B["two_hit"]+=int(lo<=t<=hi);B["two_width"].append((hi-lo)/R);B["low_hit"]+=int(t>=lb);B["up_hit"]+=int(t<=ub)
    qual={}
    nt=len(test)
    for p in PROPS:
        S=stat[p];twoc=S["two_hit"]/nt;twow=float(np.median(S["two_width"]));lc=S["low_hit"]/nt;le=float(np.median(S["low_exc"]));uc=S["up_hit"]/nt;ue=float(np.median(S["up_exc"]))
        by={}
        twofam=lowfam=upfam=True
        for g,B in S["by"].items():
            tc=B["two_hit"]/B["n"];tw=float(np.median(B["two_width"]));ll=B["low_hit"]/B["n"];uu=B["up_hit"]/B["n"]
            tq=bool(.80<=tc<=.98 and tw<.70);lq=bool(.80<=ll<=.99);uq=bool(.80<=uu<=.99);twofam&=tq;lowfam&=lq;upfam&=uq
            by[g]={"two_coverage":tc,"two_width":tw,"lower_coverage":ll,"upper_coverage":uu,"two_pass":tq,"lower_pass":lq,"upper_pass":uq}
        qual[p]={"q":qs[p],"two":{"coverage":twoc,"median_width_fraction":twow,"qualified":bool(.85<=twoc<=.95 and twow<.60 and twofam)},
                 "lower":{"coverage":lc,"median_excluded_fraction":le,"qualified":bool(.88<=lc<=.98 and le>=.20 and lowfam)},
                 "upper":{"coverage":uc,"median_excluded_fraction":ue,"qualified":bool(.88<=uc<=.98 and ue>=.20 and upfam)},"by_generator":by}
    return qual,q99,ranges
CAL={};REALRES={}
for s in REAL:
    qual,q99,ranges=calibrate(s);CAL[s]={"properties":qual,"q99":q99};d,pr=predict(s,REAL[s]["source"]);REALRES[s]={"distance":d,"q99":q99,"ood":bool(d>q99),"properties":{}}
    for p in PROPS:
        loR,hiR=ranges[p];m=pr[p]["m"];sc=pr[p]["s"];q=qual[p]["q"]
        REALRES[s]["properties"][p]={"prediction":m,"two":[float(max(loR,m-q["two"]*sc)),float(min(hiR,m+q["two"]*sc))],
          "lower":float(max(loR,m-q["lower"]*sc)),"upper":float(min(hiR,m+q["upper"]*sc)),
          "two_qualified":qual[p]["two"]["qualified"],"lower_qualified":qual[p]["lower"]["qualified"],"upper_qualified":qual[p]["upper"]["qualified"]}

vocab=[w for w,_ in collections.Counter(CIWORDS).most_common(64)];mp={w:i for i,w in enumerate(vocab)};OTHER=64
cz=np.array([mp.get(w,OTHER) for w in CIWORDS],np.int16);ctruth=source_truth(cz)
def render_exact_source(z,line_lengths,strength,seed):
    out=[];i=0;rng=np.random.default_rng(seed)
    for L in line_lengths:
        line=[]
        for _ in range(L):
            x=int(z[i%len(z)]);i+=1;pool=POOLS[strength][x];line.append(pool[int(rng.integers(len(pool)))])
        out.append(line)
    return out
csp,ccp,cau=surface_panels(render_exact_source(cz,REAL["Stars"]["line_lengths"],1.5,SEED+9999));cd,cp=predict("Stars",csp)
CIRCA={"truth":ctruth,"distance":cd,"q99":CAL["Stars"]["q99"],"ood":bool(cd>CAL["Herbal-A"]["q99"]),"properties":{}}
ranges=MODELS["Stars"][3]
for p in PROPS:
    loR,hiR=ranges[p];m=cp[p]["m"];sc=cp[p]["s"];q=CAL["Stars"]["properties"][p]["q"];two=[max(loR,m-q["two"]*sc),min(hiR,m+q["two"]*sc)];lb=max(loR,m-q["lower"]*sc);ub=min(hiR,m+q["upper"]*sc);t=ctruth[p]
    CIRCA["properties"][p]={"two":[float(two[0]),float(two[1])],"two_covers":bool(two[0]<=t<=two[1]),"lower":float(lb),"lower_covers":bool(t>=lb),"upper":float(ub),"upper_covers":bool(t<=ub)}

PROMOTED={}
for s in REAL:
    PROMOTED[s]={}
    for p in PROPS:
        if REALRES[s]["ood"]:
            PROMOTED[s][p]={"status":"blocked_ood"};continue
        q=CAL[s]["properties"][p];cr=CIRCA["properties"][p];r=REALRES[s]["properties"][p]
        two=bool(q["two"]["qualified"] and cr["two_covers"]);low=bool(q["lower"]["qualified"] and cr["lower_covers"]);up=bool(q["upper"]["qualified"] and cr["upper_covers"])
        PROMOTED[s][p]={"two":two,"lower":low,"upper":up,"interval":r["two"] if two else None,"lower_bound":r["lower"] if low else None,"upper_bound":r["upper"] if up else None}
OUT={"programme":"SOURCE-TOMO-v3-Stars","status":"complete","seed":SEED,"calibration":CAL,"real":REALRES,"circa_control":CIRCA,"promoted":PROMOTED}
print("SOURCE_TOMO_V3_JSON="+json.dumps(OUT,separators=(",",":")),flush=True)
