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
HA=set("""f10r f10v f11r f11v f13r f13v f14r f14v f15r f15v f16r f16v f17r f17v f18r f18v f19r f19v f1v f20r f20v f21r f21v f22r f22v f23r f23v f24r f24v f25r f25v f26r f26v f27r f27v f28r f28v f29r f29v f2r f2v f30r f30v f31r f31v f32r f32v f33r f33v f34r f34v f35r f35v f36r f36v f37r f37v f38r f38v f39r f39v f3r f3v f40r f40v f41r f41v f42r f42v f43r f43v f44r f44v f45r f45v f46r f46v f47r f47v f48r f48v f49r f49v f4r f4v f50r f50v f51r f51v f52r f52v f53r f53v f54r f54v f55r f55v f56r f56v f57r f5r f5v f65r f65v f66v f6r f6v f7r f7v f8r f8v f9r f9v""".split())
Q13=set("""f75r f75v f76v f77r f77v f78r f78v f79r f79v f80r f80v f81r f81v f82r f82v f83r f83v f84r f84v""".split())
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
REALLINES={"Herbal-A":real_lines(HA),"Q13":real_lines(Q13)}
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
for rep in range(4):
  for K in (8,16,32,64):
   for fam in GENS:
    for level in (0,1):
     struct_seed=SEED+rep*1000000+K*10000+level*1000+{"M1":10,"VAR2":20,"RENEW":30,"MOTIF":40}[fam]
     truth=source_truth(generate_source(make_params(fam,K,level,struct_seed),50000,struct_seed+777))
     for strength in STRENGTHS:
      for clock in CLOCKS:
       cid=f"r{rep}|K{K}|{fam}|L{level}|S{strength}|{clock}"
       for section,scode in (("Herbal-A",1),("Q13",2)):
        lines=render_config(fam,K,level,strength,clock,REAL[section]["line_lengths"],rep,scode);sp,cp,au=surface_panels(lines)
        SIM[section].append({"config_id":cid,"rep":rep,"K":K,"gen":fam,"level":level,"strength":strength,"clock":clock,"truth":truth,"source":sp.tolist(),"clock_panel":cp.tolist(),"audit":au})
  print("SIM_DONE",rep,{s:len(SIM[s]) for s in SIM},flush=True)
def model(section,panel):
    train=[r for r in SIM[section] if r["rep"]<3];X=np.array([r[panel] for r in train]);lw=LedoitWolf().fit(X);return train,X,lw.precision_
def neighbor(train,X,P,v,k=40):
    D=X-v;dd=np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",D,P,D),0));ix=np.argsort(dd)[:k];return dd,ix,[train[i] for i in ix]
MODELS={s:{"source":model(s,"source"),"clock":model(s,"clock_panel")} for s in REAL}
def infer_props(section,v):
    train,X,P=MODELS[section]["source"];dd,ix,nn=neighbor(train,X,P,v);return float(dd[ix[0]]),{p:np.array([r["truth"][p] for r in nn],float) for p in PROPS}
def qualify_source(section):
    train,X,P=MODELS[section]["source"];test=[r for r in SIM[section] if r["rep"]==3]
    ranges={p:(min(r["truth"][p] for r in SIM[section]),max(r["truth"][p] for r in SIM[section])) for p in PROPS}
    rec={p:{"all":[0,[]],"by":{g:[0,[]] for g in GENS}} for p in PROPS};nnd=[]
    for r in test:
        dd,ix,nn=neighbor(train,X,P,np.array(r["source"]));nnd.append(float(dd[ix[0]]))
        for p in PROPS:
            vals=np.array([z["truth"][p] for z in nn]);lo,hi=np.quantile(vals,[.05,.95]);ok=int(lo<=r["truth"][p]<=hi);wf=(hi-lo)/max(1e-12,ranges[p][1]-ranges[p][0])
            rec[p]["all"][0]+=ok;rec[p]["all"][1].append(wf);rec[p]["by"][r["gen"]][0]+=ok;rec[p]["by"][r["gen"]][1].append(wf)
    out={};nt=len(test)
    for p in PROPS:
        cov=rec[p]["all"][0]/nt;wid=float(np.median(rec[p]["all"][1]));bys={};robust=True
        for g in GENS:
            ng=sum(1 for r in test if r["gen"]==g);gc=rec[p]["by"][g][0]/ng;gw=float(np.median(rec[p]["by"][g][1]));q=bool(.80<=gc<=.98 and gw<.65)
            bys[g]={"coverage":gc,"width_fraction":gw,"pass":q};robust=robust and q
        agg=bool(.85<=cov<=.95 and wid<.60);out[p]={"coverage":cov,"width_fraction":wid,"aggregate_pass":agg,"by_generator":bys,"qualified":bool(agg and robust),"range":ranges[p]}
    return out,float(np.quantile(nnd,.99))
def qualify_clock(section):
    train,X,P=MODELS[section]["clock"];test=[r for r in SIM[section] if r["rep"]==3];ok=0;den=collections.Counter();hit=collections.Counter();by={g:{"ok":0,"n":0,"den":collections.Counter(),"hit":collections.Counter()} for g in GENS}
    for r in test:
        dd,ix,nn=neighbor(train,X,P,np.array(r["clock_panel"]));pred=collections.Counter(z["clock"] for z in nn).most_common(1)[0][0]
        ok+=int(pred==r["clock"]);den[r["clock"]]+=1;hit[r["clock"]]+=int(pred==r["clock"]);b=by[r["gen"]];b["ok"]+=int(pred==r["clock"]);b["n"]+=1;b["den"][r["clock"]]+=1;b["hit"][r["clock"]]+=int(pred==r["clock"])
    acc=ok/len(test);rec={c:hit[c]/den[c] for c in CLOCKS};parts={};robust=True
    for g,b in by.items():
        a=b["ok"]/b["n"];rr={c:b["hit"][c]/b["den"][c] for c in CLOCKS};q=bool(a>=.70 and min(rr.values())>=.60);parts[g]={"accuracy":a,"recall":rr,"pass":q};robust=robust and q
    agg=bool(acc>=.70 and min(rec.values())>=.60);return {"accuracy":acc,"recall":rec,"aggregate_pass":agg,"by_generator":parts,"qualified":bool(agg and robust)}
QUAL={};REALRES={}
for s in REAL:
    qs,q99=qualify_source(s);qc=qualify_clock(s);QUAL[s]={"source":qs,"clock":qc,"q99":q99};nd,vals=infer_props(s,REAL[s]["source"])
    train,X,P=MODELS[s]["clock"];dd,ix,nn=neighbor(train,X,P,REAL[s]["clock"]);votes=dict(collections.Counter(z["clock"] for z in nn))
    REALRES[s]={"nearest_distance":nd,"q99":q99,"ood":bool(nd>q99),"clock_votes":votes,"props":{}}
    for p,a in vals.items():REALRES[s]["props"][p]={"q05":float(np.quantile(a,.05)),"median":float(np.median(a)),"q95":float(np.quantile(a,.95)),"qualified":qs[p]["qualified"]}
CONTRAST={};blind_ids=sorted(set(r["config_id"] for r in SIM["Herbal-A"] if r["rep"]==3));bh={r["config_id"]:r for r in SIM["Herbal-A"] if r["rep"]==3};bq={r["config_id"]:r for r in SIM["Q13"] if r["rep"]==3}
realmed={s:{p:REALRES[s]["props"][p]["median"] for p in PROPS} for s in REAL}
for p in PROPS:
    dif=[]
    for cid in blind_ids:
        _,vh=infer_props("Herbal-A",np.array(bh[cid]["source"]));_,vq=infer_props("Q13",np.array(bq[cid]["source"]));dif.append(float(np.median(vq[p])-np.median(vh[p])))
    mu=float(np.mean(dif));sd=float(np.std(dif,ddof=1));rd=realmed["Q13"][p]-realmed["Herbal-A"][p];CONTRAST[p]={"real_Q13_minus_HA":rd,"null_mean":mu,"null_sd":sd,"effect":rd-mu,"z":((rd-mu)/sd if sd>0 else None)}
vocab=[w for w,_ in collections.Counter(CIWORDS).most_common(64)];mp={w:i for i,w in enumerate(vocab)};OTHER=64;cz=np.array([mp.get(w,OTHER) for w in CIWORDS],np.int16);ctruth=source_truth(cz)
def render_exact_source(z,line_lengths,strength,seed):
    out=[];i=0;rng=np.random.default_rng(seed)
    for L in line_lengths:
        line=[]
        for _ in range(L):
            x=int(z[i%len(z)]);i+=1;pool=POOLS[strength][x];line.append(pool[int(rng.integers(len(pool)))])
        out.append(line)
    return out
csp,ccp,cau=surface_panels(render_exact_source(cz,REAL["Herbal-A"]["line_lengths"],1.5,SEED+9999));cnd,cvals=infer_props("Herbal-A",csp);train,X,P=MODELS["Herbal-A"]["clock"];dd,ix,cnn=neighbor(train,X,P,ccp)
CIRCA={"truth":ctruth,"nearest_distance":cnd,"clock_pred":collections.Counter(z["clock"] for z in cnn).most_common(1)[0][0],"clock_votes":dict(collections.Counter(z["clock"] for z in cnn)),"intervals":{}}
for p,a in cvals.items():
    lo,hi=np.quantile(a,[.05,.95]);CIRCA["intervals"][p]={"q05":float(lo),"q95":float(hi),"contains_truth":bool(lo<=ctruth[p]<=hi)}
OUT={"programme":"SOURCE-TOMO-v2","status":"complete","seed":SEED,"generators":GENS,"clocks":CLOCKS,"source_properties":PROPS,"qualification":QUAL,"real":REALRES,"contrast":CONTRAST,"circa_control":CIRCA,"real_panels":{s:{"N":REAL[s]["N"],"source":REAL[s]["source"].tolist(),"clock":REAL[s]["clock"].tolist()} for s in REAL}}
print("SOURCE_TOMO_V2_JSON="+json.dumps(OUT,separators=(",",":")),flush=True)
