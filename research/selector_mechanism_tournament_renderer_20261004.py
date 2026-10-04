import os,base64,gzip,lzma,json,math,collections
import urllib.parse, urllib.request
import numpy as np
from concurrent.futures import ProcessPoolExecutor

K=12
LAGS=[1,2,3,5,10,20]
TARGET=json.loads(gzip.decompress(base64.b64decode(os.environ["TARGET_B64"])))\nCAUSAL=json.loads(lzma.decompress(base64.b64decode(os.environ["CAUSAL_B64"])))
HALVES={k:[[int(x) for x in s] for s in v] for k,v in TARGET["halves"].items()}
SEGS=HALVES["0"]+HALVES["1"]
P=np.array(TARGET["marginal"],float)
METS=TARGET["metrics"]
TARGET_VEC=np.array([TARGET["target"][k] for k in METS],float)
SEG_LENS=[len(s) for s in SEGS]

def fetch_source():
    url=os.environ["SOURCE_URL"]
    with urllib.request.urlopen(url,timeout=60) as r:
        rows=json.loads(r.read().decode())
    out={}
    for row in rows:
        w=row["witness_id"];rec=collections.defaultdict(list)
        for x,rid in zip(row["ids"],row["recipes"]):rec[int(rid)].append(int(x))
        out[w]=list(rec.values())
    return out

SOURCE_RECIPES=fetch_source()
print("SOURCE_COUNTS",json.dumps({k:sum(map(len,v)) for k,v in SOURCE_RECIPES.items()}),flush=True)

def H(q):
    q=np.asarray(q,float);q=q[q>0]
    return float(-(q*np.log2(q)).sum()) if len(q) else 0.0
def variance(a):
    if len(a)<2:return 0.0
    m=sum(a)/len(a);return sum((x-m)**2 for x in a)/(len(a)-1)
def pairs(S,lag):
    a=[];b=[]
    for s in S:
        if len(s)>lag:a.extend(s[:-lag]);b.extend(s[lag:])
    return a,b
def MI(a,b):
    if not a:return 0.0
    C=np.zeros((K,K),float)
    for x,y in zip(a,b):C[x,y]+=1
    n=C.sum();pa=C.sum(1);pb=C.sum(0);z=0.0
    for i in range(K):
        for j in range(K):
            if C[i,j]>0:
                q=C[i,j]/n;d=(pa[i]/n)*(pb[j]/n);z+=q*math.log2(q/d)
    return z
def cond_entropy(S,o):
    M={};N=0
    for seq in S:
        for i in range(o,len(seq)):
            ctx=tuple(seq[i-o:i]) if o else ()
            if ctx not in M:M[ctx]=[0]*K
            M[ctx][seq[i]]+=1;N+=1
    h=0.0
    for c in M.values():
        n=sum(c);h+=(n/max(N,1))*H(np.array(c)/n)
    return h
def metrics(S):
    flat=[x for s in S for x in s]
    c=np.bincount(flat,minlength=K).astype(float);pp=c/c.sum()
    hp=H(pp);coll=float(np.dot(pp,pp));d={}
    for lag in LAGS:
        a,b=pairs(S,lag);same=sum(x==y for x,y in zip(a,b))
        d[f"mi{lag}"]=MI(a,b)/max(hp,1e-9)
        d[f"rep{lag}"]=(same/max(len(a),1))/max(coll,1e-12)-1
    nt=AAA=ABA=AAB=ABB=0
    for s in S:
        for i in range(len(s)-2):
            a,b,c=s[i:i+3];nt+=1
            AAA+=a==b==c;ABA+=a==c and a!=b;AAB+=a==b and b!=c;ABB+=b==c and a!=b
    e3=float(np.sum(pp**3));eaba=float(np.sum(pp**2*(1-pp)))
    d["aaa_excess"]=(AAA/max(nt,1))/max(e3,1e-12)-1
    d["aba_excess"]=(ABA/max(nt,1))/max(eaba,1e-12)-1
    d["aababb_excess"]=((AAB+ABB)/(2*max(nt,1)))/max(eaba,1e-12)-1
    for W in (12,24,48):
        z=[]
        for k in range(K):
            xs=[]
            for s in S:
                for i in range(0,len(s)-W+1,W):xs.append(sum(x==k for x in s[i:i+W]))
            if len(xs)>=5 and .005<pp[k]<.995:z.append(variance(xs)/max(W*pp[k]*(1-pp[k]),1e-9))
        d[f"fano{W}"]=sum(z)/len(z) if z else 1.0
    cvs=[];bur=[]
    for k in range(K):
        ints=[]
        for s in S:
            ix=[i for i,x in enumerate(s) if x==k];ints.extend(ix[i]-ix[i-1] for i in range(1,len(ix)))
        if len(ints)>=10:
            mu=sum(ints)/len(ints);sd=math.sqrt(variance(ints))
            cvs.append(sd/max(mu,1e-9));bur.append((sd-mu)/max(sd+mu,1e-9))
    d["return_cv"]=sum(cvs)/len(cvs);d["return_burst"]=sum(bur)/len(bur)
    ce=[cond_entropy(S,o) for o in range(4)]
    for o in range(1,4):d[f"ng_g{o}"]=ce[o-1]-ce[o]
    C=np.zeros((K,K),float)
    for s in S:
        for a,b in zip(s,s[1:]):C[a,b]+=1
    tot=C.sum();rec=0.0
    for i in range(K):
        for j in range(i+1,K):rec+=2*min(C[i,j],C[j,i])
    d["reciprocity"]=rec/max(tot,1)
    edges=np.sort(C.ravel())[::-1];d["top10_edge"]=float(edges[:10].sum()/max(tot,1))
    ev=[];wt=[]
    for i in range(K):
        n=C[i].sum()
        if n:ev.append(2**H(C[i]/n));wt.append(n)
    d["eff_outdegree"]=float(np.average(ev,weights=wt))
    bg=collections.Counter();tg=collections.Counter();nb=ng=0
    for s in S:
        for x in zip(s,s[1:]):bg[x]+=1;nb+=1
        for x in zip(s,s[1:],s[2:]):tg[x]+=1;ng+=1
    d["top20_bigram"]=sum(v for _,v in bg.most_common(20))/max(nb,1)
    d["top50_trigram"]=sum(v for _,v in tg.most_common(50))/max(ng,1)
    first=np.bincount([s[0] for s in S if s],minlength=K).astype(float)+.5
    first/=first.sum();mid=(first+pp)/2
    d["start_js"]=.5*sum(first[i]*math.log2(first[i]/mid[i]) for i in range(K))+.5*sum(pp[i]*math.log2(pp[i]/mid[i]) for i in range(K) if pp[i]>0)
    return np.array([d[k] for k in METS],float)

def bagcounts(B):
    r=B*P;c=np.floor(r).astype(int);n=B-int(c.sum())
    c[np.argsort(r-c)[::-1][:n]]+=1
    return c
def balanced_map(recipes,rng):
    ids=[x for q in recipes for x in q];cnt=collections.Counter(ids)
    types=list(cnt.items());rng.shuffle(types);types.sort(key=lambda z:-z[1])
    target=P*len(ids);assigned=np.zeros(K);mp={}
    for t,n in types:
        cost=((assigned+n-target)/np.maximum(target,1))**2-((assigned-target)/np.maximum(target,1))**2
        s=int(np.argmin(cost));mp[t]=s;assigned[s]+=n
    return [[mp[x] for x in q] for q in recipes]
def sample_source(seed):
    rng=np.random.default_rng(seed);mapped=[]
    for wi,w in enumerate(sorted(SOURCE_RECIPES)):
        mapped.extend(balanced_map(SOURCE_RECIPES[w],np.random.default_rng(seed+97*(wi+1))))
    out=[]
    for L in SEG_LENS:
        viable=[q for q in mapped if len(q)>=L]
        if viable:
            q=viable[int(rng.integers(len(viable)))];st=int(rng.integers(len(q)-L+1));out.append(q[st:st+L])
        else:
            q=max(mapped,key=len);out.append((q*((L//len(q))+1))[:L])
    return out
def dice(seed):
    rng=np.random.default_rng(seed);out=[];slow=rng.random()<.5
    conc=float(rng.choice([100,300,1000]));block=int(rng.choice([8,16,32]))
    for L in SEG_LENS:
        s=[];q=P.copy()
        for i in range(L):
            if slow and i%block==0:q=rng.dirichlet(np.maximum(P*conc,.05))
            s.append(int(rng.choice(K,p=q)))
        out.append(s)
    return out
def deck(seed):
    rng=np.random.default_rng(seed);B=int(rng.choice([12,24,48,96]))
    rep=float(rng.choice([0,.05,.15,.25]));base=bagcounts(B);out=[]
    for L in SEG_LENS:
        s=[];bag=[]
        while len(s)<L:
            if not bag:
                bag=[k for k,c in enumerate(base) for _ in range(int(c))];rng.shuffle(bag)
            x=int(bag.pop());s.append(x)
            if rng.random()<rep:bag.insert(int(rng.integers(0,len(bag)+1)),x)
        out.append(s)
    return out
def table(seed):
    rng=np.random.default_rng(seed)
    if rng.random()<.55:
        Q=np.zeros((K,K));strength=float(rng.choice([3,6,12,24]));deg=int(rng.choice([2,3,4]))
        for i in range(K):
            pref=rng.choice(K,deg,replace=False);w=P.copy();w[pref]*=strength;Q[i]=w/w.sum()
        out=[]
        for L in SEG_LENS:
            x=int(rng.choice(K,p=P));s=[]
            for _ in range(L):s.append(x);x=int(rng.choice(K,p=Q[x]))
            out.append(s)
        return out
    Z=int(rng.choice([3,4,6,8,12]));T=np.zeros((Z,Z));E=np.zeros((Z,K))
    strength=float(rng.choice([4,8,16]))
    for z in range(Z):
        pref=rng.choice(Z,min(Z,int(rng.choice([1,2,3]))),replace=False)
        w=np.ones(Z)*.15;w[pref]+=1;T[z]=w/w.sum()
        pref2=rng.choice(K,int(rng.choice([2,3,4])),replace=False)
        v=P.copy();v[pref2]*=strength;E[z]=v/v.sum()
    out=[]
    for L in SEG_LENS:
        z=int(rng.integers(Z));s=[]
        for _ in range(L):
            s.append(int(rng.choice(K,p=E[z])));z=int(rng.choice(Z,p=T[z]))
        out.append(s)
    return out
def hybrid(seed):
    rng=np.random.default_rng(seed);src=sample_source(seed^0x9e3779b9)
    Z=int(rng.choice([3,4,6]));order=np.argsort(P);perms=[]
    for _ in range(Z):
        pm=np.arange(K)
        for __ in range(1+int(rng.integers(3))):
            i=int(rng.integers(K-1));a,b=int(order[i]),int(order[i+1]);pm[a],pm[b]=pm[b],pm[a]
        perms.append(pm)
    out=[]
    for seg in src:
        z=int(rng.integers(Z));s=[]
        for x in seg:
            s.append(int(perms[z][x]));u=rng.random()
            if u<.55:pass
            elif u<.88:z=(z+1)%Z
            else:z=int(rng.integers(Z))
        out.append(s)
    return out

def render(latent,seed):
    rng=np.random.default_rng(seed^0x5bd1e995)
    beta=float(rng.choice([0.75,1.5,3.0]))
    flat=[x for s in latent for x in s];li=0
    G=np.array(CAUSAL["global"],float);pg=(G+.5)/(G.sum()+.5*K)
    alpha=float(CAUSAL["alpha"]);theta=np.array(CAUSAL["theta"],float)
    CM={(int(s),int(p)):np.array(c,float) for s,p,c in CAUSAL["counts"]}
    page=collections.defaultdict(lambda:np.zeros(K,float))
    para=collections.defaultdict(lambda:np.zeros(K,float))
    line=collections.defaultdict(list)
    out=collections.OrderedDict()
    for fol,ln,pa,sec,piece,pos,yobs in CAUSAL["records"]:
        fk=int(fol);lk=(fk,int(ln));pk=(fk,int(pa))
        if int(pos)==0:
            y=int(yobs)
        else:
            z=int(flat[li]);li+=1
            c=CM.get((int(sec),int(piece)))
            base=(c+alpha*pg)/(c.sum()+alpha) if c is not None else pg
            rc=np.zeros(K,float)
            for q in line[lk][-6:]:rc[int(q)]+=1
            sc=np.log(np.maximum(base,1e-15))+theta[0]*np.log1p(page[fk])+theta[1]*np.log1p(para[pk])+theta[2]*rc
            sc[z]+=beta;sc-=sc.max();pr=np.exp(sc);pr/=pr.sum()
            y=int(rng.choice(K,p=pr))
            out.setdefault(pk,[]).append(y)
        page[fk][y]+=1;para[pk][y]+=1;line[lk].append(y)
    if li!=len(flat):raise RuntimeError(f"latent length mismatch {li} {len(flat)}")
    return list(out.values())

GEN={"DICE":dice,"DECK":deck,"TABLE":table,"SOURCE":sample_source,"HYBRID":hybrid}
def task(arg):
    fam,seed=arg
    return fam,metrics(render(GEN[fam](seed),seed+7919)).tolist()

if __name__=="__main__":
    NS=int(os.environ.get("NS","160"))
    jobs=[(fam,202610040000+fi*100000+i) for fi,fam in enumerate(GEN) for i in range(NS)]
    samples={f:[] for f in GEN}
    with ProcessPoolExecutor(max_workers=min(32,os.cpu_count() or 8)) as ex:
        for fam,v in ex.map(task,jobs,chunksize=2):samples[fam].append(v)
    samples={f:np.array(v,float) for f,v in samples.items()}
    ntr=min(100,NS*2//3);TR=np.arange(ntr);TE=np.arange(ntr,NS)
    pool=np.vstack([samples[f][TR] for f in GEN]);mu=pool.mean(0);sd=pool.std(0,ddof=1);sd=np.where(sd<1e-8,1,sd)
    cent={f:((samples[f][TR]-mu)/sd).mean(0) for f in GEN}
    def classify(x,allow=None):
        z=(x-mu)/sd;fs=list(allow) if allow else list(GEN)
        ds={f:float(np.mean((z-cent[f])**2)) for f in fs}
        return min(ds,key=ds.get),ds
    conf={f:collections.Counter() for f in GEN}
    for f in GEN:
        for x in samples[f][TE]:conf[f][classify(x)[0]]+=1
    recall={f:conf[f][f]/len(TE) for f in GEN}
    overall=sum(conf[f][f] for f in GEN)/(len(GEN)*len(TE))
    def pairacc(a,b):
        good=tot=0
        for f in (a,b):
            for x in samples[f][TE]:
                good+=classify(x,(a,b))[0]==f;tot+=1
        return good/tot
    pairwise={
        "DICE_DECK":pairacc("DICE","DECK"),
        "SOURCE_TABLE":pairacc("SOURCE","TABLE"),
        "SOURCE_DICE":pairacc("SOURCE","DICE"),
        "SOURCE_DECK":pairacc("SOURCE","DECK"),
        "SOURCE_HYBRID":pairacc("SOURCE","HYBRID"),
        "TABLE_HYBRID":pairacc("TABLE","HYBRID")
    }
    gate=overall>=.80 and min(recall.values())>=.65 and pairwise["DICE_DECK"]>=.80 and pairwise["SOURCE_TABLE"]>=.80
    def score(x):
        pred,ds=classify(x);pct={}
        for f in GEN:
            vals=[classify(y,(f,))[1][f] for y in samples[f][TE]]
            pct[f]=sum(v<=ds[f] for v in vals)/len(vals)
        return {"nearest":pred,"distance":ds,"within_family_distance_percentile":pct}
    target=score(TARGET_VEC)
    halves={k:score(metrics(v)) for k,v in HALVES.items()}
    groups={
      "lag_memory":[i for i,k in enumerate(METS) if k.startswith("mi") or k.startswith("rep") or k.startswith("ng_") or k in ("aaa_excess","aba_excess","aababb_excess")],
      "dispersion":[i for i,k in enumerate(METS) if k.startswith("fano") or k.startswith("return")],
      "topology":[i for i,k in enumerate(METS) if k in ("reciprocity","top10_edge","eff_outdegree","top20_bigram","top50_trigram")],
      "reset":[i for i,k in enumerate(METS) if k=="start_js"]
    }
    abl={}
    for g,ix in groups.items():
        keep=[i for i in range(len(METS)) if i not in ix]
        pmu=pool[:,keep].mean(0);psd=pool[:,keep].std(0,ddof=1);psd=np.where(psd<1e-8,1,psd)
        pc={f:((samples[f][TR][:,keep]-pmu)/psd).mean(0) for f in GEN}
        z=(TARGET_VEC[keep]-pmu)/psd;ds={f:float(np.mean((z-pc[f])**2)) for f in GEN}
        abl["without_"+g]={"nearest":min(ds,key=ds.get),"distance":ds}
    out={
      "renderer_beta_grid":[0.75,1.5,3.0],"n_control_per_family":NS,"train_per_family":len(TR),"test_per_family":len(TE),
      "calibration":{"overall_accuracy":overall,"recall":recall,"pairwise_accuracy":pairwise,"gate_pass":gate,
                     "confusion":{f:dict(conf[f]) for f in GEN}},
      "target":target,"target_halves":halves,"metric_ablations":abl,
      "target_metrics":{k:float(v) for k,v in zip(METS,TARGET_VEC)},
      "control_centroids":{f:{k:float(v) for k,v in zip(METS,samples[f][TR].mean(0))} for f in GEN}
    }
    print("TOURNAMENT_RESULT_JSON="+json.dumps(out,separators=(",",":")),flush=True)
