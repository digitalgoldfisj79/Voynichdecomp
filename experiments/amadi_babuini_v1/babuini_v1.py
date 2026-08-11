# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, collections, concurrent.futures, hashlib, json, math, statistics
from dataclasses import dataclass
import numpy as np
from numba import njit
import amadi_residuals_v1 as m

NS="AMADIBABUINIV1"
SIG=61
BASE=19
NPAIR=42
NLETTER=14
PROPS=24000
RESTARTS=12
BATCH=4
SMOKE_PROPS=1400
SMOKE_RESTARTS=2
VOW=set("aeiou")
FORMS_CV={"CV","VC"}
FORMS_FULL={"CV","VC","VCC","VCCC","CCV","CCCV","CVC"}

def seed(*x):
    return int.from_bytes(hashlib.sha256("::".join(map(str,x)).encode()).digest()[:8],"big") & 0x7fffffff

def pattern(s):
    return "".join("V" if c in VOW else "C" for c in s)

def italian_words():
    sents=m.parse_ud(m.getb(m.URLS["italian"]))
    train=[]; ctrl=[]
    for i,s in enumerate(sents):
        for raw in s:
            w=m.norm_std(raw)
            if not w: continue
            ss="".join(m.PLAIN[x] for x in w)
            if i%10 in m.TRAIN_RES: train.append(ss)
            elif i%10 in m.CTRL_RES: ctrl.append(ss)
    return train,ctrl

def make_inventory(train,mode):
    lf=collections.Counter(c for w in train for c in w)
    letters=[c for c,_ in sorted(lf.items(),key=lambda q:(-q[1],q[0]))[:NLETTER]]
    forms=FORMS_CV if mode=="CV" else FORMS_FULL
    sc=collections.Counter()
    maxlen=2 if mode=="CV" else 4
    for w in train:
        for i in range(len(w)):
            for L in range(2,maxlen+1):
                if i+L<=len(w):
                    u=w[i:i+L]
                    if pattern(u) in forms: sc[u]+=1
    syll=[]
    for u,_ in sorted(sc.items(),key=lambda q:(-q[1],-len(q[0]),q[0])):
        if u not in syll: syll.append(u)
        if len(syll)>=SIG-NLETTER: break
    if len(syll)<SIG-NLETTER: raise RuntimeError("insufficient syllable inventory")
    units=letters+syll
    assert len(units)==SIG and len(set(units))==SIG
    return units,{"letters":letters,"syllables":syll,"mode":mode}

def segment_word(w,units):
    byfirst=collections.defaultdict(list)
    for j,u in enumerate(units): byfirst[u[0]].append((j,u))
    for c in byfirst: byfirst[c].sort(key=lambda q:(-len(q[1]),q[0]))
    n=len(w); best=[None]*(n+1); best[n]=[]
    for i in range(n-1,-1,-1):
        opts=[]
        for j,u in byfirst.get(w[i],[]):
            k=i+len(u)
            if k<=n and w.startswith(u,i) and best[k] is not None:
                cand=[j]+best[k]; opts.append((len(cand),-len(u),cand))
        if opts: best[i]=min(opts,key=lambda q:(q[0],q[1],q[2]))[2]
    return best[0]

def segment_corpus(words,units):
    out=[]; kept=total=0; kw=0
    for w in words:
        total+=len(w); z=segment_word(w,units)
        if z is None: continue
        out.append(z); kept+=len(w); kw+=1
    return out,{"char_coverage":kept/max(1,total),"kept_chars":kept,"total_chars":total,"kept_words":kw,"total_words":len(words)}

@dataclass
class ULM:
    units:list[str]
    logtri:np.ndarray
    freq:np.ndarray
    control:list[list[int]]
    meta:dict

def build_ulm(train,ctrl,units):
    tr,mt=segment_corpus(train,units); ct,mc=segment_corpus(ctrl,units)
    U=len(units); C=np.full((U+1,U+1,U+1),0.15,np.float64); F=np.full(U,0.15,np.float64)
    for w in tr:
        for x in w:F[x]+=1
        z=[U,U]+w+[U,U]
        for a,b,c in zip(z,z[1:],z[2:]): C[a,b,c]+=1
    C/=C.sum(axis=2,keepdims=True); F/=F.sum()
    return ULM(units,np.log(C),F,ct,{"train":mt,"control":mc})

def take_span(words,tag,fitn,holdn):
    st=seed(NS,"span",tag)%len(words); fit=[]; hold=[]; a=b=0; j=0
    while b<holdn:
        w=words[(st+j)%len(words)]; j+=1
        if a<fitn: fit.append(w); a+=len(w)
        else: hold.append(w); b+=len(w)
        if j>len(words)*10: raise RuntimeError("span exhausted")
    return fit,hold

def render_codebook(tag):
    rg=np.random.default_rng(seed(NS,"render",tag)); singles=list(rg.permutation(BASE))
    allpairs=[(a,b) for a in range(BASE) for b in range(BASE) if a!=b]; rg.shuffle(allpairs); pairs=allpairs[:NPAIR]
    return [[x] for x in singles]+[[a,b] for a,b in pairs]

def render(words,codebook): return [[y for x in w for y in codebook[x]] for w in words]

def pair_lexicon(words):
    left=np.zeros(BASE,np.int64); right=np.zeros(BASE,np.int64); pc=collections.Counter(); N=0
    for w in words:
        for a,b in zip(w,w[1:]): pc[(a,b)]+=1; left[a]+=1; right[b]+=1; N+=1
    rows=[]
    for (a,b),c in pc.items():
        if c<3: continue
        pmi=math.log((c*N+1.0)/(left[a]*right[b]+1.0)); rows.append((c*max(0.0,pmi),c,a,b))
    rows.sort(key=lambda q:(-q[0],-q[1],q[2],q[3])); pairs=[(a,b) for _,_,a,b in rows[:NPAIR]]
    if len(pairs)<NPAIR:
        for a in range(BASE):
            for b in range(BASE):
                if (a,b) not in pairs:
                    pairs.append((a,b))
                    if len(pairs)==NPAIR: break
            if len(pairs)==NPAIR: break
    return pairs

def tokenize_pairs(words,pairs):
    pid={p:BASE+i for i,p in enumerate(pairs)}; out=[]
    for w in words:
        z=[]; i=0
        while i<len(w):
            if i+1<len(w) and (w[i],w[i+1]) in pid: z.append(pid[(w[i],w[i+1])]); i+=2
            else: z.append(w[i]); i+=1
        out.append(z)
    return out

@dataclass
class Stats:
    a:np.ndarray;b:np.ndarray;c:np.ndarray;n:np.ndarray;off:np.ndarray;adj:np.ndarray;tf:np.ndarray;chars:int

def stats(words):
    bd=SIG; cnt=collections.Counter(); tf=np.zeros(SIG+1,np.int64); chars=0
    for w in words:
        z=[bd,bd]+w+[bd,bd]; chars+=len(w)
        for x in w: tf[x]+=1
        for a,b,c in zip(z,z[1:],z[2:]):cnt[(a,b,c)]+=1
    ks=list(cnt); aa=np.array([x[0] for x in ks],np.int32);bb=np.array([x[1] for x in ks],np.int32);cc=np.array([x[2] for x in ks],np.int32);nn=np.array([cnt[x] for x in ks],np.int64)
    ls=[[] for _ in range(SIG)]
    for i,(a,b,c) in enumerate(ks):
        for t in set((a,b,c)):
            if t<SIG:ls[t].append(i)
    off=[0];flat=[]
    for q in ls:flat.extend(q);off.append(len(flat))
    return Stats(aa,bb,cc,nn,np.array(off,np.int32),np.array(flat,np.int32),tf,chars)

@njit(nogil=True,cache=False)
def score_sparse(a,b,c,n,dec,logp):
    s=0.0
    for i in range(len(n)):s+=n[i]*logp[dec[a[i]],dec[b[i]],dec[c[i]]]
    return s

@njit(nogil=True,cache=False)
def dswap(a,b,c,n,dec,off,adj,ta,tb,logp):
    va=dec[ta];vb=dec[tb];d=0.0
    for jj in range(off[ta],off[ta+1]):
        i=adj[jj];x=a[i];y=b[i];z=c[i];ox=dec[x];oy=dec[y];oz=dec[z]
        nx=vb if x==ta else (va if x==tb else ox);ny=vb if y==ta else (va if y==tb else oy);nz=vb if z==ta else (va if z==tb else oz)
        d+=n[i]*(logp[nx,ny,nz]-logp[ox,oy,oz])
    for jj in range(off[tb],off[tb+1]):
        i=adj[jj];x=a[i];y=b[i];z=c[i]
        if x==ta or y==ta or z==ta:continue
        ox=dec[x];oy=dec[y];oz=dec[z];nx=va if x==tb else ox;ny=va if y==tb else oy;nz=va if z==tb else oz
        d+=n[i]*(logp[nx,ny,nz]-logp[ox,oy,oz])
    return d

@njit(nogil=True,cache=False)
def rngstep(s):
    s^=s>>np.uint64(12);s^=s<<np.uint64(25);s^=s>>np.uint64(27);return s*np.uint64(2685821657736338717)
@njit(nogil=True,cache=False)
def rint(s,u):s=rngstep(s);return s,int(s%np.uint64(u))
@njit(nogil=True,cache=False)
def rfloat(s):s=rngstep(s);return s,float(s>>np.uint64(11))*(1.0/9007199254740992.0)

@njit(nogil=True,cache=False)
def anneal(a,b,c,n,off,adj,tf,init,logp,props,sd):
    dec=init.copy();dec[SIG]=SIG;sc=score_sparse(a,b,c,n,dec,logp);best=sc;bestd=dec.copy();state=np.uint64(sd or 1)
    ma=0.;nn=0
    for _ in range(64):
        state,x=rint(state,SIG);state,y=rint(state,SIG)
        if x==y:continue
        dd=dswap(a,b,c,n,dec,off,adj,x,y,logp);ma+=abs(dd);nn+=1
    ma/=max(1,nn);t0=max(.03,2.5*ma);te=max(.0003,.01*ma);cool=math.exp(math.log(te/t0)/max(1,props));temp=t0
    for _ in range(props):
        state,x=rint(state,SIG);state,y=rint(state,SIG)
        if x==y:temp*=cool;continue
        dd=dswap(a,b,c,n,dec,off,adj,x,y,logp);ok=dd>=0
        if not ok:
            state,u=rfloat(state);ok=u<math.exp(dd/max(temp,1e-12))
        if ok:
            q=dec[x];dec[x]=dec[y];dec[y]=q;sc+=dd
            if sc>best:best=sc;bestd=dec.copy()
        temp*=cool
    return bestd,best

def init_map(st,lm,tag,random=False):
    dec=np.empty(SIG+1,np.int32);dec[SIG]=SIG;rg=np.random.default_rng(seed(NS,"init",tag))
    if random:
        p=rg.permutation(SIG)
        for x in range(SIG):dec[x]=p[x]
    else:
        obs=sorted(range(SIG),key=lambda x:(-int(st.tf[x]),x));pl=sorted(range(SIG),key=lambda x:(-float(lm.freq[x]),x))
        for x,p in zip(obs,pl):dec[x]=p
    return dec

def agree(d1,d2,tf):
    den=max(1,int(tf[:SIG].sum()));return float(tf[:SIG][d1[:SIG]==d2[:SIG]].sum()/den)

def solve(st,lm,tag,smoke=False):
    props=SMOKE_PROPS if smoke else PROPS;mx=SMOKE_RESTARTS if smoke else RESTARTS;batch=2 if smoke else BATCH
    bd=[None,None];bs=[-1e300,-1e300];used=[0,0];ag=0.;conv=False
    for end in range(batch,mx+1,batch):
        for e in (0,1):
            for r in range(used[e],end):
                init=init_map(st,lm,f"{tag}:{e}:{r}",r>0);d,sc=anneal(st.a,st.b,st.c,st.n,st.off,st.adj,st.tf,init,lm.logtri,props,seed(NS,"ann",tag,e,r))
                if sc>bs[e]:bs[e]=float(sc);bd[e]=d.copy()
            used[e]=end
        ag=agree(bd[0],bd[1],st.tf);diff=abs(bs[0]-bs[1])/max(1,st.chars)
        if ag>=.95 and diff<=1e-7:conv=True;break
    win=0 if bs[0]>=bs[1] else 1
    return {"dec":bd[win],"fit_score":bs[win]/max(1,st.chars),"agreement":ag,"converged":conv,"score_diff":abs(bs[0]-bs[1])/max(1,st.chars),"restarts_each":used[0]}

def fixed(st,lm,dec):return float(score_sparse(st.a,st.b,st.c,st.n,dec,lm.logtri)/max(1,st.chars))

def encrypt_units(words,tag):
    rg=np.random.default_rng(seed(NS,"key",tag));p2c=rg.permutation(SIG);inv=np.empty(SIG,np.int32)
    for p,c in enumerate(p2c):inv[c]=p
    return [[int(p2c[x]) for x in w] for w in words],inv

def recover_acc(truth,tok,dec):
    if len(truth)!=len(tok):return 0.0
    good=tot=0
    for a,b in zip(truth,tok):
        if len(a)!=len(b):tot+=max(len(a),len(b));continue
        tot+=len(a);good+=sum(x==int(dec[y]) for x,y in zip(a,b))
    return good/max(1,tot)

def control_run(lm,tag,smoke=False):
    n=900 if smoke else 3500;fw,hw=take_span(lm.control,tag,n,n);fc,truth=encrypt_units(fw,tag)
    p2c=np.empty(SIG,np.int32)
    for c,p in enumerate(truth):p2c[p]=c
    hc=[[int(p2c[x]) for x in w] for w in hw];cb=render_codebook(tag);fr=render(fc,cb);hr=render(hc,cb)
    pairs=pair_lexicon(fr);ft=tokenize_pairs(fr,pairs);ht=tokenize_pairs(hr,pairs);sol=solve(stats(ft),lm,tag,smoke);hs=fixed(stats(ht),lm,sol["dec"])
    return {"score":hs,"recovery":recover_acc(hw,ht,sol["dec"]),"agreement":sol["agreement"],"converged":sol["converged"],"pair_count":len(pairs),"token_ratio_fit":sum(map(len,ft))/max(1,sum(map(len,fc))),"token_ratio_hold":sum(map(len,ht))/max(1,sum(map(len,hc)))}

def neg_words(kind,rep,total=3500):
    rg=np.random.default_rng(seed(NS,"neg",kind,rep));out=[];n=0;lens=np.array([3,4,5,6,7,8,9,10,11,12]);freq=np.ones(BASE)/BASE
    while n<total:
        L=int(rg.choice(lens));w=[]
        if kind=="iid":w=[int(x) for x in rg.choice(BASE,L,p=freq)]
        elif kind=="markov":
            x=int(rg.integers(0,BASE))
            for j in range(L):x=(x+int(rg.integers(0,5))+rep+j)%BASE;w.append(x)
        elif kind=="motif":
            mot=[int(x) for x in rg.choice(BASE,max(1,min(3,L)),p=freq)];w=(mot*((L+len(mot)-1)//len(mot)))[:L]
        elif kind=="copy":
            if out and rg.random()<.7:w=(list(out[int(rg.integers(0,len(out)))])+[int(rg.integers(0,BASE))]*L)[:L]
            else:w=[int(x) for x in rg.choice(BASE,L,p=freq)]
        elif kind=="slot":
            cls=[list(range(0,5)),list(range(5,10)),list(range(10,15)),list(range(15,19))];w=[int(rg.choice(cls[j%4])) for j in range(L)]
        out.append(w);n+=L
    return out

def neg_eval(lm,kind,rep,smoke=False):
    w=neg_words(kind,rep,1200 if smoke else 4200);cut=len(w)//2;f=w[:cut];h=w[cut:];pairs=pair_lexicon(f);ft=tokenize_pairs(f,pairs);ht=tokenize_pairs(h,pairs);sol=solve(stats(ft),lm,f"NEG:{kind}:{rep}",smoke)
    return {"kind":kind,"rep":rep,"score":fixed(stats(ht),lm,sol["dec"]),"converged":sol["converged"]}

def manifest():
    pages,meta=m.parse_rf();T,H,C1,H2,C2=m.target_split(pages);c=sorted(C2,key=lambda f:hashlib.sha256(f"{NS}::{f}".encode()).digest());bh=c[:len(c)//2];bc=c[len(c)//2:]
    chars=lambda fs:sum(len(x) for f in fs for x in pages[f])
    return pages,{"source":meta,"FIT_A":{"folios":T+H,"chars":chars(T+H),"count":len(T)+len(H)},"BAB_H1":{"folios":bh,"chars":chars(bh),"count":len(bh)},"BAB_C1":{"folios":bc,"chars":chars(bc),"count":len(bc)}}

def target_run(lm,pages,man,smoke=False):
    fw=m.combine(pages,man["FIT_A"]["folios"]);hw=m.combine(pages,man["BAB_H1"]["folios"]);pairs=pair_lexicon(fw);ft=tokenize_pairs(fw,pairs);ht=tokenize_pairs(hw,pairs);sol=solve(stats(ft),lm,"TARGET",smoke);hs=fixed(stats(ht),lm,sol["dec"])
    return {"H1_score":hs,"fit_score":sol["fit_score"],"agreement":sol["agreement"],"converged":sol["converged"],"pair_count":len(pairs),"fit_tokens":sum(map(len,ft)),"H1_tokens":sum(map(len,ht)),"pairs":pairs}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["smoke","qualify","target","manifest"],default="smoke");ap.add_argument("--family",choices=["CV","FULL"],default="CV");ap.add_argument("--workers",type=int,default=8);ap.add_argument("--qual-url");a=ap.parse_args()
    train,ctrl=italian_words();units,inv=make_inventory(train,a.family);lm=build_ulm(train,ctrl,units);base={"family":a.family,"inventory":inv,"lm_meta":lm.meta}
    if a.mode=="manifest":
        pages,man=manifest();print("RESULT_JSON",json.dumps({"base":base,"manifest":man},sort_keys=True));return
    if lm.meta["train"]["char_coverage"]<.995 or lm.meta["control"]["char_coverage"]<.995:
        print("RESULT_JSON",json.dumps({"base":base,"status":"SURFACE_INCOMPATIBLE_PLAINTEXT_COVERAGE"},sort_keys=True));return
    if a.mode in ("smoke","qualify"):
        reps=2 if a.mode=="smoke" else 12;sm=a.mode=="smoke";rows=[]
        for r in range(reps):
            z=control_run(lm,f"Q:{a.family}:{r}",sm);rows.append(z);print("Q",json.dumps(z,sort_keys=True),flush=True)
        floor=float(np.quantile([x["score"] for x in rows],.05));rec=statistics.median(x["recovery"] for x in rows);conv=all(x["converged"] for x in rows);kinds=["iid","markov","motif","copy","slot"];jobs=[(k,r) for k in kinds for r in range(2 if sm else 12)];nr=[]
        def one(q):return neg_eval(lm,q[0],q[1],sm)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
            for z in ex.map(one,jobs):nr.append(z);print("N",json.dumps(z,sort_keys=True),flush=True)
        fp=sum(x["score"]>=floor for x in nr);summary={"ABS_FLOOR":floor,"median_recovery":rec,"all_converged":conv,"false_positives":fp,"neg_trials":len(nr),"pass":bool(rec>=.95 and conv and fp<=(1 if sm else 2))}
        print("RESULT_JSON",json.dumps({"base":base,"controls":rows,"negatives":nr,"summary":summary},sort_keys=True));return
    if not a.qual_url:raise RuntimeError("target requires --qual-url")
    q=json.loads(m.getb(a.qual_url).decode())
    if not q["summary"]["pass"]:raise RuntimeError("qualification did not pass")
    pages,man=manifest();res=target_run(lm,pages,man,False);floor=q["summary"]["ABS_FLOOR"];res["ABS_FLOOR"]=floor;res["abs_pass"]=res["H1_score"]>=floor;res["verdict"]="BAB_H1_CANDIDATE" if (res["converged"] and res["abs_pass"]) else ("UNRESOLVED_SEARCH" if not res["converged"] else "CLOSED_NEGATIVE")
    print("RESULT_JSON",json.dumps({"base":base,"manifest":man,"target":res,"BAB_C1_opened":False},sort_keys=True))

if __name__=="__main__":main()
