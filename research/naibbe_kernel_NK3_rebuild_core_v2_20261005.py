#!/usr/bin/env python3
"""
NK3 rebuild v2 — corrected prospective family-context test.

Corrections relative to NK3R-v1:
- strict +P0 only
- no target-conditioned same-family recurrence feature
- no target-conditioned page/bifolium family-count feature in the primary test
- no invalid held-out label-permutation null
- discovery-only frequency and section-propensity nuisance controls
- primary external context = neighbouring tokens only (lags -2,-1,+1,+2) + physical line position
- context shuffle preserves page + line-position class
- physical bifolium cluster SE is primary inferential scale
"""
import collections, hashlib, json, math, re, urllib.request
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

SEED=20261005
CORPUS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
CORPUS_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"

m={"__name__":"nk3rv2_latent"}
exec(compile(urllib.request.urlopen(LAT_URL,timeout=120).read().decode(),LAT_URL,"exec"),m)
segment=m["segment"];ST=m["ST"];folds=m["folds"]
K8_GROUPS=[(0,),(1,),(2,10),(3,7,8),(4,11),(5,),(6,),(9,)]
G8={c:i for i,g in enumerate(K8_GROUPS) for c in g}
SECS=["HERBAL","ASTRO","BIO","PHARMA","RECIPES","UNK"]

PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),
(25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),
(43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),
(71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),
(94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112),(108,111)]
BIF={n:f"B{a:03d}_{b:03d}" for a,b in PAIRS for n in (a,b)}

def fnum(f):
    z=re.match(r"f(\d+)",str(f));return int(z.group(1)) if z else None
def section(f):
    n=fnum(f)
    if n is None:return "UNK"
    if n<=66:return "HERBAL"
    if n<=73:return "ASTRO"
    if 75<=n<=84:return "BIO"
    if n<=102:return "PHARMA"
    if n<=116:return "RECIPES"
    return "UNK"
def safe_form(t):
    try:
        ps=segment(t);cs=[G8[ST[p]] for p in ps];return ps,cs
    except Exception:return None
def plenbin(n):return min(max(n,1),7)-1
def rlenbin(n):return min(max(n,1),12)-1
def freqbin(n):
    if n<=0:return 0
    if n<=1:return 1
    if n<=2:return 2
    if n<=4:return 3
    if n<=8:return 4
    if n<=16:return 5
    if n<=32:return 6
    return 7
def linepos(r):
    if r["pos"]==0:return 0
    if r["pos"]==r["line_len"]-1:return 2
    return 1

def load_obj():
    dat=urllib.request.urlopen(CORPUS_URL,timeout=120).read()
    got=hashlib.sha256(dat).hexdigest()
    if got!=CORPUS_SHA:raise RuntimeError(("corpus_sha",got))
    return json.loads(dat)
OBJ=load_obj()

def build_rows(tid):
    rows=[];eid=0
    for fol,ld in OBJ["pages"].items():
        n=fnum(fol)
        if n not in BIF:continue
        bif=BIF[n]
        if bif not in folds:continue
        for line_ord,(ls,rec) in enumerate(ld.items()):
            if str(rec.get("u",""))!="+P0":continue
            txt=rec.get("t",{}).get(tid,"")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower()) and safe_form(t.lower()) is not None]
            if not toks:continue
            forms=[safe_form(t) for t in toks]
            for pos,t in enumerate(toks):
                ps,cs=forms[pos]
                rows.append(dict(eid=eid,folio=fol,line=str(ls),line_ord=line_ord,pos=pos,line_len=len(toks),
                                 bif=bif,fold=int(folds[bif]),section=section(fol),token=t,ps=ps,cs=cs,
                                 first=cs[0],final=cs[-1],plen=plenbin(len(ps)),rlen=rlenbin(len(t)),
                                 gall=min(sum(ch in "fkpt" for ch in t),3)))
                eid+=1
    by=collections.defaultdict(list)
    for r in rows:by[(r["folio"],r["line"])].append(r)
    for rs in by.values():
        rs.sort(key=lambda z:z["pos"])
        for i,r in enumerate(rs):
            for lag in (1,2):
                r[f"_p{lag}"]=rs[i-lag]["token"] if i-lag>=0 else None
                r[f"_n{lag}"]=rs[i+lag]["token"] if i+lag<len(rs) else None
    return rows

def internal_feature(t):
    sf=safe_form(t)
    if sf is None:return None
    ps,cs=sf;x=np.zeros(8+8+7+12+8+64+5,float);o=0
    x[o+cs[0]]=1;o+=8;x[o+cs[-1]]=1;o+=8
    x[o+plenbin(len(ps))]=1;o+=7;x[o+rlenbin(len(t))]=1;o+=12
    cc=collections.Counter(cs)
    for c,n in cc.items():x[o+c]=n/len(cs)
    o+=8
    if len(cs)>1:
        for a,b in zip(cs[:-1],cs[1:]):x[o+a*8+b]+=1/(len(cs)-1)
    o+=64
    for i,g in enumerate("fkpt"):x[o+i]=float(g in t)
    x[o+4]=sum(t.count(g) for g in "fkpt")/max(1,len(t))
    return x

def discovery_nuisance(rows):
    rr=[r for r in rows if r["fold"] in (2,3)]
    freq=collections.Counter(r["token"] for r in rr)
    sec=collections.defaultdict(collections.Counter)
    for r in rr:sec[r["token"]][r["section"]]+=1
    prop={}
    dom={}
    for t,c in sec.items():
        n=sum(c.values());prop[t]={s:c[s]/n for s in SECS};dom[t]=c.most_common(1)[0][0]
    return freq,prop,dom

def build_items(rows_subset,fam_map,freq,secprop,secdom):
    usable=[r for r in rows_subset if r["token"] in fam_map and fam_map[r["token"]] is not None and fam_map[r["token"]]>=0]
    out=[]
    for r in usable:
        t=r["token"];y=int(fam_map[t])
        n0={
          f"first={r['first']}":1.,f"final={r['final']}":1.,f"plen={r['plen']}":1.,
          f"rlen={r['rlen']}":1.,f"gall={r['gall']}":1.,f"sec={r['section']}":1.,
          f"freq={freqbin(freq.get(t,0))}":1.,f"raw1={t[:1]}":1.,f"rawN={t[-1:]}":1.,
          f"raw2={t[:2]}":1.,f"rawN2={t[-2:]}":1.,
          "seen_disc_type":float(t in freq)
        }
        pp=secprop.get(t,{})
        for s in SECS:n0[f"secprop_{s}"]=float(pp.get(s,0.))
        n0[f"disc_domsec={secdom.get(t,'UNSEEN')}"]=1.
        xc={f"linepos={linepos(r)}":1.,f"linelen={min(r['line_len'],12)}":1.}
        for lag in (1,2):
            for side in ("p","n"):
                nt=r[f"_{side}{lag}"]
                pre=f"{side}{lag}"
                sf=safe_form(nt) if nt else None
                if sf is None:
                    xc[f"{pre}_none"]=1.
                    continue
                ps,cs=sf
                xc[f"{pre}_first={cs[0]}"]=1.;xc[f"{pre}_final={cs[-1]}"]=1.
                xc[f"{pre}_plen={plenbin(len(ps))}"]=1.;xc[f"{pre}_rlen={rlenbin(len(nt))}"]=1.
                if nt in fam_map and fam_map[nt] is not None and fam_map[nt]>=0:
                    xc[f"{pre}_fam={int(fam_map[nt])}"]=1.
        out.append(dict(row=r,y=y,n0=n0,xc=xc,block=r["bif"],token=t,
                        shufgrp=(r["folio"],linepos(r))))
    return out

def _fit(X,y,C):
    return LogisticRegression(C=C,max_iter=500,solver="lbfgs",n_jobs=1).fit(X,y)
def _logp(model,X,y):
    p=model.predict_proba(X);mp={int(c):i for i,c in enumerate(model.classes_)}
    z=np.full(len(y),-50.,float)
    for i,v in enumerate(y):
        j=mp.get(int(v))
        if j is not None:z[i]=math.log2(max(float(p[i,j]),1e-15))
    return z
def _mat(v0,vc,items,fit=False):
    a=[x["n0"] for x in items];b=[x["xc"] for x in items]
    if fit:return v0.fit_transform(a),vc.fit_transform(b)
    return v0.transform(a),vc.transform(b)
def _choose_C(tr,va):
    Cs=[.03,.1,.3,1.,3.]
    v0=DictVectorizer();vc=DictVectorizer();X0,Xc=_mat(v0,vc,tr,True);V0,Vc=_mat(v0,vc,va,False)
    yt=np.array([x["y"] for x in tr]);yv=np.array([x["y"] for x in va])
    out={}
    for name,X,V in (("base",X0,V0),("ctx",sparse.hstack([X0,Xc],format="csr"),sparse.hstack([V0,Vc],format="csr"))):
        cand=[]
        for C in Cs:
            try:
                md=_fit(X,yt,C);cand.append((-float(np.mean(_logp(md,V,yv))),C))
            except Exception:pass
        if not cand:raise RuntimeError(("C selection failed",name))
        loss,C=min(cand);out[name]={"C":C,"val_bits":loss}
    return out
def cluster_se(g,blocks):
    N=len(g);mu=float(np.mean(g));by=collections.defaultdict(list)
    for x,b in zip(g,blocks):by[b].append(float(x))
    B=len(by)
    if B<2:return None,None
    s=sum((sum(v)-len(v)*mu)**2 for v in by.values())
    se=math.sqrt((B/(B-1))*s/(N*N))
    return se,(mu/se if se>0 else None)

def evaluate_family(rows,fam_map,nshuffle=250,seed=SEED):
    freq,secprop,secdom=discovery_nuisance(rows)
    tr=build_items([r for r in rows if r["fold"] in (2,3)],fam_map,freq,secprop,secdom)
    va=build_items([r for r in rows if r["fold"]==4],fam_map,freq,secprop,secdom)
    te=build_items([r for r in rows if r["fold"] in (0,1)],fam_map,freq,secprop,secdom)
    if min(map(len,(tr,va,te)))<100:return {"status":"insufficient_occurrences","n":[len(tr),len(va),len(te)]}
    cls=set(x["y"] for x in tr)
    if len(cls)<2:return {"status":"insufficient_classes","classes":len(cls)}
    sel=_choose_C(tr,va);fit=tr+va
    v0=DictVectorizer();vc=DictVectorizer();X0,Xc=_mat(v0,vc,fit,True);T0,Tc=_mat(v0,vc,te,False)
    yf=np.array([x["y"] for x in fit]);yt=np.array([x["y"] for x in te])
    mb=_fit(X0,yf,sel["base"]["C"]);mc=_fit(sparse.hstack([X0,Xc],format="csr"),yf,sel["ctx"]["C"])
    lb=_logp(mb,T0,yt);lc=_logp(mc,sparse.hstack([T0,Tc],format="csr"),yt);g=lc-lb
    obs=float(np.mean(g));blocks=[x["block"] for x in te];se,z=cluster_se(g,blocks)
    folds={}
    for f in (0,1):
        ix=[i for i,x in enumerate(te) if x["row"]["fold"]==f];folds[str(f)]=float(np.mean(g[ix])) if ix else None
    groups=collections.defaultdict(list)
    for i,x in enumerate(te):groups[x["shufgrp"]].append(i)
    rng=np.random.default_rng(seed+len(te)+len(cls));null=[]
    for _ in range(nshuffle):
        perm=np.arange(len(te))
        for ix in groups.values():
            if len(ix)>1:
                a=np.array(ix);perm[a]=rng.permutation(a)
        lp=_logp(mc,sparse.hstack([T0,Tc[perm]],format="csr"),yt)
        null.append(float(np.mean(lp-lb)))
    nm=float(np.mean(null));ns=float(np.std(null,ddof=1));nz=(obs-nm)/ns if ns>0 else None
    gate=bool(obs>0 and z is not None and z>2 and folds["0"]>0 and folds["1"]>0 and nz is not None and nz>2)
    by=collections.defaultdict(list)
    for x,b in zip(g,blocks):by[b].append(float(x))
    return {"status":"ok","n":{"train":len(tr),"val":len(va),"test":len(te),"classes_train":len(cls),"test_types":len(set(x["token"] for x in te))},
            "selected_C":sel,"observed_context_gain_bits":obs,"physical_block_se":se,"physical_block_z0":z,
            "fold_gain_bits":folds,"context_shuffle":{"mean":nm,"sd":ns,"effect":obs-nm,"z":nz,"n":len(null)},
            "block_gain_mean":{b:float(np.mean(v)) for b,v in by.items()},"gate":gate}

def compare_block_effect(a,b):
    A=a.get("block_gain_mean",{});B=b.get("block_gain_mean",{});ks=sorted(set(A)&set(B))
    if len(ks)<2:return None
    d=np.array([A[k]-B[k] for k in ks],float)
    mu=float(np.mean(d));se=float(np.std(d,ddof=1)/math.sqrt(len(d)))
    return {"blocks":len(ks),"mean_block_gain_difference":mu,"block_se":se,"z":(mu/se if se>0 else None)}
