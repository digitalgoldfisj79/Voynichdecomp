#!/usr/bin/env python3
"""
NK3 rebuild core — prospective external-context test for independently defined token families.

Scientific firewall:
- strict +P0 running text only
- physical bifolium folds from frozen FORM
- family construction may use current-token FORM / spelling / image only
- no external context, section, Stolfi/Mauro label, or exact held-out token lookup in family construction
- discovery folds 2/3; model selection fold 4; final folds 0/1
- nuisance model controls current-token shape + section + frequency
- augmented model adds external context only
- final inference uses context-shuffle and type-label permutation nulls plus physical-block SE
"""
import collections, hashlib, json, math, re, urllib.request
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

SEED=20261005
CORPUS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
CORPUS_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
LAT_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"

m={"__name__":"nk3r_latent"}
exec(compile(urllib.request.urlopen(LAT_URL,timeout=120).read().decode(),LAT_URL,"exec"),m)
segment=m["segment"]; ST=m["ST"]; folds=m["folds"]
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
    z=re.match(r"f(\d+)",str(f)); return int(z.group(1)) if z else None
def section(f):
    n=fnum(f)
    if n is None:return "UNK"
    if n<=66:return "HERBAL"
    if n<=73:return "ASTRO"
    if 75<=n<=84:return "BIO"
    if 85<=n<=102:return "PHARMA"
    if 103<=n<=116:return "RECIPES"
    return "UNK"
def safe_form(t):
    try:
        ps=segment(t);cs=[G8[ST[p]] for p in ps]
        return ps,cs
    except Exception:return None
def plenbin(n):return min(max(n,1),7)-1
def rlenbin(n):return min(max(n,1),12)-1
def freqbin(n):
    if n<=1:return 0
    if n<=2:return 1
    if n<=4:return 2
    if n<=8:return 3
    if n<=16:return 4
    if n<=32:return 5
    return 6
def gapbin(x):
    if x is None:return 7
    if x<=1:return 0
    if x<=2:return 1
    if x<=4:return 2
    if x<=8:return 3
    if x<=16:return 4
    if x<=32:return 5
    return 6
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
            # strict programme population; prior NK3 used a broader contains-"P" check.
            if str(rec.get("u","")) != "+P0":continue
            txt=rec.get("t",{}).get(tid,"")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower()) and safe_form(t.lower()) is not None]
            if not toks:continue
            forms=[safe_form(t) for t in toks]
            for pos,t in enumerate(toks):
                ps,cs=forms[pos]
                rows.append(dict(eid=eid,folio=fol,line=str(ls),pos=pos,line_len=len(toks),
                                 bif=bif,fold=int(folds[bif]),section=section(fol),token=t,line_ord=line_ord,
                                 ps=ps,cs=cs,first=cs[0],final=cs[-1],plen=plenbin(len(ps)),
                                 rlen=rlenbin(len(t)),gall=min(sum(ch in "fkpt" for ch in t),3)))
                eid+=1
    # annotate immediate neighbours on each physical line
    by=collections.defaultdict(list)
    for r in rows:by[(r["folio"],r["line"])].append(r)
    for rs in by.values():
        rs.sort(key=lambda z:z["pos"])
        for i,r in enumerate(rs):
            r["_prev"]=rs[i-1]["token"] if i else None
            r["_next"]=rs[i+1]["token"] if i+1<len(rs) else None
    return rows

def type_counts(rows):
    return collections.Counter(r["token"] for r in rows)

def type_dominant_section(rows):
    by=collections.defaultdict(collections.Counter)
    for r in rows:by[r["token"]][r["section"]]+=1
    return {t:c.most_common(1)[0][0] for t,c in by.items()}

def internal_feature(t):
    sf=safe_form(t)
    if sf is None:return None
    ps,cs=sf
    x=np.zeros(8+8+7+12+8+64+5,float);o=0
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

def _shape_key_token(t,count,domsec):
    sf=safe_form(t)
    if sf is None:return None
    ps,cs=sf
    return (cs[0],cs[-1],plenbin(len(ps)),rlenbin(len(t)),freqbin(count.get(t,0)),domsec)

def build_feature_dicts(rows,fam_map,counts_all):
    """
    Returns occurrence records with nuisance and context dictionaries.
    Context family traces are leave-one-out within split.
    """
    usable=[r for r in rows if r["token"] in fam_map and fam_map[r["token"]] is not None and fam_map[r["token"]]>=0]
    if not usable:return []
    K=max(fam_map[t] for t in fam_map if fam_map[t] is not None and fam_map[t]>=0)+1
    page=collections.defaultdict(collections.Counter);bif=collections.defaultdict(collections.Counter)
    for r in usable:
        f=int(fam_map[r["token"]]);page[r["folio"]][f]+=1;bif[r["bif"]][f]+=1
    # recurrence gaps of same family within folio text order
    byg=collections.defaultdict(list)
    for i,r in enumerate(usable):byg[r["folio"]].append((r["line_ord"],r["pos"],i))
    prevgap={};nextgap={}
    for fol,ls in byg.items():
        ls.sort(key=lambda z:(z[0],z[1]))
        last={}; seq=[]
        for rank,(_,_,i) in enumerate(ls):
            f=int(fam_map[usable[i]["token"]]);prevgap[i]=(rank-last[f] if f in last else None);last[f]=rank;seq.append((rank,i,f))
        nxt={}
        for rank,i,f in reversed(seq):
            nextgap[i]=(nxt[f]-rank if f in nxt else None);nxt[f]=rank
    out=[]
    dom=type_dominant_section(usable)
    for i,r in enumerate(usable):
        t=r["token"];f=int(fam_map[t]);sf=safe_form(t);ps,cs=sf
        n0={
          f"first={r['first']}":1.,f"final={r['final']}":1.,f"plen={r['plen']}":1.,
          f"rlen={r['rlen']}":1.,f"gall={r['gall']}":1.,f"sec={r['section']}":1.,
          f"freq={freqbin(counts_all[t])}":1.,f"raw1={t[:1]}":1.,f"rawN={t[-1:]}":1.,
          f"raw2={t[:2]}":1.,f"rawN2={t[-2:]}":1.
        }
        c={}
        for side,nt in (("p",r["_prev"]),("n",r["_next"])):
            if nt is None or safe_form(nt) is None:
                c[f"{side}_none"]=1.
            else:
                nps,ncs=safe_form(nt)
                c[f"{side}_first={ncs[0]}"]=1.;c[f"{side}_final={ncs[-1]}"]=1.
                c[f"{side}_plen={plenbin(len(nps))}"]=1.;c[f"{side}_rlen={rlenbin(len(nt))}"]=1.
                if nt in fam_map and fam_map[nt] is not None and fam_map[nt]>=0:
                    c[f"{side}_fam={int(fam_map[nt])}"]=1.
        c[f"linepos={linepos(r)}"]=1.;c[f"linelen={min(r['line_len'],12)}"]=1.
        c[f"prevgap={gapbin(prevgap.get(i))}"]=1.;c[f"nextgap={gapbin(nextgap.get(i))}"]=1.
        pc=page[r["folio"]];bc=bif[r["bif"]];pt=sum(pc.values())-1;bt=sum(bc.values())-1
        for j in range(K):
            c[f"pageF{j}"]=(pc[j]-(1 if j==f else 0))/pt if pt>0 else 0.
            c[f"bifF{j}"]=(bc[j]-(1 if j==f else 0))/bt if bt>0 else 0.
        out.append(dict(row=r,y=f,n0=n0,xc=c,
                        shufgrp=(r["fold"],r["section"],linepos(r)),
                        block=r["bif"],token=t,
                        type_shape=_shape_key_token(t,counts_all,dom.get(t,r["section"]))))
    return out

def _fit_logreg(X,y,C):
    return LogisticRegression(C=C,max_iter=400,solver="lbfgs",multi_class="auto",n_jobs=1).fit(X,y)

def _true_logp(model,X,y):
    p=model.predict_proba(X);classes=model.classes_;mp={int(c):i for i,c in enumerate(classes)}
    ix=np.array([mp.get(int(v),-1) for v in y],int);ok=ix>=0
    out=np.full(len(y),-50.,float)
    rr=np.where(ok)[0]
    out[rr]=np.log2(np.maximum(p[rr,ix[rr]],1e-15))
    return out

def _mat(v0,vc,items,fit=False):
    d0=[x["n0"] for x in items];dc=[x["xc"] for x in items]
    if fit:return v0.fit_transform(d0),vc.fit_transform(dc)
    return v0.transform(d0),vc.transform(dc)

def _choose_C(train,val):
    Cs=[.03,.1,.3,1.,3.]
    v0=DictVectorizer(sparse=True);vc=DictVectorizer(sparse=True)
    X0,Xc=_mat(v0,vc,train,True);V0,Vc=_mat(v0,vc,val,False)
    yt=np.array([x["y"] for x in train]);yv=np.array([x["y"] for x in val])
    out={}
    for kind,X,V in (("base",X0,V0),("ctx",sparse.hstack([X0,Xc],format="csr"),sparse.hstack([V0,Vc],format="csr"))):
        best=None
        for C in Cs:
            try:
                md=_fit_logreg(X,yt,C); lp=_true_logp(md,V,yv);loss=-float(np.mean(lp))
            except Exception:continue
            if best is None or loss<best[0]:best=(loss,C)
        if best is None:raise RuntimeError(("no_C",kind))
        out[kind]={"C":best[1],"val_bits":best[0]}
    return out

def _cluster_se(gains,blocks):
    vals=collections.defaultdict(list)
    for g,b in zip(gains,blocks):vals[b].append(float(g))
    B=len(vals);N=len(gains);mu=float(np.mean(gains))
    if B<2:return None,None
    s=0.
    for vs in vals.values():
        ng=len(vs);bs=sum(vs);s+=(bs-ng*mu)**2
    se=math.sqrt((B/(B-1))*s/(N*N))
    return se,(mu/se if se>0 else None)

def evaluate_family(rows,fam_map,nshuffle=250,nlabel=250,seed=SEED):
    counts=type_counts([r for r in rows if r["fold"] in (2,3)])
    tr=build_feature_dicts([r for r in rows if r["fold"] in (2,3)],fam_map,counts)
    va=build_feature_dicts([r for r in rows if r["fold"]==4],fam_map,counts)
    te=build_feature_dicts([r for r in rows if r["fold"] in (0,1)],fam_map,counts)
    if min(len(tr),len(va),len(te))<100:return {"status":"insufficient_occurrences","n":[len(tr),len(va),len(te)]}
    ncls=len(set(x["y"] for x in tr))
    if ncls<2:return {"status":"insufficient_classes","classes":ncls}
    sel=_choose_C(tr,va)
    fit=tr+va
    v0=DictVectorizer(sparse=True);vc=DictVectorizer(sparse=True)
    X0,Xc=_mat(v0,vc,fit,True);T0,Tc=_mat(v0,vc,te,False)
    yf=np.array([x["y"] for x in fit]);yt=np.array([x["y"] for x in te])
    mb=_fit_logreg(X0,yf,sel["base"]["C"]);mc=_fit_logreg(sparse.hstack([X0,Xc],format="csr"),yf,sel["ctx"]["C"])
    lb=_true_logp(mb,T0,yt);lc=_true_logp(mc,sparse.hstack([T0,Tc],format="csr"),yt);gain=lc-lb
    obs=float(np.mean(gain));blocks=[x["block"] for x in te];bse,bz=_cluster_se(gain,blocks)
    # fold-specific signs
    foldgain={}
    for f in (0,1):
        ix=[i for i,x in enumerate(te) if x["row"]["fold"]==f]
        foldgain[str(f)]=float(np.mean(gain[ix])) if ix else None
    # context-row shuffle preserving fold/section/line-position.
    groups=collections.defaultdict(list)
    for i,x in enumerate(te):groups[x["shufgrp"]].append(i)
    rng=np.random.default_rng(seed+len(te)+ncls)
    sn=[]
    for _ in range(nshuffle):
        perm=np.arange(len(te))
        for ix in groups.values():
            if len(ix)>1:
                src=np.array(ix);perm[src]=rng.permutation(src)
        lcp=_true_logp(mc,sparse.hstack([T0,Tc[perm]],format="csr"),yt)
        sn.append(float(np.mean(lcp-lb)))
    sm=float(np.mean(sn));ssd=float(np.std(sn,ddof=1));sz=(obs-sm)/ssd if ssd>0 else None
    # type-consistent family-label permutation within coarse current-shape strata on FINAL only.
    toks=sorted(set(x["token"] for x in te))
    trows=collections.defaultdict(list);tlabel={};tshape={}
    for i,x in enumerate(te):trows[x["token"]].append(i);tlabel[x["token"]]=x["y"];tshape[x["token"]]=x["type_shape"][:4] if x["type_shape"] else None
    strata=collections.defaultdict(list)
    for t in toks:strata[tshape[t]].append(t)
    pn=[]
    pb=mb.predict_proba(T0);pc=mc.predict_proba(sparse.hstack([T0,Tc],format="csr"))
    cb={int(c):i for i,c in enumerate(mb.classes_)};cc={int(c):i for i,c in enumerate(mc.classes_)}
    for _ in range(nlabel):
        yl=dict(tlabel)
        for ts in strata.values():
            labs=[tlabel[t] for t in ts];rng.shuffle(labs)
            for t,z in zip(ts,labs):yl[t]=z
        vals=[]
        for t,ixs in trows.items():
            y=int(yl[t])
            if y not in cb or y not in cc:continue
            for i in ixs:vals.append(math.log2(max(pc[i,cc[y]],1e-15))-math.log2(max(pb[i,cb[y]],1e-15)))
        if vals:pn.append(float(np.mean(vals)))
    pm=float(np.mean(pn)) if pn else None;psd=float(np.std(pn,ddof=1)) if len(pn)>1 else None
    pz=((obs-pm)/psd if psd and psd>0 else None)
    return {
      "status":"ok","n":{"train":len(tr),"val":len(va),"test":len(te),"classes_train":ncls,"test_types":len(toks)},
      "selected_C":sel,"observed_context_gain_bits":obs,
      "physical_block_se":bse,"physical_block_z0":bz,"fold_gain_bits":foldgain,
      "context_shuffle":{"mean":sm,"sd":ssd,"effect":obs-sm,"z":sz,"n":len(sn)},
      "type_label_perm":{"mean":pm,"sd":psd,"effect":(obs-pm if pm is not None else None),"z":pz,"n":len(pn)},
      "gate":bool(sz is not None and sz>2 and pz is not None and pz>2 and foldgain["0"] is not None and foldgain["1"] is not None and foldgain["0"]>0 and foldgain["1"]>0)
    }
