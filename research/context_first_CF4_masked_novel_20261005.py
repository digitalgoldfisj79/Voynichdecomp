#!/usr/bin/env python3
"""
CF4 masked novel-type bridge after context-first CF3 graph passes.
Family graph is recomputed with the frozen CF3 method from folds2/3 only.
Current token form is invisible to context classifier. Surface is revealed only for scoring.
"""
import collections, itertools, json, math, re, urllib.request
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/23e2892e7123878fed7703c9d5f59d69562fd502/research/context_first_CF1_CF3_graph_20261005.py"
src=urllib.request.urlopen(BASE_URL,timeout=120).read().decode()
prefix=src.split("OUT={}")[0]
ns={"__name__":"cf4base"};exec(compile(prefix,BASE_URL,"exec"),ns)

build_rows=ns["build_rows"];eligible=ns["eligible"];ctx_vocab=ns["ctx_vocab"];baseline=ns["baseline"];prof=ns["prof"]
domsec=ns["domsec"];matched_random_pairs=ns["matched_random_pairs"];sim_disc=ns["sim_disc"];components=ns["components"]
relabel_graph_null=ns["relabel_graph_null"];fbin=ns["fbin"]
safe_form=ns["c"]["safe_form"];ST=ns["c"]["ST"]
SEED=20261005;ALPHA=20.0
GAL=set("fkpt")

# Classical acceptors copied from frozen prior closure work.
MSLOTS=[['','ch','sh','y'],['','eee','ee','e','q','a'],['','o'],['','iiii','iii','ii','i','d'],
        ['','l','k','r','s','t','p','f','cth','ckh','cph','cfh','n','m','y']]
MSET=set()
def _mr(i,s):
    if i==len(MSLOTS):
        if s:MSET.add(s)
        return
    for x in MSLOTS[i]:_mr(i+1,s+x)
_mr(0,'');MMAX=max(map(len,MSET));_MU={}
def mauro_ok(tok):
    if tok in _MU:return _MU[tok]
    n=len(tok);cur={0};ans=False
    for _ in range(5):
        nxt=set()
        for p in cur:
            for j in range(p+1,min(n,p+MMAX)+1):
                if tok[p:j] in MSET:nxt.add(j)
        if n in nxt:ans=True;break
        cur=nxt
        if not cur:break
    _MU[tok]=ans;return ans
R='(?:d|l|r|s|n|x)';O='(?:o|a|y)';Y='(?:y|o)';A='(?:a|o)';N='(?:n|r|l|m|s)'
IN=f'(?:i|ii|iii){N}';Final=f'(?:{Y}|{A}m|{A}{IN})';OptOFinal=f'(?:|{Final}|{O}{Final})'
OR=f'(?:{R}|{O}{R}|{O}{O}{R})';CrP=f'(?:|{OR}|{OR}{OR})';Q=f'(?:q|{Y}q)'
CrustPrefix=f'(?:{CrP}|{Q}{CrP})';CrS=f'(?:|{OR}|{OR}{OR}|{OR}{OR}{OR})'
CrustSuffix=f'(?:{CrS}{OptOFinal})';CrW='(?:'+'|'.join(['']+[OR*i for i in range(1,6)])+')'
WholeCrust=f'(?:{CrW}{OptOFinal}|{Q}{CrW}{OptOFinal})';OE='(?:e|oe)';OEE='(?:ee|oee)'
CH='(?:ch|sh)';OCH=f'(?:{CH}|o{CH}|y{CH})';MtP=f'(?:|{OE}|{OEE}|{OEE}{OE})'
MantlePrefix=f'(?:{MtP}|{OCH}{MtP})';MtS=f'(?:{OEE}|{OEE}{OCH}|{OCH}|{OCH}{OE}|{OCH}{OEE}|{OCH}{OCH}|{OCH}{OE}{OCH}|{OCH}{OE}{OEE}|{OCH}{OCH}{OE})'
WholeMantle=f'(?:{MtS}|{OE}|{OE}{MtS})';G='(?:t|p|k|f)';Gallows=f'(?:{G}|c{G}h)'
OGallows=f'(?:{Gallows}|o{Gallows}|y{Gallows})';Core=f'(?:{OGallows}|{OGallows}{OE})'
MantleSuffix=f'(?:|{MtS})';MantleCore=f'(?:{MantlePrefix}{Core}{MantleSuffix}|{WholeMantle})'
Normal=f'(?:{CrustPrefix}{MantleCore}{CrustSuffix}|{WholeCrust})';STRE=re.compile('^'+Normal+'$')
def stolfi_ok(tok):return bool(STRE.fullmatch(tok))

def lev(a,b):
    p=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        q=[i]
        for j,y in enumerate(b,1):q.append(min(q[-1]+1,p[j]+1,p[j-1]+(x!=y)))
        p=q
    return p[-1]/max(1,len(a),len(b))

def surface(tok):
    z=safe_form(tok)
    if z is None:return None
    ps,cs=z
    return {
      "entry":str(cs[0]),"final":str(cs[-1]),"np":str(min(len(ps),4)),
      "rawlen":str(min(len(tok),12)),"gall":str(min(sum(ch in GAL for ch in tok),2)),
      "stolfi":str(int(stolfi_ok(tok))),"mauro":str(int(mauro_ok(tok))),
      "path":'-'.join(map(str,cs[:4]))+('+' if len(cs)>4 else '')
    }

FEATURES=["entry","final","np","rawlen","gall","stolfi","mauro","path"]

def discover_graph(rows,tid):
    types,dc,vc,tc=eligible(rows);vmap=ctx_vocab(rows);base,glob=baseline(rows,vmap)
    E2=prof(rows,vmap,set(types),2,base,glob);E3=prof(rows,vmap,set(types),3,base,glob)
    ds=domsec(rows);rng=np.random.default_rng(SEED+sum(map(ord,tid)))
    rp=matched_random_pairs(types,dc,ds,rng,6000);rv=np.array([sim_disc(a,b,E2,E3) for a,b in rp])
    q=float(np.quantile(rv,.995));edges=[]
    for i,a in enumerate(types):
        for b in types[i+1:]:
            if sim_disc(a,b,E2,E3)>q:edges.append((a,b))
    return types,dc,ds,vmap,base,glob,q,edges,components(edges)

def context_dict(r,vset):
    d={"section="+str(r["section"]):1.0,"lp="+str(r["lp"]):1.0}
    for lag in (-2,-1,1,2):
        x=r[f"n{lag:+d}"]
        d[f"L{lag}="+(x if x in vset else "OTHER")]=1.0
    return d

def fit_context_classifier(rows,comp,vmap):
    fam={}
    for i,cc in enumerate(comp):
        for t in cc:fam[t]=f"F{i}"
    classes=[f"F{i}" for i in range(len(comp))]+["OTHER"]
    vset=set(vmap)
    # Discovery train / validation C selection; current token only supplies label, never feature.
    tr=[r for r in rows if r["fold"] in (2,3)]
    va=[r for r in rows if r["fold"]==4]
    vec=DictVectorizer(sparse=True)
    X=vec.fit_transform([context_dict(r,vset) for r in tr]);y=np.array([fam.get(r["token"],"OTHER") for r in tr])
    Xv=vec.transform([context_dict(r,vset) for r in va]);yv=np.array([fam.get(r["token"],"OTHER") for r in va])
    best=None
    for C in (.05,.2,1.0,5.0):
        clf=LogisticRegression(C=C,max_iter=400,multi_class='auto',solver='lbfgs')
        clf.fit(X,y);p=clf.predict_proba(Xv);ll=log_loss(yv,p,labels=clf.classes_)
        if best is None or ll<best[0]:best=(ll,C)
    # final refit on 2/3/4 with frozen C and vectorizer vocabulary fitted train+val context ONLY
    fit=[r for r in rows if r["fold"] in (2,3,4)]
    vec2=DictVectorizer(sparse=True);X2=vec2.fit_transform([context_dict(r,vset) for r in fit]);y2=np.array([fam.get(r["token"],"OTHER") for r in fit])
    clf=LogisticRegression(C=best[1],max_iter=500,multi_class='auto',solver='lbfgs');clf.fit(X2,y2)
    return fam,vec2,clf,best

def counts_model(rows,fam,classes):
    # section baseline and family×section residual surface models.
    bg=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    fc=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(collections.Counter)))
    cats=collections.defaultdict(set)
    for r in rows:
        if r["fold"] not in (2,3,4):continue
        s=surface(r["token"])
        if s is None:continue
        sec=str(r["section"]);f=fam.get(r["token"],"OTHER")
        for feat,val in s.items():
            bg[feat][sec][val]+=1;fc[feat][f][sec][val]+=1;cats[feat].add(val)
    return bg,fc,cats

def probs(feat,val,sec,f,bg,fc,cats):
    # baseline section probability with Laplace; family is hierarchically shrunk to baseline.
    co=bg[feat][sec];K=max(1,len(cats[feat]));tot=sum(co.values());pb=(co[val]+.5)/(tot+.5*K)
    cf=fc[feat][f][sec];nt=sum(cf.values());pf=(cf[val]+ALPHA*pb)/(nt+ALPHA)
    return max(pb,1e-12),max(pf,1e-12)

def blockstat(vals,blocks):
    vals=np.array(vals,float);mu=float(vals.mean());by=collections.defaultdict(float);nby=collections.Counter()
    for v,b in zip(vals,blocks):by[b]+=float(v);nby[b]+=1
    B=len(by)
    if B<2:return {"effect":mu,"se":None,"z":None,"n":len(vals),"blocks":B}
    # cluster jackknife-style SE using centered cluster sums
    ss=sum((by[b]-nby[b]*mu)**2 for b in by)
    se=math.sqrt((B/(B-1))*ss/(len(vals)**2))
    return {"effect":mu,"se":se,"z":mu/se if se else None,"n":len(vals),"blocks":B}

def surface_inspection(edges,types,dc,ds,tid):
    rng=np.random.default_rng(SEED+8000+sum(map(ord,tid)))
    obs_ed=float(np.mean([lev(a,b) for a,b in edges])) if edges else None
    obs_entry=[];obs_final=[]
    for a,b in edges:
        sa,sb=surface(a),surface(b)
        if sa and sb:obs_entry.append(sa["entry"]==sb["entry"]);obs_final.append(sa["final"]==sb["final"])
    oe=float(np.mean(obs_entry)) if obs_entry else None;of=float(np.mean(obs_final)) if obs_final else None
    ne=[];nen=[];nfn=[]
    for _ in range(1500):
        rp=matched_random_pairs(types,dc,ds,rng,len(edges)*5)
        if len(rp)<len(edges):continue
        rng.shuffle(rp);rp=rp[:len(edges)]
        ne.append(float(np.mean([lev(a,b) for a,b in rp])))
        ee=[];ff=[]
        for a,b in rp:
            sa,sb=surface(a),surface(b)
            if sa and sb:ee.append(sa["entry"]==sb["entry"]);ff.append(sa["final"]==sb["final"])
        nen.append(float(np.mean(ee)));nfn.append(float(np.mean(ff)))
    return {"normalized_ED":{"obs":obs_ed,"null_mean":float(np.mean(ne)),"null_sd":float(np.std(ne,ddof=1)),
                              "z":float((obs_ed-np.mean(ne))/np.std(ne,ddof=1))},
            "same_entry":{"obs":oe,"null_mean":float(np.mean(nen)),"null_sd":float(np.std(nen,ddof=1)),
                          "z":float((oe-np.mean(nen))/np.std(nen,ddof=1))},
            "same_final":{"obs":of,"null_mean":float(np.mean(nfn)),"null_sd":float(np.std(nfn,ddof=1)),
                          "z":float((of-np.mean(nfn))/np.std(nfn,ddof=1))}}

def run(tid):
    rows=build_rows(tid);types,dc,ds,vmap,base,glob,q,edges,comp=discover_graph(rows,tid)
    fam,vec,clf,sel=fit_context_classifier(rows,comp,vmap)
    classes=list(clf.classes_);bg,fc,cats=counts_model(rows,fam,classes)
    train_types={r["token"] for r in rows if r["fold"] in (2,3,4)}
    novel=[r for r in rows if r["fold"] in (0,1) and r["token"] not in train_types and surface(r["token"]) is not None]
    X=vec.transform([context_dict(r,set(vmap)) for r in novel]);P=clf.predict_proba(X)
    # map classifier columns.
    ci={c:i for i,c in enumerate(clf.classes_)}
    gains={f:[] for f in FEATURES};total=[];blocks=[]
    for ii,r in enumerate(novel):
        sf=surface(r["token"]);sec=str(r["section"]);gall=0.
        for feat in FEATURES:
            val=sf[feat];pb0,_=probs(feat,val,sec,"OTHER",bg,fc,cats)
            mix=0.
            for c in clf.classes_:
                _,pf=probs(feat,val,sec,c,bg,fc,cats);mix+=P[ii,ci[c]]*pf
            gg=math.log2(max(mix,1e-12))-math.log2(pb0);gains[feat].append(gg);gall+=gg
        total.append(gall);blocks.append(r["bif"])
    stats={f:blockstat(gains[f],blocks) for f in FEATURES};stats["TOTAL"]=blockstat(total,blocks)

    # Alignment null: permute mapping among non-OTHER context classes and surface family models; OTHER fixed.
    fclasses=[x for x in clf.classes_ if x!="OTHER"];perms=list(itertools.permutations(fclasses))
    null_total=[];null_st=[];null_mu=[]
    for perm in perms:
        mp=dict(zip(fclasses,perm));mp["OTHER"]="OTHER";vals=[];sv=[];mv=[]
        for ii,r in enumerate(novel):
            sf=surface(r["token"]);sec=str(r["section"]);tt=ss=mm=0.
            for feat in FEATURES:
                val=sf[feat];pb0,_=probs(feat,val,sec,"OTHER",bg,fc,cats);mix=0.
                for c in clf.classes_:
                    _,pf=probs(feat,val,sec,mp[c],bg,fc,cats);mix+=P[ii,ci[c]]*pf
                gg=math.log2(max(mix,1e-12))-math.log2(pb0);tt+=gg
                if feat=="stolfi":ss=gg
                if feat=="mauro":mm=gg
            vals.append(tt);sv.append(ss);mv.append(mm)
        null_total.append(float(np.mean(vals)));null_st.append(float(np.mean(sv)));null_mu.append(float(np.mean(mv)))
    def az(obs,arr):
        arr=np.array(arr,float);return {"obs":obs,"null_mean":float(arr.mean()),"null_sd":float(arr.std(ddof=1)),
                                       "z":float((obs-arr.mean())/arr.std(ddof=1)) if arr.std(ddof=1)>0 else None,"nperm":len(arr)}
    align={"TOTAL":az(stats["TOTAL"]["effect"],null_total),
           "stolfi":az(stats["stolfi"]["effect"],null_st),
           "mauro":az(stats["mauro"]["effect"],null_mu)}

    inspect=surface_inspection(edges,types,dc,ds,tid)
    gate=bool(stats["TOTAL"]["z"] is not None and stats["TOTAL"]["z"]>2 and align["TOTAL"]["z"] is not None and align["TOTAL"]["z"]>2)
    return {"tid":tid,"q995":q,"edges":len(edges),"components":comp,"classifier":{"val_logloss":sel[0],"C":sel[1],"classes":classes},
            "novel_occurrences":len(novel),"surface_inspection":inspect,"surface_prediction":stats,
            "family_alignment_null":align,"gate":gate}

OUT={}
for tid in ("ZLZI","ZLZB","TTLI"):
    r=run(tid);OUT[tid]=r;print("CF4_"+tid+"="+json.dumps(r,separators=(",",":")),flush=True)
primary=bool(OUT["ZLZI"]["gate"] and OUT["ZLZB"]["surface_prediction"]["TOTAL"]["effect"]>0)
print("FINAL_RESULT="+json.dumps({"phase":"CF4_MASKED_NOVEL","status":"COMPLETE","results":OUT,"primary_gate":primary},separators=(",",":")),flush=True)
