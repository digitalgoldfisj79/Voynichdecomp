#!/usr/bin/env python3
"""
EDLC1 core metrics.
Frozen protocol: research/EDLC1_PROTOCOL_20261006.md

No language-specific interpretation occurs here.  The core operates on
surface token records and exposes length, edit-distance graph, matched
resampling, morphology-edge, OOV-repair and clustered-bootstrap metrics.
"""
from __future__ import annotations
import collections, hashlib, math, unicodedata
import numpy as np
import regex
from rapidfuzz.distance import Levenshtein

FREQ_THRESHOLDS=(1,2,3,5)
FREQ_BINS=((1,1),(2,3),(4,7),(8,15),(16,10**18))

def stable_int(s):
    return int(hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:16],16)

def graphemes(s):
    return regex.findall(r"\X",s)

def glen(s):
    return len(graphemes(s))

def _is_letter_cluster(g):
    return bool(regex.search(r"\p{L}",g))

def _is_edge_punct_cluster(g):
    # Only punctuation/separator is stripped at token edges. Combining
    # marks and historical letter-like abbreviation signs are retained.
    return all(unicodedata.category(ch)[0] in ("P","Z","C") for ch in g)

def clean_surface(s):
    if s is None:return None
    s=unicodedata.normalize("NFC",str(s)).casefold().strip()
    if not s:return None
    gs=graphemes(s)
    while gs and _is_edge_punct_cluster(gs[0]): gs.pop(0)
    while gs and _is_edge_punct_cluster(gs[-1]): gs.pop()
    s="".join(gs)
    if not s or not regex.search(r"\p{L}",s):return None
    return s

def folded_surface(s,lang):
    s=clean_surface(s)
    if not s:return None
    s=unicodedata.normalize("NFKC",s).casefold().replace("ſ","s")
    if lang=="latin":
        s=s.replace("j","i").replace("v","u")
    return clean_surface(s)

def freq_bin(n):
    for i,(a,b) in enumerate(FREQ_BINS):
        if a<=n<=b:return i
    return len(FREQ_BINS)-1

def qtile(a,q):
    return float(np.quantile(np.asarray(a,float),q)) if a else None

def _entropy(counter):
    n=sum(counter.values())
    if n<=0:return None
    p=np.asarray(list(counter.values()),float)/n
    return float(-(p*np.log2(p)).sum())

def _basic_lengths(lengths):
    a=np.asarray(lengths,float)
    if not len(a):return {}
    mean=float(a.mean());var=float(a.var(ddof=0));sd=float(math.sqrt(var))
    return {
      "n":int(len(a)),"mean":mean,"median":float(np.median(a)),
      "sd":sd,"variance":var,"cv":float(sd/mean) if mean else None,
      "fano":float(var/mean) if mean else None,
      "q10":float(np.quantile(a,.10)),"q25":float(np.quantile(a,.25)),
      "q75":float(np.quantile(a,.75)),"q90":float(np.quantile(a,.90)),
      "hist":{str(k):int(v) for k,v in sorted(collections.Counter(map(int,a)).items())}
    }

def corpus_summary(records):
    toks=[r["form"] for r in records if r.get("form")]
    fc=collections.Counter(toks)
    type_lengths=[glen(t) for t in fc]
    tok_lengths=[glen(t) for t in toks]
    chars_tok=collections.Counter(g for t in toks for g in graphemes(t))
    chars_type=collections.Counter(g for t in fc for g in graphemes(t))
    ht=_entropy(chars_tok);hy=_entropy(chars_type)
    return {
      "tokens":len(toks),"types":len(fc),
      "length_token":_basic_lengths(tok_lengths),
      "length_type":_basic_lengths(type_lengths),
      "alphabet_token":len(chars_tok),"alphabet_type":len(chars_type),
      "char_entropy_token":ht,"char_entropy_type":hy,
      "effective_alphabet_token":float(2**ht) if ht is not None else None,
      "effective_alphabet_type":float(2**hy) if hy is not None else None,
      "top_types":fc.most_common(20)
    }

class BKNode:
    __slots__=("word","children")
    def __init__(self,word):
        self.word=word;self.children={}

class BKTree:
    def __init__(self,words=()):
        self.root=None
        for w in words:self.add(w)
    def add(self,w):
        if self.root is None:self.root=BKNode(w);return
        node=self.root
        while True:
            d=Levenshtein.distance(w,node.word)
            nxt=node.children.get(d)
            if nxt is None:
                node.children[d]=BKNode(w);return
            node=nxt
    def query(self,w,radius):
        if self.root is None:return []
        out=[];stack=[self.root]
        while stack:
            n=stack.pop();d=Levenshtein.distance(w,n.word)
            if d<=radius:out.append((d,n.word))
            lo=max(0,d-radius);hi=d+radius
            for k,ch in n.children.items():
                if lo<=k<=hi:stack.append(ch)
        return out

def close_edges(words,maxd=3):
    words=sorted(set(words));idx={w:i for i,w in enumerate(words)}
    tree=BKTree(words);by={d:[] for d in range(1,maxd+1)}
    for w in words:
        i=idx[w]
        for d,v in tree.query(w,maxd):
            if d==0:continue
            j=idx[v]
            if i<j:by[d].append((w,v))
    return by

def gini(vals):
    a=np.sort(np.asarray(vals,float))
    if len(a)==0 or a.sum()==0:return 0.0
    n=len(a)
    return float((2*np.sum(np.arange(1,n+1)*a)/(n*a.sum()))-(n+1)/n)

def _component_share(words,edges):
    n=len(words)
    if n==0:return None
    par=list(range(n));sz=[1]*n;ix={w:i for i,w in enumerate(words)}
    def find(a):
        while par[a]!=a:
            par[a]=par[par[a]];a=par[a]
        return a
    def union(a,b):
        a=find(a);b=find(b)
        if a==b:return
        if sz[a]<sz[b]:a,b=b,a
        par[b]=a;sz[a]+=sz[b]
    for a,b in edges:union(ix[a],ix[b])
    return float(max(sz[find(i)] for i in range(n))/n)

def ed_graph_from_counter(fc,minfreq=1,maxd=3):
    words=sorted(t for t,n in fc.items() if n>=minfreq)
    V=len(words);den=V*(V-1)/2 if V>1 else 0
    by=close_edges(words,maxd)
    deg1=collections.Counter();deg2=collections.Counter();deg3=collections.Counter()
    for d,pairs in by.items():
        for a,b in pairs:
            if d==1:deg1[a]+=1;deg1[b]+=1
            if d<=2:deg2[a]+=1;deg2[b]+=1
            if d<=3:deg3[a]+=1;deg3[b]+=1
    near=collections.Counter()
    for w in words:
        if deg1[w]>0:near["1"]+=1
        elif deg2[w]>0:near["2"]+=1
        elif deg3[w]>0:near["3"]+=1
        else:near["4+"]+=1
    p1=len(by.get(1,()))
    p2=p1+len(by.get(2,()))
    p3=p2+len(by.get(3,()))
    d1=np.array([deg1[w] for w in words],float)
    d2=np.array([deg2[w] for w in words],float)
    return {
      "minfreq":minfreq,"V":V,
      "frac_with_ed1":float(np.mean(d1>0)) if V else None,
      "frac_with_ed2":float(np.mean(d2>0)) if V else None,
      "frac_with_ed3":float(np.mean([deg3[w]>0 for w in words])) if V else None,
      "mean_degree_ed1":float(d1.mean()) if V else None,
      "median_degree_ed1":float(np.median(d1)) if V else None,
      "mean_degree_ed2":float(d2.mean()) if V else None,
      "pair_count_ed1":p1,"pair_count_ed2":p2,"pair_count_ed3":p3,
      "pair_density_ed1":float(p1/den) if den else None,
      "pair_density_ed2":float(p2/den) if den else None,
      "pair_density_ed3":float(p3/den) if den else None,
      "nearest_distance":{"1":near["1"],"2":near["2"],"3":near["3"],"4+":near["4+"]},
      "largest_component_ed1":_component_share(words,by.get(1,[])),
      "largest_component_ed2":_component_share(words,by.get(1,[])+by.get(2,[])),
      "degree_gini_ed1":gini(d1),"degree_gini_ed2":gini(d2),
      "_edges":by
    }

def ed_panels(records,thresholds=FREQ_THRESHOLDS):
    fc=collections.Counter(r["form"] for r in records if r.get("form"))
    out={}
    for q in thresholds:
        z=ed_graph_from_counter(fc,q)
        z.pop("_edges",None);out[str(q)]=z
    return out

def morphology_decomposition(records,minfreq=1):
    fc=collections.Counter(r["form"] for r in records if r.get("form"))
    lem=collections.defaultdict(set);ver=collections.defaultdict(set)
    for r in records:
        f=r.get("form");l=r.get("lemma")
        if not f or not l:continue
        lem[f].add(l)
        if r.get("lemma_verified"):ver[f].add(l)
    words=sorted(t for t,n in fc.items() if n>=minfreq)
    by=close_edges(words,2)
    def classify(a,b,m):
        A=m.get(a,set());B=m.get(b,set())
        if not A or not B:return "AMBIGUOUS"
        return "SAME_LEMMA" if A&B else "DIFFERENT_LEMMA"
    res={}
    for label,m in (("all_lemmas",lem),("verified_only",ver)):
        out={}
        for d in (1,2):
            c=collections.Counter();mass=collections.Counter();examples=[]
            for a,b in by[d]:
                k=classify(a,b,m);c[k]+=1;mass[k]+=fc[a]*fc[b]
                if len(examples)<30 and k=="SAME_LEMMA":
                    examples.append({"a":a,"b":b,"freq_a":fc[a],"freq_b":fc[b],
                                     "lemmas":sorted(m.get(a,set())&m.get(b,set()))[:5]})
            n=sum(c.values());mm=sum(mass.values())
            out[str(d)]={"edge_counts":dict(c),
                         "edge_shares":{k:v/n for k,v in c.items()} if n else {},
                         "weighted_mass":dict(mass),
                         "weighted_shares":{k:v/mm for k,v in mass.items()} if mm else {},
                         "same_lemma_examples":examples}
        res[label]=out
    # Within-lemma surface-pair distance distribution.
    rev=collections.defaultdict(set)
    for f,ls in lem.items():
        if fc[f]>=minfreq:
            for l in ls:rev[l].add(f)
    dist=collections.Counter();pairs=0
    for l,forms in rev.items():
        ff=sorted(forms)
        for i,a in enumerate(ff):
            for b in ff[i+1:]:
                d=Levenshtein.distance(a,b,score_cutoff=4)
                k=str(d) if d<=3 else "4+"
                dist[k]+=1;pairs+=1
    res["within_lemma_pair_distance"]={"pairs":pairs,"counts":dict(dist),
       "shares":{k:v/pairs for k,v in dist.items()} if pairs else {}}
    return res

def _nearest_cat(word,tree):
    for r in (1,2,3):
        z=[x for x in tree.query(word,r) if x[0]>0 or x[1]!=word]
        if z:return r
    return 4

def oov_repair(records,nfold=5):
    # Stable block-hash folds. Only genuinely unseen heldout types are scored.
    allout=[]
    for f in range(nfold):
        train=collections.Counter();test=collections.Counter()
        for r in records:
            t=r.get("form");b=r.get("block")
            if not t or b is None:continue
            if stable_int(b)%nfold==f:test[t]+=1
            else:train[t]+=1
        tree=BKTree(train.keys())
        oov={t:n for t,n in test.items() if t not in train}
        tc=collections.Counter();ec=collections.Counter()
        for t,n in oov.items():
            d=_nearest_cat(t,tree);tc[str(d)]+=1;ec[str(d)]+=n
        allout.append({"fold":f,"train_types":len(train),"test_types":len(test),
                       "oov_types":len(oov),"oov_events":sum(oov.values()),
                       "type_distance":dict(tc),"event_distance":dict(ec)})
    T=collections.Counter();E=collections.Counter()
    for z in allout:
        T.update(z["type_distance"]);E.update(z["event_distance"])
    def rates(c):
        n=sum(c.values())
        return {"n":n,
          "ed1":c["1"]/n if n else None,
          "ed2":(c["1"]+c["2"])/n if n else None,
          "ed3":(c["1"]+c["2"]+c["3"])/n if n else None,
          "ge4":c["4"]/n if n else None,
          "counts":dict(c)}
    return {"folds":allout,"type_weighted":rates(T),"event_weighted":rates(E)}

def _stratum(t,n):
    return (glen(t),freq_bin(n))

def matched_resample(target_records,control_records,minfreq=1,nrep=200,seed=20261006):
    tf=collections.Counter(r["form"] for r in target_records if r.get("form"))
    cf=collections.Counter(r["form"] for r in control_records if r.get("form"))
    T=collections.defaultdict(list);C=collections.defaultdict(list)
    for t,n in tf.items():
        if n>=minfreq:T[_stratum(t,n)].append(t)
    for t,n in cf.items():
        if n>=minfreq:C[_stratum(t,n)].append(t)
    strata=sorted(set(T)&set(C))
    k={s:min(len(T[s]),len(C[s])) for s in strata}
    keep=sum(k.values());targetV=sum(len(v) for v in T.values())
    rng=np.random.default_rng(seed)
    vals=[]
    for rep in range(nrep):
        ts=[];cs=[]
        for s in strata:
            kk=k[s]
            if kk<=0:continue
            ts.extend(rng.choice(T[s],size=kk,replace=False).tolist())
            cs.extend(rng.choice(C[s],size=kk,replace=False).tolist())
        tg=ed_graph_from_counter(collections.Counter({x:1 for x in ts}),1)
        cg=ed_graph_from_counter(collections.Counter({x:1 for x in cs}),1)
        vals.append({
          "target_ed1":tg["pair_density_ed1"],"control_ed1":cg["pair_density_ed1"],
          "delta_ed1":tg["pair_density_ed1"]-cg["pair_density_ed1"],
          "target_ed2":tg["pair_density_ed2"],"control_ed2":cg["pair_density_ed2"],
          "delta_ed2":tg["pair_density_ed2"]-cg["pair_density_ed2"]
        })
    def stat(key):
        a=np.asarray([z[key] for z in vals],float)
        return {"mean":float(a.mean()),"sd":float(a.std(ddof=1)) if len(a)>1 else None,
                "q025":float(np.quantile(a,.025)),"q975":float(np.quantile(a,.975))}
    de1=np.asarray([z["delta_ed1"] for z in vals],float)
    de2=np.asarray([z["delta_ed2"] for z in vals],float)
    def dstat(a):
        sd=float(a.std(ddof=1)) if len(a)>1 else 0.0;mu=float(a.mean())
        p=float((1+np.sum(np.abs(a-mu)>=abs(mu)))/(len(a)+1)) if len(a) else None
        # paired-draw z is descriptive; empirical interval is primary.
        return {"mean_effect":mu,"matched_draw_sd":sd,"z":mu/sd if sd else None,
                "q025":float(np.quantile(a,.025)),"q975":float(np.quantile(a,.975)),
                "fraction_positive":float(np.mean(a>0))}
    return {"minfreq":minfreq,"nrep":nrep,"common_types":keep,
            "target_types":targetV,"coverage":keep/targetV if targetV else None,
            "strata":len(strata),"ed1":dstat(de1),"ed2":dstat(de2),
            "target_ed1":stat("target_ed1"),"control_ed1":stat("control_ed1"),
            "target_ed2":stat("target_ed2"),"control_ed2":stat("control_ed2")}

def shuffled_slot_pair_density(words,rng):
    words=list(words);bylen=collections.defaultdict(list)
    for w in words:bylen[glen(w)].append(graphemes(w))
    slots=[]
    for L,arr in bylen.items():
        n=len(arr)
        cols=[]
        for j in range(L):
            c=[x[j] for x in arr];rng.shuffle(c);cols.append(c)
        for i in range(n):slots.append("".join(cols[j][i] for j in range(L)))
    mult=collections.Counter(slots);uniq=sorted(mult)
    den=len(slots)*(len(slots)-1)/2 if len(slots)>1 else 0
    collisions=sum(n*(n-1)/2 for n in mult.values())
    by=close_edges(uniq,2)
    e1=sum(mult[a]*mult[b] for a,b in by[1])
    e2=collisions+e1+sum(mult[a]*mult[b] for a,b in by[2])
    return {"slots":len(slots),"unique":len(uniq),"ed0_collisions":int(collisions),
            "pair_density_ed1":float(e1/den) if den else None,
            "pair_density_ed2":float(e2/den) if den else None}

def positional_shuffle_null(records,minfreq=1,nrep=200,seed=20261006):
    fc=collections.Counter(r["form"] for r in records if r.get("form"))
    words=sorted(t for t,n in fc.items() if n>=minfreq)
    obs=ed_graph_from_counter(collections.Counter({w:1 for w in words}),1)
    rng=np.random.default_rng(seed);vals=[]
    for _ in range(nrep):vals.append(shuffled_slot_pair_density(words,rng))
    def stat(key,obsval):
        a=np.asarray([z[key] for z in vals],float);sd=float(a.std(ddof=1))
        return {"obs":obsval,"null_mean":float(a.mean()),"null_sd":sd,
                "z":float((obsval-a.mean())/sd) if sd else None,
                "q025":float(np.quantile(a,.025)),"q975":float(np.quantile(a,.975))}
    return {"minfreq":minfreq,"nrep":nrep,
            "ed1":stat("pair_density_ed1",obs["pair_density_ed1"]),
            "ed2":stat("pair_density_ed2",obs["pair_density_ed2"]),
            "collision_mean":float(np.mean([z["ed0_collisions"] for z in vals]))}

def cluster_bootstrap_primary(records,minfreq=3,nrep=200,seed=20261006):
    by=collections.defaultdict(list)
    for r in records:
        if r.get("form") and r.get("block") is not None:by[str(r["block"])].append(r["form"])
    blocks=sorted(by)
    if len(blocks)<2:return {"nrep":0,"reason":"<2 blocks"}
    rng=np.random.default_rng(seed);rows=[]
    for _ in range(nrep):
        pick=rng.choice(blocks,size=len(blocks),replace=True)
        fc=collections.Counter()
        tok=[]
        for b in pick:
            fc.update(by[b]);tok.extend(by[b])
        gm=ed_graph_from_counter(fc,minfreq)
        L=[glen(t) for t in tok]
        bl=_basic_lengths(L)
        rows.append((bl["cv"],gm["pair_density_ed1"],gm["pair_density_ed2"]))
    a=np.asarray(rows,float)
    names=("length_cv","ed1_pair_density","ed2_pair_density")
    return {"nrep":nrep,"blocks":len(blocks),
            **{names[j]:{"mean":float(a[:,j].mean()),"sd":float(a[:,j].std(ddof=1)),
                         "q025":float(np.quantile(a[:,j],.025)),"q975":float(np.quantile(a[:,j],.975))}
               for j in range(3)}}
