#!/usr/bin/env python3
"""S57: held-out known-entry vs continuous prose controls; Voynich boundary diagnostics.
This is a discriminating *instrument* experiment, not decipherment.
No language labels/chronology are inferred from the manuscript.
"""
import argparse,collections,csv,hashlib,json,math,re,os,sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import requests

SEED=20261019
RNG=np.random.default_rng(SEED)
URL="https://www.gutenberg.org/files/49513/49513-h/49513-h.htm"
def words(t):
    return re.findall(r"[a-z]+",t.lower())
def norm_f(w,kind):
    if kind=="word":return [w]
    if kind=="tri":return [w[i:i+3] for i in range(max(0,len(w)-2))] or [w]
    raise ValueError(kind)
def vec(ws,weights,kind):
    z=collections.Counter(v for w in ws for v in norm_f(w,kind))
    a={x:float(n*weights.get(x,0)) for x,n in z.items()}
    denom=math.sqrt(sum(y*y for y in a.values())) or 1.
    return {x:y/denom for x,y in a.items()}
def sim(a,b):
    if len(a)>len(b):a,b=b,a
    return sum(v*b.get(k,0) for k,v in a.items())
def cosine_pair(a,b,weights,kind):
    return sim(vec(a,weights,kind),vec(b,weights,kind))
def derive_weights(entries,kind):
    df=collections.Counter()
    for ent in entries:
        df.update(set(v for w in ent for v in norm_f(w,kind)))
    return {k:math.log((len(entries)+1)/(1+n))+.25 for k,n in df.items()}
def maxdiff(x,cur,idx):
    return min(cur,key=lambda q:abs(q-idx)) if cur else None
def parse_control():
    response=requests.get(URL,timeout=60)
    response.raise_for_status()
    digest=hashlib.sha256(response.content).hexdigest()
    soup=BeautifulSoup(response.content,"html.parser")
    print("HTML",len(response.content),digest,flush=True)
    allheads=soup.find_all(re.compile("^h[234]$"))
    counts=collections.Counter(h.name for h in allheads)
    headings=[(h.name,h.get_text(" ",strip=True)[:70]) for h in allheads[:38]]
    # Mark each short capitalized h3/h4/h2 as a potential plant entry start.
    # We use direct traversal through DOM order, stopping at the next equal/higher-level heading.
    eligible=[]
    for heading in allheads:
        h=heading.get_text(" ",strip=True)
        if len(h)>65 or len(h)<3 or len(words(h))>8:continue
        if not (h.upper()==h or (len(h.split())<5 and h[0].isupper())):continue
        if re.search(r"CHAPTER|BOOK |CONTENTS|INDEX|PLATE|LIST OF|INTRODUCTION|KEY TO|ENGLISH PHYSICIAN|THE COMPLETE",h,re.I):continue
        lv=int(heading.name[1]);t=[]
        for node in heading.next_elements:
            if getattr(node,"name",None) in ["h1","h2","h3","h4"] and node is not heading:
                if int(node.name[1])<=lv:break
            if getattr(node,"name",None)=="p":
                tt=node.get_text(" ",strip=True)
                if len(tt)>35:t.extend(words(tt))
                if len(t)>1700:break
        if 100<=len(t)<=1700:
            eligible.append(dict(heading=h,words=t[:1000],tag=heading.name))
    # Restrict to serial h3/h4 entries rather than distant sections.
    pools=collections.Counter(x["tag"] for x in eligible)
    chosen="h3" if pools["h3"]>=45 else "h4" if pools["h4"]>=45 else max(pools,key=pools.get,default="")
    entries=[x for x in eligible if x["tag"]==chosen]
    print("HEADINGS",json.dumps({"counts":counts,"sample":headings,"candidate_counts":pools,"chosen":chosen,"entries":len(entries),"sample_entries":[(x["heading"],len(x["words"])) for x in entries[:10]]}),flush=True)
    return entries,digest
def build_control(entries):
    assert len(entries)>=40,"QUALIFICATION UNAVAILABLE: fewer than 40 readable plant entries"
    out=[]
    for i in range(len(entries)-1):
        e=entries[i]["words"]; nex=entries[i+1]["words"]
        if min(len(e),len(nex))<100:continue
        cut=len(e)//2
        true=(e[max(0,cut-40):cut],e[cut:cut+40])
        split=(e[-40:],nex[:40])
        if min(map(len,true+split))<25:continue
        out.append(dict(i=i,inside=true,between=split))
    assert len(out)>=30
    return out
def control_instrument(entries,pairs):
    # Hold out both plant entry vocabulary and all boundary examples from the held-out fold.
    # Train/calibrate only a threshold against matched within-corpus examples.
    folds=[[],[],[],[],[]]
    perm=np.random.default_rng(SEED).permutation(len(pairs))
    for i,ix in enumerate(perm):folds[i%5].append(int(ix))
    rows=[];byfold=[]
    for fi,test in enumerate(folds):
        forbidden={pairs[j]["i"] for j in test}|{pairs[j]["i"]+1 for j in test}
        train_entries=[e["words"] for k,e in enumerate(entries) if k not in forbidden]
        weights={k:derive_weights(train_entries,k) for k in ["word","tri"]}
        # Unsupervised, no fitted semantic classifier on Voynich.
        for j in test:
            row=pairs[j]
            for label,key in [(1,"inside"),(0,"between")]:
                a,b=row[key]
                vals={typ:cosine_pair(a,b,weights,typ) for typ in weights}
                rows.append(dict(pair=j,fold=fi,label=label,**vals))
    outputs={}
    for kind in ["word","tri"]:
        positive=np.asarray([r[kind] for r in rows if r["label"]==1])
        negative=np.asarray([r[kind] for r in rows if r["label"]==0])
        # paired, direction-consistent random label swaps
        gaps=positive-negative
        null=np.asarray([np.mean(gaps*np.where(RNG.random(len(gaps))<.5,-1,1))
                         for _ in range(4000)])
        outputs[kind]=dict(n_pairs=len(gaps),auc=float(roc_auc_score([r["label"] for r in rows],[r[kind] for r in rows])),
             mean_true=float(positive.mean()),mean_distinct=float(negative.mean()),
             mean_gap=float(gaps.mean()),null_sd=float(null.std(ddof=1)),
             permutation_p=float((1+np.sum(null>=gaps.mean()))/(1+len(null))))
    return outputs
def source_voynich(rep):
    sys.path.insert(0,str(rep))
    from voynich_toolkit.source_lattice import SourceLattice
    doc=SourceLattice(rep/"ZL_ivtff_raw.txt",verify=True)
    pp={}
    census=0
    for line in doc.stream("all_spaces","preferred_first"):
        loc=line.locus
        if loc.category!="para" or loc.page_vars.get("I")!="H":continue
        texts=[t.text.lower() for t in line.tokens if not t.has_unreadable and "?" not in t.text]
        if not texts:continue
        p=pp.setdefault(loc.page,dict(id=loc.page,quire=loc.page_vars.get("Q"),
          currier=loc.page_vars.get("L"),scribe=loc.page_vars.get("H"),pars=[],line_count=0))
        if any(e.kind=="par_start" for e in loc.elements) or not p["pars"]:p["pars"].append([])
        p["pars"][-1].extend(texts);p["line_count"]+=1;census+=1
    pages=list(pp.values());pages=[p for p in pages if len(p["pars"])>0 and p["currier"] in ("A","B")]
    for p in pages:p["words"]=[v for x in p["pars"] for v in x]
    pages.sort(key=lambda p:(int(re.search(r"\d+",p["id"]).group()),
                            0 if p["id"].endswith("r") else 1))
    assert len(pages)>110 and census>1000,(len(pages),census)
    return pages,dict(loci=census,n_pages=len(pages),tokens=sum(len(p["words"]) for p in pages),
                      by_lang=dict(collections.Counter(p["currier"] for p in pages)))
def boundaries(pages,meta):
    page_byid={p["id"]:p for p in pages}
    for p in pages:p.update(meta.get(p["id"],{}))
    pairs=[]
    for i in range(len(pages)-1):
        a,b=pages[i:i+2];na=int(re.search(r"\d+",a["id"]).group());nb=int(re.search(r"\d+",b["id"]).group())
        rel=("leaf_rv" if na==nb and a["id"].endswith("r") and b["id"].endswith("v") else
             "next_leaf" if nb==na+1 and a["id"].endswith("v") and b["id"].endswith("r") else None)
        if rel and all(a[k]==b[k] for k in ("quire","currier","scribe")):
            pairs.append(dict(a=a,b=b,relation=rel))
    # Conjugate leaves: distinct leaf IDs on same physical bifolium.
    seen=set()
    for i,a in enumerate(pages):
        for b in pages[i+1:]:
            if not a.get("sheet") or a["sheet"]!=b.get("sheet"):continue
            an=int(re.search(r"\d+",a["id"]).group());bn=int(re.search(r"\d+",b["id"]).group())
            if an==bn or (a["id"],b["id"]) in seen:continue
            if not all(a[k]==b[k] for k in ("quire","currier","scribe")):continue
            seen.add((a["id"],b["id"]))
            pairs.append(dict(a=a,b=b,relation="conjugate"))
    return pairs
def source_map(path):
    d={}
    with open(path,encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if row["folio"] not in d:d[row["folio"]]=dict(sheet=row["sheet"])
    return d
def sampled_comparison(pages,pairs,mode,metric,iterations=4000):
    weights=derive_weights([p["words"] for p in pages],metric)
    fun=lambda a,b:cosine_pair(a,b,weights,metric)
    contexts={}
    for p in pages:
        contexts[p["id"]]=dict(head=p["words"][:40],tail=p["words"][-40:])
    sample=[]
    for row in pairs:
        a,b=row["a"],row["b"]
        if len(a["words"])<40 or len(b["words"])<40:continue
        # Keep matched quire, hand, Currier and word-count ratio; wrong sheet.
        other=[q for q in pages if
           all(a[z]==q[z] for z in ["quire","currier","scribe"])
           and q["id"]!=a["id"] and q["id"]!=b["id"]
           and q.get("sheet")!=a.get("sheet")
           and .4<=len(q["words"])/len(b["words"])<=2.5 and len(q["words"])>=40]
        if len(other)<2:continue
        obs=fun(contexts[a["id"]]["tail"],contexts[b["id"]]["head"])
        rev=fun(contexts[b["id"]]["tail"],contexts[a["id"]]["head"])
        controls=np.asarray([fun(contexts[a["id"]]["tail"],contexts[q["id"]]["head"]) for q in other])
        sample.append(dict(ids=[a["id"],b["id"]],obs=obs,rev=rev,controls=controls,quire=a["quire"],currier=a["currier"]))
    if not sample:return {"n":0}
    observed=np.mean([r["obs"] for r in sample])
    choices=np.column_stack([np.zeros(iterations) for r in sample])
    for i,r in enumerate(sample):
        choices[:,i]=r["controls"][RNG.integers(len(r["controls"]),size=iterations)]
    null=choices.mean(1)
    # Cluster bootstrap quire uncertainty, no independent page-pair assumption
    clusters={q:[r for r in sample if r["quire"]==q] for q in set(r["quire"] for r in sample)}
    keys=list(clusters)
    qboot=np.asarray([np.mean([r["obs"]-float(r["controls"].mean()) for k in RNG.choice(keys,len(keys),replace=True) for r in clusters[k]])
                      for _ in range(2000)])
    # Direct forward / reverse label-swap test
    directional=np.asarray([r["obs"]-r["rev"] for r in sample])
    dn=np.asarray([np.mean(directional*RNG.choice([-1,1],size=len(directional))) for _ in range(iterations)])
    return dict(n=len(sample),n_quires=len(clusters),observed=float(observed),matched_null=float(null.mean()),
                delta=float(observed-null.mean()),null_sd=float(null.std(ddof=1)),
                null_p=float((1+np.sum(null>=observed))/(iterations+1)),
                delta_quire_bootstrap_ci=[float(x) for x in np.quantile(qboot,[.025,.975])],
                forward_minus_reverse=float(directional.mean()),
                forward_reverse_p=float((1+np.sum(dn>=directional.mean()))/(iterations+1)),
                top_pairs=sorted([dict(ids=r["ids"],score=round(r["obs"],4),null=round(r["controls"].mean(),4))
                                for r in sample],key=lambda o:o["score"]-o["null"],reverse=True)[:6])
def entry_opener_test(pages):
    # Exact first two-glyph token-shape distribution at page start, compared with the
    # first word of the NEXT paragraph within the same page. Paired permutation.
    rows=[]
    for p in pages:
        if len(p["pars"])<2 or not p["pars"][0] or not p["pars"][1]:continue
        rows.append((p,p["pars"][0][0][:2],p["pars"][1][0][:2]))
    out={}
    for lang in ["A","B"]:
        pairs=[(a,b) for p,a,b in rows if p["currier"]==lang]
        if len(pairs)<10:continue
        aa=[a for a,b in pairs];bb=[b for a,b in pairs]
        freq=collections.Counter(aa+bb)
        top={s for s,_ in freq.most_common(16)}
        a=np.asarray([x if x in top else "OTHER" for x in aa]);b=np.asarray([x if x in top else "OTHER" for x in bb])
        cats=list(set(a)|set(b))
        def js(c,d):
            ac=np.asarray([np.mean(c==cat) for cat in cats]);bc=np.asarray([np.mean(d==cat) for cat in cats])
            m=(ac+bc)/2
            return .5*np.sum(np.where(ac>0,ac*np.log2(np.maximum(ac,1e-15)/m),0))+ .5*np.sum(np.where(bc>0,bc*np.log2(np.maximum(bc,1e-15)/m),0))
        val=js(a,b);sims=[]
        for _ in range(4000):
            flip=RNG.random(len(a))<.5
            sims.append(js(np.where(flip,b,a),np.where(flip,a,b)))
        sims=np.asarray(sims)
        out[lang]=dict(n=len(pairs),observed_js=val,permutation_mean=float(sims.mean()),
                       null_sd=float(sims.std(ddof=1)),p=float((1+np.sum(sims>=val))/4001))
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--replication",required=True);ap.add_argument("--sheet-map",required=True);ap.add_argument("--output",required=True);args=ap.parse_args()
    rep=Path(args.replication).resolve()
    entries,control_digest=parse_control()
    known=build_control(entries)
    ctrl=control_instrument(entries,known)
    print("CONTROL="+json.dumps(ctrl),flush=True)
    # Predeclared qualification: either surface-language feature must discriminate
    # known continuous versus true separated herbal entries with AUC >= .70
    qualified={m:v["auc"]>=.70 and v["permutation_p"]<=.01 and v["n_pairs"]>=30
               for m,v in ctrl.items()}
    pages,corpus=source_voynich(rep)
    pairs=boundaries(pages,source_map(args.sheet_map))
    vms={}
    for rel in ["leaf_rv","next_leaf","conjugate"]:
        relpairs=[p for p in pairs if p["relation"]==rel]
        vms[rel]={m:sampled_comparison(pages,relpairs,rel,m) for m in ["word","tri"]}
    result=dict(program="S57",seed=SEED,control_source=URL,control_sha256=control_digest,
       control_n_entries=len(entries),control_n_pairs=len(known),control_scores=ctrl,
       control_qualified=qualified,corpus=corpus,raw_pair_counts=dict(collections.Counter(z["relation"] for z in pairs)),
       vms=vms,opener=entry_opener_test(pages),
       classification="UNIDENTIFIABLE" if not any(qualified.values()) else "QUALIFIED_FEATURES_ONLY",
       warning="No decipherment, no calibrated probability that entries are standalone; controls are seventeenth-century English herbal in nineteenth-century edition and not a medieval matched language control. Neighboring-page and conjugate contrasts are descriptive of current foliation, not historical original prose order.")
    Path(args.output).mkdir(parents=True,exist_ok=True)
    Path(args.output,"results.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print("RESULT="+json.dumps({k:result[k] for k in ["control_n_entries","control_n_pairs","control_qualified","corpus","raw_pair_counts","classification"]}),flush=True)
    print("VMS="+json.dumps(result["vms"]),flush=True)
    print("OPENER="+json.dumps(result["opener"]),flush=True)
if __name__=="__main__":main()
