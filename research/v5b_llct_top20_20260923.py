#!/usr/bin/env python3
import collections, hashlib, json, math, os, pickle, random, unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

SEED=20260923
NPERM=200
K=20
HERE=Path(__file__).resolve().parent
OUT=HERE/"v5b_llct_checkpoints"
OUT.mkdir(exist_ok=True)

def atomic_pickle(obj,name):
    p=OUT/name; t=p.with_suffix(p.suffix+".tmp")
    with open(t,"wb") as f: pickle.dump(obj,f)
    os.replace(t,p)

def has_letter(s): return any(ch.isalpha() for ch in s)
def norm(s): return unicodedata.normalize("NFC",s).lower()

def parse_llct(path):
    root=ET.parse(path).getroot(); units=[]
    for e in root.iter():
        if e.tag.split("}")[-1]!="LM" or "document_id" not in e.attrib: continue
        seen={}
        for d in e.iter():
            if d is e or d.tag.split("}")[-1]!="LM" or "form" not in d.attrib: continue
            try: i=int(d.attrib["id"])
            except: continue
            if i not in seen: seen[i]=dict(d.attrib)
        ordered=[seen[i] for i in sorted(seen)]
        cur_seg=None; cur=[]
        def flush():
            nonlocal cur_seg,cur
            if cur_seg in ("formulaic","free") and cur:
                units.append({"group":"F" if cur_seg=="formulaic" else "R","tokens":cur[:]})
            cur_seg=None; cur=[]
        for a in ordered:
            s=a.get("seg","")
            if s=="subs" or s not in ("formulaic","free"): flush(); continue
            if cur_seg is not None and s!=cur_seg: flush()
            if cur_seg is None: cur_seg=s
            f=a.get("form","")
            if has_letter(f): cur.append(norm(f))
        flush()
    return units

def topk(units,k):
    c=collections.Counter(t for u in units for t in u)
    return {t for t,_ in c.most_common(k)}, c

def score(units,types):
    h1=o1=h2=o2=0
    for u in units:
        for i in range(1,len(u)):
            if u[i] in types:
                o1+=1; h1 += (u[i]==u[i-1])
        for i in range(2,len(u)):
            if u[i] in types:
                o2+=1; h2 += (u[i]==u[i-2])
    return {1:(h1,o1,h1/o1 if o1 else float("nan")),
            2:(h2,o2,h2/o2 if o2 else float("nan"))}

def run(label,units):
    types,freq=topk(units,K)
    obs=score(units,types)
    rng=random.Random(SEED)
    work=[u[:] for u in units]
    null={1:[],2:[]}
    for p in range(NPERM):
        for u in work: rng.shuffle(u)
        s=score(work,types)
        for lag in (1,2): null[lag].append(s[lag][2])
        if (p+1)%25==0:
            atomic_pickle({"done":p+1,"null":null},f"{label}_p{p+1}.pkl")
    rows=[]
    for lag in (1,2):
        vals=null[lag]; m=sum(vals)/len(vals)
        sd=math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1))
        h,o,r=obs[lag]
        rows.append(dict(corpus=f"LLCT_{label}",k=K,lag=lag,n_units=len(units),
            n_tokens=sum(map(len,units)),observed_hits=h,observed_opportunities=o,
            observed_rate=r,null_mean_rate=m,null_sd_rate=sd,effect_rate=r-m,
            z=((r-m)/sd if sd else None),obs_over_null=(r/m if m else None),
            p_upper=(1+sum(x>=r for x in vals))/(NPERM+1),
            top_types=[t for t,_ in freq.most_common(K)]))
    atomic_pickle(rows,f"{label}_DONE.pkl")
    return rows

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("xml"); args=ap.parse_args()
    md5=hashlib.md5(Path(args.xml).read_bytes()).hexdigest()
    if md5!="ec45ae3a6844fad40a00d2e8050cfc74":
        raise RuntimeError(f"LLCT md5 mismatch: {md5}")
    raw=parse_llct(args.xml)
    groups={g:[u["tokens"] for u in raw if u["group"]==g and len(u["tokens"])>=3] for g in ("F","R")}
    rows=[]
    rows+=run("FORMULAIC",groups["F"])
    rows+=run("FREE",groups["R"])
    out=HERE/"RESULTS_v5b_llct_top20_20260923.json"
    tmp=out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"seed":SEED,"nperm":NPERM,"k":K,"md5":md5,"rows":rows},indent=2))
    os.replace(tmp,out)
    for r in rows: print("V5B",json.dumps(r,sort_keys=True))

if __name__=="__main__": main()
