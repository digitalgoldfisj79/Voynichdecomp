#!/usr/bin/env python3
import collections, json, math, os, pickle, random
from pathlib import Path
from xd1_adapters_20260920 import vms, nuremberg

SEED=20260923
NPERM=1000
LAGS=(1,2)
KS=(10,20,50)
HERE=Path(__file__).resolve().parent
OUT=HERE/"v5_checkpoints"
OUT.mkdir(exist_ok=True)

def atomic_pickle(obj,name):
    p=OUT/name
    t=p.with_suffix(p.suffix+".tmp")
    with open(t,"wb") as f: pickle.dump(obj,f)
    os.replace(t,p)

def topk(lines,k):
    c=collections.Counter(t for r in lines for t in r["tokens"])
    return {t for t,_ in c.most_common(k)}

def stat(lines,types,lag):
    hit=opp=0
    for r in lines:
        toks=r["tokens"]
        for i in range(lag,len(toks)):
            if toks[i] in types:
                opp += 1
                hit += (toks[i]==toks[i-lag])
    return hit,opp,(hit/opp if opp else float("nan"))

def run(label,lines):
    rng=random.Random(SEED)
    rows=[]
    for k in KS:
        types=topk(lines,k)
        for lag in LAGS:
            oh,oo,orate=stat(lines,types,lag)
            null=[]
            for _ in range(NPERM):
                sh=[]
                for r in lines:
                    u=list(r["tokens"]); rng.shuffle(u)
                    sh.append({"tokens":u})
                _,_,rr=stat(sh,types,lag)
                null.append(rr)
            m=sum(null)/len(null)
            sd=math.sqrt(sum((x-m)**2 for x in null)/(len(null)-1))
            rows.append(dict(corpus=label,k=k,lag=lag,observed_hits=oh,
                observed_opportunities=oo,observed_rate=orate,null_mean_rate=m,
                null_sd_rate=sd,effect_rate=orate-m,z=((orate-m)/sd if sd else None),
                obs_over_null=(orate/m if m else None),
                p_upper=(1+sum(x>=orate for x in null))/(NPERM+1),
                top_types=sorted(types)))
        atomic_pickle(rows,f"{label}_k{k}.pkl")
    return rows

def main():
    V=vms()
    Nu,Ne=nuremberg()
    corpora=[V,Nu,Ne]
    allrows=[]
    for c in corpora:
        lines=[r for r in c["lines"] if len(r["tokens"])>=3]
        rr=run(c["label"],lines)
        allrows.extend(rr)
        atomic_pickle(dict(label=c["label"],n_lines=len(lines),
                           n_tokens=sum(len(r["tokens"]) for r in lines),
                           rows=rr),f"{c['label']}_done.pkl")
    out=HERE/"RESULTS_v5_frequency_matched_20260923.json"
    tmp=out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(seed=SEED,nperm=NPERM,rows=allrows),indent=2))
    os.replace(tmp,out)
    for r in allrows:
        if r["k"]==20:
            print("V5",json.dumps({x:r[x] for x in ["corpus","k","lag","observed_rate","null_mean_rate","null_sd_rate","effect_rate","z","obs_over_null","p_upper"]},sort_keys=True))

if __name__=="__main__": main()
