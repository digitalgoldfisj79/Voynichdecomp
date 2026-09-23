#!/usr/bin/env python3
import collections, json, math, os, pickle, random
from pathlib import Path
from xd1_adapters_20260920 import vms, nuremberg

SEED=20260923
NPERM=200
K=20
HERE=Path(__file__).resolve().parent
OUT=HERE/"v5_primary_checkpoints"
OUT.mkdir(exist_ok=True)

def atomic_pickle(obj,name):
    p=OUT/name; t=p.with_suffix(p.suffix+".tmp")
    with open(t,"wb") as f: pickle.dump(obj,f)
    os.replace(t,p)

def encode(lines):
    freq=collections.Counter(t for r in lines for t in r["tokens"])
    top={t for t,_ in freq.most_common(K)}
    vocab={t:i for i,t in enumerate(freq)}
    top_ids={vocab[t] for t in top}
    arrs=[[vocab[t] for t in r["tokens"]] for r in lines if len(r["tokens"])>=3]
    return arrs,top_ids,sorted(top),sum(map(len,arrs))

def score(arrs,top_ids):
    out={1:[0,0],2:[0,0]}
    for a in arrs:
        n=len(a)
        for lag in (1,2):
            h=o=0
            for i in range(lag,n):
                if a[i] in top_ids:
                    o+=1
                    if a[i]==a[i-lag]: h+=1
            out[lag][0]+=h; out[lag][1]+=o
    return {lag:(h,o,h/o if o else float("nan")) for lag,(h,o) in out.items()}

def run(corpus):
    lines=[r for r in corpus["lines"] if len(r["tokens"])>=3]
    arrs,top_ids,top_types,n_tokens=encode(lines)
    obs=score(arrs,top_ids)
    rng=random.Random(SEED)
    null={1:[],2:[]}
    work=[a[:] for a in arrs]
    for p in range(NPERM):
        for a in work: rng.shuffle(a)
        s=score(work,top_ids)
        for lag in (1,2): null[lag].append(s[lag][2])
        if (p+1)%25==0:
            atomic_pickle({"done":p+1,"null":null},f"{corpus['label']}_p{p+1}.pkl")
    rows=[]
    for lag in (1,2):
        vals=null[lag]; m=sum(vals)/len(vals)
        sd=math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1))
        h,o,r=obs[lag]
        rows.append(dict(corpus=corpus["label"],k=K,lag=lag,n_lines=len(arrs),n_tokens=n_tokens,
          observed_hits=h,observed_opportunities=o,observed_rate=r,null_mean_rate=m,null_sd_rate=sd,
          effect_rate=r-m,z=(r-m)/sd if sd else None,obs_over_null=r/m if m else None,
          p_upper=(1+sum(x>=r for x in vals))/(NPERM+1),top_types=top_types))
    atomic_pickle(rows,f"{corpus['label']}_DONE.pkl")
    return rows

def main():
    V=vms(); Nu,Ne=nuremberg()
    rows=[]
    for c in (V,Nu,Ne):
        rows += run(c)
    out=HERE/"RESULTS_v5_frequency_matched_PRIMARY_20260923.json"
    tmp=out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"seed":SEED,"nperm":NPERM,"k":K,"rows":rows},indent=2))
    os.replace(tmp,out)
    for r in rows: print("V5_PRIMARY",json.dumps(r,sort_keys=True))

if __name__=="__main__": main()
