#!/usr/bin/env python3
import collections, hashlib, importlib.util, json, math, urllib.request
from pathlib import Path
import numpy as np

VERSION="supergrammar-v03-r64-current-20260920-v1"
BASE_COMMIT="92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5"
BASE_URL=f"https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/{BASE_COMMIT}/research/hf_emergent_occupancy_fold.py"
BASE_SHA="9cbb02b128f019d5b4cff735b8798068f84087afa288270dc3b13e77e1c3d623"
W_REUSE=0.12403024
NREP=50
VARIANTS=("E0_NO_REUSE","E1_R64_UNIFORM","E2_R64_BANDED","E3_LAG1_64_BANDED")
def load_base():
    p=Path("/tmp/sg11base.py");urllib.request.urlretrieve(BASE_URL,p)
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    if got!=BASE_SHA:raise RuntimeError(f"base sha mismatch {got}")
    spec=importlib.util.spec_from_file_location("sg11base",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sample_counter(c,rng):
    items=list(c);w=np.fromiter((c[x] for x in items),float);p=w/w.sum()
    return items[min(int(np.searchsorted(np.cumsum(p),rng.random(),side="right")),len(items)-1)]
def real_rates(rows):
    by=collections.defaultdict(list)
    for r in rows:by[r["page"]].append(r["token"])
    hits1=hits25=n=0
    for toks in by.values():
        for i,t in enumerate(toks):
            if i==0:continue
            n+=1
            if toks[i-1]==t:hits1+=1
            if any(toks[j]==t for j in range(max(0,i-5),max(0,i-1))):hits25+=1
    return dict(lag1=hits1/n if n else 0.0,lag2_5=hits25/n if n else 0.0,n=n)
def choose_exact(fam,row,hist,tc,rng,variant):
    base=tc.get(fam)
    if not base:return "<UNK>"
    if variant=="E0_NO_REUSE" or rng.random()>=W_REUSE:return sample_counter(base,rng)
    page=row["page"];cand=[]
    lo=1 if variant=="E3_LAG1_64_BANDED" else 2
    for lag in range(lo,min(64,len(hist))+1):
        pr=hist[-lag]
        if pr["page"]==page and pr["family"]==fam:
            if variant=="E1_R64_UNIFORM":w=1.0
            else:
                if lag==1:w=1.0
                elif lag<=5:w=1.0
                elif lag<=16:w=.35
                else:w=.15
            cand.append((pr["token"],w))
    if not cand:return sample_counter(base,rng)
    w=np.array([x[1] for x in cand],float);u=rng.random()*w.sum();j=int(np.searchsorted(np.cumsum(w),u,side="right"))
    return cand[min(j,len(cand)-1)][0]
def realise(gen_rows,tc,seed,variant):
    rng=np.random.default_rng(seed);hist=[];out=[]
    for r in gen_rows:
        tok=choose_exact(r["family"],r,hist,tc,rng,variant)
        z=dict(r);z["token"]=tok;hist.append(z);out.append(z)
    return out
def err(r,truth):
    return dict(lag1=abs(r["lag1"]-truth["lag1"]),lag2_5=abs(r["lag2_5"]-truth["lag2_5"]),
                joint=(abs(r["lag1"]-truth["lag1"])+abs(r["lag2_5"]-truth["lag2_5"]))/2)
def signflip(vals):
    a=np.asarray(vals,float);e=float(a.mean());sd=float(np.sqrt(np.sum(a*a))/len(a))
    return dict(effect=e,null_sd=sd,effect_over_null_sd=(abs(e)/sd if sd else None),positive_folds=int((a>0).sum()),negative_folds=int((a<0).sum()))
def main():
    b=load_base();rows,folds=b.load_rows()
    out=dict(version=VERSION,base_commit=BASE_COMMIT,base_sha256=BASE_SHA,w_reuse=W_REUSE,nrep=NREP,folds=[])
    fold_diffs_20=[];fold_diffs_21=[];fold_diffs_23=[];single20=0;singleN=0
    for f in range(5):
        train=[r for r in rows if folds[r["bifolium"]]!=f];test=[r for r in rows if folds[r["bifolium"]]==f]
        fm=b.FamilyModel(train);bank=b.H1Bank(train,fm);tc=collections.defaultdict(collections.Counter)
        for r in train:tc[r["family"]][r["token"]]+=1
        truth=real_rates(test);acc={v:[] for v in VARIANTS}
        for rep in range(NREP):
            famseed=20260920+f*10000+rep
            gen=b.generate_family(fm,bank,test,famseed,"A4_SG1_FULL_NO_CODEBOOK")
            for vi,v in enumerate(VARIANTS):
                rr=realise(gen,tc,9000000+f*100000+rep*10+vi,v);acc[v].append(err(real_rates(rr),truth))
        means={v:{k:float(np.mean([x[k] for x in acc[v]])) for k in ("lag1","lag2_5","joint")} for v in VARIANTS}
        d20=means["E0_NO_REUSE"]["lag2_5"]-means["E2_R64_BANDED"]["lag2_5"]
        d21=means["E1_R64_UNIFORM"]["lag2_5"]-means["E2_R64_BANDED"]["lag2_5"]
        d23=means["E3_LAG1_64_BANDED"]["joint"]-means["E2_R64_BANDED"]["joint"]
        fold_diffs_20.append(d20);fold_diffs_21.append(d21);fold_diffs_23.append(d23)
        for a,c in zip(acc["E0_NO_REUSE"],acc["E2_R64_BANDED"]):
            singleN+=1;single20+=int(c["lag2_5"]<a["lag2_5"])
        out["folds"].append(dict(fold=f,n_test=len(test),truth=truth,mean_error=means,
                                 E2_minus_E0_lag25_improvement=d20,E2_minus_E1_lag25_improvement=d21,
                                 E2_vs_E3_joint_improvement=d23))
    s20=signflip(fold_diffs_20);s21=signflip(fold_diffs_21);s23=signflip(fold_diffs_23)
    out["contrasts"]=dict(E2_vs_E0_lag25=s20,E2_vs_E1_lag25=s21,E2_vs_E3_joint=s23,
                          single_run_E2_beats_E0_lag25=single20/singleN)
    out["decisions"]=dict(
      R64_CURRENT_PASS=bool(s20["effect"]>0 and s20["effect_over_null_sd"]>=2 and s20["positive_folds"]>=4 and single20/singleN>=.75),
      BANDING_OVER_UNIFORM_RESOLVED=bool(s21["effect"]>0 and s21["effect_over_null_sd"]>=2 and s21["positive_folds"]>=4),
      LAG1_EXCLUSION_RESOLVED=bool(s23["effect"]>0 and s23["effect_over_null_sd"]>=2 and s23["positive_folds"]>=4))
    payload=json.dumps(out,sort_keys=True,separators=(",",":"));out["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    print("SGV03_R64="+json.dumps(out,sort_keys=True))
if __name__=="__main__":main()
