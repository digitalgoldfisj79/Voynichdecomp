#!/usr/bin/env python3
"""S56. Joint line-bootstrap x representation x mask x smoothing x start.
Primary unit: pooled bifolium. Immutable public source inputs. Deterministic pickle checkpoints.
No chronology interpretation is built into the resulting consensus ranks.
"""
import argparse, csv, hashlib, importlib.util, json, os, pickle, sys, time
from collections import Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

VERSION = "s56_joint_v1_20261018"
SEED = 20261021
VARIANTS = {"1": None, "2": (-1,-1), "3": (-1,1), "4": (1,-1)}
GROUPS = ["1a","1b","1.5a","2a","3a","3b","3c","4a","4c"]

def atomic_pickle(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+".tmp")
    with t.open("wb") as f: pickle.dump(obj,f,protocol=4)
    os.replace(t,path)

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+".tmp")
    with t.open("w") as f:json.dump(obj,f,indent=2,allow_nan=False)
    os.replace(t,path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def load_configuration(rep_root,representation):
    rep_root=Path(rep_root).resolve()
    team=rep_root/"solve/botanical_enrichment/team_golf"
    core=imported("golf_core_s56",team/"golf_stability_20261016/core.py")
    zl=team/"golf_published_stability_20261016"
    rf=team/"golf_rf1b_stability_20261016"
    pool=json.loads((zl/"ensemble_pool50_point_l010.json").read_text())["design"]
    pg=json.loads((zl/"ensemble_paragraph207_point_l010.json").read_text())["design"]
    meta=pool["meta"]
    assert len(meta)==50 and len(pg["meta"])==207
    block={m["id"]:b for m,b in zip(pg["meta"],pg["blocks"])}
    labels=np.asarray([m["group"] for m in meta])
    masks=np.asarray(pool["masks"],bool)
    assert masks.shape==(100,50)
    anchors=[s["anchor_order"] for s in pool["starts"]]
    assert len(anchors)==8
    if representation=="zl":
        cd=core.load_count_data()
        alphabet=cd["alphabet"]
        expected={k:cd["pool_counts"][k] for k in ["word","line"]}
        source=rep_root/"ZL_ivtff_raw.txt"; unit_mode="zattera"
        baseline=core.distance_from_counts(expected)
        target=core.load_data()["pool_ds"]["combined"]
    else:
        info=json.loads((rf/"inputs.json").read_text())
        dat=np.load(rf/"inputs.npz")
        alphabet=info["alphabet"]
        expected={k:dat["pool_"+k+"_counts"] for k in ["word","line"]}
        source=rf/"RF1b-er.txt";unit_mode="raw"
        baseline=core.distance_from_counts(expected)
        target=dat["pool_distance"]
    assert [m["id"] for m in meta] == [m["id"] for m in (json.loads((rf/"inputs.json").read_text())["pool_meta"])]
    maxerr=float(np.max(np.abs(baseline-target)))
    assert maxerr<1e-12, ("BASE GEOMETRY GATE FAILED", maxerr)
    return dict(core=core,root=rep_root,meta=meta,labels=labels,masks=masks,
                anchors=anchors,block=block,alphabet=alphabet,expected=expected,
                source=source,unit_mode=unit_mode,baseline_error=maxerr)

def prepare_lines(c,rep,cache):
    from voynich_toolkit.source_lattice import SourceLattice
    from voynich_toolkit.glyph_units import units
    cache=Path(cache)
    src_hash=sha(c["source"])
    cache_sig=dict(version=VERSION,source_sha256=src_hash,
                   representation=rep,alphabet=c["alphabet"])
    if cache.exists():
        with cache.open("rb") as f: old=pickle.load(f)
        if old["signature"]==cache_sig: return old["lines"], old["gate"]
    ix={g:i for i,g in enumerate(c["alphabet"])}
    size=len(ix)
    per={m["bifolium"]:[] for m in c["meta"]}
    doc=SourceLattice(str(c["source"]),verify=False)
    total_loci=0
    for line in doc.stream("all_spaces","preferred_first"):
        page=line.locus.page
        if line.locus.category!="para" or page not in c["block"]:continue
        total_loci+=1
        ts=line.tokens
        good=[not t.has_unreadable and "?" not in t.text for t in ts]
        wc,lc=Counter(),Counter()
        def glyphs(w):
            return tuple("other" if g.rare else g.text for g in units(w,c["unit_mode"]))
        for t,ok in zip(ts,good):
            if ok:
                g=glyphs(t.text)
                if g:wc.update(zip(("^",)+g,g+("$",)))
        first=None
        for k in range(len(ts)+1):
            ok=k<len(ts) and good[k]
            if ok and first is None:first=k
            if not ok and first is not None:
                g=glyphs("".join(t.text for t in ts[first:k]))
                if g:
                    lc.update(zip(g,g[1:]))
                    if first==0:lc[("^",g[0])]+=1
                    if k==len(ts):lc[(g[-1],"$")]+=1
                first=None
        matrices=[]
        for cc in [wc,lc]:
            mat=np.zeros((size,size),np.int32)
            for (g,h),count in cc.items():
                assert g in ix and h in ix,("UNKNOWN UNIT",g,h)
                mat[ix[g],ix[h]]=count
            matrices.append(mat)
        per[c["block"][page]].append(tuple(matrices))
    lines=[per[m["bifolium"]] for m in c["meta"]]
    assert all(len(a)>0 for a in lines)
    err={}
    for mode,k in [("word",0),("line",1)]:
        rebuild=np.asarray([np.sum([x[k] for x in group],axis=0) for group in lines])
        err[mode]=int(np.max(np.abs(rebuild-c["expected"][mode])))
        assert err[mode]==0,("LINE RECONSTRUCTION GATE FAILED",rep,mode,err[mode])
    gate=dict(source=src_hash,lines=total_loci,per_sheet=[len(v) for v in lines],
              errors=err,base_distance_error=c["baseline_error"],representation=rep)
    atomic_pickle(cache,dict(signature=cache_sig,gate=gate,lines=lines))
    return lines,gate

def fit_one(z):
    b,rep,rep_root,output=z
    os.environ["OMP_NUM_THREADS"]="1"
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        c=load_configuration(rep_root,rep)
        lines,gate=prepare_lines(c,rep,Path(output)/rep/"lines.pkl")
        dst=Path(output)/rep/"rep_%03d.pkl"%b
        if dst.exists():
            old=pickle.load(dst.open("rb"))
            if old["version"]==VERSION and old["representation"]==rep:
                return dict(rep=rep,b=b,source="checkpoint",fit_count=sum(len(x["fits"]) for x in old["settings"]))
        rng=np.random.default_rng(SEED+b)
        counts={"word":[],"line":[]}
        for sheet in lines:
            picks=rng.integers(len(sheet),size=len(sheet))
            for typ,ki in [("word",0),("line",1)]:
                counts[typ].append(np.sum([sheet[v][ki] for v in picks],axis=0))
        counts={k:np.asarray(v) for k,v in counts.items()}
        d=c["core"].distance_from_counts(counts)
        x,_=c["core"].COMMON.embed(d)
        labels=c["labels"]
        mask=c["masks"][(b-1)%len(c["masks"])]
        xm=x[mask];lm=labels[mask]
        w=np.ones(len(xm))/len(xm)
        centers={g:(xm[lm==g].mean(0) if np.any(lm==g)
                    else x[labels==g].mean(0)) for g in GROUPS}
        result=[]
        for lam in [.01,.05]:
            fits=[]
            for k,anchor in enumerate(c["anchors"]):
                initial=np.asarray([centers[g] for g in anchor])
                grown=c["core"].orient_curve(c["core"].grow_curve(xm,w,initial,lam=lam),x,c["meta"])
                model=c["core"].refine_curve(xm,w,grown["nodes"],grown["edges"],lam=lam)
                t=c["core"].project_curve(x,model)["positions"]
                med={g:float(np.median(t[labels==g])) for g in GROUPS}
                signs=[int(np.sign(med[a]-med[z])) for a,z in
                       [("1a","1b"),("3a","3b"),("4a","4c")]]
                fits.append(dict(start=k,objective=float(model["objective"]),
                    converged=bool(model["converged"]),positions=np.asarray(t),
                    signs=signs))
            result.append(dict(penalty=lam,fits=fits))
        atomic_pickle(dst,dict(version=VERSION,representation=rep,b=b,
                               settings=result,mask=mask,gate=gate))
        return dict(rep=rep,b=b,source="new",fit_count=16,
                    convergence=sum(f["converged"] for s in result for f in s["fits"]))

def run(args):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        c=load_configuration(args.replication,args.representation)
        sys.path.insert(0,str(c["root"]))
        lines,gate=prepare_lines(c,args.representation,Path(args.output)/args.representation/"lines.pkl")
    atomic_json(Path(args.output)/args.representation/"source_gate.json",gate)
    print("SOURCE_GATE="+json.dumps(gate),flush=True)
    tasks=[(b,args.representation,str(c["root"]),args.output)
           for b in range(args.start,args.start+args.count)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures=[ex.submit(fit_one,t) for t in tasks]
        for fut in as_completed(futures):
            print("FIT="+json.dumps(fut.result()),flush=True)
    matches=list((Path(args.output)/args.representation).glob("rep_*.pkl"))
    print("SHARD_COMPLETE="+json.dumps(dict(rep=args.representation,start=args.start,
                                            count=args.count,checkpoint_files=len(matches))),flush=True)

def aggregate(args):
    source=Path(args.input); base=Path(args.replication)
    design=json.loads((base/"solve/botanical_enrichment/team_golf/golf_published_stability_20261016/ensemble_pool50_point_l010.json").read_text())["design"]
    meta=design["meta"]; n=len(meta)
    labels=np.asarray([v["group"] for v in meta])
    results={}; availability={}
    for rep in ["zl","rf"]:
        rs={}
        for b in range(1,101):
            fn=source/rep/("rep_%03d.pkl"%b)
            assert fn.exists(),("MISSING CHECKPOINT",fn)
            rec=pickle.load(fn.open("rb"))
            assert rec["version"]==VERSION and rec["b"]==b
            rs[b]=rec
        results[rep]=rs
    def choose(settings,conditions):
        eligible=[r for r in settings if r["converged"] and r["signs"][0]==-1
              and (conditions is None or tuple(r["signs"][1:])==conditions)]
        return min(eligible,key=lambda r:(r["objective"],r["start"])) if eligible else None
    order_result={}
    for path,cond in VARIANTS.items():
        matrices=[];status=[];select_counts={}
        for rep in ["zl","rf"]:
            for penalty_index,lam in enumerate([.01,.05]):
                after=np.zeros((n,n));ties=np.zeros((n,n));joint=np.zeros((n,n))
                selected=0;absence=0;obj_diffs=[]
                for b in range(1,101):
                    rec=results[rep][b];ss=rec["settings"][penalty_index]["fits"]
                    best=choose(ss,cond)
                    if best is None:absence+=1;continue
                    selected+=1
                    t=np.asarray(best["positions"]);mask=np.asarray(rec["mask"],bool)
                    if best["start"]<0 or len(t)!=n:raise ValueError("BAD FIT")
                    both=mask[:,None]&mask[None,:]
                    delta=t[:,None]-t[None,:]
                    after+=both&(delta>1e-7)
                    ties+=both&(np.abs(delta)<=1e-7)
                    joint+=both
                assert selected>=20,("SPARSE CONDITIONAL PATH",path,rep,lam,selected)
                assert np.all(joint>0),("PAIR MISSING",path,rep,lam)
                fraction=(after+.5*ties)/joint
                np.testing.assert_allclose(fraction+fraction.T,np.ones((n,n)),atol=1e-12)
                matrices.append(fraction)
                status.append(dict(representation=rep,penalty=lam,selected=selected,
                                   omitted=absence,min_joint=int(joint[np.triu_indices(n,1)].min())))
        pair=sum(matrices)/len(matrices)
        score=(pair.sum(axis=1)-.5)/(n-1)
        ranks=np.empty(n,int);ranks[np.argsort(score,kind="stable")]=np.arange(1,n+1)
        iu=np.triu_indices(n,1);dec=np.maximum(pair[iu],1-pair[iu])>=.95
        same=labels[iu[0]]==labels[iu[1]]
        order=np.argsort(ranks)
        def blocks(threshold):
            cuts=[]
            for j in range(1,n):
                if np.all(pair[np.ix_(order[:j],order[j:])] <= 1-threshold+1e-12):cuts.append(j)
            edges=[0]+cuts+[n]
            return [[meta[i]["id"] for i in order[edges[k]:edges[k+1]]]
                    for k in range(len(edges)-1)]
        key=dict(id=path,eligible=status,n_stages=len(blocks(.95)),
             stage_sizes=[len(x) for x in blocks(.95)],
             decisive_all=float(dec.mean()),decisive_same_group=float(dec[same].mean()),
             stages=blocks(.95),rank={m["id"]:int(r) for m,r in zip(meta,ranks)},
             score={m["id"]:float(z) for m,z in zip(meta,score)})
        order_result[path]=key
    prior=json.loads((base.parent/"path-v2/collection.json").read_text())
    prior_meta=prior["cohorts"]["bifolia"]["meta"]
    assert [x["id"] for x in meta]==[x["id"] for x in prior_meta]
    historical={}
    for key,obj in order_result.items():
        pub=prior["cohorts"]["bifolia"]["paths"][key]
        ordered=sorted(meta,key=lambda m:obj["rank"][m["id"]])
        oldr=pub["ranks"]
        rr=np.asarray([obj["rank"][m["id"]] for m in meta])
        inversions=int(sum((rr[i]-rr[j])*(oldr[i]-oldr[j])<0
                           for i in range(n) for j in range(i+1,n)))
        historical[key]=dict(rank_inversions_vs_fixed=int(inversions),
                             mean_absolute_rank_shift=float(np.mean(np.abs(rr-np.asarray(oldr)))))
    res=dict(version=VERSION,scope="pool50",n_replicates=100,
             planned_fits=100*2*2*8,representation=["zl","rf"],
             penalties=[.01,.05],early_constraint="1a before 1b",
             points="published common 90-percent masks",
             texts="independent line bootstrap within each sheet and representation",
             paths=order_result,published_rank_comparison=historical,
             warning="Retained-fit frequencies are model sensitivity, not chronology probabilities. Threshold stage boundaries are descriptive, not null-tested.")
    atomic_json(Path(args.output)/"s56_joint_results.json",res)
    with open(Path(args.output)/"s56_joint_ranks.csv","w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["bifolium","group"]+[f"path_{k}_rank" for k in VARIANTS])
        for m in meta:writer.writerow([m["id"],m["group"]]+[order_result[k]["rank"][m["id"]] for k in VARIANTS])
    print("FINAL="+json.dumps({k:dict(stages=v["n_stages"],sizes=v["stage_sizes"],
                     same_group_decisive=v["decisive_same_group"]) for k,v in order_result.items()}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["run","aggregate"],required=True)
    p.add_argument("--representation",choices=["zl","rf"],default="zl")
    p.add_argument("--replication",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--input",default="")
    p.add_argument("--start",type=int,default=1)
    p.add_argument("--count",type=int,default=25)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    if a.mode=="run":
        assert a.start>=1 and a.start+a.count<=101
        run(a)
    else:aggregate(a)

if __name__=="__main__":main()
