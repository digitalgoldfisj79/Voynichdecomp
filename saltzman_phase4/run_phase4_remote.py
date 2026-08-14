#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import glob,hashlib,json,lzma,math,sys
import numpy as np
import scorer
from mechanisms import make_plan,encode_with_plan,self_tests as mechanism_self_tests
from persistence_operators import states_for_arm,apply_k2_states,occurrence_all,self_tests as operator_self_tests
from lag_diagnostics import LAGS,autocorr_pm1,theory_for_arm,median_acfs,rmse,self_tests as lag_self_tests

ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA="594d6363522225de8ec46c0e3dab23f04b6cdd341578159772a966e7d1e86930"
SCHEDULE="SWITCH_LINE"
REPS=("ATOMIC","LITERAL")
TARGET_E1=float(scorer.Q_VMS[3])

def verify_protocol():
    p=(ROOT/"protocol.json").read_bytes()
    got=hashlib.sha256(p).hexdigest()
    expected=(ROOT/"PROTOCOL_SHA256").read_text().strip()
    assert got==PROTOCOL_SHA==expected,(got,PROTOCOL_SHA,expected)
    return json.loads(p.decode("utf-8"))

def arms_from_protocol(P):
    arms=["IDENTITY","OCCURRENCE_K2","OCCURRENCE_KALL"]
    arms += [f"FIXED_RUN{L}_K2" for L in P["arms"]["fixed_block_K2"]]
    arms += [f'MARKOV_{x["label"]}_K2' for x in P["arms"]["markov_K2"]]
    assert len(arms)==len(set(arms))
    return tuple(arms)

def eligible_docs(rem):
    d=sorted(k for k,w in rem.items() if len(w)>=2000)
    assert len(d)==190,len(d)
    return d

def w10(words):
    z=list(words)[:2000]
    return [z[i:i+10] for i in range(0,len(z),10) if len(z[i:i+10])>=2]

def compact(r):
    return {k:r[k] for k in ("Q3","Q4","distance3","distance4","gate3","gate4")}

def worker(args):
    doc,words,arms,nreps,nperm=args
    plain=w10(words)
    ntok=sum(len(x) for x in plain)
    assert ntok==2000
    e1avail=scorer.adj_repeats(plain)>0
    buckets={(a,r):[] for a in arms for r in REPS}
    lag_rep={a:[] for a in arms if theory_for_arm(a) is not None}
    for rep in range(nreps):
        plan=make_plan(plain,SCHEDULE,scorer.seed_of("P4-plan",doc,rep))
        state_seed=scorer.seed_of("P4-state",doc,rep)
        states={a:states_for_arm(a,ntok,state_seed) for a in arms}
        for a,s in states.items():
            if s is not None:
                lag_rep[a].append(autocorr_pm1(s))
        for repn in REPS:
            base=encode_with_plan(plain,plan,repn)
            stat_seed=scorer.seed_of("P4-stat",doc,repn,rep)
            occall_seed=scorer.seed_of("P4-occurrence-all",doc,rep)
            transformed={}
            for arm in arms:
                if arm=="IDENTITY":
                    out=base
                elif arm=="OCCURRENCE_KALL":
                    out=occurrence_all(base,occall_seed)
                else:
                    out=apply_k2_states(base,states[arm])
                transformed[arm]=out
            assert transformed["OCCURRENCE_K2"]==transformed["FIXED_RUN1_K2"]
            for arm in arms:
                r=scorer.one_eval(scorer.prep(transformed[arm]),stat_seed,e1avail,nperm)
                buckets[(arm,repn)].append(r)
    cells=[]
    for arm in arms:
        for repn in REPS:
            rows=buckets[(arm,repn)]
            ag=scorer.aggregate(rows)
            cells.append({"corpus":doc,"arm":arm,"representation":repn,**ag,"rows":[compact(r) for r in rows]})
    lag_cells=[]
    for arm,acfs in lag_rep.items():
        med=median_acfs(acfs)
        theory=theory_for_arm(arm)
        lag_cells.append({"corpus":doc,"arm":arm,"lags":med,"theory":theory,"rmse_to_theory":rmse(med,theory)})
    return doc,cells,lag_cells

def run_docs(rem,docs,arms,nreps,nperm):
    tasks=[(d,rem[d],arms,nreps,nperm) for d in docs]
    cells=[];lags=[]
    workers=min(2,len(tasks))
    with ProcessPoolExecutor(max_workers=max(1,workers)) as ex:
        fs=[ex.submit(worker,t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):
            d,c,l=f.result();cells.extend(c);lags.extend(l)
            print("SCORE",i,len(tasks),d,flush=True)
    cells.sort(key=lambda x:(x["corpus"],x["arm"],x["representation"]))
    lags.sort(key=lambda x:(x["corpus"],x["arm"]))
    return cells,lags

def robust_map(cells,field="d3"):
    by={}
    for c in cells:
        by.setdefault((c["corpus"],c["arm"]),{})[c["representation"]]=c
    out={}
    for key,z in by.items():
        if not all(r in z for r in REPS):continue
        if field=="d3":
            v=[z[r]["distance_of_median_Q3"] for r in REPS]
            if all(x is not None and math.isfinite(x) for x in v):out[key]=max(v)
        elif field=="e1err":
            v=[]
            for r in REPS:
                e=z[r]["median_Q4"][3]
                if e is None or e<=0 or not math.isfinite(e):break
                v.append(abs(math.log(e/TARGET_E1)))
            if len(v)==2:out[key]=max(v)
        else:raise ValueError(field)
    return out

def arm_summary(cells,arm):
    d=robust_map(cells,"d3");e=robust_map(cells,"e1err")
    docs=sorted({x[0] for x in d})
    common=[x for x in docs if all((x,a) in d for a in ("IDENTITY","OCCURRENCE_K2",arm))]
    if not common:return None
    I=np.array([d[(x,"IDENTITY")] for x in common],float)
    O=np.array([d[(x,"OCCURRENCE_K2")] for x in common],float)
    C=np.array([d[(x,arm)] for x in common],float)
    medI=float(np.median(I));medO=float(np.median(O));medC=float(np.median(C))
    denom=medI-medO
    retention=(medI-medC)/denom if denom>0 else None
    ec=[x for x in common if all((x,a) in e for a in ("OCCURRENCE_K2",arm))]
    if ec:
        EO=np.array([e[(x,"OCCURRENCE_K2")] for x in ec],float)
        EC=np.array([e[(x,arm)] for x in ec],float)
        ewin=int(np.sum(EC<EO))
        eout={"n_e1":len(ec),"median_e1_log_error":float(np.median(EC)),"median_occurrence_k2_e1_log_error":float(np.median(EO)),"e1_wins_vs_occurrence_k2":ewin,"e1_win_fraction":ewin/len(ec)}
    else:eout={"n_e1":0}
    return {"arm":arm,"n_common":len(common),"median_identity_d3":medI,"median_occurrence_k2_d3":medO,"median_arm_d3":medC,"median_d3_improvement_vs_identity":float(np.median(I-C)),"wins_vs_identity":int(np.sum(C<I)),"win_fraction_vs_identity":float(np.mean(C<I)),"retention_vs_occurrence_k2":retention,**eout}

def bootstrap_median(values,nresamp,seed):
    x=np.asarray(values,float);n=len(x)
    rng=np.random.default_rng(seed);meds=np.empty(nresamp,float);chunk=500
    for i in range(0,nresamp,chunk):
        m=min(chunk,nresamp-i);idx=rng.integers(0,n,size=(m,n));meds[i:i+m]=np.median(x[idx],axis=1)
    return {"n":n,"median":float(np.median(x)),"ci95":[float(np.quantile(meds,.025)),float(np.quantile(meds,.975))]}

def shape_contrast(cells,P):
    d=robust_map(cells,"d3")
    def contrast(middle,endpoints,label):
        req=middle+endpoints
        docs=sorted({doc for doc,arm in d if arm in req});docs=[doc for doc in docs if all((doc,a) in d for a in req)]
        vals=[]
        for doc in docs:
            mid=np.mean([d[(doc,a)] for a in middle]);end=np.mean([d[(doc,a)] for a in endpoints]);vals.append(float(end-mid))
        seed=scorer.seed_of("P4-shape-bootstrap",label)
        return bootstrap_median(vals,P["shape_test"]["bootstrap"]["resamples"],seed)
    return {"fixed":contrast(P["shape_test"]["fixed_middle"],P["shape_test"]["fixed_endpoints"],"FIXED"),"markov":contrast(P["shape_test"]["markov_middle"],P["shape_test"]["markov_endpoints"],"MARKOV")}

def lag_summary(lag_cells):
    by={}
    for x in lag_cells:by.setdefault(x["arm"],[]).append(x)
    out={}
    for arm,rows in sorted(by.items()):
        med=median_acfs([r["lags"] for r in rows]);theory=theory_for_arm(arm)
        out[arm]={"n_docs":len(rows),"median_empirical":med,"theory":theory,"rmse_median_to_theory":rmse(med,theory),"median_doc_rmse":float(np.median([r["rmse_to_theory"] for r in rows if r["rmse_to_theory"] is not None]))}
    return out

def gate_hits(cells):
    g3=g4=0;hits=[]
    for c in cells:
        for i,r in enumerate(c["rows"]):
            g3+=int(r["gate3"]);g4+=int(r["gate4"])
            if r["gate4"]:hits.append({"corpus":c["corpus"],"arm":c["arm"],"representation":c["representation"],"replicate":i,"Q4":r["Q4"],"distance4":r["distance4"]})
    return {"individual_gate3_passes":g3,"individual_gate4_passes":g4,"gate4_hits":hits}

def curve_order(summaries,P):
    sm={x["arm"]:x for x in summaries if x}
    fixed=[(L,sm[f"FIXED_RUN{L}_K2"]["median_arm_d3"]) for L in P["arms"]["fixed_block_K2"]]
    markov=[(x["expected_run"],sm[f'MARKOV_{x["label"]}_K2']["median_arm_d3"]) for x in P["arms"]["markov_K2"]]
    fdown=sum(fixed[i+1][1] <= fixed[i][1] for i in range(len(fixed)-1));mdown=sum(markov[i+1][1] <= markov[i][1] for i in range(len(markov)-1))
    return {"fixed":fixed,"markov":markov,"fixed_nonincreasing_steps":fdown,"markov_nonincreasing_steps":mdown,"longer_persistence_trend":bool(fdown>=5 and mdown>=4)}

def adjudicate(summaries,shape,curves):
    sm={x["arm"]:x for x in summaries if x};O=sm["OCCURRENCE_K2"];R4=sm["FIXED_RUN4_K2"]
    benchmark=O["median_arm_d3"]<O["median_identity_d3"] and O["win_fraction_vs_identity"]>=.60
    p3rep=R4["median_arm_d3"]<R4["median_identity_d3"] and R4["win_fraction_vs_identity"]>=.60
    fs=shape["fixed"];ms=shape["markov"];fstrong=fs["ci95"][0]>0;mstrong=ms["ci95"][0]>0
    if not benchmark or not p3rep:return "P3_MECHANISM_NOT_REPLICATED"
    if fstrong and mstrong:return "INTERMEDIATE_PERSISTENCE_BOTH"
    if (fstrong and ms["median"]>=0) or (mstrong and fs["median"]>=0):return "INTERMEDIATE_PERSISTENCE_PARTIAL"
    if curves["longer_persistence_trend"]:return "LONGER_PERSISTENCE_TREND"
    return "BROAD_SHORT_PERSISTENCE_OR_UNRESOLVED"

def load_rem():return json.loads(lzma.decompress((ROOT/"inputs/rem_docs.json.xz").read_bytes()).decode("utf-8"))

def gather(pattern,key):
    out=[]
    for f in sorted(glob.glob(pattern,recursive=True)):out.extend(json.load(open(f))[key])
    return out

def main():
    P=verify_protocol();mechanism_self_tests();operator_self_tests();lag_self_tests();scorer.self_test_ed1();mode=sys.argv[1];arms=arms_from_protocol(P)
    if mode=="score":
        shard=int(sys.argv[2]);nshards=int(sys.argv[3]);rem=load_rem();docs=eligible_docs(rem)[shard::nshards]
        cells,lags=run_docs(rem,docs,arms,P["simulation"]["replicates_per_document"],P["simulation"]["permutation_replicates"])
        json.dump({"protocol_sha256":PROTOCOL_SHA,"docs":docs,"arms":arms,"cells":cells,"lag_cells":lags},open(f"results/SCORE_SHARD_{shard}_OF_{nshards}.json","w"),ensure_ascii=False)
    elif mode=="merge":
        root=sys.argv[2];cells=gather(root+"/**/SCORE_SHARD_*.json","cells");lags=gather(root+"/**/SCORE_SHARD_*.json","lag_cells");assert len({c["corpus"] for c in cells})==190
        sums=[arm_summary(cells,a) for a in arms];shape=shape_contrast(cells,P);curves=curve_order(sums,P)
        summary={"experiment":P["experiment"],"protocol_sha256":PROTOCOL_SHA,"parent_phase3":P["parent_phase3"],"arm_summaries":sums,"shape_test":shape,"curve_order":curves,"lag_diagnostics":lag_summary(lags),"adjudication":adjudicate(sums,shape,curves),"gates_secondary":gate_hits(cells)}
        json.dump(summary,open("results/SUMMARY.json","w"),indent=2);json.dump({"cells":cells,"lag_cells":lags},open("results/COMBINED.json","w"),ensure_ascii=False);print(json.dumps(summary,indent=2))
    else:raise SystemExit(mode)

if __name__=="__main__":main()
