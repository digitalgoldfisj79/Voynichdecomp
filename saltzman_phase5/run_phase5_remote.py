#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import glob,hashlib,json,lzma,math,sys
import numpy as np
import scorer
from mechanisms import make_plan,encode_with_plan,self_tests as mechanism_self_tests
from persistence_operators import states_for_arm,apply_k2_states,self_tests as operator_self_tests
from design_utils import WIDTHS,lineate_words,ols_slope,slope_x,adjudicate as design_adjudicate,self_tests as design_self_tests

ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA="74aaeffd009fbcebaa57327c2867ee15bf501385cdb822c9af9528a98c73a96a"
SCHEDULE="SWITCH_LINE"
REPS=("ATOMIC","LITERAL")
TARGET=np.asarray(scorer.Q_VMS,float)
TARGET_E1=float(TARGET[3])

def verify_protocol():
    p=(ROOT/"protocol.json").read_bytes()
    got=hashlib.sha256(p).hexdigest()
    expected=(ROOT/"PROTOCOL_SHA256").read_text().strip()
    assert got==PROTOCOL_SHA==expected,(got,PROTOCOL_SHA,expected)
    P=json.loads(p.decode("utf-8"))
    assert tuple(P["line_scale"]["token_widths"])==WIDTHS
    return P

def arms_from_protocol(P):
    arms=["IDENTITY","OCCURRENCE_K2"]
    arms += [f"FIXED_RUN{L}_K2" for L in P["arms"]["fixed_block_K2"]]
    arms += [f'MARKOV_{x["label"]}_K2' for x in P["arms"]["markov_K2"]]
    assert len(arms)==len(set(arms))
    return tuple(arms)

def eligible_docs(rem):
    d=sorted(k for k,w in rem.items() if len(w)>=2000)
    assert len(d)==190,len(d)
    return d

def compact(r):
    return {k:r[k] for k in ("Q3","Q4","distance3","distance4","gate3","gate4")}

def worker(args):
    doc,words,width,arms,nreps,nperm=args
    plain=lineate_words(words,width)
    ntok=sum(len(x) for x in plain)
    assert ntok==2000,(doc,width,ntok)
    e1avail=scorer.adj_repeats(plain)>0
    buckets={(a,r):[] for a in arms for r in REPS}
    for rep in range(nreps):
        plan=make_plan(plain,SCHEDULE,scorer.seed_of("P5-plan",doc,rep))
        state_seed=scorer.seed_of("P5-state",doc,rep)
        states={a:states_for_arm(a,ntok,state_seed) for a in arms}
        for repn in REPS:
            base=encode_with_plan(plain,plan,repn)
            stat_seed=scorer.seed_of("P5-stat",doc,repn,rep)
            transformed={}
            for arm in arms:
                if arm=="IDENTITY":
                    out=base
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
            cells.append({
                "corpus":doc,"line_width":int(width),"arm":arm,"representation":repn,
                **ag,"rows":[compact(r) for r in rows]
            })
    return doc,cells

def run_docs(rem,docs,width,arms,nreps,nperm):
    tasks=[(d,rem[d],width,arms,nreps,nperm) for d in docs]
    cells=[]
    workers=min(2,len(tasks))
    with ProcessPoolExecutor(max_workers=max(1,workers)) as ex:
        fs=[ex.submit(worker,t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):
            d,c=f.result();cells.extend(c)
            print("SCORE",width,i,len(tasks),d,flush=True)
    cells.sort(key=lambda x:(x["line_width"],x["corpus"],x["arm"],x["representation"]))
    return cells

def robust_map(cells,field="d3"):
    by={}
    for c in cells:
        key=(c["corpus"],int(c["line_width"]),c["arm"])
        by.setdefault(key,{})[c["representation"]]=c
    out={}
    for key,z in by.items():
        if not all(r in z for r in REPS):
            continue
        if field=="d3":
            v=[z[r]["distance_of_median_Q3"] for r in REPS]
            if all(x is not None and math.isfinite(x) for x in v):
                out[key]=max(v)
        elif field=="e1err":
            v=[]
            for r in REPS:
                e=z[r]["median_Q4"][3]
                if e is None or e<=0 or not math.isfinite(e):
                    break
                v.append(abs(math.log(e/TARGET_E1)))
            if len(v)==2:
                out[key]=max(v)
        else:
            raise ValueError(field)
    return out

def arm_summary(cells,width,arm):
    d=robust_map(cells,"d3"); e=robust_map(cells,"e1err")
    docs=sorted({x[0] for x in d if x[1]==width})
    common=[x for x in docs if all((x,width,a) in d for a in ("IDENTITY","OCCURRENCE_K2",arm))]
    if not common:
        return None
    I=np.array([d[(x,width,"IDENTITY")] for x in common],float)
    O=np.array([d[(x,width,"OCCURRENCE_K2")] for x in common],float)
    C=np.array([d[(x,width,arm)] for x in common],float)
    medI=float(np.median(I)); medO=float(np.median(O)); medC=float(np.median(C))
    ec=[x for x in common if all((x,width,a) in e for a in ("OCCURRENCE_K2",arm))]
    if ec:
        EO=np.array([e[(x,width,"OCCURRENCE_K2")] for x in ec],float)
        EC=np.array([e[(x,width,arm)] for x in ec],float)
        eout={
            "n_e1":len(ec),
            "median_e1_log_error":float(np.median(EC)),
            "median_occurrence_k2_e1_log_error":float(np.median(EO)),
            "e1_wins_vs_occurrence_k2":int(np.sum(EC<EO)),
            "e1_win_fraction":float(np.mean(EC<EO))
        }
    else:
        eout={"n_e1":0}
    return {
        "line_width":int(width),"arm":arm,"n_common":len(common),
        "median_identity_d3":medI,"median_occurrence_k2_d3":medO,"median_arm_d3":medC,
        "median_d3_improvement_vs_identity":float(np.median(I-C)),
        "wins_vs_identity":int(np.sum(C<I)),"win_fraction_vs_identity":float(np.mean(C<I)),
        **eout
    }

def bootstrap_median(values,nresamp,seed):
    x=np.asarray(values,float); n=len(x)
    if n==0:
        return {"n":0,"median":None,"ci95":[None,None]}
    rng=np.random.default_rng(seed); meds=np.empty(nresamp,float); chunk=500
    for i in range(0,nresamp,chunk):
        m=min(chunk,nresamp-i)
        idx=rng.integers(0,n,size=(m,n))
        meds[i:i+m]=np.median(x[idx],axis=1)
    return {
        "n":n,"median":float(np.median(x)),
        "ci95":[float(np.quantile(meds,.025)),float(np.quantile(meds,.975))]
    }

def family_arms(P,family):
    T=P["primary_test"]
    if family=="fixed":
        return T["fixed_short_arms"],T["fixed_long_arms"]
    if family=="markov":
        return T["markov_short_arms"],T["markov_long_arms"]
    raise ValueError(family)

def contrast_values(d,width,short,long):
    docs=sorted({doc for (doc,w,a) in d if w==width and a in short+long})
    docs=[doc for doc in docs if all((doc,width,a) in d for a in short+long)]
    vals={}
    for doc in docs:
        s=float(np.mean([d[(doc,width,a)] for a in short]))
        l=float(np.mean([d[(doc,width,a)] for a in long]))
        vals[doc]=l-s
    return vals

def family_contrasts(cells,P,field="d3"):
    d=robust_map(cells,field)
    out={}
    for fam in ("fixed","markov"):
        short,long=family_arms(P,fam)
        out[fam]={}
        for width in WIDTHS:
            vals=contrast_values(d,width,short,long)
            seed=scorer.seed_of("P5-bootstrap",field,fam,width)
            out[fam][int(width)]=bootstrap_median(
                list(vals.values()),P["primary_test"]["bootstrap"]["resamples"],seed
            )
    return out

def interaction_slopes(cells,P,field="d3"):
    d=robust_map(cells,field); xs=slope_x(WIDTHS); out={}
    for fam in ("fixed","markov"):
        short,long=family_arms(P,fam)
        perw={w:contrast_values(d,w,short,long) for w in WIDTHS}
        docs=sorted(set.intersection(*[set(perw[w]) for w in WIDTHS]))
        slopes=[]
        for doc in docs:
            ys=[perw[w][doc] for w in WIDTHS]
            slopes.append(ols_slope(xs,ys))
        seed=scorer.seed_of("P5-bootstrap",field,fam,"slope")
        out[fam]=bootstrap_median(
            slopes,P["primary_test"]["bootstrap"]["resamples"],seed
        )
    return out

def representation_curves(cells,arms):
    out={}
    for width in WIDTHS:
        out[str(width)]={}
        for arm in arms:
            out[str(width)][arm]={}
            for repn in REPS:
                vals=[
                    c["distance_of_median_Q3"] for c in cells
                    if c["line_width"]==width and c["arm"]==arm and c["representation"]==repn
                    and c["distance_of_median_Q3"] is not None
                ]
                out[str(width)][arm][repn]=float(np.median(vals)) if vals else None
    return out

def q3_component_descriptives(cells,arms):
    out={}
    for width in WIDTHS:
        out[str(width)]={}
        for arm in arms:
            out[str(width)][arm]={}
            for repn in REPS:
                sub=[c for c in cells if c["line_width"]==width and c["arm"]==arm and c["representation"]==repn]
                qout={}
                for j in range(3):
                    vals=[c["median_Q3"][j] for c in sub if c["median_Q3"][j] is not None and c["median_Q3"][j]>0]
                    lr=np.log(np.asarray(vals,float)/TARGET[j]) if vals else np.array([])
                    qout[f"Q{j+1}_median_abs_log_error"]=float(np.median(np.abs(lr))) if len(lr) else None
                    qout[f"Q{j+1}_median_signed_log_bias"]=float(np.median(lr)) if len(lr) else None
                out[str(width)][arm][repn]=qout
    return out

def tau_map(P,family):
    if family=="fixed":
        return {f"FIXED_RUN{L}_K2":float(L) for L in P["arms"]["fixed_block_K2"]}
    return {f'MARKOV_{x["label"]}_K2':float(x["tau_int"]) for x in P["arms"]["markov_K2"]}

def discrete_tau_minima(cells,P):
    d=robust_map(cells,"d3"); out={}
    for fam in ("fixed","markov"):
        amap=tau_map(P,fam); out[fam]={}
        for width in WIDTHS:
            curve=[]
            for arm,tau in amap.items():
                vals=[v for (doc,w,a),v in d.items() if w==width and a==arm]
                curve.append({"arm":arm,"tau_int":tau,"median_robust_d3":float(np.median(vals)) if vals else None})
            valid=[x for x in curve if x["median_robust_d3"] is not None]
            best=min(valid,key=lambda x:(x["median_robust_d3"],x["tau_int"]))
            out[fam][str(width)]={"tau_star":best["tau_int"],"arm":best["arm"],"curve":curve}
    return out

def replication_ok(arm_summaries):
    sm={(x["line_width"],x["arm"]):x for x in arm_summaries if x}
    width=10
    for arm in ("OCCURRENCE_K2","FIXED_RUN4_K2","MARKOV_M4_K2"):
        x=sm[(width,arm)]
        if not (x["median_arm_d3"]<x["median_identity_d3"] and x["win_fraction_vs_identity"]>=0.60):
            return False
    return True

def gate_hits(cells):
    g3=g4=0; hits=[]
    for c in cells:
        for i,r in enumerate(c["rows"]):
            g3+=int(r["gate3"]); g4+=int(r["gate4"])
            if r["gate4"]:
                hits.append({
                    "corpus":c["corpus"],"line_width":c["line_width"],"arm":c["arm"],
                    "representation":c["representation"],"replicate":i,
                    "Q4":r["Q4"],"distance4":r["distance4"]
                })
    return {"individual_gate3_passes":g3,"individual_gate4_passes":g4,"gate4_hits":hits}

def load_rem():
    return json.loads(lzma.decompress((ROOT/"inputs/rem_docs.json.xz").read_bytes()).decode("utf-8"))

def gather(pattern,key):
    out=[]
    for f in sorted(glob.glob(pattern,recursive=True)):
        z=json.load(open(f))
        assert z["protocol_sha256"]==PROTOCOL_SHA
        out.extend(z[key])
    return out

def main():
    P=verify_protocol(); mechanism_self_tests(); operator_self_tests(); design_self_tests(); scorer.self_test_ed1()
    mode=sys.argv[1]; arms=arms_from_protocol(P)
    if mode=="score":
        width=int(sys.argv[2]); shard=int(sys.argv[3]); nshards=int(sys.argv[4])
        assert width in WIDTHS
        rem=load_rem(); docs=eligible_docs(rem)[shard::nshards]
        cells=run_docs(rem,docs,width,arms,P["simulation"]["replicates_per_document"],P["simulation"]["permutation_replicates"])
        json.dump({
            "protocol_sha256":PROTOCOL_SHA,"line_width":width,"docs":docs,"arms":arms,"cells":cells
        },open(f"results/SCORE_W{width}_SHARD_{shard}_OF_{nshards}.json","w"),ensure_ascii=False)
    elif mode=="merge":
        root=sys.argv[2]
        cells=gather(root+"/**/SCORE_W*_SHARD_*.json","cells")
        got={w:len({c["corpus"] for c in cells if c["line_width"]==w}) for w in WIDTHS}
        assert all(got[w]==190 for w in WIDTHS),got
        arm_sums=[arm_summary(cells,w,a) for w in WIDTHS for a in arms]
        contrasts=family_contrasts(cells,P,"d3")
        slopes=interaction_slopes(cells,P,"d3")
        e1_contrasts=family_contrasts(cells,P,"e1err")
        e1_slopes=interaction_slopes(cells,P,"e1err")
        repok=replication_ok(arm_sums)
        verdict=design_adjudicate(
            repok,contrasts,slopes,float(P["primary_test"]["material_slope_margin"])
        )
        summary={
            "experiment":P["experiment"],"protocol_sha256":PROTOCOL_SHA,
            "parent_phase4":P["parent_phase4"],"line_width_document_counts":got,
            "arm_summaries":arm_sums,
            "primary_short_long_contrasts":contrasts,
            "primary_interaction_slopes_per_log2_width":slopes,
            "replication_guard_pass":repok,
            "adjudication":verdict,
            "secondary":{
                "discrete_tau_minima":discrete_tau_minima(cells,P),
                "representation_specific_d3_curves":representation_curves(cells,arms),
                "q3_component_descriptives":q3_component_descriptives(cells,arms),
                "e1_short_long_contrasts":e1_contrasts,
                "e1_interaction_slopes_per_log2_width":e1_slopes,
                "gates":gate_hits(cells)
            }
        }
        json.dump(summary,open("results/SUMMARY.json","w"),indent=2)
        json.dump({"cells":cells},open("results/COMBINED.json","w"),ensure_ascii=False)
        print(json.dumps(summary,indent=2))
    else:
        raise SystemExit(mode)

if __name__=="__main__":
    main()
