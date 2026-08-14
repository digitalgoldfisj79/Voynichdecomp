#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob, hashlib, json, lzma, math, sys
import numpy as np
import scorer
from mechanisms import make_plan, encode_with_plan, self_tests as mechanism_self_tests
from persistence_operators import apply_k2_states, self_tests as persistence_self_tests
from dwell_operators import states_for_phase6_arm, self_tests as dwell_self_tests
from design_utils import adjudicate as design_adjudicate, self_tests as design_self_tests

ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA="0b1928efa25cc54e82e0f4edea08d5cd42777b123e00899c9666acb768ae5d3b"
SCHEDULE="SWITCH_LINE"
WIDTH=10
REPS=("ATOMIC","LITERAL")
TARGET=np.asarray(scorer.Q_VMS,float)
TARGET_E1=float(TARGET[3])

def verify_protocol():
    p=(ROOT/"protocol.json").read_bytes()
    got=hashlib.sha256(p).hexdigest()
    expected=(ROOT/"PROTOCOL_SHA256").read_text().strip()
    assert got==PROTOCOL_SHA==expected,(got,PROTOCOL_SHA,expected)
    P=json.loads(p.decode("utf-8"))
    assert P["invariants"]["line_width_tokens"]==WIDTH
    return P

def arms_from_protocol(P):
    arms=["IDENTITY","OCCURRENCE_K2"]
    for tau in ("tau3","tau4"):
        for law in ("fixed","semi","geometric"):
            arms.append(P["arms"][tau][law]["arm"])
    assert len(arms)==8 and len(set(arms))==8
    return tuple(arms)

def eligible_docs(rem):
    d=sorted(k for k,w in rem.items() if len(w)>=2000)
    assert len(d)==190,len(d)
    return d

def lineate(words):
    w=words[:2000]
    out=[w[i:i+WIDTH] for i in range(0,len(w),WIDTH)]
    assert len(out)==200 and sum(map(len,out))==2000 and all(len(x)==10 for x in out)
    return out

def compact(r):
    return {k:r[k] for k in ("Q3","Q4","distance3","distance4","gate3","gate4")}

def worker(args):
    doc,words,arms,nreps,nperm=args
    plain=lineate(words)
    ntok=2000
    e1avail=scorer.adj_repeats(plain)>0
    buckets={(a,r):[] for a in arms for r in REPS}
    for rep in range(nreps):
        plan=make_plan(plain,SCHEDULE,scorer.seed_of("P5-plan",doc,rep))
        state_seed=scorer.seed_of("P5-state",doc,rep)
        states={a:states_for_phase6_arm(a,ntok,state_seed) for a in arms}
        for repn in REPS:
            base=encode_with_plan(plain,plan,repn)
            stat_seed=scorer.seed_of("P5-stat",doc,repn,rep)
            for arm in arms:
                out=base if arm=="IDENTITY" else apply_k2_states(base,states[arm])
                r=scorer.one_eval(scorer.prep(out),stat_seed,e1avail,nperm)
                buckets[(arm,repn)].append(r)
    cells=[]
    for arm in arms:
        for repn in REPS:
            rows=buckets[(arm,repn)]
            ag=scorer.aggregate(rows)
            cells.append({
                "corpus":doc,"line_width":WIDTH,"arm":arm,"representation":repn,
                **ag,"rows":[compact(r) for r in rows]
            })
    return doc,cells

def run_docs(rem,docs,arms,nreps,nperm):
    tasks=[(d,rem[d],arms,nreps,nperm) for d in docs]
    cells=[]
    workers=min(2,len(tasks))
    with ProcessPoolExecutor(max_workers=max(1,workers)) as ex:
        fs=[ex.submit(worker,t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):
            d,c=f.result(); cells.extend(c)
            print("SCORE",i,len(tasks),d,flush=True)
    cells.sort(key=lambda x:(x["corpus"],x["arm"],x["representation"]))
    return cells

def robust_map(cells,field="d3"):
    by={}
    for c in cells:
        key=(c["corpus"],c["arm"])
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

def endpoint_summary(d,arm):
    docs=sorted({doc for (doc,a) in d if a in ("IDENTITY","OCCURRENCE_K2",arm)})
    common=[doc for doc in docs if all((doc,a) in d for a in ("IDENTITY","OCCURRENCE_K2",arm))]
    vals=np.array([d[(doc,arm)] for doc in common],float)
    I=np.array([d[(doc,"IDENTITY")] for doc in common],float)
    return {
        "arm":arm,"n_common":len(common),"median_robust_d3":float(np.median(vals)),
        "median_identity_d3":float(np.median(I)),
        "wins_vs_identity":int(np.sum(vals<I)),
        "win_fraction_vs_identity":float(np.mean(vals<I))
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

def tau_arms(P,tau):
    z=P["arms"][f"tau{tau}"]
    return z["fixed"]["arm"],z["semi"]["arm"],z["geometric"]["arm"]

def pairwise_stats(d,P,tau,field):
    fixed,semi,geom=tau_arms(P,tau)
    docs=sorted({doc for (doc,a) in d if a in (fixed,semi,geom)})
    docs=[doc for doc in docs if all((doc,a) in d for a in (fixed,semi,geom))]
    F=np.array([d[(doc,fixed)] for doc in docs],float)
    S=np.array([d[(doc,semi)] for doc in docs],float)
    G=np.array([d[(doc,geom)] for doc in docs],float)
    nres=P["primary_test"]["bootstrap"]["resamples"]
    return {
        "FG":bootstrap_median(G-F,nres,scorer.seed_of("P6-bootstrap",field,tau,"FG")),
        "FS":bootstrap_median(S-F,nres,scorer.seed_of("P6-bootstrap",field,tau,"FS")),
        "SG":bootstrap_median(G-S,nres,scorer.seed_of("P6-bootstrap",field,tau,"SG"))
    }

def within_law_tau_contrasts(d,P,field):
    out={}
    nres=P["primary_test"]["bootstrap"]["resamples"]
    for law in ("fixed","semi","geometric"):
        a3=P["arms"]["tau3"][law]["arm"]; a4=P["arms"]["tau4"][law]["arm"]
        docs=sorted({doc for (doc,a) in d if a in (a3,a4)})
        docs=[doc for doc in docs if (doc,a3) in d and (doc,a4) in d]
        vals=np.array([d[(doc,a4)]-d[(doc,a3)] for doc in docs],float)
        out[law]=bootstrap_median(vals,nres,scorer.seed_of("P6-bootstrap",field,law,"tau4-minus-tau3"))
    return out

def replication_guard(d,P):
    expected=P["replication_guard"]["expected_phase5_w10_median_robust_d3"]
    tol=float(P["replication_guard"]["tolerance_abs"])
    summaries={}
    ok=True
    for arm,exp in expected.items():
        s=endpoint_summary(d,arm); summaries[arm]=s
        if abs(s["median_robust_d3"]-float(exp))>tol:
            ok=False
    return ok,summaries

def representation_d3(cells,arms):
    out={}
    for arm in arms:
        out[arm]={}
        for repn in REPS:
            vals=[c["distance_of_median_Q3"] for c in cells if c["arm"]==arm and c["representation"]==repn
                  and c["distance_of_median_Q3"] is not None]
            out[arm][repn]=float(np.median(vals)) if vals else None
    return out

def q3_component_descriptives(cells,arms):
    out={}
    for arm in arms:
        out[arm]={}
        for repn in REPS:
            sub=[c for c in cells if c["arm"]==arm and c["representation"]==repn]
            q={}
            for j in range(3):
                vals=[c["median_Q3"][j] for c in sub if c["median_Q3"][j] is not None and c["median_Q3"][j]>0]
                lr=np.log(np.asarray(vals,float)/TARGET[j]) if vals else np.array([])
                q[f"Q{j+1}_median_abs_log_error"]=float(np.median(np.abs(lr))) if len(lr) else None
                q[f"Q{j+1}_median_signed_log_bias"]=float(np.median(lr)) if len(lr) else None
            out[arm][repn]=q
    return out

def gate_hits(cells):
    g3=g4=0; hits=[]
    for c in cells:
        for i,r in enumerate(c["rows"]):
            g3+=int(r["gate3"]); g4+=int(r["gate4"])
            if r["gate4"]:
                hits.append({
                    "corpus":c["corpus"],"arm":c["arm"],"representation":c["representation"],
                    "replicate":i,"Q4":r["Q4"],"distance4":r["distance4"]
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
    P=verify_protocol(); mechanism_self_tests(); persistence_self_tests(); dwell_self_tests(); design_self_tests(); scorer.self_test_ed1()
    mode=sys.argv[1]; arms=arms_from_protocol(P)
    if mode=="score":
        shard=int(sys.argv[2]); nshards=int(sys.argv[3])
        rem=load_rem(); docs=eligible_docs(rem)[shard::nshards]
        cells=run_docs(rem,docs,arms,P["simulation"]["replicates_per_document"],P["simulation"]["permutation_replicates"])
        json.dump({"protocol_sha256":PROTOCOL_SHA,"docs":docs,"arms":arms,"cells":cells},
                  open(f"results/SCORE_SHARD_{shard}_OF_{nshards}.json","w"),ensure_ascii=False)
    elif mode=="merge":
        root=sys.argv[2]
        cells=gather(root+"/**/SCORE_SHARD_*.json","cells")
        docs={c["corpus"] for c in cells}
        assert len(docs)==190,len(docs)
        d3=robust_map(cells,"d3"); e1=robust_map(cells,"e1err")
        repok,endpoint_summaries=replication_guard(d3,P)
        primary={str(t):pairwise_stats(d3,P,t,"d3") for t in (3,4)}
        verdict=design_adjudicate(repok,primary,float(P["primary_test"]["materiality_margin_d3"]))
        secondary={
            "E1_pairwise":{str(t):pairwise_stats(e1,P,t,"e1err") for t in (3,4)},
            "tau4_minus_tau3_d3":within_law_tau_contrasts(d3,P,"d3"),
            "tau4_minus_tau3_e1":within_law_tau_contrasts(e1,P,"e1err"),
            "representation_specific_d3":representation_d3(cells,arms),
            "Q3_component_descriptives":q3_component_descriptives(cells,arms),
            "gates":gate_hits(cells)
        }
        summary={
            "experiment":P["experiment"],"protocol_sha256":PROTOCOL_SHA,
            "document_count":len(docs),"replication_guard_pass":repok,
            "endpoint_summaries":endpoint_summaries,
            "primary_pairwise_d3":primary,
            "adjudication":verdict,
            "secondary":secondary
        }
        json.dump(summary,open("results/SUMMARY.json","w"),indent=2)
        json.dump({"protocol_sha256":PROTOCOL_SHA,"cells":cells},open("results/COMBINED.json","w"))
        print(json.dumps(summary,indent=2),flush=True)
    else:
        raise SystemExit("usage: run_phase6_remote.py score SHARD NSHARDS | merge ROOT")

if __name__=="__main__":
    main()
