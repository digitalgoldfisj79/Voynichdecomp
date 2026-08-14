#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob,hashlib,json,lzma,math,sys
import numpy as np
import scorer
from mechanisms import make_plan,encode_with_plan,self_tests as mechanism_self_tests
from persistence_operators import apply_k2_states,self_tests as persistence_self_tests
from reset_operators import states_for_phase7_arm,self_tests as reset_self_tests
from design_utils import adjudicate as design_adjudicate,self_tests as design_self_tests

ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA="36a9656f7dfa647a9c215e5dd1681e316c51d16064d9d9c0efb07cafb70b79f4"
SCHEDULE="SWITCH_LINE"; WIDTH=10; REPS=("ATOMIC","LITERAL")
TARGET=np.asarray(scorer.Q_VMS,float); TARGET_E1=float(TARGET[3])

def verify_protocol():
    b=(ROOT/"protocol.json").read_bytes(); got=hashlib.sha256(b).hexdigest()
    exp=(ROOT/"PROTOCOL_SHA256").read_text().strip()
    assert got==PROTOCOL_SHA==exp,(got,PROTOCOL_SHA,exp)
    return json.loads(b.decode("utf-8"))

def strata(P):
    return P["strata"]

def labels_for(key):
    u=key.upper()
    return f"{u}__POST",f"{u}__PRE",f"{u}__ORIGIN_ONLY"

def all_arms(P):
    out=["CIPHER_ONLY"]
    for k in strata(P):
        out.extend(labels_for(k))
    return tuple(out)

def eligible_docs(rem):
    d=sorted(k for k,w in rem.items() if len(w)>=2000); assert len(d)==190,len(d); return d

def lineate(words):
    x=words[:2000]; lines=[x[i:i+WIDTH] for i in range(0,2000,WIDTH)]
    assert len(lines)==200 and all(len(z)==WIDTH for z in lines); return lines

def compact(r):
    return {k:r[k] for k in ("Q3","Q4","distance3","distance4","gate3","gate4")}

def worker(args):
    doc,words,P,nreps,nperm=args
    plain=lineate(words); e1avail=scorer.adj_repeats(plain)>0; arms=all_arms(P)
    buckets={(a,r):[] for a in arms for r in REPS}
    for rep in range(nreps):
        plan=make_plan(plain,SCHEDULE,scorer.seed_of("P5-plan",doc,rep))
        state_seed=scorer.seed_of("P5-state",doc,rep)
        state_by={k:states_for_phase7_arm(z["state_law"],2000,WIDTH,state_seed) for k,z in strata(P).items()}
        for repn in REPS:
            cipher=encode_with_plan(plain,plan,repn)
            stat_seed=scorer.seed_of("P5-stat",doc,repn,rep)
            r=scorer.one_eval(scorer.prep(cipher),stat_seed,e1avail,nperm)
            buckets[("CIPHER_ONLY",repn)].append(r)
            for key,states in state_by.items():
                post_lab,pre_lab,origin_lab=labels_for(key)
                post=apply_k2_states(cipher,states)
                rotated_plain=apply_k2_states(plain,states)
                pre=encode_with_plan(rotated_plain,plan,repn)
                origin=rotated_plain
                for lab,out in ((post_lab,post),(pre_lab,pre),(origin_lab,origin)):
                    rr=scorer.one_eval(scorer.prep(out),stat_seed,e1avail,nperm)
                    buckets[(lab,repn)].append(rr)
    cells=[]
    for arm in arms:
      for repn in REPS:
        rows=buckets[(arm,repn)]; ag=scorer.aggregate(rows)
        cells.append({"corpus":doc,"arm":arm,"representation":repn,**ag,"rows":[compact(r) for r in rows]})
    return doc,cells

def run_docs(rem,docs,P,nreps,nperm):
    cells=[]; tasks=[(d,rem[d],P,nreps,nperm) for d in docs]
    with ProcessPoolExecutor(max_workers=max(1,min(2,len(tasks)))) as ex:
      fs=[ex.submit(worker,t) for t in tasks]
      for i,f in enumerate(as_completed(fs),1):
        d,c=f.result(); cells.extend(c); print("SCORE",i,len(tasks),d,flush=True)
    cells.sort(key=lambda x:(x["corpus"],x["arm"],x["representation"])); return cells

def robust_map(cells,field="d3"):
    by={}
    for c in cells: by.setdefault((c["corpus"],c["arm"]),{})[c["representation"]]=c
    out={}
    for key,z in by.items():
      if not all(r in z for r in REPS): continue
      if field=="d3":
        v=[z[r]["distance_of_median_Q3"] for r in REPS]
        if all(x is not None and math.isfinite(x) for x in v): out[key]=max(v)
      elif field=="e1err":
        v=[]
        for r in REPS:
          e=z[r]["median_Q4"][3]
          if e is None or e<=0 or not math.isfinite(e): break
          v.append(abs(math.log(e/TARGET_E1)))
        if len(v)==2: out[key]=max(v)
    return out

def bootstrap_median(values,nresamp,seed):
    x=np.asarray(values,float); n=len(x)
    if not n:return {"n":0,"median":None,"ci95":[None,None]}
    rng=np.random.default_rng(seed); meds=np.empty(nresamp,float)
    for i in range(0,nresamp,500):
      m=min(500,nresamp-i); idx=rng.integers(0,n,size=(m,n)); meds[i:i+m]=np.median(x[idx],axis=1)
    return {"n":n,"median":float(np.median(x)),"ci95":[float(np.quantile(meds,.025)),float(np.quantile(meds,.975))]}

def paired(d,a,b,P,field,label):
    docs=sorted(doc for doc in {x[0] for x in d} if (doc,a) in d and (doc,b) in d)
    vals=[d[(doc,a)]-d[(doc,b)] for doc in docs]
    return bootstrap_median(vals,P["primary_test"]["bootstrap"]["resamples"],scorer.seed_of("P8-bootstrap",field,label))

def order_stats(d,P,field):
    out={}
    for key in strata(P):
      post,pre,_=labels_for(key)
      out[key]=paired(d,pre,post,P,field,key)
    return out

def endpoint_median(d,arm):
    v=[x for (doc,a),x in d.items() if a==arm]; return float(np.median(v)) if v else None

def replication_guard(d,P):
    ref=json.load(open(ROOT/"PHASE7_REFERENCE.json")); tol=float(P["replication_guard"]["tolerance_abs"])
    detail={}; ok=bool(ref["replication_guard_pass"])
    for key in strata(P):
      post,_,_=labels_for(key); p7arm=ref["strata"][key]["phase7_arm"]
      exp=float(ref["strata"][key]["median_robust_d3"]); got=endpoint_median(d,post)
      detail[key]={"phase7_arm":p7arm,"expected":exp,"observed":got,"abs_diff":abs(got-exp)}; ok &= abs(got-exp)<=tol
    return bool(ok),detail,ref

def baseline_gains(d,P,field):
    out={}
    for key in strata(P):
      post,pre,_=labels_for(key)
      out[key]={"POST_gain":paired(d,"CIPHER_ONLY",post,P,field,key+"|POST_GAIN"),
                 "PRE_gain":paired(d,"CIPHER_ONLY",pre,P,field,key+"|PRE_GAIN")}
    return out

def representation_d3(cells,arms):
    out={}
    for arm in arms:
      out[arm]={}
      for repn in REPS:
        v=[c["distance_of_median_Q3"] for c in cells if c["arm"]==arm and c["representation"]==repn and c["distance_of_median_Q3"] is not None]
        out[arm][repn]=float(np.median(v)) if v else None
    return out

def gate_hits(cells):
    g3=g4=0; hits=[]
    for c in cells:
      for i,r in enumerate(c["rows"]):
        g3+=int(r["gate3"]);g4+=int(r["gate4"])
        if r["gate4"]:hits.append({"corpus":c["corpus"],"arm":c["arm"],"representation":c["representation"],"replicate":i,"Q4":r["Q4"],"distance4":r["distance4"]})
    return {"individual_gate3_passes":g3,"individual_gate4_passes":g4,"gate4_hits":hits}

def load_rem():
    return json.loads(lzma.decompress((ROOT/"inputs/rem_docs.json.xz").read_bytes()).decode("utf-8"))

def gather(pattern):
    cells=[]
    for f in sorted(glob.glob(pattern,recursive=True)):
      z=json.load(open(f)); assert z["protocol_sha256"]==PROTOCOL_SHA; cells.extend(z["cells"])
    return cells

def main():
    P=verify_protocol(); mechanism_self_tests();persistence_self_tests();reset_self_tests();design_self_tests();scorer.self_test_ed1()
    mode=sys.argv[1]
    if mode=="score":
      shard=int(sys.argv[2]);nshards=int(sys.argv[3]);rem=load_rem();docs=eligible_docs(rem)[shard::nshards]
      cells=run_docs(rem,docs,P,P["invariants"]["replicates_per_document"],P["invariants"]["permutation_replicates"])
      json.dump({"protocol_sha256":PROTOCOL_SHA,"docs":docs,"cells":cells},open(f"results/SCORE_SHARD_{shard}_OF_{nshards}.json","w"),ensure_ascii=False)
    elif mode=="merge":
      cells=gather(sys.argv[2]+"/**/SCORE_SHARD_*.json");docs={c["corpus"] for c in cells};assert len(docs)==190,len(docs)
      d3=robust_map(cells,"d3");e1=robust_map(cells,"e1err")
      repok,repdetail,ref=replication_guard(d3,P);primary=order_stats(d3,P,"d3")
      verdict=design_adjudicate(repok,primary,float(P["primary_test"]["materiality_margin_d3"]))
      summary={"experiment":P["experiment"],"protocol_sha256":PROTOCOL_SHA,"document_count":len(docs),
               "replication_guard_pass":repok,"replication_guard":repdetail,
               "phase7_reference":ref,"primary_order_d3":primary,"adjudication":verdict,
               "secondary":{"E1_order":order_stats(e1,P,"e1err"),
                             "cipher_only_gains_d3":baseline_gains(d3,P,"d3"),
                             "representation_specific_d3":representation_d3(cells,all_arms(P)),
                             "gates":gate_hits(cells)}}
      json.dump(summary,open("results/SUMMARY.json","w"),indent=2)
      json.dump({"protocol_sha256":PROTOCOL_SHA,"cells":cells},open("results/COMBINED.json","w"))
      print(json.dumps(summary,indent=2),flush=True)
    else:raise SystemExit("usage: run_phase8_remote.py score SHARD NSHARDS | merge ROOT")
if __name__=="__main__":main()
