#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import OrderedDict,defaultdict
import glob,hashlib,json,lzma,math,pickle,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA="a0d4668c13d28e3c13c602696d14d1df53ddcda49b5fe87a10323ca972ab8577"
WIDTH=10
REPS=("ATOMIC","LITERAL")

def verify_protocol():
    b=(ROOT/"protocol.json").read_bytes(); got=hashlib.sha256(b).hexdigest()
    exp=(ROOT/"PROTOCOL_SHA256").read_text().strip()
    assert got==PROTOCOL_SHA==exp,(got,PROTOCOL_SHA,exp)
    P=json.loads(b.decode("utf-8"))
    panel=json.load(open(ROOT/"SEALED_VOYNICH_CONSEQUENCE_PANEL.json"))
    ref=json.load(open(ROOT/"FINAL_PIPELINE_REFERENCE.json")) if (ROOT/"FINAL_PIPELINE_REFERENCE.json").exists() else None
    return P,panel,ref

def panel_metrics(panel):
    out=[]
    for fam in panel["families"].values(): out.extend(fam)
    assert len(out)==24 and len(set(out))==24
    return tuple(out)

def seed_of(*parts):
    # Exact helper is imported from the frozen scorer at runtime and cross-checked once.
    s="|".join(map(str,parts))
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16],16) & 0x7fffffff

def lineate(words):
    x=words[:2000]; lines=[x[i:i+WIDTH] for i in range(0,2000,WIDTH)]
    assert len(lines)==200 and all(len(z)==WIDTH for z in lines)
    return lines

def boundary_diag(lines,folios=None):
    rep_cross=[]; rep_within=[]; len_cross=[]; len_within=[]
    for i in range(len(lines)-1):
        if folios is not None and folios[i]!=folios[i+1]: continue
        a,b=lines[i],lines[i+1]
        if len(a)<2 or len(b)<1: continue
        rep_cross.append(float(a[-1]==b[0])); rep_within.append(float(a[-2]==a[-1]))
        len_cross.append(abs(len(a[-1])-len(b[0]))); len_within.append(abs(len(a[-2])-len(a[-1])))
    return {
      "repeat_cross_minus_within":float(np.mean(rep_cross)-np.mean(rep_within)) if rep_cross else None,
      "wordlen_absdiff_cross_minus_within":float(np.mean(len_cross)-np.mean(len_within)) if len_cross else None,
      "n_boundaries":len(rep_cross)
    }

def build_vms(P,panel):
    import score_85_metrics as m85
    metrics=panel_metrics(panel)
    with open(ROOT/"inputs/enriched_records.pkl","rb") as f: records=pickle.load(f)
    tokens=[r["token"] for r in records]
    groups=OrderedDict()
    for r in records:
        k=(r["folio"],r["line_no"])
        groups.setdefault(k,[]).append(r)
    lines=[]; folios=[]
    for (folio,line_no),rows in groups.items():
        rows=sorted(rows,key=lambda r:r.get("pos",0))
        line=[r["token"] for r in rows]
        if line:
            lines.append(line); folios.append(folio)
    flat=[t for line in lines for t in line]
    assert flat==tokens,"record/line reconstruction changed token order"
    got=m85.compute_metrics(tokens,lines=lines,
        subset_iterations=P["simulation"]["vms_metric_subset_iterations"],
        subset_words=P["simulation"]["metric_subset_words"],seed=P["simulation"]["vms_metric_seed"],
        do_levenshtein=P["simulation"]["vms_do_levenshtein"],
        ngram_max_len=P["simulation"]["ngram_max_len"],verbose=False)
    out={
      "protocol_sha256":PROTOCOL_SHA,
      "n_tokens":len(tokens),"n_lines":len(lines),
      "metrics":{k:float(got[k]) for k in metrics},
      "tolerances":{k:float(m85.TOLERANCES[k]) for k in metrics},
      "boundary":boundary_diag(lines,folios),
      "metric_code_git_blob_sha1":P["inputs"]["metric_code_git_blob_sha1"],
      "vms_records_git_blob_sha1":P["inputs"]["vms_records_git_blob_sha1"]
    }
    json.dump(out,open(ROOT/"results/VMS_BASELINE.json","w"),indent=2)
    print(json.dumps(out,indent=2),flush=True)

def eligible_docs(rem):
    d=sorted(k for k,w in rem.items() if len(w)>=2000); assert len(d)==190,len(d); return d

def pipeline_specs(ref):
    assert ref and ref["phase8_replication_guard_pass"] is True
    specs=[{"name":"FINAL_PIPELINE","reset":ref["canonical"]["reset"],"order":ref["canonical"]["order"],"confirmatory":True}]
    for x in ref.get("sensitivity_pipelines",[]):
        specs.append({"name":x["name"],"reset":x["reset"],"order":x["order"],"confirmatory":False})
    return specs

def state_arm(reset):
    if reset=="continuous": return "TAU3_FIXED_CONTINUOUS_K2"
    if reset=="line_reset": return "TAU3_FIXED_LINE_RESET_K2"
    raise ValueError(reset)

def transform(plain,plan,repn,states,order,encode_with_plan,apply_k2_states):
    cipher=encode_with_plan(plain,plan,repn)
    if order=="POST": return cipher,apply_k2_states(cipher,states)
    if order=="PRE":
        rot=apply_k2_states(plain,states)
        return cipher,encode_with_plan(rot,plan,repn)
    raise ValueError(order)

def worker(args):
    doc,words,P,panel,ref=args
    import score_85_metrics as m85
    from mechanisms import make_plan,encode_with_plan
    from persistence_operators import apply_k2_states
    from reset_operators import states_for_phase7_arm
    import scorer
    metrics=panel_metrics(panel); plain=lineate(words); specs=pipeline_specs(ref)
    arms=["CIPHER_ONLY"]+[x["name"] for x in specs]
    data={(a,r):[] for a in arms for r in REPS}
    bounds={(a,r):[] for a in arms for r in REPS}
    for rep in P["simulation"]["production_replicates_per_document"]:
        # Preserve exact Phase-5/6/7 seeds; scorer is used only for seed_of, never Q scoring.
        plan=make_plan(plain,"SWITCH_LINE",scorer.seed_of("P5-plan",doc,rep))
        state_seed=scorer.seed_of("P5-state",doc,rep)
        for repn in REPS:
            metric_seed=42000+int(rep)
            # Cipher-only is common to every pipeline.
            cipher=encode_with_plan(plain,plan,repn)
            cm=m85.compute_metrics([t for z in cipher for t in z],lines=cipher,
                subset_iterations=P["simulation"]["metric_subset_iterations"],
                subset_words=P["simulation"]["metric_subset_words"],seed=metric_seed,
                do_levenshtein=False,ngram_max_len=P["simulation"]["ngram_max_len"],verbose=False)
            data[("CIPHER_ONLY",repn)].append({k:float(cm[k]) for k in metrics})
            bounds[("CIPHER_ONLY",repn)].append(boundary_diag(cipher))
            for spec in specs:
                states=states_for_phase7_arm(state_arm(spec["reset"]),2000,WIDTH,state_seed)
                _,out=transform(plain,plan,repn,states,spec["order"],encode_with_plan,apply_k2_states)
                mm=m85.compute_metrics([t for z in out for t in z],lines=out,
                    subset_iterations=P["simulation"]["metric_subset_iterations"],
                    subset_words=P["simulation"]["metric_subset_words"],seed=metric_seed,
                    do_levenshtein=False,ngram_max_len=P["simulation"]["ngram_max_len"],verbose=False)
                data[(spec["name"],repn)].append({k:float(mm[k]) for k in metrics})
                bounds[(spec["name"],repn)].append(boundary_diag(out))
    cells=[]
    for arm in arms:
      for repn in REPS:
        cells.append({"corpus":doc,"arm":arm,"representation":repn,
                      "metrics_by_replicate":data[(arm,repn)],"boundary_by_replicate":bounds[(arm,repn)]})
    return doc,cells

def run_docs(rem,docs,P,panel,ref):
    tasks=[(d,rem[d],P,panel,ref) for d in docs]; cells=[]
    with ProcessPoolExecutor(max_workers=max(1,min(2,len(tasks)))) as ex:
      fs=[ex.submit(worker,t) for t in tasks]
      for i,f in enumerate(as_completed(fs),1):
        d,c=f.result();cells.extend(c);print("SCORE",i,len(tasks),d,flush=True)
    cells.sort(key=lambda x:(x["corpus"],x["arm"],x["representation"]));return cells

def aggregate_cell(c,metrics):
    out={}
    for k in metrics: out[k]=float(np.median([z[k] for z in c["metrics_by_replicate"]]))
    bd={}
    for k in ("repeat_cross_minus_within","wordlen_absdiff_cross_minus_within"):
        vals=[z[k] for z in c["boundary_by_replicate"] if z[k] is not None]
        bd[k]=float(np.median(vals)) if vals else None
    return out,bd

def robust_losses(cells,vms,panel):
    metrics=panel_metrics(panel); by={}
    for c in cells:
        m,b=aggregate_cell(c,metrics);by.setdefault((c["corpus"],c["arm"]),{})[c["representation"]]=(m,b)
    losses={}; boundaries={}; permetric={}
    for key,z in by.items():
        if not all(r in z for r in REPS):continue
        lm={}
        for k in metrics:
            vals=[abs(z[r][0][k]-vms["metrics"][k])/vms["tolerances"][k] for r in REPS]
            lm[k]=float(max(vals))
        losses[key]=float(np.median(list(lm.values())));permetric[key]=lm
        boundaries[key]={}
        for k in ("repeat_cross_minus_within","wordlen_absdiff_cross_minus_within"):
            vals=[z[r][1][k] for r in REPS if z[r][1][k] is not None]
            boundaries[key][k]=float(max(vals,key=lambda x:abs(x-vms["boundary"][k]))) if vals else None
    return losses,permetric,boundaries

def bootstrap_median(vals,nres,seed):
    x=np.asarray(vals,float);n=len(x)
    rng=np.random.default_rng(seed);out=np.empty(nres,float)
    for i in range(0,nres,500):
        m=min(500,nres-i);idx=rng.integers(0,n,size=(m,n));out[i:i+m]=np.median(x[idx],axis=1)
    return {"n":n,"median":float(np.median(x)),"ci95":[float(np.quantile(out,.025)),float(np.quantile(out,.975))]}

def adjudicate(losses,permetric,panel,ref):
    docs=sorted({d for d,a in losses if a=="CIPHER_ONLY" and (d,"FINAL_PIPELINE") in losses})
    gains=[losses[(d,"CIPHER_ONLY")]-losses[(d,"FINAL_PIPELINE")] for d in docs]
    primary=bootstrap_median(gains,10000,918273)
    family={}
    for fam,ms in panel["families"].items():
        vals=[]
        for d in docs:
            vals.append(float(np.median([permetric[(d,"CIPHER_ONLY")][m]-permetric[(d,"FINAL_PIPELINE")][m] for m in ms])))
        family[fam]={"n":len(vals),"median_gain":float(np.median(vals))}
    pos=sum(x["median_gain"]>0 for x in family.values())
    catastrophic=any(x["median_gain"] < -0.5 for x in family.values())
    if primary["ci95"][0]>0 and pos>=5 and not catastrophic: verdict="CONFIRMATORY_BROAD_IMPROVEMENT"
    elif primary["ci95"][1]<=0: verdict="CONFIRMATORY_NO_GAIN"
    else: verdict="CONFIRMATORY_NARROW_OR_MIXED"
    causal=bool(verdict=="CONFIRMATORY_BROAD_IMPROVEMENT" and ref["canonical"]["reset_resolved"] and ref["canonical"]["order_resolved"])
    return verdict,primary,family,causal

def sensitivity_summaries(losses,ref):
    out={}
    docs=sorted({d for d,a in losses if a=="CIPHER_ONLY"})
    for spec in ref.get("sensitivity_pipelines",[]):
        arm=spec["name"]; ds=[d for d in docs if (d,arm) in losses]
        if ds:
            out[arm]=bootstrap_median([losses[(d,"CIPHER_ONLY")]-losses[(d,arm)] for d in ds],10000,seed_of("P9-sens",arm))
    return out

def boundary_summary(boundaries,vms,arms):
    out={}
    for arm in arms:
      sub=[z for (d,a),z in boundaries.items() if a==arm]
      out[arm]={}
      for k in ("repeat_cross_minus_within","wordlen_absdiff_cross_minus_within"):
        vals=[z[k] for z in sub if z[k] is not None]
        out[arm][k]={"median":float(np.median(vals)) if vals else None,
                     "vms":vms["boundary"][k],
                     "median_abs_error":float(np.median([abs(x-vms["boundary"][k]) for x in vals])) if vals else None}
    return out

def load_rem():
    return json.loads(lzma.decompress((ROOT/"inputs/rem_docs.json.xz").read_bytes()).decode("utf-8"))

def gather(pattern):
    cells=[]
    for f in sorted(glob.glob(pattern,recursive=True)):
        z=json.load(open(f));assert z["protocol_sha256"]==PROTOCOL_SHA;cells.extend(z["cells"])
    return cells

def main():
    P,panel,ref=verify_protocol();mode=sys.argv[1]
    if mode=="vms":
        (ROOT/"results").mkdir(exist_ok=True);build_vms(P,panel)
    elif mode=="score":
        assert ref is not None and ref["phase8_replication_guard_pass"] is True
        shard=int(sys.argv[2]);nshards=int(sys.argv[3]);rem=load_rem();docs=eligible_docs(rem)[shard::nshards]
        cells=run_docs(rem,docs,P,panel,ref)
        json.dump({"protocol_sha256":PROTOCOL_SHA,"docs":docs,"cells":cells},open(ROOT/f"results/SCORE_SHARD_{shard}_OF_{nshards}.json","w"))
    elif mode=="merge":
        assert ref is not None
        vms=json.load(open(ROOT/"inputs/VMS_BASELINE.json"));cells=gather(sys.argv[2]+"/**/SCORE_SHARD_*.json")
        assert len({c["corpus"] for c in cells})==190
        losses,pm,bounds=robust_losses(cells,vms,panel)
        verdict,primary,family,causal=adjudicate(losses,pm,panel,ref)
        arms=["CIPHER_ONLY","FINAL_PIPELINE"]+[x["name"] for x in ref.get("sensitivity_pipelines",[])]
        summary={"experiment":P["experiment"],"protocol_sha256":PROTOCOL_SHA,"phase8_reference":ref,
                 "vms_baseline":vms,"adjudication":verdict,"primary_gain":primary,"family_gains":family,
                 "causal_architecture_claim_supported":causal,"sensitivity":sensitivity_summaries(losses,ref),
                 "boundary_holdout":boundary_summary(bounds,vms,arms)}
        json.dump(summary,open(ROOT/"results/SUMMARY.json","w"),indent=2)
        print(json.dumps(summary,indent=2),flush=True)
    else:raise SystemExit("usage: run_phase9_remote.py vms | score SHARD NSHARDS | merge ROOT")
if __name__=="__main__":main()
