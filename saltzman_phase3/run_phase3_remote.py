#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,lzma,math,os,sys,glob
import numpy as np
import scorer
from mechanisms import make_plan,encode_with_plan,self_tests as mechanism_self_tests
from origin_operators import ARCHITECTURES,ENTROPIES,transform_origin,full_shuffle,self_tests as origin_self_tests
ROOT=Path(__file__).resolve().parent
SCHEDULE="SWITCH_LINE";REPS=("ATOMIC","LITERAL");TARGET_E1=float(scorer.Q_VMS[3]);K_ORDER={2:0,3:1,4:2,"ALL":3}
def arm_name(a,k):return f"{a}_K{k}"
CANDIDATES=tuple(arm_name(a,k) for k in ENTROPIES for a in ARCHITECTURES);ALL_ARMS=("IDENTITY",)+CANDIDATES+("FULL_SHUFFLE",)
def split_docs(rem):
    elig=sorted(d for d,w in rem.items() if len(w)>=2000);assert len(elig)==190
    rank=sorted(elig,key=lambda d:hashlib.sha256(("ASC-ORIGIN-STATE-v0.1|SPLIT|"+d).encode()).hexdigest())
    return rank[::2],rank[1::2]
def w10(words):
    z=list(words)[:2000];return [z[i:i+10] for i in range(0,len(z),10) if len(z[i:i+10])>=2]
def compact(r):return {k:r[k] for k in ("Q3","Q4","distance3","distance4","gate3","gate4")}
def worker(args):
    doc,words,arms,nreps,nperm,phase=args;plain=w10(words);e1avail=scorer.adj_repeats(plain)>0
    buckets={(a,r):[] for a in arms for r in REPS};parsed={}
    for arm in arms:
        if arm not in ("IDENTITY","FULL_SHUFFLE"):
            arch,ks=arm.rsplit("_K",1);parsed[arm]=(arch,ks if ks=="ALL" else int(ks))
    for rep in range(nreps):
        plan=make_plan(plain,SCHEDULE,scorer.seed_of("P3-plan",doc,rep))
        for repn in REPS:
            base=encode_with_plan(plain,plan,repn)
            for arm in arms:
                if arm=="IDENTITY":out=base
                elif arm=="FULL_SHUFFLE":out=full_shuffle(base,scorer.seed_of("P3-full",doc,rep))
                else:
                    arch,K=parsed[arm];out=transform_origin(base,plain,arch,K,scorer.seed_of("P3-origin",doc,rep))
                r=scorer.one_eval(scorer.prep(out),scorer.seed_of("P3-stat",phase,doc,arm,repn,rep),e1avail,nperm);buckets[(arm,repn)].append(r)
    cells=[]
    for arm in arms:
        for repn in REPS:
            rows=buckets[(arm,repn)];ag=scorer.aggregate(rows);cells.append({"corpus":doc,"arm":arm,"representation":repn,**ag,"rows":[compact(r) for r in rows]})
    return doc,cells
def run_docs(rem,docs,arms,nreps,nperm,phase):
    cells=[];tasks=[(d,rem[d],tuple(arms),nreps,nperm,phase) for d in docs];workers=min(2,len(tasks))
    with ProcessPoolExecutor(max_workers=max(1,workers)) as ex:
        fs=[ex.submit(worker,t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):d,c=f.result();cells.extend(c);print(phase,i,len(tasks),d,flush=True)
    return sorted(cells,key=lambda x:(x["corpus"],x["arm"],x["representation"]))
def robust_map(cells,field="d3"):
    by={}
    for c in cells:by.setdefault((c["corpus"],c["arm"]),{})[c["representation"]]=c
    out={}
    for key,z in by.items():
        if not all(r in z for r in REPS):continue
        if field=="d3":
            v=[z[r]["distance_of_median_Q3"] for r in REPS]
            if all(x is not None and math.isfinite(x) for x in v):out[key]=max(v)
        else:
            v=[]
            for r in REPS:
                e=z[r]["median_Q4"][3]
                if e is None or e<=0 or not math.isfinite(e):break
                v.append(abs(math.log(e/TARGET_E1)))
            if len(v)==2:out[key]=max(v)
    return out
def summarize_candidate(cells,arm):
    d=robust_map(cells,"d3");e=robust_map(cells,"e1err");docs=sorted({x[0] for x in d});common=[x for x in docs if (x,"IDENTITY") in d and (x,"OCCURRENCE_KALL") in d and (x,arm) in d]
    if not common:return None
    I=float(np.median([d[(x,"IDENTITY")] for x in common]));O=float(np.median([d[(x,"OCCURRENCE_KALL")] for x in common]));C=float(np.median([d[(x,arm)] for x in common]));den=I-O;ret=(I-C)/den if den>0 else None;wins=sum(d[(x,arm)]<d[(x,"IDENTITY")] for x in common);ed=bool(ret is not None and C<I and ret>=.75 and wins/len(common)>=.60)
    ec=[x for x in common if (x,arm) in e and (x,"OCCURRENCE_KALL") in e]
    if ec:eC=float(np.median([e[(x,arm)] for x in ec]));eO=float(np.median([e[(x,"OCCURRENCE_KALL")] for x in ec]));ew=sum(e[(x,arm)]<e[(x,"OCCURRENCE_KALL")] for x in ec);ef=ew/len(ec)
    else:eC=eO=None;ew=0;ef=None
    rec=bool(len(ec)>=30 and eC is not None and eC<eO and ef>=.60)
    return {"arm":arm,"n_common":len(common),"median_identity_d3":I,"median_occurrence_all_d3":O,"median_candidate_d3":C,"retention_vs_occurrence_all":ret,"wins_vs_identity":wins,"win_fraction_vs_identity":wins/len(common),"ed1_success":ed,"n_e1_common":len(ec),"median_candidate_e1_log_error":eC,"median_occurrence_all_e1_log_error":eO,"e1_wins_vs_occurrence_all":ew,"e1_win_fraction":ef,"recurrence_success":rec}
def benchmark_summary(cells):
    d=robust_map(cells,"d3");docs=sorted({x[0] for x in d});common=[x for x in docs if (x,"IDENTITY") in d and (x,"OCCURRENCE_KALL") in d];I=float(np.median([d[(x,"IDENTITY")] for x in common]));O=float(np.median([d[(x,"OCCURRENCE_KALL")] for x in common]));wins=sum(d[(x,"OCCURRENCE_KALL")]<d[(x,"IDENTITY")] for x in common);return {"n_common":len(common),"median_identity_d3":I,"median_occurrence_all_d3":O,"occurrence_wins":wins,"occurrence_win_fraction":wins/len(common),"benchmark_positive":bool(O<I and wins/len(common)>=.60)}
def select(cells):
    sums=[summarize_candidate(cells,a) for a in CANDIDATES];sums=[x for x in sums if x];q=[x for x in sums if x["ed1_success"]]
    if q:
        def kof(a):s=a.rsplit("_K",1)[1];return s if s=="ALL" else int(s)
        mink=min((kof(x["arm"]) for x in q),key=lambda k:K_ORDER[k]);minimal=[x for x in q if kof(x["arm"])==mink]
    else:mink=None;minimal=[]
    pool=q if q else sums
    def rank(x):
        ef=x["e1_win_fraction"] if x["e1_win_fraction"] is not None else -1;ee=x["median_candidate_e1_log_error"] if x["median_candidate_e1_log_error"] is not None else 1e9
        return (not x["recurrence_success"],-ef,ee,-(x["retention_vs_occurrence_all"] if x["retention_vs_occurrence_all"] is not None else -1),x["arm"])
    rec=min(pool,key=rank) if pool else None;adv=[x["arm"] for x in minimal]
    if rec and rec["arm"] not in adv:adv.append(rec["arm"])
    if not q:adv=adv[:3]
    return {"benchmark":benchmark_summary(cells),"candidate_summaries":sums,"n_ed1_qualifiers":len(q),"minimal_entropy":mink,"minimal_entropy_qualifiers":[x["arm"] for x in minimal],"recurrence_candidate":rec["arm"] if rec else None,"advanced":adv}
def gate_hits(cells):
    hits=[];g3=g4=0
    for c in cells:
        for i,r in enumerate(c["rows"]):
            g3+=int(r["gate3"]);g4+=int(r["gate4"])
            if r["gate4"]:hits.append({"corpus":c["corpus"],"arm":c["arm"],"representation":c["representation"],"replicate":i,"Q4":r["Q4"],"distance4":r["distance4"]})
    return {"individual_gate3_passes":g3,"individual_gate4_passes":g4,"gate4_hits":hits}
def load_rem():return json.loads(lzma.decompress((ROOT/"inputs/rem_docs.json.xz").read_bytes()).decode("utf-8"))
def gather(pattern):
    cells=[]
    for f in sorted(glob.glob(pattern,recursive=True)):cells.extend(json.load(open(f))["cells"])
    return sorted(cells,key=lambda x:(x["corpus"],x["arm"],x["representation"]))
def main():
    mechanism_self_tests();origin_self_tests();scorer.self_test_ed1();mode=sys.argv[1]
    if mode=="discovery":
        shard=int(sys.argv[2]);n=int(sys.argv[3]);rem=load_rem();disc,_=split_docs(rem);docs=disc[shard::n];cells=run_docs(rem,docs,ALL_ARMS,10,50,"DISCOVERY");json.dump({"cells":cells,"docs":docs},open(f"results/DISCOVERY_SHARD_{shard}_OF_{n}.json","w"),ensure_ascii=False)
    elif mode=="select":
        cells=gather(sys.argv[2]+"/**/DISCOVERY_SHARD_*.json");assert len({c['corpus'] for c in cells})==95;sel=select(cells);json.dump(sel,open("results/SELECTION.json","w"),indent=2);json.dump({"cells":cells},open("results/DISCOVERY_COMBINED.json","w"),ensure_ascii=False)
    elif mode=="confirmation":
        shard=int(sys.argv[2]);n=int(sys.argv[3]);sel=json.load(open(sys.argv[4]));adv=tuple(dict.fromkeys(("IDENTITY","OCCURRENCE_KALL","FULL_SHUFFLE",*sel["advanced"])));rem=load_rem();_,conf=split_docs(rem);docs=conf[shard::n];cells=run_docs(rem,docs,adv,20,100,"CONFIRMATION");json.dump({"cells":cells,"docs":docs,"arms":adv},open(f"results/CONFIRMATION_SHARD_{shard}_OF_{n}.json","w"),ensure_ascii=False)
    elif mode=="merge":
        cells=gather(sys.argv[2]+"/**/CONFIRMATION_SHARD_*.json");assert len({c['corpus'] for c in cells})==95;sel=json.load(open(sys.argv[3]));adv=tuple(dict.fromkeys(("IDENTITY","OCCURRENCE_KALL","FULL_SHUFFLE",*sel["advanced"])));cs=[summarize_candidate(cells,a) for a in adv if a not in ("IDENTITY","OCCURRENCE_KALL","FULL_SHUFFLE")];cs=[x for x in cs if x];b=benchmark_summary(cells);confirmed=[x for x in cs if x["ed1_success"]];joint=[x for x in confirmed if x["recurrence_success"]];verdict="NO_REPLICATION" if not b["benchmark_positive"] else ("ORIGIN_STATE_WITH_RECURRENCE" if joint else ("ORIGIN_STATE_ED1_ONLY" if confirmed else "OCCURRENCE_RANDOM_REQUIRED"));summary={"experiment":"ASC-ORIGIN-STATE-v0.1","freeze_sha256":"ad6a85bc24114a9e5c19c85ce6c3fee6a4dbfbbd6dbc1cbb0776f3cdda888be6","discovery_selection":sel,"confirmation_benchmark":b,"confirmation_candidates":cs,"confirmed_ed1":[x["arm"] for x in confirmed],"confirmed_joint":[x["arm"] for x in joint],"verdict":verdict,"confirmation_gates":gate_hits(cells)};json.dump(summary,open("results/SUMMARY.json","w"),indent=2);json.dump({"cells":cells},open("results/CONFIRMATION_COMBINED.json","w"),ensure_ascii=False);print(json.dumps(summary,indent=2))
    else:raise SystemExit(mode)
if __name__=="__main__":main()
