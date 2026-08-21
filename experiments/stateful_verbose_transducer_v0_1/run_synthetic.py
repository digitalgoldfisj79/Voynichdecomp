#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, statistics
from pathlib import Path
import svt_v01 as svt

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def load_language(repo:Path,iso:str):
    root=repo/"experiments"/"recoverability_frontier_v0_5"; languages=svt.core.load_languages(root/"corpus_manifest_v050.json",repo/".cache"/"svt-v01")
    if iso not in languages: raise SystemExit(f"language {iso!r} not in pinned manifest")
    return languages[iso]
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--stage",choices=("segmentation","oracle_head","joint","locked"),required=True); p.add_argument("--split",choices=("dev","test"),default="dev"); p.add_argument("--iso",default="de"); p.add_argument("--length",type=int,default=384); p.add_argument("--replicates",type=int,default=4); p.add_argument("--iterations",type=int,default=80000); p.add_argument("--restarts",type=int,default=8); p.add_argument("--beam",type=int,default=8); p.add_argument("--boundary-weight",type=float,default=.35); args=p.parse_args()
    language=load_language(args.repo,args.iso); model=svt.mono.build_language_model(language); positives=[svt.make_svt_trial(language,args.split,args.length,mode,r) for mode in svt.MODES for r in range(args.replicates)]; rows=[]
    if args.stage=="segmentation":
        for t in positives:
            paths=svt.top_segmentations(t.surface,t.surface_line_starts,len(language.alphabet),args.beam); f=[svt.boundary_f1(x.starts,t.head_positions) for x in paths]; rows.append({"family":"FSVT","mode":t.head.mode,"period":t.head.period,"replicate":t.head.replicate,"top1_boundary_f1":f[0],"best_lattice_boundary_f1":max(f)})
        top=[r["top1_boundary_f1"] for r in rows]; best=[r["best_lattice_boundary_f1"] for r in rows]; summary={"trials":len(rows),"mean_boundary_f1":statistics.fmean(top),"median_boundary_f1":statistics.median(top),"best_lattice_mean_boundary_f1":statistics.fmean(best)}
    elif args.stage=="oracle_head":
        rows=[svt.solve_true_heads(t,language,model,args.iterations,args.restarts) for t in positives]; rec=[r["recovery"] for r in rows]; summary={"trials":len(rows),"mean_recovery":statistics.fmean(rec),"median_recovery":statistics.median(rec),"at_least_85":sum(v>=.85 for v in rec),"structure_accuracy":statistics.fmean(r["structure_correct"] for r in rows)}
    else:
        rows=[svt.solve_svt_trial(t,language,model,args.iterations,args.restarts,args.beam,args.boundary_weight) for t in positives]; summary=svt.summarize_joint(rows)
    controls=[]
    if args.stage=="locked":
        for t in positives:
            controls.append(svt.solve_svt_trial(svt.make_nonfactorable_control(t,language),language,model,args.iterations,args.restarts,args.beam,args.boundary_weight)); controls.append(svt.solve_svt_trial(svt.make_shuffled_control(t),language,model,args.iterations,args.restarts,args.beam,args.boundary_weight))
    config=Path(__file__).resolve().parent/"FROZEN_CONFIG_v0_1.json"; payload={"programme":"stateful_verbose_transducer_v0_1","stage":args.stage,"split":args.split,"iso":args.iso,"length":args.length,"replicates_per_mode":args.replicates,"config_sha256":sha256_file(config),"rows":rows,"controls":controls,"summary":summary}; locked=False
    if args.stage=="locked" and args.split=="test" and len(rows)>=20 and controls:
        rec=[float(r["recovery"]) for r in rows]; bf=[float(r["boundary_f1"]) for r in rows]; fpr=statistics.fmean(float(r["recovery"])>=.70 for r in controls); payload["hostile_control_fpr_at_recovery_0_70"]=fpr; locked=statistics.fmean(rec)>=.75 and statistics.median(rec)>=.85 and sum(v>=.70 for v in rec)>=16 and statistics.fmean(bf)>=.80 and statistics.fmean(bool(r["structure_correct"]) for r in rows)>=.75 and fpr<=.05
    payload["locked_gate_pass"]=locked; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps({"output":str(args.output),"locked_gate_pass":locked,"summary":summary},indent=2))
if __name__=="__main__": main()
