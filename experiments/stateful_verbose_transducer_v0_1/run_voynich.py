#!/usr/bin/env python3
"""Gate-protected target application for SVT v0.1."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import svt_v01 as svt

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def validate_gate(gate_path:Path,config_path:Path)->dict:
    gate=json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("programme")!="stateful_verbose_transducer_v0_1": raise SystemExit("refusing target access: wrong programme gate")
    if gate.get("stage")!="locked" or gate.get("split")!="test": raise SystemExit("refusing target access: not an untouched locked-test gate")
    if gate.get("locked_gate_pass") is not True: raise SystemExit("refusing target access: locked synthetic gate failed")
    if gate.get("config_sha256")!=sha256_file(config_path): raise SystemExit("refusing target access: frozen-config SHA mismatch")
    return gate
def load_lines(path:Path)->list[list[str]]:
    lines=[]
    for raw in path.read_text(encoding="utf-8").splitlines():
        text="".join(raw.strip().split())
        if text: lines.append(list(text))
    if not lines: raise SystemExit("empty target representation")
    return lines
def encode_lines(lines:list[list[str]])->tuple[list[int],list[int],dict[str,int]]:
    mapping={}; surface=[]; starts=[]
    for line in lines:
        starts.append(len(surface))
        for symbol in line:
            if symbol not in mapping: mapping[symbol]=len(mapping)
            surface.append(mapping[symbol])
    return surface,starts,mapping
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,required=True); p.add_argument("--gate",type=Path,required=True); p.add_argument("--transcription",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--isos",default="de,en,fi,tr"); p.add_argument("--iterations",type=int,default=80000); p.add_argument("--restarts",type=int,default=8); p.add_argument("--beam",type=int,default=8); args=p.parse_args()
    here=Path(__file__).resolve().parent; config=here/"FROZEN_CONFIG_v0_1.json"; gate=validate_gate(args.gate,config)
    lines=load_lines(args.transcription); surface,line_starts,mapping=encode_lines(lines); root=args.repo/"experiments"/"recoverability_frontier_v0_5"; languages=svt.core.load_languages(root/"corpus_manifest_v050.json",args.repo/".cache"/"svt-v01-target"); results=[]
    for iso in [x.strip() for x in args.isos.split(",") if x.strip()]:
        if iso not in languages: continue
        language=languages[iso]
        if len(mapping)>len(language.alphabet):
            results.append({"iso":iso,"status":"ABSTAIN_ALPHABET_CARDINALITY","surface_symbols":len(mapping),"plaintext_symbols":len(language.alphabet)}); continue
        model=svt.mono.build_language_model(language); sol=svt.solve_surface(surface,line_starts,language,model,args.iterations,args.restarts,args.beam,svt.BOUNDARY_WEIGHT,int(svt.core.stable_seed("svt-target",iso)))
        results.append({"iso":iso,"status":"SOLVED_STRUCTURALLY","joint_score":sol.joint_score,"head_score_per_symbol":sol.head_score_per_symbol,"boundary_score_per_surface":sol.boundary_score_per_surface,"selected_mode":sol.head_solution.mode,"selected_period":sol.head_solution.period,"head_count":len(sol.path.starts),"predicted_starts":sol.path.starts,"prediction":sol.head_solution.prediction})
    payload={"programme":"stateful_verbose_transducer_v0_1","status":"TARGET_SCORED","gate_sha256":sha256_file(args.gate),"config_sha256":gate["config_sha256"],"transcription_sha256":sha256_file(args.transcription),"line_count":len(lines),"surface_length":len(surface),"surface_symbol_count":len(mapping),"symbol_mapping":mapping,"results":results,"interpretation_lock":"Language ranking is diagnostic only unless separately calibrated; no semantic/manual selection is permitted."}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); print(json.dumps({"output":str(args.output),"results":len(results)},indent=2))
if __name__=="__main__": main()
