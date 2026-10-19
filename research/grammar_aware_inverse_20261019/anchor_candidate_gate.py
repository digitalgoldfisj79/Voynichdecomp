#!/usr/bin/env python3
"""Admissibility gate for proposed independent VMS text-to-meaning anchors.

Checks *completeness and internal non-circularity declarations* only.
Does NOT establish historical authenticity, botanical identity, or source independence.
Read-only: never updates catalogue, handoff or inference results.
"""
import argparse, json, pathlib, re, sys

PROHIBITED_ONLY = {"plant","root","leaf","flower","star","nymph",
                   "human_figure","container","glyph","illustration","figure"}
TEXT_TYPES = {"running", "label", "circular", "spatial"}
MIN_CITATIONS=2

def check(x):
    errs=[]
    if not isinstance(x,dict):return ["not an object"]
    required=["id","hypothesis","voynich","external_witness",
              "evidence","negative_control","heldout"]
    for r in required:
        if not x.get(r):errs.append(f"missing {r}")
    v=x.get("voynich") or {}
    t=v.get("text_type")
    if t not in TEXT_TYPES:errs.append("voynich.text_type must be one of running,label,circular,spatial")
    if not (re.fullmatch(r"f\d+[rv](?:[1-9])?",str(v.get("folio","")))):
        errs.append("voynich.folio missing/cannot parse")
    if not v.get("line_or_segment"):
        errs.append("voynich.line_or_segment must localize text, not just folio")
    if not v.get("word_index") and v.get("word_index")!=0:
        errs.append("voynich.word_index must identify a specific text unit")
    if v.get("image_object_id") in (None,""):
        errs.append("voynich.image_object_id must localize independently observed image feature")
    w=x.get("external_witness") or {}
    if not w.get("shelfmark") or not w.get("folio_or_section"):
        errs.append("external_witness requires independently traceable shelfmark and folio/section")
    semantic=str(w.get("semantic_identity","")).lower().strip()
    if semantic in PROHIBITED_ONLY or not semantic:
        errs.append("generic image class is NOT a verified semantic identity")
    if not w.get("date_before_or_contemporary") or not w.get("historical_reading_reviewed"):
        errs.append("external identity needs a historically reviewed, chronological reading")
    evidence=x.get("evidence") or []
    if not isinstance(evidence,list) or len(evidence)<MIN_CITATIONS:
        errs.append("at least two documented independent evidence records required")
    else:
        kind=set()
        for ev in evidence:
            if not isinstance(ev,dict) or not ev.get("source_id") or not ev.get("locatable_citation"):
                errs.append("each evidence requires source_id and locatable_citation")
            if isinstance(ev,dict):kind.add(ev.get("kind"))
        if "external_text" not in kind or "image_alignment" not in kind:
            errs.append("evidence must independently cover external text AND image alignment")
        if len({str(ev.get("source_id")) for ev in evidence if isinstance(ev,dict)})<2:
            errs.append("citations must come from at least two independently identified records")
    nc=x.get("negative_control") or {}
    for p in ["same_text_type","same_section","same_scribe_or_matched_unknown","matched_line_position",
              "negative_folios","frozen_selection_rule"]:
        if not nc.get(p):errs.append(f"missing negative_control.{p}")
    hold=x.get("heldout") or {}
    if not hold.get("independent_bifolium") or not hold.get("no_anchor_label_used_for_selection"):
        errs.append("heldout must be independent physical bifolium and label-blind")
    hyp=x.get("hypothesis") or {}
    if not hyp.get("specific_prediction") or not hyp.get("competing_production_prediction"):
        errs.append("specify a DIFFERENT measurable prediction for semantic vs production mechanisms")
    if hyp.get("specific_prediction")==hyp.get("competing_production_prediction"):
        errs.append("semantic and production predictions identical => NONIDENTIFIABLE")
    if x.get("label_derived_from_voynich_text") is not False:
        errs.append("label provenance must certify independence from Voynich text")
    return errs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("manifest",nargs="?",help="JSON list of candidate anchor objects")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        generic={"id":"reject","voynich":{"folio":"f75r","text_type":"running"},
                 "external_witness":{"semantic_identity":"plant"}}
        e=check(generic)
        assert "generic image class is NOT a verified semantic identity" in e
        print(json.dumps({"self_test":"PASS","generic_noncitable_candidate_rejected":True,
              "issues":len(e),"note":"This checks format and declared independence, not factual truth."}))
        return
    if not args.manifest:p.error("manifest path or --self-test required")
    manifest=json.loads(pathlib.Path(args.manifest).read_text())
    if not isinstance(manifest,list):raise ValueError("manifest must be array")
    out=[{"id":x.get("id"),"admissible_for_expert_review":not(bool(e:=check(x))),"issues":e}
         for x in manifest]
    print(json.dumps({"n":len(out),"admissible_for_manual_review":sum(x["admissible_for_expert_review"] for x in out),
          "results":out,"warning":"Schema passing is not independent evidence; expert historical review remains mandatory."},
          indent=2))
if __name__=="__main__":main()
