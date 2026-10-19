#!/usr/bin/env python3
"""Evaluate real Anchor-1 admissibility and OPTIONAL human review exports.

This instrument NEVER infers that ytoail means a plant. Real source is 1 folio,
so a 20-seed simulation cannot provide independent replication.
"""
import argparse,collections,json,pathlib,statistics,sys

ECOLOGY="Herbal-A label";MIN_FOLIOS=20;MIN_REVIEWERS=3
def check_review(r):
    errs=[]
    if r.get("schema")!="ANCHOR1_VISUAL_REVIEW_20261019":errs.append("wrong review schema")
    if r.get("status")!="EXPLORATORY_NOT_FULLY_BLIND":errs.append("blindness status altered")
    if r.get("target")!="Beinecke MS 408 f2r":errs.append("wrong target")
    obs=r.get("observations",{})
    if not obs.get("target_morphology"):errs.append("no independent morphology description")
    if not obs.get("production_only_rival"):errs.append("no recorded production-only rival")
    sc=obs.get("comparisons",[])
    if sorted(x.get("code") for x in sc)!=["A","B","C","D"]:errs.append("incomplete comparison panel")
    if any(x.get("score") not in ["0","1","2","3"] for x in sc):errs.append("missing or malformed blind visual score")
    bbox=obs.get("label_rectangle_normalised")
    if bbox is not None:
      if (not isinstance(bbox,dict)
        or not (0<=bbox.get("left",-1)<bbox.get("right",-1)<=1)
        or not (0<=bbox.get("top",-1)<bbox.get("bottom",-1)<=1)):
          errs.append("invalid label rectangle")
    return errs

def run(manifest,review_paths):
    if manifest.get("schema")!="VMS-ANCHOR1-F2R-20261019":raise ValueError("source manifest mismatch")
    if manifest["target"]["text_type"]!="label" or manifest["target"]["transcription"]!="ytoail":
      raise ValueError("must use f2r label ecology; no fallback to f2r initial running word")
    reviewed=[];warnings=[]
    for p in review_paths:
      r=json.loads(pathlib.Path(p).read_text())
      e=check_review(r)
      if e:warnings.append({"file":p,"issues":e})
      else:reviewed.append(r)
    image_scores=[]
    for r in reviewed:
      image_scores.append({x["code"]:int(x["score"]) for x in r["observations"]["comparisons"]})
    code_medians={k:(statistics.median(x[k] for x in image_scores) if image_scores else None)
                  for k in ["A","B","C","D"]}
    reviewed_roles=[r.get("expertise") for r in reviewed]
    folio_independent_count=1
    reviews_full_blind=False
    object_read=manifest["target"]["image_object"]["reviewed"]
    target_species_verified=manifest["admissibility"]["vms_species_verified"]
    conditions={
      "single_canon_folio_locus_verified":True,
      "candidate_image_species_independently_verified":bool(target_species_verified),
      "physical_label_pixel_bbox_verified":any(r["observations"].get("label_rectangle_normalised") is not None for r in reviewed),
      "external_source_caption_crop_and_botany_adjudicated":False,
      "three_independent_reviewers":len({r.get("reviewer") for r in reviewed if r.get("reviewer")})>=MIN_REVIEWERS,
      "fully_blind_botanical_crops":reviews_full_blind,
      "twenty_distinct_heldout_physically_independent_herbal_label_folios":folio_independent_count>=MIN_FOLIOS,
      "matched_production_only_negative_collected":False,
      "meaning_prediction_preset_without_text_lookups":False
    }
    qualified=all(conditions.values())
    return {"schema":"ANCHOR1_ADJUDICATION_v1",
      "anchor":"Beinecke MS408 f2r ZLZI line14 @Lp ytoail",
      "label_semantics":"UNKNOWN","qualifies_for_semantic_decode":qualified,
      "status":"QUALIFIED_FOR_CONFIRMATORY_TEST" if qualified else "NOT_QUALIFIED_MISSING_INDEPENDENT_ANCHORS",
      "source_witnesses":"Egerton 747 botanical alternatives; NO same-species matching has been certified.",
      "review_count":len(reviewed),"rejected_review_files":warnings,"expertise_roles":reviewed_roles,
      "exploratory_visual_similarity_medians":code_medians,
      "conditions":conditions,
      "unresolved_core":"Even if one witness illustration looks highly similar, the same observation can be caused by shared artistic exemplars/production; one folio and disputed species cannot discriminate encoded name from nonsemantic label.",
      "do_not_pool":"Running text, Zodiac circular labels, pharmaceutical labels, and single-glyph alphabet annotations are separate ecologies."}

def main():
   p=argparse.ArgumentParser()
   p.add_argument("--manifest",required=True)
   p.add_argument("--reviews",nargs="*",default=[])
   p.add_argument("--out")
   a=p.parse_args()
   d=run(json.loads(pathlib.Path(a.manifest).read_text()),a.reviews)
   s=json.dumps(d,indent=2)
   if a.out:pathlib.Path(a.out).write_text(s)
   print("ANCHOR1_ADJUDICATION="+json.dumps(d,sort_keys=True,separators=(",",":")))
if __name__=="__main__":main()
