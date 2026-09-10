#!/usr/bin/env python3
import json, runpy
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

BASE="experiments/vms_oversized_parchment_class_20260910.py"
ns=runpy.run_path(BASE)
OUT=ns["OUT"]
df=ns["df"].copy()  # final (2400px) representation
exact_test=ns["exact_test"]
CANDIDATES=ns["CANDIDATES"]
CONTROLS=ns["CONTROLS"]

# 1) Presentation-mode sensitivity. This directly asks whether the positive
# signal exists in candidates that are not measured as complete spread frames.
pair_units={"q15_b87_90","q19_b99_102"}
spread_units={x["unit"] for x in CANDIDATES}-pair_units
controls={x["unit"] for x in CONTROLS}
pair_df=df[df.unit.isin(pair_units|controls)].reset_index(drop=True)
spread_df=df[df.unit.isin(spread_units|controls)].reset_index(drop=True)
mode={
 "pair_candidates_vs_controls":exact_test(pair_df,"max_rms"),
 "spread_candidates_vs_controls":exact_test(spread_df,"max_rms"),
 "pair_candidate_units":sorted(pair_units),
 "spread_candidate_units":sorted(spread_units),
}

# Same-quire Q17 check (descriptive n=1 vs n=1, no p-value).
q17cand=float(df.loc[df.unit=="q17_b94_95","max_rms"].iloc[0])
q17ctrl=float(df.loc[df.unit=="q17_b93_96","max_rms"].iloc[0])
mode["within_q17_descriptive"]={"candidate":q17cand,"control":q17ctrl,"difference":q17cand-q17ctrl,"ratio":q17cand/q17ctrl if q17ctrl else None}

# 2) Independent segmentation representation at 2400px.
# Instead of Otsu, use a fixed global lightness threshold defined from image
# quantiles. This is intentionally simple; agreement is stronger evidence than
# tuning another adaptive segmentation to the result.
def quantile_page_mask(img):
    L=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)[:,:,0].astype(float)
    lo=float(np.quantile(L,.10)); hi=float(np.quantile(L,.90)); thr=lo+.45*(hi-lo)
    m=(L>thr).astype(np.uint8)*255
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k,iterations=2)
    n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
    if n<2: raise RuntimeError("no parchment component alt mask")
    idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    z=(lab==idx).astype(np.uint8)
    cs,_=cv2.findContours(z,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cs: raise RuntimeError("no parchment contour alt mask")
    out=np.zeros_like(z); cv2.drawContours(out,[max(cs,key=cv2.contourArea)],-1,1,cv2.FILLED)
    return out

# Functions created by run_path use ns as their globals; replacing this entry
# changes only the segmentation representation, not edge definitions or tests.
ns["page_mask"]=quantile_page_mask
rows=[]
for c,specs in ((1,CANDIDATES),(0,CONTROLS)):
    for spec in specs:
        print("ALT_MASK",c,spec["unit"],flush=True)
        r=ns["measure_unit"](spec,2400); r["candidate"]=c; rows.append(r)
alt=pd.DataFrame(rows)
alt.to_csv(OUT/"features_2400_alt_quantile_mask.csv",index=False)
alt_primary=exact_test(alt,"max_rms")

# Pair-mode subset under the alternative mask: strongest available test against
# a complete-spread presentation artifact.
alt_pair=alt[alt.unit.isin(pair_units|controls)].reset_index(drop=True)
alt_pair_test=exact_test(alt_pair,"max_rms")

# Candidate-unit values are emitted to make dominance/audit transparent.
unit_values=df[["unit","quire","candidate","max_rms","mean_rms","max_qrange","mean_qrange"]].sort_values(["candidate","unit"],ascending=[False,True]).to_dict(orient="records")
alt_values=alt[["unit","quire","candidate","max_rms"]].sort_values(["candidate","unit"],ascending=[False,True]).to_dict(orient="records")

audit={
 "protocol":"vms_oversized_parchment_class_20260910_audit_v01",
 "base_protocol":ns["PROTOCOL"],
 "presentation_mode_sensitivity":mode,
 "alternative_mask_2400":{"method":"Lab L > q10 + .45*(q90-q10), largest connected light component","primary":alt_primary,"pair_candidates_vs_controls":alt_pair_test},
 "unit_values_2400_otsu":unit_values,
 "unit_values_2400_alt_mask":alt_values,
 "audit_decision_rule":"Physical-class signal survives this audit only if pair-only candidates remain positive vs controls and the alternative-mask primary remains positive and >=2 null SD. Exact p for pair-only is secondary because n=2.",
}
(OUT/"audit.json").write_text(json.dumps(audit,indent=2))
print("AUDIT_JSON")
print(json.dumps(audit,indent=2),flush=True)
