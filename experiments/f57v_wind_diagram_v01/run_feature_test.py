#!/usr/bin/env python3
"""Matched visual-feature test for f57v wind-diagram programme v0.1.

Input CSV requires:
id,stratum_id,label,<frozen feature columns>
label in {wind,control,target}
Feature values: 0, 0.5, 1, or blank.

No semantic text features are accepted.
"""
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict

FEATURES = [
    "circular_layout","concentric_annuli","radial_sectoring",
    "fourfold_orthogonal_humans","radial_personifications","inward_orientation",
    "explicit_breath","central_emblem","centre_directed_relation",
    "modular_4_8_12","annular_marks_or_writing","rotational_symmetry",
]
GROUPS = {
    "geometry": ["circular_layout","concentric_annuli","radial_sectoring",
                 "modular_4_8_12","annular_marks_or_writing","rotational_symmetry"],
    "anthropomorph": ["fourfold_orthogonal_humans","radial_personifications",
                      "inward_orientation","explicit_breath","centre_directed_relation"],
}

def fval(x):
    x=x.strip()
    if x=="":
        return None
    v=float(x)
    if v not in (0.0,0.5,1.0):
        raise ValueError(f"feature value must be 0, 0.5, 1 or blank; got {x!r}")
    return v

def load_rows(path):
    with open(path,newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    req={"id","stratum_id","label",*FEATURES}
    missing=req-set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    for r in rows:
        if r["label"] not in {"wind","control","target"}:
            raise ValueError(f"bad label {r['label']}")
        r["_x"]={k:fval(r[k]) for k in FEATURES}
    return rows

def gower(a,b,features):
    diffs=[]
    for k in features:
        x,y=a["_x"][k],b["_x"][k]
        if x is None or y is None:
            continue
        diffs.append(abs(x-y))
    if not diffs:
        return math.nan
    return sum(diffs)/len(diffs)

def exact_sign_p(k,n):
    if n<=0: return math.nan
    return sum(math.comb(n,i) for i in range(k,n+1))/(2**n)

def median(xs):
    ys=sorted(xs); n=len(ys)
    if not n: return math.nan
    return ys[n//2] if n%2 else (ys[n//2-1]+ys[n//2])/2

def primary(rows, features):
    targets=[r for r in rows if r["label"]=="target"]
    if len(targets)!=1:
        raise ValueError("exactly one target required")
    t=targets[0]
    strata=defaultdict(lambda: {"wind":[],"control":[]})
    for r in rows:
        if r["label"] in {"wind","control"}:
            strata[r["stratum_id"]][r["label"]].append(r)
    deltas={}
    for sid,b in strata.items():
        if not b["wind"] or not b["control"]:
            continue
        dw=[gower(t,r,features) for r in b["wind"]]
        dc=[gower(t,r,features) for r in b["control"]]
        dw=[x for x in dw if not math.isnan(x)]; dc=[x for x in dc if not math.isnan(x)]
        if dw and dc:
            deltas[sid]=sum(dc)/len(dc)-sum(dw)/len(dw)
    vals=list(deltas.values()); k=sum(v>0 for v in vals)
    return {"n_strata":len(vals),"positive_strata":k,"median_delta":median(vals),
            "sign_p":exact_sign_p(k,len(vals)),"deltas":deltas}

def positive_control(rows, features):
    winds=[r for r in rows if r["label"]=="wind"]
    ctrls=[r for r in rows if r["label"]=="control"]
    correct=0; total=0; details=[]
    for w in winds:
        w_other=[x for x in winds if x["stratum_id"]!=w["stratum_id"]]
        c_other=[x for x in ctrls if x["stratum_id"]!=w["stratum_id"]]
        if not w_other or not c_other: continue
        dw=[gower(w,x,features) for x in w_other]; dc=[gower(w,x,features) for x in c_other]
        dw=[x for x in dw if not math.isnan(x)]; dc=[x for x in dc if not math.isnan(x)]
        if not dw or not dc: continue
        mdw=sum(dw)/len(dw); mdc=sum(dc)/len(dc); ok=mdw<mdc
        correct+=int(ok); total+=1; details.append((w["id"],ok,mdw,mdc))
    return {"n":total,"correct":correct,"accuracy":correct/total if total else math.nan,"details":details}

def fmt(x):
    return "NA" if isinstance(x,float) and math.isnan(x) else (f"{x:.6f}" if isinstance(x,float) else str(x))

def report(rows):
    ablations = {
        "primary_all": FEATURES,
        "geometry_only": GROUPS["geometry"],
        "anthropomorph_only": GROUPS["anthropomorph"],
        "remove_explicit_breath":[f for f in FEATURES if f!="explicit_breath"],
        "remove_centre":[f for f in FEATURES if f not in {"central_emblem","centre_directed_relation"}],
    }
    for name,feats in ablations.items():
        pc=positive_control(rows,feats); pr=primary(rows,feats)
        print(f"[{name}]")
        print(f"positive_control_n={pc['n']} accuracy={fmt(pc['accuracy'])}")
        print(f"n_strata={pr['n_strata']} positive={pr['positive_strata']} median_delta={fmt(pr['median_delta'])} sign_p={fmt(pr['sign_p'])}")
        for sid,d in sorted(pr["deltas"].items()): print(f"  {sid}: delta={d:.6f}")
        print()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("coded_csv"); args=ap.parse_args()
    report(load_rows(args.coded_csv))

if __name__=="__main__": main()
