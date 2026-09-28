#!/usr/bin/env python3
"""Post-outcome completion of preregistered descriptive charm-vs-background bootstrap.
No selection, boundary, tokenizer, null or gate changes."""
import json,re,random,hashlib,statistics,math
from pathlib import Path
from research.xd1_charm_run import BASE,ITEM,get,parse_lines,tokenize,ranges_to_lines,observed,analytic_null

NBOOT=2000
def seed(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],"big")
def effect(lineobjs,d):
    toks=[x["tokens"] for x in lineobjs if len(x["tokens"])>=2]
    o,_,_=observed(toks,d);n=analytic_null(toks,d)
    return o-n,o,n

def q(vals,p):
    v=sorted(vals);x=(len(v)-1)*p;lo=int(x);hi=min(lo+1,len(v)-1);f=x-lo
    return v[lo]*(1-f)+v[hi]*f

def main():
    mf=json.load(open("research/XD1_CHARM_MANIFEST_20260928.json"))
    meta=json.loads(get(f"{BASE}/v1/metadata/json/{ITEM}"))
    wanted={r[0] for u in mf["primary_units"] for r in u["ranges"]}
    pages={}
    for p in meta["pages"]:
        lab=p.get("label");u=p.get("transcriptionDiplomaticURL")
        if not u:continue
        mm=re.fullmatch(r"(\d+)([rv])",str(lab or ""))
        if not ((mm and 1<=int(mm.group(1))<=89) or lab in wanted):continue
        ls=parse_lines(get(BASE+u).decode("utf-8","replace"))
        for x in ls:x["tokens"]=tokenize(x["text"])
        pages[lab]=ls
    units={u["id"]:ranges_to_lines(pages,u["ranges"]) for u in mf["primary_units"]}
    selected={(r[0],i) for u in mf["primary_units"] for r in u["ranges"] for i in range(r[1],r[2]+1)}
    mixed={(f,i) for f,i in mf["excluded_mixed_boundary_lines"]}
    bg_by_folio={}
    for fol,ls in pages.items():
        mm=re.fullmatch(r"(\d+)([rv])",str(fol or ""))
        if not (mm and 1<=int(mm.group(1))<=89) or fol=="1r":continue
        keep=[]
        for x in ls:
            if (fol,x["line_index"]) in selected or (fol,x["line_index"]) in mixed:continue
            keep.append(x)
        if keep:bg_by_folio[fol]=keep
    unit_names=list(units); folios=list(bg_by_folio)
    rng=random.Random(seed("XD1-CHARM-postoutcome-background-bootstrap"))
    vals={1:[],2:[]}
    for _ in range(NBOOT):
        cu=[rng.choice(unit_names) for __ in unit_names]
        bf=[rng.choice(folios) for __ in folios]
        cls=[x for k in cu for x in units[k]]
        bls=[x for f in bf for x in bg_by_folio[f]]
        for d in (1,2):
            ce,_,_=effect(cls,d);be,_,_=effect(bls,d)
            vals[d].append(ce-be)
    point={}
    primary=[x for k in unit_names for x in units[k]]
    background=[x for f in folios for x in bg_by_folio[f]]
    for d in (1,2):
        ce,co,cn=effect(primary,d);be,bo,bn=effect(background,d)
        ds=vals[d]
        point[f"lag{d}"]={
          "charm_effect_vs_analytic_null":ce,
          "background_effect_vs_analytic_null":be,
          "difference_charm_minus_background":ce-be,
          "bootstrap_sd":statistics.pstdev(ds),
          "difference_over_bootstrap_sd":(ce-be)/statistics.pstdev(ds),
          "bootstrap_ci95":[q(ds,.025),q(ds,.975)],
          "charm_observed":co,"charm_analytic_null":cn,
          "background_observed":bo,"background_analytic_null":bn
        }
    out={"protocol":"XD1-CHARM-20260928-v1.1","status":"POST_OUTCOME_PREREGISTERED_SECONDARY",
         "note":"Secondary descriptive comparison only; no primary gate is changed.",
         "bootstrap_replicates":NBOOT,"charm_clusters":len(unit_names),
         "background_folio_clusters":len(folios),"results":point}
    Path("xd1_charm_secondary").mkdir(exist_ok=True)
    Path("xd1_charm_secondary/charm_vs_background_bootstrap.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
