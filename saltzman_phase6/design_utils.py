#!/usr/bin/env python3
def ci_inside(ci,lo,hi):
    return ci[0] >= lo and ci[1] <= hi

def gradient_supported(stats,margin):
    return (stats["FG"]["ci95"][0] > margin
            and stats["FS"]["ci95"][0] > 0.0
            and stats["SG"]["ci95"][0] > 0.0)

def fg_excludes_zero(stats):
    lo,hi=stats["FG"]["ci95"]
    return lo>0.0 or hi<0.0

def adjudicate(replication_ok, by_tau, margin):
    if not replication_ok:
        return "P5_W10_ENDPOINTS_NOT_REPLICATED"
    all_equiv=True
    for tau in ("3","4"):
        for k in ("FG","FS","SG"):
            if not ci_inside(by_tau[tau][k]["ci95"],-margin,margin):
                all_equiv=False
    if all_equiv:
        return "TAU_SUFFICIENT_NO_DWELL_EFFECT"
    g3=gradient_supported(by_tau["3"],margin)
    g4=gradient_supported(by_tau["4"],margin)
    if g3 and g4:
        return "REGULARITY_GRADIENT_BOTH_TAU"
    if g3:
        return "REGULARITY_GRADIENT_TAU3_ONLY"
    if g4:
        return "REGULARITY_GRADIENT_TAU4_ONLY"
    if fg_excludes_zero(by_tau["3"]) or fg_excludes_zero(by_tau["4"]):
        return "DWELL_LAW_MATTERS_NONMONOTONIC"
    return "MIXED_OR_UNRESOLVED"

def _s(m,lo,hi):
    return {"median":m,"ci95":[lo,hi],"n":190}

def self_tests():
    eq={t:{k:_s(0,-.01,.01) for k in ("FG","FS","SG")} for t in ("3","4")}
    assert adjudicate(True,eq,.04)=="TAU_SUFFICIENT_NO_DWELL_EFFECT"
    g={t:{k:_s(0,-.01,.01) for k in ("FG","FS","SG")} for t in ("3","4")}
    g["3"]={"FG":_s(.09,.06,.12),"FS":_s(.04,.01,.07),"SG":_s(.05,.01,.08)}
    assert adjudicate(True,g,.04)=="REGULARITY_GRADIENT_TAU3_ONLY"
    n={t:{k:_s(0,-.01,.01) for k in ("FG","FS","SG")} for t in ("3","4")}
    n["4"]["FG"]=_s(.05,.01,.09)
    assert adjudicate(True,n,.04)=="DWELL_LAW_MATTERS_NONMONOTONIC"
    assert adjudicate(False,eq,.04)=="P5_W10_ENDPOINTS_NOT_REPLICATED"
    return True

if __name__=="__main__":
    assert self_tests()
    print("design_utils self-tests: OK")
