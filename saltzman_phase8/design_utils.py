#!/usr/bin/env python3
def _inside(ci,lo,hi):
    return ci[0] >= lo and ci[1] <= hi

def adjudicate(replication_ok, stats, margin):
    if not replication_ok:
        return "PHASE7_POST_ENDPOINTS_NOT_REPLICATED"
    vals=list(stats.values())
    post=sum(x["ci95"][0] > margin for x in vals)
    pre=sum(x["ci95"][1] < -margin for x in vals)
    if post==4:
        return "POST_ORDER_REQUIRED_ALL"
    if pre==4:
        return "PRE_ORDER_REQUIRED_ALL"
    if all(_inside(x["ci95"],-margin,margin) for x in vals):
        return "ORDER_EQUIVALENT_ALL"
    if post>=3 and pre==0:
        return "POST_ORDER_ROBUST_MAJORITY"
    if pre>=3 and post==0:
        return "PRE_ORDER_ROBUST_MAJORITY"
    meds=[x["median"] for x in vals]
    if all(x>0 for x in meds) or all(x<0 for x in meds):
        return "ORDER_DIRECTIONALLY_CONSISTENT"
    return "ORDER_CONTEXT_DEPENDENT"

def self_tests():
    m=.04
    z=lambda x,lo,hi:{"median":x,"ci95":[lo,hi]}
    ks=["a","b","c","d"]
    mk=lambda row:{k:row[i] for i,k in enumerate(ks)}
    assert adjudicate(False,mk([z(.1,.06,.12)]*4),m)=="PHASE7_POST_ENDPOINTS_NOT_REPLICATED"
    assert adjudicate(True,mk([z(.1,.06,.12)]*4),m)=="POST_ORDER_REQUIRED_ALL"
    assert adjudicate(True,mk([z(-.1,-.12,-.06)]*4),m)=="PRE_ORDER_REQUIRED_ALL"
    assert adjudicate(True,mk([z(0,-.02,.02)]*4),m)=="ORDER_EQUIVALENT_ALL"
    assert adjudicate(True,mk([z(.1,.06,.12),z(.1,.05,.11),z(.08,.05,.1),z(.02,-.02,.06)]),m)=="POST_ORDER_ROBUST_MAJORITY"
    assert adjudicate(True,mk([z(-.1,-.12,-.06),z(-.1,-.11,-.05),z(-.08,-.1,-.05),z(-.02,-.06,.02)]),m)=="PRE_ORDER_ROBUST_MAJORITY"
    assert adjudicate(True,mk([z(.02,-.02,.06)]*4),m)=="ORDER_DIRECTIONALLY_CONSISTENT"
    return True

if __name__=="__main__":
    assert self_tests()
    print("Phase 8 design self-tests: OK")
