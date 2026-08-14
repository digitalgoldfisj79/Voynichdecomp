#!/usr/bin/env python3
def _inside(ci,lo,hi):
    return ci[0] >= lo and ci[1] <= hi

def adjudicate(replication_ok, reset_stats, margin):
    if not replication_ok:
        return "P6_CONTINUOUS_ENDPOINTS_NOT_REPLICATED"
    F=reset_stats["fixed"]; G=reset_stats["geometric"]
    if F["ci95"][0] > margin and G["ci95"][0] > margin:
        return "CONTINUITY_REQUIRED_BOTH"
    if F["ci95"][1] < -margin and G["ci95"][1] < -margin:
        return "LINE_RESET_BETTER_BOTH"
    if _inside(F["ci95"],-margin,margin) and _inside(G["ci95"],-margin,margin):
        return "RESET_EQUIVALENT_BOTH"
    if ((F["ci95"][0] > margin and G["ci95"][1] < -margin) or
        (G["ci95"][0] > margin and F["ci95"][1] < -margin)):
        return "DWELL_DEPENDENT_RESET"
    if F["median"] != 0 and G["median"] != 0 and (F["median"] > 0)==(G["median"] > 0):
        return "CONSISTENT_DIRECTION_NOT_ROBUST"
    return "MIXED_OR_UNRESOLVED"

def self_tests():
    m=.04
    z=lambda med,lo,hi:{"median":med,"ci95":[lo,hi]}
    assert adjudicate(False,{"fixed":z(.1,.08,.12),"geometric":z(.1,.08,.12)},m)=="P6_CONTINUOUS_ENDPOINTS_NOT_REPLICATED"
    assert adjudicate(True,{"fixed":z(.1,.06,.12),"geometric":z(.09,.05,.11)},m)=="CONTINUITY_REQUIRED_BOTH"
    assert adjudicate(True,{"fixed":z(-.1,-.12,-.06),"geometric":z(-.09,-.11,-.05)},m)=="LINE_RESET_BETTER_BOTH"
    assert adjudicate(True,{"fixed":z(.0,-.02,.02),"geometric":z(.01,-.01,.03)},m)=="RESET_EQUIVALENT_BOTH"
    assert adjudicate(True,{"fixed":z(.1,.06,.12),"geometric":z(-.1,-.12,-.06)},m)=="DWELL_DEPENDENT_RESET"
    assert adjudicate(True,{"fixed":z(.02,-.01,.05),"geometric":z(.03,.0,.06)},m)=="CONSISTENT_DIRECTION_NOT_ROBUST"
    return True

if __name__=="__main__":
    assert self_tests()
    print("Phase 7 design self-tests: OK")
