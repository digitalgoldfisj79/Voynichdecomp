#!/usr/bin/env python3
import importlib.util, json, sys
spec=importlib.util.spec_from_file_location("xd1","research/xd1_core_20260920.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

obj={"label":"fixture","lines":[
 {"block":"p1","unit":"u","segment":"A","line_order":10,"tokens":["cc"]},
 {"block":"p1","unit":"u","segment":"A","line_order":1,"tokens":["aa"]},
 {"block":"p1","unit":"u","segment":"A","line_order":2,"tokens":["bb"]},
 {"block":"p1","unit":"u","segment":"B","line_order":11,"tokens":["dd"]},
]}
lines=m.normalize_input(obj)
assert [r["line_order"] for r in lines]==[1,2,10,11], [r["line_order"] for r in lines]
jp=m.junction_pairs(lines)
lbs=[r for r in jp if r["boundary"]=="LINE_BREAK"]
assert len(lbs)==2, lbs
tr=m.transitions(lines)
starts=[r for r in tr if r["boundary"]=="PAGE_START"]
assert len(starts)==2, starts
assert [r["target"] for r in tr]==["aa","bb","cc","dd"], [r["target"] for r in tr]
print(json.dumps({"status":"PASS","line_order":[r["line_order"] for r in lines],
                  "linebreak_pairs":len(lbs),"segment_starts":len(starts),
                  "transition_order":[r["target"] for r in tr]},sort_keys=True))

import numpy as np
def naive(ids,lo,hi):
    n=len(ids); eligible=max(0,n-lo)
    if eligible==0:return None
    hits=0
    for i in range(lo,n):
        a=max(0,i-hi); z=i-lo+1
        if a<z and np.any(ids[a:z]==ids[i]): hits+=1
    return hits/eligible
rng=np.random.default_rng(123)
for n in (2,5,17,80):
    ids=rng.integers(0,9,size=n,dtype=np.int32)
    for lo,hi in m.P5_BANDS:
        a=naive(ids,lo,hi); b=m.band_rate(ids,lo,hi)
        if a is None: assert b is None
        else: assert abs(a-b)<1e-15,(n,lo,hi,a,b,ids)
print("P5_VECTOR_EQUIVALENCE=PASS")
