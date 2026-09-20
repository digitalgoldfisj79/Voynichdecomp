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
