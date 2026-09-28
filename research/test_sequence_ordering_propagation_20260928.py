#!/usr/bin/env python3
"""Propagation regression: shared ordering + v03.1 certifier + XD1 fixture."""
import importlib.util, pathlib, sys
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import sequence_ordering_20260928 as seq
import supergrammar_v031_certifier_20260928 as sg
import xd1_core_20260920 as xd1

labels=["10","1","11","2","2a"]
assert sorted(labels,key=seq.order_key)==["1","2","2a","10","11"]
assert [seq.numeric_line_no(x) for x in ("1","2a","10","11")]==[1,2,10,11]
assert [xd1.order_key(x) for x in labels]==[seq.order_key(x) for x in labels]

obj={"pages":{"f1r":{}}}
for lab,tok in [("10","cc"),("1","aa"),("11","dd"),("2","bb")]:
    obj["pages"]["f1r"][lab]={"u":"P","t":{"ZLZI":tok}}
lines=sg.build_lines(obj,"ZLZI")
pairs=[r for r in sg.build_junction_pairs(lines) if r["boundary"]=="LINE_BREAK"]
assert [r["target"] for r in pairs]==["b","c","d"],pairs
assert sg.davis_hand("f115r",1)=="S2"
assert sg.davis_hand("f115r",13)=="S3"

old=(HERE/"supergrammar_v03_certifier_20260920.py").read_text()
new=(HERE/"supergrammar_v031_certifier_20260928.py").read_text()
assert 'mm=re.match(r"(\\\\d+)",line_label)' in old
assert 'numeric_line_no(line_label)' in new
print("PROPAGATION_AUDIT=PASS")
print("HISTORICAL_V03_BUG_PRESERVED=YES")
print("V031_SHARED_ORDERING=PASS")
