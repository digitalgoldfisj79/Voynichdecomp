#!/usr/bin/env python3
"""Shared physical-line ordering primitives for sequence-sensitive Voynich pipelines.

Introduced after the XD1/v03 propagation audit. New sequence-sensitive code must import
this module rather than maintaining an independent line-label parser.
"""
import re
_NUM_PREFIX=re.compile(r"^(\d+)")

def numeric_line_no(v):
    if isinstance(v,bool): return int(v)
    if isinstance(v,(int,float)): return int(v)
    s=str(v)
    m=_NUM_PREFIX.match(s)
    if not m:
        raise ValueError(f"line label has no numeric prefix: {v!r}")
    return int(m.group(1))

def order_key(v):
    # Exact compatibility with the already-regression-tested XD1 v2 key.
    if isinstance(v,(int,float)) and not isinstance(v,bool):
        return (0,float(v),"")
    s=str(v)
    try:
        return (0,float(s),"")
    except ValueError:
        return (1,0.0,s)
