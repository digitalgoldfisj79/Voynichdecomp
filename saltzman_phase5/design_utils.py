#!/usr/bin/env python3
import math
import numpy as np

WIDTHS=(6,8,10,12,16,20)

def lineate_words(words,width):
    w=int(width)
    if w<2:
        raise ValueError("width must be >=2")
    z=list(words)[:2000]
    out=[z[i:i+w] for i in range(0,len(z),w) if len(z[i:i+w])>=2]
    return out

def ols_slope(xs,ys):
    x=np.asarray(xs,float); y=np.asarray(ys,float)
    if len(x)!=len(y) or len(x)<2:
        raise ValueError("bad slope inputs")
    xc=x-x.mean()
    den=float(np.dot(xc,xc))
    if den<=0:
        raise ValueError("degenerate x")
    return float(np.dot(xc,y-y.mean())/den)

def slope_x(widths=WIDTHS):
    return [math.log2(float(w)/10.0) for w in widths]

def adjudicate(replication_ok, contrasts, slopes, margin=0.04):
    if not replication_ok:
        return "P4_CURVE_NOT_REPLICATED_W10"
    families=("fixed","markov")
    widths=tuple(WIDTHS)
    all_short=all(
        contrasts[f][w]["ci95"][0] > 0
        for f in families for w in widths
    )
    equiv=all(
        slopes[f]["ci95"][0] >= -margin and slopes[f]["ci95"][1] <= margin
        for f in families
    )
    line_shift=all(slopes[f]["ci95"][1] < -margin for f in families)
    if all_short and equiv:
        return "ABSOLUTE_SHORT_SCALE_ROBUST"
    if all_short:
        return "SHORT_SCALE_ROBUST_WITH_LINE_INTERACTION"
    if line_shift:
        return "LINE_RELATIVE_SHIFT_SUPPORTED"
    return "MIXED_OR_UNRESOLVED"

def self_tests():
    words=[f"w{i}" for i in range(2000)]
    for w in WIDTHS:
        lines=lineate_words(words,w)
        assert sum(map(len,lines))==2000,(w,sum(map(len,lines)))
        assert all(len(x)>=2 for x in lines)
    xs=slope_x()
    assert abs(ols_slope(xs,[2*x+1 for x in xs])-2.0)<1e-12
    pos={f:{w:{"ci95":[0.10,0.20]} for w in WIDTHS} for f in ("fixed","markov")}
    eq={f:{"ci95":[-0.01,0.01]} for f in ("fixed","markov")}
    neg={f:{"ci95":[-0.09,-0.05]} for f in ("fixed","markov")}
    broken={f:{w:{"ci95":[0.10,0.20]} for w in WIDTHS} for f in ("fixed","markov")}
    broken["fixed"][20]={"ci95":[-0.02,0.01]}
    assert adjudicate(True,pos,eq)=="ABSOLUTE_SHORT_SCALE_ROBUST"
    assert adjudicate(True,pos,neg)=="SHORT_SCALE_ROBUST_WITH_LINE_INTERACTION"
    assert adjudicate(True,broken,neg)=="LINE_RELATIVE_SHIFT_SUPPORTED"
    assert adjudicate(False,pos,eq)=="P4_CURVE_NOT_REPLICATED_W10"
    return True

if __name__=="__main__":
    assert self_tests()
    print("phase5 design self-tests: OK")
