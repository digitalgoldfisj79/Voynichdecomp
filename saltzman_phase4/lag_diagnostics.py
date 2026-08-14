#!/usr/bin/env python3
import math
import numpy as np

LAGS=tuple(range(1,25))

def autocorr_pm1(states,lags=LAGS):
    x=np.asarray([1.0 if int(s) else -1.0 for s in states],dtype=float)
    out={}
    for h in lags:
        if h>=len(x):
            out[str(h)]=None
            continue
        a=x[:-h]; b=x[h:]
        sa=a.std(); sb=b.std()
        if sa==0 or sb==0:
            out[str(h)]=None
        else:
            out[str(h)]=float(np.corrcoef(a,b)[0,1])
    return out

def theory_fixed(run_length,lags=LAGS):
    L=float(run_length)
    return {str(h):max(1.0-float(h)/L,0.0) for h in lags}

def theory_markov(p_stay,lags=LAGS):
    lam=2.0*float(p_stay)-1.0
    return {str(h):float(lam**h) for h in lags}

def theory_occurrence(lags=LAGS):
    return {str(h):0.0 for h in lags}

def theory_for_arm(arm,lags=LAGS):
    if arm in ("OCCURRENCE_K2","FIXED_RUN1_K2"):
        return theory_occurrence(lags)
    if arm.startswith("FIXED_RUN") and arm.endswith("_K2"):
        L=int(arm[len("FIXED_RUN"):-len("_K2")])
        return theory_fixed(L,lags)
    if arm.startswith("MARKOV_M") and arm.endswith("_K2"):
        lookup={"M2":.5,"M3":2/3,"M4":.75,"M5":.8,"M8":.875,"M12":11/12}
        label=arm[len("MARKOV_"):-len("_K2")]
        return theory_markov(lookup[label],lags)
    return None

def rmse(empirical,theory):
    vals=[]
    for k,t in theory.items():
        e=empirical.get(str(k))
        if e is not None and t is not None and math.isfinite(e) and math.isfinite(t):
            vals.append((e-t)**2)
    return float(math.sqrt(sum(vals)/len(vals))) if vals else None

def median_acfs(acfs,lags=LAGS):
    out={}
    for h in lags:
        vals=[a.get(str(h)) for a in acfs]
        vals=[float(v) for v in vals if v is not None and math.isfinite(float(v))]
        out[str(h)]=float(np.median(vals)) if vals else None
    return out

def self_tests():
    assert theory_fixed(4,{1,2,3,4,5})=={"1":.75,"2":.5,"3":.25,"4":0.0,"5":0.0}
    m=theory_markov(.75,{1,2,3})
    assert abs(m["1"]-.5)<1e-12 and abs(m["2"]-.25)<1e-12 and abs(m["3"]-.125)<1e-12
    o=theory_occurrence({1,2})
    assert o=={"1":0.0,"2":0.0}
    return True

if __name__=="__main__":
    assert self_tests()
    print("lag_diagnostics self-tests: OK")
