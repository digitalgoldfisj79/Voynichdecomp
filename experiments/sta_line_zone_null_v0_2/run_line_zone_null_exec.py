#!/usr/bin/env python3
"""Execution wrapper for frozen v0.2 protocol.

Pretarget implementation fixes only:
- cache invariant event counts in C1;
- forbid planted swap partner k from being the source position i and explicitly
  require the intended (i,j) lag-2 pair to become equal.
No scientific threshold, source, null, seed family, or adjudication is changed.
"""
import importlib.util, random
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE/'run_line_zone_null.py'
spec=importlib.util.spec_from_file_location('lzbase',BASE)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixed_null_pseudo_control(ref,lines,n=m.N_NULL_PSEUDO):
    pseudo=m.permutation_counts(lines,n,m.SEED+60000); passes=0; z2s=[]; contrasts=[]; pmaxs=[]; pspecs=[]
    events=m.counts_by_lag(lines)['events']
    for r in range(n):
        obs={'lag':{lag:int(pseudo['lag'][lag][r]) for lag in m.LAGS},'events':events,
             'lag2_left':int(pseudo['lag2_left'][r]),'lag2_right':int(pseudo['lag2_right'][r])}
        s=m.summarize_obs(obs,ref); passes+=int(s['full_lag2_specific_gate']); z2s.append(s['lags']['2']['z'])
        contrasts.append(s['lag2_specificity_contrast']); pmaxs.append(s['p_maxT_lag2']); pspecs.append(s['p_specificity'])
    frac=passes/n
    return {'n':n,'gate_passes':passes,'gate_fraction':frac,'median_z2':m.med(z2s),'p99_z2':m.qtile(z2s,.99),
            'median_contrast':m.med(contrasts),'median_pmax':m.med(pmaxs),'median_pspecificity':m.med(pspecs),'pass':bool(frac<=.05)}


def fixed_plant_line_lag2(tokens,rng,q=m.PLANT_Q):
    t=list(tokens); n=len(t); used=set(); swaps=0
    order=m.boundary_starts(n,2); rng.shuffle(order)
    for i in order:
        j=i+2
        if i in used or j in used or t[i]==t[j] or rng.random()>=q: continue
        src=t[i]; before=m.line_match_counts(t); candidates=[]
        for k in m.target_zone_positions(n,j):
            if k in (i,j) or k in used or t[k]!=src: continue
            tt=t.copy(); tt[j],tt[k]=tt[k],tt[j]
            if tt[i]!=tt[j]: continue
            after=m.line_match_counts(tt); d2=after[2]-before[2]
            if d2<=0: continue
            collateral=sum(abs(after[L]-before[L]) for L in (1,3,4))
            candidates.append((-d2,collateral,k,tt))
        if not candidates: continue
        candidates.sort(key=lambda z:(z[0],z[1],z[2])); _,_,k,tt=candidates[0]
        if m.line_zone_counters(t)!=m.line_zone_counters(tt): raise RuntimeError('plant changed line-zone inventory')
        t=tt; used.update((i,j,k)); swaps+=1
    return tuple(t),swaps

m.null_pseudo_control=fixed_null_pseudo_control
m.plant_line_lag2=fixed_plant_line_lag2

if __name__=='__main__':
    m.main()
