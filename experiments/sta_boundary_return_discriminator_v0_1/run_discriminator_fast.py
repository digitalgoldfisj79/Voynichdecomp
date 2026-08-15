#!/usr/bin/env python3
"""Performance-only runner for the frozen STA boundary-return discriminator v0.1.

Scientific protocol is unchanged. This wrapper:
1. evaluates the identical q grid / conditional likelihood using the algebraic
   separation of equality and non-equality events, avoiding 501 full-vector scans;
2. runs independent CN1 shuffled controls in two forked worker processes.

The original run_discriminator.py remains the binding specification and is also
running independently. Outputs should be deterministic-equivalent at the grid-q
and gate/verdict level.
"""
import importlib.util, math, multiprocessing as mp, random
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE/'run_discriminator.py'
spec=importlib.util.spec_from_file_location('frozen_discriminator',BASE)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

ORIG_FIT_Q=m.fit_q

def fast_fit_q(model, lines, eligible_only=True):
    eq_p0=[]
    neq_log_const=0.0
    neq_n=0
    total=0
    for x in lines:
        t=x['tokens']; n=len(t); sec=x['section']
        for j in range(2,n):
            if eligible_only and not m.boundary_eligible_target(j,n):
                continue
            p0=model.b2_marginal_prob(sec,n,m.p.edge_pos(j,n),t[j])
            total += 1
            if t[j]==t[j-2]:
                eq_p0.append(p0)
            else:
                neq_log_const += math.log(max(p0,1e-300))
                neq_n += 1
    if total==0:
        return 0.0,float('-inf'),0
    ep=np.asarray(eq_p0,dtype=float)
    best_ll=float('-inf'); best_q=0.0
    for q in m.Q_GRID:
        # For non-equality events: p=(1-q)*p0 exactly.
        ll=neq_log_const + neq_n*math.log1p(-float(q))
        if ep.size:
            ll += float(np.log(np.maximum(float(q)+(1.0-float(q))*ep,1e-300)).sum())
        if ll>best_ll+1e-12:
            best_ll=ll; best_q=float(q)
    return best_q,best_ll,total

m.fit_q=fast_fit_q

_CN1_LINES=None
_CN1_SMOKE=False

def _cn1_worker(r):
    sh=m.n1_shuffle(_CN1_LINES,random.Random(m.SEED+900000+r))
    fits=m.fit_fold_models(sh,m.K_PRIMARY)
    q=m.med([d['q_boundary'] for d in fits.values()])
    pred=bool(m.predictive_identity(fits,bootstrap=(100 if _CN1_SMOKE else 2000))['pass'])
    return q,pred

def parallel_calibration_cn1(lines,n=100,smoke=False):
    global _CN1_LINES,_CN1_SMOKE
    _CN1_LINES=lines; _CN1_SMOKE=smoke
    reps=3 if smoke else n
    ctx=mp.get_context('fork')
    with ctx.Pool(processes=2) as pool:
        vals=pool.map(_cn1_worker,range(reps),chunksize=1)
    qs=[q for q,_ in vals]; positive=sum(int(p) for _,p in vals)
    return {'n':reps,'median_q':m.med(qs),'p95_q':m.qtile(qs,.95),'positive_gate_fraction':positive/reps,
            'pass':bool(m.med(qs)<=.02 and positive/reps<=.05)}

m.calibration_cn1=parallel_calibration_cn1

if __name__=='__main__':
    m.main()
