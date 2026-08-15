#!/usr/bin/env python3
"""Performance-only runner for the frozen STA boundary-return discriminator v0.1.

Scientific protocol is unchanged. This wrapper:
1. evaluates the identical q grid / conditional likelihood using the algebraic
   separation of equality and non-equality events, avoiding 501 full-vector scans;
2. vectorizes only k-modes Hamming-distance calculations while preserving the
   frozen farthest-first initialization, tie-breaking and modal updates;
3. runs independent CN1 shuffled controls in two forked worker processes.

The original run_discriminator.py remains the binding specification and is also
running independently. Outputs should be deterministic-equivalent at the grid-q,
cluster-assignment, gate and verdict levels.
"""
import collections, importlib.util, math, multiprocessing as mp, random
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE/'run_discriminator.py'
spec=importlib.util.spec_from_file_location('frozen_discriminator',BASE)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

ORIG_FIT_Q=m.fit_q
ORIG_KMODES=m.kmodes

def fast_kmodes(lines,K):
    feats=[m.boundary_feature(x) for x in lines]
    counts=collections.Counter(feats); uniq=sorted(counts)
    if not uniq:
        return [],[]
    K=min(K,len(uniq)); mx=max(counts.values())
    first=min(v for v in uniq if counts[v]==mx); prot=[first]
    uarr=np.asarray(uniq,dtype='U8')
    uindex={v:i for i,v in enumerate(uniq)}
    while len(prot)<K:
        parr=np.asarray(prot,dtype='U8')
        md=(uarr[:,None,:]!=parr[None,:,:]).sum(axis=2).min(axis=1)
        for z in prot:
            md[uindex[z]]=-1
        maxd=int(md.max())
        # uniq is lexicographically sorted, so first eligible max-distance item
        # is exactly the original min(v ...) tie-break.
        ix=int(np.flatnonzero(md==maxd)[0])
        prot.append(uniq[ix])
    arr=np.asarray(feats,dtype='U8')
    assign=None
    for _ in range(30):
        parr=np.asarray(prot,dtype='U8')
        dist=(arr[:,None,:]!=parr[None,:,:]).sum(axis=2)
        new=np.argmin(dist,axis=1).astype(int).tolist()  # first min = lowest z
        if assign==new:
            break
        assign=new
        aa=np.asarray(assign,dtype=int)
        for z in range(K):
            idx=np.flatnonzero(aa==z)
            if not len(idx):
                continue
            nv=[]
            for col in range(arr.shape[1]):
                c=collections.Counter(arr[idx,col].tolist()); mm=max(c.values())
                nv.append(min(k for k,v in c.items() if v==mm))
            prot[z]=tuple(nv)
    return prot,assign

m.kmodes=fast_kmodes

def fast_fit_q(model, lines, eligible_only=True):
    eq_p0=[]; neq_log_const=0.0; neq_n=0; total=0
    for x in lines:
        t=x['tokens']; n=len(t); sec=x['section']
        for j in range(2,n):
            if eligible_only and not m.boundary_eligible_target(j,n):
                continue
            p0=model.b2_marginal_prob(sec,n,m.p.edge_pos(j,n),t[j]); total+=1
            if t[j]==t[j-2]:
                eq_p0.append(p0)
            else:
                neq_log_const += math.log(max(p0,1e-300)); neq_n+=1
    if total==0:
        return 0.0,float('-inf'),0
    ep=np.asarray(eq_p0,dtype=float); best_ll=float('-inf'); best_q=0.0
    for q in m.Q_GRID:
        q=float(q)
        ll=neq_log_const + neq_n*math.log1p(-q)
        if ep.size:
            ll += float(np.log(np.maximum(q+(1.0-q)*ep,1e-300)).sum())
        if ll>best_ll+1e-12:
            best_ll=ll; best_q=q
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
    _CN1_LINES=lines; _CN1_SMOKE=smoke; reps=3 if smoke else n
    ctx=mp.get_context('fork')
    with ctx.Pool(processes=2) as pool:
        vals=pool.map(_cn1_worker,range(reps),chunksize=1)
    qs=[q for q,_ in vals]; positive=sum(int(p) for _,p in vals)
    return {'n':reps,'median_q':m.med(qs),'p95_q':m.qtile(qs,.95),'positive_gate_fraction':positive/reps,
            'pass':bool(m.med(qs)<=.02 and positive/reps<=.05)}

m.calibration_cn1=parallel_calibration_cn1

if __name__=='__main__':
    m.main()
