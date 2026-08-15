#!/usr/bin/env python3
"""Target-blind engineering checks for STA line-zone null v0.2."""
import importlib.util, itertools, random
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
EXEC=HERE/'run_line_zone_null_exec.py'
spec=importlib.util.spec_from_file_location('lzexec',EXEC)
e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
m=e.m


def toy_line(tokens):
    return {'folio':'toy','section':'T','tokens':tuple(tokens)}


def exhaustive_counts(tokens):
    n=len(tokens); groups=m.p.groups_for(n,'N1')
    gp=[]
    for g in groups:
        pos=[int(i) for i in g]
        gp.append(list(itertools.permutations(pos)))
    vals=[]
    for combo in itertools.product(*gp):
        order=list(range(n))
        for g,perm in zip(groups,combo):
            pos=[int(i) for i in g]
            for dst,src in zip(pos,perm): order[dst]=src
        t=tuple(tokens[k] for k in order)
        vals.append(m.counts_by_lag([toy_line(t)]))
    means={lag:np.mean([v['lag'][lag] for v in vals]) for lag in m.LAGS}
    return means,len(vals)


def main():
    # 1. Exact line-zone preservation.
    lines=[toy_line(('A','B','A','C','B','A')),toy_line(('X','X','Y','Z'))]
    assert m.preservation_audit(lines,n_lines=2,n_rep=20)

    # 2. Monte Carlo permutation means agree with exhaustive index-permutation means on a small toy line.
    ex,n=exhaustive_counts(lines[0]['tokens'])
    ref=m.permutation_counts([lines[0]],12000,m.SEED+123)
    for lag in m.LAGS:
        mu=float(np.mean(ref['lag'][lag]))
        assert abs(mu-ex[lag])<0.04,(lag,mu,ex[lag])

    # 3. Planted swap preserves each line-zone multiset and creates positive lag-2 signal.
    t=('A','B','A','C','B','A')
    before=m.line_zone_counters(t); c0=m.line_match_counts(t)[2]
    planted,ns=e.fixed_plant_line_lag2(t,random.Random(7),q=1.0)
    assert ns>=1,(planted,ns)
    assert before==m.line_zone_counters(planted)
    assert m.line_match_counts(planted)[2]>c0,(t,planted,c0,m.line_match_counts(planted)[2])

    # 4. Full lag-2 gate behaves as specified on a fabricated joint reference distribution.
    rng=np.random.default_rng(99)
    R=10000
    ref2={'lag':{1:rng.poisson(100,R),2:rng.poisson(100,R),3:rng.poisson(100,R),4:rng.poisson(100,R)},
          'lag2_left':rng.poisson(50,R),'lag2_right':rng.poisson(50,R)}
    obs={'lag':{1:100,2:150,3:100,4:100},'events':{1:1000,2:1000,3:1000,4:1000},'lag2_left':75,'lag2_right':75}
    s=m.summarize_obs(obs,ref2)
    assert s['full_lag2_specific_gate'],s

    print('BUILD_CHECK_PASS',{'exhaustive_states':n,'toy_exact_means':ex,'planted_swaps':ns,'fabricated_z2':s['lags']['2']['z']})

if __name__=='__main__': main()
