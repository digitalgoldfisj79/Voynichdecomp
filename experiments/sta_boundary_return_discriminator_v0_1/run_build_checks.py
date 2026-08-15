#!/usr/bin/env python3
"""Target-blind build checks for STA boundary-return discriminator v0.1."""
import importlib.util, random
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('disc',HERE/'run_discriminator.py')
d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)

assert d.boundary_eligible_target(2,10)
assert d.boundary_eligible_target(3,10)
assert not d.boundary_eligible_target(4,10)
assert d.boundary_eligible_target(8,10)
assert d.boundary_eligible_target(9,10)

# Fully synthetic corpus: no RF1b/Voynich target data are read here.
rng=random.Random(12345)
vocab=['AB','AC','AD','BA','BC','BD','CA','CB','CD','DA','DB','DC']
lines=[]
for f in range(60):
    for l in range(4):
        n=8+(f+l)%5
        t=[rng.choice(vocab) for _ in range(n)]
        lines.append({'folio':f'syn{f:03d}','section':'SYN','tokens':tuple(t)})

prot,a=d.kmodes(lines,8)
assert len(prot)==8 and len(a)==len(lines)
fm=d.FoldModel(lines,8)
q0,_,n0=d.fit_q(fm,lines,True)
assert n0>100

# Plant a known boundary-only return signal and verify the detector moves upward.
planted=[]; rng=random.Random(67890)
for x in lines:
    t=list(x['tokens']); n=len(t)
    for j in range(2,n):
        if d.boundary_eligible_target(j,n) and rng.random()<0.12:
            t[j]=t[j-2]
    planted.append({'folio':x['folio'],'section':x['section'],'tokens':tuple(t)})
fm1=d.FoldModel(planted,8)
q1,_,n1=d.fit_q(fm1,planted,True)
assert n1==n0
assert q1>q0, (q0,q1)

s0=d.score(lines); s1=d.score(planted)
assert s1['E2_N0']>s0['E2_N0']
print('BUILD_CHECK_PASS',{'q_unplanted':q0,'q_planted':q1,'E2_unplanted':s0['E2_N0'],'E2_planted':s1['E2_N0']})
