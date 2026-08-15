#!/usr/bin/env python3
import importlib.util, itertools, math, random
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fm',HERE/'run_fixed_marginal_v0_1.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def exhaustive(src,tar):
    vals=[]
    for q in set(itertools.permutations(tar)):
        vals.append(sum(a==b for a,b in zip(src,q)))
    mu=sum(vals)/len(vals); var=sum((x-mu)**2 for x in vals)/len(vals)
    return mu,var

def rows(src,tar):
    return [{'source':a,'target':b,'eq':a==b} for a,b in zip(src,tar)]

cases=[(['a','b'],['a','b']),(['a','a','b'],['a','b','b']),(['a','a','b','c'],['a','b','b','c'])]
for s,t in cases:
    fake=rows(s,t); o,mu,var,_,_=m.stratum_moments(fake); em,ev=exhaustive(s,t)
    assert abs(mu-em)<1e-12 and abs(var-ev)<1e-12,(s,t,mu,var,em,ev)

# Toy fixed-marginal panel: target shuffling must preserve both endpoint marginals exactly.
ev=[]
for i,(s,t) in enumerate(zip(['a','a','b','b','c','c'],['a','b','b','c','c','a'])):
    ev.append({'folio':'f1','fold':0,'section':'T','n':8,'lb':'6-9','si':'L0','tj':'L2','source':s,'target':t,'eq':s==t,'z':0,'left':True,'right':False,'line':i})
summary=m.summarize(ev,'D2')
assert m.marginal_audit(summary,123)

# Strong synthetic identity coupling must exceed the same fixed marginals' expectation.
ev2=[]
for i in range(200):
    s=['a','b','c','d'][i%4]
    t=s if i<120 else ['a','b','c','d'][(i+1)%4]
    ev2.append({'folio':f'f{i%20}','fold':i%5,'section':'T','n':8,'lb':'6-9','si':'L0','tj':'L2','source':s,'target':t,'eq':s==t,'z':0,'left':True,'right':False,'line':i})
s2=m.summarize(ev2,'D2')
assert s2['ratio']>1.5 and s2['z']>2.58,(s2['ratio'],s2['z'])
print('BUILD_CHECK_PASS',{'toy_mu':summary['expected'],'identity_ratio':s2['ratio'],'identity_z':s2['z']})
