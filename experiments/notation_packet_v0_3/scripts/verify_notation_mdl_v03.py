#!/usr/bin/env python3
import pickle,importlib.util,sys,math,json
from collections import defaultdict,Counter
import numpy as np
from scipy.special import gammaln
spec=importlib.util.spec_from_file_location('m','/mnt/data/run_notation_mdl_fast_v03.py');m=importlib.util.module_from_spec(spec);sys.modules['m']=m;spec.loader.exec_module(m)
R=pickle.load(open(m.DATA,'rb'));S=m.make_segs(R);alphabet=sorted(set(''.join(r['token'] for r in R)));N=len(R)
ref=json.load(open('/mnt/data/voynich_notation_v0_3/mdl_fast_results_v0_3.json'))
for name in ['UNSPLIT','P70_lossless','No_suffix','Flat_no_gallows']:
 parts=S[name];voc=[set(x[j] for x in parts) for j in range(len(parts[0]))];dc=m.dict_code(voc,alphabet)
 counts=defaultdict(Counter)
 for r,p in zip(R,parts):
  for j,v in enumerate(p):counts[m.context(j,p,r,'independent')][v]+=1
 alpha=.5;bits=dc
 for ctx,c in counts.items():
  V=len(voc[ctx[0]]);n=sum(c.values())
  lp=gammaln(alpha*V)-gammaln(n+alpha*V)+sum(gammaln(x+alpha)-gammaln(alpha) for x in c.values())
  bits-=lp/math.log(2)
 target=[x for x in ref['summary'] if x['code']=='exact_independent' and x['alpha']==.5 and x['segmentation']==name][0]['mean_bpt']
 print(name,bits/N,target,(bits/N-target))
 assert abs(bits/N-target)<1e-9
print('PASS')
