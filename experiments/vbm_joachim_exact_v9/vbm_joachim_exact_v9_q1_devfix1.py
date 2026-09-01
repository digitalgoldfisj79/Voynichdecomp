#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, importlib.util, sys, tempfile, urllib.request, collections
import numpy as np

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/ac988685e602363286aee75467cc2113af00be28/experiments/vbm_joachim_exact_v9/vbm_joachim_exact_v9_q1.py'
p=tempfile.NamedTemporaryFile(suffix='.py',delete=False).name
urllib.request.urlretrieve(BASE,p)
spec=importlib.util.spec_from_file_location('q1base',p);q=importlib.util.module_from_spec(spec);sys.modules['q1base']=q;spec.loader.exec_module(q)

# Protocol-consistent operational fix: build maximal representable event segments
# instead of discarding an entire sentence because one nucleus is outside the
# frozen 64-state vocabulary or exceeds the 1-5 consonant limit.
def loose_events(words):
    s=''.join(q.norm_token(w) for w in words);out=[];run=''
    if not s:return []
    for c in s:
        if c in q.VSET:
            if run:out.append(run);run=''
            out.append(c)
        else:run+=c
    if run:out.append(run)
    return out

def segments(ev,nset):
    out=[];cur=[]
    for x in ev:
        bad=(x not in q.VSET and (len(x)>5 or x not in nset))
        if bad:
            if len(cur)>=4:out.append(cur)
            cur=[]
        else:cur.append(x)
    if len(cur)>=4:out.append(cur)
    return out

def build_language(name,sents):
    nuc=collections.Counter();rawtrain=[]
    for i,ws in enumerate(sents):
        if i%10 not in range(0,6):continue
        ev=loose_events(ws)
        if not ev:continue
        rawtrain.append(ev)
        for x in ev:
            if x not in q.VSET and 1<=len(x)<=5:nuc[x]+=1
    nuclei=[x for x,_ in sorted(nuc.items(),key=lambda kv:(-kv[1],kv[0]))[:q.NCAND]];nset=set(nuclei)
    sem_names=list(q.VOWELS)+nuclei;sid={x:i for i,x in enumerate(sem_names)};K=len(sem_names);B=K
    cls=np.array([0]*5+[1]*len(nuclei),np.int8)
    cost=np.array([q.math.log2(5)]*5+[q.math.log2(5)+len(x)*q.math.log2(21) for x in nuclei],float)
    C=np.full((K+1,K+1,K+1),0.25,dtype=np.float64);F=np.full(K,0.25,dtype=np.float64);used=0;nsegments=0
    for ev in rawtrain:
        for sg in segments(ev,nset):
            a=[sid[x] for x in sg];used+=len(a);nsegments+=1
            for x in a:F[x]+=1
            z=[B,B]+a+[B,B]
            for x,y,zv in zip(z,z[1:],z[2:]):C[x,y,zv]+=1
    C/=C.sum(axis=2,keepdims=True);F/=F.sum()
    pools={6:[],7:[],8:[],9:[]}
    for i,ws in enumerate(sents):
        r=i%10
        if r not in pools:continue
        for sg in segments(loose_events(ws),nset):pools[r].append(np.array([sid[x] for x in sg],np.int16))
    if used<10000 or min(sum(len(z) for z in pools[r]) for r in (6,7,8))<10000:raise RuntimeError((name,'insufficient segmented corpus',used,{r:sum(len(z) for z in v) for r,v in pools.items()}))
    L=q.Lang(name,nuclei,sem_names,cls,cost,np.log(C),F,pools)
    L._segment_meta={'lm_events':used,'lm_segments':nsegments,'pool_events':{r:sum(len(z) for z in v) for r,v in pools.items()}}
    return L

q.build_language=build_language

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['DEV','CAL','VAL'],required=True);args=ap.parse_args();q.run_phase(args.phase)
