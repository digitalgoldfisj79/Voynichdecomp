#!/usr/bin/env python3
import urllib.request
import numpy as np
from collections import Counter
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0ccea68e5eef0b551cff7cb2703c20c9868e294c/experiments/bnf_free_switch_m19_v0_7/run_m19.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
ns={'__name__':'m19_base'}
exec(compile(src,'run_m19.py','exec'),ns)

def generate_control_amended(plain,lang,rep):
    for attempt in range(1000):
        rng=np.random.default_rng(ns['seed']('values',lang,rep,attempt));vals=[]
        for c in plain:
            if c==' ':vals.append(None)
            else:
                vs=ns['LETTER_VALS'][ns['A2I'][c]]
                vals.append(ns['V2I'][int(rng.choice(vs))])
        cnt=Counter();letters=0
        for v in vals:
            if v is None:continue
            if letters<ns['TRAIN']:cnt[v]+=1
            letters+=1
        dup=[v for v,_ in sorted(cnt.items(),key=lambda kv:(-kv[1],kv[0]))[:6]]
        if len(dup)!=6:continue
        rawforms={v:[v] for v in range(ns['NV'])}
        for j,v in enumerate(dup):rawforms[v].append(ns['NV']+j)
        perm=np.arange(25);rng2=np.random.default_rng(ns['seed']('opaque',lang,rep,attempt));rng2.shuffle(perm)
        raw2surf={raw:int(perm[raw]) for raw in range(25)};surf2val=np.full(25,-1,np.int16)
        for v,forms in rawforms.items():
            for raw in forms:surf2val[raw2surf[raw]]=v
        out=[];letters=0;used=set();rng3=np.random.default_rng(ns['seed']('surface',lang,rep,attempt))
        for v in vals:
            if v is None:out.append(' ');continue
            raw=int(rng3.choice(rawforms[v]));sid=raw2surf[raw];out.append(chr(65+sid))
            if letters<ns['TRAIN']:used.add(sid)
            letters+=1
        if len(used)==25:
            print('CONTROL_GENERATION',lang,rep,'attempt',attempt,flush=True)
            assert ns['valid_map'](surf2val)
            return ''.join(out),surf2val
    raise RuntimeError(('control rejection exhausted',lang,rep))

ns['generate_control']=generate_control_amended
ns['main']()
