#!/usr/bin/env python3
import urllib.request
import numpy as np

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/be4d9d85503cc0d616636f207e93ac5dc06d5c59/experiments/bnf_m19_sta_hierarchy_v1_7/run_v17.py'
src=urllib.request.urlopen(BASE,timeout=120).read().decode('utf-8')
b={'__name__':'v17base'}
exec(compile(src,'run_v17.py','exec'),b)

def choose_span_v17(pool,n,tag):
    pos=[i for i,c in enumerate(pool) if c!=' ']
    if len(pos)<n:raise RuntimeError(('pool short',tag,len(pos),n))
    st=b['seed']('span',tag)%(len(pos)-n+1)
    a=pos[st];z=pos[st+n-1]+1
    return pool[a:z].strip()

# The frozen generator reaches its span helper through the imported base namespace.
b['ns']['choose_span']=choose_span_v17

def delta_selftest():
    rng=np.random.default_rng(b['seed']('delta-selftest'))
    for K in [19,23,26,36,38]:
        B=rng.integers(0,20,size=(K,K),dtype=np.int64)
        st=rng.integers(0,20,size=K,dtype=np.int64)
        en=rng.integers(0,20,size=K,dtype=np.int64)
        freq=rng.integers(1,50,size=K,dtype=np.int64)
        S={'B':B,'st':st,'en':en,'freq':freq,'denom':int(B.sum()+st.sum()+en.sum()+freq.sum())}
        comp=(rng.normal(size=(b['NV'],b['NV'])),rng.normal(size=b['NV']),rng.normal(size=b['NV']))
        m=b['init_map'](K,rng)
        for _ in range(100):
            x,ch=b['proposal'](m,rng)
            d=b['delta_score'](S,m,x,ch,comp)
            full=b['score_num'](S,x,comp)-b['score_num'](S,m,comp)
            if abs(d-full)>1e-10:raise RuntimeError(('delta selftest',K,d,full,ch))
            if rng.random()<.5:m=x
    print('DELTA_SELFTEST PASS',flush=True)

delta_selftest()
b['main']()
