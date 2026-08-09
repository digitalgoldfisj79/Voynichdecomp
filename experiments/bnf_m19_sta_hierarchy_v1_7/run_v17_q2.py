#!/usr/bin/env python3
import urllib.request,hashlib,math
import numpy as np

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/be4d9d85503cc0d616636f207e93ac5dc06d5c59/experiments/bnf_m19_sta_hierarchy_v1_7/run_v17.py'
b={'__name__':'v17base'}
exec(compile(urllib.request.urlopen(BASE,timeout=120).read().decode(),'run_v17.py','exec'),b)

QNS='M19STAv17Q2'
def qseed(*parts):
    h=hashlib.sha256(('::'.join([QNS]+list(map(str,parts)))).encode()).digest()
    return int.from_bytes(h[:8],'big') & 0xffffffff
b['seed']=qseed

# Amendment 003: choose first deterministic fresh dev+test span capable of emitting all 19 values.
def support_span(pool,n,tag):
    if not (isinstance(tag,tuple) and len(tag)>=3 and tag[0]=='v17qual'):
        # No binding call other than v17 controls is expected, but keep deterministic semantics.
        pos=[i for i,c in enumerate(pool) if c!=' ']
        if len(pos)<n:raise RuntimeError(('pool short',tag,len(pos),n))
        st=qseed('span-generic',tag)%(len(pos)-n+1);return pool[pos[st]:pos[st+n-1]+1].strip()
    la,K=tag[1],int(tag[2]);pos=[i for i,c in enumerate(pool) if c!=' ']
    if len(pos)<n:raise RuntimeError(('pool short',tag,len(pos),n))
    for attempt in range(1000):
        st=qseed('span-support',la,K,attempt)%(len(pos)-n+1);span=pool[pos[st]:pos[st+n-1]+1].strip();chars=set(c for c in span if c!=' ');vals=set()
        for c in chars:
            vals.update(b['V2I'][v] for v in b['LETTER_VALS'][b['A2I'][c]])
        if len(vals)==b['NV']:
            print('SUPPORT_SPAN',la,K,'attempt',attempt,'letters',''.join(sorted(chars)),flush=True);return span
    raise RuntimeError(('no support-complete span',la,K))
b['ns']['choose_span']=support_span

# Amendment 004: same proposal kernel/search budget, exact vectorized full rescoring.
def optimize_full(S,comp,tag,K):
    steps=26000 if K<=26 else 40000;restarts=6 if K<=26 else 8;best=(-1e100,None)
    for rr in range(restarts):
        rng=np.random.default_rng(qseed('opt',tag,rr));m=b['init_map'](K,rng);s=b['score_num'](S,m,comp)
        ds=[]
        for _ in range(50):
            x,ch=b['proposal'](m,rng);ds.append(abs(b['score_num'](S,x,comp)-s))
        t0=max(1e-6,float(np.median(ds))*4);local_best=(s,m.copy())
        for k in range(steps):
            frac=k/max(1,steps-1);temp=max(1e-7,t0*(0.01**frac));x,ch=b['proposal'](m,rng);s2=b['score_num'](S,x,comp);d=s2-s
            if d>=0 or rng.random()<math.exp(max(-50,d/temp)):
                m=x;s=s2
                if s>local_best[0]:local_best=(s,m.copy())
        m=local_best[1].copy();s=b['score_num'](S,m,comp)
        for _ in range(8):
            bd=1e-12;bx=None;cnt=np.bincount(m,minlength=b['NV'])
            for a in range(K):
                for bb in range(a+1,K):
                    if m[a]==m[bb]:continue
                    x=m.copy();x[a],x[bb]=x[bb],x[a];s2=b['score_num'](S,x,comp);d=s2-s
                    if d>bd:bd=d;bx=x
            if np.any(cnt==2) and np.any(cnt==1):
                for sv in np.flatnonzero(cnt==2):
                    for dv in np.flatnonzero(cnt==1):
                        for i in np.flatnonzero(m==sv):
                            x=m.copy();x[i]=dv;s2=b['score_num'](S,x,comp);d=s2-s
                            if d>bd:bd=d;bx=x
            if bx is None:break
            m=bx;s=b['score_num'](S,m,comp)
        if s>best[0]:best=(s,m.copy())
    assert b['valid_map'](best[1],K)
    return best
b['optimize']=optimize_full

# Numerical equivalence self-test before any source/language scoring.
def selftest():
    rng=np.random.default_rng(qseed('selftest'))
    for K in [19,22,26,36,38]:
        S={'B':rng.integers(0,20,(K,K),dtype=np.int64),'st':rng.integers(0,20,K,dtype=np.int64),'en':rng.integers(0,20,K,dtype=np.int64),'freq':rng.integers(1,50,K,dtype=np.int64)}
        S['denom']=int(S['B'].sum()+S['st'].sum()+S['en'].sum()+S['freq'].sum());comp=(rng.normal(size=(b['NV'],b['NV'])),rng.normal(size=b['NV']),rng.normal(size=b['NV']));m=b['init_map'](K,rng)
        for _ in range(100):
            x,ch=b['proposal'](m,rng);d=b['delta_score'](S,m,x,ch,comp);f=b['score_num'](S,x,comp)-b['score_num'](S,m,comp)
            if abs(d-f)>1e-10:raise RuntimeError(('Q2 score selftest',K,d,f))
            m=x
    print('Q2_SCORE_SELFTEST PASS',flush=True)

selftest()
b['main']()
