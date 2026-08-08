#!/usr/bin/env python3
import urllib.request
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c615b94c03ca35569e8d582e995791fe40457490/experiments/bnf_onomancy_folio_pilot_v0_1/run_pilot.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
ns={'__name__':'bnf_pilot_base'}
exec(compile(src,'run_pilot.py','exec'),ns)
np=ns['np']; ALPH=ns['ALPH']; A2I=ns['A2I']; N=ns['N']; normalize_plain=ns['normalize_plain']

def build_lm_fast(sents, keep_space):
    K=N+1 if keep_space else N
    V=K**4
    idx_chunks=[]
    unig=np.ones(N,dtype=np.float64)*0.1
    chars=0
    for raw in sents:
        s=normalize_plain(raw,keep_space=keep_space)
        if len(s)<4: continue
        arr=[]
        for c in s:
            if c==' ' and keep_space: arr.append(N)
            elif c in A2I:
                arr.append(A2I[c]); unig[A2I[c]] += 1
        a=np.asarray(arr,dtype=np.int64)
        if len(a)<4: continue
        idx=((a[:-3]*K+a[1:-2])*K+a[2:-1])*K+a[3:]
        idx_chunks.append(idx)
        chars += len(a)
    all_idx=np.concatenate(idx_chunks) if idx_chunks else np.empty(0,dtype=np.int64)
    counts=np.bincount(all_idx,minlength=V).astype(np.float64)
    alpha=0.05
    logp=np.log((counts+alpha)/(counts.sum()+alpha*V))
    unig/=unig.sum()
    return logp,K,unig,chars

ns['build_lm']=build_lm_fast
ns['main']()
