# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import math,sys
import numpy as np
from numba import njit
sys.path.insert(0,'experiments/amadi_residuals_v1');sys.path.insert(0,'experiments/amadi_expanded_vbm_v1')
import vbm_typed_v1 as b

@njit(nogil=True)
def anneal_kernel(a,bb,c,n,off,adj,dec,logp,denom,Kc,N,cidx,vidx,props,sd):
    np.random.seed(sd)
    cnt=np.zeros(20,np.int32)
    for t in range(N):cnt[dec[t]]+=1
    raw=b.score_raw(a,bb,c,n,dec,logp)
    best=raw;bestdec=dec.copy()
    sm=0.;nn=0
    for _ in range(96):
        t=np.random.randint(0,N);vals=cidx if t<Kc else vidx;nv=int(vals[np.random.randint(0,len(vals))]);ov=int(dec[t])
        if nv==ov or cnt[ov]<=1:continue
        sm+=abs(b.d_one(a,bb,c,n,dec,off,adj,t,nv,logp)/denom);nn+=1
    t0=max(1e-6,(sm/max(1,nn))*5.)
    for k in range(props):
        t=np.random.randint(0,N);vals=cidx if t<Kc else vidx;nv=int(vals[np.random.randint(0,len(vals))]);ov=int(dec[t])
        if nv==ov or cnt[ov]<=1:continue
        dr=b.d_one(a,bb,c,n,dec,off,adj,t,nv,logp);dn=dr/denom;frac=k/max(1,props-1);temp=max(1e-8,t0*(0.002**frac))
        if dr>=0 or np.random.random()<math.exp(max(-60.,dn/temp)):
            dec[t]=nv;cnt[ov]-=1;cnt[nv]+=1;raw+=dr
            if raw>best:best=raw;bestdec=dec.copy()
    return best/denom,bestdec

def one_restart_fast(S,lm,Kc,tag,ens,rr,props=b.PROPS):
    rng=np.random.default_rng(b.seed(b.NS,tag,ens,rr));d=b.init_dec(S.N,Kc,rng)
    sc,d=anneal_kernel(S.a,S.b,S.c,S.n,S.off,S.adj,d,lm.logtri,S.denom,Kc,S.N,b.CIDX,b.VIDX,props,b.seed(b.NS,'fast',tag,ens,rr))
    raw=sc*S.denom
    cnt=np.bincount(d[:S.N],minlength=20).astype(int)
    order=np.argsort(-S.freq)
    for _ in range(4):
        changed=False
        for tt in order:
            t=int(tt);old=int(d[t]);bestd=0.;bestv=old
            for vv in b.allowed(t,Kc):
                nv=int(vv)
                if nv==old or cnt[old]<=1:continue
                dr=b.d_one(S.a,S.b,S.c,S.n,d,S.off,S.adj,t,nv,lm.logtri)
                if dr>bestd+1e-10:bestd=dr;bestv=nv
            if bestv!=old:
                d[t]=bestv;cnt[old]-=1;cnt[bestv]+=1;raw+=bestd;changed=True
        if not changed:break
    return raw/S.denom,d

b.one_restart=one_restart_fast
if __name__=='__main__':b.main()
