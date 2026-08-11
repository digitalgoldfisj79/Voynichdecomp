# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import numpy as np
from numba import njit
import amadi_driver_v1b as d
m=d.m

@njit(nogil=True,cache=False)
def done_fixed(a,b,c,n,dec,off,adj,t,newv,logp):
    nv=np.int32(newv); dlt=0.0
    for jj in range(off[t],off[t+1]):
        i=adj[jj]; x=a[i]; y=b[i]; z=c[i]
        ox=np.int32(dec[x]); oy=np.int32(dec[y]); oz=np.int32(dec[z])
        nx=nv if x==t else ox; ny=nv if y==t else oy; nz=nv if z==t else oz
        dlt+=n[i]*(logp[int(nx),int(ny),int(nz)]-logp[int(ox),int(oy),int(oz)])
    return dlt
m.done=done_fixed

if __name__=="__main__": m.main()
