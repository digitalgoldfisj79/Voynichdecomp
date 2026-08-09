#!/usr/bin/env python3
import argparse, base64, hashlib, urllib.request, zlib
import numpy as np
import stage_a1_h20 as m

MODEL_COMMIT='28fe88d1c1877c588ff7b7262efd3012c357bb76'
BASE=f'https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/{MODEL_COMMIT}/experiments/tranchedino_sta_mixed_unit_v2_0/model_transport/'
NAMES=['model.part0','model.part1','model.part2a','model.part2b','model.part3']

def load_model():
    b64=''.join(urllib.request.urlopen(BASE+n,timeout=60).read().decode().strip() for n in NAMES)
    comp=base64.b64decode(b64);assert hashlib.sha256(comp).hexdigest()==m.MODEL_Z_SHA
    raw=zlib.decompress(comp);pos=0;n,pos=m.read_varint(raw,pos);idx=[];cnt=[];prev=0
    for _ in range(n):
        d,pos=m.read_varint(raw,pos);c,pos=m.read_varint(raw,pos);prev+=d;idx.append(prev);cnt.append(c)
    uni=[]
    for _ in range(19):u,pos=m.read_varint(raw,pos);uni.append(u)
    assert pos==len(raw) and n==24159
    tab=np.full((19**3,19),.5,dtype=np.float64)
    for i,c in zip(idx,cnt):tab[i//19,i%19]+=c
    tab/=tab.sum(axis=1,keepdims=True);uc=np.asarray(uni,float);up=(uc+.5)/(uc.sum()+.5*19)
    return np.log(tab.ravel()),np.log(up),up

def run(cipher,quad,uni,puni,name,restarts):
    init=m.freq_init(cipher,puni);slot=np.concatenate([np.full(int(x),i,np.int32) for i,x in enumerate(m.MULT)]);bestk=None;best=-1e100;hist=[]
    for r in range(restarts):
        sd=m.seedint(f'TRANCHSTA20TFIT::{name}::{r}');rng=np.random.default_rng(sd)
        if r==0:k=init.copy()
        elif r%3==1 and bestk is not None:
            k=bestk.copy()
            for _ in range(2+r%7):i,j=rng.integers(0,36,2);k[i],k[j]=k[j],k[i]
        else:k=rng.permutation(slot).astype(np.int32)
        k,sc=m.polish(cipher,k,quad,uni,m.MULT,12,sd)
        if sc>best:best=float(sc);bestk=k.copy()
        if (r+1)%6==0:hist.append([r+1,best]);print('TFIT',name,r+1,best,flush=True)
    return bestk,best,hist

def main():
    ap=argparse.ArgumentParser();ap.add_argument('ensemble');ap.add_argument('--restarts',type=int,default=72);a=ap.parse_args()
    quad,uni,puni=load_model();fol,vocab,ids,T,H,C=m.parse_rf();seg,meta,cov,_,_=m.retained_segments(fol,T,ids);cipher=m.flatten(seg)
    k,s,h=run(cipher,quad,uni,puni,a.ensemble,a.restarts);freq=np.bincount(cipher[cipher>=0],minlength=36)
    print('TFIT_RESULT='+__import__('json').dumps({'ensemble':a.ensemble,'score':s,'key':k.tolist(),'history':h,'T_coverage':cov,'freq':freq.tolist()},separators=(',',':')),flush=True)
if __name__=='__main__':main()
