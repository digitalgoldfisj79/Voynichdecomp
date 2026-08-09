#!/usr/bin/env python3
import argparse, base64, hashlib, urllib.request, zlib, sys
from pathlib import Path
import numpy as np
ROOT='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/e3a7362adf53d514dd16d37a1a812073357cbd8f/experiments/tranchedino_sta_mixed_unit_v2_0/'
p=Path('/tmp/stage_a1_h20.py');p.write_bytes(urllib.request.urlopen(ROOT+'stage_a1_h20.py',timeout=60).read());sys.path.insert(0,'/tmp');import stage_a1_h20 as m
BASE=ROOT+'model_transport/';NAMES=['model.part0','model.part1','model.part2a','model.part2b','model.part3']
def model():
 b64=''.join(urllib.request.urlopen(BASE+n,timeout=60).read().decode().strip() for n in NAMES);comp=base64.b64decode(b64);assert hashlib.sha256(comp).hexdigest()==m.MODEL_Z_SHA;raw=zlib.decompress(comp);pos=0;n,pos=m.read_varint(raw,pos);idx=[];cnt=[];prev=0
 for _ in range(n):d,pos=m.read_varint(raw,pos);c,pos=m.read_varint(raw,pos);prev+=d;idx.append(prev);cnt.append(c)
 u=[]
 for _ in range(19):x,pos=m.read_varint(raw,pos);u.append(x)
 assert pos==len(raw) and n==24159;tab=np.full((19**3,19),.5)
 for i,c in zip(idx,cnt):tab[i//19,i%19]+=c
 tab/=tab.sum(1,keepdims=True);uc=np.asarray(u,float);up=(uc+.5)/(uc.sum()+9.5);return np.log(tab.ravel()),np.log(up),up
def run(cipher,q,u,pu,name):
 init=m.freq_init(cipher,pu);slots=np.concatenate([np.full(int(x),i,np.int32) for i,x in enumerate(m.MULT)]);bk=None;bs=-1e100;hist=[]
 for r in range(96):
  sd=m.seedint(f'TRANCHSTA22TFIT::{name}::{r}');rng=np.random.default_rng(sd)
  if r==0:k=init.copy()
  elif r%3==1 and bk is not None:
   k=bk.copy()
   for _ in range(2+r%7):i,j=rng.integers(0,36,2);k[i],k[j]=k[j],k[i]
  else:k=rng.permutation(slots).astype(np.int32)
  k,s=m.polish(cipher,k,q,u,m.MULT,12,sd)
  if s>bs:bs=float(s);bk=k.copy()
  if (r+1)%6==0:hist.append([r+1,bs]);print('V22_TFIT',name,r+1,bs,flush=True)
 return bk,bs,hist
def main():
 a=argparse.ArgumentParser();a.add_argument('ensemble');x=a.parse_args();q,u,pu=model();fol,vocab,ids,T,H,C=m.parse_rf();seg,meta,cov,_,_=m.retained_segments(fol,T,ids);cipher=m.flatten(seg);k,s,h=run(cipher,q,u,pu,x.ensemble);freq=np.bincount(cipher[cipher>=0],minlength=36);print('V22_TFIT_RESULT='+__import__('json').dumps({'ensemble':x.ensemble,'score':s,'key':k.tolist(),'history':h,'coverage':cov,'freq':freq.tolist()},separators=(',',':')))
if __name__=='__main__':main()
