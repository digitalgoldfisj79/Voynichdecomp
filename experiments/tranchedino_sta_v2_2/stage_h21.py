#!/usr/bin/env python3
from __future__ import annotations
import base64, hashlib, json, urllib.request, zlib, sys
from pathlib import Path
import numpy as np
ROOT='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/e3a7362adf53d514dd16d37a1a812073357cbd8f/experiments/tranchedino_sta_mixed_unit_v2_0/'
p=Path('/tmp/stage_a1_h20.py');p.write_bytes(urllib.request.urlopen(ROOT+'stage_a1_h20.py',timeout=60).read());sys.path.insert(0,'/tmp');import stage_a1_h20 as m
BASE=ROOT+'model_transport/';NAMES=['model.part0','model.part1','model.part2a','model.part2b','model.part3']
CANON=np.asarray([4,0,4,18,16,17,8,17,15,3,10,8,18,10,16,11,15,9,9,3,11,12,13,7,13,2,12,1,14,6,6,5,14,7,5,1],dtype=np.int32)
P05=-2.3672276834921244

def model():
    b64=''.join(urllib.request.urlopen(BASE+n,timeout=60).read().decode().strip() for n in NAMES)
    comp=base64.b64decode(b64);assert hashlib.sha256(comp).hexdigest()==m.MODEL_Z_SHA
    raw=zlib.decompress(comp);pos=0;n,pos=m.read_varint(raw,pos);idx=[];cnt=[];prev=0
    for _ in range(n):
        d,pos=m.read_varint(raw,pos);c,pos=m.read_varint(raw,pos);prev+=d;idx.append(prev);cnt.append(c)
    u=[]
    for _ in range(19):x,pos=m.read_varint(raw,pos);u.append(x)
    assert pos==len(raw) and n==24159
    tab=np.full((19**3,19),.5,dtype=np.float64)
    for i,c in zip(idx,cnt):tab[i//19,i%19]+=c
    tab/=tab.sum(axis=1,keepdims=True);uc=np.asarray(u,float);up=(uc+.5)/(uc.sum()+.5*19)
    return np.log(tab.ravel()),np.log(up)

def seedint(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'big')&0x7fffffff

def score_shuffle(segs,meta,key,q,u,r):
    out=[]
    for s,(f,si,sub) in zip(segs,meta):
        a=np.asarray(s,np.int32).copy();rng=np.random.default_rng(seedint(f'TRANCHSTA21Hshuffle::{r}::{f}::{si}:{sub}'));rng.shuffle(a);out.append(a.tolist())
    return float(m.score_key(m.flatten(out),key,q,u))

def main():
    q,u=model();fol,vocab,ids,T,H20,C20=m.parse_rf()
    protected=sorted(C20,key=lambda f:hashlib.sha256(('TRANCHSTA21split::'+f).encode()).digest())
    H21=protected[:34];C21=protected[34:]
    assert len(H21)==34 and len(C21)==34 and set(H21).isdisjoint(C21)
    seg,meta,cov,alln,ret=m.retained_segments(fol,H21,ids);cipher=m.flatten(seg)
    obs=float(m.score_key(cipher,CANON,q,u))
    null=np.asarray([score_shuffle(seg,meta,CANON,q,u,r) for r in range(200)],dtype=float)
    med=float(np.median(null));q99=float(np.quantile(null,.99))
    buckets=[]
    for bi in range(4):
        fs=[f for f in H21 if hashlib.sha256(('TRANCHSTA21Hbucket::'+f).encode()).digest()[0]%4==bi]
        bseg,bmeta,bcov,ball,bret=m.retained_segments(fol,fs,ids);bcipher=m.flatten(bseg)
        bo=float(m.score_key(bcipher,CANON,q,u));bn=np.asarray([score_shuffle(bseg,bmeta,CANON,q,u,r) for r in range(200)],dtype=float)
        bmed=float(np.median(bn));buckets.append({'bucket':bi,'folios':len(fs),'coverage':bcov,'observed':bo,'null_median':bmed,'delta':bo-bmed})
    gate=bool(cov>=.97 and obs>=P05 and obs>q99 and all(x['delta']>0 for x in buckets))
    payload={'H21_folios':H21,'C21_folios_sha256':hashlib.sha256(('\n'.join(C21)).encode()).hexdigest(),'coverage':cov,'all_chars':alln,'retained':ret,'segments':len(seg),'observed':obs,'positive_control_p05':P05,'absolute_floor_delta':obs-P05,'null_median':med,'null_q99':q99,'observed_minus_q99':obs-q99,'buckets':buckets,'gate':gate,'verdict':'H21 TRANCHEDINO-STA CANDIDATE' if gate else 'NO TRANCHEDINO-STA ALPHABETIC SIGNAL'}
    print('H21_RESULT='+json.dumps(payload,separators=(',',':')),flush=True)
if __name__=='__main__':main()
