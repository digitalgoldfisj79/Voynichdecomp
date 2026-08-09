#!/usr/bin/env python3
from __future__ import annotations
import base64, collections, hashlib, json, math, re, urllib.request, zlib
import numpy as np
from numba import njit

RF_URL='https://www.voynich.nu/data/sta/RF1b.txt'
RF_SHA='81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17'
MODEL_COMMIT='7847af568f95173fd2f514f8fd31ece8372017fa'
MODEL_BASE=f'https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/{MODEL_COMMIT}/experiments/tranchedino_sta_mixed_unit_v2_0/model_transport/model.part'
MODEL_Z_SHA='b3f56ce629172cb3825b2312b608fed149dbadd2a553dda5b2c401f54642bc8f'
P05=-2.3672276834921244
MULT=np.array([1,2,1]+[2]*16,dtype=np.int32)
assert MULT.sum()==36 and len(MULT)==19
HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8','Accept-Language':'en-GB,en;q=0.9','Referer':'https://www.voynich.nu/extra/sta.html'}

def get(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url,headers=headers or {'User-Agent':'Mozilla/5.0'}),timeout=60).read()

def read_varint(buf,pos):
    n=0;shift=0
    while True:
        b=buf[pos];pos+=1;n|=(b&127)<<shift
        if not b&128:return n,pos
        shift+=7

def load_model():
    b64=''.join(get(MODEL_BASE+str(i)).decode().strip() for i in range(4))
    comp=base64.b64decode(b64)
    assert hashlib.sha256(comp).hexdigest()==MODEL_Z_SHA
    raw=zlib.decompress(comp);pos=0
    n,pos=read_varint(raw,pos); idx=[]; cnt=[]; prev=0
    for _ in range(n):
        d,pos=read_varint(raw,pos);c,pos=read_varint(raw,pos);prev+=d;idx.append(prev);cnt.append(c)
    uni=[]
    for _ in range(19):u,pos=read_varint(raw,pos);uni.append(u)
    assert pos==len(raw) and n==24159
    tab=np.full((19**3,19),.5,dtype=np.float64)
    for i,c in zip(idx,cnt):tab[i//19,i%19]+=c
    tab/=tab.sum(axis=1,keepdims=True)
    uc=np.asarray(uni,dtype=np.float64); up=(uc+.5)/(uc.sum()+.5*19)
    return np.log(tab.ravel()),np.log(up),up

def parse_rf():
    b=get(RF_URL,HEADERS); assert hashlib.sha256(b).hexdigest()==RF_SHA
    text=b.decode('utf-8'); rx=re.compile(r'[A-Z][1-9a-z]')
    fol=collections.defaultdict(list); counts=collections.Counter();total=0
    for line in text.splitlines():
        if not line.startswith('<') or '>' not in line:continue
        lab,rhs=line.split('>',1)
        if '.' not in lab:continue
        f=lab[1:].split('.',1)[0]
        rhs=re.sub(r'\[[^\]]*\]','<BREAK>',rhs)
        for p in re.split(r'<(?:-|~)>|<BREAK>',rhs):
            toks=rx.findall(p)
            if toks:fol[f].append(toks);counts.update(toks);total+=len(toks)
    order=sorted(counts,key=lambda x:(-counts[x],x));cum=0;k995=None
    for i,x in enumerate(order,1):
        cum+=counts[x]
        if k995 is None and cum/total>=.995:k995=i
    assert total==157254 and k995==36
    vocab=order[:36]; ids={x:i for i,x in enumerate(vocab)}
    folios=sorted(fol,key=lambda f:hashlib.sha256(('TRANCHSTA20split::'+f).encode()).digest())
    nt=round(.5*len(folios));nh=round(.2*len(folios)); T=folios[:nt];H=folios[nt:nt+nh];C=folios[nt+nh:]
    return fol, vocab, ids, T,H,C

def retained_segments(fol,fs,ids):
    out=[]; meta=[]; alln=ret=0
    for f in fs:
        for si,line in enumerate(fol[f]):
            alln+=len(line); cur=[]
            sub=0
            for x in line:
                if x in ids:cur.append(ids[x]);ret+=1
                else:
                    if cur:out.append(cur);meta.append((f,si,sub));sub+=1;cur=[]
            if cur:out.append(cur);meta.append((f,si,sub))
    return out,meta,ret/alln,alln,ret

def flatten(segs):
    a=[]
    for i,s in enumerate(segs):
        if i:a.append(-1)
        a.extend(s)
    return np.asarray(a,dtype=np.int32)

@njit(cache=True)
def score_key(cipher,key,quad,uni):
    V=19;total=0.;n=0;h0=h1=h2=-1
    for c in cipher:
        if c<0:h0=h1=h2=-1;continue
        p=key[c]
        if h0<0 or h1<0 or h2<0: total+=uni[p]
        else: total+=quad[((h0*V+h1)*V+h2)*V+p]
        n+=1;h0,h1,h2=h1,h2,p
    return total/max(n,1)

@njit(cache=True)
def rng_step(state):
    state^=state>>np.uint64(12);state^=state<<np.uint64(25);state^=state>>np.uint64(27);return state*np.uint64(2685821657736338717)
@njit(cache=True)
def rng_int(state,upper):state=rng_step(state);return state,int(state%np.uint64(upper))
@njit(cache=True)
def polish(cipher,key,quad,uni,mult,sweeps,seed):
    cur=key.copy();best=score_key(cipher,cur,quad,uni);state=np.uint64(seed if seed else 1);inds=np.empty(36,np.int32)
    for _ in range(sweeps):
        order=np.arange(19,dtype=np.int32)
        for z in range(18,0,-1):state,j=rng_int(state,z+1);tmp=order[z];order[z]=order[j];order[j]=tmp
        imp=0
        for ii in range(18):
            a=order[ii]
            for jj in range(ii+1,19):
                b=order[jj];m=0
                for s in range(36):
                    if cur[s]==a or cur[s]==b:inds[m]=s;m+=1
                ra=mult[a];bs=best;bm=-1
                for mask in range(1<<m):
                    ca=0
                    for q in range(m):ca+=(mask>>q)&1
                    if ca!=ra:continue
                    old=np.empty(m,np.int32)
                    for q in range(m):old[q]=cur[inds[q]];cur[inds[q]]=a if ((mask>>q)&1) else b
                    sc=score_key(cipher,cur,quad,uni)
                    if sc>bs+1e-12:bs=sc;bm=mask
                    for q in range(m):cur[inds[q]]=old[q]
                if bm>=0:
                    for q in range(m):cur[inds[q]]=a if ((bm>>q)&1) else b
                    best=bs;imp+=1
        if imp==0:break
    return cur,best

def seedint(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'big')&0x7fffffff

def freq_init(cipher,plain_uni):
    f=np.bincount(cipher[cipher>=0],minlength=36).astype(float);f/=f.sum();slots=[]
    for l,m in enumerate(MULT):
        for _ in range(int(m)):slots.append((plain_uni[l]/m,l))
    slots=sorted(slots,key=lambda x:(-x[0],x[1]));syms=sorted(range(36),key=lambda s:(-f[s],s));k=np.empty(36,np.int32)
    for s,(_,l) in zip(syms,slots):k[s]=l
    return k

def run_ensemble(cipher,quad,uni,puni,name):
    init=freq_init(cipher,puni);slot=np.concatenate([np.full(int(m),l,np.int32) for l,m in enumerate(MULT)]);bestk=None;best=-1e100;hist=[]
    for r in range(36):
        sd=seedint(f'TRANCHSTA20H::{name}::{r}');rng=np.random.default_rng(sd)
        if r==0:k=init.copy()
        elif r%3==1 and bestk is not None:
            k=bestk.copy()
            for _ in range(2+r%7):i,j=rng.integers(0,36,2);k[i],k[j]=k[j],k[i]
        else:k=rng.permutation(slot).astype(np.int32)
        k,sc=polish(cipher,k,quad,uni,MULT,12,sd)
        if sc>best:best=float(sc);bestk=k.copy()
        if (r+1)%6==0:hist.append((r+1,best));print('FIT',name,r+1,best,flush=True)
    return bestk,best,hist

def agreement(a,b,freq):return float(freq[a==b].sum()/freq.sum())

def shuffled_score(segs,meta,key,quad,uni,r):
    out=[]
    for s,(f,si,sub) in zip(segs,meta):
        q=np.asarray(s,np.int32).copy();rng=np.random.default_rng(seedint(f'TRANCHSTA20Hshuffle::{r}::{f}::{si}:{sub}'));rng.shuffle(q);out.append(q.tolist())
    return float(score_key(flatten(out),key,quad,uni))

def main():
    quad,uni,puni=load_model();fol,vocab,ids,T,H,C=parse_rf();Tseg,Tmeta,Tcov,_,_=retained_segments(fol,T,ids);Hseg,Hmeta,Hcov,Hall,Hret=retained_segments(fol,H,ids)
    tc=flatten(Tseg);hc=flatten(Hseg);freq=np.bincount(tc[tc>=0],minlength=36).astype(float)
    ka,sa,ha=run_ensemble(tc,quad,uni,puni,'A');kb,sb,hb=run_ensemble(tc,quad,uni,puni,'B');agr=agreement(ka,kb,freq)
    key=ka if sa>=sb else kb;obs=float(score_key(hc,key,quad,uni))
    null=np.asarray([shuffled_score(Hseg,Hmeta,key,quad,uni,r) for r in range(200)]);q99=float(np.quantile(null,.99));med=float(np.median(null))
    buckets=[]
    for bi in range(4):
        fs=[f for f in H if hashlib.sha256(('TRANCHSTA20Hbucket::'+f).encode()).digest()[0]%4==bi]
        seg,meta,cov,_,_=retained_segments(fol,fs,ids);bc=flatten(seg);bo=float(score_key(bc,key,quad,uni));bn=np.asarray([shuffled_score(seg,meta,key,quad,uni,r) for r in range(200)]);buckets.append({'bucket':bi,'folios':len(fs),'coverage':cov,'observed':bo,'null_median':float(np.median(bn)),'delta':bo-float(np.median(bn))})
    gate=bool(Hcov>=.97 and agr>=.90 and obs>=P05 and obs>q99 and all(x['delta']>0 for x in buckets))
    result={'vocab':vocab,'T_folios':len(T),'H_folios':len(H),'C_folios':len(C),'T_segments':len(Tseg),'H_segments':len(Hseg),'H_all':Hall,'H_retained':Hret,'H_coverage':Hcov,'T_score_A':sa,'T_score_B':sb,'T_AB_agreement':agr,'T_history_A':ha,'T_history_B':hb,'candidate_map_indices':key.tolist(),'H_observed':obs,'positive_control_p05':P05,'null_median':med,'null_q99':q99,'H_minus_null_q99':obs-q99,'buckets':buckets,'gate':gate,'verdict':'H20 TRANCHEDINO-STA CANDIDATE' if gate else 'NO TRANCHEDINO-STA ALPHABETIC SIGNAL'}
    print('H20_RESULT='+json.dumps(result,separators=(',',':')),flush=True)
if __name__=='__main__':main()
