#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import linear_sum_assignment
from rapidfuzz.distance import Levenshtein

CSV_SHA='c5eba63cbe8055d3506d099043f5df23fd427df709546df6de70e084fedd3cf6'
ALPH='abcdefghilmnopqrstu'; V=len(ALPH); CID={c:i for i,c in enumerate(ALPH)}
TRANS=str.maketrans({'j':'i','v':'u','w':'u','y':'i','x':'s','z':'s'})
GEMS=('bb','cc','dd','ff','gg','ll','nn','pp','rr','ss','tt')
RATES=(0.01,0.03,0.06,0.10)

def seedint(s:str)->int:return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'big')&0x7fffffff
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def norm_word(x):
    x=str(x).lower().translate(TRANS);return ''.join(c for c in x if c in CID)
def words(x):return [w for raw in str(x).split() if (w:=norm_word(raw))]

def load_source(path:Path):
    assert sha(path)==CSV_SHA
    d=pd.read_csv(path).fillna('');d['words']=d.text.astype(str).map(words);d['letters19']=d.words.map(lambda z:''.join(z))
    pages=sorted(d.loc[d.letters19.str.len()>0,'page'].unique());cut=pages[int(len(pages)*.72)];assert cut==183
    tr=d[(d.page<cut)&(d.letters19.str.len()>0)].copy();te=d[(d.page>=cut)&(d.letters19.str.len()>0)].copy()
    hp=sorted(te.page.unique(),key=lambda p:hashlib.sha256(f'TRANCHSTA23B1source::{int(p)}'.encode()).digest())
    D=te[te.page.isin(set(hp[:17]))].copy();Q=te[te.page.isin(set(hp[17:]))].copy()
    assert int(tr.letters19.str.len().sum())==172347 and int(te.letters19.str.len().sum())==54750
    return tr,D,Q

def build_model(tr):
    qc=collections.Counter();uc=np.zeros(V)
    for s in tr.letters19:
        a=[CID[c] for c in s]
        for x in a:uc[x]+=1
        for i in range(3,len(a)):qc[tuple(a[i-3:i+1])]+=1
    tab=np.full((V**3,V),.5,dtype=np.float64)
    for q,n in qc.items():tab[((q[0]*V+q[1])*V+q[2]),q[3]]+=n
    tab/=tab.sum(1,keepdims=True);up=(uc+.5)/(uc.sum()+.5*V)
    return np.log(tab.ravel()),np.log(up)

def semantics(tr):
    train=collections.Counter(w for z in tr.words for w in z);top96=[w for w,_ in train.most_common(96)]
    sem=[];cl=[]
    for c in ALPH:
        for _ in range(1 if c in 'ac' else 2):sem.append(c);cl.append(0)
    for g in GEMS:sem.append(g);cl.append(1)
    for _ in range(7):sem.append('');cl.append(2)
    assert len(sem)==54
    sem.extend(top96);cl.extend([3]*96)
    maxlen=max(map(len,sem));out=np.full((150,maxlen),-1,np.int16);lens=np.zeros(150,np.int16)
    for i,s in enumerate(sem):
        lens[i]=len(s)
        for j,c in enumerate(s):out[i,j]=CID[c]
    trainwords=[w for z in tr.words for w in z];ae=collections.Counter();ge=collections.Counter();sub=0
    for w in trainwords:
        i=0
        while i<len(w):
            if i+1<len(w) and w[i:i+2] in GEMS:ge[w[i:i+2]]+=1;i+=2
            else:ae[w[i]]+=1;i+=1
            sub+=1
    scale=12000/sum(map(len,trainwords));fixed=[]
    for c in ALPH:
        for _ in range(1 if c in 'ac' else 2):fixed.append(ae[c]*scale/(1 if c in 'ac' else 2))
    for g in GEMS:fixed.append(ge[g]*scale)
    wr=np.asarray([train[w]*scale for w in top96],float)
    return train,top96,sem,np.asarray(cl,np.int8),out,lens,np.asarray(fixed,float),float(sub*scale),wr

def collect(panel,tag,target=12000):
    ls=[z for z in panel.words.tolist() if z];st=seedint(tag)%len(ls);out=[];n=0;i=0
    while n<target:
        z=ls[(st+i)%len(ls)];cur=[];m=0
        for w in z:
            if n+m+len(w)>target and (out or cur):break
            cur.append(w);m+=len(w)
        if cur:out.append(cur);n+=m
        i+=1
    return out

def generate(panel,top96,sem,rep,pnull,panelname):
    plain_words=collect(panel,f'TRANCHSTA23B1window::{panelname}::{rep}',12000)
    rng=np.random.default_rng(seedint(f'TRANCHSTA23B1key::{panelname}::{rep}'))
    cb=list(rng.choice(top96,size=38,replace=False));cbset=set(cb)
    alpha={c:[i for i in range(36) if sem[i]==c] for c in ALPH};gem={g:sem.index(g,36,47) for g in GEMS};null=list(range(47,54));word={w:54+top96.index(w) for w in cb}
    used=list(range(54))+[54+top96.index(w) for w in cb]
    lrng=np.random.default_rng(seedint(f'TRANCHSTA23B1label::{panelname}::{rep}'));perm=lrng.permutation(92);surf={sid:int(perm[i]) for i,sid in enumerate(used)};truth=np.full(92,-1,np.int16)
    for sid,x in surf.items():truth[x]=sid
    clines=[];plines=[]
    for ws in plain_words:
        ev=[];plines.append(''.join(ws))
        for w in ws:
            if w in cbset:
                ev.append(surf[word[w]])
                if rng.random()<pnull:ev.append(surf[int(rng.choice(null))])
            else:
                i=0
                while i<len(w):
                    if i+1<len(w) and w[i:i+2] in gem:sid=gem[w[i:i+2]];i+=2
                    else:sid=int(rng.choice(alpha[w[i]]));i+=1
                    ev.append(surf[sid])
                    if rng.random()<pnull:ev.append(surf[int(rng.choice(null))])
        clines.append(ev)
    return clines,plines,truth,cb

def flatten(lines):
    z=[]
    for i,line in enumerate(lines):
        if i:z.append(-1)
        z.extend(line)
    return np.asarray(z,np.int16)

@njit(cache=True)
def scores(cipher,mapping,out,lens,logq,logu):
    total=0.;chars=0;events=0;h0=h1=h2=-1
    for s in cipher:
        if s<0:h0=h1=h2=-1;continue
        events+=1;sid=mapping[s];ln=lens[sid]
        for j in range(ln):
            x=out[sid,j]
            if h0<0 or h1<0 or h2<0:total+=logu[x]
            else:total+=logq[((h0*19+h1)*19+h2)*19+x]
            chars+=1;h0,h1,h2=h1,h2,x
    if chars==0 or events==0:return -1e9,-1e9,chars,events
    ratio=chars/events
    if ratio<.90 or ratio>1.12:return -1e8,-1e8,chars,events
    return total/events,total/chars,chars,events

@njit(cache=True)
def polish(cipher,mapping,out,lens,sclass,logq,logu,maxcycles):
    cur=mapping.copy();eventscore,charscore,n,e=scores(cipher,cur,out,lens,logq,logu);M=len(cur)
    for cyc in range(maxcycles):
        imp=0
        for a in range(M-1):
            for b in range(a+1,M):
                x=cur[a];y=cur[b]
                if (sclass[x]==0)!=(sclass[y]==0):continue
                if lens[x]==lens[y]:
                    same=True
                    for q in range(lens[x]):
                        if out[x,q]!=out[y,q]:same=False;break
                    if same:continue
                cur[a]=y;cur[b]=x;sc,cs,nn,ee=scores(cipher,cur,out,lens,logq,logu)
                if sc>eventscore+1e-11:eventscore=sc;charscore=cs;n=nn;e=ee;imp+=1
                else:cur[a]=x;cur[b]=y
        used=np.zeros(out.shape[0],np.uint8)
        for x in cur:used[x]=1
        for a in range(M):
            old=cur[a]
            if old<54:continue
            best=old;bs=eventscore;bcs=charscore;bn=n;be=e
            for cand in range(54,out.shape[0]):
                if used[cand] and cand!=old:continue
                cur[a]=cand;sc,cs,nn,ee=scores(cipher,cur,out,lens,logq,logu)
                if sc>bs+1e-11:best=cand;bs=sc;bcs=cs;bn=nn;be=ee
            cur[a]=best
            if best!=old:used[old]=0;used[best]=1;eventscore=bs;charscore=bcs;n=bn;e=be;imp+=1
        if imp==0:break
    return cur,eventscore,charscore,n,e

def init_freq(lines,pguess,fixed,subexp,wordrates):
    vals=[x for z in lines for x in z];counts=np.bincount(vals,minlength=92).astype(float);obs=np.maximum(counts,.3)
    sf=max(len(vals),1)/12000.0
    fx=np.concatenate([fixed,[subexp*pguess/7]*7])*sf;wr=wordrates*sf
    cost=np.abs(np.log(obs[:,None])-np.log(np.maximum(fx,.3)[None,:]));r,c=linear_sum_assignment(cost);m=np.full(92,-1,np.int16);assigned=set()
    for a,b in zip(r,c):m[a]=b;assigned.add(int(a))
    left=[i for i in range(92) if i not in assigned];cw=np.abs(np.log(obs[left,None])-np.log(np.maximum(wr,.3)[None,:]));r2,c2=linear_sum_assignment(cw)
    for a,b in zip(r2,c2):m[left[a]]=54+b
    return m

def perturb(m,sclass,seed,level):
    z=m.copy();rng=np.random.default_rng(seed);alpha=[i for i,x in enumerate(z) if sclass[x]==0];res=[i for i,x in enumerate(z) if sclass[x]!=0]
    for _ in range(2+2*level):
        a,b=rng.choice(alpha,2,replace=False);z[a],z[b]=z[b],z[a]
    for _ in range(3+2*level):
        a,b=rng.choice(res,2,replace=False);z[a],z[b]=z[b],z[a]
    used=set(int(x) for x in z)
    for _ in range(level):
        wordpos=[i for i,x in enumerate(z) if x>=54];a=int(rng.choice(wordpos));unused=[x for x in range(54,150) if x not in used]
        if unused:
            old=int(z[a]);new=int(rng.choice(unused));z[a]=new;used.discard(old);used.add(new)
    return z

def solve_ensemble(fitlines,name,rep,sclass,out,lens,logq,logu,fixed,subexp,wordrates):
    cipher=flatten(fitlines);best=None
    for pi,pguess in enumerate(RATES):
        base=init_freq(fitlines,pguess,fixed,subexp,wordrates)
        for restart in range(4):
            if restart==0 and name=='A':m=base.copy()
            else:m=perturb(base,sclass,seedint(f'TRANCHSTA23B1::{name}::{rep}::{pi}::{restart}'),restart+1)
            m,es,cs,n,e=polish(cipher,m,out,lens,sclass,logq,logu,8)
            row=(es,cs,m.copy(),pguess,restart,n,e)
            if best is None or es>best[0]+1e-12:best=row
    return best

def decode_lines(lines,m,sem):return [''.join(sem[int(m[x])] for x in z) for z in lines]
def edit_acc(pred,true):
    p=''.join(pred);t=''.join(true);return 1-Levenshtein.distance(p,t)/max(1,len(t))
def occ_metrics(lines,m,truth,sem):
    cnt=collections.Counter(x for z in lines for x in z);den=sum(cnt.values());exact=sum(v for s,v in cnt.items() if sem[int(m[s])]==sem[int(truth[s])]);by={}
    def cl(s):
        if s=='':return 'null'
        if s in GEMS:return 'gem'
        if len(s)==1:return 'alpha'
        return 'word'
    for kind in ('alpha','gem','word'):
        d=sum(v for s,v in cnt.items() if cl(sem[int(truth[s])])==kind);h=sum(v for s,v in cnt.items() if cl(sem[int(truth[s])])==kind and sem[int(m[s])]==sem[int(truth[s])]);by[kind]=h/d if d else 1.0
    tp=sum(v for s,v in cnt.items() if cl(sem[int(truth[s])])=='null' and cl(sem[int(m[s])])=='null');pred=sum(v for s,v in cnt.items() if cl(sem[int(m[s])])=='null');tru=sum(v for s,v in cnt.items() if cl(sem[int(truth[s])])=='null');pr=tp/pred if pred else 0.;re=tp/tru if tru else 1.;f1=2*pr*re/(pr+re) if pr+re else 0.;by['null_f1']=f1
    return exact/den if den else 1.,by

def agreement(m1,m2,lines,sem):
    cnt=collections.Counter(x for z in lines for x in z);return sum(v for s,v in cnt.items() if sem[int(m1[s])]==sem[int(m2[s])])/sum(cnt.values())
def heldout_score(lines,m,fitseen,out,lens,logq,logu):
    total=0.;n=0;scorable=0;events=0
    for line in lines:
        h=[]
        for s in line:
            events+=1
            if s not in fitseen:h=[];continue
            scorable+=1;sid=int(m[s])
            for j in range(int(lens[sid])):
                x=int(out[sid,j])
                if len(h)<3:total+=logu[x]
                else:total+=logq[((h[-3]*19+h[-2])*19+h[-1])*19+x]
                n+=1;h.append(x)
    return (total/n if n else -1e9,scorable/max(events,1),n)

def run_q(csvpath,rep):
    tr,D,Q=load_source(Path(csvpath));logq,logu=build_model(tr);train,top96,sem,sclass,out,lens,fixed,subexp,wr=semantics(tr);pnull=RATES[rep%4];lines,plain,truth,cb=generate(Q,top96,sem,rep,pnull,'Q');nfit=max(1,int(math.floor(.8*len(lines))));fit=lines[:nfit];hold=lines[nfit:]
    A=solve_ensemble(fit,'A',rep,sclass,out,lens,logq,logu,fixed,subexp,wr);B=solve_ensemble(fit,'B',rep,sclass,out,lens,logq,logu,fixed,subexp,wr);best=A if A[0]>=B[0] else B;m=best[2]
    fullrec=edit_acc(decode_lines(lines,m,sem),plain);semacc,parts=occ_metrics(lines,m,truth,sem);agr=agreement(A[2],B[2],lines,sem);fit_cs_A=scores(flatten(fit),A[2],out,lens,logq,logu)[1];fit_cs_B=scores(flatten(fit),B[2],out,lens,logq,logu)[1]
    seen=set(x for z in fit for x in z);hs,hcov,hn=heldout_score(hold,m,seen,out,lens,logq,logu)
    return {'replicate':rep,'p_null':pnull,'lines':len(lines),'fit_lines':nfit,'heldout_lines':len(hold),'ensemble_A':{'search_event_score':A[0],'fit_char_score':fit_cs_A,'p_guess':A[3],'restart':A[4]},'ensemble_B':{'search_event_score':B[0],'fit_char_score':fit_cs_B,'p_guess':B[3],'restart':B[4]},'AB_fit_char_score_delta':abs(fit_cs_A-fit_cs_B),'AB_occ_semantic_agreement':agr,'selected':('A' if A[0]>=B[0] else 'B'),'plaintext_recovery':fullrec,'semantic_occ_recovery':semacc,'component_occ_recovery':parts,'heldout_char_score':hs,'heldout_surface_coverage':hcov,'heldout_decoded_chars':hn,'minimum_gate_irrecoverably_failed':bool(fullrec<.75 or semacc<.70 or parts['word']<.60 or parts['gem']<.70 or parts['null_f1']<.75 or hcov<.95)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('csv');ap.add_argument('--q-rep',type=int);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        tr,D,Q=load_source(Path(a.csv));logq,logu=build_model(tr);train,top96,sem,sclass,out,lens,fixed,subexp,wr=semantics(tr);lines,plain,truth,cb=generate(D,top96,sem,0,.01,'D');m=init_freq(lines,.01,fixed,subexp,wr);print(json.dumps({'source_ok':True,'lines':len(lines),'events':sum(map(len,lines)),'init_score':scores(flatten(lines),m,out,lens,logq,logu)[0]}));return
    if a.q_rep is None:raise SystemExit('--q-rep required')
    print('B1_Q_RESULT='+json.dumps(run_q(a.csv,a.q_rep),separators=(',',':')),flush=True)
if __name__=='__main__':main()
