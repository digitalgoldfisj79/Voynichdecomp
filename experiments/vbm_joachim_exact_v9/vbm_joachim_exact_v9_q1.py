#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, collections, hashlib, json, math, re, urllib.request
from dataclasses import dataclass
from typing import Any
import numpy as np
from numba import njit
from unidecode import unidecode

NS='VBMJOACHIMEXACTV9Q1'
Q0NS='VBMJOACHIMEXACTV9Q0'
VMS_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/gpt56/vbm-bridge-factor-v0.2-20260821/voynich_transcriptions_slim.json'
URLS={
 'german':'https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu',
 'italian':'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu'}
H1={'f28v','f31v','f88r','f5r','f34r','f81v'}
C1={'f85r1','f53v','f33r','f10r','f23r','f111r'}
ATOMS=('ckh','cth','cph','cfh','ch','sh','qo')
VOWELS='aeiou'; VSET=set(VOWELS)
CONSONANTS='bcdfghjklmnpqrstvwxyz'; assert len(CONSONANTS)==21
UA={'User-Agent':'VBMJoachimExactV9Q1/2026-09-01'}
FIT_EVENTS=6000; HOLD_EVENTS=3000; NCAND=64
DEV_REPS=range(0,4); CAL_REPS=range(100,106); VAL_REPS=range(200,206)
RESTARTS=8; MAX_CYCLES=7
LN2=math.log(2.0)


def seed(*parts):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,parts)).encode()).digest()[:8],'big') & 0x7fffffff

def get_bytes(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=120) as r:return r.read()
def get_json(url):return json.loads(get_bytes(url).decode('utf-8'))

def parse_ud(raw:bytes):
    sents=[]; cur=[]
    for line in raw.decode('utf-8','replace').splitlines():
        if not line:
            if cur:sents.append(cur);cur=[]
            continue
        if line.startswith('#'):continue
        c=line.split('\t')
        if len(c)>=2 and c[0].isdigit():cur.append(c[1])
    if cur:sents.append(cur)
    return sents

def norm_token(x):
    s=unidecode(str(x)).lower().translate(str.maketrans({'j':'i','v':'u','w':'u','y':'i','x':'s','z':'s'}))
    return ''.join(c for c in s if 'a'<=c<='z')

def raw_events(words):
    s=''.join(norm_token(w) for w in words); out=[]; run=''
    if not s:return None
    for c in s:
        if c in VSET:
            if run:
                if len(run)>5:return None
                out.append(run);run=''
            out.append(c)
        else:
            run+=c
            if len(run)>5:return None
    if run:
        if len(run)>5:return None
        out.append(run)
    return out if len(out)>=4 else None

@dataclass
class Lang:
    name:str
    nuclei:list[str]
    sem_names:list[str]
    sem_class:np.ndarray
    sem_cost_bits:np.ndarray
    logtri:np.ndarray
    sem_freq:np.ndarray
    pools:dict[int,list[np.ndarray]]


def build_language(name,sents):
    nuc=collections.Counter(); train_raw=[]
    for i,ws in enumerate(sents):
        if i%10 not in range(0,6):continue
        ev=raw_events(ws)
        if not ev:continue
        train_raw.append(ev)
        for x in ev:
            if x not in VSET:nuc[x]+=1
    nuclei=[x for x,_ in sorted(nuc.items(),key=lambda kv:(-kv[1],kv[0]))[:NCAND]]
    nset=set(nuclei); sem_names=list(VOWELS)+nuclei; sid={x:i for i,x in enumerate(sem_names)}; K=len(sem_names); B=K
    cls=np.array([0]*5+[1]*len(nuclei),np.int8)
    cost=np.array([math.log2(5)]*5+[math.log2(5)+len(x)*math.log2(21) for x in nuclei],float)
    C=np.full((K+1,K+1,K+1),0.25,dtype=np.float64); F=np.full(K,0.25,dtype=np.float64); used=0
    for ev in train_raw:
        if any((x not in VSET and x not in nset) for x in ev):continue
        a=[sid[x] for x in ev]; used+=len(a)
        for x in a:F[x]+=1
        z=[B,B]+a+[B,B]
        for x,y,q in zip(z,z[1:],z[2:]):C[x,y,q]+=1
    C/=C.sum(axis=2,keepdims=True);F/=F.sum()
    pools={6:[],7:[],8:[],9:[]}
    for i,ws in enumerate(sents):
        r=i%10
        if r not in pools:continue
        ev=raw_events(ws)
        if not ev or any((x not in VSET and x not in nset) for x in ev):continue
        pools[r].append(np.array([sid[x] for x in ev],np.int16))
    if used<10000 or min(len(pools[r]) for r in (6,7,8))<100:raise RuntimeError((name,'insufficient corpus',used,{r:len(v) for r,v in pools.items()}))
    return Lang(name,nuclei,sem_names,cls,cost,np.log(C),F,pools)

# Q0b source parser, used only on Q0 TRAIN to obtain empirical surface inventory/counts.
def left_half(w):
    for a in ATOMS:
        if len(w)>=len(a)+1 and w.startswith(a):return a
    return w[0]
def parse_vms_token(w):
    if not re.fullmatch(r'[a-z]+',w):return None
    if len(w)==1:return (w,'',w)
    L=left_half(w);return (L,w[len(L):-1],w[-1]) if len(w)>=len(L)+1 else None
def split_folio(fid):
    h=hashlib.sha256(f'{Q0NS}::{fid}'.encode()).hexdigest()[:8]
    return 'INTERNAL_HOLDOUT' if int(h,16)%5==0 else 'TRAIN'

def surface_inventory():
    data=get_json(VMS_URL); nc=collections.Counter();bc=collections.Counter()
    for fid,lines in sorted(data['pages'].items()):
        if fid in H1 or fid in C1 or split_folio(fid)!='TRAIN':continue
        for ln in sorted(lines,key=lambda x:int(x) if str(x).isdigit() else 999999):
            txt=lines[ln].get('t',{}).get('ZLZI','');seg=[]
            def flush():
                nonlocal seg
                if not seg:return
                tr=[parse_vms_token(x) for x in seg]
                for _,N,_ in tr:
                    if N:nc[N]+=1
                for a,b in zip(tr,tr[1:]):bc[a[2]+'|'+b[0]]+=1
                seg=[]
            for w in txt.split():
                if parse_vms_token(w) is None:flush()
                else:seg.append(w)
            flush()
    nactive=[(x,c) for x,c in sorted(nc.items()) if c>=5]
    bactive=[(x,c) for x,c in sorted(bc.items()) if c>=3]
    if len(nactive)<NCAND or len(bactive)<5:raise RuntimeError(('active inventory too small',len(nactive),len(bactive)))
    names=[f'N:{x}' for x,_ in nactive]+[f'B:{x}' for x,_ in bactive]
    weights=np.array([c for _,c in nactive]+[c for _,c in bactive],float)
    sclass=np.array([1]*len(nactive)+[0]*len(bactive),np.int8)
    return names,weights,sclass,{'active_nucleus':len(nactive),'active_bridge':len(bactive),'nucleus_events':sum(c for _,c in nactive),'bridge_events':sum(c for _,c in bactive)}

def collect_window(pool,lang,rep,need_fit=FIT_EVENTS,need_hold=HOLD_EVENTS):
    start=seed(NS,'window',lang,rep)%len(pool); fit=[];hold=[]; nf=nh=0;j=0; phase=0
    while nh<need_hold:
        if j>len(pool)*4:raise RuntimeError(('pool exhausted',lang,rep,nf,nh,len(pool)))
        line=pool[(start+j)%len(pool)];j+=1;pos=0
        while pos<len(line):
            if nf<need_fit:
                take=min(len(line)-pos,need_fit-nf);q=line[pos:pos+take];pos+=take;nf+=take
                if len(q):fit.append(q.copy())
            else:
                take=min(len(line)-pos,need_hold-nh);q=line[pos:pos+take];pos+=take;nh+=take
                if len(q):hold.append(q.copy())
                if nh>=need_hold:break
    return fit,hold

def make_key(L:Lang,sweights,sclass,rep,tag):
    M=len(sweights); key=np.full(M,-1,np.int16);rng=np.random.default_rng(seed(NS,'key',L.name,rep,tag))
    for cls,sems in [(0,np.arange(0,5,dtype=np.int16)),(1,np.arange(5,69,dtype=np.int16))]:
        sur=np.where(sclass==cls)[0].copy();rng.shuffle(sur); ss=sems.copy();rng.shuffle(ss)
        for s,v in zip(sur[:len(ss)],ss):key[s]=v
        probs=L.sem_freq[sems].astype(float);probs/=probs.sum()
        if len(sur)>len(ss):key[sur[len(ss):]]=rng.choice(sems,size=len(sur)-len(ss),p=probs)
    assert np.all(key>=0)
    return key

def encrypt(lines,key,sweights,rep,lang,tag):
    rng=np.random.default_rng(seed(NS,'encrypt',lang,rep,tag));M=len(key); by={}
    for sem in range(69):
        ix=np.where(key==sem)[0];w=sweights[ix].astype(float);w/=w.sum();by[sem]=(ix,w)
    out=[]
    for line in lines:
        z=[]
        for sem in line:
            ix,w=by[int(sem)];z.append(int(rng.choice(ix,p=w)))
        out.append(np.array(z,np.int32))
    return out

@dataclass
class Sparse:
    a:np.ndarray;b:np.ndarray;c:np.ndarray;n:np.ndarray;off:np.ndarray;adj:np.ndarray;freq:np.ndarray;seen:np.ndarray;B:int

def sparse_stats(lines,M):
    B=M;cnt=collections.Counter();freq=np.zeros(M,np.int64)
    for line in lines:
        for s in line:freq[int(s)]+=1
        z=[B,B]+[int(x) for x in line]+[B,B]
        for x,y,q in zip(z,z[1:],z[2:]):cnt[(x,y,q)]+=1
    ks=list(cnt);a=np.array([k[0] for k in ks],np.int32);b=np.array([k[1] for k in ks],np.int32);c=np.array([k[2] for k in ks],np.int32);n=np.array([cnt[k] for k in ks],np.int64)
    lists=[[] for _ in range(M)]
    for i,(x,y,q) in enumerate(ks):
        for t in set((x,y,q)):
            if t!=B:lists[t].append(i)
    off=[0];flat=[]
    for z in lists:flat.extend(z);off.append(len(flat))
    seen=np.where(freq>0)[0].astype(np.int32)
    return Sparse(a,b,c,n,np.array(off,np.int32),np.array(flat,np.int32),freq,seen,B)

@njit(cache=True)
def score_sparse(a,b,c,n,mapping,logtri,semB):
    total=0.0
    for i in range(len(n)):
        x=semB if a[i]==len(mapping) else mapping[a[i]]
        y=semB if b[i]==len(mapping) else mapping[b[i]]
        z=semB if c[i]==len(mapping) else mapping[c[i]]
        total+=n[i]*logtri[x,y,z]
    return total

@njit(cache=True)
def delta_one(t,newv,a,b,c,n,off,adj,mapping,logtri,semB):
    old=mapping[t];d=0.0;M=len(mapping)
    if old==newv:return 0.0
    for jj in range(off[t],off[t+1]):
        i=adj[jj];x=a[i];y=b[i];z=c[i]
        ox=semB if x==M else mapping[x];oy=semB if y==M else mapping[y];oz=semB if z==M else mapping[z]
        nx=newv if x==t else ox;ny=newv if y==t else oy;nz=newv if z==t else oz
        d+=n[i]*(logtri[nx,ny,nz]-logtri[ox,oy,oz])
    return d

@njit(cache=True)
def best_move(t,cands,a,b,c,n,off,adj,mapping,logtri,semB,cost_nats):
    old=mapping[t];best=old;bd=0.0
    for q in cands:
        q=int(q)
        if q==old:continue
        d=delta_one(t,q,a,b,c,n,off,adj,mapping,logtri,semB)-(cost_nats[q]-cost_nats[old])
        if d>bd+1e-10:bd=d;best=q
    return best,bd

def freq_init(S:Sparse,L:Lang,sclass,rng=None):
    M=len(sclass);m=np.zeros(M,np.int16)
    for cls,sems in [(0,np.arange(0,5,dtype=np.int16)),(1,np.arange(5,69,dtype=np.int16))]:
        sur=np.where((sclass==cls)&(S.freq>0))[0]
        if rng is not None:
            probs=L.sem_freq[sems].astype(float);probs/=probs.sum();m[sur]=rng.choice(sems,size=len(sur),p=probs)
        else:
            order=sur[np.argsort(-S.freq[sur])];target=L.sem_freq[sems].astype(float);target/=target.sum();rem=target.copy();tot=max(1,float(S.freq[sur].sum()))
            for t in order:
                j=int(np.argmax(rem));m[t]=sems[j];rem[j]-=S.freq[t]/tot
        # unseen values arbitrary; never scored
        unseen=np.where((sclass==cls)&(S.freq==0))[0]
        if len(unseen):m[unseen]=sems[0]
    return m

def objective(S,L,m):
    semB=69;ll=score_sparse(S.a,S.b,S.c,S.n,m,L.logtri,semB);cost=float(np.sum(L.sem_cost_bits[m[S.seen]])*LN2);return ll-cost

def solve(lines,L,sclass,tag):
    M=len(sclass);S=sparse_stats(lines,M);semB=69;costn=L.sem_cost_bits*LN2;cands0=np.arange(0,5,dtype=np.int16);cands1=np.arange(5,69,dtype=np.int16)
    best=None
    for r in range(RESTARTS):
        rng=np.random.default_rng(seed(NS,'solver',tag,r));m=freq_init(S,L,sclass,None if r==0 else rng)
        # deterministic/randomized coordinate polishing
        for cyc in range(MAX_CYCLES):
            order=S.seen.copy()
            if r>0 or cyc>0:rng.shuffle(order)
            changes=0
            for t in order:
                cand=cands0 if sclass[t]==0 else cands1
                nv,d=best_move(int(t),cand,S.a,S.b,S.c,S.n,S.off,S.adj,m,L.logtri,semB,costn)
                if nv!=m[t]:m[t]=nv;changes+=1
            if changes==0:break
        sc=objective(S,L,m)
        if best is None or sc>best[0]:best=(sc,m.copy(),r)
    return best[1],S,{'objective':best[0],'best_restart':best[2],'seen_surface_types':len(S.seen),'restarts':RESTARTS,'cycles':MAX_CYCLES}

def fit_accuracy(lines,m,truth):
    cnt=collections.Counter(int(x) for z in lines for x in z);den=sum(cnt.values());hit=sum(n for s,n in cnt.items() if int(m[s])==int(truth[s]));return hit/max(1,den)

def hold_metrics(lines,m,fitseen,truth,L):
    seen=set(map(int,fitseen));hit=den=0;li=lt=0.0;nt=0;events=0
    B=69
    for line in lines:
        hi=[B,B];ht=[B,B]
        for s0 in line:
            s=int(s0);events+=1
            if s not in seen:
                hi=[B,B];ht=[B,B];continue
            pi=int(m[s]);tt=int(truth[s]);hit+=pi==tt;den+=1
            li+=L.logtri[hi[-2],hi[-1],pi];lt+=L.logtri[ht[-2],ht[-1],tt];nt+=1;hi.append(pi);ht.append(tt)
        # explicit line endings only if at least one scored event since last reset are omitted equally; metric is event NLL
    return {'semantic_accuracy':hit/max(1,den),'scored_fraction':den/max(1,events),'scored_events':den,
            'inferred_nll':-li/max(1,nt),'true_key_nll':-lt/max(1,nt),'regret':(-li+lt)/max(1,nt)}

def summarize(rows,phase):
    out={'phase':phase,'languages':{}}
    for lang in ('german','italian'):
        rr=[x for x in rows if x['language']==lang];ga=np.array([x['global']['semantic_accuracy'] for x in rr]);fa=np.array([x['fresh']['semantic_accuracy'] for x in rr]);gn=np.array([x['global']['inferred_nll'] for x in rr]);fn=np.array([x['fresh']['inferred_nll'] for x in rr]);gr=np.array([x['global']['regret'] for x in rr]);sf=np.array([x['global']['scored_fraction'] for x in rr])
        gaps=ga-fa;ngaps=fn-gn
        d={'n':len(rr),'global_accuracy':ga.tolist(),'fresh_accuracy':fa.tolist(),'paired_accuracy_gap':gaps.tolist(),'global_regret':gr.tolist(),'paired_nll_gap':ngaps.tolist(),'scored_fraction':sf.tolist(),
           'median_global_accuracy':float(np.median(ga)),'min_global_accuracy':float(np.min(ga)),'median_fresh_accuracy':float(np.median(fa)),'median_accuracy_gap':float(np.median(gaps)),'pairs_gap_gt_020':int(np.sum(gaps>.20)),'median_global_regret':float(np.median(gr)),'median_nll_gap':float(np.median(ngaps)),'median_scored_fraction':float(np.median(sf))}
        if phase in ('CAL','VAL'):
            gates={'median_global_acc_ge_060':d['median_global_accuracy']>=.60,'all_global_acc_ge_045':d['min_global_accuracy']>=.45,'median_fresh_acc_le_030':d['median_fresh_accuracy']<=.30,'median_acc_gap_ge_030':d['median_accuracy_gap']>=.30,'five_of_six_gap_gt_020':d['pairs_gap_gt_020']>=5,'median_global_regret_le_035':d['median_global_regret']<=.35,'median_nll_gap_ge_025':d['median_nll_gap']>=.25,'median_scored_fraction_ge_090':d['median_scored_fraction']>=.90};d['gates']=gates;d['pass']=all(gates.values())
        else:
            gates={'median_global_acc_ge_050':d['median_global_accuracy']>=.50,'median_gap_ge_020':d['median_accuracy_gap']>=.20};d['dev_gates']=gates;d['dev_pass']=all(gates.values())
        out['languages'][lang]=d
    out['overall_pass']=all(v.get('pass',v.get('dev_pass',False)) for v in out['languages'].values())
    return out

def run_phase(phase):
    reps={'DEV':DEV_REPS,'CAL':CAL_REPS,'VAL':VAL_REPS}[phase];names,sweights,sclass,smeta=surface_inventory();rows=[];lmeta={}
    for lname in ('german','italian'):
        sents=parse_ud(get_bytes(URLS[lname]));L=build_language(lname,sents);lmeta[lname]={'nuclei':L.nuclei,'pool_sizes':{str(k):len(v) for k,v in L.pools.items()}}
        residue={'DEV':6,'CAL':7,'VAL':8}[phase];pool=L.pools[residue]
        for rep in reps:
            fitplain,holdplain=collect_window(pool,lname,rep);gkey=make_key(L,sweights,sclass,rep,'GLOBAL');fkey=make_key(L,sweights,sclass,rep,'FRESH_HOLD')
            fitcipher=encrypt(fitplain,gkey,sweights,rep,lname,'FIT');ghold=encrypt(holdplain,gkey,sweights,rep,lname,'GLOBAL_HOLD');fhold=encrypt(holdplain,fkey,sweights,rep,lname,'FRESH_HOLD')
            mapping,S,sm=solve(fitcipher,L,sclass,f'{phase}:{lname}:{rep}');fm=fit_accuracy(fitcipher,mapping,gkey);gm=hold_metrics(ghold,mapping,S.seen,gkey,L);fr=hold_metrics(fhold,mapping,S.seen,fkey,L)
            row={'phase':phase,'language':lname,'replicate':rep,'fit_mapping_accuracy':fm,'solver':sm,'global':gm,'fresh':fr};rows.append(row);print('CASE='+json.dumps(row,sort_keys=True,separators=(',',':')),flush=True)
    summ=summarize(rows,phase);out={'protocol':'VBM_JOACHIM_EXACT_V9_Q1_PROTOCOL.md','phase':phase,'surface_meta':smeta,'solver_config':{'restarts':RESTARTS,'max_cycles':MAX_CYCLES,'fit_events':FIT_EVENTS,'hold_events':HOLD_EVENTS,'nucleus_candidates':NCAND},'language_meta':lmeta,'cases':rows,'summary':summ}
    print('VBM_V9_Q1_RESULT='+json.dumps(out,sort_keys=True,separators=(',',':')),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['DEV','CAL','VAL'],required=True);args=ap.parse_args();run_phase(args.phase)
