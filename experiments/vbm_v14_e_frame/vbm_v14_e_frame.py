#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "scipy>=1.13,<2", "scikit-learn>=1.5,<2"]
# ///
from __future__ import annotations
import collections, hashlib, json, math, re, urllib.request
import numpy as np
from scipy.spatial.distance import jensenshannon

NS='VBMV14EFRAME20260902'
BRANCH='experiment/vbm-v14-e-frame-mediation-20260902'
V11_URL=f'https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/{BRANCH}/experiments/vbm_v11_structural/vbm_v11_structural_part1.py'
UA={'User-Agent':'VBMV14EFrame/2026-09-02'}
NNULL=10000

def seed(*xs):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,(NS,)+xs)).encode()).digest()[:8],'big') & 0x7fffffff

def get_text(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=120) as r:return r.read().decode('utf-8')

def load_v11():
    ns={'__name__':'v11base'};exec(compile(get_text(V11_URL),V11_URL,'exec'),ns);return ns

def eskel(s):return re.sub(r'e+','E',s)
def ecount(s):return s.count('e')

def occurrences(segments,split='TRAIN',half=None):
    out=[]
    for s in segments:
        if s['split']!=split:continue
        if half is not None and s.get('half')!=half:continue
        tr=s['triples']
        for i,(L,N,R) in enumerate(tr):
            if not N:continue
            xl=tr[i-1][2] if i>0 else 'EDGE'
            xr=tr[i+1][0] if i+1<len(tr) else 'EDGE'
            out.append({'n':N,'frame':(L,R),'L':L,'R':R,'XL':xl,'XR':xr})
    return out

def full_setup(segments):
    tr=occurrences(segments,'TRAIN');cnt=collections.Counter(o['n'] for o in tr);elig=sorted([n for n,c in cnt.items() if c>=20])
    groups=collections.defaultdict(list)
    for n in elig:groups[eskel(n)].append(n)
    pairs=[]
    for sk,ls in groups.items():
        for i in range(len(ls)):
            for j in range(i+1,len(ls)):
                if ecount(ls[i])!=ecount(ls[j]):pairs.append((ls[i],ls[j]))
    vals=np.array([cnt[n] for n in elig],float);qs=np.quantile(vals,np.linspace(0,1,11))
    dec={n:min(9,max(0,int(np.searchsorted(qs,cnt[n],side='right')-2))) for n in elig}
    return tr,cnt,elig,pairs,dec

def vocab_from(occ,key,edge=False):
    xs=sorted({o[key] for o in occ if (edge or o[key]!='EDGE')},key=str)
    return xs

def side_vector(vals,vocab,alpha=.5):
    idx={x:i for i,x in enumerate(vocab)};a=np.full(len(vocab)+1,alpha,float)
    for z in vals:a[idx.get(z,len(vocab))]+=1
    a/=a.sum();return a

def frame_vectors(occ,elig,vL,vR):
    by=collections.defaultdict(list)
    for o in occ:
        if o['n'] in elig:by[o['n']].append(o)
    X={}
    for n in elig:
        z=by.get(n,[]);a=side_vector([q['L'] for q in z],vL);b=side_vector([q['R'] for q in z],vR);x=np.r_[a,b];x/=x.sum();X[n]=x
    return X

def distance_matrix_dict(X,elig):
    D={}
    for i,a in enumerate(elig):
        for b in elig[i+1:]:D[(a,b)]=D[(b,a)]=float(jensenshannon(X[a],X[b]))
    return D

def null_pools_A(elig,pairs,cnt,dec):
    pools={}
    for a,b in pairs:
        base=[c for c in elig if eskel(c)!=eskel(a) and dec[c]==dec[b]]
        cand=[c for c in base if abs(len(c)-len(b))<=1 and abs(ecount(c)-ecount(b))<=1]
        if not cand:cand=[c for c in base if abs(len(c)-len(b))<=1]
        if not cand:cand=base
        if not cand:raise RuntimeError(f'no A null candidate for {a},{b}')
        pools[(a,b)]=sorted(cand)
    return pools

def low_distance_test(occ,elig,pairs,pools,vL,vR,tag):
    X=frame_vectors(occ,elig,vL,vR);D=distance_matrix_dict(X,elig);obs=float(np.median([D[(a,b)] for a,b in pairs]));rng=np.random.default_rng(seed('A',tag));null=np.empty(NNULL,float)
    for r in range(NNULL):null[r]=np.median([D[(a,pools[(a,b)][int(rng.integers(len(pools[(a,b)])))])] for a,b in pairs])
    sd=float(np.std(null,ddof=1));z=(float(np.mean(null))-obs)/sd if sd>0 else 0.;p=float((1+np.sum(null<=obs))/(NNULL+1))
    return {'pairs':len(pairs),'obs_median_js':obs,'null_mean':float(np.mean(null)),'null_sd':sd,'z':z,'p':p}

def branch_A(segments,tr,cnt,elig,pairs,dec):
    vL=vocab_from(tr,'L');vR=vocab_from(tr,'R');pools=null_pools_A(elig,pairs,cnt,dec)
    full=low_distance_test(tr,elig,pairs,pools,vL,vR,'FULL');ha=low_distance_test(occurrences(segments,'TRAIN','A'),elig,pairs,pools,vL,vR,'A');hb=low_distance_test(occurrences(segments,'TRAIN','B'),elig,pairs,pools,vL,vR,'B')
    gate=bool(full['pairs']>=20 and full['z']>=2.5 and full['p']<=.01 and ha['z']>=1.5 and hb['z']>=1.5)
    return {'full':full,'half_A':ha,'half_B':hb,'gate':gate,'verdict':'A_E_LADDERS_SHARE_TOKEN_FRAMES' if gate else 'A_NO_STRONG_FRAME_SHARING'}

def split_maps(occ,elig):
    frames=collections.defaultdict(collections.Counter);env=collections.defaultdict(lambda:collections.defaultdict(lambda:[collections.Counter(),collections.Counter()]))
    totals=collections.Counter()
    for o in occ:
        n=o['n']
        if n not in elig:continue
        f=o['frame'];frames[n][f]+=1;totals[n]+=1;env[n][f][0][o['XL']]+=1;env[n][f][1][o['XR']]+=1
    return frames,env,totals

def env_vec(counter,vocab,alpha=.5):
    idx={x:i for i,x in enumerate(vocab)};a=np.full(len(vocab)+1,alpha,float)
    for z,c in counter.items():a[idx.get(z,len(vocab))]+=c
    a/=a.sum();return a

def residual_metric(a,b,frames,env,totals,vXL,vXR):
    shared=sorted(set(frames[a])&set(frames[b]),key=str)
    if len(shared)<2:return None
    eff=sum(min(frames[a][f],frames[b][f]) for f in shared)
    if eff<10:return None
    vals=[];ws=[]
    for f in shared:
        xa=np.r_[env_vec(env[a][f][0],vXL),env_vec(env[a][f][1],vXR)];xb=np.r_[env_vec(env[b][f][0],vXL),env_vec(env[b][f][1],vXR)];xa/=xa.sum();xb/=xb.sum();w=min(frames[a][f],frames[b][f]);vals.append(float(jensenshannon(xa,xb)));ws.append(w)
    d=float(np.average(vals,weights=ws));ov=eff/max(1,min(totals[a],totals[b]));return d,float(ov),int(eff),len(shared)

def branch_B_one(occ,elig,pairs,cnt,dec,vXL,vXR,tag):
    frames,env,totals=split_maps(occ,elig);cache={}
    def met(a,b):
        k=(a,b)
        if k not in cache:cache[k]=residual_metric(a,b,frames,env,totals,vXL,vXR)
        return cache[k]
    ep=[(a,b) for a,b in pairs if met(a,b) is not None]
    if not ep:return {'pairs':0,'z':None,'p':1.0,'obs':None}
    obs=float(np.median([met(a,b)[0] for a,b in ep]));pools={}
    for a,b in ep:
        ov=met(a,b)[1];base=[c for c in elig if c!=a and eskel(c)!=eskel(a) and dec[c]==dec[b] and met(a,c) is not None]
        cand=[c for c in base if abs(len(c)-len(b))<=1 and abs(met(a,c)[1]-ov)<=.10]
        if not cand:cand=[c for c in base if abs(len(c)-len(b))<=1 and abs(met(a,c)[1]-ov)<=.20]
        if not cand:cand=[c for c in base if abs(met(a,c)[1]-ov)<=.20]
        if not cand:cand=base
        if not cand:raise RuntimeError(f'no B null candidate for {tag}:{a},{b}')
        pools[(a,b)]=sorted(cand)
    rng=np.random.default_rng(seed('B',tag));null=np.empty(NNULL,float)
    for r in range(NNULL):null[r]=np.median([met(a,pools[(a,b)][int(rng.integers(len(pools[(a,b)])))])[0] for a,b in ep])
    sd=float(np.std(null,ddof=1));z=(float(np.mean(null))-obs)/sd if sd>0 else 0.;p=float((1+np.sum(null<=obs))/(NNULL+1))
    return {'pairs':len(ep),'obs_median_conditional_js':obs,'null_mean':float(np.mean(null)),'null_sd':sd,'z':z,'p':p}

def branch_B(segments,tr,cnt,elig,pairs,dec):
    vXL=sorted({o['XL'] for o in tr if o['XL']!='EDGE'},key=str);vXR=sorted({o['XR'] for o in tr if o['XR']!='EDGE'},key=str)
    full=branch_B_one(tr,elig,pairs,cnt,dec,vXL,vXR,'FULL');ha=branch_B_one(occurrences(segments,'TRAIN','A'),elig,pairs,cnt,dec,vXL,vXR,'A');hb=branch_B_one(occurrences(segments,'TRAIN','B'),elig,pairs,cnt,dec,vXL,vXR,'B')
    if full['pairs']<15:gate=False;ver='B_UNDERPOWERED_WITHIN_FRAME'
    else:
        gate=bool(full['z'] is not None and full['z']>=2.5 and full['p']<=.01 and ha['pairs']>=8 and hb['pairs']>=8 and ha['z'] is not None and hb['z'] is not None and ha['z']>=1.5 and hb['z']>=1.5);ver='B_E_RELATION_PERSISTS_WITHIN_FRAME' if gate else 'B_NO_RESIDUAL_E_SIMILARITY'
    return {'full':full,'half_A':ha,'half_B':hb,'gate':gate,'verdict':ver}

def c_setup(tr,hold,elig):
    bysk=collections.defaultdict(set)
    for n in elig:bysk[eskel(n)].add(ecount(n))
    sks=sorted([s for s,m in bysk.items() if len(m)>=2]);types={n for n in elig if eskel(n) in sks};T=[o for o in tr if o['n'] in types];H=[o for o in hold if o['n'] in types]
    return sks,types,T,H

def outcome_vocab(T,key):return sorted({o[key] for o in T if o[key]!='EDGE'},key=str)+['OTHER','EDGE']
def obin(z,vocab):
    if z=='EDGE':return len(vocab)-1
    try:return vocab.index(z)
    except ValueError:return len(vocab)-2

def aggregate_side(occ,key,vocab):
    agg=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    for o in occ:agg[o['n']][o['frame']][obin(o[key],vocab)]+=1
    return agg

def score_C_side(T,H,key,vocab,mmap):
    V=len(vocab);skc=collections.defaultdict(lambda:np.zeros(V,float));fsc=collections.defaultdict(lambda:np.zeros(V,float));fsm=collections.defaultdict(lambda:np.zeros(V,float))
    for o in T:
        n=o['n'];sk=eskel(n);f=o['frame'];m=mmap[n];j=obin(o[key],vocab);skc[sk][j]+=1;fsc[(f,sk)][j]+=1;fsm[(f,sk,m)][j]+=1
    psk={sk:(a+.5)/(a.sum()+.5*V) for sk,a in skc.items()};ll0=ll1=0.;N=0
    for o in H:
        n=o['n'];sk=eskel(n);f=o['frame'];m=mmap[n];j=obin(o[key],vocab);base=psk[sk];a=fsc[(f,sk)];b=fsm[(f,sk,m)];p0=(a[j]+base[j])/(a.sum()+1.);p1=(b[j]+base[j])/(b.sum()+1.);ll0+=math.log(p0);ll1+=math.log(p1);N+=1
    return ll0/max(1,N),ll1/max(1,N),N

def branch_C(tr,hold,elig):
    sks,types,T,H=c_setup(tr,hold,elig);m0={n:ecount(n) for n in types};vL=outcome_vocab(T,'XL');vR=outcome_vocab(T,'XR')
    a0,a1,nL=score_C_side(T,H,'XL',vL,m0);b0,b1,nR=score_C_side(T,H,'XR',vR,m0);obs=.5*((a1-a0)+(b1-b0))
    groups=collections.defaultdict(list)
    for n in sorted(types):groups[eskel(n)].append(n)
    rng=np.random.default_rng(seed('C_NULL'));null=np.empty(NNULL,float)
    for r in range(NNULL):
        mm={}
        for sk,ls in groups.items():
            vals=[m0[n] for n in ls];rng.shuffle(vals)
            for n,v in zip(ls,vals):mm[n]=v
        x0,x1,_=score_C_side(T,H,'XL',vL,mm);y0,y1,_=score_C_side(T,H,'XR',vR,mm);null[r]=.5*((x1-x0)+(y1-y0))
    p=float((1+np.sum(null>=obs))/(NNULL+1));p99=float(np.quantile(null,.99));gate=bool(len(sks)>=10 and obs>0 and obs>p99 and p<=.01)
    return {'eligible_skeletons':len(sks),'train_occ':len(T),'hold_occ':len(H),'delta':float(obs),'null_mean':float(np.mean(null)),'null_p99':p99,'p':p,'gate':gate,'verdict':'C_ECOUNT_HAS_RESIDUAL_PREDICTIVE_INFORMATION' if gate else 'C_NO_RESIDUAL_ECOUNT_INFORMATION'}

def main():
    v11=load_v11();data=v11['get_json'](v11['DATA_URL']);segments,_lines=v11['build_corpus'](data);tr,cnt,elig,pairs,dec=full_setup(segments);hold=occurrences(segments,'HOLD')
    meta={'eligible_nuclei':len(elig),'e_ladder_pairs':len(pairs),'train_occ':len(tr),'hold_occ':len(hold)};print('V14_META='+json.dumps(meta,sort_keys=True),flush=True)
    A=branch_A(segments,tr,cnt,elig,pairs,dec);print('V14_A='+json.dumps(A,sort_keys=True),flush=True)
    B=branch_B(segments,tr,cnt,elig,pairs,dec);print('V14_B='+json.dumps(B,sort_keys=True),flush=True)
    C=branch_C(tr,hold,elig);print('V14_C='+json.dumps(C,sort_keys=True),flush=True)
    if A['gate'] and not B['gate'] and not C['gate']:ver='V14_E_LADDER_EFFECT_LARGELY_FRAME_MEDIATED'
    elif A['gate'] and (B['gate'] or C['gate']):ver='V14_E_RELATION_EXTENDS_BEYOND_TOKEN_FRAME'
    elif (not A['gate']) and (B['gate'] or C['gate']):ver='V14_RESIDUAL_E_STRUCTURE_WITHOUT_FRAME_MEDIATION'
    else:ver='V14_NO_ADDITIONAL_E_MECHANISM_RESOLVED'
    print('VBM_V14_FINAL_RESULT='+json.dumps({'verdict':ver,'A':A,'B':B,'C':C,'meta':meta,'plaintext_opened':False,'gpu_used':False},sort_keys=True),flush=True)
if __name__=='__main__':main()
