#!/usr/bin/env python3
import json, os, math
import numpy as np
from collections import defaultdict

SEED=20260813
DATA='/mnt/data/joint_lag/voynich_transcriptions_slim.json'
OUT='/mnt/data/joint_lag/v04_ed1'

def subtype(u):
    if not u: return ''
    return u[1:] if len(u)>=2 else ''

def load_p0(path=DATA, frame='ZLZI'):
    obj=json.load(open(path,encoding='utf-8')); lines=[]
    for page,pd in obj['pages'].items():
        def kf(x):
            try:return (0,int(x))
            except:return (1,str(x))
        for lid in sorted(pd,key=kf):
            rec=pd[lid]
            if subtype(rec.get('u',''))!='P0': continue
            s=rec.get('t',{}).get(frame)
            if s:
                toks=s.split()
                if toks: lines.append(toks)
    return lines

def ed1(a,b):
    if a==b: return False
    la,lb=len(a),len(b)
    if abs(la-lb)>1:return False
    if la==lb:return sum(x!=y for x,y in zip(a,b))==1
    if la>lb:a,b=b,a;la,lb=lb,la
    i=j=0; skipped=0
    while i<la and j<lb:
        if a[i]==b[j]: i+=1;j+=1
        else:
            skipped+=1;j+=1
            if skipped>1:return False
    return True

def groups(n,null):
    if null=='N0': return [np.arange(n,dtype=np.int32)]
    if null=='N1':
        if n<5:
            return [g for g in (np.arange(min(2,n),dtype=np.int32),np.arange(min(2,n),n,dtype=np.int32)) if len(g)]
        return [np.arange(0,2,dtype=np.int32),np.arange(2,n-2,dtype=np.int32),np.arange(n-2,n,dtype=np.int32)]
    raise ValueError(null)

def prep(lines):
    out=[]
    for toks in lines:
        uniq=list(dict.fromkeys(toks)); mp={t:i for i,t in enumerate(uniq)}
        ids=np.asarray([mp[t] for t in toks],dtype=np.int16 if len(uniq)<32767 else np.int32)
        R=np.zeros((len(uniq),len(uniq)),dtype=bool)
        for i,a in enumerate(uniq):
            for j in range(i+1,len(uniq)):
                if ed1(a,uniq[j]): R[i,j]=R[j,i]=True
        out.append((ids,R,uniq))
    return out

def actual(prepared,interior=False):
    s=d=0
    for ids,R,_ in prepared:
        n=len(ids)
        if interior:
            starts=np.arange(2,n-3,dtype=np.int32) if n>=6 else np.array([],dtype=np.int32)
        else:
            starts=np.arange(0,n-1,dtype=np.int32)
        if len(starts):
            s+=int(R[ids[starts],ids[starts+1]].sum()); d+=len(starts)
    return s,d

def simulate(prepared,null,nperm,seed,interior=False):
    rng=np.random.default_rng(seed); totals=np.zeros(nperm,dtype=np.int32)
    for ids,R,_ in prepared:
        n=len(ids)
        if n<2: continue
        mat=np.broadcast_to(ids,(nperm,n)).copy()
        for g in groups(n,null):
            if len(g)>1:
                keys=rng.random((nperm,len(g))); order=np.argsort(keys,axis=1)
                vals=mat[:,g].copy(); mat[:,g]=np.take_along_axis(vals,order,axis=1)
        if interior:
            starts=np.arange(2,n-3,dtype=np.int32) if n>=6 else np.array([],dtype=np.int32)
        else: starts=np.arange(0,n-1,dtype=np.int32)
        if len(starts): totals += R[mat[:,starts],mat[:,starts+1]].sum(axis=1,dtype=np.int32)
    return totals

def summ(obs,arr,denom):
    arr=np.asarray(arr,dtype=float); mu=arr.mean(); sd=arr.std(ddof=1)
    return {'actual':int(obs),'denom':int(denom),'rate':obs/denom if denom else None,'null_mean':float(mu),'null_sd':float(sd),'null_rate':float(mu/denom) if denom else None,'ratio':float(obs/mu) if mu else None,'z':float((obs-mu)/sd) if sd else None}

def analyse(lines,null,nperm,seed,interior=False):
    p=prep(lines); obs,den=actual(p,interior); arr=simulate(p,null,nperm,seed,interior)
    return summ(obs,arr,den)

def permute_lines(lines,null,seed):
    rng=np.random.default_rng(seed); out=[]
    for toks in lines:
        a=np.array(toks,dtype=object)
        for g in groups(len(a),null):
            if len(g)>1:a[g]=rng.permutation(a[g])
        out.append(a.tolist())
    return out

def calibrate(lines,null,interior,seed):
    ratios=[];zs=[]; reps=[]
    for r in range(20):
        pseudo=permute_lines(lines,null,seed+10000*r)
        rr=analyse(pseudo,null,200,seed+10000*r+1,interior)
        ratios.append(rr['ratio']);zs.append(rr['z']);reps.append(rr)
    mr=float(np.nanmean(ratios)); nz=sum(abs(z)>=2 for z in zs if z is not None and math.isfinite(z))
    return {'mean_ratio':mr,'n_abs_z_ge2':nz,'pass':bool(.95<=mr<=1.05 and nz<=2),'replicates':reps}

def inject(lines,frac,seed):
    rng=np.random.default_rng(seed); out=[x[:] for x in lines]; cand=[]
    for li,t in enumerate(out):
        n=len(t)
        for i in range(2,max(2,n-3)):
            cand.append((li,i))
    rng.shuffle(cand); target=max(1,int(frac*len(cand))); occupied=defaultdict(set); chosen=[]
    for li,i in cand:
        if {i,i+1}.isdisjoint(occupied[li]):
            chosen.append((li,i));occupied[li].update([i,i+1])
            if len(chosen)>=target:break
    for li,i in chosen:
        out[li][i+1]=out[li][i]+'§'
    return out,len(chosen)

def main():
    os.makedirs(OUT,exist_ok=True); lines=load_p0()
    print('lines tokens',len(lines),sum(map(len,lines)),flush=True)
    controls={}
    controls['C0_N0']=calibrate(lines,'N0',False,SEED+100)
    controls['C0_N1']=calibrate(lines,'N1',False,SEED+200)
    controls['C0_N3']=calibrate(lines,'N1',True,SEED+300)
    pseudo=permute_lines(lines,'N1',SEED+400); inj,ns=inject(pseudo,.02,SEED+401)
    c1=analyse(inj,'N1',1000,SEED+402,True); controls['C1']={'n_sites':ns,'stat':c1,'pass':bool(c1['ratio']>=1.15 and c1['z']>=3)}
    res={'N0':analyse(lines,'N0',1000,SEED+1000,False),'N1':analyse(lines,'N1',1000,SEED+2000,False),'N3':analyse(lines,'N1',1000,SEED+3000,True)}
    valid=all(x['pass'] for x in controls.values())
    strong=valid and res['N1']['ratio']>=1.10 and res['N1']['z']>=3 and res['N3']['ratio']>=1.10 and res['N3']['z']>=3
    partial=valid and ((res['N1']['ratio']>=1.10 and res['N1']['z']>=3) != (res['N3']['ratio']>=1.10 and res['N3']['z']>=3))
    dec={'H1_VMS_ED1_INLINE': 'STRONG_SUPPORT' if strong else ('PARTIAL' if partial else 'UNSUPPORTED'),'valid':valid}
    out={'seed':SEED,'lines':len(lines),'tokens':sum(map(len,lines)),'controls':controls,'ED1':res,'decision':dec}
    fn=os.path.join(OUT,'RESULTS_p0_ed1_laafu_v0_1_20260813.json');json.dump(out,open(fn,'w'),indent=2)
    print('DEC',dec,'WROTE',fn,flush=True)
if __name__=='__main__': main()
