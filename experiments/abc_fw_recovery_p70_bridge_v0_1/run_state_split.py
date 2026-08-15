#!/usr/bin/env python3
import json, math, os
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np

SEED=20260813
NPERM=2000
CATS_E2=('empty_core','nonempty_core')
CATS_ED1=('both_empty','mixed','both_nonempty')

@lru_cache(maxsize=None)
def is_ed1(a,b):
    if a==b: return False
    la,lb=len(a),len(b)
    if abs(la-lb)>1: return False
    if la==lb: return sum(x!=y for x,y in zip(a,b))==1
    if la>lb: a,b=b,a; la,lb=lb,la
    i=j=d=0
    while i<la and j<lb:
        if a[i]==b[j]: i+=1; j+=1
        else:
            d+=1; j+=1
            if d>1:return False
    return True

def groups_for(n,null):
    if null=='N0': return [np.arange(n,dtype=np.int32)] if n else []
    if null=='N1':
        if n<5:
            a=np.arange(min(2,n),dtype=np.int32)
            b=np.arange(min(2,n),n,dtype=np.int32)
            return [g for g in (a,b) if len(g)]
        return [np.arange(0,2,dtype=np.int32),np.arange(2,n-2,dtype=np.int32),np.arange(n-2,n,dtype=np.int32)]
    raise ValueError(null)

def load_p70(path='enriched_records.json'):
    obj=json.load(open(path,encoding='utf-8'))
    states=defaultdict(set)
    for r in obj['records']:
        states[r['token']].add(bool(r['empty_core']))
    mapping={t:(1 if next(iter(s)) else 0) for t,s in states.items() if len(s)==1}
    ambiguous={t:sorted(s) for t,s in states.items() if len(s)>1}
    return mapping,ambiguous

def load_p0_zlzi(path='voynich_transcriptions_slim.json'):
    obj=json.load(open(path,encoding='utf-8')); lines=[]
    for page,pd in obj['pages'].items():
        def kf(x):
            try:return (0,int(x))
            except:return (1,str(x))
        for lid in sorted(pd,key=kf):
            rec=pd[lid]
            u=rec.get('u','')
            if not (len(u)>=2 and u[1:]=='P0'): continue
            s=rec.get('t',{}).get('ZLZI')
            if not s: continue
            toks=s.split()
            if toks: lines.append({'page':page,'line':lid,'tokens':toks})
    return lines

def prep(lines,state_map):
    vocab={}; rev=[]
    def tid(t):
        if t not in vocab:
            vocab[t]=len(rev); rev.append(t)
        return vocab[t]
    out=[]; labelled=0; total=0; state_counts=Counter()
    for meta in lines:
        toks=meta['tokens']; ids=np.asarray([tid(t) for t in toks],dtype=np.int32)
        st=np.asarray([state_map.get(t,-1) for t in toks],dtype=np.int8)
        total+=len(toks); labelled+=int((st>=0).sum())
        state_counts['empty_core']+=int((st==1).sum()); state_counts['nonempty_core']+=int((st==0).sum()); state_counts['unlabelled']+=int((st<0).sum())
        n=len(toks); ed=np.zeros((n,n),dtype=bool)
        for i in range(n):
            for j in range(i+1,n):
                if is_ed1(toks[i],toks[j]): ed[i,j]=ed[j,i]=True
        out.append({'ids':ids,'state':st,'ed':ed,'n':n,'meta':meta})
    return out,rev,{'tokens':total,'labelled':labelled,'coverage':labelled/total if total else 0,'state_counts':dict(state_counts)}

def actual(lines):
    e2w=Counter(); e2i=Counter(); edw=Counter(); edi=Counter(); totals=Counter()
    for x in lines:
        ids=x['ids']; st=x['state']; ed=x['ed']; n=x['n']
        if n>=3:
            for i in range(n-2):
                if ids[i]==ids[i+2] and st[i]>=0:
                    e2w['empty_core' if st[i]==1 else 'nonempty_core']+=1
        if n>=7:
            for i in range(2,n-4):
                if ids[i]==ids[i+2] and st[i]>=0:
                    e2i['empty_core' if st[i]==1 else 'nonempty_core']+=1
        if n>=2:
            for i in range(n-1):
                if ed[i,i+1] and st[i]>=0 and st[i+1]>=0:
                    if st[i]==1 and st[i+1]==1:k='both_empty'
                    elif st[i]==0 and st[i+1]==0:k='both_nonempty'
                    else:k='mixed'
                    edw[k]+=1
        if n>=6:
            for i in range(2,n-3):
                if ed[i,i+1] and st[i]>=0 and st[i+1]>=0:
                    if st[i]==1 and st[i+1]==1:k='both_empty'
                    elif st[i]==0 and st[i+1]==0:k='both_nonempty'
                    else:k='mixed'
                    edi[k]+=1
    return {'E2_whole':e2w,'E2_interior':e2i,'ED1_whole':edw,'ED1_interior':edi}

def simulate(lines,null,nperm,seed,need_interior=False):
    rng=np.random.default_rng(seed)
    E2w={k:np.zeros(nperm,dtype=np.int32) for k in CATS_E2}; E2i={k:np.zeros(nperm,dtype=np.int32) for k in CATS_E2}
    EDw={k:np.zeros(nperm,dtype=np.int32) for k in CATS_ED1}; EDi={k:np.zeros(nperm,dtype=np.int32) for k in CATS_ED1}
    for x in lines:
        n=x['n']
        if n<2: continue
        perm=np.broadcast_to(np.arange(n,dtype=np.int32),(nperm,n)).copy()
        for g in groups_for(n,null):
            if len(g)>1:
                keys=rng.random((nperm,len(g))); order=np.argsort(keys,axis=1)
                vals=perm[:,g].copy(); perm[:,g]=np.take_along_axis(vals,order,axis=1)
        ids=x['ids'][perm]; st=x['state'][perm]
        if n>=3:
            eq=ids[:,:-2]==ids[:,2:]
            s0=st[:,:-2]
            E2w['empty_core'] += (eq & (s0==1)).sum(axis=1,dtype=np.int32)
            E2w['nonempty_core'] += (eq & (s0==0)).sum(axis=1,dtype=np.int32)
            if need_interior and n>=7:
                starts=np.arange(2,n-4,dtype=np.int32)
                if len(starts):
                    ei=ids[:,starts]==ids[:,starts+2]; si=st[:,starts]
                    E2i['empty_core'] += (ei & (si==1)).sum(axis=1,dtype=np.int32)
                    E2i['nonempty_core'] += (ei & (si==0)).sum(axis=1,dtype=np.int32)
        p=perm
        em=x['ed'][p[:,:-1],p[:,1:]]
        s1=st[:,:-1]; s2=st[:,1:]
        labelled=(s1>=0)&(s2>=0)
        EDw['both_empty'] += (em & labelled & (s1==1)&(s2==1)).sum(axis=1,dtype=np.int32)
        EDw['both_nonempty'] += (em & labelled & (s1==0)&(s2==0)).sum(axis=1,dtype=np.int32)
        EDw['mixed'] += (em & labelled & (s1!=s2)).sum(axis=1,dtype=np.int32)
        if need_interior and n>=6:
            starts=np.arange(2,n-3,dtype=np.int32)
            if len(starts):
                ei=x['ed'][p[:,starts],p[:,starts+1]]; a=st[:,starts]; b=st[:,starts+1]; lab=(a>=0)&(b>=0)
                EDi['both_empty'] += (ei & lab & (a==1)&(b==1)).sum(axis=1,dtype=np.int32)
                EDi['both_nonempty'] += (ei & lab & (a==0)&(b==0)).sum(axis=1,dtype=np.int32)
                EDi['mixed'] += (ei & lab & (a!=b)).sum(axis=1,dtype=np.int32)
    return {'E2_whole':E2w,'E2_interior':E2i,'ED1_whole':EDw,'ED1_interior':EDi}

def summ(obs,arr):
    a=np.asarray(arr,dtype=float); mu=float(a.mean()); sd=float(a.std(ddof=1)); ratio=float(obs/mu) if mu>0 else None; z=float((obs-mu)/sd) if sd>0 else None
    p=float((1+np.sum(np.abs(a-mu)>=abs(obs-mu)))/(len(a)+1)) if sd>0 else None
    return {'observed':int(obs),'null_mean':mu,'null_sd':sd,'ratio':ratio,'z':z,'empirical_p_two_sided':p}

def main():
    state_map,amb=load_p70(); raw=load_p0_zlzi(); lines,rev,cov=prep(raw,state_map); obs=actual(lines)
    n0=simulate(lines,'N0',NPERM,SEED+31000,False)
    n1=simulate(lines,'N1',NPERM,SEED+32000,True)
    out={'metadata':{'seed':SEED,'nperm':NPERM,'P0_lines':len(lines),'P0_tokens':sum(x['n'] for x in lines),'vocab':len(rev)},'mapping':{'coverage':cov,'ambiguous_P70_types':len(amb)},'S1_E2':{},'S2_ED1':{}}
    for cat in CATS_E2:
        out['S1_E2'][cat]={
            'N0_whole':summ(obs['E2_whole'][cat],n0['E2_whole'][cat]),
            'N1_whole':summ(obs['E2_whole'][cat],n1['E2_whole'][cat]),
            'N3_interior':summ(obs['E2_interior'][cat],n1['E2_interior'][cat])}
    for cat in CATS_ED1:
        out['S2_ED1'][cat]={
            'N0_whole':summ(obs['ED1_whole'][cat],n0['ED1_whole'][cat]),
            'N1_whole':summ(obs['ED1_whole'][cat],n1['ED1_whole'][cat]),
            'N3_interior':summ(obs['ED1_interior'][cat],n1['ED1_interior'][cat])}
    coverage_pass=cov['coverage']>=.98
    e=out['S1_E2']['empty_core'];
    strong=coverage_pass and all(e[k]['ratio'] is not None and e[k]['ratio']>=1.10 and e[k]['z'] is not None and e[k]['z']>=3 for k in ('N1_whole','N3_interior'))
    positional=coverage_pass and e['N0_whole']['ratio'] is not None and e['N0_whole']['ratio']>1 and e['N0_whole']['z'] is not None and e['N0_whole']['z']>0 and any(e[k]['ratio'] is None or e[k]['ratio']<1.10 or e[k]['z'] is None or e[k]['z']<2 for k in ('N1_whole','N3_interior'))
    out['S1_verdict']='INDEPENDENT_EMPTY_CORE_RETURN_SUPPORTED' if strong else ('POSITIONAL_SCAFFOLDING_SUPPORTED' if positional else 'PARTIAL_OR_UNRESOLVED')
    same=[out['S2_ED1']['both_empty']['N3_interior'],out['S2_ED1']['both_nonempty']['N3_interior']]; mixed=out['S2_ED1']['mixed']['N3_interior']
    edstrong=coverage_pass and any(q['ratio'] is not None and q['ratio']>=1.10 and q['z'] is not None and q['z']>=3 for q in same) and (mixed['ratio'] is None or mixed['ratio']<1.10 or mixed['z'] is None or abs(mixed['z'])<2)
    n0same=any(out['S2_ED1'][c]['N0_whole']['ratio'] is not None and out['S2_ED1'][c]['N0_whole']['ratio']>1 for c in ('both_empty','both_nonempty'))
    out['S2_verdict']='INTERIOR_STATE_PRESERVING_ED1_SUPPORTED' if edstrong else ('BOUNDARY_POSITION_COUPLED' if coverage_pass and n0same else 'PARTIAL_OR_UNRESOLVED')
    if out['S1_verdict']=='INDEPENDENT_EMPTY_CORE_RETURN_SUPPORTED' and out['S2_verdict']=='INTERIOR_STATE_PRESERVING_ED1_SUPPORTED': arch='TWO_LOCAL_KERNELS'
    elif out['S1_verdict']!='INDEPENDENT_EMPTY_CORE_RETURN_SUPPORTED' and out['S2_verdict']=='INTERIOR_STATE_PRESERVING_ED1_SUPPORTED': arch='ONE_ED1_LOCAL_KERNEL_PLUS_POSITIONAL_RECURRENCE'
    elif out['S1_verdict']=='INDEPENDENT_EMPTY_CORE_RETURN_SUPPORTED' and out['S2_verdict']!='INTERIOR_STATE_PRESERVING_ED1_SUPPORTED': arch='ONE_EMPTY_CORE_RETURN_KERNEL_PLUS_BOUNDARY_COUPLED_ED1'
    else: arch='NO_NEW_LOCAL_KERNELS_YET'
    out['S3_architecture_consequence']=arch
    p=Path('results/abc_fw_recovery_p70_bridge_v0_1'); p.mkdir(parents=True,exist_ok=True)
    (p/'STATE_SPLIT_RESULTS_20260815.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    md=['# P70 core-state × LAAFU discriminator — results','',f"Mapping coverage: **{cov['coverage']:.4%}** ({cov['labelled']}/{cov['tokens']} P0 ZLZI token occurrences); ambiguous P70 spellings: {len(amb)}.",'','## S1 — exact lag-2 by repeated endpoint core-state','|state|null|observed|null mean|ratio|z|p|','|---|---|---:|---:|---:|---:|---:|']
    for cat in CATS_E2:
        for k in ('N0_whole','N1_whole','N3_interior'):
            q=out['S1_E2'][cat][k]; md.append(f"|{cat}|{k}|{q['observed']}|{q['null_mean']:.3f}|{q['ratio']:.3f}|{q['z']:.2f}|{q['empirical_p_two_sided']:.4g}|")
    md += ['',f"S1 verdict: **{out['S1_verdict']}**.",'','## S2 — ED1 by P70 endpoint core-state','|state|null|observed|null mean|ratio|z|p|','|---|---|---:|---:|---:|---:|---:|']
    for cat in CATS_ED1:
        for k in ('N0_whole','N1_whole','N3_interior'):
            q=out['S2_ED1'][cat][k]; md.append(f"|{cat}|{k}|{q['observed']}|{q['null_mean']:.3f}|{q['ratio']:.3f}|{q['z']:.2f}|{q['empirical_p_two_sided']:.4g}|")
    md += ['',f"S2 verdict: **{out['S2_verdict']}**.",'',f"## Architecture consequence: **{arch}**",'', 'Same-corpus mechanistic follow-up; not independent confirmation. Cross-transliteration robustness remains required before generalisation.']
    (p/'STATE_SPLIT_RESULTS_20260815.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))
if __name__=='__main__':main()
