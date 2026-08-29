#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, math, pickle, re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
SEED=20260829; REPS=500; POWER_TRIALS=100
MULTI=('ckh','cth','cph','cfh','ikh','ith','iph','ifh','ch','sh')
@dataclass
class Occ:
    folio:str;currier:str;section:str;token:str;pos:str;linebin:str;prev:str|None;nxt:str|None

def clean(s):
    s=re.sub(r'<!.*?>','',s)
    for x in ('<%>','<$>','<->'):s=s.replace(x,'')
    s=re.sub(r'<[^>]*>','',s);o=[]
    for x in re.split(r'[\s\.,]+',s.strip()):
        if not x or any(c in x for c in "[]{}?@'/:;0123456789"):continue
        x=re.sub('[^a-z]','',x.lower())
        if x:o.append(x)
    return o

def units(t,rep='eva'):
    if rep=='char':return tuple(t)
    o=[];i=0
    while i<len(t):
        q=next((u for u in MULTI if t.startswith(u,i)),None)
        if q:o.append(q);i+=len(q)
        else:o.append(t[i]);i+=1
    return tuple(o)

def parse(src,smap):
    out=[];cur='UNK'
    for raw in src.read_text(errors='replace').splitlines():
        if not raw.startswith('<'):continue
        h=re.match(r'^<([^>]+)>\s*<!\s*(.*?)>\s*$',raw)
        if h and '.' not in h.group(1):
            m=re.search(r'\$L=([^\s>]+)',h.group(2));cur=m.group(1) if m else 'UNK';continue
        m=re.match(r'^<([^>]+)>\s*(.*)$',raw)
        if not m or ',' not in m.group(1) or '.' not in m.group(1):continue
        left,code=m.group(1).rsplit(',',1)
        if 'P' not in code:continue
        f=left.split('.',1)[0];ts=clean(m.group(2));n=len(ts)
        if n<2:continue
        lb='S' if n<=6 else ('M' if n<=10 else 'L')
        for i,t in enumerate(ts):
            if i==0:p='I'
            elif i==n-1:p='F'
            else:p='M'+str(min(2,int(3*i/max(1,n-1))))
            out.append(Occ(f,cur,smap.get(f,'UNK'),t,p,lb,ts[i-1] if i else None,ts[i+1] if i+1<n else None))
    return out

def context(o,rep):
    if o.prev is None:pe='^';pl=0
    else:u=units(o.prev,rep);pe=u[-1];pl=min(9,len(u))
    if o.nxt is None:ne='$';nl=0
    else:u=units(o.nxt,rep);ne=u[0];nl=min(9,len(u))
    return (pe,pl,ne,nl)

def discover_pairs(occs):
    freq=collections.Counter(o.token for o in occs);S=set(freq);pairs={}
    for long in S:
        u=units(long,'eva')
        if len(u)<2:continue
        for i,ins in enumerate(u):
            short=''.join(u[:i]+u[i+1:])
            if short in S and short:
                key=(short,long,ins);pairs[key]={'short':short,'long':long,'inserted':ins,'short_n':freq[short],'long_n':freq[long],'total_n':freq[short]+freq[long],'short_len':len(units(short,'eva')),'long_len':len(u)}
    return list(pairs.values())

def disjoint(pairs,target):
    cand=[p for p in pairs if ((p['inserted'] in ('e','i'))==target) and p['total_n']>=8 and min(p['short_n'],p['long_n'])>=2]
    cand.sort(key=lambda p:(-min(p['short_n'],p['long_n']),-p['total_n'],p['short'],p['long'],p['inserted']))
    used=set();sel=[]
    for p in cand:
        if p['short'] in used or p['long'] in used:continue
        used|={p['short'],p['long']};sel.append(p)
    for j,p in enumerate(sel):p['pair_id']=j
    return sel

def mi(x,y):
    n=len(x)
    if n<2:return 0.0
    cx=collections.Counter(x);cy=collections.Counter(y);cxy=collections.Counter(zip(x,y));s=0.0
    for (a,b),c in cxy.items():s+=(c/n)*math.log2(c*n/(cx[a]*cy[b]))
    return s

def cmi(groups):
    N=sum(len(x) for x,y in groups)
    if not N:return float('nan')
    return sum(len(x)/N*mi(x.tolist(),y.tolist()) for x,y in groups)

def prepare(occs,pairs,rep,scope='full',short_len=None):
    ps=[p for p in pairs if short_len is None or p['short_len']==short_len]
    mp={}
    for p in ps:mp[p['short']]=(p['pair_id'],0);mp[p['long']]=(p['pair_id'],1)
    cmap={};raw=collections.defaultdict(list)
    for o in occs:
        if scope!='full' and o.currier!=scope:continue
        q=mp.get(o.token)
        if q is None:continue
        ck=context(o,rep)
        if ck not in cmap:cmap[ck]=len(cmap)
        pid,var=q;z=(pid,o.currier,o.section,o.pos,o.linebin);raw[z].append((var,cmap[ck]))
    groups=[];rows=0
    for z,v in raw.items():
        if len(v)<4:continue
        x=np.array([a for a,b in v],np.int8)
        if x.min()==x.max():continue
        y=np.array([b for a,b in v],np.int32);groups.append((x,y));rows+=len(x)
    return groups,{'selected_pairs':len(ps),'eligible_rows':rows,'eligible_strata':len(groups),'contexts':len(cmap),'scope':scope,'rep':rep,'short_len_filter':short_len}

def perm(groups,seed,reps=REPS,keep_null=False):
    obs=cmi(groups);rng=np.random.default_rng(seed);v=[]
    for _ in range(reps):v.append(cmi([(rng.permutation(x),y) for x,y in groups]))
    a=np.array(v,float);mu=float(a.mean()) if len(a) else float('nan');sd=float(a.std(ddof=1)) if len(a)>1 else float('nan');eff=obs-mu;z=eff/sd if sd>0 else float('nan');p=(1+np.sum(np.abs(a-mu)>=abs(eff)))/(len(a)+1) if len(a) else 1
    d={'observed_cmi_bits':float(obs),'null_mean':mu,'effect_bits':float(eff),'null_sd':sd,'z':float(z),'p_empirical_2s':float(p),'reps':reps}
    if keep_null:d['_null']=v
    return d

def power(groups,null_mean,null_sd,seed):
    rng=np.random.default_rng(seed);out={}
    cutoff=null_mean+2*null_sd
    for OR in (1.5,2.0):
        hit=0;vals=[]
        for _ in range(POWER_TRIALS):
            gg=[]
            for x,y in groups:
                n=len(x);m=int(x.sum())
                if m<=0 or m>=n:gg.append((x.copy(),y));continue
                flag=(y%2).astype(float);w=np.where(flag>0,OR,1.0);w=w/w.sum();choose=rng.choice(np.arange(n),size=m,replace=False,p=w);xx=np.zeros(n,np.int8);xx[choose]=1;gg.append((xx,y))
            val=cmi(gg);vals.append(val);hit+=val>=cutoff
        out[str(OR)]={'recovery_rate_at_null_plus_2sd':hit/POWER_TRIALS,'mean_injected_cmi':float(np.mean(vals)),'cutoff':float(cutoff),'trials':POWER_TRIALS}
    return out

def runset(occs,pairs,rep,target,seed):
    D={}
    for si,scope in enumerate(('full','A','B')):
        gp,meta=prepare(occs,pairs,rep,scope);d=perm(gp,seed+si*100,REPS,keep_null=(scope=='full'));null=d.pop('_null',None);D[scope]={'meta':meta,'test':d}
        if scope=='full' and gp:D[scope]['power']=power(gp,d['null_mean'],d['null_sd'],seed+900)
    for sl in (3,4):
        D[f'{sl}->{sl+1}']={}
        for si,scope in enumerate(('full','A','B')):
            gp,meta=prepare(occs,pairs,rep,scope,sl)
            D[f'{sl}->{sl+1}'][scope]={'meta':meta,'test':perm(gp,seed+1000+sl*100+si*10,REPS) if gp else None}
    return D

def fmt(label,d):
    if d is None:return f'{label}: unavailable (insufficient matched strata).'
    z=d['z'];lead='the metric does not resolve this — ' if not math.isfinite(z) or abs(z)<2 else ''
    return f"{lead}{label}: effect={d['effect_bits']:.6f} bits/occurrence; matched-null SD={d['null_sd']:.6f}; z={z:.2f}; observed={d['observed_cmi_bits']:.6f}; p={d['p_empirical_2s']:.4f}."

def gate(D):
    ok=True
    for rep in ('eva','char'):
        q=D['target'][rep];vals=[q[s]['test'] for s in ('full','A','B')]
        if any(v is None or not math.isfinite(v['z']) or abs(v['z'])<2 for v in vals):ok=False;continue
        if not (np.sign(vals[0]['effect_bits'])==np.sign(vals[1]['effect_bits'])==np.sign(vals[2]['effect_bits'])):ok=False
    return bool(ok)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--section-map',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sm=json.loads(a.section_map.read_text())['mapping'];occs=parse(a.source,sm);pairs=discover_pairs(occs);target=disjoint(pairs,True);control=disjoint(pairs,False)
    R={'programme':'JLCD_near_neighbour_bound','seed':SEED,'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'all_pairs':len(pairs),'target_pairs':target,'control_pairs':control,'target':{},'control':{}}
    for ri,rep in enumerate(('eva','char')):
        R['target'][rep]=runset(occs,target,rep,True,SEED+ri*10000);R['control'][rep]=runset(occs,control,rep,False,SEED+50000+ri*10000)
        with (a.out/f'bound_{rep}.pkl').open('wb') as f:pickle.dump({'target':R['target'][rep],'control':R['control'][rep]},f,pickle.HIGHEST_PROTOCOL)
    R['target_replication_gate']=gate(R)
    (a.out/'NEIGHBOUR_BOUND.json').write_text(json.dumps(R,indent=2,default=str))
    L=['# JLCD v0.1 — exact near-neighbour bound','','# RETRACTED FINDINGS','','None at bound inception.','','# CURRENT FINDINGS','',f"Discovered one-unit pairs={len(pairs)}; selected disjoint e/i pairs={len(target)}; selected disjoint non-e/i pairs={len(control)}.",'']
    for rep in ('eva','char'):
        L += [f'## Exact e/i insertion pairs — {rep}','']
        for s in ('full','A','B'):
            q=R['target'][rep][s];L.append(fmt(s,q['test'])+f" Eligible={q['meta']['eligible_rows']} rows in {q['meta']['eligible_strata']} strata.")
        L += ['','Power injection (full):',json.dumps(R['target'][rep]['full'].get('power',{}),indent=2),'']
        for step in ('3->4','4->5'):
            L.append(f'### {step}')
            for s in ('full','A','B'):
                q=R['target'][rep][step][s];L.append(fmt(s,q['test'])+f" Eligible={q['meta']['eligible_rows']} rows; selected pairs={q['meta']['selected_pairs']}.")
        L += ['','## Non-e/i one-unit insertion control — '+rep,'']
        for s in ('full','A','B'):
            q=R['control'][rep][s];L.append(fmt(s,q['test'])+f" Eligible={q['meta']['eligible_rows']} rows in {q['meta']['eligible_strata']} strata.")
        L.append('')
    L += ['## Decision',f"Exact e/i near-neighbour effect replicates full/A/B and both representations: {'PASS' if R['target_replication_gate'] else 'FAIL'}.",'','This bound is narrower than stripping all e/i: each compared lexical family differs by exactly one attested e or i insertion, no token type participates in more than one selected family, and labels are permuted only within pair × Currier × section × position × line-length strata.']
    (a.out/'NEIGHBOUR_BOUND.md').write_text('\n'.join(L));print('\n'.join(L))
if __name__=='__main__':main()
