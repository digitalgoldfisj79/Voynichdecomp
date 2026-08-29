#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, math, pickle, re, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_jlcd_neighbour_bound as nb
SEED=20260829;REPS=5000;EPS=1e-12
@dataclass
class EdgeLine: currier:str;section:str;first:str;last:str

def clean(s):
    s=re.sub(r'<!.*?>','',s)
    for x in ('<%>','<$>','<->'):s=s.replace(x,'')
    s=re.sub(r'<[^>]*>','',s);o=[]
    for x in re.split(r'[\s\.,]+',s.strip()):
        if not x or any(c in x for c in "[]{}?@'/:;0123456789"):continue
        x=re.sub('[^a-z]','',x.lower())
        if x:o.append(x)
    return o

def parse_lines(src,smap):
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
        f=left.split('.',1)[0];ts=clean(m.group(2))
        if len(ts)>=2:out.append(EdgeLine(cur,smap.get(f,'UNK'),ts[0],ts[-1]))
    return out

def mh(lines,pairs,swaps=None):
    role={}
    for p in pairs:role[p['short']]=(p['pair_id'],0);role[p['long']]=(p['pair_id'],1)
    tabs=collections.defaultdict(lambda:[0,0,0,0])
    for j,ln in enumerate(lines):
        a,b=(ln.last,ln.first) if swaps is not None and swaps[j] else (ln.first,ln.last)
        for tok,pos in ((a,'I'),(b,'F')):
            r=role.get(tok)
            if r is None:continue
            pid,long=r;k=(pid,ln.section)
            if long:tabs[k][0 if pos=='I' else 1]+=1
            else:tabs[k][2 if pos=='I' else 3]+=1
    num=den=0.;usable=occ=0
    for a,b,c,d in tabs.values():
        n=a+b+c+d
        if n<4 or a+b==0 or c+d==0:continue
        usable+=1;occ+=n;num+=a*d/n;den+=b*c/n
    return float(math.log((num+EPS)/(den+EPS))),usable,occ

def test(lines,pairs,seed):
    obs,u,o=mh(lines,pairs);rng=np.random.default_rng(seed);v=[]
    for _ in range(REPS):
        swaps=rng.random(len(lines))<.5;v.append(mh(lines,pairs,swaps)[0])
    a=np.asarray(v);mu=float(a.mean());sd=float(a.std(ddof=1));eff=obs-mu;z=eff/sd if sd else float('nan');p=(1+np.sum(np.abs(a-mu)>=abs(eff)))/(REPS+1)
    return {'observed_log_or':obs,'observed_or':float(math.exp(obs)),'null_mean':mu,'effect_log_or':float(eff),'null_sd':sd,'z':float(z),'p_empirical_2s':float(p),'usable_pair_section_strata':u,'edge_occurrences':o,'selected_pairs':len(pairs)}

def fmt(label,d):
    lead='the metric does not resolve this — ' if abs(d['z'])<2 else ''
    return f"{lead}{label}: effect={d['effect_log_or']:.6f} log-odds; matched-null SD={d['null_sd']:.6f}; z={d['z']:.2f}; observed OR={d['observed_or']:.3f}; p={d['p_empirical_2s']:.4f}."

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--section-map',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sm=json.loads(a.section_map.read_text())['mapping'];occs=nb.parse(a.source,sm);pairs=nb.discover_pairs(occs);target=nb.disjoint(pairs,True);control=nb.disjoint(pairs,False);lines=parse_lines(a.source,sm)
    R={'programme':'JLCD_position_counter_bound','source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'target_pairs':len(target),'control_pairs':len(control),'target':{},'control':{}}
    for si,scope in enumerate(('full','A','B')):
        ls=lines if scope=='full' else [x for x in lines if x.currier==scope]
        R['target'][scope]=test(ls,target,SEED+si*100);R['control'][scope]=test(ls,control,SEED+10000+si*100)
    targetpass=all(abs(R['target'][s]['z'])>=2 and R['target'][s]['effect_log_or']>0 for s in ('full','A','B'));R['target_positional_replication']=targetpass
    with (a.out/'position_counter_bound.pkl').open('wb') as f:pickle.dump(R,f,pickle.HIGHEST_PROTOCOL)
    (a.out/'POSITION_COUNTER_BOUND.json').write_text(json.dumps(R,indent=2))
    L=['# JLCD v0.1 — positional counter bound','','# RETRACTED FINDINGS','','None.','','# CURRENT FINDINGS','',f"Selected disjoint e/i insertion pairs={len(target)}; non-e/i insertion pairs={len(control)}.",'','## e/i insertion families','']
    for s in ('full','A','B'):L.append(fmt(s,R['target'][s])+f" Edge occurrences={R['target'][s]['edge_occurrences']}.")
    L += ['','## non-e/i insertion families','']
    for s in ('full','A','B'):L.append(fmt(s,R['control'][s])+f" Edge occurrences={R['control'][s]['edge_occurrences']}.")
    L += ['','## Decision',f"e/i additions reproduce the robust positional length rule across full/A/B: {'YES' if targetpass else 'NO'}.",'','The distinguishing test is whether the specific e/i lengthening operation proposed as a counter exhibits the same line-initial preference previously found for related short/long forms generally. The null reverses only the two observed edge tokens inside each physical line, preserving all line-level content.']
    (a.out/'POSITION_COUNTER_BOUND.md').write_text('\n'.join(L));print('\n'.join(L))
if __name__=='__main__':main()
