#!/usr/bin/env python3
from __future__ import annotations
import argparse,collections,hashlib,json,pickle,re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
SEED=20260829;REPS=2000;MULTI=('ckh','cth','cph','cfh','ikh','ith','iph','ifh','ch','sh')
@dataclass
class Tok: currier:str;section:str;pos:str;token:str

def clean(s):
 s=re.sub(r'<!.*?>','',s)
 for x in ('<%>','<$>','<->'):s=s.replace(x,'')
 s=re.sub(r'<[^>]*>','',s);o=[]
 for x in re.split(r'[\s\.,]+',s.strip()):
  if not x or any(c in x for c in "[]{}?@'/:;0123456789"):continue
  x=re.sub('[^a-z]','',x.lower())
  if x:o.append(x)
 return o

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
  f=left.split('.',1)[0];ts=clean(m.group(2))
  if len(ts)>=2:
   out.append(Tok(cur,smap.get(f,'UNK'),'I',ts[0]));out.append(Tok(cur,smap.get(f,'UNK'),'F',ts[-1]))
 return out

def units(t,rep):
 if rep=='char':return list(t)
 o=[];i=0
 while i<len(t):
  q=next((u for u in MULTI if t.startswith(u,i)),None)
  if q:o.append(q);i+=len(q)
  else:o.append(t[i]);i+=1
 return o

def nullsum(obs,v):
 a=np.asarray(v,float);mu=float(a.mean());sd=float(a.std(ddof=1));e=obs-mu;z=e/sd if sd else float('nan');p=(1+int(np.sum(np.abs(a-mu)>=abs(e))))/(len(a)+1)
 return {'observed':float(obs),'null_mean':mu,'effect':float(e),'null_sd':sd,'z':float(z),'p_empirical_2s':p,'reps':len(v),'null_min':float(a.min()),'null_max':float(a.max())}

def test(tokens,rep,seed):
 counts=collections.defaultdict(lambda:collections.Counter())
 secs=collections.defaultdict(set)
 for x in tokens:counts[(x.currier,x.section,x.token)][x.pos]+=1;secs[x.currier].add(x.section)
 types=sorted({x.token for x in tokens});S=set(types);pairs=set()
 for t in types:
  u=units(t,rep)
  if len(u)<2:continue
  for su in (u[1:],u[:-1]):
   s=''.join(su)
   if s in S and len(units(s,rep))==len(u)-1:pairs.add((s,t))
 rows=[]
 for short,long in sorted(pairs):
  for cur in sorted(secs):
   for sec in sorted(secs[cur]):
    cs=counts[(cur,sec,short)];cl=counts[(cur,sec,long)];sI,sF,lI,lF=cs['I'],cs['F'],cl['I'],cl['F'];nS=sI+sF;nL=lI+lF
    if nS+nL<4 or nS==0 or nL==0 or sI+lI==0 or sF+lF==0:continue
    rows.append((cur,sec,short,long,sI,sF,lI,lF))
 def stat(rr):
  vals=[];w=[]
  for _,_,_,_,sI,sF,lI,lF in rr:
   vals.append(lI/(lI+sI)-lF/(lF+sF));w.append((lI+sI)*(lF+sF)/(lI+sI+lF+sF))
  return float(np.average(vals,weights=w))
 obs=stat(rows);rng=np.random.default_rng(seed);null=[]
 for _ in range(REPS):
  rr=[]
  for cur,sec,s,l,sI,sF,lI,lF in rows:
   nS=sI+sF;nL=lI+lF;nI=sI+lI
   # Exact random-label null conditional on pair, section, Currier, long/short totals and I/F totals.
   newLI=int(rng.hypergeometric(ngood=nL,nbad=nS,nsample=nI));newSI=nI-newLI
   rr.append((cur,sec,s,l,newSI,nS-newSI,newLI,nL-newLI))
  null.append(stat(rr))
 d=nullsum(obs,null);d.update({'pair_strata':len(rows),'unique_pairs':len({(r[2],r[3]) for r in rows}),'null':'hypergeometric exact-margin random-label null','direction':'positive means longer one-edge-extension family member is relatively more line-initial than line-final'});return d

def fmt(name,d):
 lead='the metric does not resolve this — ' if abs(d['z'])<2 else '';return f"{lead}{name}: effect={d['effect']:.6f}; matched-null SD={d['null_sd']:.6f}; z={d['z']:.2f}; observed={d['observed']:.6f}; empirical p={d['p_empirical_2s']:.6f}."

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--section-map',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);sm=json.loads(a.section_map.read_text())['mapping'];t=parse(a.source,sm)
 R={'programme':'WLCP_v0.1_extension_exact_bound','seed':SEED,'reps':REPS,'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'section_map_sha256':hashlib.sha256(a.section_map.read_bytes()).hexdigest(),'tests':{}}
 for j,rep in enumerate(('eva','char')):
  R['tests'][rep]={}
  for k,scope in enumerate(('full','A','B')):
   ss=t if scope=='full' else [x for x in t if x.currier==scope];R['tests'][rep][scope]=test(ss,rep,SEED+j*10000+k*100)
 with (a.out/'extension_exact.pkl').open('wb') as f:pickle.dump(R,f,pickle.HIGHEST_PROTOCOL)
 pass_all=True
 for scope in ('full','A','B'):
  e=R['tests']['eva'][scope];c=R['tests']['char'][scope];pass_all &= abs(e['z'])>=2 and abs(c['z'])>=2 and np.sign(e['effect'])==np.sign(c['effect'])
 R['replicated_both_curriers_both_representations']=bool(pass_all);(a.out/'EXTENSION_EXACT_RESULTS.json').write_text(json.dumps(R,indent=2))
 L=['# WLCP v0.1 — exact extension-family bounding test','','# RETRACTED FINDINGS','','The earlier extension-family result used a uniform feasible-cell draw instead of the correct conditional hypergeometric random-label null. Those z-scores are retracted and replaced below.','','# CURRENT FINDINGS','']
 for scope in ('full','A','B'):
  L += [fmt(f'{scope} EVA extension-family positional effect',R['tests']['eva'][scope]),fmt(f'{scope} character extension-family positional effect',R['tests']['char'][scope]),'']
 L += ['## Decision',f"Replicates in full corpus, Currier A, Currier B, and both representations: {'YES' if pass_all else 'NO'}.",'','This is a morphology-controlled positional test. It does not distinguish cipher from generated/scribal line grammar; it only establishes whether position selects longer versus shorter members within mechanically defined one-edge extension families.']
 (a.out/'EXTENSION_EXACT_RESULTS.md').write_text('\n'.join(L));print('\n'.join(L))
if __name__=='__main__':main()
