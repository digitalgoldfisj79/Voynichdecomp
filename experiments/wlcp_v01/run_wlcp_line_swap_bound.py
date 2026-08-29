#!/usr/bin/env python3
from __future__ import annotations
import argparse,collections,hashlib,json,math,pickle,re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
SEED=20260829;REPS=5000;EPS=1e-12;MULTI=('ckh','cth','cph','cfh','ikh','ith','iph','ifh','ch','sh')
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
  if len(ts)>=2:out.append(EdgeLine(cur,smap.get(f,'UNK'),ts[0],ts[-1]))
 return out

def units(t,rep):
 if rep=='char':return list(t)
 o=[];i=0
 while i<len(t):
  q=next((u for u in MULTI if t.startswith(u,i)),None)
  if q:o.append(q);i+=len(q)
  else:o.append(t[i]);i+=1
 return o

def choose_disjoint_pairs(lines,rep):
 freq=collections.Counter(t for ln in lines for t in (ln.first,ln.last));S=set(freq);cand=set()
 for t in S:
  u=units(t,rep)
  if len(u)<2:continue
  for su in (u[1:],u[:-1]):
   s=''.join(su)
   if s in S and len(units(s,rep))==len(u)-1:cand.add((s,t))
 ranked=sorted(cand,key=lambda p:(-min(freq[p[0]],freq[p[1]]),-(freq[p[0]]+freq[p[1]]),p[0],p[1]))
 used=set();sel=[]
 for p in ranked:
  if p[0] in used or p[1] in used:continue
  used.update(p);sel.append(p)
 return sel,freq

def mh_log_or(lines,pairs,swaps=None):
 role={}
 for i,(s,l) in enumerate(pairs):role[s]=(i,0);role[l]=(i,1)
 tabs=collections.defaultdict(lambda:[0,0,0,0]) # longI,longF,shortI,shortF
 for j,ln in enumerate(lines):
  a,b=(ln.last,ln.first) if swaps is not None and swaps[j] else (ln.first,ln.last)
  for tok,pos in ((a,'I'),(b,'F')):
   r=role.get(tok)
   if r is None:continue
   pi,islong=r;k=(pi,ln.section)
   if islong:tabs[k][0 if pos=='I' else 1]+=1
   else:tabs[k][2 if pos=='I' else 3]+=1
 num=den=0.0;usable=0;occ=0
 for a,b,c,d in tabs.values():
  n=a+b+c+d
  if n<4 or a+b==0 or c+d==0:continue
  usable+=1;occ+=n;num+=a*d/n;den+=b*c/n
 return float(math.log((num+EPS)/(den+EPS))),usable,occ

def nullsum(obs,v):
 a=np.asarray(v,float);mu=float(a.mean());sd=float(a.std(ddof=1));e=obs-mu;z=e/sd if sd else float('nan');p=(1+int(np.sum(np.abs(a-mu)>=abs(e))))/(len(a)+1)
 return {'observed':float(obs),'null_mean':mu,'effect':float(e),'null_sd':sd,'z':float(z),'p_empirical_2s':p,'reps':len(v),'null_min':float(a.min()),'null_max':float(a.max())}

def test(lines,rep,seed):
 pairs,freq=choose_disjoint_pairs(lines,rep);obs,usable,occ=mh_log_or(lines,pairs);rng=np.random.default_rng(seed);null=[]
 for _ in range(REPS):
  swaps=rng.random(len(lines))<0.5;v,_,_=mh_log_or(lines,pairs,swaps);null.append(v)
 d=nullsum(obs,null);d.update({'selected_disjoint_pairs':len(pairs),'usable_pair_section_strata':usable,'edge_occurrences_in_stat':occ,'null':'independent 0.5 I/F swap within every physical line','pair_selection':'greedy maximum-power lexical matching using edge-token frequencies only; no token occurs in more than one selected pair','effect_scale':'natural-log Mantel-Haenszel common odds ratio; exp(observed) is long-form initial-vs-final odds ratio'});d['observed_odds_ratio']=float(math.exp(obs));return d

def fmt(name,d):
 lead='the metric does not resolve this — ' if abs(d['z'])<2 else '';return f"{lead}{name}: effect={d['effect']:.6f}; matched-null SD={d['null_sd']:.6f}; z={d['z']:.2f}; observed log-OR={d['observed']:.6f} (OR={d['observed_odds_ratio']:.3f}); empirical p={d['p_empirical_2s']:.6f}."

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--section-map',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);sm=json.loads(a.section_map.read_text())['mapping'];all_lines=parse(a.source,sm)
 R={'programme':'WLCP_v0.1_line_swap_bound','seed':SEED,'reps':REPS,'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'section_map_sha256':hashlib.sha256(a.section_map.read_bytes()).hexdigest(),'tests':{}}
 for j,rep in enumerate(('eva','char')):
  R['tests'][rep]={}
  for k,scope in enumerate(('full','A','B')):
   ss=all_lines if scope=='full' else [x for x in all_lines if x.currier==scope];R['tests'][rep][scope]=test(ss,rep,SEED+j*10000+k*100)
 with (a.out/'line_swap_bound.pkl').open('wb') as f:pickle.dump(R,f,pickle.HIGHEST_PROTOCOL)
 pass_all=True
 for scope in ('full','A','B'):
  e=R['tests']['eva'][scope];c=R['tests']['char'][scope];pass_all &= abs(e['z'])>=2 and abs(c['z'])>=2 and np.sign(e['effect'])==np.sign(c['effect'])
 R['replicated_full_A_B_both_representations']=bool(pass_all);(a.out/'LINE_SWAP_RESULTS.json').write_text(json.dumps(R,indent=2))
 L=['# WLCP v0.1 — line-clustered disjoint-family bound','','# RETRACTED FINDINGS','','The exact-margin extension-family test remains valid as a marginal randomisation test, but its nominal variance can be anti-conservative because lexical pairs can share tokens and physical lines. Its z-scores are therefore not used for the final headline; the disjoint-family, within-line-swap results below supersede them for inference.','','# CURRENT FINDINGS','']
 for scope in ('full','A','B'):
  L += [fmt(f'{scope} EVA disjoint-family line-position effect',R['tests']['eva'][scope]),fmt(f'{scope} character disjoint-family line-position effect',R['tests']['char'][scope]),'']
 L += ['## Decision',f"Replicates in full corpus, Currier A, Currier B, and both representations: {'YES' if pass_all else 'NO'}.",'','The null swaps the two observed edge tokens within each physical line, so line composition, section, Currier, token frequencies, word lengths and line-level clustering are all preserved. Lexical extension pairs are disjoint, preventing the same token type from contributing to multiple families. A surviving effect therefore identifies a robust positional rule within related short/long forms. It still cannot distinguish cipher from non-cipher line grammar.']
 (a.out/'LINE_SWAP_RESULTS.md').write_text('\n'.join(L));print('\n'.join(L))
if __name__=='__main__':main()
