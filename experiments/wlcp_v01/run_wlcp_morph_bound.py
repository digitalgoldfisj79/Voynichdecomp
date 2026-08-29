#!/usr/bin/env python3
from __future__ import annotations
import argparse,collections,hashlib,json,pickle,re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
SEED=20260829;REPS=500;MULTI=('ckh','cth','cph','cfh','ikh','ith','iph','ifh','ch','sh')
@dataclass
class Tok: folio:str;currier:str;section:str;pos:str;token:str

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
  f=left.split('.',1)[0];ts=clean(m.group(2));n=len(ts)
  if n<2:continue
  out.append(Tok(f,cur,smap.get(f,'UNK'),'I',ts[0]));out.append(Tok(f,cur,smap.get(f,'UNK'),'F',ts[-1]))
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

def stratified(tokens,rep,depth,seed):
 rows=[]
 for x in tokens:
  u=units(x.token,rep);k=min(depth,len(u));edge=(tuple(u[:k]),tuple(u[-k:]));rows.append([x.pos,len(u),(x.currier,x.section,edge)])
 G=collections.defaultdict(list)
 for i,r in enumerate(rows):G[r[2]].append(i)
 keep=[]
 for ix in G.values():
  ps={rows[i][0] for i in ix}
  if ps=={'I','F'}:keep.extend(ix)
 keep=np.array(sorted(keep),int);pos=np.array([rows[i][0] for i in keep]);L=np.array([rows[i][1] for i in keep],float);strat=[rows[i][2] for i in keep]
 obs=float(L[pos=='I'].mean()-L[pos=='F'].mean());groups=collections.defaultdict(list)
 for j,s in enumerate(strat):groups[s].append(j)
 rng=np.random.default_rng(seed);v=[]
 for _ in range(REPS):
  pp=pos.copy()
  for ix in groups.values():
   if len(ix)>1:pp[ix]=rng.permutation(pp[ix])
  v.append(float(L[pp=='I'].mean()-L[pp=='F'].mean()))
 d=nullsum(obs,v);d.update({'edge_depth':depth,'matched_tokens':len(keep),'matched_strata':len(groups),'I':int(np.sum(pos=='I')),'F':int(np.sum(pos=='F'))});return d

def extpairs(tokens,rep,seed):
 counts=collections.defaultdict(lambda:collections.Counter())
 for x in tokens:counts[(x.currier,x.section,x.token)][x.pos]+=1
 types=sorted({x.token for x in tokens});S=set(types);pairs=set()
 for t in types:
  u=units(t,rep)
  if len(u)<2:continue
  for su in (u[1:],u[:-1]):
   s=''.join(su)
   if s in S:pairs.add((s,t) if len(units(s,rep))<len(u) else (t,s))
 pair_rows=[]
 for short,long in sorted(pairs):
  for cur in ('A','B'):
   secs={x.section for x in tokens if x.currier==cur}
   for sec in secs:
    cS=counts[(cur,sec,short)];cL=counts[(cur,sec,long)];tot=sum(cS.values())+sum(cL.values())
    if tot<4 or (cS['I']+cL['I']==0) or (cS['F']+cL['F']==0) or sum(cS.values())==0 or sum(cL.values())==0:continue
    pair_rows.append((cur,sec,short,long,cS['I'],cS['F'],cL['I'],cL['F']))
 def stat(rows):
  vals=[];weights=[]
  for _,_,_,_,sI,sF,lI,lF in rows:
   ri=lI/max(1,lI+sI);rf=lF/max(1,lF+sF);w=(lI+sI)*(lF+sF)/max(1,lI+sI+lF+sF);vals.append(ri-rf);weights.append(w)
  return float(np.average(vals,weights=weights)) if vals else float('nan')
 obs=stat(pair_rows);rng=np.random.default_rng(seed);v=[]
 for _ in range(REPS):
  rr=[]
  for cur,sec,s,l,sI,sF,lI,lF in pair_rows:
   nS=sI+sF;nL=lI+lF;nI=sI+lI;n=nS+nL
   lo=max(0,nI-nS);hi=min(nL,nI);newLI=int(rng.integers(lo,hi+1)) if hi>=lo else lI;newSI=nI-newLI
   rr.append((cur,sec,s,l,newSI,nS-newSI,newLI,nL-newLI))
  v.append(stat(rr))
 d=nullsum(obs,v);d.update({'pair_strata':len(pair_rows),'unique_pairs':len({(r[2],r[3]) for r in pair_rows}),'interpretation':'positive = within one-edge extension families, the longer member is relatively more common line-initial than line-final'});return d

def fmt(name,d):
 lead='the metric does not resolve this — ' if abs(d['z'])<2 else '';return f"{lead}{name}: effect={d['effect']:.6f}; matched-null SD={d['null_sd']:.6f}; z={d['z']:.2f}; observed={d['observed']:.6f}."

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--section-map',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);sm=json.loads(a.section_map.read_text())['mapping'];t=parse(a.source,sm)
 R={'programme':'WLCP_v0.1_morph_position_bound','source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'section_map_sha256':hashlib.sha256(a.section_map.read_bytes()).hexdigest(),'tests':{}}
 for j,rep in enumerate(('eva','char')):
  R['tests'][rep]={}
  for scope in ('full','A','B'):
   ss=t if scope=='full' else [x for x in t if x.currier==scope];R['tests'][rep][scope]={'edge1':stratified(ss,rep,1,SEED+j*10000+len(scope)+1),'edge2':stratified(ss,rep,2,SEED+j*10000+len(scope)+2),'extension_pairs':extpairs(ss,rep,SEED+j*10000+len(scope)+3)}
  with (a.out/f'morph_bound_{rep}.pkl').open('wb') as f:pickle.dump(R['tests'][rep],f,pickle.HIGHEST_PROTOCOL)
 e=R['tests']['eva'];survive=abs(e['B']['edge2']['z'])>=2 and abs(R['tests']['char']['B']['edge2']['z'])>=2;R['B_position_survives_edge2_matching']=bool(survive);(a.out/'MORPH_BOUND_RESULTS.json').write_text(json.dumps(R,indent=2))
 L=['# WLCP v0.1 — morphology/position bounding audit','','# RETRACTED FINDINGS','','None.','','# CURRENT FINDINGS','']
 for scope in ('full','A','B'):
  L += [f'## {scope}',fmt('Position effect after section+Currier+first/last-unit matching',e[scope]['edge1']),fmt('Position effect after section+Currier+first2/last2-unit matching',e[scope]['edge2']),fmt('Within one-edge-extension families: long-form initial preference',e[scope]['extension_pairs']),'']
 L+=['## Decision',f"Currier-B positional length effect survives two-edge morphological matching in both representations: {'YES' if survive else 'NO'}.",'','Interpretation test: if the initial/final length effect were only a by-product of section mixture and edge-glyph composition, permuting I/F labels within Currier × section × matched edge-morphology strata would reproduce it. A surviving effect rejects that restricted composition null. The extension-family test is stricter in a different direction: it asks whether, within mechanically related short/long token pairs, the longer form is preferentially line-initial. Neither result by itself distinguishes cipher from non-cipher line grammar.']
 (a.out/'MORPH_BOUND_RESULTS.md').write_text('\n'.join(L));print('\n'.join(L))
if __name__=='__main__':main()
