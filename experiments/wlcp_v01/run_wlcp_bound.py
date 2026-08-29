#!/usr/bin/env python3
from __future__ import annotations
import argparse,collections,hashlib,json,math,pickle,re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
SEED=20260829; REPS=500; EPS=1e-12; MULTI=('ckh','cth','cph','cfh','ikh','ith','iph','ifh','ch','sh')
@dataclass
class Line: folio:str;currier:str;section:str;tokens:list[str]
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
  f=left.split('.',1)[0];t=clean(m.group(2))
  if len(t)>=2:out.append(Line(f,cur,smap.get(f,'UNK'),t))
 return out
def units(t,rep):
 if rep=='char':return list(t)
 o=[];i=0
 while i<len(t):
  q=next((u for u in MULTI if t.startswith(u,i)),None)
  if q:o.append(q);i+=len(q)
  else:o.append(t[i]);i+=1
 return o
def tf(f):return hashlib.sha256(('WLCP:'+f).encode()).digest()[0]%5==0
def pclass(i,n):
 if i==0:return 'I'
 if i==n-1:return 'F'
 return 'M'+str(min(4,int(5*i/max(1,n-1))))
def build(lines,rep,include_section=True):
 rows=[]
 for ln in lines:
  ls=[len(units(t,rep)) for t in ln.tokens];n=len(ls);lb=str(min(n,15))
  for i in range(1,n):
   s=(ln.currier,ln.section if include_section else 'ALL',pclass(i,n),lb);rows.append((ln.folio,s,ls[i-1],ls[i]))
 return rows
def nullsum(obs,v):
 a=np.asarray(v,float);mu=float(a.mean());sd=float(a.std(ddof=1));e=obs-mu;z=e/sd if sd else float('nan');p=(1+int(np.sum(np.abs(a-mu)>=abs(e))))/(len(a)+1)
 return {'observed':float(obs),'null_mean':mu,'effect':float(e),'null_sd':sd,'z':float(z),'p_empirical_2s':p,'reps':len(v),'null_min':float(a.min()),'null_max':float(a.max())}
def arrays(rows):
 strata={s:i for i,s in enumerate(sorted({r[1] for r in rows}))};L=max(max(r[2],r[3]) for r in rows)+1
 f=np.array([tf(r[0]) for r in rows],bool);s=np.array([strata[r[1]] for r in rows],int);p=np.array([r[2] for r in rows],int);y=np.array([r[3] for r in rows],int)
 return f,s,p,y,len(strata),L,strata
def fit(s,p,y,S,L,alpha=.5):
 B=np.full((S,L),alpha);Q=np.full((S,L,L),alpha);np.add.at(B,(s,y),1);np.add.at(Q,(s,p,y),1);B/=B.sum(1,keepdims=True);Q/=Q.sum(2,keepdims=True);return B,Q
def gain(s,p,y,B,Q):return float(np.mean(np.log2(np.maximum(Q[s,p,y],EPS)/np.maximum(B[s,y],EPS))))
def cond_test(lines,rep,seed,include_section=True,reps=REPS):
 rows=build(lines,rep,include_section);f,s,p,y,S,L,_=arrays(rows);tr=~f;te=f;B,Q=fit(s[tr],p[tr],y[tr],S,L);obs=gain(s[te],p[te],y[te],B,Q)
 groups_tr=[np.where(tr & (s==k))[0] for k in range(S)];groups_te=[np.where(te & (s==k))[0] for k in range(S)];rng=np.random.default_rng(seed);v=[]
 for _ in range(reps):
  yp=y.copy()
  for ix in groups_tr:
   if len(ix)>1:yp[ix]=rng.permutation(yp[ix])
  for ix in groups_te:
   if len(ix)>1:yp[ix]=rng.permutation(yp[ix])
  b,q=fit(s[tr],p[tr],yp[tr],S,L);v.append(gain(s[te],p[te],yp[te],b,q))
 d=nullsum(obs,v);d.update({'transitions':len(y),'train':int(tr.sum()),'test':int(te.sum()),'strata':S,'model':'P(L_i | prev_length, currier, section, position_bin, line_length_bin) vs P(L_i | currier, section, position_bin, line_length_bin)' if include_section else 'same without section'});return d
def section_transport(lines,rep):
 rows=build(lines,rep,False);secs=sorted({x.section for x in lines});out=[]
 for sec in secs:
  testfolios={x.folio for x in lines if x.section==sec};trrows=[r for r in rows if r[0] not in testfolios];terows=[r for r in rows if r[0] in testfolios]
  if len(terows)<100 or len(trrows)<100:continue
  allr=trrows+terows;strata={s:i for i,s in enumerate(sorted({r[1] for r in allr}))};L=max(max(r[2],r[3]) for r in allr)+1
  def cv(rs):return np.array([strata[r[1]] for r in rs]),np.array([r[2] for r in rs]),np.array([r[3] for r in rs])
  st,pt,yt=cv(trrows);se,pe,ye=cv(terows);B,Q=fit(st,pt,yt,len(strata),L);out.append({'section':sec,'n_test':len(terows),'gain_bits_per_transition':gain(se,pe,ye,B,Q)})
 return out
def fmt(name,d):
 lead='the metric does not resolve this — ' if abs(d['z'])<2 else '';return f"{lead}{name}: effect={d['effect']:.6f}; matched-null SD={d['null_sd']:.6f}; z={d['z']:.2f}; observed={d['observed']:.6f}."
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--section-map',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 sm=json.loads(a.section_map.read_text())['mapping'];lines=parse(a.source,sm);R={'programme':'WLCP_v0.1_bounding','seed':SEED,'reps':REPS,'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'section_map_sha256':hashlib.sha256(a.section_map.read_bytes()).hexdigest(),'tests':{}}
 for j,rep in enumerate(('eva','char')):
  d={'full':cond_test(lines,rep,SEED+j*10000+1,True),'A':cond_test([x for x in lines if x.currier=='A'],rep,SEED+j*10000+2,True),'B':cond_test([x for x in lines if x.currier=='B'],rep,SEED+j*10000+3,True),'no_section_full':cond_test(lines,rep,SEED+j*10000+4,False),'section_transport':section_transport(lines,rep)};R['tests'][rep]=d
  with (a.out/f'bound_{rep}.pkl').open('wb') as f:pickle.dump(d,f,pickle.HIGHEST_PROTOCOL)
 e=R['tests']['eva'];c=R['tests']['char'];same=np.sign(e['full']['effect'])==np.sign(e['A']['effect'])==np.sign(e['B']['effect']);gate=abs(e['full']['z'])>=2 and abs(e['A']['z'])>=2 and abs(e['B']['z'])>=2 and same and abs(c['full']['z'])>=2 and np.sign(c['full']['effect'])==np.sign(e['full']['effect'])
 R['conditional_transition_gate']=bool(gate);R['endpoint_after_bounding']='WL-1' if gate else 'WL-1-positional-only-or-unresolved';(a.out/'BOUND_RESULTS.json').write_text(json.dumps(R,indent=2))
 L=['# WLCP v0.1 — positional/section bounding audit','','# RETRACTED FINDINGS','']
 if not gate:L+=['The earlier wording “reproducible conditional length structure” is narrowed: adjacency/Markov effects have not survived the preregistered position+section conditioning gate across both Currier systems.','']
 else:L+=['None.','']
 L+=['# CURRENT FINDINGS','',fmt('Position+section-conditioned transition gain, full corpus',e['full']),fmt('Position+section-conditioned transition gain, Currier A',e['A']),fmt('Position+section-conditioned transition gain, Currier B',e['B']),'',fmt('Character-representation conditioned gain, full corpus',c['full']),'','## Section holdout transport (EVA)','']
 for x in e['section_transport']:L.append(f"- {x['section']}: gain={x['gain_bits_per_transition']:.6f} bits/transition; n={x['n_test']}.")
 L+=['','## Decision',f"Conditional transition gate: {'PASS' if gate else 'FAIL'}.",f"Bounded endpoint: {R['endpoint_after_bounding']}",'','This test compares a held-out model using previous-token length against a baseline already conditioned on Currier, manuscript section, normalized within-line position, and line-length bin. Its null preserves the complete target-length distribution inside those same strata while destroying adjacency. It therefore tests whether the earlier transition signal is more than positional/sectional composition.']
 (a.out/'BOUND_RESULTS.md').write_text('\n'.join(L));print('\n'.join(L))
if __name__=='__main__':main()
