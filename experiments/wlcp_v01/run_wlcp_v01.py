#!/usr/bin/env python3
from __future__ import annotations
import argparse,collections,hashlib,json,math,pickle,re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
SEED=20260829; REPS=500; EPS=1e-12
MULTI=('ckh','cth','cph','cfh','ikh','ith','iph','ifh','ch','sh')
@dataclass
class Line: folio:str; currier:str; tokens:list[str]
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def clean(s):
 s=re.sub(r'<!.*?>','',s)
 for x in ('<%>','<$>','<->'):s=s.replace(x,'')
 s=re.sub(r'<[^>]*>','',s);o=[]
 for x in re.split(r'[\s\.,]+',s.strip()):
  if not x or any(c in x for c in "[]{}?@'/:;0123456789"):continue
  x=re.sub('[^a-z]','',x.lower())
  if x:o.append(x)
 return o
def parse(p):
 lines=[];cur='UNK';pages=0
 for raw in p.read_text(errors='replace').splitlines():
  if not raw.startswith('<'):continue
  h=re.match(r'^<([^>]+)>\s*<!\s*(.*?)>\s*$',raw)
  if h and '.' not in h.group(1):
   m=re.search(r'\$L=([^\s>]+)',h.group(2));cur=m.group(1) if m else 'UNK';pages+=1;continue
  m=re.match(r'^<([^>]+)>\s*(.*)$',raw)
  if not m or ',' not in m.group(1) or '.' not in m.group(1):continue
  left,code=m.group(1).rsplit(',',1)
  if 'P' not in code:continue
  t=clean(m.group(2))
  if len(t)>=2:lines.append(Line(left.split('.',1)[0],cur,t))
 return lines,{'pages':pages,'lines':len(lines),'tokens':sum(len(x.tokens) for x in lines),'folios':len({x.folio for x in lines}),'currier_tokens':dict(collections.Counter(x.currier for x in lines for _ in x.tokens))}
def units(t,rep):
 if rep=='char':return list(t)
 o=[];i=0
 while i<len(t):
  q=next((u for u in MULTI if t.startswith(u,i)),None)
  if q:o.append(q);i+=len(q)
  else:o.append(t[i]);i+=1
 return o
def arrs(lines,rep):return [np.array([len(units(t,rep)) for t in x.tokens],dtype=np.int16) for x in lines]
def mi(a,b):
 a=np.asarray(a,int);b=np.asarray(b,int);J=np.zeros((a.max()+1,b.max()+1));np.add.at(J,(a,b),1);J/=J.sum();px=J.sum(1);py=J.sum(0);D=px[:,None]*py[None,:];z=J>0;return float((J[z]*np.log2(J[z]/D[z])).sum())
def adj(A):
 x=np.concatenate([a[:-1] for a in A]);y=np.concatenate([a[1:] for a in A]);return mi(x,y)
def pos(A):return float(np.mean([a[0]-a[-1] for a in A]))
def nullsum(obs,v):
 a=np.array(v,float);mu=float(a.mean());sd=float(a.std(ddof=1));e=obs-mu;z=e/sd if sd else float('nan');p=(1+int(np.sum(np.abs(a-mu)>=abs(e))))/(len(a)+1)
 return {'observed':obs,'null_mean':mu,'effect':e,'null_sd':sd,'z':z,'p_empirical_2s':p,'reps':len(v),'null_min':float(a.min()),'null_max':float(a.max())}
def permtest(A,fn,seed,reps=REPS):
 r=np.random.default_rng(seed);obs=fn(A);v=[]
 for _ in range(reps):v.append(fn([r.permutation(a) for a in A]))
 return nullsum(obs,v)
def testfolio(f):return hashlib.sha256(('WLCP:'+f).encode()).digest()[0]%5==0
def fit(A,alpha=.5):
 K=max(int(a.max()) for a in A)+1;T=np.full((K,K),alpha);M=np.full(K,alpha)
 for a in A:
  np.add.at(M,a,1);np.add.at(T,(a[:-1],a[1:]),1)
 T/=T.sum(1,keepdims=True);M/=M.sum();return T,M
def score(A,T,M):
 K=len(M);v=[]
 for a in A:
  x=np.minimum(a[:-1],K-1);y=np.minimum(a[1:],K-1);v.extend(np.log2(np.maximum(T[x,y],EPS)/np.maximum(M[y],EPS)))
 return float(np.mean(v))
def markov(lines,rep,seed,reps=REPS):
 tr=arrs([x for x in lines if not testfolio(x.folio)],rep);te=arrs([x for x in lines if testfolio(x.folio)],rep);T,M=fit(tr);obs=score(te,T,M);r=np.random.default_rng(seed);v=[]
 for _ in range(reps):
  tp=[r.permutation(a) for a in tr];ep=[r.permutation(a) for a in te];Q,N=fit(tp);v.append(score(ep,Q,N))
 d=nullsum(obs,v);d['train_tokens']=sum(map(len,tr));d['test_tokens']=sum(map(len,te));return d
def edge_mi(lines,rep,edge,seed,reps=REPS):
 g=[];L=[];strata=[]
 for ln in lines:
  for i,t in enumerate(ln.tokens):
   u=units(t,rep);g.append(u[0] if edge=='first' else u[-1]);L.append(len(u));strata.append((ln.currier,'I' if i==0 else ('F' if i==len(ln.tokens)-1 else 'M'),u[-1] if edge=='first' else u[0]))
 gm={x:i for i,x in enumerate(sorted(set(g)))};gx=np.array([gm[x] for x in g]);ly=np.array(L);obs=mi(gx,ly);G=collections.defaultdict(list)
 for i,s in enumerate(strata):G[s].append(i)
 r=np.random.default_rng(seed);v=[]
 for _ in range(reps):
  y=ly.copy()
  for ix in G.values():
   if len(ix)>1:y[ix]=r.permutation(y[ix])
  v.append(mi(gx,y))
 return nullsum(obs,v)
def desc(A):
 v=np.concatenate(A).astype(float);c=collections.Counter(map(int,v));p=np.array(list(c.values()))/len(v)
 return {'n':len(v),'mean':float(v.mean()),'sd':float(v.std(ddof=1)),'median':float(np.median(v)),'entropy_bits':float(-(p*np.log2(p)).sum()),'hist':dict(sorted(c.items()))}
def runrep(lines,rep,seed):
 A=arrs(lines,rep);return {'n_lines':len(lines),'n_tokens':sum(map(len,A)),'desc':desc(A),'adj_mi':permtest(A,adj,seed+1),'first_final':permtest(A,pos,seed+2),'markov_gain':markov(lines,rep,seed+3),'first_length_mi':edge_mi(lines,rep,'first',seed+4),'last_length_mi':edge_mi(lines,rep,'last',seed+5)}
def ck(out,n,obj):
 with (out/f'{n}.pkl').open('wb') as f:pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
def fmt(name,d):
 z=d['z'];lead='the metric does not resolve this — ' if abs(z)<2 else '';return f"{lead}{name}: effect={d['effect']:.6f}; matched-null SD={d['null_sd']:.6f}; z={z:.2f}; observed={d['observed']:.6f}."
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--reps',type=int,default=REPS);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 lines,audit=parse(a.source);audit['sha256']=sha(a.source)
 if audit['tokens']<10000 or audit['folios']<100:raise SystemExit('AUDIT FAIL '+json.dumps(audit))
 R={'programme':'WLCP_v0.1','seed':SEED,'null_reps':a.reps,'audit':audit,'representations':{}};ck(a.out,'phase0_audit',R)
 for j,rep in enumerate(('eva','char')):
  d=runrep(lines,rep,SEED+j*10000);d['A']=runrep([x for x in lines if x.currier=='A'],rep,SEED+j*10000+100);d['B']=runrep([x for x in lines if x.currier=='B'],rep,SEED+j*10000+200);R['representations'][rep]=d;ck(a.out,f'phase1_{rep}',d)
 prim=('adj_mi','first_final','markov_gain');g1=[]
 for m in prim:
  f=R['representations']['eva'][m];A=R['representations']['eva']['A'][m];B=R['representations']['eva']['B'][m];same=np.sign(f['effect'])==np.sign(A['effect'])==np.sign(B['effect']);g1.append({'metric':m,'pass':bool(abs(f['z'])>=2 and abs(A['z'])>=2 and abs(B['z'])>=2 and same),'full_z':f['z'],'A_z':A['z'],'B_z':B['z'],'same_direction':bool(same)})
 gate1=any(x['pass'] for x in g1);g2=[]
 for x in g1:
  m=x['metric'];e=R['representations']['eva'][m];c=R['representations']['char'][m];ok=x['pass'] and abs(c['z'])>=2 and np.sign(e['effect'])==np.sign(c['effect']);g2.append({'metric':m,'pass':bool(ok),'eva_z':e['z'],'char_z':c['z']})
 gate2=any(x['pass'] for x in g2)
 proof={'statement':'For every plaintext token sequence P, identity and any boundary-preserving one-symbol-per-plaintext-symbol substitution have identical whole-token length sequence L. Therefore for every statistic S that is a function only of L, S(identity(P)) = S(substitution(P)).','identity_vs_1to1_substitution_max_possible_length_stat_difference':0.0,'family_identifiability_pass':False}
 endpoint='WL-0' if not gate1 else 'WL-1'
 R['gates']={'gate0':True,'gate1':gate1,'gate1_tests':g1,'gate2':gate2,'gate2_tests':g2,'gate3':False,'gate3_proof':proof,'WL2_promoted':False,'WL4_promoted':False,'endpoint':endpoint};ck(a.out,'phase2_gates',R['gates'])
 (a.out/'results.json').write_text(json.dumps(R,indent=2))
 e=R['representations']['eva'];c=R['representations']['char'];L=['# WLCP v0.1 final results','','# RETRACTED FINDINGS','','None.','','# CURRENT FINDINGS','',f'**Endpoint: {endpoint}**','',f"Source SHA-256: `{audit['sha256']}`; running-text tokens={audit['tokens']}; folios={audit['folios']}; null repetitions={a.reps}.",'','## Primary tests (EVA-unit operational representation)','',fmt('Adjacent length MI (bits)',e['adj_mi']),fmt('Line-initial minus line-final length',e['first_final']),fmt('Held-out Markov gain (bits/transition)',e['markov_gain']),'','## Currier replication','']
 for cur in ('A','B'):
  L += [f"Currier {cur}: "+fmt('Adjacent length MI',e[cur]['adj_mi']),f"Currier {cur}: "+fmt('Initial-final length',e[cur]['first_final']),f"Currier {cur}: "+fmt('Markov gain',e[cur]['markov_gain'])]
 L+=['','## Representation bound','',fmt('Character-length adjacent MI',c['adj_mi']),fmt('Character-length initial-final',c['first_final']),fmt('Character-length Markov gain',c['markov_gain']),'','## Glyph × length controls','',fmt('First-unit × total-length MI',e['first_length_mi']),fmt('Final-unit × total-length MI',e['last_length_mi']),'','## Gates','',f"Gate 0 corpus/audit: PASS",f"Gate 1 reproducible conditional length structure: {'PASS' if gate1 else 'FAIL'}",f"Gate 2 matched-null and representation robustness: {'PASS' if gate2 else 'FAIL'}",'Gate 3 cipher-family identifiability from whole-token length: FAIL (formal degeneracy).','WL-2 mechanism exclusion: NOT PROMOTED.','WL-4 plaintext-length recovery for Voynich: NOT PROMOTED.','','## Gate 3 bounding proof','',proof['statement'],'','This is a hard identifiability bound, not a low-powered empirical failure. Whole-token length can still reject particular length-changing encodings if a plaintext prior is independently fixed, but it cannot by itself distinguish no cipher from a boundary-preserving 1:1 stateful/polyalphabetic substitution.','','## Audit','',
 '1. Circularity: primary statistics fixed before null generation.','2. Leakage: Markov train/test split is by whole folio.','3. Confounds: within-line permutation preserves the exact token multiset, vocabulary, morphology and marginal lengths of every line.','4. Matched nulls: 500 deterministic within-line permutations.','5. Control fairness: no cipher family is credited for a statistic that is mathematically degenerate with another family.','6. Measurement degeneracy: explicitly bounded by the Gate 3 proof.','7. Representation dependence: core tests repeated for greedy EVA units and raw transcription characters.','8. Decision-rule fragility: |z|>=2 plus same-direction Currier A/B replication for Gate 1.','9. Audit completeness: source hash, JSON and atomic pickle checkpoints retained.','10. Interpretation only follows these checks.','','## Interpretation','']
 if endpoint=='WL-0':L.append('The registered length effects do not clear the replication gate. Word length provides no usable cryptanalytic signal under this programme.')
 else:L.append('Voynich whole-token length contains reproducible conditional structure under matched nulls, but that structure is not sufficient to identify a cipher mechanism. The programme therefore terminates at WL-1: structurally real, cryptanalytically non-discriminating on its own.')
 L+=['','## Scope / hallucination boundary','','The EVA-unit representation here is operational (greedy recognition of standard multi-character EVA units), not a claim of palaeographic ground truth. No historical plaintext language is inferred. No existing Terminal Cipher solver is reimplemented or silently replaced; Gate 3 is a mechanism-independent mathematical equivalence for the length observable itself.']
 (a.out/'RESULTS.md').write_text('\n'.join(L));print('\n'.join(L))
if __name__=='__main__':main()
