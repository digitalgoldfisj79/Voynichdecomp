#!/usr/bin/env python3
from __future__ import annotations
import pickle, json, math, random, statistics, csv, importlib.util
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

DATA=Path('/mnt/data/notation_voynich_inputs/enriched_records.pkl')
OUT=Path('/mnt/data/voynich_notation_v0_3');OUT.mkdir(exist_ok=True)
SEEDS=[101,202,303,404,505]
ALPHA=0.25
spec=importlib.util.spec_from_file_location('mdl','/mnt/data/run_notation_mdl_fast_v03.py');mdl=importlib.util.module_from_spec(spec);spec.loader.exec_module(mdl)

def pclass(r):return 'F' if int(r['pos'])==0 else ('L' if int(r['pos'])==int(r['line_len'])-1 else 'M')

def folio_splits(R,seed):
 fol=sorted(set(str(r['folio']) for r in R)); y=[]
 for f in fol:
  secs=Counter(str(r['section']) for r in R if str(r['folio'])==f);y.append(secs.most_common(1)[0][0])
 tr,te=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(fol,y));tf={fol[i] for i in tr};ef={fol[i] for i in te}
 return np.array([str(r['folio']) in tf for r in R]),np.array([str(r['folio']) in ef for r in R])

class CharModel:
 def __init__(self,alphabet):self.A=alphabet+['<EOS>'];self.V=len(self.A);self.c=defaultdict(Counter);self.n=Counter()
 def add(self,s,slot,sec,pos):
  prev='<BOS>'
  for ch in list(s)+['<EOS>']:
   ctx=(slot,sec,pos,prev);self.c[ctx][ch]+=1;self.n[ctx]+=1;prev=ch
 def logp(self,s,slot,sec,pos):
  prev='<BOS>';z=0.
  for ch in list(s)+['<EOS>']:
   ctx=(slot,sec,pos,prev);z+=math.log((self.c[ctx][ch]+.5)/(self.n[ctx]+.5*self.V));prev=ch
  return z

class Parser:
 def __init__(self,R,parts,train,alphabet):
  self.P={''};self.G={''};self.S={''};self.pc=defaultdict(Counter);self.gc=defaultdict(Counter);self.sc=defaultdict(Counter);self.cm=CharModel(alphabet)
  for i in np.where(train)[0]:
   r=R[i];p,g,c,s=parts[i];pos=pclass(r);sec=str(r['section']);self.P.add(p);self.G.add(g);self.S.add(s)
   self.pc[pos][p]+=1;self.gc[(sec,p)][g]+=1;self.sc[(pos,p,g)][s]+=1;self.cm.add(c,2,sec,pos)
  self.P=sorted(self.P,key=lambda x:(-len(x),x));self.G=sorted(self.G,key=lambda x:(-len(x),x));self.S=sorted(self.S,key=lambda x:(-len(x),x))
  self.vp=len(self.P);self.vg=len(self.G);self.vs=len(self.S);self.cache={};self.cand_cache={};self.core_cache={}
  self.lpP={(pos,p):math.log((self.pc[pos][p]+ALPHA)/(sum(self.pc[pos].values())+ALPHA*self.vp)) for pos in ('F','M','L') for p in self.P}
  secs=sorted(set(str(r['section']) for r in R))
  self.lpG={(sec,p,g):math.log((self.gc[(sec,p)][g]+ALPHA)/(sum(self.gc[(sec,p)].values())+ALPHA*self.vg)) for sec in secs for p in self.P for g in self.G}
  self.lpS={(pos,p,g,ss):math.log((self.sc[(pos,p,g)][ss]+ALPHA)/(sum(self.sc[(pos,p,g)].values())+ALPHA*self.vs)) for pos in ('F','M','L') for p in self.P for g in self.G for ss in self.S}
 def candidates(self,tok):
  if tok in self.cand_cache:return self.cand_cache[tok]
  out=[]
  for p in self.P:
   if p and not tok.startswith(p):continue
   rem=tok[len(p):]
   for g in self.G:
    if g and not rem.startswith(g):continue
    rest=rem[len(g):]
    for ss in self.S:
     if ss and (not rest.endswith(ss) or len(ss)>len(rest)):continue
     c=rest[:-len(ss)] if ss else rest
     out.append((p,g,c,ss))
  self.cand_cache[tok]=out;return out
 def parse(self,tok,sec,pos):
  key=(tok,sec,pos)
  if key in self.cache:return self.cache[key]
  best=None;bestz=-1e999
  for p,g,c,ss in self.candidates(tok):
   ck=(c,sec,pos)
   if ck not in self.core_cache:self.core_cache[ck]=self.cm.logp(c,2,sec,pos)
   z=self.lpP[(pos,p)]+self.lpG[(sec,p,g)]+self.lpS[(pos,p,g,ss)]+self.core_cache[ck]
   if z>bestz:bestz=z;best=(p,g,c,ss)
  self.cache[key]=best;return best

def slot_labels(parts):
 p,g,c,s=parts;return 'P'*len(p)+'G'*len(g)+'C'*len(c)+'S'*len(s)

def fit_char(R,parts,train,alphabet):
 m=CharModel(alphabet)
 for i in np.where(train)[0]:
  r=R[i];sec=str(r['section']);pos=pclass(r)
  for j,v in enumerate(parts[i]):m.add(v,j,sec,pos)
 return m

def score_char(R,parts,test,m):
 z=0;n=0
 for i in np.where(test)[0]:
  r=R[i];sec=str(r['section']);pos=pclass(r)
  for j,v in enumerate(parts[i]):z+=m.logp(v,j,sec,pos)
  n+=1
 return -z/math.log(2)/n

def main():
 import sys
 seeds=[int(sys.argv[1])] if len(sys.argv)>1 else SEEDS
 R=pickle.load(open(DATA,'rb'));S=mdl.make_segs(R);alphabet=sorted(set(''.join(r['token'] for r in R)));rows=[]
 for seed in seeds:
  tr,te=folio_splits(R,seed);train_types={str(R[i]['token']) for i in np.where(tr)[0]}
  for name,parts0 in S.items():
   if name=='Random_hash': continue
   if name=='UNSPLIT':
    m=fit_char(R,parts0,tr,alphabet);bpt=score_char(R,parts0,te,m)
    rows.append({'seed':seed,'segmentation':name,'all_exact':1.0,'unseen_exact':1.0,'char_label_acc':1.0,'predicted_bpt':bpt,'gold_bpt':bpt,'parse_penalty':0.0,'n_test':int(te.sum()),'n_unseen':sum(str(R[i]['token']) not in train_types for i in np.where(te)[0])});continue
   parts=[tuple(x) for x in parts0];parser=Parser(R,parts,tr,alphabet);pred=list(parts)
   exact=0;unexact=0;nu=0;charok=0;chartot=0
   for i in np.where(te)[0]:
    r=R[i];q=parser.parse(str(r['token']),str(r['section']),pclass(r));pred[i]=q
    exact+=q==parts[i]
    if str(r['token']) not in train_types:nu+=1;unexact+=q==parts[i]
    a=slot_labels(q);b=slot_labels(parts[i]);charok+=sum(x==y for x,y in zip(a,b));chartot+=len(a)
   m=fit_char(R,parts,tr,alphabet);pb=score_char(R,pred,te,m);gb=score_char(R,parts,te,m)
   rows.append({'seed':seed,'segmentation':name,'all_exact':exact/int(te.sum()),'unseen_exact':unexact/max(1,nu),'char_label_acc':charok/max(1,chartot),'predicted_bpt':pb,'gold_bpt':gb,'parse_penalty':pb-gb,'n_test':int(te.sum()),'n_unseen':nu})
   print(seed,name,round(exact/int(te.sum()),3),round(unexact/max(1,nu),3),round(pb,3),flush=True)
 summary=[]
 for name in S:
  if name=='Random_hash': continue
  rr=[x for x in rows if x['segmentation']==name]
  x={'segmentation':name,'n_splits':len(rr)}
  for k in ['all_exact','unseen_exact','char_label_acc','predicted_bpt','gold_bpt','parse_penalty']:x['mean_'+k]=statistics.mean(z[k] for z in rr);x['sd_'+k]=statistics.pstdev(z[k] for z in rr)
  summary.append(x)
 for metric,asc in [('mean_predicted_bpt',True),('mean_unseen_exact',False)]:
  for rank,x in enumerate(sorted(summary,key=lambda z:z[metric],reverse=not asc),1):x['rank_'+metric]=rank
 obj={'schema':'voynich-train-fold-only-segmentation-v0.3','rows':rows,'summary':summary}
 (OUT/(f'trainfold_segmentation_results_seed{seeds[0]}_v0_3.json' if len(seeds)==1 else 'trainfold_segmentation_results_v0_3.json')).write_text(json.dumps(obj,indent=2),encoding='utf-8')
 fields=list(summary[0])
 with open(OUT/(f'trainfold_segmentation_summary_seed{seeds[0]}_v0_3.csv' if len(seeds)==1 else 'trainfold_segmentation_summary_v0_3.csv'),'w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(summary,key=lambda z:z['mean_predicted_bpt']))
 print('\nSUMMARY')
 for x in sorted(summary,key=lambda z:z['mean_predicted_bpt']):print(x['segmentation'],round(x['mean_predicted_bpt'],3),round(x['mean_all_exact'],3),round(x['mean_unseen_exact'],3),round(x['mean_parse_penalty'],3))
if __name__=='__main__':main()
