#!/usr/bin/env python3
"""Optimize component seam illegality with source-development glyph budget.
One selected model per Currier class; Hartlieb kept sealed during lambda selection.
"""
import json,collections,math,random,hashlib,importlib.util
from pathlib import Path
P=Path(__file__).resolve().parent;B=(Path(__file__).resolve().parent/'inputs')
spec=importlib.util.spec_from_file_location('vb',str(P/'build_v145.py'));vb=importlib.util.module_from_spec(spec);spec.loader.exec_module(vb)
spec2=importlib.util.spec_from_file_location('opt',str(P/'optimise_seams.py'));opt=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(opt)
G,WEIGHTS,INC,FREQ=opt.g,opt.weights,opt.incident,opt.freq
WEI=json.loads((P/'source_unit_weights.json').read_text())
ORIGINAL_MEAN=7.105766419088836;budget=8.25 # <16.1% inflation relative to frozen v142b, set before heldout scoring
N_WORDS=68396;LAMBDA_GRID=[0.4,0.8,1.6]

def totalchar_cost(v,unit,c,bridge):
 word=vb.U[unit];return (WEI['total'][v].get(word,0)*len(c)+WEI['first'][v].get(word,0)*len(bridge[v][c[0]]))

def trial(cls,lamb,steps=24000):
 manifest=json.loads((P/f'v145_{cls}_manifest.json').read_text());bridge=manifest['bridge_for_vowel_and_next_key']
 tri,_=vb.support(vb.words_A() if cls=='A' else vb.words_B())
 codes={v:list(c) for v,c in json.loads((P/f'v145_{cls}_minlength_mapping.json').read_text()).items()}
 r=random.Random(20261014+int(lamb*1000)+(cls=='B')*3)
 scored=[opt.scoreedge(j,codes,tri,bridge) for j in range(len(G))]
 chars=sum(totalchar_cost(v,i,c,bridge) for v,cs in codes.items() for i,c in enumerate(cs))
 initial=(sum(scored),chars);improved=0
 for z in range(steps):
  v='aeiou'[z%5];a=r.randrange(222);b=r.randrange(222)
  if a==b:continue
  edges=INC[(v,a)]|INC[(v,b)]
  prevE=sum(scored[j] for j in edges)
  pC=totalchar_cost(v,a,codes[v][a],bridge)+totalchar_cost(v,b,codes[v][b],bridge)
  codes[v][a],codes[v][b]=codes[v][b],codes[v][a]
  nC=totalchar_cost(v,a,codes[v][a],bridge)+totalchar_cost(v,b,codes[v][b],bridge)
  deltaC=nC-pC;update=[(j,opt.scoreedge(j,codes,tri,bridge)) for j in edges];deltaE=sum(x for _,x in update)-prevE
  temp=max(0,5*(1-z/steps)**2) if z<steps*.75 else 0
  delta=deltaE+lamb*deltaC
  if delta<=0 or (temp and r.random()<math.exp(-delta/temp)):
   chars+=deltaC
   if deltaE<0:improved+=1
   for j,x in update:scored[j]=x
  else:codes[v][a],codes[v][b]=codes[v][b],codes[v][a]
 outcome={'lambda':lamb,'dev_bad_seams':sum(scored),'dev_chars':chars,'dev_mean_len':chars/N_WORDS,'initial_dev_bad_seams':initial[0],'initial_dev_mean_len':initial[1]/N_WORDS,'improved_swaps':improved}
 return outcome,codes,manifest,tri

if __name__=='__main__':
 table={}
 for cls in ('A','B'):
  tried=[];models=[]
  for lam in LAMBDA_GRID:
   out,codes,manifest,tri=trial(cls,lam)
   tried.append(out);models.append((codes,manifest,tri));print(cls,out,flush=True)
  admissible=[i for i,x in enumerate(tried) if x['dev_mean_len']<=budget]
  chosen=min(admissible,key=lambda i:tried[i]['dev_bad_seams']) if admissible else min(range(len(tried)),key=lambda i:tried[i]['dev_mean_len'])
  codes,manifest,tri=models[chosen];look={v:dict(zip(vb.U,codes[v])) for v in 'aeiou'}
  for v in 'aeiou':manifest['state'][v]['unitcode']=codes[v]
  dec=vb.decoder(codes,manifest)
  out={'selection_rule':'minimize development seam illegality subject to mean ciphertext len <=8.25 across frozen 0.4/0.8/1.6 lambda grid; never inspect Hartlieb before selection', 'candidate_grid':tried, 'selected':tried[chosen], 'validation':{}}
  for source in ('train','hartlieb'):
   sourcewords=(B/f'{source}.reference.txt').read_text().splitlines();cipher=[];badwords=0;badtri=0;badbi=0;big=vb.support(vb.words_A() if cls=='A' else vb.words_B())[1]
   for w in sourcewords:
    c=vb.encode(w,manifest,look)
    if dec(c)!=w:raise ValueError(('roundtrip',cls,source,w,c))
    cipher.append(c)
    k=sum(c[i:i+3] not in tri for i in range(len(c)-2));badwords+=int(k>0);badtri+=k;badbi+=sum(c[i:i+2] not in big for i in range(len(c)-1))
   out['validation'][source]={'n':len(sourcewords),'decode_exact':True,'badwords':badwords,'badtri':badtri,'badbi':badbi,'length':sum(map(len,cipher))/len(cipher)}
   (P/f'v145_{cls}_{source}_length_constrained.vbm').write_text(' '.join(cipher)+'\n')
  (P/f'v145_{cls}_length_constrained_manifest.json').write_text(json.dumps(manifest,indent=2))
  table[cls]=out;print('SELECTED',cls,out['selected'],out['validation'],flush=True)
 (P/'v145_length_constrained_results.json').write_text(json.dumps(table,indent=2))
