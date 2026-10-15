#!/usr/bin/env python3
"""Global consistent swap hill climb of v145 assignments against development source words only.
Pinned Currier support, no updates to source segmentation, no Hartlieb fitting.
"""
import json,re,collections,random,hashlib,sys,time
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('vb',str(Path(__file__).resolve().parent/'build_v145.py'));vb=importlib.util.module_from_spec(spec);spec.loader.exec_module(vb)
P=Path(__file__).resolve().parent;B=(Path(__file__).resolve().parent/'inputs');U=vb.U;R=random.Random(20261014)
def make_graph(words):
 g=collections.Counter();node_freq=collections.Counter();
 for word in words:
  lead,r=vb.runs(word)
  seq=[('e',U.index(u),False) for u in vb.pieces(lead)]
  for v,run in r:
   parts=vb.pieces(run) if run else [''];seq.extend([(v,U.index(parts[0]),True)]+[(v,U.index(u),False) for u in parts[1:]])
  for n in seq:node_freq[n[:2]]+=1
  for a,b in zip(seq,seq[1:]):g[a,b]+=1
 keys=list(g);weights=list(g.values());idx=collections.defaultdict(set)
 for j,(a,b) in enumerate(keys):idx[a[:2]].add(j);idx[b[:2]].add(j)
 return keys,weights,idx,node_freq

words=(B/'train.reference.txt').read_text().splitlines();g,weights,incident,freq=make_graph(words);print('graph edges',len(g),'total',sum(weights),'nodes',len(freq),flush=True)

def calc_current(t,code,v,bridge):
 c=code[v][t[1]]
 return (bridge[v][c[0]] if t[2] else '')+c

def seam(a,b,tri):
 return int(a[-2:]+b[0] not in tri)+int(a[-1]+b[:2] not in tri)

def scoreedge(j,codes,tri,bridge):
 a,b=g[j];return seam(calc_current(a,codes,a[0],bridge),calc_current(b,codes,b[0],bridge),tri)*weights[j]

def opt(cls,iterations=30000):
 m=json.loads((P/f'v145_{cls}_manifest.json').read_text())
 tri,_=vb.support(vb.words_A() if cls=='A' else vb.words_B());bridge=m['bridge_for_vowel_and_next_key']
 codes={v:list(m['state'][v]['unitcode']) for v in 'aeiou'}
 pairScores=[scoreedge(j,codes,tri,bridge) for j in range(len(g))];orig=sum(pairScores);start=time.time();accepted=0;improved=0
 for it in range(iterations):
  v='aeiou'[it%5]
  a=R.randrange(222);b=R.randrange(222)
  if a==b:continue
  impacted=incident[(v,a)]|incident[(v,b)];prior=sum(pairScores[j] for j in impacted)
  codes[v][a],codes[v][b]=codes[v][b],codes[v][a]
  trial=[(j,scoreedge(j,codes,tri,bridge)) for j in impacted]
  delta=sum(x for j,x in trial)-prior
  # Allow rare uphill moves in first 40% to escape local optima with deterministic seed.
  temp= max(0.01,1.2*(1-it/iterations)**2) if it<iterations*.6 else 0
  accept=delta<=0 or (temp and R.random()<__import__('math').exp(-delta/temp))
  if accept:
   accepted+=1
   if delta<0:improved+=1
   for j,x in trial:pairScores[j]=x
  else:codes[v][a],codes[v][b]=codes[v][b],codes[v][a]
  if it%5000==4999:print(cls,'iter',it+1,'edge bad',sum(pairScores),'change',sum(pairScores)-orig,'elapsed',int(time.time()-start),flush=True)
 for v in 'aeiou':m['state'][v]['unitcode']=codes[v]
 look={v:dict(zip(U,codes[v])) for v in 'aeiou'};inv=vb.decoder(codes,m)
 output={'source_graph_edges':len(g),'source_graph_edge_events':sum(weights),'initial_bad_seam_trigrams':orig,'optimised_bad_seam_trigrams':sum(pairScores),'iterations':iterations,'accepted':accepted,'improved_swaps':improved,'source_spec':'development only, Hartlieb untouched until final validation','cases':{}}
 for source in ['train','hartlieb']:
  src=(B/f'{source}.reference.txt').read_text().splitlines();cs=[];bad=0,0;badwords=0;badtri=0
  for word in src:
   w=vb.encode(word,m,look)
   if inv(w)!=word:raise ValueError(('roundtrip failed',word,w))
   cs.append(w);k=sum(w[i:i+3] not in tri for i in range(len(w)-2));badtri+=k;badwords+=(k>0)
  (P/f'v145_{cls}_{source}_optimised.vbm').write_text(' '.join(cs)+'\n')
  output['cases'][source]={'n':len(src),'badwords':badwords,'badtrigrams':badtri,'mean_len':sum(map(len,cs))/len(cs)}
 (P/f'v145_{cls}_optimised_manifest.json').write_text(json.dumps(m,indent=2));return output

if __name__=='__main__':
 res={}
 for cls in ('A','B'):
  res[cls]=opt(cls,int(sys.argv[1]) if len(sys.argv)>1 else 25000)
  print('FINAL',cls,json.dumps(res[cls]),flush=True)
 (P/'v145_optimised_results.json').write_text(json.dumps(res,indent=2))
