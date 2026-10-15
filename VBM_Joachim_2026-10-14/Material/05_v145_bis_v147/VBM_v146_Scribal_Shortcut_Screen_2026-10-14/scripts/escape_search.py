#!/usr/bin/env python3
"""Exploratory escape-sign structural repair on fixed four digraph shortcuts.
All tuning on development only; historical mechanism not identified.
"""
import sys,json,collections,re,random,math,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_abbreviations as m
P=Path(__file__).resolve().parent.parent/'results'
family=sys.argv[1];prefix=sys.argv[2]
ref=m.target('A_REVA_reference.txt' if family=='A' else 'VMS_B_REVA_reference.txt')
sup=m.prep(ref)
train=m.tokenize(m.D,m.SUFFIX4);hold=m.tokenize(m.H,m.SUFFIX4)
freq=collections.Counter(x for row in train for x in row)
ordered=sorted(freq,key=lambda x:(-freq[x],x)); direct=ordered[:22];rare=ordered[22:]
atomcount=collections.Counter(a for w in ref for a in m.ATOMS.findall(w)); available=[a for a in m.ALPH if a!=prefix]
main=sorted(available,key=lambda a:(-atomcount[a],a))
gram=collections.Counter()
for w in ref:
 a=m.ATOMS.findall(w)
 for i in range(len(a)-1):gram[(a[i],a[i+1])]+=1
suffixes=sorted(available,key=lambda a:(-gram[(prefix,a)],a))[:len(rare)]
table={**dict(zip(direct,main)),**dict(zip(rare,[prefix+g for g in suffixes]))}
cur=dict(table);best=dict(cur)
cal=train[::40];val=train[5::37]
rr=random.Random(20261014+len(prefix)*19+(1 if family=='A' else 127)+ord(prefix[0])*41)
def score(t):return m.evaluate(cal,t,sup)['objective']
cur_s=score(cur);best_s=cur_s
for z in range(1300):
 if rr.random()<.70:
  a,b=rr.sample(direct,2)
 else:a,b=rr.sample(rare,2)
 cur[a],cur[b]=cur[b],cur[a]
 v=score(cur);temp=.047*(1-z/1300)**1.5+.001
 if v<cur_s or rr.random()<math.exp(min(0,(cur_s-v)/temp)):
  cur_s=v
  if v<best_s:best_s=v;best=dict(cur)
 else:cur[a],cur[b]=cur[b],cur[a]
result={'family':family,'escape_prefix':prefix,'source_digraphs':m.SUFFIX4,'training_objective':best_s,'validation':m.evaluate(val,best,sup),'holdout':m.evaluate(hold,best,sup),'codebook':best,'escape_tails':[best[u] for u in rare],'table_entries':len(best),'exact_decoding':False}
inv={v:k for k,v in best.items()}
def decode(w):
 atoms=m.ATOMS.findall(w);i=0;parts=[]
 while i<len(atoms):
  if atoms[i]==prefix:
   assert i+1<len(atoms)
   glyph=atoms[i]+atoms[i+1];i+=2
  else:glyph=atoms[i];i+=1
  parts.append(inv[glyph])
 return ''.join(parts)
assert all(decode(''.join(best[x] for x in seg))==w for seg,w in zip(hold,m.H))
result['exact_decoding']=True
(R:=P/f'ESC_{family}_{prefix}.json').write_text(json.dumps(result,indent=2)+'\n')
print(family,prefix, 'valcost',round(result['validation']['objective'],4), 'holdbad',round(result['holdout']['bad_trigram_rate'],4),'len',round(result['holdout']['mean_length'],3),'ge10',round(result['holdout']['share_len_ge_10'],3),'code',len(best),flush=True)
