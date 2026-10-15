#!/usr/bin/env python3
"""Compact scribal 3-column alphabet (initial, medial, terminal). No shortcuts.
Exact independent positional decoding; development-only table optimisation;
not historical mechanism attribution and NOT unseen VMS validation.
"""
import sys,re,json,math,random,time,collections,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_abbreviations as m
P=Path(__file__).resolve().parent.parent/'results'
fam=sys.argv[1]
ref=m.target('A_REVA_reference.txt' if fam=='A' else 'VMS_B_REVA_reference.txt');sup=m.prep(ref)
# Explicit atom-code position, all source letters permitted in each table.
def parsewords(ws):
 out=[]
 for w in ws:
  if len(w)==1:out.append(['I:'+w]);continue
  out.append(['I:'+w[0]]+['M:'+c for c in w[1:-1]]+['F:'+w[-1]])
 return out
train=parsewords(m.D);hold=parsewords(m.H)
source_counts={p:collections.Counter(c for row in train for x in row if x.startswith(p+':') for c in x[2:]) for p in ['I','M','F']}
refcounts={p:collections.Counter() for p in ['I','M','F']}
for w in ref:
 tok=m.ATOMS.findall(w)
 if not tok:continue
 refcounts['I'][tok[0]]+=1
 refcounts['F'][tok[-1]]+=1
 for x in tok[1:-1]:refcounts['M'][x]+=1
initial={};pool=sorted(set(''.join(m.D)));assert len(pool)==23
for pos in ['I','M','F']:
 ordered=sorted(pool,key=lambda x:(-source_counts[pos][x],x))
 signs=sorted(m.ALPH,key=lambda x:(-refcounts[pos][x],x));assert len(signs)==23
 initial.update({pos+':'+ch:sign for ch,sign in zip(ordered,signs)})
def score(rows,tbl):return m.evaluate(rows,tbl,sup)['objective']
results=[]
for seed_i in [0,1]:
 rr=random.Random(20261014+(0 if fam=='A' else 1077)+seed_i*73)
 table=dict(initial);bs=score(train[::40],table);best=dict(table);cur=bs;bests=bs
 start=time.time()
 for k in range(1800):
  pos=rr.choice(['I','M','F']);a,b=rr.sample(pool,2);i,j=pos+':'+a,pos+':'+b
  table[i],table[j]=table[j],table[i]
  v=score(train[::40],table)
  temp=.067*(1-k/1800)**1.7+.001
  if v<cur or rr.random()<math.exp(min(0,(cur-v)/temp)):
   cur=v
   if v<bests:bests=v;best=dict(table)
  else:table[i],table[j]=table[j],table[i]
 v=score(train[5::37],best)
 results.append((v,seed_i,best,bests,time.time()-start))
val,seed,table,trcost,seconds=min(results,key=lambda z:z[0])
met=m.evaluate(hold,table,sup,details=True);out=met.pop('cipher_words')
inv={pos:{v:k[2:] for k,v in table.items() if k.startswith(pos+':')} for pos in 'IMF'}
assert all(len(x)==23 for x in inv.values())
def dec(w):
 a=m.ATOMS.findall(w);n=len(a)
 decoded=''.join(inv['I' if i==0 else 'F' if i==n-1 else 'M'][glyph] for i,glyph in enumerate(a))
 return decoded
assert all(dec(cipher)==plain for cipher,plain in zip(out,m.H))
value={'family':fam,'mapping':table,'table_entries':69,'words_tested':len(hold),'development_val_objective':val,'development_fit_objective':trcost,'selected_seed':seed,'hartlieb_diagnostic':met,'exact_decode':True,'mean_EVA_atoms_per_word':sum(len(m.ATOMS.findall(w)) for w in out)/len(out),'caution':'3 independent bijections, same source full reversibility; target ref used in optimisation; reused Hartlieb is exploratory.'}
(P/f'POSITION_{fam}.json').write_text(json.dumps(value,indent=2)+'\n')
print(fam,'bad3',round(met['bad_trigram_rate'],4),'bigrams',round(met['bad_bigram_rate'],4),'len',round(met['mean_length'],3),'ge10',round(met['share_len_ge_10'],3),'exact word',round(met['exact_native_word_rate'],3),'val',round(val,3),'decode',True,flush=True)
