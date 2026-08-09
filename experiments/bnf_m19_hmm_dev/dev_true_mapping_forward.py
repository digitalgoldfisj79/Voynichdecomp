#!/usr/bin/env python3
import urllib.request
import numpy as np
from collections import Counter
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0ccea68e5eef0b551cff7cb2703c20c9868e294c/experiments/bnf_free_switch_m19_v0_7/run_m19.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
ns={'__name__':'m19_base'}
exec(compile(src,'run_m19.py','exec'),ns)

# Amendment-001 generator, used only for control development.
def gen(plain,lang,rep):
 for attempt in range(1000):
  rng=np.random.default_rng(ns['seed']('values',lang,rep,attempt));vals=[]
  for c in plain:
   if c==' ':vals.append(None)
   else:vals.append(ns['V2I'][int(rng.choice(ns['LETTER_VALS'][ns['A2I'][c]]))])
  cnt=Counter();n=0
  for v in vals:
   if v is None:continue
   if n<ns['TRAIN']:cnt[v]+=1
   n+=1
  dup=[v for v,_ in sorted(cnt.items(),key=lambda kv:(-kv[1],kv[0]))[:6]]
  raw={v:[v] for v in range(ns['NV'])}
  for j,v in enumerate(dup):raw[v].append(ns['NV']+j)
  perm=np.arange(25);r2=np.random.default_rng(ns['seed']('opaque',lang,rep,attempt));r2.shuffle(perm);r2s={x:int(perm[x]) for x in range(25)};true=np.full(25,-1,np.int16)
  for v,forms in raw.items():
   for x in forms:true[r2s[x]]=v
  r3=np.random.default_rng(ns['seed']('surface',lang,rep,attempt));out=[];used=set();n=0
  for v in vals:
   if v is None:out.append(' ');continue
   sid=r2s[int(r3.choice(raw[v]))];out.append(chr(65+sid));
   if n<ns['TRAIN']:used.add(sid)
   n+=1
  if len(used)==25:return ''.join(out),true,attempt
 raise RuntimeError('generation')

def forward_word(obs,lm):
 T=lm['T'];emit=ns['EMIT'];a=lm['st']*emit[:,obs[0]];z=float(a.sum())
 if z<=0:return -1e100
 ll=np.log(z);a/=z
 for v in obs[1:]:
  a=(a@T)*emit[:,v];z=float(a.sum())
  if z<=0:return -1e100
  ll+=np.log(z);a/=z
 # use final-letter factor as a word-final pseudo-likelihood, normalized by current mass
 z=float(np.dot(a,lm['en']))
 if z>0:ll+=np.log(z)
 return ll

def forward_corpus(words,true,lm):
 ll=0.0;n=0;bad=0
 for w in words:
  obs=[int(true[ord(c)-65]) for c in w]
  x=forward_word(obs,lm)
  if x<-1e50:bad+=1;continue
  ll+=x;n+=len(obs)
 return ll/max(1,n),n,bad

def main():
 lms,holds,meta=ns['load_sources']()
 lang='latin';rep=0;span=ns['choose_span'](holds[lang],ns['TRAIN']+ns['HOLD'],(lang,rep));cipher,true,attempt=gen(span,lang,rep);_,cho=ns['split_letters'](cipher,ns['TRAIN']);words=cho.split()
 print('CONTROL',lang,rep,'attempt',attempt,'letters',sum(map(len,words)),flush=True)
 scores=[]
 for la in ns['LANGS']:
  sc,n,bad=forward_corpus(words,true,lms[la]);scores.append((la,sc));print('FORWARD',la,sc,'n',n,'bad',bad,flush=True)
 print('RANK',sorted(scores,key=lambda x:x[1],reverse=True),flush=True)
if __name__=='__main__':main()
