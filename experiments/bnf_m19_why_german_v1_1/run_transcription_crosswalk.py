#!/usr/bin/env python3
import urllib.request,json,re
from collections import Counter,defaultdict
import numpy as np
import edlib
PARENT='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c7c50f74e1f1f88004a0f08ea379324a3d42c16d/experiments/bnf_m19_german_confirm_v1_0/run_confirm.py'
src=urllib.request.urlopen(PARENT,timeout=90).read().decode();src=src.rsplit("if __name__=='__main__':main()",1)[0]
lib={'__name__':'parent'};exec(compile(src,'run_confirm.py','exec'),lib)
b=lib['b'];inner=lib['inner'];M=lib['M'];SYMS=lib['SYMS']
TIDS=['GCGA','GCGI','FFSG','FFSG-1','FFSG-2','RGVN','PCCA','PCCI','JSLI','JGLI','ZLZB','VDRB-1','TTVE','TTIA','TTII']

def normline(s):return ''.join(c.lower() for c in s if c.isalpha())

def pairs_from_cigar(query,target,cigar):
 qi=ti=0;out=[]
 for n,op in re.findall(r'(\d+)([=XID])',cigar or ''):
  n=int(n)
  if op in '=X':
   for k in range(n):out.append((query[qi+k],target[ti+k]))
   qi+=n;ti+=n
  elif op=='I': # insertion to target relative to query
   ti+=n
  elif op=='D':
   qi+=n
 return out

def line_pairs(data,folios,tid):
 out=[]
 for f in folios:
  if f not in data['pages']:continue
  for lk,line in data['pages'][f].items():
   t=line.get('t',{});q=normline(t.get(tid,''));z=normline(t.get('ZLZI',''))
   if not q or not z:continue
   al=edlib.align(q,z,task='path',mode='NW');out.extend(pairs_from_cigar(q,z,al.get('cigar')))
 return out

def learn(data,folios,tid):
 cc=defaultdict(Counter)
 for a,z in line_pairs(data,folios,tid):cc[a][z]+=1
 mp={a:c.most_common(1)[0][0] for a,c in cc.items()};tot=sum(sum(c.values()) for c in cc.values());correct=sum(c[mp[a]] for a,c in cc.items());return mp,correct/max(1,tot),{a:dict(c) for a,c in cc.items()}

def eval_crosswalk(data,folios,tid,mp):
 ps=line_pairs(data,folios,tid);tot=known=correct=0
 for a,z in ps:
  tot+=1
  if a in mp:
   known+=1;correct+=int(mp[a]==z)
 return {'aligned_pairs':tot,'mapped_pairs':known,'coverage':known/max(1,tot),'agreement':correct/max(1,known)}

def mapped_words(data,folios,tid,mp):
 out=[];total=known=0
 for f in folios:
  if f not in data['pages']:continue
  for _,line in sorted(data['pages'][f].items(),key=lambda kv:int(kv[0]) if str(kv[0]).isdigit() else 99999):
   for tok in line.get('t',{}).get(tid,'').split():
    q=normline(tok)
    if not q:continue
    z=[];ok=True
    for c in q:
     total+=1
     if c not in mp:ok=False;break
     z.append(mp[c]);known+=1
    if ok and z:out.append(''.join(z))
 return out,known/max(1,total)

def rank(words,lms):
 rr=[]
 for la in b['LANGS']:
  sc,n,sk,cov=inner['forward_words'](words,M,SYMS,lms[la]);rr.append((la,float(sc),int(n),float(cov)))
 rr.sort(key=lambda x:x[1],reverse=True);return rr

def main():
 lms,_,_=inner['load_fresh']();data=json.loads(b['fetch'](b['SLIM']));sample,hold,pages,_=inner['split_vms'](data);T=sorted(f for f,_,_ in sample);H=sorted(f for f,_,_ in hold);A={f for f,_,_ in pages};C=sorted(A-set(T)-set(H));out={}
 for tid in TIDS:
  mp,tragr,counts=learn(data,T,tid);h=eval_crosswalk(data,H,tid,mp);ww,cov=mapped_words(data,C,tid,mp);row={'map':mp,'train_pair_agreement':tragr,'hold_crosswalk':h,'c10_word_coverage':cov,'c10_letters':sum(map(len,ww)),'c10_tokens':len(ww)}
  if h['coverage']>=.95 and h['agreement']>=.80 and cov>=.90 and sum(map(len,ww))>=10000:
   r=rank(ww,lms);row['ranking']=[(x[0],x[1]) for x in r];row['german_rank']=1+next(i for i,x in enumerate(r) if x[0]=='german');g=next(x[1] for x in r if x[0]=='german');bo=max(x[1] for x in r if x[0]!='german');row['german_margin']=g-bo
  else:row['verdict']='CROSSWALK NOT QUALIFIED'
  out[tid]=row;print('CROSSWALK',tid,json.dumps(row,separators=(',',':')),flush=True)
 print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':main()
