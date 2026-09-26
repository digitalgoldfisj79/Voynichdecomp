#!/usr/bin/env python3
from __future__ import annotations
import pickle, random, math, json, hashlib, statistics, csv, re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.special import gammaln

DATA=Path('/mnt/data/notation_voynich_inputs/enriched_records.pkl')
OUT=Path('/mnt/data/voynich_notation_v0_3'); OUT.mkdir(exist_ok=True)
SEEDS=[101,202,303,404,505]
ALPHAS=[0.1,0.5,1.0]
EMPTY='∅'
GRAMMAR_P=['o','y','d','s','ch','sh','qo']; GRAMMAR_G=['k','t','p','f','ckh','cth','cph','cfh']
GRAMMAR_S=['aiin','edy','eey','ody','ain','iin','chy','shy','dy','ey','in','ol','or','ar','al','am','an','ir','ee','eedy','oiin','oiiin','y','n','l','r','m','g','iiin','s','a','e']
CRUDE_P=['o','qo','d','y','s','q']; CRUDE_G=['k','t','p','f','cth','ckh','cph','cfh','ch','sh']
ALT2_P=['o','qo','d','y','s']; ALT2_G=CRUDE_G
ALT4_P=['o','y','d','s','c','qo']; ALT4_G=['h','k','t','p','f','hk','ht','ckh','cth','cph','cfh']
ALT3_P=['o','y','d','s','ch','sh','qo','k','t','p','f','ok','ot','qok','qot','dk','dt','yk','sk','chk','cht','shk','sht','qop','qof','cth','ckh','cph','cfh','dcth','dckh']

def clean(x):
 s=str(x); return '' if s in ('∅','None','') else s

def decompose_generic(token,prefixes,gallows,suffixes):
 rem=token;p=''
 for x in sorted(prefixes,key=len,reverse=True):
  if x and rem.startswith(x):p=x;rem=rem[len(x):];break
 g=''
 for x in sorted(gallows,key=len,reverse=True):
  if x and rem.startswith(x):g=x;rem=rem[len(x):];break
 s=''
 for x in sorted(suffixes,key=len,reverse=True):
  if x and rem.endswith(x) and len(rem)>len(x):s=x;rem=rem[:-len(x)];break
 return p,g,rem,s

def fixed(token,npfx,ngal,nsfx):
 p=token[:npfx]; rem=token[len(p):]; g=rem[:ngal]; rem=rem[len(g):]
 if nsfx and len(rem)>nsfx:return p,g,rem[:-nsfx],rem[-nsfx:]
 return p,g,rem,''

def randseg(token):
 rng=random.Random(int(hashlib.sha256(('v03|'+token).encode()).hexdigest()[:16],16)); n=len(token)
 b=sorted(rng.randrange(n+1) for _ in range(3));return token[:b[0]],token[b[0]:b[1]],token[b[1]:b[2]],token[b[2]:]

def make_segs(R):
 O={'UNSPLIT':[(str(r['token']),) for r in R], 'P70_lossless':[tuple(clean(r[k]) for k in ('prefix','gallows','core','suffix')) for r in R]}
 F={
 'Crude_chsh_gallows':lambda t:decompose_generic(t,CRUDE_P,CRUDE_G,GRAMMAR_S),
 'No_chsh_prefix':lambda t:decompose_generic(t,ALT2_P,ALT2_G,GRAMMAR_S),
 'Shift_c_plus_h':lambda t:decompose_generic(t,ALT4_P,ALT4_G,GRAMMAR_S),
 'Flat_no_gallows':lambda t:decompose_generic(t,ALT3_P,[],GRAMMAR_S),
 'No_suffix':lambda t:decompose_generic(t,GRAMMAR_P,GRAMMAR_G,[]),
 'Fixed_1_1_1':lambda t:fixed(t,1,1,1),'Fixed_1_1_2':lambda t:fixed(t,1,1,2),'Fixed_1_1_3':lambda t:fixed(t,1,1,3),
 'Fixed_1_0_2':lambda t:fixed(t,1,0,2),'Fixed_2_1_1':lambda t:fixed(t,2,1,1),'Fixed_2_1_2':lambda t:fixed(t,2,1,2),'Fixed_2_0_2':lambda t:fixed(t,2,0,2),
 'Random_hash':randseg}
 for n,fn in F.items():O[n]=[fn(str(r['token'])) for r in R]
 for n,rows in O.items():
  bad=sum(''.join(p)!=str(r['token']) for p,r in zip(rows,R))
  if bad:raise ValueError(n,bad)
 return O

def posclass(r):
 return 'F' if int(r['pos'])==0 else ('L' if int(r['pos'])==int(r['line_len'])-1 else 'M')

def folkey(f):
 m=re.match(r'f(\d+)([rv])',f);return (int(m.group(1)) if m else 9999,0 if m and m.group(2)=='r' else 1,f)

def dict_code(vocabs,alphabet):
 A=tuple(alphabet)+('<EOS>',);V=len(A); counts=defaultdict(Counter); totals=Counter();bits=0.0
 for j,vocab in enumerate(vocabs):
  n=len(vocab); bits += math.log2(max(2,n))+2*math.log2(max(2,math.log2(max(4,n))))
  for s in sorted(vocab):
   prev='<BOS>'
   for ch in list(s)+['<EOS>']:
    ctx=(j,prev);p=(counts[ctx][ch]+0.5)/(totals[ctx]+0.5*V);bits-=math.log2(p);counts[ctx][ch]+=1;totals[ctx]+=1;prev=ch
 return bits

def context(j,parts,r,mode):
 sec=str(r['section']);pos=posclass(r)
 if len(parts)==1:return (0,sec,pos)
 if mode=='independent':
  return (j,pos) if j in (0,3) else (j,sec)
 if j==0:return (0,pos)
 if j==1:return (1,sec,parts[0])
 if j==2:return (2,sec,parts[0],parts[1])
 return (3,pos,parts[0],parts[1])

def build_block_counts(R,parts,mode):
 by=defaultdict(lambda:defaultdict(Counter))
 for r,p in zip(R,parts):
  f=str(r['folio'])
  for j,v in enumerate(p):by[f][context(j,p,r,mode)][v]+=1
 return by

def preq_code(by,vocab_by_slot,order,alpha):
 Vslot={j:len(v) for j,v in enumerate(vocab_by_slot)}
 maxn=40000
 lgval=gammaln(np.arange(maxn+1,dtype=float)+alpha)
 lgctx={j:gammaln(np.arange(maxn+1,dtype=float)+alpha*V) for j,V in Vslot.items()}
 C=defaultdict(Counter);N=Counter();bits=0.0; ilog2=1.0/math.log(2)
 for f in order:
  for ctx,bc in by[f].items():
   j=ctx[0];n=sum(bc.values());old=N[ctx]
   logp=lgctx[j][old]-lgctx[j][old+n]
   for v,b in bc.items():
    cv=C[ctx][v];logp += lgval[cv+b]-lgval[cv]
   bits -= logp*ilog2
  for ctx,bc in by[f].items():C[ctx].update(bc);N[ctx]+=sum(bc.values())
 return bits

def char_block_counts(R,parts,conditioned=True):
 by=defaultdict(lambda:defaultdict(Counter))
 for r,p in zip(R,parts):
  f=str(r['folio']); sec=str(r['section']);pos=posclass(r)
  for j,s in enumerate(p):
   prev='<BOS>'
   for ch in list(s)+['<EOS>']:
    ctx=(j,sec,pos,prev) if conditioned else (j,prev)
    by[f][ctx][ch]+=1;prev=ch
 return by

def preq_char(by,order,V,alpha=0.5):
 maxn=200000
 lgval=gammaln(np.arange(maxn+1,dtype=float)+alpha)
 lgctx=gammaln(np.arange(maxn+1,dtype=float)+alpha*V)
 C=defaultdict(Counter);N=Counter();bits=0.0;ilog2=1.0/math.log(2)
 for f in order:
  for ctx,bc in by[f].items():
   n=sum(bc.values());old=N[ctx]
   logp=lgctx[old]-lgctx[old+n]
   for v,b in bc.items():
    cv=C[ctx][v];logp += lgval[cv+b]-lgval[cv]
   bits-=logp*ilog2
  for ctx,bc in by[f].items():C[ctx].update(bc);N[ctx]+=sum(bc.values())
 return bits

def main():
 R=pickle.load(open(DATA,'rb'));S=make_segs(R);fol=sorted(set(str(r['folio']) for r in R),key=folkey);nt=len(R);nc=sum(len(r['token']) for r in R);alphabet=sorted(set(''.join(r['token'] for r in R)));Vchar=len(alphabet)+1
 audit={'lossless_core_errors':sum(''.join(clean(r[k]) for k in ('prefix','gallows','core','suffix'))!=r['token'] for r in R),
        'minimal_core_errors':sum(''.join(clean(r[k]) for k in ('prefix','gallows','m_core','suffix'))!=r['token'] for r in R)}
 results=[]
 for name,parts in S.items():
  voc=[set(x[j] for x in parts) for j in range(len(parts[0]))];dc=dict_code(voc,alphabet)
  blocks={m:build_block_counts(R,parts,m) for m in ('independent','packet')}
  cb={c:char_block_counts(R,parts,c) for c in (False,True)}
  for seed in SEEDS:
   order=fol[:];random.Random(seed).shuffle(order)
   for alpha in ALPHAS:
    for mode in ('independent','packet'):
     db=preq_code(blocks[mode],voc,order,alpha);total=db+dc
     results.append({'segmentation':name,'code':'exact_'+mode,'alpha':alpha,'seed':seed,'data_bits':db,'dictionary_bits':dc,'total_bits':total,'bpt':total/nt,'bpc':total/nc})
   for cond in (False,True):
    db=preq_char(cb[cond],order,Vchar,0.5)
    results.append({'segmentation':name,'code':'char_bigram_'+('secpos' if cond else 'global'),'alpha':0.5,'seed':seed,'data_bits':db,'dictionary_bits':0.0,'total_bits':db,'bpt':db/nt,'bpc':db/nc})
  print('done',name,flush=True)
 summary=[]
 keys=sorted(set((x['code'],x['alpha']) for x in results))
 for code,alpha in keys:
  for name in S:
   vals=[x['bpt'] for x in results if x['code']==code and x['alpha']==alpha and x['segmentation']==name]
   summary.append({'code':code,'alpha':alpha,'segmentation':name,'mean_bpt':statistics.mean(vals),'sd_bpt':statistics.pstdev(vals),'n':len(vals)})
 for code,alpha in keys:
  ss=sorted([x for x in summary if x['code']==code and x['alpha']==alpha],key=lambda z:z['mean_bpt'])
  for i,x in enumerate(ss,1):x['rank']=i
 obj={'schema':'voynich-notation-two-part-prequential-mdl-v0.3','audit':audit,'n_tokens':nt,'n_chars':nc,'n_folios':len(fol),'seeds':SEEDS,'alphas':ALPHAS,'results':results,'summary':summary}
 (OUT/'mdl_fast_results_v0_3.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
 with open(OUT/'mdl_fast_summary_v0_3.csv','w',newline='',encoding='utf-8') as f:
  fields=['code','alpha','rank','segmentation','mean_bpt','sd_bpt','n'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(summary,key=lambda x:(x['code'],x['alpha'],x['rank'])))
 for code,alpha in keys:
  ss=sorted([x for x in summary if x['code']==code and x['alpha']==alpha],key=lambda z:z['mean_bpt'])[:5]
  print(code,alpha,[(x['segmentation'],round(x['mean_bpt'],3),round(x['sd_bpt'],3)) for x in ss])
 print('audit',audit)
if __name__=='__main__':main()
