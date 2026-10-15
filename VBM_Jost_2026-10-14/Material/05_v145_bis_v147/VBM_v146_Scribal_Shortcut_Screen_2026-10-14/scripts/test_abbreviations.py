#!/usr/bin/env python3
"""VBM compact manual encoding preflight: frozen token alphabets, 4 ordinary digraph shortcuts,
unique escape prefix, optimal sign permutations by development diagnostic only.
No attempt to infer Voynich semantic content. Script version 1, no independent VMS holdout.
"""
import re, json, random, math, time, hashlib, collections, statistics
from pathlib import Path
BASE=Path('/mnt/data/VBM_Scribal_Feasibility_Bound_2026-10-14/inputs')
OUT=Path(__file__).resolve().parent
D=list((BASE/'train.reference.txt').read_text().split())
H=list((BASE/'hartlieb.reference.txt').read_text().split())
ALPH=['a','d','e','f','g','i','k','l','m','n','o','p','q','r','s','t','y','ch','sh','ckh','cth','cph','cfh']
ATOMS=re.compile(r'ckh|cth|cph|cfh|ch|sh|.')
SUFFIX4=['en','er','ch','un']
def target(f):
 lines=(BASE/f).read_text().splitlines()
 if f.startswith('A_'):
  words=[w for ln in lines for w in ln.split() if re.fullmatch('[a-z]+',w)]
 else:
  words=[w for ln in lines if '\t' in ln for w in ln.split('\t',1)[1].split() if re.fullmatch('[a-z]+',w)]
 return words

def tokenize(words,abbr):
 out=[]
 for w in words:
  seg=[];i=0
  while i<len(w):
   pair=w[i:i+2];
   if pair in abbr:seg.append(pair);i+=2
   else:seg.append(w[i]);i+=1
  out.append(seg)
 return out

def construct_model(words, abbr, ref):
 seq=tokenize(words,abbr)
 freq=collections.Counter(k for ts in seq for k in ts)
 rf=collections.Counter(g for w in ref for g in ATOMS.findall(w))
 if not abbr:
  used=sorted(freq,key=lambda a:(-freq[a],a))
  glyph=sorted(ALPH,key=lambda a:(-rf[a],a))
  direct=dict(zip(used,glyph));escape={}
 else:
  ESC='g'; glyphs=sorted([g for g in ALPH if g!=ESC], key=lambda a:(-rf[a],a))
  ordered=sorted(freq,key=lambda a:(-freq[a],a))
  assert len(ordered)==23+len(abbr)
  direct=dict(zip(ordered[:22],glyphs));
  escape=dict(zip(ordered[22:], [ESC+g for g in glyphs[::-1][:len(ordered)-22]]))
 return seq,direct,escape

def prep(ref):
 trig={w[i:i+3] for w in ref for i in range(len(w)-2)}
 bi={w[i:i+2] for w in ref for i in range(len(w)-1)}
 beg={w[:2] for w in ref if len(w)>1};end={w[-2:] for w in ref if len(w)>1}
 return trig,bi,beg,end,set(ref),sum(map(len,ref))/len(ref),sum(len(w)>=10 for w in ref)/len(ref)

def evaluate(seqs,table,support,details=False):
 trig,bi,beg,end,vocab,natlen,tail=support
 n=len(seqs);bad3=bad2=boundary=exact=long=0;total=totalbadtr=totalbadbi=0
 generated=[]
 for seg in seqs:
  s=''.join(table[x] for x in seg);generated.append(s)
  m=len(s);total+=m
  invalid3=sum(s[i:i+3] not in trig for i in range(m-2));totalbadtr+=invalid3
  invalid2=sum(s[i:i+2] not in bi for i in range(m-1));totalbadbi+=invalid2
  bad3+=invalid3>0;bad2+=invalid2>0
  boundary+=int(m>1 and (s[:2] not in beg or s[-2:] not in end))
  long+=m>=10;exact+=s in vocab
 L=total/n;R3=bad3/n;R2=bad2/n;bound=boundary/n;long=long/n
 # registered optimization objective, no Voynich semantics
 cost=R3+.30*R2+.20*bound+.15*totalbadtr/n+.12*(L-natlen)**2+.08*max(0,long-tail)
 result={'n':n,'mean_length':L,'bad_trigram_rate':R3,'bad_bigram_rate':R2,'invalid_trigrams_per_word':totalbadtr/n,'invalid_bigrams_per_word':totalbadbi/n,'invalid_boundary_rate':bound,'share_len_ge_10':long,'exact_native_word_rate':exact/n,'objective':cost}
 if details:result['cipher_sha256']=hashlib.sha256('\n'.join(generated).encode()).hexdigest();result['cipher_words']=generated
 return result

def optimise(table,seqs,support,seed,iterations=950):
 rr=random.Random(seed);cur=dict(table);direct=[k for k,v in cur.items() if not v.startswith('g')];best=dict(cur)
 curS=evaluate(seqs,cur,support)['objective'];bestS=curS
 for t in range(iterations):
  i,j=rr.sample(direct,2)
  cur[i],cur[j]=cur[j],cur[i]
  c=evaluate(seqs,cur,support)['objective']
  temp=.06*(1-t/iterations)**1.5+.001
  if c<curS or rr.random()<math.exp(min(0,(curS-c)/temp)):
   curS=c
   if c<bestS:bestS=c;best=dict(cur)
  else:cur[i],cur[j]=cur[j],cur[i]
 return best,bestS

def main():
 print('length train',len(D),'hold',len(H),flush=True)
 results={'source':{'development_words':len(D),'diagnostic_reused_hartlieb_words':len(H),'notation':'reverse-atom EVA, A/B reference used to learn constraints; no historical inference'},'models':{}}
 for fam,f in [('A','A_REVA_reference.txt'),('B','VMS_B_REVA_reference.txt')]:
  print('family',fam,flush=True)
  ref=target(f);su=prep(ref)
  for name, abbr in [('simple23',[]),('four_shortcuts',SUFFIX4)]:
   tr,base,esc=construct_model(D,abbr,ref)
   hold=tokenize(H,abbr)
   train_sample=tr[::40] # deterministic 1710 development words
   table={**base,**esc};before=evaluate(hold,table,su)
   tic=time.time();opt,score=optimise(table,train_sample,su,seed=20261014+(0 if fam=='A' else 97)+(0 if not abbr else 101),iterations=950)
   met=evaluate(hold,opt,su,details=True);outputs=met.pop('cipher_words')
   # independent token-by-token decoder, no source words accessed in its logic
   decoder={v:k for k,v in opt.items()};atom_inv=ATOMS
   def decode(w):
    ats=atom_inv.findall(w);i=0;decoded=[]
    while i<len(ats):
     a=ats[i];
     if esc and a=='g':
      assert i+1<len(ats)
      code='g'+ats[i+1];i+=2
     else:code=a;i+=1
     decoded.append(decoder[code])
    return ''.join(decoded)
   decoded=[decode(w) for w in outputs]
   assert decoded==H,(fam,name)
   total_ops=sum(map(len,hold));esc_use=sum(x in esc for seq in hold for x in seq)
   data={'abbreviations':abbr,'direct_signs':len(base),'escape_signs':len(esc),'table_entries':len(table),'symbol_operations_per_word':total_ops/len(H),'escape_calls_per_word':esc_use/len(H),'baseline_no_optimisation':before,'diagnostic_hartlieb':met,'dev_sample_objective':score,'optimisation_seconds':round(time.time()-tic,2),'mapping':opt,'exact_decoder':True}
   results['models'][fam+'_'+name]=data
   (OUT/(fam+'_'+name+'_mapping.json')).write_text(json.dumps(opt,indent=2)+'\n')
   print(fam,name,{k:round(met[k],4) for k in ['bad_trigram_rate','bad_bigram_rate','mean_length','share_len_ge_10','exact_native_word_rate']},'time',round(time.time()-tic,2),flush=True)
 (OUT/'EXPERIMENT.json').write_text(json.dumps(results,indent=2)+'\n');print('saved',OUT/'EXPERIMENT.json')
if __name__=='__main__':main()
