#!/usr/bin/env python3
import urllib.request,json
import numpy as np
from collections import Counter
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0ccea68e5eef0b551cff7cb2703c20c9868e294c/experiments/bnf_free_switch_m19_v0_7/run_m19.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
b={'__name__':'m19_base'}
exec(compile(src,'run_m19.py','exec'),b)
QUAL=['latin','italian','german','french','arabic','spanish']
TRAIN_RES={2,3,4,7,8,9};QUAL_RES={1,6}

def load_fresh():
 lms={};pools={};meta={}
 for lang,u in b['LM_URLS'].items():
  ss=b['conllu'](b['fetch'](u));tr=[s for i,s in enumerate(ss) if i%10 in TRAIN_RES];qo=[s for i,s in enumerate(ss) if i%10 in QUAL_RES];lm=b['build_lm'](tr);lms[lang]=lm;pools[lang]=b['pool_text'](qo);meta[lang]={'sentences_total':len(ss),'train_sentences':len(tr),'qual_sentences':len(qo),'lm_letters':lm['letters'],'qual_letters':sum(c!=' ' for c in pools[lang])};print('FRESH',lang,meta[lang],flush=True)
 return lms,pools,meta

def gen_control(plain,lang):
 for attempt in range(1000):
  rng=np.random.default_rng(b['seed']('v08values',lang,attempt));vals=[]
  for c in plain:
   if c==' ':vals.append(None)
   else:vals.append(b['V2I'][int(rng.choice(b['LETTER_VALS'][b['A2I'][c]]))])
  cnt=Counter();n=0
  for v in vals:
   if v is None:continue
   if n<b['TRAIN']:cnt[v]+=1
   n+=1
  if len(cnt)<b['NV']:continue
  dup=[v for v,_ in sorted(cnt.items(),key=lambda kv:(-kv[1],kv[0]))[:6]];raw={v:[v] for v in range(b['NV'])}
  for j,v in enumerate(dup):raw[v].append(b['NV']+j)
  perm=np.arange(25);r2=np.random.default_rng(b['seed']('v08opaque',lang,attempt));r2.shuffle(perm);r2s={x:int(perm[x]) for x in range(25)};true=np.full(25,-1,np.int16)
  for v,forms in raw.items():
   for x in forms:true[r2s[x]]=v
  r3=np.random.default_rng(b['seed']('v08surface',lang,attempt));out=[];used=set();n=0
  for v in vals:
   if v is None:out.append(' ');continue
   sid=r2s[int(r3.choice(raw[v]))];out.append(chr(65+sid));
   if n<b['TRAIN']:used.add(sid)
   n+=1
  if len(used)==25:
   print('QUAL_GENERATION',lang,'attempt',attempt,flush=True);assert b['valid_map'](true);return ''.join(out),true,attempt
 raise RuntimeError(('generation exhausted',lang))

def forward_word_values(obs,lm):
 if not obs:return (0.0,0)
 T=lm['T'];E=b['EMIT'];a=lm['st']*E[:,obs[0]];z=float(a.sum())
 if z<=0:return (-1e100,0)
 ll=float(np.log(z));a/=z
 for v in obs[1:]:
  a=(a@T)*E[:,v];z=float(a.sum())
  if z<=0:return (-1e100,0)
  ll+=float(np.log(z));a/=z
 z=float(np.dot(a,lm['en']))
 if z>0:ll+=float(np.log(z))
 return ll,len(obs)

def forward_words(words,m,symbols,lm):
 s2i={s:i for i,s in enumerate(symbols)};ll=0.0;n=0;skipped_words=0;total_letters=0;mapped_letters=0
 for w in words:
  total_letters+=len(w);obs=[];ok=True
  for c in w:
   i=s2i.get(c)
   if i is None:ok=False;break
   obs.append(int(m[i]));mapped_letters+=1
  if not ok:
   skipped_words+=1;continue
  x,k=forward_word_values(obs,lm)
  if k:ll+=x;n+=k
 return ll/max(1,n),n,skipped_words,mapped_letters/max(1,total_letters)

def mapping_agreement(freq,m1,m2):return float(np.dot(freq,m1==m2)/max(1,freq.sum()))
def weighted_acc(S,m,true):return float(np.dot(S['freq'],m==true)/max(1,S['freq'].sum()))

def split_vms(data):
 pages=[]
 for f in data['pages']:
  w=b['extract_page'](data,f,'ZLZI')
  if w:pages.append((f,w,sum(map(len,w))))
 pages=sorted(pages,key=lambda p:b['seed']('M19HMMsplit',p[0]));nh=max(1,int(round(.2*len(pages))));hold=pages[:nh];train=pages[nh:];required=set(c for _,ws,_ in train for w in ws for c in w);cand=sorted(train,key=lambda p:b['seed']('M19HMMtrain',p[0]));sample=[];n=0;seen=set()
 for p in cand:
  sample.append(p);n+=p[2]
  for w in p[1]:seen.update(w)
  if n>=b['TRAIN'] and required.issubset(seen):break
 return sample,hold,pages,required

def combine(pp):return [w for _,ws,_ in pp for w in ws]

def lexical_z(words,m,symbols,lm,tag):
 obs,_,_=b['lexical'](words,m,symbols,lm);rng=np.random.default_rng(b['seed']('v08lex',tag));vals=[]
 for _ in range(b['LEX_NULLS']):
  x=m.copy();rng.shuffle(x);vals.append(b['lexical'](words,x,symbols,lm)[0])
 mu=float(np.mean(vals));sd=float(np.std(vals,ddof=1));return {'fraction':obs,'null_mean':mu,'null_sd':sd,'z':(obs-mu)/sd if sd>1e-15 else 0.0}
def transfer(data,hold,tid,m,symbols,lms,cand):
 words=[]
 for f,_,_ in hold:
  if f in data['pages']:words.extend(b['extract_page'](data,f,tid))
 rows=[];coverage=None
 for la in b['LANGS']:
  sc,n,sk,cov=forward_words(words,m,symbols,lms[la]);rows.append((la,sc));coverage=cov
 rows.sort(key=lambda x:x[1],reverse=True);lex=lexical_z(words,m,symbols,lms[cand],('transfer',tid,cand));return {'ranking':rows,'candidate_rank':1+next(i for i,x in enumerate(rows) if x[0]==cand),'candidate_score':next(x[1] for x in rows if x[0]==cand),'margin':rows[0][1]-rows[1][1] if rows and rows[0][0]==cand else None,'coverage':coverage,'lexical':lex}

def main():
 lms,pools,meta=load_fresh();comps={la:b['induced'](lms[la]) for la in b['LANGS']};controls=[]
 for lang in QUAL:
  if sum(c!=' ' for c in pools[lang])<b['TRAIN']+b['HOLD']:raise RuntimeError(('qual pool short',lang))
  span=b['choose_span'](pools[lang],b['TRAIN']+b['HOLD'],('v08qual',lang));cipher,true,attempt=gen_control(span,lang);ctra,cho=b['split_letters'](cipher,b['TRAIN']);symbols=[chr(65+i) for i in range(25)];trw=ctra.split();how=cho.split();Str=b['stats'](trw,symbols);Sho=b['stats'](how,symbols);rows=[];fits={}
  for cand in b['LANGS']:
   sc,m=b['optimize'](Str,comps[cand],('v08qual',lang,cand,'fit1'));fw,n,sk,cov=forward_words(how,m,symbols,lms[cand]);rows.append((cand,fw));fits[cand]=m
  rows.sort(key=lambda x:x[1],reverse=True);target=fits[lang];acc=weighted_acc(Sho,target,true);sc2,m2=b['optimize'](Str,comps[lang],('v08qual',lang,'fit2'));agr=mapping_agreement(Str['freq'],target,m2);margin=rows[0][1]-rows[1][1];row={'lang':lang,'top':rows[0][0],'margin':margin,'target_rank':1+next(i for i,x in enumerate(rows) if x[0]==lang),'target_score':next(x[1] for x in rows if x[0]==lang),'mapping_acc':acc,'fit_agreement':agr,'attempt':attempt,'ranking':rows};controls.append(row);print('QUAL',json.dumps(row,separators=(',',':')),flush=True)
 gate={'correct':sum(r['top']==r['lang'] for r in controls),'min_margin':float(min(r['margin'] for r in controls)),'median_acc':float(np.median([r['mapping_acc'] for r in controls])),'min_acc':float(min(r['mapping_acc'] for r in controls)),'min_agreement':float(min(r['fit_agreement'] for r in controls))};gate.update({'Q1':gate['correct']==6,'Q2':gate['min_margin']>=.05,'Q3':gate['median_acc']>=.95,'Q4':gate['min_acc']>=.85,'Q5':gate['min_agreement']>=.90});gate['pass']=all(gate[k] for k in ['Q1','Q2','Q3','Q4','Q5']);print('QUAL_GATE',json.dumps(gate,separators=(',',':')),flush=True);out={'protocol':'v0.8','fresh_meta':meta,'controls':controls,'gate':gate}
 if not gate['pass']:
  out['verdict']='INSTRUMENT NOT QUALIFIED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
 data=json.loads(b['fetch'](b['SLIM']));sample,hold,pages,required=split_vms(data);trw=combine(sample);how=combine(hold);symbols=sorted(set(c for w in trw for c in w));Str=b['stats'](trw,symbols);Sho=b['stats'](how,symbols);census={'all_pages':len(pages),'train_sample_pages':len(sample),'hold_pages':len(hold),'train_letters':sum(p[2] for p in sample),'hold_letters':sum(p[2] for p in hold),'required_train_symbols':sorted(required),'sample_symbols':symbols,'nsym':len(symbols),'hold_mapping_coverage':Sho['coverage']};out['vms_census']=census;print('VMS_CENSUS',json.dumps(census,separators=(',',':')),flush=True)
 if len(symbols)!=25 or Sho['coverage']<.99:
  out['verdict']='UNDERPOWERED: SURFACE ALPHABET/COVERAGE';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
 vres=[];fits={}
 for la in b['LANGS']:
  sc1,m1=b['optimize'](Str,comps[la],('v08VMS',la,'fit1'));sc2,m2=b['optimize'](Str,comps[la],('v08VMS',la,'fit2'));m=m1 if sc1>=sc2 else m2;agr=mapping_agreement(Str['freq'],m1,m2);fw,n,sk,cov=forward_words(how,m,symbols,lms[la]);row={'lang':la,'forward_score':fw,'fit_agreement':agr,'mapping':{symbols[i]:b['VALUES'][int(m[i])] for i in range(25)},'train_pair_score':max(sc1,sc2)};vres.append(row);fits[la]=m;print('VMS',json.dumps(row,separators=(',',':')),flush=True)
 rank=sorted(vres,key=lambda r:r['forward_score'],reverse=True);top,second=rank[:2];margin=top['forward_score']-second['forward_score'];primary=bool(margin>=.05 and top['fit_agreement']>=.90 and b['valid_map'](fits[top['lang']]) and Sho['coverage']>=.99);signal={'top':top['lang'],'top_score':top['forward_score'],'second':second['lang'],'second_score':second['forward_score'],'margin':margin,'fit_agreement':top['fit_agreement'],'primary':primary};trans={}
 if primary:
  cand=top['lang'];m=fits[cand];lex=lexical_z(how,m,symbols,lms[cand],('VMS',cand));signal['lexical']=lex
  if lex['z']>=5:
   for tid in ['TTLI','VDRB']:
    trans[tid]=transfer(data,hold,tid,m,symbols,lms,cand);print('TRANSFER',tid,json.dumps(trans[tid],separators=(',',':')),flush=True)
   confirmed=all(trans[t]['candidate_rank']==1 and trans[t]['margin'] is not None and trans[t]['margin']>=.03 and trans[t]['coverage']>=.90 and trans[t]['lexical']['z']>=3 for t in ['TTLI','VDRB'])
  else:confirmed=False
  signal['confirmed']=confirmed
 else:signal['confirmed']=False
 verdict='CONFIRMED M19-HMM SIGNAL' if signal.get('confirmed') else ('TRANSCRIPTION-DEPENDENT / NOT CONFIRMED' if primary else 'NO M19-HMM SIGNAL');out.update({'vms':vres,'signal':signal,'transfers':trans,'verdict':verdict});print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':main()
