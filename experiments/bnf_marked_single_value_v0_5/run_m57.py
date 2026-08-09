#!/usr/bin/env python3
import json, math, re, hashlib, urllib.request, urllib.parse
from collections import Counter
import numpy as np
from unidecode import unidecode

SEED0=20260809
ALPH='abcdefghiklmnopqrstuxyz'; N=len(ALPH); A2I={c:i for i,c in enumerate(ALPH)}
LANGS=['latin','italian','german','french','greek','hebrew','arabic','spanish']
TARGETS=['latin','italian','german','hebrew']
TRAIN_LETTERS=45000; HOLD_LETTERS=39000
REPS=2; STEPS=28000; RESTARTS=5; NULLS=1000; LEX_NULLS=128
TABLES={
'F':[1,2,3,4,5,6,7,8,9,10,10,2,12,22,4,12,24,6,16,4,20,8,24],
'M':[1,2,3,4,5,28,10,12,1,16,2,12,23,6,2,20,3,30,9,1,20,0,4],
'G':[1,2,6,4,5,8,1,6,7,1,8,8,5,6,5,2,2,1,4,1,1,3,3],
'L':[1,2,6,4,1,8,4,3,10,2,3,8,5,6,8,7,2,6,1,6,5,0,7],
'H':[1,2,6,4,5,6,3,1,3,6,2,4,1,6,7,2,8,6,1,6,1,0,7],
}
LM_URLS={
'latin':'https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-train.conllu',
'italian':'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu',
'german':'https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu',
'french':'https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-train.conllu',
'greek':'https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-train.conllu',
'hebrew':'https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/master/he_htb-ud-train.conllu',
'arabic':'https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-train.conllu',
'spanish':'https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-train.conllu',
}
SEF='https://storage.googleapis.com/sefaria-export/json/Halakhah/Mishneh Torah/Sefer Madda/Mishneh Torah, Torah Study/Hebrew/Torat Emet 363.json'
SLIM='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json'


def stable_seed(*p):
 h=hashlib.sha256(('::'.join(map(str,p))).encode()).digest();return (SEED0+int.from_bytes(h[:8],'big'))&0xffffffff

def fetch(u):
 q=urllib.parse.quote(u,safe=':/?=&%')
 req=urllib.request.Request(q,headers={'User-Agent':'Mozilla/5.0 M57/0.5'})
 with urllib.request.urlopen(req,timeout=90) as r:return r.read().decode('utf-8','replace')

def conllu(txt):
 out=[];cur=[]
 for ln in txt.splitlines():
  if not ln:
   if cur:out.append(' '.join(cur));cur=[]
   continue
  if ln.startswith('#'):continue
  c=ln.split('\t')
  if len(c)>=2 and c[0].isdigit():cur.append(c[1])
 if cur:out.append(' '.join(cur))
 return out

def norm_words(s):
 s=unidecode(s).lower().replace('j','i').replace('v','u').replace('w','u')
 out=[]
 for w in re.findall(r'[a-z]+',s):
  z=''.join(c for c in w if c in A2I)
  if z:out.append(z)
 return out

def split_train_hold(ss):return [s for i,s in enumerate(ss) if i%5!=0],[s for i,s in enumerate(ss) if i%5==0]

def concat_words(ss):
 out=[]
 for s in ss:out.extend(norm_words(s))
 return ' '.join(out)

def sefaria_words():
 obj=json.loads(fetch(SEF));chunks=[]
 def walk(x):
  if isinstance(x,str):chunks.append(x)
  elif isinstance(x,list):
   for y in x:walk(y)
 walk(obj.get('text',[]));return ' '.join(norm_words(' '.join(chunks)))

def build_codes():
 codes=[]
 for t in ['F','M','G','L','H']:
  d={}
  for i,v in enumerate(TABLES[t]):d.setdefault(v,[]).append(i)
  for v in sorted(d):codes.append({'table':t,'value':v,'cand':tuple(d[v])})
 assert len(codes)==57
 return codes
CODES=build_codes()


def build_lm(ss):
 alpha=.25
 tr=np.ones((N,N),dtype=float)*alpha; st=np.ones(N)*alpha; en=np.ones(N)*alpha; unig=np.ones(N)*alpha; vocab=set();nlet=0
 for raw in ss:
  for w in norm_words(raw):
   a=[A2I[c] for c in w];vocab.add(w);nlet+=len(a)
   if not a:continue
   st[a[0]]+=1;en[a[-1]]+=1
   for x in a:unig[x]+=1
   for x,y in zip(a,a[1:]):tr[x,y]+=1
 tr/=tr.sum(axis=1,keepdims=True);st/=st.sum();en/=en.sum();unig/=unig.sum()
 return {'tr':tr,'st':st,'en':en,'unig':unig,'vocab':vocab,'letters':nlet}

def compat_for_lm(lm):
 nc=len(CODES);lc=np.empty((nc,nc));ls=np.empty(nc);le=np.empty(nc)
 for i,c in enumerate(CODES):
  a=np.array(c['cand'],dtype=int);ls[i]=math.log(float(lm['st'][a].mean()));le[i]=math.log(float(lm['en'][a].mean()))
  for j,d in enumerate(CODES):
   b=np.array(d['cand'],dtype=int);lc[i,j]=math.log(float(lm['tr'][np.ix_(a,b)].mean()))
 return lc,ls,le


def stats_from_words(words,symbols=None):
 if symbols is None:symbols=sorted(set(c for w in words for c in w))
 s2i={s:i for i,s in enumerate(symbols)};n=len(symbols);B=np.zeros((n,n),dtype=np.int64);st=np.zeros(n,dtype=np.int64);en=np.zeros(n,dtype=np.int64);freq=np.zeros(n,dtype=np.int64);mapped=total=0
 for w in words:
  ids=[]
  for c in w:
   total+=1
   if c in s2i:ids.append(s2i[c]);mapped+=1;freq[s2i[c]]+=1
   else:ids.append(-1)
  if ids and ids[0]>=0:st[ids[0]]+=1
  if ids and ids[-1]>=0:en[ids[-1]]+=1
  for a,b in zip(ids,ids[1:]):
   if a>=0 and b>=0:B[a,b]+=1
 events=int(B.sum()+st.sum()+en.sum())
 return {'B':B,'st':st,'en':en,'freq':freq,'events':events,'symbols':symbols,'coverage':mapped/max(1,total),'letters':total}

def score_map(stats,m,comp):
 lc,ls,le=comp;B=stats['B'];
 return float((np.sum(B*lc[np.ix_(m,m)])+np.dot(stats['st'],ls[m])+np.dot(stats['en'],le[m]))/max(1,stats['events']))

def delta_replace(stats,m,s,new,comp):
 lc,ls,le=comp;old=int(m[s]);B=stats['B'];n=len(m)
 if new==old:return 0.0
 d=0.0
 for j in range(n):d+=B[s,j]*(lc[new,m[j]]-lc[old,m[j]])
 for i in range(n):d+=B[i,s]*(lc[m[i],new]-lc[m[i],old])
 d-=B[s,s]*(lc[new,new]-lc[old,old])
 d+=stats['st'][s]*(ls[new]-ls[old])+stats['en'][s]*(le[new]-le[old])
 return float(d/max(1,stats['events']))

def optimize(stats,comp,tag):
 n=len(stats['symbols']);nc=len(CODES);best=(-1e99,None)
 for rr in range(RESTARTS):
  rng=np.random.default_rng(stable_seed('opt',tag,rr));m=rng.choice(nc,size=n,replace=False).astype(np.int16);used=set(map(int,m));unused=set(range(nc))-used;s=score_map(stats,m,comp)
  # data-scaled initial temperature from legal replacement deltas
  ds=[]
  for _ in range(80):
   i=int(rng.integers(n));new=int(rng.choice(tuple(unused)));ds.append(abs(delta_replace(stats,m,i,new,comp)))
  t0=max(1e-5,float(np.median(ds))*3)
  for step in range(STEPS):
   frac=step/max(1,STEPS-1);T=t0*((.01/t0)**frac) if t0>.01 else max(1e-6,t0*(.02**frac))
   if rng.random()<.18 and n>1:
    a,b=rng.choice(n,size=2,replace=False);m2=m.copy();m2[a],m2[b]=m2[b],m2[a];s2=score_map(stats,m2,comp);d=s2-s
    if d>=0 or rng.random()<math.exp(max(-50,d/max(T,1e-8))):m,s=m2,s2
   else:
    i=int(rng.integers(n));new=int(rng.choice(tuple(unused)));old=int(m[i]);d=delta_replace(stats,m,i,new,comp)
    if d>=0 or rng.random()<math.exp(max(-50,d/max(T,1e-8))):
     m[i]=new;unused.remove(new);unused.add(old);s+=d
  # deterministic coordinate polish
  for _ in range(8):
   changed=False
   for i in range(n):
    cur=int(m[i]);bestd=0.0;bestc=cur
    for c in tuple(unused):
     d=delta_replace(stats,m,i,int(c),comp)
     if d>bestd+1e-12:bestd=d;bestc=int(c)
    if bestc!=cur:
     m[i]=bestc;unused.remove(bestc);unused.add(cur);s+=bestd;changed=True
   if not changed:break
  s=score_map(stats,m,comp)
  if s>best[0]:best=(s,m.copy())
 return best

def perm_z(stats,m,comp,tag,nnull=NULLS):
 obs=score_map(stats,m,comp);rng=np.random.default_rng(stable_seed('perm',tag));vals=[]
 for _ in range(nnull):
  x=m.copy();rng.shuffle(x);vals.append(score_map(stats,x,comp))
 mu=float(np.mean(vals));sd=float(np.std(vals,ddof=1));return obs,mu,sd,(obs-mu)/sd if sd>1e-15 else 0.0


def count_letters(s):return sum(c!=' ' for c in s)

def choose_span(pool,nletters,tag):
 pos=[i for i,c in enumerate(pool) if c!=' ']
 if len(pos)<nletters:raise RuntimeError(('short pool',tag,len(pos),nletters))
 maxstart=len(pos)-nletters;st=stable_seed('span',tag)%(maxstart+1);a=pos[st];b=pos[st+nletters-1]+1
 return pool[a:b].strip()

def split_at_letters(s,n):
 k=0
 for i,c in enumerate(s):
  if c!=' ':k+=1
  if k==n:return s[:i+1].strip(),s[i+1:].strip()
 raise RuntimeError('split fail')

def choose_subset(lang,rep,attempt):
 rng=np.random.default_rng(stable_seed('subset',lang,rep,attempt));sel=sorted(map(int,rng.choice(len(CODES),25,replace=False)))
 cov=set()
 for c in sel:cov.update(CODES[c]['cand'])
 return sel if len(cov)==N else None

def encrypt_control(plain,lang,rep):
 # deterministic rejection until all selected codes are train-observed
 for attempt in range(10000):
  sel=choose_subset(lang,rep,attempt)
  if sel is None:continue
  rng=np.random.default_rng(stable_seed('enc',lang,rep,attempt));byletter={i:[c for c in sel if i in CODES[c]['cand']] for i in range(N)}
  if any(not byletter[i] for i in range(N)):continue
  perm=np.arange(25);rng.shuffle(perm);code2surf={c:int(perm[j]) for j,c in enumerate(sel)};surf2code=np.full(25,-1,dtype=np.int16)
  for c,s in code2surf.items():surf2code[s]=c
  out=[];trueletters=[];lettercount=0;usedtrain=set()
  for ch in plain:
   if ch==' ':out.append(' ');continue
   li=A2I[ch];c=int(rng.choice(byletter[li]));sid=code2surf[c];out.append(chr(65+sid));trueletters.append(li);lettercount+=1
   if lettercount<=TRAIN_LETTERS:usedtrain.add(sid)
  if len(usedtrain)==25:return ''.join(out),np.asarray(trueletters,dtype=np.int16),surf2code,attempt
 raise RuntimeError(('cannot generate control',lang,rep))

def true_compat_accuracy(hold_words,true_letters,fitmap):
 flat=[]
 for w in hold_words:
  for c in w:flat.append(ord(c)-65)
 if len(flat)!=len(true_letters):raise RuntimeError(('align',len(flat),len(true_letters)))
 ok=0
 for s,l in zip(flat,true_letters):
  if int(l) in CODES[int(fitmap[int(s)])]['cand']:ok+=1
 return ok/max(1,len(flat))


def load_sources():
 lms={};holds={};meta={}
 for lang in LANGS:
  ss=conllu(fetch(LM_URLS[lang]));tr,ho=split_train_hold(ss) if lang in TARGETS else (ss,[]);lms[lang]=build_lm(tr);meta[lang]={'train_sentences':len(tr),'lm_letters':lms[lang]['letters']}
  if lang in TARGETS:holds[lang]=concat_words(ho);meta[lang]['hold_letters']=count_letters(holds[lang])
  print('LM',lang,meta[lang],flush=True)
 holds['hebrew']=(holds['hebrew']+' '+sefaria_words()).strip();meta['hebrew']['hold_letters_extended']=count_letters(holds['hebrew']);print('HEBREW_HOLD_EXT',meta['hebrew']['hold_letters_extended'],flush=True)
 return lms,holds,meta

def extract_page(data,f,tid):
 lines=data['pages'][f];parts=[]
 for k,line in sorted(lines.items(),key=lambda kv:int(kv[0]) if str(kv[0]).isdigit() else 99999):
  t=line.get('t',{}).get(tid,'')
  for tok in t.split():
   z=''.join(c.lower() for c in tok if c.isalpha())
   if z:parts.append(z)
 return parts

def vms_split(data):
 pages=[]
 for f in data['pages']:
  w=extract_page(data,f,'ZLZI')
  if w:pages.append((f,w,sum(map(len,w))))
 pages=sorted(pages,key=lambda p:stable_seed('M57folio',p[0]));nh=max(1,int(round(.2*len(pages))));hold=pages[:nh];train=pages[nh:]
 train=sorted(train,key=lambda p:stable_seed('M57train',p[0]));sample=[];n=0
 for p in train:
  sample.append(p);n+=p[2]
  if n>=TRAIN_LETTERS:break
 return sample,hold,pages

def combine_pages(pp):return [w for _,words,_ in pp for w in words]

def viterbi_word(word,maparr,lm,s2i):
 # hard-break words are skipped for lexical decoding
 codes=[]
 for ch in word:
  if ch not in s2i:return None
  codes.append(int(maparr[s2i[ch]]))
 tr=np.log(lm['tr']);st=np.log(lm['st']);en=np.log(lm['en']);L=len(codes)
 dp=np.full((L,N),-1e100);back=np.full((L,N),-1,dtype=np.int16)
 cand=CODES[codes[0]]['cand'];dp[0,list(cand)]=st[list(cand)]
 for i in range(1,L):
  allowed=CODES[codes[i]]['cand'];mat=dp[i-1][:,None]+tr
  for j in allowed:
   q=int(np.argmax(mat[:,j]));dp[i,j]=mat[q,j];back[i,j]=q
 allowed=CODES[codes[-1]]['cand'];last=max(allowed,key=lambda j:dp[-1,j]+en[j]);seq=[int(last)]
 for i in range(L-1,0,-1):seq.append(int(back[i,seq[-1]]))
 seq=seq[::-1];return ''.join(ALPH[i] for i in seq)

def lexical_score(words,maparr,lm,vocab,symbols):
 s2i={s:i for i,s in enumerate(symbols)};hit=tot=0
 for w in words:
  d=viterbi_word(w,maparr,lm,s2i)
  if d is None:continue
  tot+=1;hit+=int(d in vocab)
 return hit/max(1,tot),hit,tot

def lexical_z(words,maparr,lm,vocab,symbols,tag):
 obs,_,_=lexical_score(words,maparr,lm,vocab,symbols);rng=np.random.default_rng(stable_seed('lex',tag));vals=[]
 for _ in range(LEX_NULLS):
  x=maparr.copy();rng.shuffle(x);vals.append(lexical_score(words,x,lm,vocab,symbols)[0])
 mu=float(np.mean(vals));sd=float(np.std(vals,ddof=1));return {'fraction':obs,'null_mean':mu,'null_sd':sd,'z':(obs-mu)/sd if sd>1e-15 else 0.0}

def transcriber_transfer(data,holdfolios,tid,maparr,symbols,lms,comps,cand):
 words=[]
 for f,_,_ in holdfolios:
  if f in data['pages']:words.extend(extract_page(data,f,tid))
 stats=stats_from_words(words,symbols);rank=[]
 for lang in LANGS:
  _,_,_,z=perm_z(stats,maparr,comps[lang],('xfer',tid,cand,lang),400);rank.append((lang,z))
 rank.sort(key=lambda x:x[1],reverse=True);lex=lexical_z(words,maparr,lms[cand],lms[cand]['vocab'],symbols,('xfer',tid,cand))
 return {'ranking':rank,'candidate_rank':1+next(i for i,x in enumerate(rank) if x[0]==cand),'candidate_z':next(x[1] for x in rank if x[0]==cand),'lexical':lex,'coverage':stats['coverage'],'letters':stats['letters']}


def main():
 lms,holds,meta=load_sources();comps={l:compat_for_lm(lms[l]) for l in LANGS}
 # Controls
 controls=[]
 for lang in TARGETS:
  pool=holds[lang]
  for rep in range(REPS):
   span=choose_span(pool,TRAIN_LETTERS+HOLD_LETTERS,(lang,rep));ptra,phold=split_at_letters(span,TRAIN_LETTERS);cipher,trueletters,true_map,attempt=encrypt_control(span,lang,rep);ctra,chold=split_at_letters(cipher,TRAIN_LETTERS)
   trainwords=ctra.split();holdwords=chold.split();stats_tr=stats_from_words(trainwords,[chr(65+i) for i in range(25)]);stats_ho=stats_from_words(holdwords,stats_tr['symbols'])
   rankings=[];fits={}
   for cand in LANGS:
    sc,mp=optimize(stats_tr,comps[cand],('control',lang,rep,cand));obs,mu,sd,z=perm_z(stats_ho,mp,comps[cand],('controlhold',lang,rep,cand));rankings.append((cand,z,obs));fits[cand]=mp
   rankings.sort(key=lambda x:x[1],reverse=True);fit=fits[lang]
   # align hold true letters only: trueletters includes full span
   htrue=trueletters[TRAIN_LETTERS:TRAIN_LETTERS+HOLD_LETTERS]
   acc=true_compat_accuracy(holdwords,htrue,fit)
   row={'lang':lang,'rep':rep,'top':rankings[0][0],'target_rank':1+next(i for i,x in enumerate(rankings) if x[0]==lang),'target_z':next(x[1] for x in rankings if x[0]==lang),'compat_acc':acc,'generation_attempt':attempt,'ranking':[(x[0],x[1]) for x in rankings]};controls.append(row);print('CONTROL',json.dumps(row,separators=(',',':')),flush=True)
 gate={'correct':sum(r['top']==r['lang'] for r in controls),'median_acc':float(np.median([r['compat_acc'] for r in controls])),'min_acc':float(min(r['compat_acc'] for r in controls)),'median_z':float(np.median([r['target_z'] for r in controls]))}
 gate.update({'Q1':gate['correct']==8,'Q2':gate['median_acc']>=.95,'Q3':gate['min_acc']>=.85,'Q4':gate['median_z']>=10,'Q5':True});gate['pass']=all(gate[k] for k in ['Q1','Q2','Q3','Q4','Q5']);print('CONTROL_GATE',json.dumps(gate,separators=(',',':')),flush=True)
 out={'protocol':'v0.5','code_count':len(CODES),'lm_meta':meta,'controls':controls,'gate':gate}
 if not gate['pass']:
  out['verdict']='INSTRUMENT NOT QUALIFIED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
 # Voynich
 data=json.loads(fetch(SLIM));sample,hold,pages=vms_split(data);trwords=combine_pages(sample);howords=combine_pages(hold);symbols=sorted(set(c for w in trwords for c in w));stats_tr=stats_from_words(trwords,symbols);stats_ho=stats_from_words(howords,symbols)
 census={'all_pages':len(pages),'train_sample_pages':len(sample),'hold_pages':len(hold),'train_letters':sum(p[2] for p in sample),'hold_letters':sum(p[2] for p in hold),'symbols':symbols,'nsym':len(symbols),'hold_mapping_coverage':stats_ho['coverage']};out['vms_census']=census;print('VMS_CENSUS',json.dumps(census,separators=(',',':')),flush=True)
 if stats_ho['coverage']<.99:
  out['verdict']='UNDERPOWERED: HOLD MAPPING COVERAGE';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
 vres=[];fits={}
 for lang in LANGS:
  sc,mp=optimize(stats_tr,comps[lang],('VMS',lang));obs,mu,sd,z=perm_z(stats_ho,mp,comps[lang],('VMShold',lang));row={'lang':lang,'train_score':sc,'hold_score':obs,'z':z,'codes':[f"{CODES[int(c)]['table']}:{CODES[int(c)]['value']}" for c in mp]};vres.append(row);fits[lang]=mp;print('VMS',json.dumps(row,separators=(',',':')),flush=True)
 rank=sorted(vres,key=lambda r:r['z'],reverse=True);top,second=rank[:2];margin=top['z']-second['z'];primary=bool(top['z']>=10 and margin>=5)
 signal={'top':top['lang'],'top_z':top['z'],'second':second['lang'],'second_z':second['z'],'margin':margin,'primary':primary};transfers={}
 if primary:
  cand=top['lang'];mp=fits[cand];lex=lexical_z(howords,mp,lms[cand],lms[cand]['vocab'],symbols,('VMS',cand));signal['lexical']=lex
  if lex['z']>=5:
   for tid in ['TTLI','VDRB']:
    transfers[tid]=transcriber_transfer(data,hold,tid,mp,symbols,lms,comps,cand);print('TRANSFER',tid,json.dumps(transfers[tid],separators=(',',':')),flush=True)
   confirmed=all(transfers[t]['candidate_rank']==1 and transfers[t]['candidate_z']>=7 and transfers[t]['lexical']['z']>=3 and transfers[t]['coverage']>=.90 for t in ['TTLI','VDRB'])
  else:confirmed=False
  signal['confirmed']=confirmed
 else:signal['confirmed']=False
 if signal.get('confirmed'):verdict='CONFIRMED M57 SIGNAL'
 elif primary:verdict='M57 CANDIDATE NOT CONFIRMED'
 else:verdict='NO M57 SIGNAL'
 out.update({'vms':vres,'signal':signal,'transfers':transfers,'verdict':verdict});print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__':main()
