#!/usr/bin/env python3
import json,math,re,hashlib,urllib.request,urllib.parse
from collections import Counter
import numpy as np
from unidecode import unidecode

SEED0=20260809
ALPH='abcdefghiklmnopqrstuxyz';N=len(ALPH);A2I={c:i for i,c in enumerate(ALPH)}
LANGS=['latin','italian','german','french','greek','hebrew','arabic','spanish'];TARGETS=['latin','italian','german','hebrew']
TRAIN=45000;HOLD=39000;REPS=2;STEPS=14000;RESTARTS=3;NULLS=500;VMS_NULLS=1000;LEX_NULLS=128
TABLES={
'F':[1,2,3,4,5,6,7,8,9,10,10,2,12,22,4,12,24,6,16,4,20,8,24],
'M':[1,2,3,4,5,28,10,12,1,16,2,12,23,6,2,20,3,30,9,1,20,0,4],
'G':[1,2,6,4,5,8,1,6,7,1,8,8,5,6,5,2,2,1,4,1,1,3,3],
'L':[1,2,6,4,1,8,4,3,10,2,3,8,5,6,8,7,2,6,1,6,5,0,7],
'H':[1,2,6,4,5,6,3,1,3,6,2,4,1,6,7,2,8,6,1,6,1,0,7],
}
VALUES=sorted(set(sum(TABLES.values(),[])));NV=len(VALUES);V2I={v:i for i,v in enumerate(VALUES)}
LETTER_VALS=[]
for i in range(N):LETTER_VALS.append(sorted(set(TABLES[t][i] for t in TABLES)))
EMIT=np.zeros((N,NV),float)
for l,vs in enumerate(LETTER_VALS):
 for v in vs:EMIT[l,V2I[v]]=1/len(vs)
LM_URLS={
'latin':'https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-ITTB/master/la_ittb-ud-train.conllu',
'italian':'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu',
'german':'https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu',
'french':'https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-train.conllu',
'greek':'https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-Perseus/master/grc_perseus-ud-train.conllu',
'hebrew':'https://raw.githubusercontent.com/UniversalDependencies/UD_Hebrew-HTB/master/he_htb-ud-train.conllu',
'arabic':'https://raw.githubusercontent.com/UniversalDependencies/UD_Arabic-PADT/master/ar_padt-ud-train.conllu',
'spanish':'https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-AnCora/master/es_ancora-ud-train.conllu'}
SEF='https://storage.googleapis.com/sefaria-export/json/Halakhah/Mishneh Torah/Sefer Madda/Mishneh Torah, Torah Study/Hebrew/Torat Emet 363.json'
SLIM='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json'

def seed(*p):
 h=hashlib.sha256(('::'.join(map(str,p))).encode()).digest();return (SEED0+int.from_bytes(h[:8],'big'))&0xffffffff

def fetch(u):
 q=urllib.parse.quote(u,safe=':/?=&%');req=urllib.request.Request(q,headers={'User-Agent':'Mozilla/5.0 M19-v07'})
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
 s=unidecode(s).lower().replace('j','i').replace('v','u').replace('w','u');out=[]
 for w in re.findall(r'[a-z]+',s):
  z=''.join(c for c in w if c in A2I)
  if z:out.append(z)
 return out

def split_hold(ss):return [s for i,s in enumerate(ss) if i%5!=0],[s for i,s in enumerate(ss) if i%5==0]

def pool_text(ss):return ' '.join(w for s in ss for w in norm_words(s))

def sef_text():
 obj=json.loads(fetch(SEF));chunks=[]
 def walk(x):
  if isinstance(x,str):chunks.append(x)
  elif isinstance(x,list):
   for y in x:walk(y)
 walk(obj.get('text',[]));return ' '.join(norm_words(' '.join(chunks)))

def build_lm(ss):
 a=.25;T=np.ones((N,N))*a;st=np.ones(N)*a;en=np.ones(N)*a;uni=np.ones(N)*a;vocab=set();letters=0
 for s in ss:
  for w in norm_words(s):
   q=[A2I[c] for c in w];vocab.add(w);letters+=len(q)
   if not q:continue
   st[q[0]]+=1;en[q[-1]]+=1
   for x in q:uni[x]+=1
   for x,y in zip(q,q[1:]):T[x,y]+=1
 T/=T.sum(axis=1,keepdims=True);st/=st.sum();en/=en.sum();uni/=uni.sum()
 return {'T':T,'st':st,'en':en,'uni':uni,'vocab':vocab,'letters':letters}

def induced(lm):
 # observed numerical channel induced by hidden language bigram + frozen BnF emission law
 uni=lm['uni'];T=lm['T'];st=lm['st'];en=lm['en']
 start=st@EMIT;start=np.maximum(start,1e-15);start/=start.sum()
 post=uni[:,None]*EMIT;den=post.sum(axis=0);post=post/np.maximum(den,1e-15)[None,:] # letter x observed value posterior
 trans=np.empty((NV,NV))
 for v in range(NV):trans[v]=post[:,v]@T@EMIT
 trans=np.maximum(trans,1e-15);trans/=trans.sum(axis=1,keepdims=True)
 end=np.maximum(post.T@en,1e-15)
 return np.log(trans),np.log(start),np.log(end)

def stats(words,symbols=None):
 if symbols is None:symbols=sorted(set(c for w in words for c in w))
 s2i={s:i for i,s in enumerate(symbols)};n=len(symbols);B=np.zeros((n,n),np.int64);st=np.zeros(n,np.int64);en=np.zeros(n,np.int64);freq=np.zeros(n,np.int64);mapped=total=0
 for w in words:
  ids=[]
  for c in w:
   total+=1
   if c in s2i:ids.append(s2i[c]);freq[s2i[c]]+=1;mapped+=1
   else:ids.append(-1)
  if ids and ids[0]>=0:st[ids[0]]+=1
  if ids and ids[-1]>=0:en[ids[-1]]+=1
  for x,y in zip(ids,ids[1:]):
   if x>=0 and y>=0:B[x,y]+=1
 denom=int(B.sum()+st.sum()+en.sum()+freq.sum())
 return {'B':B,'st':st,'en':en,'freq':freq,'denom':max(1,denom),'symbols':symbols,'coverage':mapped/max(1,total),'letters':total}

def valid_map(m):
 c=np.bincount(m,minlength=NV);return len(m)==25 and np.all(c>=1) and np.all(c<=2) and int(np.sum(c==2))==6

def score(S,m,comp):
 lt,ls,le=comp;cnt=np.bincount(m,minlength=NV);num=float(np.sum(S['B']*lt[np.ix_(m,m)])+np.dot(S['st'],ls[m])+np.dot(S['en'],le[m]))
 # observed surface form is chosen uniformly within a numerical value's 1/2 homophones
 num-=float(np.dot(S['freq'],np.log(cnt[m])))
 return num/S['denom']

def init_map(rng):
 dup=sorted(map(int,rng.choice(NV,6,replace=False)));arr=np.array(list(range(NV))+dup,dtype=np.int16);rng.shuffle(arr);return arr

def optimize(S,comp,tag,steps=STEPS,restarts=RESTARTS):
 best=(-1e100,None)
 for rr in range(restarts):
  rng=np.random.default_rng(seed('opt',tag,rr));m=init_map(rng);s=score(S,m,comp)
  # empirical temperature from random legal neighbor deltas
  ds=[]
  for _ in range(50):
   a,b=rng.choice(25,2,replace=False);x=m.copy();x[a],x[b]=x[b],x[a];ds.append(abs(score(S,x,comp)-s))
  t0=max(1e-5,float(np.median(ds))*4)
  for k in range(steps):
   frac=k/max(1,steps-1);T=max(1e-6,t0*(0.01**frac));x=m.copy()
   if rng.random()<.72:
    a,b=rng.choice(25,2,replace=False);x[a],x[b]=x[b],x[a]
   else:
    cnt=np.bincount(m,minlength=NV);srcvals=np.flatnonzero(cnt==2);dstvals=np.flatnonzero(cnt==1);sv=int(rng.choice(srcvals));dv=int(rng.choice(dstvals));inds=np.flatnonzero(m==sv);i=int(rng.choice(inds));x[i]=dv
   s2=score(S,x,comp);d=s2-s
   if d>=0 or rng.random()<math.exp(max(-50,d/T)):m,s=x,s2
   if s>best[0]:best=(s,m.copy())
  # deterministic legal-neighbor polish: best improving swap or duplicate transfer, up to 10 passes
  m=best[1].copy() if best[1] is not None else m;s=score(S,m,comp)
  for _ in range(10):
   bd=0;bx=None
   for a in range(25):
    for b in range(a+1,25):
     if m[a]==m[b]:continue
     x=m.copy();x[a],x[b]=x[b],x[a];d=score(S,x,comp)-s
     if d>bd+1e-12:bd=d;bx=x
   cnt=np.bincount(m,minlength=NV)
   for sv in np.flatnonzero(cnt==2):
    for dv in np.flatnonzero(cnt==1):
     for i in np.flatnonzero(m==sv):
      x=m.copy();x[i]=dv;d=score(S,x,comp)-s
      if d>bd+1e-12:bd=d;bx=x
   if bx is None:break
   m=bx;s+=bd
   if s>best[0]:best=(s,m.copy())
 return best

def perm_z(S,m,comp,tag,nnull):
 obs=score(S,m,comp);rng=np.random.default_rng(seed('perm',tag));vals=[]
 for _ in range(nnull):x=m.copy();rng.shuffle(x);vals.append(score(S,x,comp))
 mu=float(np.mean(vals));sd=float(np.std(vals,ddof=1));return obs,mu,sd,(obs-mu)/sd if sd>1e-15 else 0.0

def choose_span(pool,n,tag):
 pos=[i for i,c in enumerate(pool) if c!=' ']
 if len(pos)<n:raise RuntimeError(('pool short',tag,len(pos),n))
 st=seed('span',tag)%(len(pos)-n+1);a=pos[st];b=pos[st+n-1]+1;return pool[a:b].strip()

def split_letters(text,n):
 k=0
 for i,c in enumerate(text):
  if c!=' ':k+=1
  if k==n:return text[:i+1].strip(),text[i+1:].strip()
 raise RuntimeError('split')

def generate_control(plain,lang,rep):
 rng=np.random.default_rng(seed('values',lang,rep));vals=[]
 for c in plain:
  if c==' ':vals.append(None)
  else:vs=LETTER_VALS[A2I[c]];vals.append(V2I[int(rng.choice(vs))])
 # top six values in training receive duplicate surface homophones
 cnt=Counter(v for v in vals[:next(i for i,_ in enumerate(vals) if False)] ) if False else Counter()
 letters=0
 for v in vals:
  if v is None:continue
  if letters<TRAIN:cnt[v]+=1
  letters+=1
 dup=[v for v,_ in sorted(cnt.items(),key=lambda kv:(-kv[1],kv[0]))[:6]]
 if len(dup)!=6:raise RuntimeError('dup')
 rawforms={v:[v] for v in range(NV)}
 for j,v in enumerate(dup):rawforms[v].append(NV+j)
 perm=np.arange(25);rng2=np.random.default_rng(seed('opaque',lang,rep));rng2.shuffle(perm);raw2surf={raw:int(perm[raw]) for raw in range(25)};surf2val=np.full(25,-1,np.int16)
 for v,forms in rawforms.items():
  for raw in forms:surf2val[raw2surf[raw]]=v
 out=[];letters=0;used=set()
 for v in vals:
  if v is None:out.append(' ');continue
  raw=int(rng.choice(rawforms[v]));sid=raw2surf[raw];out.append(chr(65+sid));
  if letters<TRAIN:used.add(sid)
  letters+=1
 if len(used)!=25:raise RuntimeError(('not all control forms observed',lang,rep,len(used)))
 assert valid_map(surf2val)
 return ''.join(out),surf2val

def weighted_map_acc(S,m,true):
 return float(np.dot(S['freq'],m==true)/max(1,S['freq'].sum()))

def half_words(words,nletters):
 a=[];b=[];n=0
 for w in words:
  if n+len(w)<=nletters:a.append(w);n+=len(w)
  elif n<nletters:
   k=nletters-n
   if k:a.append(w[:k])
   if k<len(w):b.append(w[k:])
   n=nletters
  else:b.append(w)
 return a,b

def agreement(freq,m1,m2):return float(np.dot(freq,m1==m2)/max(1,freq.sum()))
def load_sources():
 lms={};holds={};meta={}
 for lang,u in LM_URLS.items():
  ss=conllu(fetch(u));tr,ho=split_hold(ss) if lang in TARGETS else (ss,[]);lm=build_lm(tr);lms[lang]=lm;meta[lang]={'train_sentences':len(tr),'letters':lm['letters']}
  if lang in TARGETS:holds[lang]=pool_text(ho);meta[lang]['hold_letters']=sum(c!=' ' for c in holds[lang])
  print('LM',lang,meta[lang],flush=True)
 holds['hebrew']=(holds['hebrew']+' '+sef_text()).strip();meta['hebrew']['hold_extended']=sum(c!=' ' for c in holds['hebrew']);print('HEBREW_EXT',meta['hebrew']['hold_extended'],flush=True)
 return lms,holds,meta

def extract_page(data,f,tid):
 words=[]
 for k,line in sorted(data['pages'][f].items(),key=lambda kv:int(kv[0]) if str(kv[0]).isdigit() else 99999):
  for tok in line.get('t',{}).get(tid,'').split():
   z=''.join(c.lower() for c in tok if c.isalpha())
   if z:words.append(z)
 return words

def vms_split(data):
 pages=[]
 for f in data['pages']:
  w=extract_page(data,f,'ZLZI')
  if w:pages.append((f,w,sum(map(len,w))))
 pages=sorted(pages,key=lambda p:seed('foliosplit',p[0]));nh=max(1,int(round(.2*len(pages))));hold=pages[:nh];train=sorted(pages[nh:],key=lambda p:seed('trsample',p[0]));sample=[];n=0
 for p in train:
  sample.append(p);n+=p[2]
  if n>=TRAIN:break
 return sample,hold,pages

def combine(pp):return [w for _,ws,_ in pp for w in ws]
def viterbi(word,m,symbols,lm):
 s2i={s:i for i,s in enumerate(symbols)};obs=[]
 for c in word:
  if c not in s2i:return None
  obs.append(int(m[s2i[c]]))
 lt=np.log(np.maximum(lm['T'],1e-300));ls=np.log(np.maximum(lm['st'],1e-300));le=np.log(np.maximum(lm['en'],1e-300));loge=np.log(np.maximum(EMIT,1e-300));L=len(obs);dp=np.full((L,N),-1e100);back=np.full((L,N),-1,np.int16);dp[0]=ls+loge[:,obs[0]]
 for i in range(1,L):
  mat=dp[i-1][:,None]+lt
  back[i]=np.argmax(mat,axis=0);dp[i]=np.max(mat,axis=0)+loge[:,obs[i]]
 last=int(np.argmax(dp[-1]+le));q=[last]
 for i in range(L-1,0,-1):q.append(int(back[i,q[-1]]))
 return ''.join(ALPH[x] for x in q[::-1])
def lexical(words,m,symbols,lm):
 hit=tot=0
 for w in words:
  d=viterbi(w,m,symbols,lm)
  if d is None:continue
  tot+=1;hit+=int(d in lm['vocab'])
 return hit/max(1,tot),hit,tot
def lexical_z(words,m,symbols,lm,tag):
 obs,_,_=lexical(words,m,symbols,lm);rng=np.random.default_rng(seed('lex',tag));vals=[]
 for _ in range(LEX_NULLS):x=m.copy();rng.shuffle(x);vals.append(lexical(words,x,symbols,lm)[0])
 mu=float(np.mean(vals));sd=float(np.std(vals,ddof=1));return {'fraction':obs,'null_mean':mu,'null_sd':sd,'z':(obs-mu)/sd if sd>1e-15 else 0.0}
def transfer(data,hold,tid,m,symbols,lms,comps,cand):
 words=[]
 for f,_,_ in hold:
  if f in data['pages']:words.extend(extract_page(data,f,tid))
 S=stats(words,symbols);rank=[]
 for lang in LANGS:
  _,_,_,z=perm_z(S,m,comps[lang],('xfer',tid,cand,lang),400);rank.append((lang,z))
 rank.sort(key=lambda x:x[1],reverse=True);lex=lexical_z(words,m,symbols,lms[cand],('xfer',tid,cand));return {'ranking':rank,'candidate_rank':1+next(i for i,x in enumerate(rank) if x[0]==cand),'candidate_z':next(x[1] for x in rank if x[0]==cand),'lexical':lex,'coverage':S['coverage'],'letters':S['letters']}

def main():
 lms,holds,meta=load_sources();comps={l:induced(lms[l]) for l in LANGS};controls=[]
 for lang in TARGETS:
  for rep in range(REPS):
   span=choose_span(holds[lang],TRAIN+HOLD,(lang,rep));cipher,true=generate_control(span,lang,rep);ctra,cho=split_letters(cipher,TRAIN);trw=ctra.split();how=cho.split();symbols=[chr(65+i) for i in range(25)];Str=stats(trw,symbols);Sho=stats(how,symbols);rank=[];fits={}
   for cand in LANGS:
    _,m=optimize(Str,comps[cand],('control',lang,rep,cand));_,_,_,z=perm_z(Sho,m,comps[cand],('controlhold',lang,rep,cand),NULLS);rank.append((cand,z));fits[cand]=m
   rank.sort(key=lambda x:x[1],reverse=True);m=fits[lang];acc=weighted_map_acc(Sho,m,true);h1,h2=half_words(trw,TRAIN//2);S1=stats(h1,symbols);S2=stats(h2,symbols);_,m1=optimize(S1,comps[lang],('half1',lang,rep),9000,2);_,m2=optimize(S2,comps[lang],('half2',lang,rep),9000,2);agr=agreement(Str['freq'],m1,m2)
   row={'lang':lang,'rep':rep,'top':rank[0][0],'target_rank':1+next(i for i,x in enumerate(rank) if x[0]==lang),'target_z':next(x[1] for x in rank if x[0]==lang),'mapping_acc':acc,'half_agreement':agr,'ranking':rank};controls.append(row);print('CONTROL',json.dumps(row,separators=(',',':')),flush=True)
 gate={'correct':sum(r['top']==r['lang'] for r in controls),'median_acc':float(np.median([r['mapping_acc'] for r in controls])),'min_acc':float(min(r['mapping_acc'] for r in controls)),'median_z':float(np.median([r['target_z'] for r in controls])),'median_agreement':float(np.median([r['half_agreement'] for r in controls])),'min_agreement':float(min(r['half_agreement'] for r in controls))};gate.update({'Q1':gate['correct']==8,'Q2':gate['median_acc']>=.95,'Q3':gate['min_acc']>=.85,'Q4':gate['median_z']>=10,'Q5':gate['median_agreement']>=.90 and gate['min_agreement']>=.75});gate['pass']=all(gate[k] for k in ['Q1','Q2','Q3','Q4','Q5']);print('CONTROL_GATE',json.dumps(gate,separators=(',',':')),flush=True);out={'protocol':'v0.7','values':VALUES,'lm_meta':meta,'controls':controls,'gate':gate}
 if not gate['pass']:
  out['verdict']='INSTRUMENT NOT QUALIFIED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
 data=json.loads(fetch(SLIM));sample,hold,pages=vms_split(data);trw=combine(sample);how=combine(hold);symbols=sorted(set(c for w in trw for c in w));Str=stats(trw,symbols);Sho=stats(how,symbols);census={'pages':len(pages),'train_pages':len(sample),'hold_pages':len(hold),'train_letters':sum(p[2] for p in sample),'hold_letters':sum(p[2] for p in hold),'symbols':symbols,'nsym':len(symbols),'hold_mapping_coverage':Sho['coverage']};out['vms_census']=census;print('VMS_CENSUS',json.dumps(census,separators=(',',':')),flush=True)
 if len(symbols)!=25 or Sho['coverage']<.99:
  out['verdict']='UNDERPOWERED: SURFACE ALPHABET/COVERAGE';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
 vres=[];fits={};agreements={}
 h1,h2=half_words(trw,Str['letters']//2);S1=stats(h1,symbols);S2=stats(h2,symbols)
 for lang in LANGS:
  sc,m=optimize(Str,comps[lang],('VMS',lang));obs,mu,sd,z=perm_z(Sho,m,comps[lang],('VMShold',lang),VMS_NULLS);_,m1=optimize(S1,comps[lang],('VMShalf1',lang),9000,2);_,m2=optimize(S2,comps[lang],('VMShalf2',lang),9000,2);agr=agreement(Str['freq'],m1,m2);row={'lang':lang,'train_score':sc,'hold_score':obs,'z':z,'half_agreement':agr,'mapping':{symbols[i]:VALUES[int(m[i])] for i in range(25)}};vres.append(row);fits[lang]=m;print('VMS',json.dumps(row,separators=(',',':')),flush=True)
 rank=sorted(vres,key=lambda r:r['z'],reverse=True);top,second=rank[:2];margin=top['z']-second['z'];primary=bool(top['z']>=10 and margin>=5 and top['half_agreement']>=.80 and valid_map(fits[top['lang']]) and Sho['coverage']>=.99);signal={'top':top['lang'],'top_z':top['z'],'second':second['lang'],'second_z':second['z'],'margin':margin,'half_agreement':top['half_agreement'],'primary':primary};trans={}
 if primary:
  cand=top['lang'];m=fits[cand];lex=lexical_z(how,m,symbols,lms[cand],('VMS',cand));signal['lexical']=lex
  if lex['z']>=5:
   for tid in ['TTLI','VDRB']:
    trans[tid]=transfer(data,hold,tid,m,symbols,lms,comps,cand);print('TRANSFER',tid,json.dumps(trans[tid],separators=(',',':')),flush=True)
   confirmed=all(trans[t]['candidate_rank']==1 and trans[t]['candidate_z']>=7 and trans[t]['lexical']['z']>=3 and trans[t]['coverage']>=.90 for t in ['TTLI','VDRB'])
  else:confirmed=False
  signal['confirmed']=confirmed
 else:signal['confirmed']=False
 verdict='CONFIRMED M19 SIGNAL' if signal.get('confirmed') else ('M19 CANDIDATE NOT CONFIRMED' if primary else 'NO M19 SIGNAL');out.update({'vms':vres,'signal':signal,'transfers':trans,'verdict':verdict});print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':main()
