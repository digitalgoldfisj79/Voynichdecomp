#!/usr/bin/env python3
import urllib.request,json,math,hashlib
from collections import Counter
import numpy as np
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0ccea68e5eef0b551cff7cb2703c20c9868e294c/experiments/bnf_free_switch_m19_v0_7/run_m19.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8');b={'__name__':'m19base'};exec(compile(src,'run_m19.py','exec'),b)
QUAL=['latin','italian','german','french','arabic','spanish'];TRAIN_RES={3,4,8,9};QUAL_RES={2,7};NS=31;NDUP=12;STEPS=24000;RESTARTS=6
GROUPS=['cfh','ckh','cph','cth','ch','sh']

def sd(*p):return b['seed']('ATOMIC_EVA',*p)
def atom(w):
 out=[];i=0
 while i<len(w):
  g=next((g for g in GROUPS if w.startswith(g,i)),None)
  if g:out.append(g);i+=len(g)
  else:out.append(w[i]);i+=1
 return out

def load_fresh():
 lms={};pools={};meta={}
 for lang,u in b['LM_URLS'].items():
  ss=b['conllu'](b['fetch'](u));tr=[s for i,s in enumerate(ss) if i%10 in TRAIN_RES];qo=[s for i,s in enumerate(ss) if i%10 in QUAL_RES];lm=b['build_lm'](tr);lms[lang]=lm;pools[lang]=b['pool_text'](qo);meta[lang]={'lm_letters':lm['letters'],'qual_letters':sum(c!=' ' for c in pools[lang])};print('LM',lang,meta[lang],flush=True)
 return lms,pools,meta

def stats(words,symbols):
 s2i={s:i for i,s in enumerate(symbols)};n=len(symbols);B=np.zeros((n,n),np.int64);st=np.zeros(n,np.int64);en=np.zeros(n,np.int64);freq=np.zeros(n,np.int64);mapped=total=0
 for w in words:
  ids=[]
  for c in w:
   total+=1;i=s2i.get(c,-1);ids.append(i)
   if i>=0:freq[i]+=1;mapped+=1
  if ids and ids[0]>=0:st[ids[0]]+=1
  if ids and ids[-1]>=0:en[ids[-1]]+=1
  for x,y in zip(ids,ids[1:]):
   if x>=0 and y>=0:B[x,y]+=1
 denom=int(B.sum()+st.sum()+en.sum()+freq.sum());return {'B':B,'st':st,'en':en,'freq':freq,'denom':max(1,denom),'coverage':mapped/max(1,total),'letters':total}

def valid(m):
 c=np.bincount(m,minlength=b['NV']);return len(m)==NS and np.all(c>=1) and np.all(c<=2) and int(np.sum(c==2))==NDUP

def score(S,m,comp):
 lt,ls,le=comp;cnt=np.bincount(m,minlength=b['NV']);num=float(np.sum(S['B']*lt[np.ix_(m,m)])+np.dot(S['st'],ls[m])+np.dot(S['en'],le[m]));num-=float(np.dot(S['freq'],np.log(cnt[m])));return num/S['denom']

def init_map(rng):
 dup=sorted(map(int,rng.choice(b['NV'],NDUP,replace=False)));a=np.array(list(range(b['NV']))+dup,dtype=np.int16);rng.shuffle(a);return a

def optimize(S,comp,tag):
 best=(-1e100,None)
 for rr in range(RESTARTS):
  rng=np.random.default_rng(sd('opt',tag,rr));m=init_map(rng);s=score(S,m,comp);ds=[]
  for _ in range(40):
   i,j=rng.choice(NS,2,replace=False);x=m.copy();x[i],x[j]=x[j],x[i];ds.append(abs(score(S,x,comp)-s))
  t0=max(1e-5,float(np.median(ds))*4)
  for k in range(STEPS):
   T=max(1e-6,t0*(0.01**(k/max(1,STEPS-1))));x=m.copy()
   if rng.random()<.72:
    i,j=rng.choice(NS,2,replace=False);x[i],x[j]=x[j],x[i]
   else:
    cnt=np.bincount(m,minlength=b['NV']);sv=int(rng.choice(np.flatnonzero(cnt==2)));dv=int(rng.choice(np.flatnonzero(cnt==1)));ii=int(rng.choice(np.flatnonzero(m==sv)));x[ii]=dv
   s2=score(S,x,comp);d=s2-s
   if d>=0 or rng.random()<math.exp(max(-50,d/T)):m,s=x,s2
   if s>best[0]:best=(s,m.copy())
  m=best[1].copy();s=score(S,m,comp)
  for _ in range(8):
   bd=0.;bx=None
   for i in range(NS):
    for j in range(i+1,NS):
     if m[i]==m[j]:continue
     x=m.copy();x[i],x[j]=x[j],x[i];dd=score(S,x,comp)-s
     if dd>bd+1e-12:bd=dd;bx=x
   cnt=np.bincount(m,minlength=b['NV'])
   for sv in np.flatnonzero(cnt==2):
    for dv in np.flatnonzero(cnt==1):
     for ii in np.flatnonzero(m==sv):
      x=m.copy();x[ii]=dv;dd=score(S,x,comp)-s
      if dd>bd+1e-12:bd=dd;bx=x
   if bx is None:break
   m=bx;s+=bd
   if s>best[0]:best=(s,m.copy())
 return best

def forward_value(obs,lm):
 if not obs:return 0.,0
 T=lm['T'];E=b['EMIT'];a=lm['st']*E[:,obs[0]];z=float(a.sum())
 if z<=0:return -1e100,0
 ll=math.log(z);a/=z
 for v in obs[1:]:
  a=(a@T)*E[:,v];z=float(a.sum())
  if z<=0:return -1e100,0
  ll+=math.log(z);a/=z
 z=float(np.dot(a,lm['en']))
 if z>0:ll+=math.log(z)
 return ll,len(obs)

def forward(words,m,symbols,lm):
 s2i={s:i for i,s in enumerate(symbols)};ll=0.;n=0;tot=known=0
 for w in words:
  obs=[];ok=True;tot+=len(w)
  for c in w:
   if c not in s2i:ok=False;break
   obs.append(int(m[s2i[c]]));known+=1
  if not ok:continue
  x,k=forward_value(obs,lm);ll+=x;n+=k
 return ll/max(1,n),n,known/max(1,tot)

def agreement(freq,a,c):return float(np.dot(freq,a==c)/max(1,freq.sum()))
def acc(freq,a,c):return float(np.dot(freq,a==c)/max(1,freq.sum()))

def choose_span(pool,n,tag):
 pos=[i for i,c in enumerate(pool) if c!=' '];st=sd('span',tag)%(len(pos)-n+1);a=pos[st];z=pos[st+n-1]+1;return pool[a:z].strip()

def gen_control(plain,lang):
 pwords=plain.split()
 for attempt in range(100):
  rng=np.random.default_rng(sd('values',lang,attempt));vwords=[];cnt=Counter();n=0
  for w in pwords:
   q=[]
   for c in w:
    vi=b['V2I'][int(rng.choice(b['LETTER_VALS'][b['A2I'][c]]))];q.append(vi)
    if n<b['TRAIN']:cnt[vi]+=1
    n+=1
   if q:vwords.append(q)
  if len(cnt)<b['NV']:continue
  dup=[v for v,_ in sorted(cnt.items(),key=lambda kv:(-kv[1],kv[0]))[:NDUP]];forms={v:[v] for v in range(b['NV'])}
  for j,v in enumerate(dup):forms[v].append(b['NV']+j)
  perm=np.arange(NS);r2=np.random.default_rng(sd('opaque',lang,attempt));r2.shuffle(perm);r2s={x:int(perm[x]) for x in range(NS)};true=np.full(NS,-1,np.int16)
  for v,ff in forms.items():
   for x in ff:true[r2s[x]]=v
  r3=np.random.default_rng(sd('surface',lang,attempt));out=[];used=set();n=0
  for w in vwords:
   q=[]
   for v in w:
    sid=r2s[int(r3.choice(forms[v]))];q.append(sid)
    if n<b['TRAIN']:used.add(sid)
    n+=1
   out.append(q)
  if len(used)==NS and valid(true):return out,true,attempt
 raise RuntimeError(('control generation failed',lang))

def split_words(words,nletters):
 a=[];c=[];n=0
 for w in words:
  if n>=nletters:c.append(w);continue
  if n+len(w)<=nletters:a.append(w);n+=len(w)
  else:
   k=nletters-n
   if k:a.append(w[:k])
   if k<len(w):c.append(w[k:])
   n=nletters
 return a,c

def v09_panels(data):
 pages=[]
 for f in data['pages']:
  w=b['extract_page'](data,f,'ZLZI')
  if w:pages.append((f,w,sum(map(len,w))))
 pages=sorted(pages,key=lambda p:b['seed']('M19HMMv09split',p[0]));nh=max(1,int(round(.2*len(pages))));hold=pages[:nh];train=pages[nh:];required=set(c for _,ws,_ in train for w in ws for c in w);cand=sorted(train,key=lambda p:b['seed']('M19HMMv09train',p[0]));sample=[];n=0;seen=set()
 for p in cand:
  sample.append(p);n+=p[2]
  for w in p[1]:seen.update(w)
  if n>=b['TRAIN'] and required.issubset(seen):break
 return sample,hold,pages

def atoms_for(data,folios):return [atom(w) for f in folios for w in b['extract_page'](data,f,'ZLZI')]
def rank_fixed(words,fits,symbols,lms):
 rows=[]
 for la in b['LANGS']:
  sc,n,cov=forward(words,fits[la],symbols,lms[la]);rows.append((la,sc,cov))
 rows.sort(key=lambda x:x[1],reverse=True);return rows

def main():
 lms,pools,meta=load_fresh();comps={la:b['induced'](lms[la]) for la in b['LANGS']};controls=[]
 for lang in QUAL:
  span=choose_span(pools[lang],b['TRAIN']+b['HOLD'],('qual',lang));cw,true,attempt=gen_control(span,lang);tr,ho=split_words(cw,b['TRAIN']);symbols=list(range(NS));Str=stats(tr,symbols);Sho=stats(ho,symbols);rows=[];fits={}
  for cand in b['LANGS']:
   sc,m=optimize(Str,comps[cand],('qual',lang,cand,1));fw,n,cov=forward(ho,m,symbols,lms[cand]);rows.append((cand,fw));fits[cand]=m
  rows.sort(key=lambda x:x[1],reverse=True);sc2,m2=optimize(Str,comps[lang],('qual',lang,lang,2));wa=acc(Sho['freq'],fits[lang],true);agr=agreement(Str['freq'],fits[lang],m2);row={'lang':lang,'top':rows[0][0],'margin':rows[0][1]-rows[1][1],'rank':1+next(i for i,x in enumerate(rows) if x[0]==lang),'map_acc':wa,'agreement':agr,'attempt':attempt};controls.append(row);print('QUAL',json.dumps(row,separators=(',',':')),flush=True)
 gate={'correct':sum(r['top']==r['lang'] for r in controls),'min_margin':min(r['margin'] for r in controls),'median_acc':float(np.median([r['map_acc'] for r in controls])),'min_acc':min(r['map_acc'] for r in controls),'min_agreement':min(r['agreement'] for r in controls)};gate['pass']=gate['correct']==6 and gate['min_margin']>=.05 and gate['median_acc']>=.95 and gate['min_acc']>=.85 and gate['min_agreement']>=.90;print('GATE',json.dumps(gate,separators=(',',':')),flush=True)
 out={'controls':controls,'gate':gate}
 if not gate['pass']:
  out['verdict']='INSTRUMENT NOT QUALIFIED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
 data=json.loads(b['fetch'](b['SLIM']));sample,hold,pages=v09_panels(data);Tf=[f for f,_,_ in sample];Hf=[f for f,_,_ in hold];Af={f for f,_,_ in pages};Cf=sorted(Af-set(Tf)-set(Hf));tr=atoms_for(data,Tf);ho=atoms_for(data,Hf);symbols=sorted(set(x for w in tr for x in w));Str=stats(tr,symbols);Sho=stats(ho,symbols);print('CENSUS',json.dumps({'symbols':symbols,'ns':len(symbols),'Tfolios':len(Tf),'Hfolios':len(Hf),'Cfolios':len(Cf),'Tunits':sum(map(len,tr)),'Hunits':sum(map(len,ho)),'Cunits':sum(map(len,atoms_for(data,Cf))),'Hcoverage':Sho['coverage']},separators=(',',':')),flush=True)
 if len(symbols)!=NS or Sho['coverage']<.99:
  out['verdict']='ATOMIC ALPHABET/COVERAGE MISMATCH';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
 fits={};vres=[]
 for la in b['LANGS']:
  s1,m1=optimize(Str,comps[la],('vms',la,1));s2,m2=optimize(Str,comps[la],('vms',la,2));m=m1 if s1>=s2 else m2;agr=agreement(Str['freq'],m1,m2);fw,n,cov=forward(ho,m,symbols,lms[la]);fits[la]=m;row={'lang':la,'Hscore':fw,'agreement':agr,'train_score':max(s1,s2),'map':{symbols[i]:b['VALUES'][int(m[i])] for i in range(NS)}};vres.append(row);print('VMS',json.dumps(row,separators=(',',':')),flush=True)
 rank=sorted(vres,key=lambda x:x['Hscore'],reverse=True);margin=rank[0]['Hscore']-rank[1]['Hscore'];primary=margin>=.05 and rank[0]['agreement']>=.90;cwords=atoms_for(data,Cf);cr=rank_fixed(cwords,fits,symbols,lms);out.update({'vms':vres,'Hranking':[(x['lang'],x['Hscore'],x['agreement']) for x in rank],'Hmargin':margin,'Hprimary':primary,'Cranking':cr,'verdict':'ATOMIC SIGNAL '+rank[0]['lang'] if primary else 'EVA REPRESENTATION SENSITIVE / NO STABLE ATOMIC SIGNAL'});print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':main()
