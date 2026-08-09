#!/usr/bin/env python3
import urllib.request, tarfile, os, json, hashlib, re, math
from collections import Counter, defaultdict
import numpy as np
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

PARENT='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/c7c50f74e1f1f88004a0f08ea379324a3d42c16d/experiments/bnf_m19_german_confirm_v1_0/run_confirm.py'
src=urllib.request.urlopen(PARENT,timeout=90).read().decode();src=src.rsplit("if __name__=='__main__':main()",1)[0]
lib={'__name__':'parent'};exec(compile(src,'run_confirm.py','exec'),lib)
b=lib['b'];inner=lib['inner'];RAW=lib['RAW']
REF_TABLE='https://www.linguistics.rub.de/ref/corpus/texts.html'
REF_ARCH='https://zenodo.org/api/records/5793616/files/ReF-v1.0.2.tar.gz/content'
WINDOW={'14,2','15,1','15,2'}; ORDERS=[2,3,4,5]; ALPHA=.1; TRAIN_CAP=500000; REPS=3; CTRL_N=25000

def seed(*x):return int.from_bytes(hashlib.sha256(('20260809|BAVNUM|'+'|'.join(map(str,x))).encode()).digest()[:8],'big')&0xffffffff

def metadata():
 s=BeautifulSoup(urllib.request.urlopen(REF_TABLE,timeout=90).read(),'html.parser');t=s.find('table');heads=[x.get_text(' ',strip=True) for x in t.find_all('th')];out={}
 for r in t.find_all('tr')[1:]:
  cells=[x.get_text(' ',strip=True) for x in r.find_all(['td','th'])]
  if len(cells)>=9:
   row=dict(zip(heads,cells));fid=row.get('ID','').strip()
   if fid:out[fid]=row
 return out

def norm_token(x):
 ws=b['norm_words'](x);return ''.join(ws) if ws else ''

def parse(blob):
 try:root=ET.fromstring(blob)
 except:return []
 out=[]
 for e in root.iter():
  if e.tag.split('}')[-1]=='token':
   z=norm_token(e.attrib.get('trans') or '')
   if z:out.append(z)
 return out

def load_docs(meta):
 p='/tmp/ReF-v1.0.2.tar.gz'
 if not os.path.exists(p):urllib.request.urlretrieve(REF_ARCH,p)
 tf=tarfile.open(p,'r:gz');out={}
 for m in tf.getmembers():
  if not m.isfile() or not m.name.endswith('.xml'):continue
  fid=os.path.basename(m.name)[:-4]
  if fid not in meta or meta[fid].get('Datierung') not in WINDOW:continue
  f=tf.extractfile(m);ws=parse(f.read()) if f else []
  if len(ws)>len(out.get(fid,[])):out[fid]=ws
 return out

def cls(meta,fid):return 'bavarian' if 'bairisch' in meta[fid].get('Dialekt','').lower() else 'nonbavarian'

def split_ids(ids,label):
 a=sorted(ids,key=lambda f:seed('split',label,f));n=len(a);ntr=max(1,int(round(.6*n)));nd=max(1,int(round(.2*n)))
 if ntr+nd>=n:ntr=max(1,n-2);nd=1
 return a[:ntr],a[ntr:ntr+nd],a[ntr+nd:]

def cap_words(docs,ids,cap):
 ws=[];n=0
 for fid in sorted(ids,key=lambda f:seed('trainorder',f)):
  for w in docs[fid]:
   if n+len(w)>cap:return ws
   ws.append(w);n+=len(w)
 return ws

def choose_words(docs,ids,tag,nletters):
 pool=[w for fid in sorted(ids) for w in docs[fid] if w]
 if not pool:return []
 start=seed('span',tag)%len(pool);out=[];n=0;i=start
 while n<nletters and len(out)<len(pool)*2:
  w=pool[i%len(pool)];out.append(w);n+=len(w);i+=1
 return out

def emit_word(w,rng):
 out=[]
 for c in w:
  if c not in b['A2I']:continue
  vals=b['LETTER_VALS'][b['A2I'][c]];out.append(int(vals[int(rng.integers(0,5))]))
 return out

def numeric_words(words,tag):
 rng=np.random.default_rng(seed('emit',tag));return [z for w in words if (z:=emit_word(w,rng))]

def train_ngram(words,order):
 ng=Counter();ctx=Counter();vocab=set(b['VALUES']);END=-2;START=-1
 for w in words:
  seq=[START]*(order-1)+w+[END]
  for i in range(order-1,len(seq)):
   c=tuple(seq[i-order+1:i]);x=seq[i];ctx[c]+=1;ng[c+(x,)]+=1
 return {'order':order,'ng':ng,'ctx':ctx,'v':len(vocab)+1}

def score(model,words):
 o=model['order'];ng=model['ng'];ctx=model['ctx'];V=model['v'];START=-1;END=-2;s=0.0;n=0
 for w in words:
  seq=[START]*(o-1)+w+[END]
  for i in range(o-1,len(seq)):
   c=tuple(seq[i-o+1:i]);x=seq[i];s+=math.log((ng[c+(x,)]+ALPHA)/(ctx[c]+ALPHA*V));n+=1
 return s/max(1,n),n

def classify(models,words):
 r=sorted([(k,score(m,words)[0]) for k,m in models.items()],key=lambda x:x[1],reverse=True);return r

def build_models(docs,splits,order):
 # Equal plaintext budget before 3 stochastic channel replications.
 avail={c:sum(sum(map(len,docs[f])) for f in splits[c]['train']) for c in splits};budget=min(TRAIN_CAP,*avail.values());out={}
 for c in ['bavarian','nonbavarian']:
  plain=cap_words(docs,splits[c]['train'],budget);num=[]
  for rep in range(REPS):num.extend(numeric_words(plain,('train',c,order,rep)))
  out[c]=train_ngram(num,order)
 return out,budget

def control_eval(docs,splits,models,part,order):
 rows=[]
 for c in ['bavarian','nonbavarian']:
  ids=splits[c][part]
  for rep in range(20):
   plain=choose_words(docs,ids,(part,c,order,rep),CTRL_N);num=numeric_words(plain,(part,c,order,rep));r=classify(models,num);true=next(x[1] for x in r if x[0]==c);other=next(x[1] for x in r if x[0]!=c);rows.append({'true':c,'rep':rep,'pred':r[0][0],'margin':true-other,'ranking':r,'letters':sum(map(len,num))})
 return rows

def summarize(rows):
 by={c:[r for r in rows if r['true']==c] for c in ['bavarian','nonbavarian']};acc={c:sum(r['pred']==c for r in x)/len(x) for c,x in by.items()};bal=sum(acc.values())/2;return {'balanced_accuracy':bal,'accuracy':acc,'median_true_margin':float(np.median([r['margin'] for r in rows]))}

def vms_num_words(data,folios,tid='ZLZI'):
 out=[]
 for f in folios:
  for w in lib['words_for'](data,[f],tid):
   z=[RAW[c] for c in w if c in RAW]
   if z:out.append(z)
 return out

def main():
 meta=metadata();docs=load_docs(meta);ids={'bavarian':[f for f in docs if cls(meta,f)=='bavarian'],'nonbavarian':[f for f in docs if cls(meta,f)=='nonbavarian']};splits={}
 for c,fs in ids.items():
  tr,dv,cf=split_ids(fs,c);splits[c]={'train':tr,'dev':dv,'confirm':cf}
 print('SPLITS='+json.dumps({c:{k:v for k,v in s.items()} for c,s in splits.items()},separators=(',',':')),flush=True)
 devres={};models_by={}
 for o in ORDERS:
  models,budget=build_models(docs,splits,o);models_by[o]=models;rows=control_eval(docs,splits,models,'dev',o);su=summarize(rows);devres[o]={'budget':budget,'summary':su};print('DEV',o,json.dumps(su,separators=(',',':')),flush=True)
 bestacc=max(devres[o]['summary']['balanced_accuracy'] for o in ORDERS);best=min(o for o in ORDERS if devres[o]['summary']['balanced_accuracy']==bestacc);print('SELECTED',best,bestacc,flush=True)
 if bestacc<.85:print('RESULT_JSON='+json.dumps({'splits':splits,'dev':devres,'selected':best,'verdict':'UNDERPOWERED DEVELOPMENT'},separators=(',',':')),flush=True);return
 models=models_by[best];conf=control_eval(docs,splits,models,'confirm',best);cs=summarize(conf);gate=cs['balanced_accuracy']>=.85 and min(cs['accuracy'].values())>=.80 and cs['median_true_margin']>0;print('CONFIRM='+json.dumps({'summary':cs,'rows':conf},separators=(',',':')),flush=True)
 if not gate:print('RESULT_JSON='+json.dumps({'splits':splits,'dev':devres,'selected':best,'confirm':cs,'verdict':'UNDERPOWERED CONFIRMATION'},separators=(',',':')),flush=True);return
 data=json.loads(b['fetch'](b['SLIM']));sample,hold,pages,_=inner['split_vms'](data);T={f for f,_,_ in sample};H={f for f,_,_ in hold};A={f for f,_,_ in pages};C=sorted(A-T-H);vw=vms_num_words(data,C);vr=classify(models,vw);vm=next(x[1] for x in vr if x[0]=='bavarian')-next(x[1] for x in vr if x[0]=='nonbavarian')
 # Fixed four buckets as deterministic SHA modulo 4; explanatory only.
 buckets=[]
 for j in range(4):
  fs=[f for f in C if seed('bucket',f)%4==j];ww=vms_num_words(data,fs);rr=classify(models,ww);m=next(x[1] for x in rr if x[0]=='bavarian')-next(x[1] for x in rr if x[0]=='nonbavarian');buckets.append({'bucket':j,'folios':len(fs),'letters':sum(map(len,ww)),'ranking':rr,'bavarian_margin':m})
 out={'splits':splits,'dev':devres,'selected':best,'confirm':cs,'gate':gate,'vms_ranking':vr,'vms_bavarian_margin':vm,'buckets':buckets,'verdict':'BAVARIAN MACRO SIGNAL' if vm>0 else 'NON-BAVARIAN MACRO SIGNAL'};print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':main()
