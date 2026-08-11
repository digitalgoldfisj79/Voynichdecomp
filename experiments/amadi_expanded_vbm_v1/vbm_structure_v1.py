# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import collections,hashlib,json,math,re,sys,urllib.request
import numpy as np
from datasets import load_dataset
from unidecode import unidecode
sys.path.insert(0,'experiments/amadi_residuals_v1')
import amadi_residuals_v1 as ar
ar.HEADERS={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36','Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8','Accept-Language':'en-GB,en;q=0.9','Referer':'https://www.voynich.nu/transcr.html'}
NS='VBMV1'; PLAIN='abcdefghilmnopqrstu'; V=set('aeiou')
H1=['f28v','f31v','f88r','f5r','f34r','f81v']; C1=['f85r1','f53v','f33r','f10r','f23r','f111r']
URLS={
'german':{'train':'https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu','ctrl':['https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-dev.conllu','https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-test.conllu']},
'italian':{'train':'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu','ctrl':['https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-dev.conllu','https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-test.conllu']}}

def norm(s):
 s=unidecode(s).lower().replace('j','i').replace('v','u').replace('w','u').replace('y','i').replace('x','s').replace('z','s')
 return ''.join(c for c in s if c in PLAIN)
def cv(s):return ''.join('V' if c in V else 'C' for c in norm(s))
def parse_conllu(b):
 sents=[];cur=[]
 for ln in b.decode('utf-8','replace').splitlines():
  if not ln:
   if cur:sents.append(''.join(cur));cur=[]
   continue
  if ln.startswith('#'):continue
  c=ln.split('\t')
  if len(c)>=2 and c[0].isdigit():
   z=norm(c[1])
   if z:cur.append(z)
 if cur:sents.append(''.join(cur))
 return sents

def get(url):return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=120).read()

def corpora():
 out={}
 for la in ['german','italian']:
  tr=parse_conllu(get(URLS[la]['train']));ct=[]
  for u in URLS[la]['ctrl']:ct+=parse_conllu(get(u))
  out[la]=(tr,ct)
 ds=load_dataset('bavarian-nlp/barwiki-20250720',split='train')
 tr=[];ct=[];tc=cc=0
 for row in ds:
  rid=int(row['id']) if str(row['id']).isdigit() else int.from_bytes(hashlib.sha256(str(row['id']).encode()).digest()[:4],'big')
  dest=tr if rid%10<6 else ct
  for z in re.split(r'[.!?\n]+',row.get('text','')):
   q=norm(z)
   if len(q)>=20:
    dest.append(q)
    if dest is tr:tc+=len(q)
    else:cc+=len(q)
  if tc>=1800000 and cc>=700000:break
 out['bavarian']=(tr,ct)
 return out

def lm_train(seqs):
 # order 4 incl boundary B=2; predict symbol with previous 3
 C=np.full((3,3,3,3),0.25,dtype=float)
 for s in seqs:
  q=[2,2,2]+[1 if x=='V' else 0 for x in cv(s)]+[2]
  for a,b,c,d in zip(q,q[1:],q[2:],q[3:]):C[a,b,c,d]+=1
 C/=C.sum(axis=3,keepdims=True)
 return np.log(C)
def score_cv(seq,logp):
 q=[2,2,2]+[1 if x=='V' else 0 for x in seq]+[2];z=0.;n=0
 for a,b,c,d in zip(q,q[1:],q[2:],q[3:]):z+=logp[a,b,c,d];n+=1
 return z/max(1,n)
def span_controls(seqs,N,tag,n=16):
 flat=''.join(cv(x) for x in seqs if x)
 if len(flat)<N:raise RuntimeError((tag,'control too short',len(flat),N))
 out=[]
 for r in range(n):
  st=int.from_bytes(hashlib.sha256(f'{NS}::{tag}::{r}'.encode()).digest()[:8],'big')%(len(flat)-N+1);out.append(flat[st:st+N])
 return out

def raw_lines():
 b=ar.getb(ar.RF_URL,ar.HEADERS);assert hashlib.sha256(b).hexdigest()==ar.RF_SHA
 out=collections.defaultdict(list)
 for line in b.decode('utf-8','replace').splitlines():
  if not line.startswith('<') or '>' not in line:continue
  lab,rhs=line.split('>',1)
  if '.' not in lab or '<!' in rhs:continue
  pg=lab[1:].split('.',1)[0];rhs=re.sub(r'<(?:-|~)>','.',rhs);rhs=re.sub(r'<[^>]*>','.',rhs);rhs=rhs.replace(',','')
  seg=[]
  def flush():
   nonlocal seg
   if len(seg)>=2:out[pg].append(seg)
   seg=[]
  for rw in rhs.split('.'):
   rw=rw.strip()
   if not rw:flush();continue
   if '[' in rw or ']' in rw or '?' in rw:flush();continue
   ch=''.join(c for c in rw.replace('{','').replace('}','').lower() if 'a'<=c<='z')
   if not ch or any(c not in ar.S2I for c in ch) or len(ch)<2:flush();continue
   seg.append(ch)
  flush()
 return dict(out)
def vr(w):return w[:2] if w.startswith('qo') and len(w)>=3 else w[:1]
def core_units(w):
 a=len(vr(w));b=len(w)-1
 if b<=a:return []
 s=w[a:b];out=[]
 # reserve attested C2 suffixes
 tail=None
 if s.endswith('eed') and len(s)>=3:s=s[:-3];tail='eed'
 elif s.endswith('ed') and len(s)>=2:s=s[:-2];tail='ed'
 i=0
 while i<len(s):
  if s[i]=='e':
   j=i+1
   while j<len(s) and s[j]=='e':j+=1
   out.append('e+');i=j
  else:out.append(s[i]);i+=1
 if tail:out.append(tail)
 return out
def vbm_types(lines,folios):
 seqs=[];cores=bridges=0
 for f in folios:
  for words in lines.get(f,[]):
   q=[]
   for i,w in enumerate(words):
    cu=core_units(w);q+=['C']*len(cu);cores+=len(cu)
    if i+1<len(words):q.append('V');bridges+=1
   if q:seqs.append(''.join(q))
 return seqs,{'segments':len(seqs),'events':cores+bridges,'core_events':cores,'bridge_events':bridges,'vowel_event_fraction':bridges/max(1,cores+bridges)}
def main():
 pages,_=ar.parse_rf();T,H,prior,H2,C2=ar.target_split(pages);FIT=T+H
 lines=raw_lines();fseq,fm=vbm_types(lines,FIT);hseq,hm=vbm_types(lines,H1);N=hm['events'];target=''.join(hseq)
 cs=corpora();lms={};rows={};rank=[]
 for la,(tr,ct) in cs.items():
  lp=lm_train(tr);lms[la]=lp;c=span_controls(ct,N,la,16);scores=[score_cv(x,lp) for x in c];floor=float(np.quantile(scores,.05,method='linear'));ts=score_cv(target,lp);rows[la]={'control_scores':scores,'floor_p05':floor,'H1_score':ts,'gap':ts-floor,'pass':bool(ts>=floor),'train_chars':sum(len(norm(x)) for x in tr),'control_chars':sum(len(norm(x)) for x in ct)};rank.append((la,ts))
 rank.sort(key=lambda x:x[1],reverse=True);margin=rank[0][1]-rank[1][1];bav=rows['bavarian'];candidate=bool(bav['pass'] and rank[0][0]=='bavarian' and margin>=.02);anypass=any(x['pass'] for x in rows.values())
 out={'protocol':'VBM_PROTOCOL_V1.md','FIT':fm,'H1':hm,'VBM_H1':H1,'VBM_C1':C1,'C1_opened':False,'languages':rows,'ranking':rank,'top_margin':margin,'bavarian_candidate':candidate,'S0_pass_any_language':anypass,'verdict':'S0_PASS_TO_TYPED_SUBSTITUTION' if anypass else 'VBM_V1_TOPOLOGY_INCOMPATIBLE'}
 print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
