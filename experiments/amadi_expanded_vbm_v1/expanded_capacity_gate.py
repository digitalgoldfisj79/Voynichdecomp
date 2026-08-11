# /// script
# requires-python = ">=3.11"
# dependencies = ["Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import collections,hashlib,json,urllib.request
from unidecode import unidecode
URL='https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu'
V=set('aeiou');ALPH=set('abcdefghilmnopqrstuz');DIP={'ae','oe','au','eu','ei'}
TARGET_EVENTS={'sta':130655,'aaa':145115};TARGET_K={'sta':158,'aaa':129}

def norm(w):
 s=unidecode(w).lower().replace('j','i').replace('v','u').replace('w','u').replace('y','i').replace('x','s');return ''.join(c for c in s if c in ALPH)
def valid(s):
 if not s:return False
 nv=sum(c in V for c in s)
 if nv==2 and s in DIP:return True
 if nv!=1:return False
 j=next(i for i,c in enumerate(s) if c in V);a=j;b=len(s)-j-1
 return a<=3 and b<=3 and a+b<=5
def syl(w):
 s=norm(w);n=len(s);best=[None]*(n+1);best[n]=()
 for i in range(n-1,-1,-1):
  q=[]
  for j in range(i+1,min(n,i+7)+1):
   u=s[i:j]
   if valid(u) and best[j] is not None:q.append((u,)+best[j])
  if q:best[i]=min(q,key=lambda z:(len(z),tuple(-len(x) for x in z),z))
 return list(best[0]) if s and best[0] is not None else []
def sentences(b):
 out=[];cur=[]
 for ln in b.decode('utf-8','replace').splitlines():
  if not ln:
   if cur:out.append(cur);cur=[]
   continue
  if ln.startswith('#'):continue
  c=ln.split('\t')
  if len(c)>=2 and c[0].isdigit():cur.append(c[1])
 if cur:out.append(cur)
 return out
def flatten(S,idxs,inv=None):
 out=[];words=kept=0
 for i in idxs:
  for w in S[i]:
   q=syl(w);words+=1
   if q and (inv is None or all(x in inv for x in q)):out.extend(q);kept+=1
 return out,kept,words
def main():
 b=urllib.request.urlopen(URL,timeout=120).read();S=sentences(b);train=[i for i in range(len(S)) if i%10<6];ctrl=[i for i in range(len(S)) if i%10>=6]
 tr,_,_=flatten(S,train);C=collections.Counter(tr);ordered=sorted(C,key=lambda x:(-C[x],x));inventory=set(ordered[:1365]);ct,kw,nw=flatten(S,ctrl,inventory)
 rows={}
 for rep,N in TARGET_EVENTS.items():
  vals=[]
  L=len(ct)
  for r in range(8):
   if L<=N:q=ct
   else:
    st=int.from_bytes(hashlib.sha256(f'EXP1365::{rep}::{r}'.encode()).digest()[:8],'big')%(L-N+1);q=ct[st:st+N]
   vals.append({'rep':rep,'window':r,'events':len(q),'active_syllables':len(set(q)),'surface_K':TARGET_K[rep],'capacity_pass':len(set(q))<=TARGET_K[rep]})
  rows[rep]=vals
 out={'inventory_size':len(inventory),'train_distinct_before_cap':len(C),'control_events_retained':len(ct),'control_word_coverage':kw/max(1,nw),'rows':rows,'gate_pass':all(x['capacity_pass'] for z in rows.values() for x in z),'source_sha256':hashlib.sha256(b).hexdigest()}
 print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
