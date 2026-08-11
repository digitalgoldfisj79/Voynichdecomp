# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import hashlib,json,re
from unidecode import unidecode
from datasets import load_dataset
V=set('aeiou'); ALPH=set('abcdefghilmnopqrstuz'); DIP={'ae','oe','au','eu','ei'}
TARGET={'sta':130655,'aaa':145115}

def norm(w):
 s=unidecode(w).lower().replace('j','i').replace('v','u').replace('w','u').replace('y','i').replace('x','s')
 return ''.join(c for c in s if c in ALPH)
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
def main():
 ds=load_dataset('bavarian-nlp/barwiki-20250720',split='train')
 stream=[];words=segwords=0
 # deterministic article order, stop after enough events for 8 matched windows plus reserve
 for row in ds:
  for w in re.findall(r"[A-Za-zÀ-ÿ]+",row.get('text','')):
   s=norm(w)
   if not s:continue
   words+=1;q=syl(w)
   if q:
    segwords+=1;stream.extend(q)
  if len(stream)>=900000:break
 rows={}
 for rep,N in TARGET.items():
  z=[];L=len(stream)
  for r in range(8):
   st=int.from_bytes(hashlib.sha256(f'BAR-EXP::{rep}::{r}'.encode()).digest()[:8],'big')%(L-N+1)
   q=stream[st:st+N];k=len(set(q));z.append({'window':r,'events':N,'active_syllables':k,'capacity_1365_pass':k<=1365,'surface_sta158_pass':k<=158,'surface_aaa129_pass':k<=129})
  rows[rep]=z
 out={'dataset':'bavarian-nlp/barwiki-20250720','stream_events':len(stream),'words':words,'segmentable_words':segwords,'word_coverage':segwords/max(1,words),'overall_distinct_syllables':len(set(stream)),'rows':rows,'expanded_capacity_pass_all':all(x['capacity_1365_pass'] for zz in rows.values() for x in zz)}
 print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
