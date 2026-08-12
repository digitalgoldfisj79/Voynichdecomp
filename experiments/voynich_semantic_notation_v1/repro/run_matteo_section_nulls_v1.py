# Exact Python body recovered from HF job 6a7bc266f6d0f3ee953aa37b.
# Reformatted from `python -c`; algorithm unchanged.
import urllib.request,re,random,json,statistics
from collections import Counter,defaultdict
URL='https://raw.githubusercontent.com/sjgallagher2/PyWORDS/master/pywords/data/lingualatina_voclist.txt'
raw=urllib.request.urlopen(URL,timeout=30).read(); text=raw.decode()
V=set('aeiouy'); D={'ae','au','oe','ei','eu','ui'}; M=set('bcdgptfk'); L=set('lr')
def fs(w):
 w=re.sub('[^a-z]','',w.lower().replace('j','i')); ns=[]; i=0
 while i<len(w):
  if w[i] in V:
   if i+1<len(w) and w[i:i+2] in D: ns.append((i,i+2)); i+=2
   else: ns.append((i,i+1)); i+=1
  else:i+=1
 if not ns:return w
 if len(ns)==1:return w
 e=ns[0][1]; s=ns[1][0]; cl=w[e:s]
 if len(cl)<=1:return w[:e]
 cut=s-2 if cl[-2] in M and cl[-1] in L else s-1
 return w[:cut]
words=sorted(set(w.strip().lower() for w in text.splitlines() if re.fullmatch('[A-Za-z]+',w.strip()) and len(w.strip())>=2 and any(c in V for c in w.strip().lower())))
sy=[fs(w) for w in words]
def ep(toks):
 S=set(toks); pairs=set()
 for w in toks:
  for i in range(len(w)):
   d=w[:i]+w[i+1:]
   if d in S:pairs.add(tuple(sorted((w,d))))
 B=defaultdict(list)
 for w in toks:
  for i in range(len(w)):B[(len(w),i,w[:i],w[i+1:])].append(w)
 for ws in B.values():
  if len(ws)>1:
   ws=sorted(set(ws))
   for i in range(len(ws)):
    for j in range(i+1,len(ws)):pairs.add((ws[i],ws[j]))
 deg=Counter(); loc=Counter()
 for a,b in pairs:
  deg[a]+=1;deg[b]+=1
  if len(a)==len(b):
   k=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y); n=len(a); p='prefix' if k==0 else ('suffix' if k==n-1 else 'internal')
  else:
   long,short=(a,b) if len(a)>len(b) else (b,a); poss=[i for i in range(len(long)) if long[:i]+long[i+1:]==short]; pcs=[('prefix' if i==0 else ('suffix' if i==len(long)-1 else 'internal')) for i in poss]; p=pcs[0] if pcs and all(x==pcs[0] for x in pcs) else 'internal'
  loc[p]+=1
 N=len(pairs); ds=[deg[t] for t in toks]
 return {'pairs':N,'mean_degree':2*N/len(toks),'isolated':sum(d==0 for d in ds)/len(ds),'prefix':loc['prefix']/N if N else 0,'internal':loc['internal']/N if N else 0,'suffix':loc['suffix']/N if N else 0}
def gen(n,seed):
 r=random.Random(seed); out=set()
 while len(out)<n: out.add(r.choice(sy)+r.choice(sy))
 return sorted(out)
sections={'Stars':3121,'Herbal-A':2812,'missing':2494,'Balneological':1406,'text-only':907,'Pharmaceutical':566,'Herbal-B':437,'Cosmological':249,'Zodiac':223}
rows=[]
for si,(sec,n) in enumerate(sections.items()):
 vals=[]
 for j in range(20):
  t=gen(n,20260812+si*1000+j); d=ep(t); d['mean_len']=sum(map(len,t))/n; vals.append(d)
 for k in ['pairs','mean_degree','isolated','prefix','internal','suffix','mean_len']:
  xs=[v[k] for v in vals]
  rows.append({'section':sec,'n':n,'metric':k,'mean':statistics.mean(xs),'sd':statistics.pstdev(xs),'min':min(xs),'max':max(xs)})
print(json.dumps(rows,sort_keys=True))