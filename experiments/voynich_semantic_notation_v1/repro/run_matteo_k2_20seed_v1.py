# Exact Python body recovered from HF job 6a7bbfc8f6d0f3ee953aa36a.
# Reformatted from `python -c`; algorithm unchanged.
import urllib.request,re,random,math,json,statistics
from collections import Counter,defaultdict
raw=urllib.request.urlopen('https://raw.githubusercontent.com/sjgallagher2/PyWORDS/master/pywords/data/lingualatina_voclist.txt',timeout=30).read(); text=raw.decode(); V=set('aeiouy');D={'ae','au','oe','ei','eu','ui'};M=set('bcdgptfk');L=set('lr')
def fs(w):
 w=re.sub('[^a-z]','',w.lower().replace('j','i'));ns=[];i=0
 while i<len(w):
  if w[i] in V:
   if i+1<len(w) and w[i:i+2] in D:ns.append((i,i+2));i+=2
   else:ns.append((i,i+1));i+=1
  else:i+=1
 if len(ns)<2:return w
 e=ns[0][1];s=ns[1][0];cl=w[e:s]
 if len(cl)<=1:return w[:e]
 return w[:s-2 if cl[-2] in M and cl[-1] in L else s-1]
words=sorted(set(w.strip().lower() for w in text.splitlines() if re.fullmatch('[A-Za-z]+',w.strip()) and len(w.strip())>=2 and any(c in V for c in w.strip().lower())));pool=[fs(w) for w in words]
def q(a,p):
 a=sorted(a);x=(len(a)-1)*p;lo=int(x);hi=min(lo+1,len(a)-1);f=x-lo;return a[lo]*(1-f)+a[hi]*f
def one(seed):
 r=random.Random(seed);S=set();att=0
 while len(S)<7893:
  S.add(r.choice(pool)+r.choice(pool));att+=1
 toks=sorted(S); pairs=set()
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
 deg=Counter();loc=Counter()
 for a,b in pairs:
  deg[a]+=1;deg[b]+=1
  if len(a)==len(b):
   k=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y);p='prefix' if k==0 else ('suffix' if k==len(a)-1 else 'internal')
  else:
   long,short=(a,b) if len(a)>len(b) else (b,a);poss=[i for i in range(len(long)) if long[:i]+long[i+1:]==short];pcs=[('prefix' if i==0 else ('suffix' if i==len(long)-1 else 'internal')) for i in poss];p=pcs[0] if pcs and all(x==pcs[0] for x in pcs) else 'internal'
  loc[p]+=1
 ds=[deg[t] for t in toks];N=len(pairs)
 return {'seed':seed,'attempts':att,'pairs':N,'mean_degree':sum(ds)/len(ds),'isolated':sum(d==0 for d in ds)/len(ds),'prefix':loc['prefix']/N,'internal':loc['internal']/N,'suffix':loc['suffix']/N,'mean_len':sum(map(len,toks))/len(toks)}
R=[one(20260812+i) for i in range(20)]
keys=['pairs','mean_degree','isolated','prefix','internal','suffix','mean_len']
out={'n':20,'rows':R,'summary':{k:{'mean':statistics.mean(x[k] for x in R),'sd':statistics.stdev(x[k] for x in R),'min':min(x[k] for x in R),'max':max(x[k] for x in R)} for k in keys}}
print(json.dumps(out,sort_keys=True))