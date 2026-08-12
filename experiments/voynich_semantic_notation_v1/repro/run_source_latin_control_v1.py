# Exact Python body recovered from HF job 6a7bbfedf6d0f3ee953aa372.
# Reformatted from `python -c`; algorithm unchanged.
import urllib.request,re,json
from collections import Counter,defaultdict
text=urllib.request.urlopen('https://raw.githubusercontent.com/sjgallagher2/PyWORDS/master/pywords/data/lingualatina_voclist.txt',timeout=30).read().decode();V=set('aeiouy');W=sorted(set(w.strip().lower() for w in text.splitlines() if re.fullmatch('[A-Za-z]+',w.strip()) and len(w.strip())>=2 and any(c in V for c in w.strip().lower())));S=set(W);pairs=set()
for w in W:
 for i in range(len(w)):
  d=w[:i]+w[i+1:]
  if d in S:pairs.add(tuple(sorted((w,d))))
B=defaultdict(list)
for w in W:
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
N=len(pairs);ds=[deg[w] for w in W]; print(json.dumps({'types':len(W),'pairs':N,'mean_degree':sum(ds)/len(ds),'isolated':sum(d==0 for d in ds)/len(ds),'prefix':loc['prefix']/N,'internal':loc['internal']/N,'suffix':loc['suffix']/N,'mean_len':sum(map(len,W))/len(W)}))