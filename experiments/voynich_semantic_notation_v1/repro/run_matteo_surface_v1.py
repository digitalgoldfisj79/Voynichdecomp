# Exact Python body recovered from HF job 6a7bbf9527caad61c6eaca79.
# Reformatted from the job's `python -c` string into a .py file; algorithm unchanged.
import urllib.request,re,random,math,hashlib,json,statistics
from collections import Counter,defaultdict
URL='https://raw.githubusercontent.com/sjgallagher2/PyWORDS/master/pywords/data/lingualatina_voclist.txt'
raw=urllib.request.urlopen(URL,timeout=30).read(); text=raw.decode('utf-8')
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
sy=[fs(w) for w in words]; us=sorted(set(sy)); assert [fs(x) for x in ['tripode','pepo','corvus','vetula']]==['tri','pe','cor','ve']
def ent(vals):
 c=Counter(vals); n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values())
def quant(a,q):
 a=sorted(a); x=(len(a)-1)*q; lo=int(x); hi=min(lo+1,len(a)-1); f=x-lo; return a[lo]*(1-f)+a[hi]*f
def sm(toks):
 lens=[len(t) for t in toks]; chars=[c for t in toks for c in t]; byp=defaultdict(list); byr=defaultdict(list); bg=[]; bp=defaultdict(list)
 for t in toks:
  for i,c in enumerate(t): byp[i+1].append(c); byr[len(t)-i].append(c)
  for a,b in zip(t,t[1:]): bg.append((a,b)); bp[a].append(b)
 nc=sum(lens)
 return dict(types=len(toks),mean_len=sum(lens)/len(lens),median_len=quant(lens,.5),p10_len=quant(lens,.1),p90_len=quant(lens,.9),h_char=ent(chars),h_char_abs=sum(len(v)/nc*ent(v) for v in byp.values()),h_char_right=sum(len(v)/nc*ent(v) for v in byr.values()),h_first=ent([t[0] for t in toks]),h_last=ent([t[-1] for t in toks]),h_next_prev=sum(len(v)/len(bg)*ent(v) for v in bp.values()),bigram_types=len(set(bg)),alphabet=len(set(chars)))
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
 ds=[deg[t] for t in toks]; N=len(pairs)
 return dict(edit1_pairs=N,mean_degree=sum(ds)/len(ds),median_degree=quant(ds,.5),isolated_frac=sum(d==0 for d in ds)/len(ds),max_degree=max(ds),edit_prefix=loc['prefix']/N if N else 0,edit_internal=loc['internal']/N if N else 0,edit_suffix=loc['suffix']/N if N else 0)
def gen(mode,unif=False,N=7893):
 seed=20260812+(0 if mode=='mix' else int(mode)*1000)+(100000 if unif else 0); r=random.Random(seed); pool=us if unif else sy; out=set(); att=0
 while len(out)<N and att<5000000:
  k=r.choice([2,3,5]) if mode=='mix' else int(mode); out.add(''.join(r.choice(pool) for _ in range(k))); att+=1
 return sorted(out),att
rows=[]
for u in [False,True]:
 for m in ['2','3','5','mix']:
  t,a=gen(m,u); rows.append(dict(kind='matteo',sampling='syllable_uniform' if u else 'lemma_uniform',slots=m,attempts=a,**sm(t),**ep(t)))
k2=[x for x in rows if x['sampling']=='lemma_uniform' and x['slots']=='2'][0]; alph=sorted(set(c for w in words for c in w)); cc=Counter(c for w in words for c in w)
def iid(emp):
 r=random.Random(20260812+(1 if emp else 0)); out=set(); lo=int(k2['mean_len']); hi=lo+1; p=k2['mean_len']-lo
 while len(out)<7893:
  n=hi if r.random()<p else lo; out.add(''.join(r.choices(alph,weights=[cc[c] for c in alph],k=n) if emp else r.choices(alph,k=n)))
 return sorted(out)
for emp in [False,True]:
 t=iid(emp); rows.append(dict(kind='iid',sampling='source_char' if emp else 'uniform_char',slots='K2mean',**sm(t),**ep(t)))
print(json.dumps({'source_sha256':hashlib.sha256(raw).hexdigest(),'source_lines':len(text.splitlines()),'eligible_unique_words':len(words),'unique_first_syllables':len(us),'rows':rows},sort_keys=True))