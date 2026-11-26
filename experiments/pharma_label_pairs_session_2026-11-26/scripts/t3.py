import pickle, random, math, json, sys
from collections import Counter, defaultdict
sys.path.insert(0,'scripts'); from common import C, joined
from scipy.stats import kruskal
R=pickle.load(open('ckpt/s1_final.pkl','rb'))
parts=defaultdict(set)
for r in R:
    f=r['fragment_final']
    if f and '/' not in f and '|' not in f: parts[f].add(r['obj_class'])
units=[]; seen=set()
for c in C:
    f=c['fragment_num']
    if c['locus_type']!='Lf' or not f or f in seen or f not in parts: continue
    p=parts[f]; cat='ROOT' if p=={'root'} else 'LEAF' if p=={'leaf'} else 'BOTH' if p=={'root','leaf'} else None
    if not cat: continue
    w=joined(c['label_zl'])
    if not w: continue
    seen.add(f); units.append(dict(f=f,page=c['page'],cat=cat,first=w[0],last=w[-1],L=len(w)))
print('units',len(units),Counter(u['cat'] for u in units))
def pool(key):
    cnt=Counter(u[key] for u in units); return [u[key] if cnt[u[key]]>=5 else 'other' for u in units]
def G(cats,feat):
    tab=Counter(zip(cats,feat)); rc=Counter(cats); cc=Counter(feat); n=len(cats); g=0
    for (a,b),o in tab.items(): g+=2*o*math.log(o*n/(rc[a]*cc[b]))
    return g
def KW(cats,L):
    grp=defaultdict(list)
    for c,l in zip(cats,L): grp[c].append(l)
    return kruskal(*grp.values()).statistic
F1=pool('first'); F2=pool('last'); LL=[u['L'] for u in units]; cats=[u['cat'] for u in units]; pages=[u['page'] for u in units]
idx=defaultdict(list)
for i,p in enumerate(pages): idx[p].append(i)
def perm(c,rng):
    c=list(c)
    for p,ii in idx.items():
        v=[c[i] for i in ii]; rng.shuffle(v)
        for i,x in zip(ii,v): c[i]=x
    return c
def test(cats,F1,F2,LL,nperm,rng):
    obs=[G(cats,F1),G(cats,F2),KW(cats,LL)]; nulls=[[],[],[]]
    for _ in range(nperm):
        q=perm(cats,rng); nulls[0].append(G(q,F1)); nulls[1].append(G(q,F2)); nulls[2].append(KW(q,LL))
    out=[]
    for o,nl in zip(obs,nulls):
        m=sum(nl)/len(nl); s=(sum((x-m)**2 for x in nl)/len(nl))**.5
        out.append(dict(obs=round(o,3),null_mean=round(m,3),null_sd=round(s,3),z=round((o-m)/s,2),p=sum(1 for x in nl if x>=o)/len(nl)))
    return out
rng=random.Random(3)
res=test(cats,F1,F2,LL,10000,rng)
for name,r in zip(['first glyph G','last glyph G','length KW H'],res): print(name,r)
# descriptive table
for key,F in (('first',F1),('last',F2)):
    t=Counter(zip(cats,F)); levels=sorted(set(F))
    print(key,'levels',levels)
    for c in ('ROOT','LEAF','BOTH'): print(' ',c,[t[(c,l)] for l in levels])
for c in ('ROOT','LEAF','BOTH'):
    v=[u['L'] for u in units if u['cat']==c]; print('len',c,len(v),round(sum(v)/len(v),2))
json.dump(dict(units=units,res=res),open('out/t3_result.json','w'))
# power
pw={}
for f in (0.1,0.2,0.3,0.5):
    det=0
    for run in range(200):
        r2=random.Random(1000+run); F2p=list(F2)
        roots=[i for i,c in enumerate(cats) if c=='ROOT']
        for i in r2.sample(roots,int(round(f*len(roots)))): F2p[i]='m'
        o=G(cats,F2p); nl=[G(perm(cats,r2),F2p) for _ in range(1000)]
        m=sum(nl)/len(nl); s=(sum((x-m)**2 for x in nl)/len(nl))**.5
        det+= ((o-m)/s>=2) and (sum(1 for x in nl if x>=o)/1000<0.0167)
    pw[f]=det/200; print('power f',f,pw[f])
json.dump(dict(units=units,res=res,power=pw),open('out/t3_result.json','w'))
