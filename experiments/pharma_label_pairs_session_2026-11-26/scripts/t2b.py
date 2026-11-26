import re, random, json, sys
from collections import defaultdict
sys.path.insert(0,'scripts'); from common import C, vords, joined
ZL='/home/claude/ext/voynich-scribe/ZL3b-n.txt'
PAGES=['f88r','f88v','f89r1','f89r2','f89v2','f89v1','f99r','f99v','f100r','f100v','f101v','f102r1','f102r2','f102v2','f102v1']
seq=defaultdict(list)   # page -> [(locus, type, text)]
for line in open(ZL,encoding='utf-8',errors='replace'):
    m=re.match(r'<(f[0-9]+[rv][0-9]?)\.([0-9]+),(.)([A-Za-z])[^>]*>\s*(.*)',line.rstrip('\n'))
    if m and m.group(1) in PAGES: seq[m.group(1)].append((int(m.group(2)),m.group(4),m.group(5)))
blocks=[]; nextblock={}   # (page,locus)->block index
for p in PAGES:
    runs=[]; cur=None
    for loc,t,txt in seq[p]:
        k='P' if t=='P' else 'L'
        if cur and cur[0]==k: cur[1].append((loc,txt))
        else: cur=[k,[(loc,txt)]]; runs.append(cur)
    for i,(k,items) in enumerate(runs):
        if k=='P':
            toks=[v for _,txt in items for v in vords(txt)]
            blocks.append(dict(page=p,toks=toks,set=set(toks),n=len(toks)))
    # map each L locus to the next P block
    bi=[b for b in range(len(blocks)) if blocks[b]['page']==p]; j=0; pblocks=iter(bi)
    pos=0; pidx=[x for x in bi]
    order=0
    for i,(k,items) in enumerate(runs):
        if k=='L':
            nb=None; c=0
            for kk,_ in runs[i+1:]:
                if kk=='P': nb=pidx[sum(1 for q in runs[:runs.index(runs[i])+1+c+0] if q[0]=='P')]; break
                c+=1
            for loc,_ in items: nextblock[(p,loc)]=nb
labs=[]
for c in C:
    if c['locus_type']!='Lf' or c['page'] not in PAGES: continue
    nb=nextblock.get((c['page'],int(c['zl_locus'])))
    if nb is None: continue
    forms={joined(c['label_zl'])}|{v for v in vords(c['label_zl']) if len(v)>=3} if len(vords(c['label_zl']))>1 else {joined(c['label_zl'])}
    forms={f for f in forms if f}
    labs.append(dict(page=c['page'],locus=c['zl_locus'],forms=forms,nb=nb))
def hit(l,b): return bool(l['forms'] & blocks[b]['set'])
H=sum(hit(l,l['nb']) for l in labs)
print('pages',len(PAGES),'blocks',len(blocks),'labels with a next block',len(labs),'H',H)
rng=random.Random(7)
same={p:[i for i,b in enumerate(blocks) if b['page']==p] for p in PAGES}
def draw_n1(l):
    c=[i for i in same[l['page']] if i!=l['nb']]; return rng.choice(c) if c else None
def cands_n2(l):
    n=blocks[l['nb']]['n']; return [i for i,b in enumerate(blocks) if b['page']!=l['page'] and abs(b['n']-n)<=0.3*n]
C2=[cands_n2(l) for l in labs]
def null(kind,labs_,nd=10000):
    v=[]
    for _ in range(nd):
        h=0
        for l,c2 in zip(labs_,C2):
            b=draw_n1(l) if kind==1 else (rng.choice(c2) if c2 else None)
            if b is not None: h+=hit(l,b)
        v.append(h)
    m=sum(v)/len(v); s=(sum((x-m)**2 for x in v)/len(v))**.5; return m,s,v
m1,s1,v1=null(1,labs); m2,s2,v2=null(2,labs)
res=dict(H=H,n_labels=len(labs),n_blocks=len(blocks),
  N1=dict(mean=round(m1,2),sd=round(s1,2),z=round((H-m1)/s1,2),p=sum(1 for x in v1 if x>=H)/len(v1),clear2=round(m1+2*s1,1)),
  N2=dict(mean=round(m2,2),sd=round(s2,2),z=round((H-m2)/s2,2),p=sum(1 for x in v2 if x>=H)/len(v2),clear2=round(m2+2*s2,1)),
  no_n1_block=sum(1 for l in labs if len(same[l['page']])<2), no_n2_block=sum(1 for c in C2 if not c))
print(json.dumps(res))
print('block sizes per page',{p:[blocks[i]['n'] for i in same[p]] for p in PAGES})
print('hits by page',{p:sum(hit(l,l['nb']) for l in labs if l['page']==p) for p in PAGES})
# power: plant fraction f of labels into their own next block
pw={}
for f in (0.05,0.10,0.15,0.25):
    det=0
    for run in range(100):
        r2=random.Random(500+run); planted=set(r2.sample(range(len(labs)),int(round(f*len(labs)))))
        Hp=sum(1 for i,l in enumerate(labs) if i in planted or hit(l,l['nb']))
        # null unchanged by planting except planted labels' own block; nulls use other blocks -> reuse
        det+= (Hp>=m1+2*s1) and (Hp>=m2+2*s2)
    pw[f]=det/100
print('power',pw)
res['power']=pw; json.dump(res,open('out/t2b_result.json','w'))
