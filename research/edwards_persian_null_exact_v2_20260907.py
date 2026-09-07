import re, urllib.request, unicodedata, collections, json, random
from datasets import load_dataset

# Exact GC/v101 source Edwards cites: IVTFF 0c, 10 Apr 2020.
V101_URL='https://raw.githubusercontent.com/CS-433/ml-project-2-scikit-learn2/e370e7f14f49c30026b572fa7bcd06b5fd1ec1c7/data/GC_ivtff_0c.txt'

# Four-poet medieval Persian reference, using Ganjoor-derived public dataset.
POETS={'کمال خجندی','حافظ','عبید زاکانی','شاه نعمت\u200cالله ولی'}
NORM=str.maketrans({'ي':'ی','ى':'ی','ك':'ک','ة':'ه','ۀ':'ه'})
def persian_words(s):
    s=unicodedata.normalize('NFKC',s).translate(NORM)
    out=[]
    for ch in s:
        if '\u0600' <= ch <= '\u06ff' and unicodedata.category(ch).startswith('L'):
            out.append(ch)
        else:
            out.append(' ')
    return ''.join(out).split()

D=load_dataset('mabidan/ganjoor',split='train',streaming=True)
refs=collections.Counter(); kho=collections.Counter(); poet_counts=collections.Counter()
for r in D:
    p=r.get('poet','')
    if p not in POETS: continue
    ws=persian_words(r.get('text',''))
    refs.update(ws); poet_counts[p]+=len(ws)
    if p=='کمال خجندی': kho.update(''.join(ws))
print('REF',dict(poet_counts),'words',sum(refs.values()),'types',len(refs))

# Exact IVTFF extraction. Both '.' and ',' are treated as word boundaries: certain + uncertain spaces.
raw=urllib.request.urlopen(V101_URL).read().decode('utf-8','ignore')
def atomic_special(m):
    return chr(0xE000 + int(m.group(1)))
base=[]; linewords={}
for ln in raw.splitlines():
    m=re.match(r'^<([^>]+)>\s+(.*)$',ln)
    if not m: continue
    locus,text=m.group(1),m.group(2)
    if '.' not in locus: continue
    text=re.sub(r'<[^>]*>','',text)
    text=re.sub(r'@(\d+);',atomic_special,text)
    toks=[x.strip() for x in re.split(r'[.,]',text) if x.strip()]
    base.extend(toks); linewords[locus]=toks
print('V101','tokens',len(base),'types',len(set(base)),'8am',base.count('8am'))

# Edwards L0 reconstruction from published rules + missing-glyph definitions inferred from published frequency anchors.
VAR=str.maketrans({'w':'g','x':'y','&':'8','(':'9'})
def trans(w,mode='published'):
    w=w.translate(VAR)
    # L-zero atomic bigrams
    for a,b in [('am','μ'),('an','ν'),('ay','γ'),('az','ζ'),('c8','ς')]: w=w.replace(a,b)
    if mode=='published':
        # Published L0 missing-glyph rationale: C=cc, K=k+1, H=h+1; j merged with g variant.
        w=w.replace('C','cc').replace('K','k1').replace('H','h1').replace('j','g')
    elif mode=='C':
        w=w.replace('C','cc')
    # L0.3 / L0.4
    w=w.replace('4o','④')
    w=w.replace('cc9','κ').replace('C9','κ')
    return w

def make(mode):
    words=[trans(w,mode) for w in base]
    wc=collections.Counter(words); gc=collections.Counter(''.join(words))
    return words,wc,gc

words,wc,gc=make('published')
G=[g for g,_ in gc.most_common()]
ng=sum(gc.values())
print('POSTVAR_8MU',wc['8μ'])
print('TOP_GLYPHS',[(i+1,g,round(100*gc[g]/ng,3)) for i,g in enumerate(G[:20])])
print('TOP_WORDS',wc.most_common(25))

P=[c for c,_ in kho.most_common()]
def swap(L,a,b):
    L=L[:]; i,j=L.index(a),L.index(b); L[i],L[j]=L[j],L[i]; return L
PE=swap(swap(P,'م','ه'),'ک','س')
print('P_BASE',P[:20]); print('P_ED',PE[:20])
print('ANCHORS',{'e_rank':G.index('e')+1,'e_freq':100*gc['e']/ng,'e_maps':PE[G.index('e')],
                 '8_rank':G.index('8')+1,'8_maps':PE[G.index('8')]})

def mapdict(letters): return {g:letters[i] for i,g in enumerate(G[:len(letters)])}
def dec(w,M):
    try: return ''.join(M[x] for x in w)
    except KeyError: return None

def hitcount(lst,letters,reverse=False):
    M=mapdict(letters); n=0
    for w in lst:
        z=dec(w,M)
        if z is None: continue
        if refs.get(z,0)>0 or (reverse and refs.get(z[::-1],0)>0): n+=1
    return n

RWORDS=[w for w,_ in wc.most_common()]
sets={'1_50':RWORDS[:50],'51_100':RWORDS[50:100],'101_200':RWORDS[100:200],'201_500':RWORDS[200:500]}
OBS={k:hitcount(v,PE,False) for k,v in sets.items()}
OBSR={k:hitcount(v,PE,True) for k,v in sets.items()}
print('OBS_DIRECT',OBS); print('OBS_DIRECT_OR_REVERSE',OBSR)

# Matched null: exactly two disjoint Persian rank swaps, each no larger in frequency gap than Edwards' looser selected pair.
tot=sum(kho.values()); pf={c:kho[c]/tot for c in P}
maxgap=max(abs(pf['م']-pf['ه']),abs(pf['ک']-pf['س']))
# Relevant ranks only; minimum 20 to include enough comparable swaps.
relevant=set(''.join(RWORDS[:500])); maxr=max([G.index(g) for g in relevant if g in G and G.index(g)<len(P)] or [19])+1
L=min(len(P),max(20,maxr))
pairs=[]
for i in range(L):
    for j in range(i+1,L):
        if abs(pf[P[i]]-pf[P[j]]) <= maxgap+1e-15: pairs.append((i,j))
null=[]
for a,(i,j) in enumerate(pairs):
    for k,l in pairs[a+1:]:
        if len({i,j,k,l})<4: continue
        Q=P[:]; Q[i],Q[j]=Q[j],Q[i]; Q[k],Q[l]=Q[l],Q[k]
        null.append(tuple(hitcount(sets[name],Q,False) for name in sets))
print('NULL_FAMILY','maxgap',maxgap,'rank_span',L,'pairs',len(pairs),'two_swap_models',len(null))
for idx,name in enumerate(sets):
    arr=[x[idx] for x in null]; mu=sum(arr)/len(arr); sd=(sum((x-mu)**2 for x in arr)/len(arr))**0.5; obs=OBS[name]
    print('MATCHED',name,json.dumps({'obs':obs,'mean':mu,'sd':sd,'z':(obs-mu)/sd if sd else None,'p_ge':sum(x>=obs for x in arr)/len(arr),'max':max(arr)},ensure_ascii=False))

# Fully random bijections: diagnostic only.
rng=random.Random(20260907); random_vals={k:[] for k in sets}
for _ in range(20000):
    Q=P[:]; rng.shuffle(Q)
    for k,v in sets.items(): random_vals[k].append(hitcount(v,Q,False))
for k,arr in random_vals.items():
    mu=sum(arr)/len(arr); sd=(sum((x-mu)**2 for x in arr)/len(arr))**0.5; obs=OBS[k]
    print('RANDOM',k,json.dumps({'obs':obs,'mean':mu,'sd':sd,'z':(obs-mu)/sd if sd else None,'p_ge':sum(x>=obs for x in arr)/len(arr),'max':max(arr)},ensure_ascii=False))

# Representation sensitivity around the non-neutral missing-glyph choices.
for mode in ['minimal','C','published']:
    _,wcm,gcm=make(mode); GG=[g for g,_ in gcm.most_common()]
    # local scorer uses this representation's rank order
    def hlocal(lst):
        M={g:PE[i] for i,g in enumerate(GG[:len(PE)])}; n=0
        for w in lst:
            try:z=''.join(M[x] for x in w)
            except KeyError:continue
            n+=refs.get(z,0)>0
        return n
    rr=[w for w,_ in wcm.most_common()]
    print('REP',mode,'top8',GG[:8],'top50',hlocal(rr[:50]),'51_100',hlocal(rr[50:100]))

# Specific published line, with no post-hoc reparsing.
for locus in ['f80v.22','80v.22']:
    if locus in linewords:
        lw=[trans(w) for w in linewords[locus]]
        M=mapdict(PE)
        out=[]
        for w in lw:
            z=dec(w,M); out.append((w,z,refs.get(z,0) if z else 0))
        print('F80V22_STRICT',out)
