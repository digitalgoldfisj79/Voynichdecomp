import re, urllib.request, unicodedata, collections, json, random
from datasets import load_dataset

TARGET_POETS={'کمال خجندی','حافظ','عبید زاکانی','شاه نعمت\u200cالله ولی'}
MAP=str.maketrans({'ي':'ی','ى':'ی','ك':'ک','ة':'ه','ۀ':'ه'})

def norm_persian(s):
    s=unicodedata.normalize('NFKC',s).translate(MAP)
    out=[]
    for ch in s:
        if '\u0600' <= ch <= '\u06ff' and unicodedata.category(ch).startswith('L'):
            out.append(ch)
        else:
            out.append(' ')
    return ''.join(out)

D=load_dataset('mabidan/ganjoor',split='train',streaming=True)
poet_text={p:[] for p in TARGET_POETS}
for r in D:
    p=r.get('poet','')
    if p in TARGET_POETS:
        poet_text[p].append(r.get('text',''))
ref_words=collections.Counter(); kho_chars=collections.Counter(); corpus_letters=0
for p,chunks in poet_text.items():
    for chunk in chunks:
        ws=norm_persian(chunk).split()
        ref_words.update(ws)
        corpus_letters+=sum(map(len,ws))
        if p=='کمال خجندی':
            kho_chars.update(''.join(ws))
print('CORPUS', {p:len(v) for p,v in poet_text.items()}, 'words',sum(ref_words.values()), 'types',len(ref_words),'letters',corpus_letters)
print('KHO_TOP',kho_chars.most_common(20))

url='https://raw.githubusercontent.com/Aspect-Research/voynich-autoexploration/master/data/transcriptions/v101_claston.txt'
req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
raw=urllib.request.urlopen(req,timeout=60).read().decode('utf-8','ignore')
base_words=[]; lines={}
for ln in raw.splitlines():
    m=re.match(r'^<([^>]+)>(.*)$',ln)
    if not m: continue
    lab,txt=m.group(1),m.group(2).strip()
    txt=re.sub(r'[-=]+$','',txt)
    toks=[t.strip() for t in txt.split('.') if t.strip()]
    base_words.extend(toks); lines[lab]=toks
print('V101',len(base_words),'types',len(set(base_words)),'8am',base_words.count('8am'))

variant_trans=str.maketrans({'w':'g','x':'y','&':'8','(':'9'})
def transform_word(w,mode='published'):
    w=w.translate(variant_trans)
    for a,b in [('4o','④'),('am','μ'),('an','ν'),('ay','γ'),('az','ζ'),('c8','ς')]:
        w=w.replace(a,b)
    if mode=='published':
        w=w.replace('C','cc').replace('K','k1').replace('H','h1').replace('j','g')
    elif mode=='C':
        w=w.replace('C','cc')
    w=w.replace('cc9','κ').replace('C9','κ')
    return w

def corpus_for(mode):
    ww=[transform_word(w,mode) for w in base_words]
    return ww,collections.Counter(ww),collections.Counter(''.join(ww))

ww,wc,gc=corpus_for('published')
ng=sum(gc.values()); grank=[g for g,_ in gc.most_common()]
print('L04_TOP_GLYPHS',[(i+1,g,gc[g],round(100*gc[g]/ng,3)) for i,g in enumerate(grank[:15])])
print('L04_TOP_WORDS',wc.most_common(25))

base_prank=[c for c,_ in kho_chars.most_common()]
def swap_chars(lst,a,b):
    lst=lst[:]
    ia,ib=lst.index(a),lst.index(b)
    lst[ia],lst[ib]=lst[ib],lst[ia]
    return lst
ed_prank=swap_chars(swap_chars(base_prank,'م','ه'),'ک','س')
print('PERSIAN_RANK_BASE',base_prank[:20])
print('PERSIAN_RANK_ED',ed_prank[:20])
print('ANCHORS', {'e_rank':grank.index('e')+1,'e_maps':ed_prank[grank.index('e')], '8_rank':grank.index('8')+1,'8_maps':ed_prank[grank.index('8')]})

def mapping_for(glyph_rank,letter_rank):
    return {g:letter_rank[i] for i,g in enumerate(glyph_rank[:len(letter_rank)])}
def decode(w,m):
    try:
        return ''.join(m[ch] for ch in w)
    except KeyError:
        return None

def score_words(wordlist,glyph_rank,letter_rank,allow_reverse=False):
    m=mapping_for(glyph_rank,letter_rank); hits=0; decoded=[]
    for w in wordlist:
        z=decode(w,m); hit=False; chosen=z
        if z is not None:
            if ref_words.get(z,0)>0:
                hit=True
            if allow_reverse and not hit and ref_words.get(z[::-1],0)>0:
                hit=True; chosen=z[::-1]
        hits+=hit
        decoded.append((w,z,chosen,ref_words.get(chosen,0) if chosen else 0,hit))
    return hits,decoded

ranked_words=[w for w,_ in wc.most_common()]
train=ranked_words[:50]; hold1=ranked_words[50:100]; hold2=ranked_words[100:200]
for rev in [False,True]:
    s0,d0=score_words(train,grank,ed_prank,rev)
    s1,_=score_words(hold1,grank,ed_prank,rev)
    s2,_=score_words(hold2,grank,ed_prank,rev)
    print('ED_SCORE','reverse' if rev else 'direct',s0,s1,s2)
    if not rev:
        print('ED_TOP50',json.dumps(d0,ensure_ascii=False))

tot=sum(kho_chars.values()); pf={c:kho_chars[c]/tot for c in base_prank}
gapA=abs(pf['م']-pf['ه']); gapB=abs(pf['ک']-pf['س']); maxgap=max(gapA,gapB)
relevant=set(''.join(train+hold1+hold2))
max_rank=max((grank.index(g) for g in relevant if g in grank and grank.index(g)<len(base_prank)),default=len(base_prank)-1)+1
L=min(len(base_prank),max(max_rank,16)); pairs=[]
for i in range(L):
    for j in range(i+1,L):
        a,b=base_prank[i],base_prank[j]
        if abs(pf[a]-pf[b])<=maxgap+1e-15:
            pairs.append((i,j))
print('SWAP_GAPS',gapA,gapB,'max',maxgap,'L',L,'allowed_pairs',len(pairs))

def score_fast(wordlist,letters):
    m=mapping_for(grank,letters); h=0
    for w in wordlist:
        z=decode(w,m)
        h+=bool(z is not None and ref_words.get(z,0)>0)
    return h

vals=[]
for a in range(len(pairs)):
    i,j=pairs[a]
    for b in range(a+1,len(pairs)):
        k,l=pairs[b]
        if len({i,j,k,l})<4:
            continue
        lr=base_prank[:]
        lr[i],lr[j]=lr[j],lr[i]
        lr[k],lr[l]=lr[l],lr[k]
        vals.append((score_fast(train,lr),score_fast(hold1,lr),score_fast(hold2,lr),(i,j,k,l)))
print('NULL_N',len(vals))

def stats(idx,obs):
    arr=[v[idx] for v in vals]; n=len(arr); mu=sum(arr)/n
    sd=(sum((x-mu)**2 for x in arr)/n)**0.5
    q=sorted(arr)
    return {'mean':mu,'sd':sd,'obs':obs,'z':(obs-mu)/sd if sd else None,'ge':sum(x>=obs for x in arr),'p_ge':sum(x>=obs for x in arr)/n,'max':max(arr),'p95':q[int(.95*(n-1))],'p99':q[int(.99*(n-1))]}

obs0=score_fast(train,ed_prank); obs1=score_fast(hold1,ed_prank); obs2=score_fast(hold2,ed_prank)
print('MATCHED_NULL',json.dumps({'top50':stats(0,obs0),'51_100':stats(1,obs1),'101_200':stats(2,obs2)},ensure_ascii=False))
print('ED_SWAP_POS',[(x,base_prank.index(x)) for x in ['م','ه','ک','س']])

rng=random.Random(20260907); R=[]
for _ in range(20000):
    lr=base_prank[:]
    rng.shuffle(lr)
    R.append((score_fast(train,lr),score_fast(hold1,lr)))
def rstats(idx,obs):
    a=[x[idx] for x in R]; mu=sum(a)/len(a); sd=(sum((x-mu)**2 for x in a)/len(a))**.5
    return {'mean':mu,'sd':sd,'obs':obs,'z':(obs-mu)/sd if sd else None,'p_ge':sum(x>=obs for x in a)/len(a),'max':max(a)}
print('RANDOM_NULL',json.dumps({'top50':rstats(0,obs0),'51_100':rstats(1,obs1)},ensure_ascii=False))

rep={}
for mode in ['minimal','C','published']:
    _,wc2,gc2=corpus_for(mode)
    gr2=[g for g,_ in gc2.most_common()]
    rw=[w for w,_ in wc2.most_common()]
    tr,ho=rw[:50],rw[50:100]
    st,_=score_words(tr,gr2,ed_prank,False)
    sh,_=score_words(ho,gr2,ed_prank,False)
    rep[mode]={'top50':st,'51_100':sh,'glyph_top8':gr2[:8]}
print('REP_SENS',json.dumps(rep,ensure_ascii=False))

for key in ['80v.22','f80v.22']:
    if key in lines:
        lw=[transform_word(x,'published') for x in lines[key]]
        sc,dec=score_words(lw,grank,ed_prank,False)
        print('F80V22',key,sc,len(lw),json.dumps(dec,ensure_ascii=False))
