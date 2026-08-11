# /// script
# requires-python = ">=3.11"
# dependencies = ["Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import collections, hashlib, json, urllib.request
from unidecode import unidecode

URL='https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu'
V=set('aeiou')
ALPH=set('abcdefghilmnopqrstuz')
DIP={'ae','oe','au','eu','ei'}

# Source-grounded broad Amadi grammar: one vowel nucleus, <=3 consonants before/after,
# <=5 consonants total; listed diphthongs are admitted as one nucleus.
def norm(w:str)->str:
    s=unidecode(w).lower().replace('j','i').replace('v','u').replace('w','u').replace('y','i').replace('x','s')
    return ''.join(c for c in s if c in ALPH)

def valid_unit(s:str)->bool:
    if not s:return False
    nv=sum(c in V for c in s)
    if nv==2 and s in DIP:return True
    if nv!=1:return False
    j=next(i for i,c in enumerate(s) if c in V)
    pre=j;post=len(s)-j-1
    return pre<=3 and post<=3 and pre+post<=5

def syllabify(w:str):
    s=norm(w)
    n=len(s)
    if not s:return []
    # DP minimizes number of units, then prefers longer leftmost unit deterministically.
    best=[None]*(n+1);best[n]=()
    for i in range(n-1,-1,-1):
        opts=[]
        for j in range(i+1,min(n,i+7)+1):
            u=s[i:j]
            if valid_unit(u) and best[j] is not None:
                cand=(u,)+best[j]
                opts.append(cand)
        if opts:
            best[i]=min(opts,key=lambda z:(len(z),tuple(-len(x) for x in z),z))
    return list(best[0]) if best[0] is not None else []

def parse(raw):
    out=[]
    for ln in raw.decode('utf-8','replace').splitlines():
        if not ln or ln.startswith('#'):continue
        c=ln.split('\t')
        if len(c)>=2 and c[0].isdigit():out.append(c[1])
    return out

def main():
    b=urllib.request.urlopen(URL,timeout=120).read();words=parse(b)
    C=collections.Counter();nw=ok=chars=kept=0;patterns=collections.Counter()
    bad=[]
    for w in words:
        s=norm(w)
        if not s:continue
        nw+=1;chars+=len(s);q=syllabify(w)
        if not q:
            if len(bad)<20:bad.append(s)
            continue
        ok+=1;kept+=len(s);C.update(q)
        for u in q:
            pat=''.join('V' if x in V else 'C' for x in u);patterns[pat]+=1
    ordered=sorted(C,key=lambda x:(-C[x],x))
    def cov(k):return sum(C[x] for x in ordered[:k])/max(1,sum(C.values()))
    out={'words':nw,'segmentable_words':ok,'word_coverage':ok/max(1,nw),'char_coverage':kept/max(1,chars),'distinct_syllables':len(C),'syllable_events':sum(C.values()),'top20':[(x,C[x]) for x in ordered[:20]],'coverage_at':{str(k):cov(k) for k in [64,89,100,129,158,256,512,1024,1365]},'patterns':patterns.most_common(),'bad_examples':bad,'source_sha256':hashlib.sha256(b).hexdigest()}
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
