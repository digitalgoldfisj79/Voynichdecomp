#!/usr/bin/env python3
"""
SGT13 literal d-q-y wheel simulation.

Mechanism:
- Paragraph start draws an opener from the empirical training initial distribution.
- If current opener is d/q/y, next state is tied to a literal 3-position wheel:
  stay / clockwise (d->q->y->d) / anticlockwise / exit-to-background.
- If current opener is outside d/q/y ("X"), next category is d/q/y/X.
- X emissions are drawn from one shared background opener distribution.

Evaluation:
- true ZL3b paragraph sequences only;
- 5-fold held-out by folio number mod 5;
- compare full-opener cross entropy to IID and unrestricted first-order Markov;
- fit all data and simulate 1000 corpora preserving observed paragraph lengths.

No third-party dependencies.
"""
from __future__ import annotations
import math, random, re, urllib.request
from collections import Counter, defaultdict

URL="https://www.voynich.nu/data/ZL3b-n.txt"
ANCHORS=("d","q","y")
CW={"d":"q","q":"y","y":"d"}
CCW={"d":"y","y":"q","q":"d"}

def opener(payload):
    s=re.sub(r"<[^>]*>","",payload.strip()).strip()
    if not s or s[0] in "?@": return None
    m=re.match(r"^\[([^:\]]+):[^\]]+\]",s)
    if m: s=m.group(1)+s[m.end():]
    m=re.match(r"^\{([^}]+)\}",s)
    if m:
        a=re.sub(r"@[0-9]+;","",m.group(1))
        a=re.sub(r"[^A-Za-z]","",a)
        if not a: return None
        s=a+s[m.end():]
    for a in ("cfh","cph","ckh","cth","ch","sh"):
        if s.startswith(a): return a
    m=re.match(r"^([a-z])",s,re.I)
    return m.group(1).lower() if m else None

def load():
    raw=urllib.request.urlopen(URL).read().decode("utf-8")
    ctr=defaultdict(int); paras=defaultdict(list)
    for order,line in enumerate(raw.splitlines()):
        m=re.match(r"^<([^.,>]+)\.([0-9]+),([^>]*)>\s*(.*)$",line)
        if not m or "P" not in m.group(3): continue
        fol,ln,payload=m.group(1),int(m.group(2)),m.group(4)
        if "<%>" in payload or ctr[fol]==0: ctr[fol]+=1
        o=opener(payload)
        if o: paras[(fol,ctr[fol])].append((ln,order,o))
    seqs=[]
    for (fol,_),rows in paras.items():
        rows.sort()
        cur=[]; prev=None
        for ln,_,o in rows:
            if prev is not None and ln!=prev+1:
                if len(cur)>=2: seqs.append((fol,cur))
                cur=[]
            cur.append(o); prev=ln
        if len(cur)>=2: seqs.append((fol,cur))
    return seqs

def fnum(f):
    m=re.match(r"^f(\d+)",f)
    return int(m.group(1)) if m else 0

def cat(o): return o if o in ANCHORS else "X"

def norm(c):
    z=sum(c.values())
    return {k:v/z for k,v in c.items()}

def fit_literal(train,vocab,alpha=.5):
    non=[o for o in vocab if o not in ANCHORS]
    init=Counter({o:alpha for o in vocab})
    wheel=Counter({"stay":alpha,"cw":alpha,"ccw":alpha,"exit":alpha})
    xnext=Counter({"d":alpha,"q":alpha,"y":alpha,"X":alpha})
    bg=Counter({o:alpha for o in non})
    for _,ops in train:
        init[ops[0]]+=1
        for a,b in zip(ops,ops[1:]):
            ca,cb=cat(a),cat(b)
            if ca=="X": xnext[cb]+=1
            elif cb=="X": wheel["exit"]+=1
            elif cb==ca: wheel["stay"]+=1
            elif cb==CW[ca]: wheel["cw"]+=1
            else: wheel["ccw"]+=1
            if cb=="X": bg[b]+=1
    return dict(init=norm(init),wheel=norm(wheel),xnext=norm(xnext),bg=norm(bg))

def lp_literal(m,ops):
    L=math.log(m["init"][ops[0]])
    for a,b in zip(ops,ops[1:]):
        ca,cb=cat(a),cat(b)
        if ca=="X": p=m["xnext"][cb]
        elif cb=="X": p=m["wheel"]["exit"]
        elif cb==ca: p=m["wheel"]["stay"]
        elif cb==CW[ca]: p=m["wheel"]["cw"]
        else: p=m["wheel"]["ccw"]
        if cb=="X": p*=m["bg"][b]
        L+=math.log(p)
    return L

def fit_full(train,vocab,alpha=.5):
    init=Counter({o:alpha for o in vocab})
    P={a:Counter({b:alpha for b in vocab}) for a in vocab}
    for _,ops in train:
        init[ops[0]]+=1
        for a,b in zip(ops,ops[1:]): P[a][b]+=1
    return norm(init),{a:norm(c) for a,c in P.items()}

def lp_full(m,ops):
    init,P=m
    return math.log(init[ops[0]])+sum(math.log(P[a][b]) for a,b in zip(ops,ops[1:]))

def fit_iid(train,vocab,alpha=.5):
    c=Counter({o:alpha for o in vocab})
    for _,ops in train:c.update(ops)
    return norm(c)

def lp_iid(p,ops): return sum(math.log(p[o]) for o in ops)

def draw(p,R):
    u=R.random(); s=0
    for k,v in p.items():
        s+=v
        if u<=s:return k
    return next(reversed(p))

def next_o(a,m,R):
    ca=cat(a)
    if ca=="X": cb=draw(m["xnext"],R)
    else:
        z=draw(m["wheel"],R)
        cb=ca if z=="stay" else CW[ca] if z=="cw" else CCW[ca] if z=="ccw" else "X"
    return draw(m["bg"],R) if cb=="X" else cb

def metrics(seqs):
    F=B=same=n=0; joint=Counter(); row=Counter(); col=Counter()
    for _,ops in seqs:
        for a,b in zip(ops,ops[1:]):
            n+=1;joint[a,b]+=1;row[a]+=1;col[b]+=1;same+=a==b
            F+=((a,b) in (("d","q"),("q","y"),("y","d")))
            B+=((a,b) in (("q","d"),("y","q"),("d","y")))
    mi=sum((c/n)*math.log2(c*n/(row[a]*col[b])) for (a,b),c in joint.items())
    return dict(n=n,F=F,B=B,diff=F-B,share=F/(F+B),sameRate=same/n,mi=mi)

def main():
    seqs=load(); vocab=sorted({o for _,ops in seqs for o in ops})
    folds=[]
    for f in range(5):
        tr=[s for s in seqs if fnum(s[0])%5!=f]; te=[s for s in seqs if fnum(s[0])%5==f]
        lit=fit_literal(tr,vocab); full=fit_full(tr,vocab); iid=fit_iid(tr,vocab)
        n=sum(len(ops) for _,ops in te)
        def bits(L): return -L/math.log(2)/n
        folds.append(dict(fold=f,n=n,wheel=lit["wheel"],
          literal=bits(sum(lp_literal(lit,ops) for _,ops in te)),
          fullMarkov=bits(sum(lp_full(full,ops) for _,ops in te)),
          iid=bits(sum(lp_iid(iid,ops) for _,ops in te))))
    N=sum(x["n"] for x in folds)
    avg=lambda k:sum(x[k]*x["n"] for x in folds)/N
    iid,lit,full=avg("iid"),avg("literal"),avg("fullMarkov")
    print("HELDOUT bits/opener")
    print("IID",iid,"literal",lit,"fullMarkov",full)
    print("literal gain",iid-lit,"full gain",iid-full,
          "fraction", (iid-lit)/(iid-full), "gap",lit-full)
    for x in folds:print(x)

    m=fit_literal(seqs,vocab); obs=metrics(seqs); sims=[]
    for rep in range(1000):
        R=random.Random(20261002+rep); ss=[]
        for fol,ops0 in seqs:
            ops=[draw(m["init"],R)]
            for _ in range(1,len(ops0)):ops.append(next_o(ops[-1],m,R))
            ss.append((fol,ops))
        sims.append(metrics(ss))
    print("FIT",m)
    print("OBS",obs)
    for k in ("diff","share","sameRate","mi"):
        vals=[x[k] for x in sims]; mu=sum(vals)/len(vals)
        sd=(sum((x-mu)**2 for x in vals)/(len(vals)-1))**.5
        print(k,"obs",obs[k],"sim",mu,"sd",sd,"z",(obs[k]-mu)/sd)

if __name__=="__main__": main()
