import sys,re,math,collections as C
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
atom=re.compile('ckh|cth|cph|cfh|ch|sh|oq|.')
def load(p):
    out=[]
    for raw in open(p,encoding='utf-8'):
        raw=raw.rstrip('\n'); s=raw.split('\t')[-1]
        ts=[t for t in s.split() if '?' not in t and "'" not in t]
        if ts: out.append(ts)
    return out
def blocks(lines,n):
    seq=[(t,i) for i,l in enumerate(lines) for t in l]
    return [seq[i:i+n] for i in range(0,len(seq)-n+1,n)]
def h2(b):
    ts=[t for t,l in b];ats=[atom.findall(t) for t in ts]
    pairs=[i for i in range(len(b)-1) if b[i][1]==b[i+1][1]]
    bg=C.Counter()
    for aa in ats:bg.update(zip(aa,aa[1:]))
    for i in pairs:bg[(ats[i][-1],ats[i+1][0])]+=1
    pre=C.Counter()
    for (a,z),v in bg.items():pre[a]+=v
    nb=sum(bg.values());return -sum(v/nb*math.log2(v/pre[a]) for (a,z),v in bg.items())
def auc_h2(A,B,ga,gb):
    X=np.r_[A,B].reshape(-1,1);y=np.r_[np.zeros(len(A)),np.ones(len(B))];groups=np.r_[ga,gb+100000];vals=[]
    for seed in [42,43,44,45,46]:
        for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,y,groups):
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000));clf.fit(X[tr],y[tr]);vals.append(roc_auc_score(y[te],clf.predict_proba(X[te])[:,1]))
    return float(np.mean(vals))
ref=load(sys.argv[1]); odd=load('daten/vms_B_odd.txt'); even=load('daten/vms_B_even.txt')
A=np.array([h2(b) for b in blocks(ref,120)]); ga=np.arange(len(A))//5
To=C.Counter(t for l in odd for t in l); Te=C.Counter(t for l in even for t in l)
print(f"Referenz {sys.argv[1]}: {sum(map(len,ref))} Token, {len(A)} Bloecke, h2 {A.mean():.3f}; Selbstbaseline odd-Typen in even: {100*sum(v for t,v in To.items() if t in Te)/sum(To.values()):.1f}")
for f in sys.argv[2:]:
    g=load(f); G=np.array([h2(b) for b in blocks(g,120)])
    rng=np.random.default_rng(42); sel=np.sort(rng.choice(len(G),len(A),replace=False)); B=G[sel]; gb=sel//5
    toks=[t for l in g for t in l]; n=len(toks); c=C.Counter(toks[:21522])
    dk=100*sum(1 for t in toks if t in To)/n; de=100*sum(1 for t in toks if t in Te)/n
    print(f"{f:<28} h2 {G.mean():.3f} Luecke {G.mean()-A.mean():+.3f} AUC-h2 {auc_h2(A,B,ga,gb):.3f} | Typen21522 {len(c)} Hapax {100*sum(v==1 for v in c.values())/len(c):.1f} | Deckung odd {dk:.1f} even {de:.1f} (Memo {dk-de:+.1f})")
