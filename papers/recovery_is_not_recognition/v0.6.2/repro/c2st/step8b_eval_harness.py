"""Step 8b: REPLACEMENT EVAL HARNESS.
(1) C2ST: train a classifier to tell each candidate from held-out real VMS, per-chunk features,
    5-fold CV ROC-AUC. AUC~0.5 = indistinguishable (good); AUC->1 = bad. Tolerance-free, no xsec.
(2) Standardized discrepancy: |candidate-real| per metric in within-corpus bootstrap-SE units (the
    fix for the cross-section-SD circularity). Reports worst metric + count beyond 3 SE.
Acceptance test: a good metric must REJECT line-shuffle (AUC>>0.5) even though it passes 84-battery 74.6."""
import pickle, math, statistics, random
import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

cand=pickle.load(open('/tmp/vms/work/candidates.pkl','rb'))
GAL=set('pfkt')

def chunks_of(tokens, line_lens, target=120):
    lines=[]; i=0
    for L in line_lens:
        if i>=len(tokens): break
        lines.append(tokens[i:i+L]); i+=L
    out=[]; cur=[]; n=0
    for ln in lines:
        cur.append(ln); n+=len(ln)
        if n>=target: out.append(cur); cur=[]; n=0
    if cur: out.append(cur)
    return out

def feats(chunk):
    toks=[t for ln in chunk for t in ln]
    if len(toks)<20: return None
    wl=[len(t) for t in toks]
    a=np.array(wl,float)
    ac=np.corrcoef(a[:-1],a[1:])[0,1] if len(a)>3 and a.std()>0 else 0.0
    ttr=len(set(toks))/len(toks)
    fr=Counter(toks); hapax=sum(1 for w in fr if fr[w]==1)/len(fr)
    chars=''.join(toks); cc=Counter(chars); tot=len(chars)
    H1=-sum((n/tot)*math.log2(n/tot) for n in cc.values())
    tr=defaultdict(Counter)
    for x,y in zip(chars,chars[1:]): tr[x][y]+=1
    t2=sum(sum(c.values()) for c in tr.values())
    H2=sum((sum(c.values())/t2)*(-sum((n/sum(c.values()))*math.log2(n/sum(c.values())) for n in c.values())) for c in tr.values()) if t2 else 0
    chardist_max=max(cc.values())/tot
    digs=set(zip(chars,chars[1:])); digcov=len(digs)/(len(cc)**2) if cc else 0
    wt=defaultdict(Counter)
    for t in toks:
        for x,y in zip(t,t[1:]): wt[x][y]+=1
    wtt=sum(sum(c.values()) for c in wt.values())
    wH=sum((sum(c.values())/wtt)*(-sum((n/sum(c.values()))*math.log2(n/sum(c.values())) for n in c.values())) for c in wt.values()) if wtt else 0
    opener=np.mean([1.0 if ln and ln[0] and ln[0][0] in GAL else 0.0 for ln in chunk]) if chunk else 0
    fh=[]
    for ln in chunk:
        s=''.join(ln)
        if not s: continue
        g=[i for i,ch in enumerate(s) if ch in GAL]
        if g: fh.append(np.mean([1.0 if i<len(s)/2 else 0.0 for i in g]))
    charpos=np.mean(fh) if fh else 0.5
    adj=np.mean([1.0 if toks[i]==toks[i-1] else 0.0 for i in range(1,len(toks))])
    return [statistics.mean(wl),statistics.pstdev(wl),ac,ttr,hapax,H1,H2,chardist_max,digcov,wH,opener,charpos,adj]

def featmatrix(name):
    toks,lens=cand[name]
    X=[feats(c) for c in chunks_of(toks,lens)]
    return np.array([x for x in X if x is not None])

ref=featmatrix('real_A')
print(f"real_A reference chunks: {len(ref)}")
print(f"\n{'candidate':18s} {'C2ST AUC (5-fold)':>20s}  verdict")
rng=np.random.default_rng(0)
auc_rows={}
for name in ['real_B','line-shuffle','word-shuffle','gen_template_v10','delex_char3']:
    Xc=featmatrix(name)
    n=min(len(ref),len(Xc))
    ri=rng.choice(len(ref),n,replace=False); ci=rng.choice(len(Xc),n,replace=False)
    X=np.vstack([ref[ri],Xc[ci]]); y=np.r_[np.zeros(n),np.ones(n)]
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
    cv=StratifiedKFold(5,shuffle=True,random_state=0)
    aucs=cross_val_score(clf,X,y,cv=cv,scoring='roc_auc')
    m,s=aucs.mean(),aucs.std()
    auc_rows[name]=(m,s)
    verdict=('INDISTINGUISHABLE (~real)' if m<0.6 else 'detectable' if m<0.8 else 'EASILY REJECTED')
    print(f"{name:18s} {m:8.3f} +/- {s:5.3f}      {verdict}")

def corpus_metrics(toks):
    wl=[len(t) for t in toks]; fr=Counter(toks); chars=''.join(toks); cc=Counter(chars); tot=len(chars)
    H1=-sum((n/tot)*math.log2(n/tot) for n in cc.values())
    return {'wl_mean':statistics.mean(wl),'ttr':len(fr)/len(toks),
            'hapax':sum(1 for w in fr if fr[w]==1)/len(fr),
            'H1':H1,'chardist_max':max(cc.values())/tot}
realA=cand['real_A'][0]
lines=[]; i=0
for L in cand['real_A'][1]:
    if i>=len(realA): break
    lines.append(realA[i:i+L]); i+=L
boot=[]
R=random.Random(3)
for _ in range(50):
    samp=[lines[R.randrange(len(lines))] for _ in range(len(lines))]
    boot.append(corpus_metrics([t for l in samp for t in l]))
keys=list(boot[0]); se={k:statistics.pstdev([b[k] for b in boot]) for k in keys}
base=corpus_metrics(realA)
print(f"\nStandardized discrepancy |cand-real|/bootSE  (worst metric, #metrics>3 SE):")
for name in ['real_B','line-shuffle','word-shuffle','gen_template_v10','delex_char3']:
    cm=corpus_metrics(cand[name][0])
    z={k:abs(cm[k]-base[k])/se[k] if se[k]>0 else 0 for k in keys}
    worst=max(z,key=z.get); n3=sum(1 for k in z if z[k]>3)
    print(f"  {name:18s} worst={worst}({z[worst]:.1f} SE)  #>3SE={n3}/{len(keys)}")

pickle.dump({'auc':auc_rows},open('/tmp/vms/work/eval_harness.pkl','wb'))
print("\nOK -> eval_harness.pkl")
