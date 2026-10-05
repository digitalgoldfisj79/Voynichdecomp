#!/usr/bin/env python3
import urllib.request,numpy as np,traceback
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cf"};exec(compile(urllib.request.urlopen(BASE).read().decode(),BASE,"exec"),m)
rows,_=m["synth_rows"](False)
def feats(r):
 d={"lp="+str(r["lp"]):1.}
 for lag in (-2,-1,1,2):d[f"L{lag}="+str(r[f'n{lag:+d}'])]=1.
 d["near="+str(r["n-1"])+"|"+str(r["n+1"])]=1.
 return d
for a,b in [("V08_0","V08_3"),("V13_1","V15_1")]:
 tr=[r for r in rows if r["fold"] in (2,3) and r["token"] in (a,b)]
 va=[r for r in rows if r["fold"]==4 and r["token"] in (a,b)]
 print("PAIR",a,b,"train",len(tr),{x:sum(r["token"]==x for r in tr) for x in (a,b)},"val",len(va),{x:sum(r["token"]==x for r in va) for x in (a,b)},flush=True)
 try:
  v=DictVectorizer();X=v.fit_transform([feats(r) for r in tr]);V=v.transform([feats(r) for r in va])
  y=np.array([r["token"]==b for r in tr],int);z=np.array([r["token"]==b for r in va],int)
  print("SHAPE",X.shape,V.shape,"classes",np.unique(y,return_counts=True),flush=True)
  md=LogisticRegression(C=.1,max_iter=500,solver="liblinear").fit(X,y)
  p=md.predict_proba(V)[:,1];prior=(y.sum()+.5)/(len(y)+1)
  base=np.mean(np.log2(np.maximum(np.where(z==1,prior,1-prior),1e-12)))
  ll=np.mean(np.log2(np.maximum(np.where(z==1,p,1-p),1e-12)))
  print("GAIN",ll-base,flush=True)
 except Exception as e:
  print("ERROR",repr(e));traceback.print_exc()
