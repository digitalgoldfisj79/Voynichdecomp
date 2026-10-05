import urllib.request,numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cf"};exec(compile(urllib.request.urlopen(URL).read().decode(),URL,"exec"),m)
rows,_=m["synth_rows"](False);a="V10_1";b="V10_3"
tr=[r for r in rows if r["fold"] in (2,3) and r["token"] in (a,b)]
va=[r for r in rows if r["fold"]==4 and r["token"] in (a,b)]
print("counts",len(tr),len(va),sum(r["token"]==a for r in tr),sum(r["token"]==b for r in tr),sum(r["token"]==a for r in va),sum(r["token"]==b for r in va))
def feats(r):
 d={"lp="+str(r["lp"]):1.}
 for lag in (-2,-1,1,2):
  x=r[f"n{lag:+d}"];d[f"L{lag}="+str(x)]=1.
 d["PAIR11="+str(r["n-1"])+"|"+str(r["n+1"])]=1.
 d["PAIR22="+str(r["n-2"])+"|"+str(r["n+2"])]=1.
 d["QUAD="+str(r["n-2"])+"|"+str(r["n-1"])+"|"+str(r["n+1"])]=1.
 return d
v=DictVectorizer();X=v.fit_transform([feats(r) for r in tr]);V=v.transform([feats(r) for r in va])
yt=np.array([r["token"]==b for r in tr],int)
print("shape",X.shape,"classes",set(yt))
try:
 md=LogisticRegression(C=.1,max_iter=500,solver="liblinear").fit(X,yt)
 print("OK",md.classes_,md.predict_proba(V)[:3])
except Exception as e:print("ERR",repr(e))
