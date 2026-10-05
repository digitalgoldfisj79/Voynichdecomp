#!/usr/bin/env python3
import urllib.request,numpy as np,collections
from sklearn.metrics import roc_auc_score
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cf"};exec(compile(urllib.request.urlopen(URL).read().decode(),URL,"exec"),m)
def run(shuffle=False):
 rows,_=m["synth_rows"](shuffle)
 types,dc,vc,tc=m["eligible_types"](rows);vmap,_=m["context_vocab"](rows)
 base,glob=m["fit_baseline"](rows,vmap,(2,3));obs,exp=m["profiles"](rows,vmap,(2,3),set(types),base,glob)
 E=m["embedding"](types,obs,exp);S=E@E.T;np.fill_diagonal(S,-9)
 truth=lambda a,b:int(a.split("_")[0]==b.split("_")[0])
 out={}
 for k in [1,2,3,4,5]:
  pairs=set()
  top=[set(np.argsort(-S[i])[:k]) for i in range(len(types))]
  for i in range(len(types)):
   for j in top[i]:
    if i in top[j]:pairs.add(tuple(sorted((i,int(j)))))
  y=[truth(types[i],types[j]) for i,j in pairs]
  out[k]={"n":len(pairs),"true":sum(y),"precision":sum(y)/len(y) if y else None,
          "recall":sum(y)/91.0}
 # reciprocal rank score = 1/max(rank_i(j),rank_j(i)); report high thresholds
 ranks=np.argsort(np.argsort(-S,axis=1),axis=1)+1
 allp=[]
 for i in range(len(types)):
  for j in range(i+1,len(types)):
   r=max(int(ranks[i,j]),int(ranks[j,i])); allp.append((r,truth(types[i],types[j]),types[i],types[j],float(S[i,j])))
 for R in [1,2,3,4,5,8,10]:
  z=[x for x in allp if x[0]<=R];y=[x[1] for x in z]
  out["R"+str(R)]={"n":len(z),"true":sum(y),"precision":sum(y)/len(y) if y else None,"recall":sum(y)/91 if y else 0}
 return out
print("ORDERED",run(False))
print("SHUFFLED",run(True))
