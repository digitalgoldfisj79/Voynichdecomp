#!/usr/bin/env python3
import urllib.request, numpy as np, json
from sklearn.metrics import roc_auc_score
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/af48d2b45ad416cfac9589c47f696fa6bb0d54ff/research/context_first_lexical_equivalence_20261005.py"
m={"__name__":"cfmod"};exec(compile(urllib.request.urlopen(URL).read().decode(),URL,"exec"),m)
def normemb(types,obs,exp):
 out={}
 for t in types:
  mm=m["multiplier"](obs[t],exp[t]);x=np.log(np.maximum(mm,1e-9)).ravel();x-=x.mean();n=np.linalg.norm(x);out[t]=x/n if n else x
 return out
def run(shuffle=False,seedoff=0):
 rows,_=m["synth_rows"](shuffle)
 types,dc,vc,tc=m["eligible_types"](rows);vmap,_=m["context_vocab"](rows)
 base,glob=m["fit_baseline"](rows,vmap,(2,3))
 o2,e2=m["profiles"](rows,vmap,(2,),set(types),base,glob);o3,e3=m["profiles"](rows,vmap,(3,),set(types),base,glob)
 E2=normemb(types,o2,e2);E3=normemb(types,o3,e3)
 rec=[]
 for i,a in enumerate(types):
  for b in types[i+1:]:
   cross=.5*(float(np.dot(E2[a],E3[b]))+float(np.dot(E2[b],E3[a])))
   selfsim=.5*(float(np.dot(E2[a],E3[a]))+float(np.dot(E2[b],E3[b])))
   eqdef=selfsim-cross
   avg=.25*(float(np.dot(E2[a],E2[b]))+float(np.dot(E3[a],E3[b]))+
            float(np.dot(E2[a],E3[b]))+float(np.dot(E2[b],E3[a])))
   truth=a.split("_")[0]==b.split("_")[0]
   rec.append((a,b,truth,cross,selfsim,eqdef,avg))
 y=np.array([r[2] for r in rec],int)
 for name,ix,sgn in [("cross",3,1),("neg_eqdef",5,-1),("avg",6,1),("self",4,1)]:
  sc=np.array([r[ix]*sgn for r in rec])
  print("AUC",shuffle,name,roc_auc_score(y,sc))
 # thresholds on avg and eqdef chosen as generic grid
 for ath in [.1,.15,.2,.25,.3,.35,.4,.45,.5]:
  rr=[r for r in rec if r[6]>=ath]
  if rr:
   print("TH",shuffle,ath,len(rr),sum(r[2] for r in rr),sum(r[2] for r in rr)/len(rr))
 # rank top by avg and cross/self gap
 print("TOPAVG",shuffle,[(r[0],r[1],r[2],round(r[6],3),round(r[5],3)) for r in sorted(rec,key=lambda z:z[6],reverse=True)[:30]])
 return rec
run(False);run(True)
