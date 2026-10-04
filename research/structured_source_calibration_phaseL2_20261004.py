#!/usr/bin/env python3
import json,math,urllib.request,numpy as np
from numba import njit
from sklearn.metrics import normalized_mutual_info_score,roc_auc_score
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0d818e9f00012661077c83b27416ad0d2c99d581/research/structured_source_calibration_phaseL1_20261004.py"
m={"__name__":"L1"};exec(compile(urllib.request.urlopen(URL).read().decode(),URL,"exec"),m)
V,NSIG,NSEC,SECLEN,NFIT,NVAL=[m[x] for x in ("V","NSIG","NSEC","SECLEN","NFIT","NVAL")];SIG=m["SIG_OF"]
@njit
def fb(E,A,pi):
 n,K=E.shape;B=np.empty((n,K));mx=np.empty(n)
 for t in range(n):
  mm=E[t,0]
  for k in range(1,K):
   if E[t,k]>mm:mm=E[t,k]
  mx[t]=mm
  for k in range(K):B[t,k]=math.exp(E[t,k]-mm)
 al=np.empty((n,K));c=np.empty(n)
 s=0.
 for k in range(K):al[0,k]=pi[k]*B[0,k];s+=al[0,k]
 c[0]=s
 for k in range(K):al[0,k]/=s
 for t in range(1,n):
  s=0.
  for j in range(K):
   q=0.
   for i in range(K):q+=al[t-1,i]*A[i,j]
   al[t,j]=q*B[t,j];s+=al[t,j]
  c[t]=s
  for j in range(K):al[t,j]/=s
 be=np.ones((n,K))
 for t in range(n-2,-1,-1):
  for i in range(K):
   q=0.
   for j in range(K):q+=A[i,j]*B[t+1,j]*be[t+1,j]
   be[t,i]=q/c[t+1]
 g=np.empty((n,K))
 for t in range(n):
  s=0.
  for k in range(K):g[t,k]=al[t,k]*be[t,k];s+=g[t,k]
  for k in range(K):g[t,k]/=s
 xi=np.zeros((K,K))
 for t in range(n-1):
  den=0.
  for i in range(K):
   for j in range(K):den+=al[t,i]*A[i,j]*B[t+1,j]*be[t+1,j]
  for i in range(K):
   for j in range(K):xi[i,j]+=al[t,i]*A[i,j]*B[t+1,j]*be[t+1,j]/den
 ll=0.
 for t in range(n):ll+=math.log(c[t])+mx[t]
 return ll,g,xi
@njit
def em(E,A,pi,iters):
 last=-1e300
 for it in range(iters):
  ll,g,xi=fb(E,A,pi)
  for i in range(V):
   s=0.
   for j in range(V):A[i,j]=xi[i,j]+.15;s+=A[i,j]
   for j in range(V):A[i,j]/=s
  s=0.
  for k in range(V):pi[k]=g[0,k]+.1;s+=pi[k]
  for k in range(V):pi[k]/=s
  if it>12 and abs(ll-last)<1e-5:break
  last=ll
 return A,pi
def fit(Ef,Ev,seed,R=5):
 best=None
 for r in range(R):
  q=np.random.default_rng(seed+173*r);A=q.gamma(1,1,(V,V));A[np.arange(V),np.arange(V)]+=1;A=A/A.sum(1,keepdims=True);pi=q.dirichlet(np.ones(V))
  A,pi=em(Ef,A,pi,70);lv=fb(Ev,A,pi)[0]
  if best is None or lv>best[0]:best=(lv,A.copy(),pi.copy())
 return best
def cnmi(z,p):
 return float(np.mean([normalized_mutual_info_score(z[SIG[z]==s],p[SIG[z]==s]) for s in range(NSIG)]))
def auc(z,G,seed):
 q=np.random.default_rng(seed);Y=[];S=[]
 for s in range(NSIG):
  ix=np.where(SIG[z]==s)[0]
  for _ in range(600):
   a,b=q.choice(ix,2,replace=False);Y.append(int(z[a]==z[b]));S.append(float(G[a]@G[b]))
 return float(roc_auc_score(Y,S))
def ev(E,z,A,pi,seed):
 ll,G,_=fb(E,A,pi);p=G.argmax(1)
 return {"ll":ll,"nmi":float(normalized_mutual_info_score(z,p)),"collision":cnmi(z,p),"auc":auc(z,G,seed)}
def one(fam,seed):
 rng=np.random.default_rng(seed);U,Ve,Vr=m["encoder"](seed);R=[]
 for sec in range(NSEC):
  z,At,pt=m["source_sequence"](fam,sec,SECLEN,rng);obs,sig=m["render"](z,U,Ve,Vr,rng);EC=m["control_E"](sig);EF=m["form_E"](obs,U,Ve,Vr)
  for ch,E in (("CONTROL",EC),("FORM",EF)):
   lv,A,pi=fit(E[:NFIT],E[NFIT:NFIT+NVAL],seed+1000*sec+(50000 if ch=="FORM" else 0))
   Etr=E[:NFIT+NVAL];A,pi=em(Etr,A.copy(),pi.copy(),50);zt=z[NFIT+NVAL:];Et=E[NFIT+NVAL:]
   R.append({"section":sec,"channel":ch,"val_ll":lv,"blind":ev(Et,zt,A,pi,seed+sec),"oracle":ev(Et,zt,At,pt,seed+sec+99)})
 def md(ch,w,k):return float(np.median([x[w][k] for x in R if x["channel"]==ch]))
 return {"family":fam,"seed":seed,"control":{"blind_collision":md("CONTROL","blind","collision"),"oracle_collision":md("CONTROL","oracle","collision"),"blind_auc":md("CONTROL","blind","auc"),"oracle_auc":md("CONTROL","oracle","auc")},"form":{"blind_collision":md("FORM","blind","collision"),"oracle_collision":md("FORM","oracle","collision"),"blind_auc":md("FORM","blind","auc"),"oracle_auc":md("FORM","oracle","auc")},"sections":R}
if __name__=="__main__":
 out=[]
 for fam in ("LANG","NOTATION","TABLE"):
  for seed in (20265001,20265002,20265003):
   r=one(fam,seed);out.append(r);print("L2_REP="+json.dumps(r,separators=(",",":")),flush=True)
 sm={}
 for fam in ("LANG","NOTATION","TABLE"):
  rr=[x for x in out if x["family"]==fam];sm[fam]={}
  for ch in ("control","form"):
   for k in ("blind_collision","oracle_collision","blind_auc","oracle_auc"):sm[fam][ch+"_"+k]=float(np.median([x[ch][k] for x in rr]))
 print("L2_FINAL="+json.dumps(sm,separators=(",",":")),flush=True)
