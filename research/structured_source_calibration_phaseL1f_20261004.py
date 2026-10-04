#!/usr/bin/env python3
import json,urllib.request,numpy as np
from hmmlearn.hmm import CategoricalHMM
from sklearn.metrics import normalized_mutual_info_score,roc_auc_score
URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/0d818e9f00012661077c83b27416ad0d2c99d581/research/structured_source_calibration_phaseL1_20261004.py"
m={"__name__":"L1"};exec(compile(urllib.request.urlopen(URL).read().decode(),URL,"exec"),m)
V,NSIG,NSEC,SECLEN,NFIT,NVAL=[m[x] for x in ("V","NSIG","NSEC","SECLEN","NFIT","NVAL")];SIG=m["SIG_OF"]
EM=np.full((V,NSIG),.001/(NSIG-1));EM[np.arange(V),SIG]=.999;EM/=EM.sum(1,keepdims=True)
def fit(a,b,seed,R=4):
 best=None
 for r in range(R):
  g=np.random.default_rng(seed+173*r);h=CategoricalHMM(n_components=V,n_iter=80,tol=1e-4,implementation="log",init_params="",params="st",random_state=seed+173*r);h.n_features=NSIG;h.startprob_=g.dirichlet(np.ones(V));A=g.gamma(1,1,(V,V));A[np.arange(V),np.arange(V)]+=1;h.transmat_=A/A.sum(1,keepdims=True);h.emissionprob_=EM.copy();h.fit(a[:,None]);lv=h.score(b[:,None])
  if best is None or lv>best[0]:best=(lv,h)
 return best
def cnmi(z,p):
 return float(np.mean([normalized_mutual_info_score(z[SIG[z]==s],p[SIG[z]==s]) for s in range(NSIG)]))
def auc(z,G,seed):
 g=np.random.default_rng(seed);Y=[];S=[]
 for s in range(NSIG):
  ix=np.where(SIG[z]==s)[0]
  for _ in range(500):
   a,b=g.choice(ix,2,replace=False);Y.append(int(z[a]==z[b]));S.append(float(G[a]@G[b]))
 return float(roc_auc_score(Y,S))
def one(fam,seed):
 rng=np.random.default_rng(seed);U,Ve,Vr=m["encoder"](seed);rec=[];secs=[];ts=[];fs=[]
 for sec in range(NSEC):
  z,A,pi=m["source_sequence"](fam,sec,SECLEN,rng);obs,sig=m["render"](z,U,Ve,Vr,rng);E=m["form_E"](obs,U,Ve,Vr);f=E[:,:NSIG].argmax(1);secs += [sec]*SECLEN;ts += sig.tolist();fs += f.tolist()
  for ch,o in (("CONTROL",sig),("FORM",f)):
   lv,h=fit(o[:NFIT],o[NFIT:NFIT+NVAL],seed+1000*sec+(50000 if ch=="FORM" else 0));zt=z[NFIT+NVAL:];te=o[NFIT+NVAL:,None];p=h.predict(te);G=h.predict_proba(te);rec.append({"section":sec,"channel":ch,"nmi":float(normalized_mutual_info_score(zt,p)),"collision":cnmi(zt,p),"auc":auc(zt,G,seed+sec)})
 def md(ch,k):return float(np.median([x[k] for x in rec if x["channel"]==ch]))
 secs=np.array(secs);ts=np.array(ts);fs=np.array(fs)
 return {"family":fam,"seed":seed,"sec_mi":float(normalized_mutual_info_score(secs,ts)),"sec_mi_form":float(normalized_mutual_info_score(secs,fs)),"sig_acc":float(np.mean(ts==fs)),"control":{"nmi":md("CONTROL","nmi"),"collision":md("CONTROL","collision"),"auc":md("CONTROL","auc")},"form":{"nmi":md("FORM","nmi"),"collision":md("FORM","collision"),"auc":md("FORM","auc")}}
if __name__=="__main__":
 out=[]
 for fam in ("LANG","NOTATION","TABLE"):
  for seed in (20264001,20264002,20264003):
   r=one(fam,seed);out.append(r);print("L1F_REP="+json.dumps(r,separators=(",",":")),flush=True)
 sm={}
 for fam in ("LANG","NOTATION","TABLE"):
  rr=[x for x in out if x["family"]==fam];sm[fam]={k:float(np.median([x[k] for x in rr])) for k in ("sec_mi","sec_mi_form","sig_acc")}
  for ch in ("control","form"):
   for k in ("nmi","collision","auc"):sm[fam][ch+"_"+k]=float(np.median([x[ch][k] for x in rr]))
 print("L1F_FINAL="+json.dumps(sm,separators=(",",":")),flush=True)
