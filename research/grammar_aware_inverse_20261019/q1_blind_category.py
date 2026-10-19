#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<3", "scipy>=1.13,<2", "scikit-learn>=1.5,<2"]
# ///
"""GAI-Q1: BLIND category recovery under a frozen FORM socket.
Preregistration: research/grammar_aware_inverse_20261019/Q1_BLIND_CATEGORY_PREREG.md.
No direct Voynich decoding. Source truth is NEVER used in training or validation.
"""
import argparse,json,math,time,urllib.request
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score,roc_auc_score
ROOT="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
Q0_URL=ROOT+"163118f760c697ef7688643ab4b7af16995d8c6e/research/grammar_aware_inverse_20261019/q0_frozen_form_oracle.py"
z={"__name__":"q0_module"}
exec(compile(urllib.request.urlopen(Q0_URL,timeout=180).read().decode(),Q0_URL,"exec"),z)
k0=z["k0"]
BASE_START=np.asarray(k0["P_START_CLASS"],float)
BASE_ROUTE=np.asarray(k0["P_CONT"],float)
LEGAL=np.asarray(k0["LEGAL"],bool)
assert LEGAL.shape==(8,12) and BASE_ROUTE.shape==(8,12)
K=8;D=12
LB0=np.log(np.maximum(BASE_START,1e-300))
LOGBR=np.where(LEGAL,np.log(np.maximum(BASE_ROUTE,1e-300)),-np.inf)
FAMILY_IDS={"LANG":0,"NOTATION":1,"TABLE":2,"NONE":3}

def emit(tag,obj):
 print(tag+"="+json.dumps(obj,default=float,separators=(",",":"),sort_keys=True),flush=True)

def tokenize_features(obs,smoke_check=True):
 n=len(obs);start=np.zeros((n,D),float);route=np.zeros((n,8,D),float);const=np.zeros(n)
 PP=k0["PCLASS"];G=k0["G_OF"];START_IN=k0["START_IN"]
 for j,tok in enumerate(obs):
  c0=int(PP[tok[0]]);start[j,c0]=1.
  cp=k0["P_START_PIECE"][c0,tok[0]]
  if cp<=0:raise RuntimeError(("invalid start piece",j,tok))
  cons=math.log(cp)
  inc=START_IN
  for dep,pp in enumerate(tok):
   ps=k0["P_STOP"][pp,min(dep,3),inc]
   if dep==len(tok)-1:
    cons+=math.log(ps);break
   cons+=math.log1p(-ps)
   nc=int(PP[tok[dep+1]]);g=G[int(PP[pp])]
   if not LEGAL[g,nc]:raise RuntimeError(("FORM legality violated",j,g,nc))
   pc=k0["P_NEXT"][g,nc,tok[dep+1]]
   if pc<=0:raise RuntimeError(("invalid next piece",j,g,nc))
   cons+=math.log(pc)
   route[j,g,nc]+=1
   inc=int(PP[pp])
  const[j]=cons
 if smoke_check:
  # Assert exact normalized reference likelihood under no SELECT bias, no approximation.
  U=np.zeros((1,2));Ve=np.ones((2,D));Vr=np.ones((2,D))
  for j in range(min(12,n)):
   ll=k0["token_loglik"](obs[j],0,U,Ve,Vr)
   calc=const[j]+float(start[j]@LB0)+float((route[j]*np.where(LEGAL,np.log(np.maximum(BASE_ROUTE,1e-300)),0)).sum())
   if abs(ll-calc)>1e-8:raise AssertionError(("incorrect_form_likelihood",j,ll,calc))
 return {"s":start,"r":route,"c":const}

def emission_params(b0,br):
 # b0 and br are shape K x 12
 L0=LB0[None,:]+b0
 L0-=logsumexp(L0,axis=1,keepdims=True)
 Lr=LOGBR[None,:,:]+br[:,None,:]
 Lr-=logsumexp(Lr,axis=2,keepdims=True)
 return L0,Lr

def emits(X,b0,br):
 L0,Lr=emission_params(b0,br)
 return X["s"]@L0.T+np.einsum("ngd,kgd->nk",X["r"],np.where(np.isfinite(Lr),Lr,0.0))+X["c"][:,None]

def fb(E,A,pi,transition):
 n,k=E.shape
 Lp=np.log(np.maximum(pi,1e-300))
 LA=np.log(np.maximum(A,1e-300))
 alpha=np.empty((n,k));scale=np.empty(n)
 a=Lp+E[0];scale[0]=logsumexp(a);alpha[0]=a-scale[0]
 for t in range(1,n):
  if transition:
   a=logsumexp(alpha[t-1,:,None]+LA,axis=0)+E[t]
  else:
   a=Lp+E[t]
  scale[t]=logsumexp(a);alpha[t]=a-scale[t]
 beta=np.zeros((n,k))
 if transition:
  for t in range(n-2,-1,-1):
   beta[t]=logsumexp(LA+E[t+1,None,:]+beta[t+1,None,:],axis=1)-scale[t+1]
 logits=alpha+beta
 logits-=logsumexp(logits,axis=1,keepdims=True)
 gamma=np.exp(logits)
 xi=np.zeros((k,k),float)
 if transition:
  for t in range(n-1):
   v=alpha[t,:,None]+LA+E[t+1,None,:]+beta[t+1,None,:]
   v-=logsumexp(v)
   xi+=np.exp(v)
 return float(scale.sum()),gamma,xi

def fit_emission(X,gamma,b0,br,l2):
 ce=gamma.T@X["s"]
 cr=np.einsum("nk,ngd->kgd",gamma,X["r"])
 new0=np.empty_like(b0);newr=np.empty_like(br)
 for state in range(K):
  total=ce[state].sum()
  def objective(x):
   bb=x[:D];rr=x[D:]
   le=LB0+bb;le-=logsumexp(le)
   lr=LOGBR+rr[None,:];lr-=logsumexp(lr,axis=1,keepdims=True)
   valid=np.where(LEGAL,lr,0.)
   start_prob=np.exp(le)
   rout_prob=np.exp(np.where(LEGAL,lr,-np.inf))
   ce0=ce[state]
   cr0=cr[state]
   loss=-(ce0@le)-np.sum(cr0*valid)+.5*l2*float(x@x)
   de=total*start_prob-ce0 + l2*bb
   dr=(cr0.sum(axis=1)[:,None]*rout_prob-cr0).sum(axis=0)+l2*rr
   return float(loss),np.r_[de,dr]
  o=minimize(objective,np.r_[b0[state],br[state]],method="L-BFGS-B",jac=True,
             options={"maxiter":45,"ftol":1e-9})
  new0[state]=o.x[:D];newr[state]=o.x[D:]
 return new0,newr

def initial_gamma(X,seed):
 arr=np.concatenate((X["s"]*2.0,X["r"].reshape(len(X["s"]),-1)),axis=1)
 rng=np.random.default_rng(seed)
 arr=arr+rng.normal(0,.02,arr.shape)
 lab=KMeans(n_clusters=K,n_init=4,max_iter=100,random_state=seed%2147483647).fit_predict(arr)
 gamma=np.full((len(lab),K),.04/(K-1))
 gamma[np.arange(len(lab)),lab]=.96
 return gamma

def fit_one(X,mode,l2,seed,iters=9):
 n=len(X["s"]);Gamma=initial_gamma(X,seed)
 b0=np.zeros((K,D));br=np.zeros((K,D))
 b0,br=fit_emission(X,Gamma,b0,br,l2)
 pi=np.ones(K)/K
 A=np.full((K,K),1./K)
 last=-1e300
 for it in range(iters):
  E=emits(X,b0,br)
  ll,gamma,xi=fb(E,A,pi,mode=="HMM")
  b0,br=fit_emission(X,gamma,b0,br,l2)
  if mode=="HMM":
   A=(xi+.5);A/=A.sum(axis=1,keepdims=True)
   pi=(gamma[0]+.5);pi/=pi.sum()
  else:
   pi=(gamma.sum(axis=0)+.5);pi/=pi.sum()
   A=np.tile(pi,(K,1))
  if not np.isfinite(ll):raise RuntimeError("nonfinite likelihood")
  if it>=5 and ll-last <1e-5:break
  last=ll
 return {"mode":mode,"l2":l2,"seed":seed,"b0":b0,"br":br,"A":A,"pi":pi,"train_ll":ll,"em_iters":it+1}

def score_model(model,X):
 E=emits(X,model["b0"],model["br"])
 return fb(E,model["A"],model["pi"],model["mode"]=="HMM")

def solve_family(family,nsource,seed,args):
 # Training and test have independent chain resets, fixed hidden generative instance.
 rng=np.random.default_rng(seed)
 srcfam=family if family!="NONE" else "LANG"
 pi,A,_=z["build_source"](nsource,srcfam,rng)
 controller=(rng.permutation(nsource)%K).astype(int)
 U,Ve,Vr=z["frozen_controller"](rng)
 g=[];X=[]
 for count in [args.train,args.val,args.test]:
  x=z["sample_sequence"](A,pi,count,rng)
  tokens=z["render"](x,controller,U,Ve,Vr,rng,none=family=="NONE")
  g.append(controller[x]);X.append(tokenize_features(tokens))
 # Source truth unreferenced until all blind model choices and test posteriors fixed.
 choices=[]
 for mode in ("IID","HMM"):
  for l2 in (.3,1.):
   for restart in range(3):
    model=fit_one(X[0],mode,l2,seed+1000+17*restart+int(l2*10))
    val_ll=score_model(model,X[1])[0]
    choices.append((val_ll,args.val,model))
 selected=max(choices,key=lambda x:x[0])
 bymode={}
 for mode in ("IID","HMM"):
  cand=[x for x in choices if x[2]["mode"]==mode]
  best=max(cand,key=lambda x:x[0])
  bymode[mode]=best[2]
 # Untouched TEST. Only now refer to hidden controller truth.
 model=selected[2];test_ll,posterior,_=score_model(model,X[2])
 true_g=g[2];pred=posterior.argmax(axis=1)
 nmi=float(normalized_mutual_info_score(true_g,pred))
 ari=float(adjusted_rand_score(true_g,pred))
 ll_h=score_model(bymode["HMM"],X[2])[0];ll_i=score_model(bymode["IID"],X[2])[0]
 gain=(ll_h-ll_i)/args.test/math.log(2)
 rng_pair=np.random.default_rng(seed+777)
 ind1=rng_pair.integers(args.test,size=6000)
 ind2=rng_pair.integers(args.test,size=6000)
 keep=ind1!=ind2;ind1=ind1[keep];ind2=ind2[keep]
 scores=np.sum(posterior[ind1]*posterior[ind2],axis=1)
 matches=(true_g[ind1]==true_g[ind2]).astype(int)
 auc=float(roc_auc_score(matches,scores)) if len(np.unique(matches))==2 else None
 return {"seed":seed,"n_source":nsource,"family":family,"n_train":args.train,"n_val":args.val,
   "n_test":args.test,"blind_selected":model["mode"],"selected_l2":model["l2"],"selected_restart_seed":model["seed"],
   "selected_val_nats_token":selected[0]/args.val,"hmm_minus_iid_test_bits_token":gain,
   "test_source_select_group_nmi":nmi,"test_group_ari":ari,"test_group_pair_auc":auc,
   "test_loglik_selected_nats_token":test_ll/args.test,
   "test_truth_group_distribution":np.bincount(true_g,minlength=K).tolist(),
   "em_iters":model["em_iters"],"models_tried":len(choices),
   "no_truth_selected_models":True,
   "limits":"Synthetic FORM only; no CONNECT/LINE_ENTRY. Blind latent 8-class model; no real Voynich."}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--n",type=int,choices=[32,64],default=32)
 p.add_argument("--train",type=int,default=1200)
 p.add_argument("--val",type=int,default=400)
 p.add_argument("--test",type=int,default=400)
 p.add_argument("--seeds",type=int,default=2)
 p.add_argument("--seed",type=int,default=20265000)
 p.add_argument("--families",nargs="+",default=["LANG","NOTATION","TABLE","NONE"])
 args=p.parse_args()
 emit("GAI_Q1_AUDIT",{"schema":"GAI-Q1-20261019","source":"frozen renderer", "n_source":args.n,
  "train":args.train,"val":args.val,"test":args.test,"n_seeds":args.seeds,
  "families":args.families,"candidate_bank":"mode IID/HMM x L2 .3/1.0 x 3 restarts",
  "oracle_not_used_in_training":True,"true_model_params_not_given":True})
 out=[];start=time.monotonic()
 for f in args.families:
  for i in range(args.seeds):
   seed=args.seed+i+FAMILY_IDS[f]*10000
   result=solve_family(f,args.n,seed,args);out.append(result);emit("GAI_Q1_CASE",result)
 summ=[]
 for f in args.families:
  rows=[x for x in out if x["family"]==f]
  summ.append({"family":f,"n":len(rows),
   "median_nmi":float(np.median([x["test_source_select_group_nmi"] for x in rows])),
   "median_pair_auc":float(np.median([x["test_group_pair_auc"] for x in rows])),
   "median_hmm_gain":float(np.median([x["hmm_minus_iid_test_bits_token"] for x in rows])),
   "pct_nmi_ge_06":float(np.mean([x["test_source_select_group_nmi"]>=.6 for x in rows])),
   "pct_hmm_gain_ge_002":float(np.mean([x["hmm_minus_iid_test_bits_token"]>=.02 for x in rows]))})
 emit("GAI_Q1_FINAL",{"complete":True,"schema":"GAI-Q1-SYNTHETIC-BLIND-20261019",
  "results":out,"summary":summ,"elapsed_s":time.monotonic()-start,
  "interpretation":"All model selection on synthetic TRAIN/VAL. Real Voynich never scored; blind source category gate separate."})
if __name__=="__main__":main()
