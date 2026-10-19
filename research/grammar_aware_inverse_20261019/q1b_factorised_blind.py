#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<3", "scipy>=1.13,<2", "scikit-learn>=1.5,<2"]
# ///
"""GAI-Q1B: independently train emission-only SELECT categories and source dynamics.
Blind, synthetic-only. No source labels or family used for model selection.
"""
import argparse,math,time,json,urllib.request
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score,roc_auc_score
ROOT="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
Q1_URL=ROOT+"47c8ad7d107c24b48e07b4385e2582a8d5f6f899/research/grammar_aware_inverse_20261019/q1_blind_category.py"
u={"__name__":"q1mod"};exec(compile(urllib.request.urlopen(Q1_URL,timeout=120).read().decode(),Q1_URL,"exec"),u)
q=u
K=8
def emit(k,o):print(k+"="+json.dumps(o,separators=(",",":"),sort_keys=True,default=str),flush=True)
def fit_transition_only(train_x,em_model,prior):
 E=q["emits"](train_x,em_model["b0"],em_model["br"])
 pi=em_model["pi"].copy()
 _,iidgamma,_=q["fb"](E,np.tile(pi,(K,1)),pi,False)
 counts=iidgamma[:-1].T@iidgamma[1:]
 A=counts+prior;A/=A.sum(axis=1,keepdims=True)
 for it in range(10):
  ll,gamma,xi=q["fb"](E,A,pi,True)
  A=xi+prior;A/=A.sum(axis=1,keepdims=True)
  pi=gamma[0]+prior;pi/=pi.sum()
 return {"A":A,"pi":pi,"prior":prior}

def seq_score(em_model,dynamics,x):
 E=q["emits"](x,em_model["b0"],em_model["br"])
 return q["fb"](E,dynamics["A"],dynamics["pi"],True)

def pairauc(posterior,true_labels,seed,n=12000):
 rng=np.random.default_rng(seed)
 a=rng.integers(len(true_labels),size=n)
 b=rng.integers(len(true_labels),size=n)
 ix=a!=b;a=a[ix];b=b[ix]
 y=(true_labels[a]==true_labels[b]).astype(int)
 x=np.sum(posterior[a]*posterior[b],axis=1)
 return float(roc_auc_score(y,x)) if len(set(y))==2 else None

def run(family,nsource,seed,nt,nv,nte):
 rng=np.random.default_rng(seed)
 pi,A,_=q["z"]["build_source"](nsource,"LANG" if family=="NONE" else family,rng)
 gmap=(rng.permutation(nsource)%K).astype(int)
 U,Ve,Vr=q["z"]["frozen_controller"](rng)
 Z=[];G=[];X=[]
 for n in (nt,nv,nte):
  z=q["z"]["sample_sequence"](A,pi,n,rng)
  obs=q["z"]["render"](z,gmap,U,Ve,Vr,rng,none=family=="NONE")
  Z.append(z);G.append(gmap[z]);X.append(q["tokenize_features"](obs))
 # FIT EMISSIONS WITH NO TIME DATA AND NO TRUTH
 candidates=[]
 for reg in (.3,1.):
  for start in range(3):
   m=q["fit_one"](X[0],"IID",reg,seed+1000+17*start+int(10*reg))
   ll=q["score_model"](m,X[1])[0]
   candidates.append((ll,m))
 val,em_model=max(candidates,key=lambda x:x[0])
 # Freeze emissions. Estimate temporal model separately, with identical emissions.
 dyn_c=[]
 for prior in (.5,3.):
  d=fit_transition_only(X[0],em_model,prior)
  valid_ll=seq_score(em_model,d,X[1])[0]
  dyn_c.append((valid_ll,d))
 best_dyn_val,best_dyn=max(dyn_c,key=lambda x:x[0])
 selected_seq=bool(best_dyn_val>val)
 # Test only after models selected.
 iid_ll,posterior,_=q["score_model"](em_model,X[2])
 hmm_ll,hmm_gamma,_=seq_score(em_model,best_dyn,X[2])
 # SOURCE TRUTH is first referenced here.
 truth_g=G[2];truth_z=Z[2]
 pred=posterior.argmax(axis=1)
 seq_pred=hmm_gamma.argmax(axis=1)
 iid_nmi=float(normalized_mutual_info_score(truth_g,pred))
 seq_nmi=float(normalized_mutual_info_score(truth_g,seq_pred))
 ci_auc=pairauc(posterior,truth_g,seed+777)
 item_auc=pairauc(posterior,truth_z,seed+777)
 gain=(hmm_ll-iid_ll)/nte/math.log(2)
 return {"family":family,"seed":seed,"source_types":nsource,"n_train":nt,"n_val":nv,"n_test":nte,
  "emission_val_nats_token":val/nv,"dyn_val_nats_token":best_dyn_val/nv,"selected_seq":selected_seq,
  "selected_emission_l2":em_model["l2"],"selected_emission_seed":em_model["seed"],
  "selected_transition_prior":best_dyn["prior"],
  "test_iid_class_nmi":iid_nmi,"test_hmm_class_nmi_diagnostic":seq_nmi,
  "test_iid_class_ari":float(adjusted_rand_score(truth_g,pred)),
  "test_same_class_auc":ci_auc,"test_same_source_item_auc_exploratory":item_auc,
  "test_seq_gain_bits_per_token":gain,
  "test_iid_ll_nats_token":iid_ll/nte,"test_hmm_ll_nats_token":hmm_ll/nte,
  "calibration":"Synthetic only. True source labels revealed only after model selection; no real VMS decoding."}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--n",type=int,choices=[32,64],default=32)
 p.add_argument("--train",type=int,default=1200)
 p.add_argument("--val",type=int,default=400)
 p.add_argument("--test",type=int,default=400)
 p.add_argument("--seeds",type=int,default=2)
 p.add_argument("--seed",type=int,default=20267000)
 p.add_argument("--families",nargs="+",default=["LANG","NOTATION","TABLE","NONE"])
 args=p.parse_args()
 emit("GAI_Q1B_AUDIT",{"engine":"IID exact FORM emission K8 then emission-frozen HMM transition",
  "source_q1":Q1_URL,"real_voynich_not_scored":True,"source_truth_not_fit":True,
  "seed_start":args.seed,"n_source":args.n,"cases_per_family":args.seeds,
  "train":args.train,"val":args.val,"test":args.test,
  "families":args.families,"pre_registered_gate":"Q1B_FACTORISED_BLIND_PREREG.md"})
 t0=time.monotonic();results=[]
 for family in args.families:
  for i in range(args.seeds):
   seed=args.seed+i+q["FAMILY_IDS"][family]*10000
   record=run(family,args.n,seed,args.train,args.val,args.test)
   results.append(record);emit("GAI_Q1B_CASE",record)
 summary=[]
 for family in args.families:
  subset=[x for x in results if x["family"]==family]
  summary.append({"family":family,"n":len(subset),
   "median_iid_class_nmi":float(np.median([x["test_iid_class_nmi"] for x in subset])),
   "median_iid_class_auc":float(np.median([x["test_same_class_auc"] for x in subset])),
   "median_same_item_auc":float(np.median([x["test_same_source_item_auc_exploratory"] for x in subset])),
   "median_hmm_extra_bits":float(np.median([x["test_seq_gain_bits_per_token"] for x in subset])),
   "class_gate_fraction":float(np.mean([x["test_iid_class_nmi"]>=.6 for x in subset])),
   "false_seq_alarm_fraction":float(np.mean([x["test_seq_gain_bits_per_token"]>=.02 for x in subset]))})
 emit("GAI_Q1B_FINAL",{"schema":"GAI-Q1B-SYNTHETIC-20261019","complete":True,
   "results":results,"summary":summary,"runtime_sec":time.monotonic()-t0})
if __name__=="__main__":main()
