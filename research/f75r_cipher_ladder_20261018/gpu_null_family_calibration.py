#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5,<3", "Unidecode>=1.3,<2"]
# ///
"""
F75R-NULL3 / 2026-10-19: family-specific instrument calibration for one-null ladder.
No Voynich inference unless registered no-null false positive and true-null power qualify.
This file freezes synthetic train/test construction and matched fit/heldout score.
Source text only from the historical-unmatched UD corpus; NEVER claim historical decipherment.
Randomization is deterministic; all trials atomically pickled in /tmp, summary printed to logs.
"""
import argparse,collections,hashlib,json,math,os,pickle,random,sys,urllib.request
from pathlib import Path
import torch
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
MURL=BASE+"257a0d1198c3e8c0b416fda44255a0c8675424aa/research/f75r_cipher_ladder_20261018/gpu_mono_solver.py"
NURL=BASE+"510abaf6db45ee0384b9874bcb1649a5c0091be1/research/f75r_cipher_ladder_20261018/gpu_single_null.py"
m={"__name__":"monomod"};exec(compile(urllib.request.urlopen(MURL,timeout=90).read().decode(),MURL,"exec"),m)
v={"__name__":"nullmod"};exec(compile(urllib.request.urlopen(NURL,timeout=90).read().decode(),NURL,"exec"),v)
get=m.__getitem__; emit=get("emit");A=get("A");SPACE=get("SPACE");conllu=get("conllu");make_model=get("make_model");getvms=get("getvms");glyphs=get("glyphs");mean_score=get("mean_score");anneal=get("anneal")
encode=v["encode_existing"];joined=v["joined"];renumber=v["renumber"];solve=v["solve_null_candidates"];channel_obj=v["channel_obj"];chars_only=v["chars_only"]
OUT=Path("/tmp/vms_null3");OUT.mkdir(exist_ok=True)
def write_atomic(path,data):
 tmp=path.with_suffix(".tmp")
 with tmp.open("wb") as f:pickle.dump(data,f,protocol=5);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def corpus_control(dev,rate,seed,ntrain=352,nhold=465):
 rng=random.Random(seed)
 letters=[c for c,_ in collections.Counter("".join(dev)).most_common(19 if rate>0 else 20)]
 allowed=set(letters)
 # Require alphabet coverage while retaining natural corpus token shapes; sample independent dev indices.
 cand=[w for w in dev if len(w)>1 and set(w).issubset(allowed)]
 if len(cand)<1100:raise RuntimeError(("not enough synthetic plaintext",len(cand)))
 rng.shuffle(cand)
 tr=cand[:ntrain];hold=cand[ntrain:ntrain+nhold]
 enc_idx=list(range(len(letters)));rng.shuffle(enc_idx)
 mapping={letter:enc_idx[i] for i,letter in enumerate(letters)}
 nullidx=len(letters) if rate>0 else None
 def crypt(words):
  seq=[]
  for word in words:
   w=[]
   for c in word:
    w.append(mapping[c])
    if nullidx is not None and rng.random()<rate:w.append(nullidx)
   seq.append(w)
  return seq
 return crypt(tr),crypt(hold),{"rate":rate,"seed":seed,"letters":"".join(letters),
  "true_null":nullidx,"train_words":len(tr),"heldout_words":len(hold)}
def score_key(key,wordseq,model,nullsym=None):
 arr=renumber(wordseq,nullsym) if nullsym is not None else wordseq
 arr=[w for w in arr if w]
 enc=joined(arr)
 ids=torch.tensor(enc,device=model.device,dtype=torch.long)
 extra=[i for i in range(26) if i not in key]
 full=torch.tensor(key+extra,device=model.device,dtype=torch.long)[None,:]
 ll=float(mean_score(full,ids,model).item())
 nc=chars_only(wordseq)
 return channel_obj(ll,len(enc),nc,nc-chars_only(arr),nullsym is not None)/nc
def trial(words,hold,model,seed,steps,pop):
 torch.manual_seed(seed)
 n=max(max(w) for w in words)+1
 if n>26:raise RuntimeError(("alphabet",n))
 x=solve(words,n,model,seed+117,steps,pop)
 mono=x["mono"];null=x["best"]
 mhold=score_key(mono["key"][:n],hold,model)
 nhold=score_key(null["mapping"],hold,model,null["null_symbol"])
 out={"selected_null":null["null_symbol"],"selected_deleted_rate":null["deletion_rate"],
 "mono_fit_per_glyph":mono["per_cipher_glyph"],"null_fit_per_glyph":null["score_per_cipher_glyph"],
 "train_gain":null["score_delta_vs_mono"],"mono_holdout":mhold,"null_holdout":nhold,
 "heldout_gain":nhold-mhold,
 "fit_key":mono["key"][:n],
 "null_key":null["mapping"]}
 return out
def run(lang,nreps,steps,pop,seed,mode):
 if not torch.cuda.is_available():raise RuntimeError("GPU_REQUIRED")
 torch.manual_seed(seed)
 model_words,source=conllu(lang,"train")
 hold_words,source_hold=conllu(lang,"dev")
 model=make_model(model_words[:90000]).to("cuda:0")
 emit("NULL3_AUDIT",{"lang":lang,"torch":torch.__version__,"device":torch.cuda.get_device_name(0),
   "mode":mode,"steps":steps,"pop":pop,"reps":nreps,"seed":seed,
   "training_source":source,"heldout_source":source_hold,"solver_sha":hashlib.sha256(urllib.request.urlopen(NURL).read()).hexdigest(),
   "notes":"UD sources are not matched 15th century witnesses; no language conclusion."})
 if mode=="synthetic":
  records=[];rates=[0.,.05,.10,.15]
  for rate in rates:
   for k in range(nreps):
    ii=seed+int(rate*10000)*100+k*101
    tr,ho,meta=corpus_control(hold_words,rate,ii)
    r=trial(tr,ho,model,ii,steps,pop)
    obs={**meta,**{k:v for k,v in r.items() if k not in ("fit_key","null_key")}}
    obs["correct_null"]=bool(meta["true_null"] is not None and meta["true_null"]==r["selected_null"])
    records.append(obs)
    write_atomic(OUT/f"{lang}_synthetic_{int(rate*100)}_{k}.pkl",obs)
    emit("NULL3_TRIAL",obs)
  neg=[x["heldout_gain"] for x in records if x["rate"]==0]
  mn=sum(neg)/len(neg)
  sd=(sum((x-mn)**2 for x in neg)/(len(neg)-1))**.5 if len(neg)>1 else None
  metrics={}
  for rate in rates:
   a=[x for x in records if x["rate"]==rate]
   metrics[str(rate)]={"n":len(a),"mean_gain":sum(x["heldout_gain"] for x in a)/len(a),
     "mean_minus_no_null":sum(x["heldout_gain"] for x in a)/len(a)-mn,
     "correct_null":sum(x["correct_null"] for x in a) if rate else None,
     "gain_positive":sum(x["heldout_gain"]>0 for x in a)}
  emit("NULL3_SYNTH_RESULT",{"lang":lang,"no_null_mean":mn,"no_null_sd":sd,"metric":"heldout log score gain per cipher glyph",
   "nreps":nreps,"by_rate":metrics,"warning":"No-null calibration only; not enough samples to fix a 5% FPR gate unless >=20 negatives."})
 elif mode=="real":
  vms=getvms("f75r");hold=getvms("f76r")
  primary=sorted(set(g for w in vms for g in glyphs(w)));lookup={g:i for i,g in enumerate(primary)}
  train=encode(vms,primary);accept=[w for w in hold if set(glyphs(w)).issubset(lookup)]
  test=encode(accept,primary);records=[]
  for k in range(nreps):
   ss=seed+k*101
   r=trial(train,test,model,ss,steps,pop)
   obs={**{k:v for k,v in r.items() if k not in ("fit_key","null_key")},"lang":lang,
    "seed":ss,"glyph":primary[r["selected_null"]],"train_folio":"f75r","test_folio":"f76r",
    "train_words":len(vms),"test_words_scored":len(accept)}
   records.append(obs)
   write_atomic(OUT/f"{lang}_real_{k}.pkl",obs)
   emit("NULL3_REAL_TRIAL",obs)
  emit("NULL3_REAL_RESULT",{"lang":lang,"nreps":nreps,
   "gains":[r["heldout_gain"] for r in records],
   "selected_glyphs":[r["glyph"] for r in records],
   "warning":"Do not compare across languages or promote without matched synthetic false positive/power audit."})
 else:raise ValueError(mode)
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--lang",choices=["la","it"],required=True)
 p.add_argument("--mode",choices=["synthetic","real"],required=True)
 p.add_argument("--reps",type=int,default=4)
 p.add_argument("--steps",type=int,default=450)
 p.add_argument("--pop",type=int,default=128)
 p.add_argument("--seed",type=int,default=1931)
 a=p.parse_args();run(a.lang,a.reps,a.steps,a.pop,a.seed,a.mode)
