#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5,<3", "Unidecode>=1.3,<2"]
# ///
"""F75R cipher ladder rung 1B: fixed-key, refitted-key and recurrence-preserving null bounding."""
import collections,hashlib,json,math,os,pickle,random,re,sys,urllib.request
from pathlib import Path
import torch
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
CODE=BASE+"257a0d1198c3e8c0b416fda44255a0c8675424aa/research/f75r_cipher_ladder_20261018/gpu_mono_solver.py"
RESULTS=BASE+"7ffdfbc22ade63639e9f72a772c9771d8bd14d5f/research/f75r_cipher_ladder_20261018/results/rung1_gpu_results.json"
m={"__name__":"ladder_base"}
exec(compile(urllib.request.urlopen(CODE,timeout=90).read().decode(),CODE,"exec"),m)
for n,v in m.items():
 if not n.startswith("_"):globals().setdefault(n,v)
def line_groups(page,trans="TTLI"):
 p=json.loads(download(VMS)[0])["pages"][page]
 return [re.findall("[a-z]+",x.get("t",{}).get(trans,"")) for x in p.values() if x.get("t",{}).get(trans,"")]
def shuffled_lines(groups,seed):
 rng=random.Random(seed); groups=[x[:] for x in groups]
 for line in groups:rng.shuffle(line)
 return [w for line in groups for w in line]
def markov2_type_null(words,seed):
 rng=random.Random(seed)
 # Conditionals estimated only from target folio; preserve exact token frequency,
 # recurrence, word-length and every line's word count using one synthetic replacement per TYPE.
 seq=[glyphs(w) for w in words]
 start=collections.Counter(w[0] for w in seq if w)
 pos=collections.defaultdict(collections.Counter);prev=collections.defaultdict(collections.Counter)
 pair=collections.defaultdict(collections.Counter)
 for w in seq:
  for j,ch in enumerate(w):
   p="end" if j==len(w)-1 else ("start" if j==0 else "middle")
   pos[p][ch]+=1
   if j>=1: prev[(p,w[j-1])][ch]+=1
   if j>=2: pair[(p,w[j-2],w[j-1])][ch]+=1
 def weighted(counter):
  if not counter:counter=start
  return rng.choices(list(counter),weights=list(counter.values()))[0]
 dictionary={}; reverse=set()
 for w in sorted(set(words),key=lambda x:(len(glyphs(x)),x)):
  n=len(glyphs(w));candidate=None
  for _ in range(30):
   a=[]
   for j in range(n):
    p="end" if j==n-1 else ("start" if j==0 else "middle")
    if j>=2 and pair.get((p,a[-2],a[-1])): d=pair[(p,a[-2],a[-1])]
    elif j>=1 and prev.get((p,a[-1])):d=prev[(p,a[-1])]
    else:d=pos[p]
    a.append(weighted(d))
   candidate=tuple(a)
   if candidate not in reverse:break
  reverse.add(candidate); dictionary[w]=candidate
 return [dictionary[w] for w in words]
def from_glyph_lists(x,idx):
 out=[]
 for w in x:
  for g in w:out.append(idx[g])
  out.append(SPACE)
 return out[:-1]
def score_permutations(model,key,seqs,device):
 return [float(mean_score(key[None,:],torch.tensor(s,device=device,dtype=torch.long),model).item()) for s in seqs]
def stat(real,arr):
 avg=sum(arr)/len(arr);sd=(sum((v-avg)**2 for v in arr)/(len(arr)-1))**.5 if len(arr)>1 else None
 effect=real-avg
 return {"real":real,"null_mean":avg,"null_sd":sd,"effect":effect,"effect_over_null_sd":effect/sd if sd else None,"nulls":arr}
def go(lang,seed,steps,pop):
 if not torch.cuda.is_available():raise RuntimeError("GPU_REQUIRED")
 device="cuda:0";torch.manual_seed(seed)
 data=json.load(urllib.request.urlopen(RESULTS,timeout=60))["jobs"][lang]
 result=data["result"];assert result["status"]=="EXPLORATORY_NOT_DECIPHERED"
 tr,_=conllu(lang,"train");model=make_model(tr[:90000]).to(device)
 keydata=result["key"];idx={g:i for i,g in enumerate(sorted(keydata))}
 key=torch.tensor([A.index(keydata[g]) for g in sorted(keydata)]+[i for i in range(26) if i not in [A.index(keydata[g]) for g in sorted(keydata)]],dtype=torch.long,device=device)
 assert len(key)==26 and len(set(key.tolist()))==26
 groups=line_groups("f75r");words=[w for g in groups for w in g]
 hg=line_groups("f76r");hold=[w for g in hg for w in g if set(glyphs(w)).issubset(idx)]
 trainids=from_glyph_lists([glyphs(w) for w in words],idx)
 holdids=from_glyph_lists([glyphs(w) for w in hold],idx)
 fit=float(mean_score(key[None,:],torch.tensor(trainids,device=device),model).item())
 ho=float(mean_score(key[None,:],torch.tensor(holdids,device=device),model).item())
 trials=100
 perm_train=[from_glyph_lists([glyphs(w) for w in shuffled_lines(groups,seed+100+i)],idx) for i in range(trials)]
 # For heldout drop words with unknown glyphs before shuffling, avoiding representation leakage.
 hf=[ [w for w in g if set(glyphs(w)).issubset(idx)] for g in hg]
 perm_hold=[from_glyph_lists([glyphs(w) for w in shuffled_lines(hf,seed+400+i)],idx) for i in range(trials)]
 fixed_train=stat(fit,score_permutations(model,key,perm_train,device))
 fixed_hold=stat(ho,score_permutations(model,key,perm_hold,device))
 emit("LADDER_BOUND_FIXED",{"lang":lang,"train":fixed_train,"holdout":fixed_hold})
 # Fair search-budget refit on null text, no advantage to trained original.
 refit=[]
 for j in range(5):
  perm=perm_train[j]
  _,v=anneal(torch.tensor(perm,device=device,dtype=torch.long),len(idx),model,seed+7000+j,steps,pop)
  refit.append(v)
 bound_refit=stat(fit,refit)
 emit("LADDER_BOUND_REFIT",{"lang":lang,**bound_refit})
 # A recurrence/type-frequency preserving Markov(2) glyph null.
 retype=[]
 for j in range(5):
  pseudo=markov2_type_null(words,seed+8000+j)
  ps=from_glyph_lists(pseudo,idx)
  _,v=anneal(torch.tensor(ps,device=device,dtype=torch.long),len(idx),model,seed+9000+j,steps,pop)
  retype.append(v)
 bound_markov=stat(fit,retype)
 emit("LADDER_BOUND_MARKOV2",{"lang":lang,**bound_markov})
 # Observe false positives: these bounds do not validate language identification.
 out={"schema":"F75R_MONO_BOUND_20261018","lang":lang,"seed":seed,"base_job":data["job_url"],
 "fixed_train":fixed_train,"fixed_holdout":fixed_hold,"refit_word_order":bound_refit,
 "refit_type_recurrence_markov2":bound_markov,
 "caveat":"Both nulls probe ordinary graphotactics. No historical language identified by this test."}
 emit("LADDER_BOUNDS_RESULT",out)
 tmp=OUT/(lang+"_bounds.tmp");p=OUT/(lang+"_bounds.pkl")
 with tmp.open("wb") as f:pickle.dump(out,f,protocol=5);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser()
 p.add_argument("--lang",choices=list(LANG),required=True)
 p.add_argument("--seed",type=int,default=2143)
 p.add_argument("--steps",type=int,default=600)
 p.add_argument("--pop",type=int,default=192)
 a=p.parse_args();go(a.lang,a.seed,a.steps,a.pop)
