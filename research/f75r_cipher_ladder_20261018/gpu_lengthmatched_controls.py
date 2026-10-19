#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5,<3", "Unidecode>=1.3,<2"]
# ///
"""F75R null ladder, matched *word-length and word-type recurrence* synthetic controls.
Keep original Voynich types/occurrence positions as a fixed nuisance template; assign
independently sourced UD heldout real-language word types to template types by glyph length.
Then monoalphabetically encrypt with optional deterministic per-word-type inserted null.
The cipher symbol substitutions are consistent across all words; unknown VMS symbol
identities are NOT used as hypothetical plaintext. Controls cannot prove actual meaning.
"""
import argparse,collections,hashlib,json,os,pickle,random,urllib.request
from pathlib import Path
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
SRC=BASE+"67ae96aaede567fd43e79ff2ba3ec461eae58962/research/f75r_cipher_ladder_20261018/gpu_null_family_calibration.py"
mod={"__name__":"null3_reuse"}
exec(compile(urllib.request.urlopen(SRC,timeout=90).read().decode(),SRC,"exec"),mod)
import torch
emit=mod["emit"];conllu=mod["conllu"];make_model=mod["make_model"];getvms=mod["getvms"];glyphs=mod["glyphs"];trial=mod["trial"]
OUT=Path("/tmp/null3_lengthmatched");OUT.mkdir(exist_ok=True)
def save(name,x):
 p=OUT/name;tmp=OUT/(name+".tmp")
 with tmp.open("wb") as f:pickle.dump(x,f,5);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def synthetic(words,train_template,held_template,rate,seed):
 rng=random.Random(seed)
 letters=[c for c,_ in collections.Counter("".join(words)).most_common(19 if rate else 20)]
 accepted=[w for w in set(words) if all(c in letters for c in w)]
 bags=collections.defaultdict(list)
 for w in accepted:bags[len(w)].append(w)
 for a in bags.values():a.sort();rng.shuffle(a)
 types=sorted(set(train_template+held_template))
 used=collections.defaultdict(int);wordmap={}
 misses=[];collisions=[]
 for t in types:
  length=len(glyphs(t))
  bag=bags[length]
  if not bag:
   misses.append([t,length]);continue
  k=used[length];wordmap[t]=bag[k%len(bag)];used[length]+=1
  if k>=len(bag):collisions.append([t,length,wordmap[t]])
 if misses:raise ValueError(("no matching UD source word length",misses[:10]))
 ords=list(range(len(letters)));rng.shuffle(ords)
 map_cipher={c:ords[i] for i,c in enumerate(letters)}
 true_null=len(letters) if rate else None
 # Fixed null mask per plaintext word TYPE, so repeated types remain identical
 lookup={}
 for w in sorted(set(wordmap.values())):
  glyphs_out=[]
  for c in w:
   glyphs_out.append(map_cipher[c])
   if rate and rng.random()<rate:glyphs_out.append(true_null)
  lookup[w]=glyphs_out
 def enc(template):return [lookup[wordmap[t]][:] for t in template]
 a=enc(train_template);b=enc(held_template)
 ntrain=sum(len(w) for w in a);nheld=sum(len(w) for w in b)
 return a,b,{"rate":rate,"seed":seed,"train_n":len(a),"holdout_n":len(b),
  "unique_template_types":len(types),"lexicon_types_used":len(set(wordmap.values())),
  "word_type_collisions":len(collisions),"collision_examples":collisions[:5],
  "train_mean_glyphs":ntrain/len(a),"holdout_mean_glyphs":nheld/len(b),
  "true_null":true_null,
  "word_length_training_max":max(len(x) for x in a),
  "word_length_holdout_max":max(len(x) for x in b)}
def run(lang,rate,reps,steps,pop,seed):
 tr,ma=conllu(lang,"train");dev,db=conllu(lang,"dev")
 model=make_model(tr[:90000]).to("cuda:0")
 t=getvms("f75r");h=getvms("f76r")
 # Templates are nuisance structure only. The underlying plaintext words
 # and glyph-to-character mapping are independently chosen per seed.
 emit("NULL3_LENGTH_AUDIT",{"language":lang,"rate":rate,"replicates":reps,
 "device":torch.cuda.get_device_name(0),"steps":steps,"pop":pop,"seed":seed,
 "model_source":ma,"plaintext_source":db,"train_words":len(t),
 "holdout_words":len(h),"meaning":"Synthetic calibration; no VMS decipherment"})
 out=[]
 for i in range(reps):
  qseed=seed+i*101
  a,b,meta=synthetic(dev,t,h,rate,qseed)
  res=trial(a,b,model,qseed,steps,pop)
  z={**meta,**{k:v for k,v in res.items() if k not in ["fit_key","null_key"]}}
  z["correct_null"]=rate>0 and z["true_null"]==z["selected_null"]
  save(f"{lang}_{int(rate*100)}_{i}.pkl",z)
  emit("NULL3_LENGTH_TRIAL",z)
  out.append(z)
 vals=[x["heldout_gain"] for x in out]
 mn=sum(vals)/len(vals)
 sd=(sum((x-mn)**2 for x in vals)/(len(vals)-1))**.5 if len(vals)>1 else None
 emit("NULL3_LENGTH_RESULT",{"language":lang,"rate":rate,"n":len(out),
  "mean_gain":mn,"sd":sd,"positive_gain":sum(x>0 for x in vals),
  "correct_null":sum(x["correct_null"] for x in out),
  "collision_counts":[x["word_type_collisions"] for x in out],
  "warning":"Fully length-matched ONLY for no-null. Null insertion increases lengths; token frequency counts preserved except any lexical collisions."})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--lang",choices=["la","it"],required=True)
 p.add_argument("--rate",type=float,choices=[0.,.05],required=True)
 p.add_argument("--reps",type=int,default=8)
 p.add_argument("--steps",type=int,default=650);p.add_argument("--pop",type=int,default=192)
 p.add_argument("--seed",type=int,default=48921)
 x=p.parse_args();run(x.lang,x.rate,x.reps,x.steps,x.pop,x.seed)
