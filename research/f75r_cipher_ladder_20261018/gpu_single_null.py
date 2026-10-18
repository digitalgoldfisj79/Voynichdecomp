#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5,<3", "Unidecode>=1.3,<2"]
# ///
"""Rung 2: exhaustively enumerate single-null glyphs + GPU-search each remaining substitution.
The evidence gate runs on independent planted null-cipher text BEFORE Voynich.
Every candidate has the same SEARCH budget; pnull=0.05 is frozen.
"""
import argparse,hashlib,json,math,os,pickle,random,sys,urllib.request
from pathlib import Path
import torch
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
SRC=BASE+"257a0d1198c3e8c0b416fda44255a0c8675424aa/research/f75r_cipher_ladder_20261018/gpu_mono_solver.py"
BRESULT=BASE+"7ffdfbc22ade63639e9f72a772c9771d8bd14d5f/research/f75r_cipher_ladder_20261018/results/rung1_gpu_results.json"
mod={"__name__":"mono_base"}
exec(compile(urllib.request.urlopen(SRC,timeout=90).read().decode(),SRC,"exec"),mod)
for name,value in mod.items():
 if not name.startswith("_"):globals().setdefault(name,value)
PN=.05
def encode_existing(words,glyph_order):
 idx={g:i for i,g in enumerate(glyph_order)}
 return [[idx[g] for g in glyphs(w)] for w in words]
def joined(seq,skip=None):
 v=[]
 for w in seq:
  c=[x for x in w if x!=skip]
  if not c:continue
  v.extend(c);v.append(SPACE)
 return v[:-1]
def renumber(seq,deleted):
 return [[x-int(x>deleted) for x in w if x!=deleted] for w in seq]
def chars_only(seq):return sum(len(w) for w in seq)
def channel_obj(ll,n_plain,n0,n_deleted,one_null):
 # n0 is original cipher GLYPHS, not spaces; trigram score via n_plain output events
 emitted=n0-n_deleted
 return ll*max(0,n_plain-2)+((emitted*math.log1p(-PN)+n_deleted*math.log(PN)) if one_null else 0.)
@torch.no_grad()
def solve_null_candidates(words,n,model,seed,steps,pop):
 dev=model.device;orig=chars_only(words)
 # The all-emitted mono baseline, same n and same GPU budget.
 all_ids=torch.tensor(joined(words),device=dev,dtype=torch.long)
 mono_key,mono_ll=anneal(all_ids,n,model,seed,steps,pop)
 base=channel_obj(mono_ll,len(all_ids),orig,0,False)/orig
 best=None;results=[]
 for nullsym in range(n):
  nested=renumber(words,nullsym)
  seq=joined(nested)
  if len(seq)<6:continue
  ids=torch.tensor(seq,device=dev,dtype=torch.long)
  key,ll=anneal(ids,n-1,model,seed+137+nullsym,steps,pop)
  deleted=sum(sum(v==nullsym for v in word) for word in words)
  value=channel_obj(ll,len(seq),orig,deleted,True)/orig
  item={"null_symbol":nullsym,"deletion_rate":deleted/max(1,orig),
   "score_per_cipher_glyph":value,"score_delta_vs_mono":value-base,
   "mapping":key[:n-1].tolist(),"raw_trigram":ll,"n_chars_out":len(seq)}
  results.append(item)
  if best is None or value>best["score_per_cipher_glyph"]:best=item
  if nullsym%5==4:emit("NULL_LADDER_PROGRESS",{"done":nullsym+1,"of":n,"best_symbol":best["null_symbol"],
   "best_advantage":best["score_delta_vs_mono"]})
 results.sort(key=lambda d:d["score_per_cipher_glyph"],reverse=True)
 return {"mono":{"per_cipher_glyph":base,"trigram":mono_ll,"key":mono_key[:n].tolist()},
         "best":best,"top5":results[:5],"enumerated":len(results)}
def planted(lang,model,seed,steps,pop):
 devwords,_=conllu(lang,"dev")
 alph=[x for x,_ in __import__("collections").Counter("".join(devwords)).most_common(20)]
 usable=[w for w in devwords if len(w)>1 and set(w)<=set(alph)]
 if len(usable)<200:raise RuntimeError("not enough plant words")
 rng=random.Random(seed);rng.shuffle(usable)
 plain=usable[:350]; target=" ".join(plain)
 sub=list(range(len(alph)));rng.shuffle(sub);bychar={c:sub[i] for i,c in enumerate(alph)}
 nullid=len(alph);encrypted=[]
 for w in plain:
  cw=[]
  for c in w:
   cw.append(bychar[c])
   if rng.random()<PN:cw.append(nullid)
  encrypted.append(cw)
 out=solve_null_candidates(encrypted,nullid+1,model,seed+10000,steps,pop)
 best=out["best"];text=decoded(torch.tensor(best["mapping"]),joined(renumber(encrypted,best["null_symbol"])),nullid)
 # Exact alignment only when null selected correctly; if wrong => fail closed.
 acc=sum(a==b for a,b in zip(text,target))/max(len(target),len(text))
 qual={"null_correct":best["null_symbol"]==nullid,
        "char_accuracy":acc,"true_null":nullid,"picked":best["null_symbol"],
        "n_plain_chars":len(target),"null_advantage":best["score_delta_vs_mono"],
        "qualified":best["null_symbol"]==nullid and acc>=.8 and best["score_delta_vs_mono"]>0,
        "top_nulls":[{"symbol":a["null_symbol"],"gain":a["score_delta_vs_mono"]} for a in out["top5"]]}
 emit("NULL_LADDER_PLANT",qual)
 return qual
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--lang",choices=list(LANG),required=True)
 p.add_argument("--steps",type=int,default=650)
 p.add_argument("--pop",type=int,default=192)
 p.add_argument("--seed",type=int,default=3927)
 a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError("GPU_REQUIRED")
 torch.manual_seed(a.seed);random.seed(a.seed)
 meta=json.load(urllib.request.urlopen(BRESULT,timeout=90))["jobs"][a.lang]
 tr,source=conllu(a.lang,"train");model=make_model(tr[:90000]).to("cuda:0")
 emit("NULL_LADDER_AUDIT",{"lang":a.lang,"gpu":torch.cuda.get_device_name(0),"steps":a.steps,
 "pop":a.pop,"pnull":PN,"source":source,"baseline":meta["job_url"]})
 qual=planted(a.lang,model,a.seed,a.steps,a.pop)
 if not qual["qualified"]:
  emit("NULL_LADDER_GATE",{"status":"FAIL_PLANTED_INSTRUMENT","no_voynich":True})
  return
 vw=getvms("f75r");hw=getvms("f76r")
 alphabet=sorted(set(g for w in vw for g in glyphs(w)))
 index={g:i for i,g in enumerate(alphabet)}
 words=encode_existing(vw,alphabet)
 out=solve_null_candidates(words,len(alphabet),model,a.seed+100, a.steps,a.pop)
 best=out["best"];nullglyph=alphabet[best["null_symbol"]]
 # Only tokens wholly inside the training glyph vocabulary can enter the heldout test.
 held=[w for w in hw if all(g in index for g in glyphs(w))]
 held_source=encode_existing(held,alphabet)
 old_n=chars_only(held_source)
 mono_key=torch.tensor(meta["result"]["key"] and [A.index(meta["result"]["key"][g]) for g in alphabet]+
 [c for c in range(26) if c not in [A.index(meta["result"]["key"][g]) for g in alphabet]], device="cuda:0")
 ids=torch.tensor(joined(held_source),device="cuda:0")
 mono_ll=float(mean_score(mono_key[None,:],ids,model).item())
 mono_hold=channel_obj(mono_ll,len(ids),old_n,0,False)/old_n
 hnested=renumber(held_source,best["null_symbol"])
 hids=torch.tensor(joined(hnested),device="cuda:0")
 nkey=torch.tensor(best["mapping"]+[i for i in range(26) if i not in best["mapping"]],device="cuda:0")
 null_ll=float(mean_score(nkey[None,:],hids,model).item())
 n_deleted=old_n-chars_only(hnested)
 null_hold=channel_obj(null_ll,len(hids),old_n,n_deleted,True)/old_n
 outcome={"lang":a.lang,"planted":qual,"search":out,"selected_nullglyph":nullglyph,
   "heldout_mono_per_cipher_glyph":mono_hold,"heldout_null_per_cipher_glyph":null_hold,
   "holdout_delta":null_hold-mono_hold,"heldout_words":len(held),
   "interpretation":"NO LANGUAGE / CIPHER PROMOTION without matched family-null search and heldout effect calibration"}
 emit("NULL_LADDER_RESULT",outcome)
 with (OUT/(a.lang+"_null1.pkl.tmp")).open("wb") as f:pickle.dump(outcome,f,5);f.flush();os.fsync(f.fileno())
 os.replace(OUT/(a.lang+"_null1.pkl.tmp"),OUT/(a.lang+"_null1.pkl"))
if __name__=="__main__":main()
