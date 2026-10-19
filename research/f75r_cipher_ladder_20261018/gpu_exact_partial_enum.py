#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5,<3","numpy>=1.26,<3","wordfreq>=3.1,<4","Unidecode>=1.3,<2"]
# ///
"""F75R-INCREMENTAL-EXACT-6, plain glyph -> letter search, 2026-10-19.
Exact enumeration at K<=6; beam search beyond K6 handled separately.
Dictionary only wordfreq 3.1 (modern) + UD Latin/ Yiddish (not 15th century).
Distinguish this from a decipherment: lookups produce many coincidences.
"""
import argparse, collections, hashlib, json, math, os, re, sys, time, unicodedata, urllib.request
from pathlib import Path
import numpy as np
import torch
from unidecode import unidecode
from wordfreq import top_n_list
VMS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json"
LANGS=["it","de","cs","fr","es","pl","nl","ro","sk","sl","hu","el","ru","uk","pt","ca","en","he"]
EXTRAS={"la":"https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-Perseus/master/la_perseus-ud-train.conllu",
"yi":"https://raw.githubusercontent.com/UniversalDependencies/UD_Yiddish-YiTB/master/yi_yitb-ud-train.conllu"}
GLYPHS=["ch","d","e","k","sh","y"]
COMPS=("ckh","cth","cph","cfh","ikh","ith","iph","ifh","ch","sh")
ABC="abcdefghijklmnopqrstuvwxyz"
CACHE=Path("/tmp/f75rexact");CACHE.mkdir(exist_ok=True)
def emit(tag,obj):print(tag+"="+json.dumps(obj,ensure_ascii=False,separators=(",",":"),default=str),flush=True)
def normalize(s):
 s=unidecode(s.lower().replace("ß","ss")).lower()
 return re.sub("[^a-z]","",s)
def tokenize(w):
 out=[];i=0
 while i<len(w):
  g=next((x for x in COMPS if w.startswith(x,i)),w[i])
  out.append(g);i+=len(g)
 return out
def read_page(p="f75r",trans="TTLI"):
 b=urllib.request.urlopen(VMS_URL,timeout=90).read()
 rows=json.loads(b)["pages"][p]
 w=[x for z in rows.values() for x in re.findall("[a-z]+",z.get("t",{}).get(trans,""))]
 return w,hashlib.sha256(b).hexdigest()
def pattern(xs):
 c={};out=[]
 for x in xs:
  if x not in c:c[x]=len(c)
  out.append(c[x])
 return tuple(out)
def lexicon(limit):
 lex=collections.defaultdict(set);audit={}
 for lang in LANGS:
  try:forms=top_n_list(lang,limit)
  except Exception as e:
   emit("LEXICON_WARNING",{"lang":lang,"reason":str(e)});continue
  count=0
  for x in forms:
   y=normalize(x)
   if 3<=len(y)<=10 and len(y)==len(x) or (lang in ["el","ru","uk","he"] and 3<=len(y)<=10):
    if 3<=len(y)<=10:
     lex[y].add(lang);count+=1
  audit[lang]={"n_requested":limit,"n_eligible":count}
 for lang,url in EXTRAS.items():
  try:
   b=urllib.request.urlopen(url,timeout=90).read()
   audit[lang]={"url":url,"sha256":hashlib.sha256(b).hexdigest()}
   for line in b.decode("utf8","replace").splitlines():
    if not line or line.startswith("#"):continue
    c=line.split("\t")
    if len(c)==10 and c[0].isdigit() and c[3]!="PUNCT":
     y=normalize(c[1])
     if 3<=len(y)<=10:lex[y].add(lang)
  except Exception as e: audit[lang]={"error":str(e)}
 audit["unique_forms"]=len(lex)
 emit("LEXICON_AUDIT",audit)
 return lex,audit
def build_lookups(words,glyphs,lex):
 counts=collections.Counter(w for w in words if set(tokenize(w)).issubset(glyphs) and len(tokenize(w))>=3)
 # Distinct *word types* scored; preserve but cap frequency weighting to avoid a single repeated word dominating.
 by_pattern=collections.defaultdict(list)
 for word,languages in lex.items():by_pattern[(len(word),pattern(word))].append(word)
 dev=torch.device("cuda:0")
 lookups=[];audit=[]
 for w,freq in sorted(counts.items(),key=lambda a:(-a[1],a[0])):
  units=tokenize(w);rg=sorted(set(units),key=glyphs.index);r=len(rg)
  # r>=6 matched against sparse whole-word lookup on GPU.
  weight=min(3.0,math.sqrt(freq))*(1+0.16*max(0,len(units)-3))
  if len(units)>=4: weight*=1.25
  if r<=5:
   arr=np.zeros(26**r,dtype=np.uint8)
   for candidate in by_pattern.get((len(units),pattern(units)),[]):
    pairs=dict(zip(units,candidate))
    idx=0
    for g in rg:idx=idx*26+ord(pairs[g])-97
    arr[idx]=1
   lookup={"kind":"dense","table":torch.from_numpy(arr).to(dev),
           "indices":[glyphs.index(x) for x in rg],
           "powers":[26**i for i in range(r-1,-1,-1)]}
   nz=int(np.count_nonzero(arr))
  else:
   # Fast GPU membership test in sorted 64-bit integer codes, ≤10 chars, codes fit signed int64.
   candidates=by_pattern.get((len(units),pattern(units)),[])
   codes=sorted(set(sum((ord(ch)-97)*26**(len(x)-j-1) for j,ch in enumerate(x)) for x in candidates))
   lookup={"kind":"sparse","table":torch.tensor(codes if codes else [-1],dtype=torch.long,device=dev),
           "indices":[glyphs.index(x) for x in units],
           "powers":[26**i for i in range(len(units)-1,-1,-1)]}
   nz=len(codes)
  lookups.append((w,freq,weight,lookup))
  audit.append({"w":w,"occurrences":freq,"glyph_len":len(units),"unique":r,"dictionary_hits":nz,"weight":weight})
 emit("LOOKUP_AUDIT",{"glyphs":glyphs,"eligible_tokens":sum(counts.values()),
 "unique_types":len(counts),"lookups":audit})
 return lookups
@torch.no_grad()
def score(keys,lookups,topn=0):
 scores=torch.zeros(keys.shape[0],device=keys.device,dtype=torch.float32)
 hitcount=torch.zeros(keys.shape[0],device=keys.device,dtype=torch.int16)
 for word,freq,weight,z in lookups:
  if z["kind"]=="dense":
   j=torch.zeros(keys.shape[0],device=keys.device,dtype=torch.long)
   for g,p in zip(z["indices"],z["powers"]):j=j+keys[:,g]*p
   found=z["table"][j].bool()
  else:
   j=torch.zeros(keys.shape[0],device=keys.device,dtype=torch.long)
   for g,p in zip(z["indices"],z["powers"]):j=j+keys[:,g]*p
   ix=torch.searchsorted(z["table"],j)
   found=z["table"][ix.clamp(max=z["table"].numel()-1)]==j
  scores+=found.float()*weight;hitcount+=found.short()
 return scores,hitcount
def decode(w,key,gl):
 ids={g:int(key[i]) for i,g in enumerate(gl)}
 return "".join(ABC[ids[x]] for x in tokenize(w)) if set(tokenize(w)).issubset(ids) else None
def report_top(keys,values,words,hold,gl,lex,top=15):
 result=[]
 for i,(key,sc,hits) in enumerate(values[:top]):
  mapping={g:ABC[key[j]] for j,g in enumerate(gl)}
  def observed(wlist):
   counter=collections.Counter(w for w in wlist if set(tokenize(w)).issubset(mapping) and len(tokenize(w))>=3)
   matches=[]
   for w,n in counter.items():
    plain=decode(w,key,gl)
    if plain in lex:matches.append({"cipher":w,"plain":plain,"occurrences":n,"lang":sorted(lex[plain])})
   matches.sort(key=lambda x:(-x["occurrences"],-len(x["plain"]),x["cipher"]))
   return {"eligible_tokens":sum(counter.values()),"eligible_types":len(counter),
           "dictionary_matching_types":len(matches),"dictionary_matching_tokens":sum(x["occurrences"] for x in matches),
           "words":matches[:25]}
  result.append({"rank":i+1,"score":sc,"n_matching_types_search":hits,
                 "mapping":mapping,"training":observed(words),"untouched_f76r":observed(hold)})
 return result
def candidates(start,stop,k,device):
 # Enumerate 26^k states, filter injective. This is literally exhaustive, not annealing.
 span=26**(k-1)
 for x in range(start,stop):
  a=0
  while a<span:
   b=min(span,a+ARGS.batch)
   n=torch.arange(x*span+a,x*span+b,device=device,dtype=torch.long)
   cols=[];z=n
   for _ in range(k):
    cols.append(z%26);z=torch.div(z,26,rounding_mode="floor")
   keys=torch.stack(cols[::-1],dim=1).contiguous()
   good=torch.ones(len(keys),dtype=torch.bool,device=device)
   for i in range(k):
    for j in range(i+1,k):good &= keys[:,i]!=keys[:,j]
   yield keys[good],n[good]
   a=b
def top_mixed(existing,newkeys,newscores,newhits,maxkeep):
 # Deterministic score followed by key lexicographic tie-breaker.
 r=min(maxkeep,len(newkeys))
 if not r:return existing
 ix=torch.topk(newscores,k=r,largest=True,sorted=True).indices
 kk=newkeys[ix].cpu().tolist();ss=newscores[ix].cpu().tolist();hh=newhits[ix].cpu().tolist()
 arr=existing+[(x,float(y),int(z)) for x,y,z in zip(kk,ss,hh)]
 arr.sort(key=lambda a:(-a[1],-a[2],a[0]))
 return arr[:maxkeep]
def main():
 global ARGS
 p=argparse.ArgumentParser()
 p.add_argument("--k",type=int,choices=[4,5,6,7,8],default=6)
 p.add_argument("--first-start",type=int,default=0)
 p.add_argument("--first-stop",type=int,default=26)
 p.add_argument("--batch",type=int,default=200000)
 p.add_argument("--keep",type=int,default=500)
 p.add_argument("--lexicon",type=int,default=25000)
 p.add_argument("--mode",choices=["real","type_scramble"],default="real")
 p.add_argument("--seed",type=int,default=119)
 p.add_argument("--smoke",type=int,default=0,help="number batches to check; 0 full exhaustive")
 ARGS=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
 import random
 gpu=torch.device("cuda:0");t0=time.time()
 train,sha=read_page("f75r");hold,_=read_page("f76r")
 gl={4:["d","e","sh","y"],5:["ch","d","e","sh","y"],6:GLYPHS,
     7:["d","e","k","o","q","sh","y"],8:["d","e","k","l","o","q","sh","y"]}[ARGS.k]
 if ARGS.mode=="type_scramble":
  # One shuffle per Voynich TYPE; preserves length, glyph multiset, and all recurrence counts.
  rng=random.Random(ARGS.seed);mapping={}
  for w in sorted(set(train)):
   x=tokenize(w);rng.shuffle(x);mapping[w]="".join(x)
  train=[mapping[w] for w in train]
 lex,audit=lexicon(ARGS.lexicon)
 lookups=build_lookups(train,gl,lex)
 emit("EXACT_AUDIT",{"gpu":torch.cuda.get_device_name(),"page":"f75r","holdout":"f76r",
  "version":"exact-6-20261019","k":ARGS.k,"mode":ARGS.mode,"source_sha256":sha,
  "exact_keys_total":math.perm(26,ARGS.k),"first_start":ARGS.first_start,"first_stop":ARGS.first_stop,
  "batch":ARGS.batch,"keep":ARGS.keep,"gl":gl,"lexicon_forms":len(lex),
  "smoke":ARGS.smoke,"warning":"Search selects on same f75r word hits; rank is NOT a decipherment claim."})
 best=[];seen=0;batches=0
 for keys,_ in candidates(ARGS.first_start,ARGS.first_stop,ARGS.k,gpu):
  if not len(keys):continue
  s,h=score(keys,lookups)
  best=top_mixed(best,keys,s,h,ARGS.keep)
  seen+=len(keys);batches+=1
  if batches%60==0:emit("EXACT_PROGRESS",{"seen_injective":seen,"batches":batches,
    "elapsed_s":round(time.time()-t0,2),"best":best[0][1],"rate_keys_per_s":round(seen/(time.time()-t0),1)})
  if ARGS.smoke and batches>=ARGS.smoke:break
 summary=report_top(None,best,train,hold,gl,lex,20)
 emit("EXACT_RESULT",{"schema":"F75R_EXACT_ENUM_20261019","mode":ARGS.mode,"k":ARGS.k,
 "first_start":ARGS.first_start,"first_stop":ARGS.first_stop,"seen":seen,
 "batches":batches,"elapsed_s":time.time()-t0,"complete":not bool(ARGS.smoke),
 "best_score":best[0][1] if best else None,
 "top":summary,
 "topkeys":[{"key":"".join(ABC[x] for x in key),"score":val,"hits":n} for key,val,n in best],
 "dictionary_audit":audit})
if __name__=="__main__":main()
