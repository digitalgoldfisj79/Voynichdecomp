#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5,<3", "Unidecode>=1.3,<2"]
# ///
"""F75R progressive cipher ladder 2026-10-18, instrument-first.
GPU batched substitution annealing, six independently sourced real-language controls.
All observations log to HF Job logs; local pickle checkpoints are atomic but ephemeral.
Do NOT promote candidate Voynich mappings if planted decoder recovery fails.
"""
import argparse,collections,hashlib,json,math,os,pickle,random,re,time,unicodedata,urllib.request
from pathlib import Path
import torch
from unidecode import unidecode

ROOT="https://raw.githubusercontent.com/"
VMS=ROOT+"digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json"
LANG={
 "la":("UD_Latin-Perseus","la_perseus"),
 "it":("UD_Italian-ISDT","it_isdt"),
 "de":("UD_German-GSD","de_gsd"),
 "cs":("UD_Czech-CAC","cs_cac"),
 "he":("UD_Hebrew-HTB","he_htb"),
 "yi":("UD_Yiddish-YiTB","yi_yitb")
}
EVA=("ckh","cth","cph","cfh","ikh","ith","iph","ifh","ch","sh")
A="abcdefghijklmnopqrstuvwxyz"; SPACE=26
OUT=Path("/tmp/f75r_cipher_ladder"); OUT.mkdir(exist_ok=True)
def emit(k,obj): print(k+"="+json.dumps(obj,ensure_ascii=False,sort_keys=True,default=str),flush=True)
def download(url):
 b=urllib.request.urlopen(url,timeout=120).read()
 return b,hashlib.sha256(b).hexdigest()
def clean(s):
 s=unicodedata.normalize("NFKD",unidecode(s.lower().replace("ß","ss")))
 return re.sub("[^a-z]","",s.lower())
def conllu(lang,split):
 repo,prefix=LANG[lang]
 u=ROOT+f"UniversalDependencies/{repo}/master/{prefix}-ud-{split}.conllu"
 b,h=download(u)
 words=[]
 for row in b.decode("utf-8","replace").splitlines():
  if not row or row[0]=="#": continue
  c=row.split("\t")
  if len(c)!=10 or not c[0].isdigit() or c[3]=="PUNCT":continue
  w=clean(c[1])
  if 1<=len(w)<=22:words.append(w)
 return words,{"url":u,"sha256":h,"tokens":len(words)}
def getvms(page,trans="TTLI"):
 p=json.loads(download(VMS)[0])["pages"][page]
 words=[]
 for num,row in p.items():
  s=row.get("t",{}).get(trans,"")
  words.extend(re.findall(r"[a-z]+",s))
 return words
def glyphs(w):
 out=[];i=0
 while i<len(w):
  g=next((g for g in EVA if w.startswith(g,i)),w[i])
  out.append(g);i+=len(g)
 return out
def tokens_to_ids(words,glyph_index):
 seq=[]
 for w in words:
  for g in glyphs(w):
   if g not in glyph_index:raise ValueError(("unknown glyph",g))
   seq.append(glyph_index[g])
  seq.append(len(glyph_index))
 return seq[:-1]
def make_model(words):
 t=" ".join(words);trip=collections.Counter(t[i:i+3] for i in range(len(t)-2))
 pref=collections.Counter(t[i:i+2] for i in range(len(t)-2))
 vals=torch.empty(27**3,dtype=torch.float32)
 for x in range(27):
  for y in range(27):
   a=(A+" ")[x]+(A+" ")[y];d=pref[a]+27*.1
   for z in range(27):
    vals[x*729+y*27+z]=math.log((trip[a+(A+" ")[z]]+.1)/d)
 return vals
def mean_score(keys,ids,logprobs):
 # keys: batch x 26 injective glyph -> plaintext characters; ids uses cipher symbol len(syms) as blank
 m=keys.shape[0]
 lookup=torch.cat((keys[:,:],torch.full((m,1),SPACE,device=keys.device,dtype=torch.long)),1)
 s=lookup[:,ids].long()
 ix=s[:,:-2]*729+s[:,1:-1]*27+s[:,2:]
 return logprobs[ix].mean(dim=-1)
@torch.no_grad()
def anneal(ids,n,model,seed,steps=850,pop=256):
 rng=torch.Generator(device="cpu");rng.manual_seed(seed)
 device=model.device; rows=torch.arange(pop,device=device)
 # 26 columns allow swaps with otherwise unused letters.
 keys=torch.stack([torch.randperm(26,generator=rng) for _ in range(pop)]).to(device)
 cur=mean_score(keys,ids,model)
 best_score=-float("inf");best_key=None
 for st in range(steps):
  f=st/max(1,steps-1)
  temp=.24*(1-f)**2+.004
  ix=torch.randint(n,(pop,),device=device)
  jx=torch.randint(26,(pop,),device=device)
  new=keys.clone();u=keys[rows,ix].clone();v=keys[rows,jx].clone()
  new[rows,ix]=v;new[rows,jx]=u
  s2=mean_score(new,ids,model)
  accept=(s2>cur)|(torch.rand(pop,device=device)<torch.exp(torch.clamp((s2-cur)/temp,-50,0)))
  keys=torch.where(accept[:,None],new,keys)
  cur=torch.where(accept,s2,cur)
  idx=int(torch.argmax(cur).item());val=float(cur[idx].item())
  if val>best_score: best_score=val; best_key=keys[idx].clone()
  if st and st%250==0:emit("LADDER_PROGRESS",{"step":st,"steps":steps,"best_trigram_loglik":best_score})
 assert best_key is not None
 return best_key,best_score
def decoded(key,ids,n):
 key=key[:n].tolist()
 return "".join(" " if q==n else A[key[q]] for q in ids)
def lexical(text,lexicon):
 w=text.split();hits=[s for s in w if s in lexicon]
 return {"words":len(w),"matched":len(hits),"match_rate":round(len(hits)/max(len(w),1),5),
 "examples":hits[:15],"first_words":w[:28]}
def match_recovery(recovered,truth,n):
 return sum(int(a==b) for a,b in zip(recovered[:n].tolist(),truth[:n]))/n
def checkpoint(name,info):
 p=OUT/(name+".pkl");tmp=p.with_suffix(".tmp")
 with tmp.open("wb") as f:pickle.dump(info,f,protocol=5);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def control(lang,dev,n,steps,pop,model,seed,lex):
 # Plaintext is wholly from UD dev, not model train. All symbols represent distinct letters.
 alphabet=[c for c,_ in collections.Counter("".join(dev)).most_common(20)]
 allowed=set(alphabet)
 viable=[w for w in dev if len(w)>=2 and set(w)<=allowed]
 rng=random.Random(seed);rng.shuffle(viable)
 selected=viable[:360]
 if len(selected)<120:raise RuntimeError(f"insufficient dev material: {len(selected)}")
 # A true monoalphabetic plant, transcribed into n=20 opaque pseudo-glyphs.
 n0=len(alphabet); permutation=list(range(n0));rng.shuffle(permutation)
 cipher_by_plain={alphabet[i]:permutation[i] for i in range(n0)}
 ids=[]
 for w in selected:
  ids.extend(cipher_by_plain[c] for c in w);ids.append(n0)
 ids=ids[:-1];truth=[None]*n0
 for c,i in cipher_by_plain.items():truth[i]=A.index(c)
 device=model.device
 key,score=anneal(torch.tensor(ids,device=device,dtype=torch.long),n0,model,seed+211,steps,pop)
 plaintext=decoded(key,ids,n0)
 gold=" ".join(selected)
 char_acc=sum(a==b for a,b in zip(plaintext,gold))/len(gold)
 map_acc=match_recovery(key,truth,n0)
 qual={"plain_chars":len(gold),"n_glyphs":n0,"char_accuracy":round(char_acc,5),
 "map_accuracy":round(map_acc,5),"score":score,"lexical":lexical(plaintext,lex),
 "qualified":char_acc>=.8 and map_acc>=.75}
 emit("LADDER_PLANT",qual)
 checkpoint(f"{lang}_plant",qual)
 return qual
def positional_null(words,seed):
 # Within each word-length x glyph-position bucket, permute glyph occurrences;
 # preserves word lengths and positional glyph margins; destroys word identity.
 rng=random.Random(seed)
 seq=[glyphs(w) for w in words]
 groups=collections.defaultdict(list)
 for i,w in enumerate(seq):
  for j,g in enumerate(w):
   pos="only" if len(w)==1 else "first" if j==0 else "last" if j==len(w)-1 else "middle"
   groups[pos].append((i,j,g))
 for group in groups.values():
  old=[z[2] for z in group];rng.shuffle(old)
  for (i,j,_),g in zip(group,old):seq[i][j]=g
 return seq
def seq_null(words,index,seed):
 s=positional_null(words,seed);flat=[]
 for word in s:
  flat.extend(index[g] for g in word);flat.append(len(index))
 return flat[:-1]
def run(args):
 torch.manual_seed(args.seed);random.seed(args.seed)
 if not torch.cuda.is_available():raise RuntimeError("GPU_REQUIRED: cannot claim GPU run")
 device=torch.device("cuda:0")
 emit("LADDER_AUDIT",{"phase":"F75R_MONO_CAESAR","lang":args.lang,"seed":args.seed,
 "torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),"steps":args.steps,
 "pop":args.pop,"nulls":args.nulls,"transcription":args.trans,"repo_branch":
 "experiment/f75r-progressive-cipher-ladder-20261018",
 "note":"Real-language corpora are exploratory UD controls, NOT 15c adjudication."})
 tr,traudit=conllu(args.lang,"train");dev,devaudit=conllu(args.lang,"dev")
 emit("LADDER_SOURCES",{"train":traudit,"dev":devaudit,"normalization":"Unidecode->lowercase ASCII a-z"})
 lex=set(tr);model=make_model(tr[:args.max_train_tokens]).to(device)
 kval=control(args.lang,dev,20,args.steps,args.pop,model,args.seed,lex)
 if not kval["qualified"]:
  emit("LADDER_GATE",{"status":"FAILED_PLANTED_RECOVERY","action":"NO_VOYNICH_SCORE","lang":args.lang})
  return
 vw=getvms("f75r",args.trans);hw=getvms("f76r",args.trans)
 glyph_vocab=sorted(set(g for w in vw+hw for g in glyphs(w)))
 # Holdout-only novel glyphs are carried but never selected for calibration.
 primary=sorted(set(g for w in vw for g in glyphs(w)))
 unseen=sorted(set(glyph_vocab)-set(primary))
 if len(glyph_vocab)>26:raise ValueError("More glyphs than the Latin alphabet, monoalphabetic model invalid")
 idx={g:i for i,g in enumerate(glyph_vocab)}
 seq=torch.tensor(tokens_to_ids(vw,idx),device=device,dtype=torch.long)
 hold=torch.tensor(tokens_to_ids(hw,idx),device=device,dtype=torch.long)
 # Rung 0: Caesar under explicit (noncanonical) glyph order, 26 trial shifts.
 base=list(range(26))
 caesar=[]
 for shift in range(26):
  key=torch.tensor([(i+shift)%26 for i in base],device=device)
  score=mean_score(key[None,:],seq,model)[0].item()
  caesar.append((score,shift))
 caesar.sort(reverse=True)
 emit("LADDER_CAESAR",{"alphabet_order":glyph_vocab,
 "warning":"Alphabet order conventional only; shift is not a historical cipher hypothesis",
 "best":[{"shift":i,"score":round(s,6)} for s,i in caesar[:3]]})
 key,score=anneal(seq,len(idx),model,args.seed+4000,args.steps,args.pop)
 decoded_train=decoded(key,tokens_to_ids(vw,idx),len(idx))
 decoded_hold=decoded(key,tokens_to_ids(hw,idx),len(idx))
 holdscore=float(mean_score(key[None,:],hold,model).item())
 arr=[]
 for k in range(args.nulls):
  nullids=torch.tensor(seq_null(vw,idx,args.seed+10000+k),device=device,dtype=torch.long)
  _,nullsc=anneal(nullids,len(idx),model,args.seed+50000+k,
                   max(120,args.steps//2),args.pop)
  arr.append(nullsc)
 mu=sum(arr)/len(arr) if arr else None
 sd=(sum((x-mu)**2 for x in arr)/(len(arr)-1))**.5 if len(arr)>=2 else None
 delta=score-mu if mu is not None else None
 z=delta/sd if sd and sd>0 else None
 result={"status":"EXPLORATORY_NOT_DECIPHERED","language":args.lang,"trans":args.trans,
 "source_folio":"f75r","holdout_folio":"f76r","train_words":len(vw),
 "holdout_words":len(hw),"n_glyphs":len(idx),"unknown_holdout_glyphs":unseen,
 "key":{g:A[int(key[idx[g]].item())] for g in idx},
 "rung":"mono","fit_trigram":score,"holdout_trigram":holdscore,
 "null_scores":arr,"null_mean":mu,"null_sd":sd,"effect":delta,"effect_over_null_sd":z,
 "lexical_train":lexical(decoded_train,lex),"lexical_holdout":lexical(decoded_hold,lex),
 "warning":"Char ngram optimisation is NOT a calibrated language score; positional nulls are imperfect controls."}
 emit("LADDER_RESULT",result)
 checkpoint(f"{args.lang}_{args.trans}_mono",result)
 emit("LADDER_GATE",{"status":"MONO_SCREEN_DONE","candidate_promotion":bool(z is not None and z>=2),
 "effect":delta,"null_sd":sd,"z":z})
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--lang",choices=sorted(LANG),required=True)
 p.add_argument("--trans",choices=["TTLI","ZLZI"],default="TTLI")
 p.add_argument("--seed",type=int,default=1731)
 p.add_argument("--steps",type=int,default=800)
 p.add_argument("--pop",type=int,default=256)
 p.add_argument("--nulls",type=int,default=5)
 p.add_argument("--max-train-tokens",type=int,default=90000)
 run(p.parse_args())
if __name__=="__main__":main()
