#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.5,<3","numpy>=1.26,<3","wordfreq>=3.1,<4","Unidecode>=1.3,<2"]
# ///
"""Extend exact six-glyph partial substitution candidates one glyph at a time.
K7 onward is PRUNED/BEAM (not exhaustive). All mapping keys remain injective.
Input is SHA-pinned external GitHub JSON merging completed K6 shards.
Print result JSON; no translation or language identification claim.
"""
import argparse,collections,json,math,os,sys,time,urllib.request
import torch
RAW="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
def grab(u):return urllib.request.urlopen(u,timeout=90).read().decode()
def parse():
 p=argparse.ArgumentParser()
 p.add_argument("--input",required=True,help="public raw GitHub URL JSON manifest of exact6 keys")
 p.add_argument("--beam",type=int,default=10000)
 p.add_argument("--init",type=int,default=3000)
 p.add_argument("--up-to",type=int,default=12)
 p.add_argument("--batch",type=int,default=60000)
 p.add_argument("--lexicon",type=int,default=25000)
 return p.parse_args()
args=parse()
SOURCE=RAW+"d38f3bb58be3bf1612fb3be879c8b72075fa5f30/research/f75r_cipher_ladder_20261018/gpu_exact_partial_enum.py"
g={"__name__":"solver_module"};exec(compile(grab(SOURCE),SOURCE,"exec"),g)
ABC=g["ABC"];emit=g["emit"];tokenize=g["tokenize"];lexicon=g["lexicon"];build_lookups=g["build_lookups"];score=g["score"];report_top=g["report_top"];read_page=g["read_page"]
if not torch.cuda.is_available():raise RuntimeError("GPU_REQUIRED")
device="cuda:0"; t0=time.time()
manifest=json.loads(grab(args.input));gl=list(manifest["glyphs"])
if len(gl) not in (6,7,8):raise ValueError("input must be K6/K7/K8 exact")
train,sha=read_page("f75r");hold,_=read_page("f76r")
lex,audit=lexicon(args.lexicon)
initial=manifest["topkeys"][:args.init]
keys=torch.tensor([[ABC.index(ch) for ch in x["key"]] for x in initial],device=device,dtype=torch.long)
if keys.size(1)!=6:raise RuntimeError("wrong key length")
def pick_extension(remaining,gl,train):
 # maximize new complete word type coverage (not any language criterion)
 eligible=collections.Counter()
 for w in train:
  ss=set(tokenize(w))
  if len(tokenize(w))>=3:eligible[frozenset(ss)]+=1
 current=set(gl)
 def metric(g):
  newly=[(w,v) for w,v in eligible.items() if w<=current|{g} and not w<=current]
  return sum(min(3,math.sqrt(v)) for _,v in newly)
 return max(remaining,key=lambda q:(metric(q),q))
allgs=sorted(set(g for w in train for g in tokenize(w)))
emit("BEAM_AUDIT",{"exact_seed_keys":len(initial),"glyphs":gl,"up_to":args.up_to,
"beam":args.beam,"source_vms_sha":sha,"source_manifest":args.input,
"not_exhaustive_after_k6":True,"limit":args.lexicon})
results=[]
for k in range(len(gl)+1,args.up_to+1):
 rem=[x for x in allgs if x not in gl]
 if not rem:break
 newglyph=pick_extension(rem,gl,train);gl.append(newglyph)
 cand=[]
 for a in range(0,len(keys),args.batch//26 or 1):
  part=keys[a:a+max(1,args.batch//26)]
  nums=torch.arange(26,device=device,dtype=torch.long)
  ext=nums[None,:].expand(len(part),26)
  not_used=(part[:,:,None]!=ext[:,None,:]).all(dim=1)
  extended=torch.cat([part[:,None,:].expand(-1,26,-1),ext[:,:,None]],dim=-1)
  cand.append(extended[not_used])
 expanded=torch.cat(cand)
 lookups=build_lookups(train,gl,lex)
 allscores=[];allhits=[]
 for i in range(0,len(expanded),args.batch):
  sc,hi=score(expanded[i:i+args.batch],lookups)
  allscores.append(sc);allhits.append(hi)
 allscores=torch.cat(allscores);allhits=torch.cat(allhits)
 n=min(len(expanded),args.beam)
 indices=torch.topk(allscores+allhits.float()*1e-6,k=n,sorted=True).indices
 # deterministic sort score desc, keys lex; tied rank does not indicate proof.
 keys=expanded[indices]
 vals=allscores[indices].tolist();hits=allhits[indices].tolist();strkeys=keys.cpu().tolist()
 merged=sorted(zip(strkeys,vals,hits),key=lambda x:(-x[1],-x[2],x[0]))
 keys=torch.tensor([x[0] for x in merged],dtype=torch.long,device=device)
 best=report_top(None,merged[:20],train,hold,gl,lex,15)
 result={"k":k,"added_glyph":newglyph,"glyphs":gl[:],"candidate_extensions":len(expanded),
 "retained":n,"best":best[:12],
 "bestscore":merged[0][1],
 "topkeys":[{"key":"".join(ABC[j] for j in x[0]),"score":x[1],"hits":x[2]} for x in merged[:100]]}
 results.append(result)
 emit("BEAM_STAGE",result)
emit("BEAM_FINAL",{"schema":"F75R_PROGRESSIVE_GPU_20261019","exact_k6_input":args.input,
"gl":gl,"stages":[{"k":x["k"],"glyph":x["added_glyph"],"best_score":x["bestscore"],
"trained_hits":x["best"][0]["training"]["dictionary_matching_types"],
"heldout_hits":x["best"][0]["untouched_f76r"]["dictionary_matching_types"]} for x in results],
"elapsed_sec":time.time()-t0,
"conclusion":"Discovery only; many-language short-word hits and heldout selection bias require matched null search."})
