#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["wordfreq>=3.1,<4","Unidecode>=1.3,<2"]
# ///
"""Inspect fully exhaustive K8 shortlist on novel f76r word types; not a promotion.
Warning: sorting by heldout results is post-hoc exploration, never validation.
"""
import collections,hashlib,json,re,urllib.request
from wordfreq import top_n_list
from unidecode import unidecode
ROOT="https://raw.githubusercontent.com/"
BASE=ROOT+"digitalgoldfisj79/Voynichdecomp/"
MANIFEST=BASE+"experiment/f75r-progressive-cipher-ladder-20261018/research/f75r_cipher_ladder_20261018/results/exact8_merged_20261019.json"
TRANS=BASE+"main/voynich_transcriptions_slim.json"
A=("ckh","cth","cph","cfh","ikh","ith","iph","ifh","ch","sh")
LF=["it","de","cs","fr","es","pl","nl","ro","sk","sl","hu","el","ru","uk","pt","ca","en","he"]
EX={"la":"https://raw.githubusercontent.com/UniversalDependencies/UD_Latin-Perseus/master/la_perseus-ud-train.conllu",
"yi":"https://raw.githubusercontent.com/UniversalDependencies/UD_Yiddish-YiTB/master/yi_yitb-ud-train.conllu"}
def emit(k,x):print(k+"="+json.dumps(x,ensure_ascii=False,separators=(",",":")),flush=True)
def sget(u):return urllib.request.urlopen(u,timeout=90).read()
def norm(t):return re.sub("[^a-z]","",unidecode(t.lower().replace("ß","ss")).lower())
def tok(w):
 r=[];i=0
 while i<len(w):
  x=next((x for x in A if w.startswith(x,i)),w[i]);r.append(x);i+=len(x)
 return r
def lex():
 d=collections.defaultdict(set)
 for l in LF:
  for w in top_n_list(l,25000):
   x=norm(w)
   if 3<=len(x)<=10 and (len(x)==len(w) or l in ("el","ru","uk","he")):d[x].add(l)
 for l,u in EX.items():
  for row in sget(u).decode("utf-8","replace").splitlines():
   x=row.split("\t")
   if len(x)!=10 or not x[0].isdigit() or x[3]=="PUNCT":continue
   w=norm(x[1])
   if 3<=len(w)<=10:d[w].add(l)
 return d
def main():
 m=json.loads(sget(MANIFEST))
 assert m["exact_complete"] and m["assignments"]==62990928000
 w=json.loads(sget(TRANS))["pages"]
 def words(f):return [x for z in w[f].values() for x in re.findall("[a-z]+",z.get("t",{}).get("TTLI",""))]
 train=words("f75r");test=words("f76r")
 gl=m["glyphs"];known=set(gl)
 covtrain=collections.Counter(w for w in train if len(tok(w))>=3 and set(tok(w)).issubset(known))
 nov=collections.Counter(w for w in test if len(tok(w))>=3 and set(tok(w)).issubset(known) and w not in covtrain)
 repeated=collections.Counter(w for w in test if w in covtrain)
 dictionary=lex()
 emit("NOVEL_AUDIT",{"exact_keys":m["assignments"],"top_candidates":len(m["topkeys"]),
 "train_types":len(covtrain),"novel_test_types":len(nov),"novel_test_tokens":sum(nov.values()),"repeated_test_tokens":sum(repeated.values()),"lex_forms":len(dictionary)})
 metrics=[]
 for ix,v in enumerate(m["topkeys"]):
  enc={g:c for g,c in zip(gl,v["key"])}
  def decode(w):return "".join(enc[x] for x in tok(w))
  def matches(wc):
   out=[]
   for w,n in wc.items():
    p=decode(w)
    if p in dictionary:out.append({"cipher":w,"decoded":p,"n":n,"lang":sorted(dictionary[p])})
   return out
  tr=matches(covtrain);nv=matches(nov);lang=collections.Counter()
  for w in nv:
   for x in w["lang"]:lang[x]+=1
  lc=lang.most_common()
  metrics.append({"train_rank":ix+1,"key":v["key"],"train_search_score":v["score"],
   "train_match_types":len(tr),"train_match_tokens":sum(x["n"] for x in tr),
   "novel_match_types":len(nv),"novel_match_tokens":sum(x["n"] for x in nv),
   "dominant_novel_language":lc[:5],
   "novel_matches":sorted(nv,key=lambda x:-x["n"])[:18]})
 # Freeze valid ordered shortlist, don't reselect on test.
 emit("NOVEL_BY_SEARCH_RANK",{"top":metrics[:30]})
 # Explicitly labeled POST-HOC diagnostic; these are not independently selected candidates
 best=sorted(metrics,key=lambda x:(-x["novel_match_types"],-x["train_search_score"]))[:12]
 emit("NOVEL_POSTHOC_DIAGNOSTIC",{"warning":"Re-ranking 2600 keys on heldout words leaks validation; do not use for inference.",
   "top":best})
if __name__=="__main__":main()
