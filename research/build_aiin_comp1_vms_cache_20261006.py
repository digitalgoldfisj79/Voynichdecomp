#!/usr/bin/env python3
"""
Build VMS-only cache for AIIN-COMP1.
No German/ReF parsing. Uses exact frozen physical fold loader and strict +P0,
excluding first two tokens per physical line.
"""
import argparse,collections,hashlib,json,re,urllib.request
V_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json"
FOLD_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/420d4363973174760a78000f1049339dbd26fa46/research/hf_emergent_occupancy_fold.py"
FOLD_SHA="e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888"
PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),(25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),(43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),(71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),(94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112)]
BIF_BY_NUM={n:f'B{a:03d}_{b:03d}' for a,b in PAIRS for n in (a,b)}
def parse_num(f):
    m=re.match(r"f(\d+)",str(f)); return int(m.group(1)) if m else None
def canon_sha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

ns={"__name__":"fold"}
src=urllib.request.urlopen(FOLD_URL,timeout=120).read().decode()
exec(compile(src,FOLD_URL,"exec"),ns)
_,folds=ns["load_rows"]()
assert ns["canon_sha"](folds)==FOLD_SHA
obj=json.loads(urllib.request.urlopen(V_URL,timeout=120).read())
byfold={i:collections.Counter() for i in range(5)}
for fol,ld in obj["pages"].items():
    n=parse_num(fol)
    if n not in BIF_BY_NUM: continue
    bif=BIF_BY_NUM[n]
    if bif not in folds: continue
    f=folds[bif]
    for ls,rec in ld.items():
        if str(rec.get("u",""))!="+P0": continue
        toks=[t.lower() for t in rec.get("t",{}).get("ZLZI","").split() if re.fullmatch(r"[a-z]+",t.lower())]
        for pos,t in enumerate(toks):
            if pos>=2: byfold[f][t]+=1
disc=byfold[2]+byfold[3]
final=byfold[0]+byfold[1]
val=byfold[4]
pool=[w for w,n in disc.most_common(1800)]
# prospective family exclusion: all exact terminal "aiin"; keep aiiin distinct.
train=[w for w in pool if not w.endswith("aiin")][:735]
train_set=set(train)
validation=[w for w,n in val.most_common() if w not in train_set and not w.endswith("aiin")][:735]
# held-out final tokens absent from all discovery types and validation, as in GER2 strict final.
disc_all=set(disc); val_all=set(val)
strict=[w for w,n in final.most_common() if w not in disc_all and w not in val_all]
aiin=[w for w in strict if w.endswith("aiin")]
# Build 4-char suffix-family controls from same strict pool, matched by family size and length.
fam=collections.defaultdict(list)
for w in strict:
    if len(w)>=4: fam[w[-4:]].append(w)
aiin_n=len(aiin)
controls=[]
for suf,ws in fam.items():
    if suf=="aiin" or not ws: continue
    if max(2,int(.5*aiin_n)) <= len(ws) <= max(3,int(1.5*aiin_n)):
        controls.append({"suffix":suf,"tokens":ws})
out={
 "schema":"VMS_AIIN_COMP1_V1","fold_sha":FOLD_SHA,
 "train":train,"train_n":len(train),"discovery_pool_n":len(pool),
 "validation":validation,"validation_n":len(validation),
 "strict_final":strict,"strict_final_n":len(strict),
 "aiin":aiin,"aiin_n":aiin_n,
 "suffix_controls":controls,
 "counts":{"disc_tokens":sum(disc.values()),"val_tokens":sum(val.values()),"final_tokens":sum(final.values()),
           "disc_types":len(disc),"val_types":len(val),"final_types":len(final)}
}
raw=json.dumps(out,separators=(",",":"))
ap=argparse.ArgumentParser(); ap.add_argument("--out"); args=ap.parse_args()
if args.out:
    open(args.out,"w").write(raw)
print("AIIN_COMP1_CACHE_SHA="+hashlib.sha256(raw.encode()).hexdigest())
print("AIIN_COMP1_AUDIT="+json.dumps({k:out[k] for k in ["train_n","discovery_pool_n","validation_n","strict_final_n","aiin_n","counts"]},separators=(",",":")))
if not args.out:
    print("AIIN_COMP1_CACHE="+raw)
