#!/usr/bin/env python3
import collections,json,math,urllib.request,itertools
import numpy as np

# Current MRC hierarchy implementation only; no p70 features/metadata.
BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/1b4d2812b86e44f97151305b589d7d01dbb409af/research/boundary_reset_hierarchy_20261004.py"
ns={"__name__":"mrc_hierarchy_base"}
exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),ns)

# Frozen independent physical metadata export; explicitly not p70-derived.
META_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/80d5c9778cb27e3dd20b049ab8804a7bb8832fd2/research/data/physical_metadata_nonp70_20261004.tsv"

def digits(s):
    z="".join(ch for ch in str(s) if ch.isdigit())
    return int(z) if z else None

# page metadata -> numbered-leaf metadata signatures
raw=collections.defaultdict(list)
for i,ln in enumerate(urllib.request.urlopen(META_URL,timeout=60).read().decode().splitlines()):
    if i==0 or not ln.strip():continue
    f,q,b,c,h,w,l=ln.split("\t")
    n=digits(f)
    if n is None:continue
    raw[n].append({"folio":f,"quire":q,"bifolio":b,"currier":c,"hand":h,
                   "word_count":int(w or 0),"line_count":int(l or 0)})

def sig(vals):
    vals=[str(x) for x in vals if x not in (None,"","NA","null","None")]
    return "/".join(sorted(set(vals))) if vals else "NA"
def mode(vals):
    vals=[str(x) for x in vals if x not in (None,"","NA","null","None")]
    return collections.Counter(vals).most_common(1)[0][0] if vals else "NA"

META={}
for n,rr in raw.items():
    META[n]={
      "num":n,
      "quire":mode([x["quire"] for x in rr]),
      "bifolio":mode([x["bifolio"] for x in rr]),
      "currier":sig([x["currier"] for x in rr]),
      "hand":sig([x["hand"] for x in rr]),
      "word_count":sum(x["word_count"] for x in rr),
      "line_count":sum(x["line_count"] for x in rr)
    }

# independent true-mate map from bifolio id
bg=collections.defaultdict(set)
for n,m in META.items():
    if m["bifolio"]!="NA":bg[(m["quire"],m["bifolio"])].add(n)
MATE={}
for b,ss in bg.items():
    zz=sorted(ss)
    if len(zz)==2:
        MATE[zz[0]]=zz[1];MATE[zz[1]]=zz[0]

# Frozen current MRC heldout residuals. Same parameters as parent packet PASS.
TR,TE=ns["build_split"]((2,3,4),(0,1))
L2=10.0
LAM=.25
F=ns["packet_tilts"](TE,L2)

# enrich folio groups
for n,x in F.items():
    x["nscore"]=sum(len(s["Y"]) for s in x["seqs"])
    x["meta"]=META.get(n,{"quire":"NA","bifolio":"NA","currier":"NA","hand":"NA"})
    
def loss_rate(target,donor_num):
    loss,nn=ns["donor_bits"](target,F[donor_num]["tilt"],LAM)
    return loss/nn

def tier_candidates(t,m):
    tm=F[t]; mm=F[m]["meta"]
    cq=[c for c in F if c not in (t,m) and F[c]["meta"]["quire"]==tm["meta"]["quire"]]
    tests=[
      lambda c:F[c]["section"]==F[m]["section"] and F[c]["meta"]["currier"]==mm["currier"] and F[c]["meta"]["hand"]==mm["hand"],
      lambda c:F[c]["section"]==F[m]["section"] and F[c]["meta"]["currier"]==mm["currier"],
      lambda c:F[c]["section"]==F[m]["section"] and F[c]["meta"]["hand"]==mm["hand"],
      lambda c:F[c]["section"]==F[m]["section"],
      lambda c:True
    ]
    for ti,fn in enumerate(tests,1):
        z=[c for c in cq if fn(c)]
        if z:return ti,z
    return None,[]

def physical_match_distance(t,m,c):
    dsep=abs(abs(c-t)-abs(m-t))
    n_m=max(F[m]["nscore"],1);n_c=max(F[c]["nscore"],1)
    dlen=abs(math.log(n_c/n_m))
    return dsep+dlen

def exact_signflip(vals):
    v=np.asarray(vals,float);B=len(v)
    if B==0:return {"B":0,"p_add":None,"obs":None}
    obs=float(v.mean())
    total=1<<B;hit=0
    # exact exhaustive, batchless; <=2^20 in this test.
    for mask in range(total):
        s=0.0
        for i,x in enumerate(v):
            s += x if ((mask>>i)&1) else -x
        if s/B >= obs-1e-15:hit+=1
    return {"B":B,"n_permutations":total,"obs":obs,"p_exact":hit/total,
            "sd":float(v.std(ddof=1)) if B>1 else 0.0,
            "mean_over_sd":float(obs/v.std(ddof=1)) if B>1 and v.std(ddof=1)>0 else None}

rows=[]
for t in sorted(F):
    m=MATE.get(t)
    if m not in F:continue
    tier,cands=tier_candidates(t,m)
    if not cands:continue
    cands=sorted(cands,key=lambda c:(physical_match_distance(t,m,c),c))
    chosen=cands[:min(3,len(cands))]
    lm=loss_rate(F[t],m)
    lc=[loss_rate(F[t],c) for c in chosen]
    allq=[c for c in F if c not in (t,m) and F[c]["meta"]["quire"]==F[t]["meta"]["quire"]]
    allq_losses=[(c,loss_rate(F[t],c)) for c in allq]
    nearest=min(allq,key=lambda c:(abs(c-t),c)) if allq else None
    ln=loss_rate(F[t],nearest) if nearest is not None else None
    rank=1+sum(l<lm-1e-15 for c,l in allq_losses)
    rows.append({
      "target":t,"mate":m,"bifolio":F[t]["meta"]["quire"]+":"+F[t]["meta"]["bifolio"],"tier":tier,
      "chosen":chosen,"mate_loss":lm,"control_loss":float(np.mean(lc)),
      "advantage":float(np.mean(lc)-lm),
      "nearest":nearest,"nearest_loss":ln,
      "nearest_advantage":(ln-lm) if ln is not None else None,
      "allq_n":len(allq),"rank":rank,
      "cross_register":bool(F[t]["meta"]["currier"]!=F[m]["meta"]["currier"] or F[t]["meta"]["hand"]!=F[m]["meta"]["hand"])
    })

def blockify(field,subset=None):
    d=collections.defaultdict(list)
    for r in rows:
        if subset is not None and not subset(r):continue
        v=r.get(field)
        if v is not None:d[r["bifolio"]].append(v)
    return {b:float(np.mean(v)) for b,v in d.items()}

primary_blocks=blockify("advantage")
nearest_blocks=blockify("nearest_advantage")
cross_blocks=blockify("advantage",lambda r:r["cross_register"])

# all-same-quire true-mate ranks
ranks=[r["rank"] for r in rows]
top1=sum(x==1 for x in ranks)

# Generic same-quire effect vs cross-quire same section/currier/hand.
qrows=[]
for t in sorted(F):
    m=MATE.get(t)
    if m not in F:continue
    sameq=[c for c in F if c not in (t,m) and F[c]["meta"]["quire"]==F[t]["meta"]["quire"]]
    cross=[c for c in F if c!=t and F[c]["meta"]["quire"]!=F[t]["meta"]["quire"]
           and F[c]["section"]==F[t]["section"]
           and F[c]["meta"]["currier"]==F[t]["meta"]["currier"]
           and F[c]["meta"]["hand"]==F[t]["meta"]["hand"]]
    if not sameq or not cross:continue
    sq=float(np.mean([loss_rate(F[t],c) for c in sameq]))
    # match donor token mass to median same-quire donor mass; up to 3
    target_mass=float(np.median([F[c]["nscore"] for c in sameq]))
    cross=sorted(cross,key=lambda c:(abs(math.log(max(F[c]["nscore"],1)/max(target_mass,1))),abs(c-t),c))[:3]
    cr=float(np.mean([loss_rate(F[t],c) for c in cross]))
    qrows.append({"target":t,"bifolio":F[t]["meta"]["quire"]+":"+F[t]["meta"]["bifolio"],"sameq_loss":sq,"cross_loss":cr,
                  "sameq_advantage":cr-sq,"cross":cross})
qd=collections.defaultdict(list)
for r in qrows:qd[r["bifolio"]].append(r["sameq_advantage"])
qblocks={b:float(np.mean(v)) for b,v in qd.items()}

tier_counts=dict(collections.Counter(r["tier"] for r in rows))
out={
 "firewall":"NO_P70",
 "metadata_source":"vms_boundary_localisation_paragraph_v01 frozen export commit 80d5c9778cb27e3dd20b049ab8804a7bb8832fd2",
 "mrc":{"L2":L2,"lambda":LAM,"heldout_folios":len(F)},
 "primary":{
   "target_directions":len(rows),"bifolium_blocks":len(primary_blocks),"tier_counts":tier_counts,
   "mean_directional_advantage_bits":float(np.mean([r["advantage"] for r in rows])) if rows else None,
   "block_test":exact_signflip(list(primary_blocks.values())),
   "blocks":primary_blocks
 },
 "nearest_same_quire":{
   "block_test":exact_signflip(list(nearest_blocks.values())),
   "blocks":nearest_blocks
 },
 "all_same_quire_rank":{
   "targets":len(ranks),"top1":top1,
   "median_rank":float(np.median(ranks)) if ranks else None,
   "mean_rank":float(np.mean(ranks)) if ranks else None
 },
 "generic_quire_effect":{
   "directions":len(qrows),"blocks":len(qblocks),
   "block_test":exact_signflip(list(qblocks.values())),
   "block_values":qblocks
 },
 "cross_register":{
   "directions":sum(r["cross_register"] for r in rows),"blocks":len(cross_blocks),
   "block_test":exact_signflip(list(cross_blocks.values())) if len(cross_blocks)>=3 else {"underpowered":True,"B":len(cross_blocks)},
   "block_values":cross_blocks
 },
 "directional_rows":rows
}
out["decision"]={
 "mate_specific_pass":bool(out["primary"]["block_test"].get("p_exact",1)<.01 and out["primary"]["block_test"].get("obs",-1)>0),
 "nearest_pass":bool(out["nearest_same_quire"]["block_test"].get("p_exact",1)<.01 and out["nearest_same_quire"]["block_test"].get("obs",-1)>0),
 "generic_quire_present":bool(out["generic_quire_effect"]["block_test"].get("p_exact",1)<.01 and out["generic_quire_effect"]["block_test"].get("obs",-1)>0)
}
print("PACKET_SOURCE_PRODUCTION_JSON="+json.dumps(out,separators=(",",":")),flush=True)
