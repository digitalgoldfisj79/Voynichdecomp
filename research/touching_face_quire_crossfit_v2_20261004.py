# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-learn"]
# ///
#!/usr/bin/env python3
import collections,json,math,re,urllib.request
import numpy as np

BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/b337f81a606dacee3066c0739f613484db9cdeba/research/touching_face_discriminator_20261004.py"
b={"__name__":"touch_base"}
exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),b)

LINES=b["LINES"]; CONTACTS=b["CONTACTS"]
RAW_META=b["load_meta"]()

QRANGES={
 "Q9":set(range(67,69)),
 "Q10":set(range(69,71)),
 "Q11":set(range(71,73)),
 "Q15":set(range(87,91)),
 "Q17":set(range(93,97)),
 "Q19":set(range(99,103)),
}
PAIRS=[(67,68),(69,70),(71,72),(87,90),(88,89),(93,96),(94,95),(99,102),(100,101)]
BIF={}
for a,c in PAIRS:
    for n in (a,c): BIF[n]=f"{a}_{c}"

def num(x):
    m=re.match(r"f?(\d+)",str(x).lower())
    return int(m.group(1)) if m else None

def in_quire(f,q):
    return num(f) in QRANGES[q]

def build_quire_holdout(q):
    tr=[l for l in LINES if not in_quire(l["folio"],q)]
    tg=[l for l in LINES if in_quire(l["folio"],q)]
    base=b["fit_struct"](tr); trp=b["attach"](tr,base); md=b["fit_mix"](trp,4,3.0)
    return b["apply_line_model"](tg,base,md)

def make_meta(q,faces):
    M={}
    for f in faces:
        n=num(f); old=RAW_META.get(b["normface"](f),{})
        M[f]={
          "quire":q,
          "bifolio":BIF.get(n,f"UNMAPPED_{n}"),
          "currier":old.get("currier","NA"),
          "hand":old.get("hand","NA"),
        }
    return M

def tier_pool(target,donor,faces,meta,touch_neighbors,exclude_same_bif=False):
    tm=meta[target]; dm=meta[donor]
    pool=[]
    for c in faces:
        if c in (target,donor) or c in touch_neighbors.get(target,set()): continue
        cm=meta[c]
        if exclude_same_bif and cm["bifolio"]==tm["bifolio"]: continue
        pool.append(c)
    tests=[
      lambda c:faces[c]["section"]==faces[donor]["section"] and meta[c]["currier"]==dm["currier"] and meta[c]["hand"]==dm["hand"],
      lambda c:faces[c]["section"]==faces[donor]["section"] and meta[c]["currier"]==dm["currier"],
      lambda c:faces[c]["section"]==faces[donor]["section"] and meta[c]["hand"]==dm["hand"],
      lambda c:faces[c]["section"]==faces[donor]["section"],
      lambda c:True
    ]
    for ti,fn in enumerate(tests,1):
        z=[c for c in pool if fn(c)]
        if z:
            z=sorted(z,key=lambda c:(abs(math.log(max(faces[c]["nscore"],1)/max(faces[donor]["nscore"],1))),c))
            return ti,z[:min(3,len(z))]
    return None,[]

def analyze(q,TE):
    faces=b["groups_by_face"](TE); meta=make_meta(q,faces)
    resolved=[];unresolved=[]
    for qq,a,c in CONTACTS:
        if qq!=q: continue
        aa,ra=b["resolve"](a,faces); cc,rc=b["resolve"](c,faces)
        if aa is None or cc is None:
            unresolved.append({"a":a,"b":c,"a_resolution":ra,"b_resolution":rc});continue
        resolved.append((aa,cc))
    touch=collections.defaultdict(set)
    for a,c in resolved: touch[a].add(c);touch[c].add(a)
    rows=[]
    for a,c in resolved:
        edge="|".join(sorted((a,c)))
        same_bif=(meta[a]["bifolio"]==meta[c]["bifolio"])
        for t,d in ((a,c),(c,a)):
            lt=b["donor_bits"](faces[t],faces[d])
            tier,ctrl=tier_pool(t,d,faces,meta,touch,exclude_same_bif=True)
            if not ctrl: continue
            lc=[b["donor_bits"](faces[t],faces[x]) for x in ctrl]
            r={"quire":q,"edge":edge,"target":t,"donor":d,"same_bif_touch":same_bif,
               "tier":tier,"controls":ctrl,"touch_loss":lt,"control_loss":float(np.mean(lc)),
               "advantage":float(np.mean(lc)-lt),"nscore":faces[t]["nscore"]}
            sb=[x for x in faces if x!=t and x not in touch.get(t,set())
                and meta[x]["bifolio"]==meta[t]["bifolio"]]
            if sb:
                sb=sorted(sb,key=lambda x:(abs(math.log(max(faces[x]["nscore"],1)/max(faces[d]["nscore"],1))),x))
                x=sb[0]; lb=b["donor_bits"](faces[t],faces[x])
                r.update({"own_bif_non_touch":x,"own_bif_loss":lb,
                          "touch_adv_over_own_bif":float(lb-lt)})
            rows.append(r)
    return {"quire":q,"n_faces":len(faces),"resolved":len(resolved),"unresolved":unresolved,"rows":rows}

def blocks(rows,field,pred=lambda r:True):
    z=collections.defaultdict(list)
    for r in rows:
        if pred(r) and field in r:z[r["edge"]].append(r[field])
    return {k:float(np.mean(v)) for k,v in z.items()}

def pack(rows,field,pred=lambda r:True):
    x=blocks(rows,field,pred)
    return {"blocks":x,"test":b["exact_signflip"](list(x.values()))}

def main():
    per=[];allrows=[]
    for q in ("Q9","Q10","Q11","Q15","Q17","Q19"):
        x=analyze(q,build_quire_holdout(q))
        per.append({k:v for k,v in x.items() if k!="rows"}); allrows.extend(x["rows"])
    out={
      "firewall":"NO_P70",
      "design":"leave-one-codicological-quire-out by frozen folio ranges; exact face-level donors; frozen MRC architecture; donor L2=10 lambda=.25",
      "per_quire":per,
      "n_directional_rows":len(allrows),
      "all_touch_vs_local_non_touch_diff_bif_controls":pack(allrows,"advantage"),
      "cross_bif_touch_vs_local_non_touch":pack(allrows,"advantage",lambda r:not r["same_bif_touch"]),
      "same_bif_touch_vs_local_non_touch":pack(allrows,"advantage",lambda r:r["same_bif_touch"]),
      "cross_bif_touch_vs_own_bif_non_touch":pack(allrows,"touch_adv_over_own_bif",lambda r:not r["same_bif_touch"]),
      "same_bif_touch_vs_same_bif_non_touch":pack(allrows,"touch_adv_over_own_bif",lambda r:r["same_bif_touch"]),
      "directional_rows":allrows
    }
    print("TOUCHING_FACE_QUIRE_CROSSFIT_V2_JSON="+json.dumps(out,separators=(",",":")),flush=True)
if __name__=="__main__":main()
