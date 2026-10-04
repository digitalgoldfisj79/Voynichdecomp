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
meta=b["load_meta"]()

def qnum(x):
    m=re.search(r"\d+",str(x))
    return int(m.group()) if m else None

def line_q(l):
    m=meta.get(b["normface"](l["folio"]))
    return qnum(m["quire"]) if m else None

def build_quire_holdout(q):
    qn=qnum(q)
    tr=[l for l in LINES if line_q(l)!=qn]
    tg=[l for l in LINES if line_q(l)==qn]
    base=b["fit_struct"](tr); trp=b["attach"](tr,base); md=b["fit_mix"](trp,4,3.0)
    return b["apply_line_model"](tg,base,md)

def analyze(q,TE):
    faces=b["groups_by_face"](TE)
    resolved=[];unresolved=[]
    for qq,a,c in CONTACTS:
        if qnum(qq)!=qnum(q): continue
        aa,ra=b["resolve"](a,faces); cc,rc=b["resolve"](c,faces)
        if aa is None or cc is None:
            unresolved.append({"a":a,"b":c,"a_resolution":ra,"b_resolution":rc});continue
        if aa not in meta or cc not in meta:
            unresolved.append({"a":a,"b":c,"reason":"metadata_missing"});continue
        resolved.append((aa,cc))
    touch=collections.defaultdict(set)
    for a,c in resolved: touch[a].add(c);touch[c].add(a)
    rows=[]
    for a,c in resolved:
        edge="|".join(sorted((a,c)))
        same_bif=(meta[a]["bifolio"]==meta[c]["bifolio"] and qnum(meta[a]["quire"])==qnum(meta[c]["quire"]))
        for t,d in ((a,c),(c,a)):
            lt=b["donor_bits"](faces[t],faces[d])
            tier,ctrl=b["tier_pool"](t,d,faces,meta,touch,exclude_same_bif=True)
            if not ctrl: continue
            lc=[b["donor_bits"](faces[t],faces[x]) for x in ctrl]
            r={"quire":q,"edge":edge,"target":t,"donor":d,"same_bif_touch":same_bif,
               "tier":tier,"controls":ctrl,"touch_loss":lt,"control_loss":float(np.mean(lc)),
               "advantage":float(np.mean(lc)-lt),"nscore":faces[t]["nscore"]}
            sb=[x for x in faces if x!=t and x not in touch.get(t,set()) and x in meta
                and qnum(meta[x]["quire"])==qnum(meta[t]["quire"])
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
    qs=sorted(set(q for q,_,_ in CONTACTS),key=qnum)
    per=[];allrows=[]
    for q in qs:
        TE=build_quire_holdout(q)
        x=analyze(q,TE);per.append({k:v for k,v in x.items() if k!="rows"});allrows.extend(x["rows"])
    out={
      "firewall":"NO_P70",
      "design":"leave-one-quire-out; all touching faces and same-quire controls held out together; frozen MRC architecture and donor L2=10 lambda=.25",
      "per_quire":per,
      "n_directional_rows":len(allrows),
      "all_touch_vs_local_non_touch_diff_bif_controls":pack(allrows,"advantage"),
      "cross_bif_touch_vs_local_non_touch":pack(allrows,"advantage",lambda r:not r["same_bif_touch"]),
      "same_bif_touch_vs_local_non_touch":pack(allrows,"advantage",lambda r:r["same_bif_touch"]),
      "cross_bif_touch_vs_own_bif_non_touch":pack(allrows,"touch_adv_over_own_bif",lambda r:not r["same_bif_touch"]),
      "same_bif_touch_vs_same_bif_non_touch":pack(allrows,"touch_adv_over_own_bif",lambda r:r["same_bif_touch"]),
      "directional_rows":allrows
    }
    print("TOUCHING_FACE_QUIRE_CROSSFIT_JSON="+json.dumps(out,separators=(",",":")),flush=True)
if __name__=="__main__":main()
