#!/usr/bin/env python3
import collections,itertools,json,math,re,urllib.request
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# Frozen MRC upstream model. NO P70.
BASE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
ns={"__name__":"touching_face_base"}
exec(compile(urllib.request.urlopen(BASE_URL,timeout=60).read().decode(),BASE_URL,"exec"),ns)
LINES=ns["LINES"]; fit_struct=ns["fit_struct"]; attach=ns["attach"]; fit_mix=ns["fit_mix"]; K=12

META_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/80d5c9778cb27e3dd20b049ab8804a7bb8832fd2/research/data/physical_metadata_nonp70_20261004.tsv"

# Only downstream-eligible explicit edges from public.vms_touching_face_topology_v01.
# Rosettes aggregate, literal self-contacts, and ambiguous Q19 101v edge are excluded.
CONTACTS=[
("Q9","67v1","67v2"),("Q15","90r1","90r2"),
("Q9","68r1","68v3"),("Q9","68r1","68r3"),("Q9","68r2","68r3"),
("Q10","70r1","70r2"),("Q10","70r1","70v2"),
("Q11","71v","72r1"),("Q11","71v","72r3"),("Q11","71v","72v3"),
("Q11","72r1","72r2"),("Q11","72v2","72v3"),
("Q15","88v","89r1"),("Q15","88v","89v2"),("Q15","89r1","89r2"),
("Q15","89r1","89v2"),("Q15","89v1","90v2"),
("Q17","94v","95r1"),("Q17","94v","95v2"),("Q17","95r1","95r2"),
("Q19","100v","101r"),("Q19","100v","101v2"),
("Q19","101v1","102r1"),("Q19","101v1","102r2"),("Q19","101v1","102v2"),
("Q19","102r1","102r2")
]

L2=10.0
LAM=.25

def state_q(p,b):
    sc=np.log(np.maximum(p,1e-15))+b
    sc-=sc.max(); q=np.exp(sc); return q/q.sum()

def apply_line_model(lines,base,md):
    ap=attach(lines,base); B=md["B"]; HP=md["hp"]; out=[]
    for l in ap:
        P,Y=l["P"],l["Y"]
        if len(Y)<3: continue
        prior=HP[l["house"]]
        A=np.log(prior+1e-15)
        for z in range(4):
            for i in (0,1):
                q=state_q(P[i],B[z]); A[z]+=math.log(max(q[Y[i]],1e-300))
        A-=A.max(); post=np.exp(A); post/=post.sum()
        Q=[]
        for i in range(2,len(Y)):
            q=np.zeros(K,float)
            for z in range(4): q+=post[z]*state_q(P[i],B[z])
            Q.append(q)
        if Q:
            out.append({"P":np.stack(Q),"Y":Y[2:].copy(),
                        "section":l["events"][0]["section"],"bif":l["bif"],
                        "fold":int(l["fold"]),"folio":l["folio"],"line":int(l["line"])})
    return out

def build_split(trainfolds,targetfolds):
    tr=[l for l in LINES if l["fold"] in trainfolds]
    tg=[l for l in LINES if l["fold"] in targetfolds]
    base=fit_struct(tr); trp=attach(tr,base); md=fit_mix(trp,4,3.0)
    return apply_line_model(tr,base,md),apply_line_model(tg,base,md)

def fit_tilt(P,Y,l2=L2):
    P=np.asarray(P,float); Y=np.asarray(Y,int)
    def fg(b):
        bb=b-b.mean()
        sc=np.log(np.maximum(P,1e-15))+bb[None,:]
        z=logsumexp(sc,axis=1); q=np.exp(sc-z[:,None])
        loss=-float(np.sum(sc[np.arange(len(Y)),Y]-z))+.5*l2*float(np.dot(bb,bb))
        D=q.copy(); D[np.arange(len(Y)),Y]-=1
        g=D.sum(0)+l2*bb; g-=g.mean()
        return loss,g
    r=minimize(lambda b:fg(b),np.zeros(K),jac=True,method="L-BFGS-B",
               options={"maxiter":80,"ftol":1e-10})
    return r.x-r.x.mean()

def tilt_prob(p,b,lam=LAM):
    sc=np.log(np.maximum(p,1e-15))+lam*b
    sc-=sc.max(); q=np.exp(sc); return q/q.sum()

def groups_by_face(seqs):
    g=collections.defaultdict(list)
    for s in seqs: g[s["folio"]].append(s)
    out={}
    for f,ls in g.items():
        P=np.vstack([x["P"] for x in ls]); Y=np.concatenate([x["Y"] for x in ls])
        out[f]={"face":f,"seqs":ls,"P":P,"Y":Y,"nscore":len(Y),
                "section":collections.Counter(x["section"] for x in ls).most_common(1)[0][0],
                "tilt":fit_tilt(P,Y)}
    return out

def donor_bits(target,donor):
    loss=0.; n=0
    for s in target["seqs"]:
        for p,y in zip(s["P"],s["Y"]):
            q=tilt_prob(p,donor["tilt"])
            loss+=-math.log2(max(q[int(y)],1e-300)); n+=1
    return loss/n if n else np.nan

def normface(s):
    s=str(s).strip().lower()
    return s if s.startswith("f") else "f"+s

def load_meta():
    lines=urllib.request.urlopen(META_URL,timeout=60).read().decode().splitlines()
    raw=collections.defaultdict(list)
    for i,ln in enumerate(lines):
        if i==0 or not ln.strip(): continue
        f,q,b,c,h,w,l=ln.split("\t")
        raw[normface(f)].append({"quire":q,"bifolio":b,"currier":c,"hand":h,
                                 "word_count":int(w or 0),"line_count":int(l or 0)})
    def sig(vals):
        z=[str(x) for x in vals if x not in (None,"","NA","null","None")]
        return "/".join(sorted(set(z))) if z else "NA"
    def mode(vals):
        z=[str(x) for x in vals if x not in (None,"","NA","null","None")]
        return collections.Counter(z).most_common(1)[0][0] if z else "NA"
    out={}
    for f,rr in raw.items():
        out[f]={"quire":mode([x["quire"] for x in rr]),
                "bifolio":mode([x["bifolio"] for x in rr]),
                "currier":sig([x["currier"] for x in rr]),
                "hand":sig([x["hand"] for x in rr])}
    return out

def resolve(label,faces):
    x=normface(label)
    if x in faces: return x,None
    # Only accept unique prefix resolution (e.g. unsuffixed label where one page exists).
    cand=sorted([f for f in faces if f.startswith(x)])
    if len(cand)==1: return cand[0],"unique_prefix"
    return None,("missing" if not cand else "ambiguous:"+",".join(cand))

def exact_signflip(vals):
    v=np.asarray(vals,float); n=len(v)
    if n==0: return {"B":0}
    obs=float(v.mean())
    if n<=20:
        means=[]
        for mask in range(1<<n):
            s=np.array([1.0 if ((mask>>i)&1) else -1.0 for i in range(n)])
            means.append(float(np.mean(s*v)))
        arr=np.asarray(means)
    else:
        rng=np.random.default_rng(20261004)
        arr=np.mean(rng.choice((-1.,1.),size=(200000,n))*v[None,:],axis=1)
    return {"B":n,"effect":obs,"null_sd":float(arr.std(ddof=1)),
            "effect_over_null_sd":float(obs/arr.std(ddof=1)) if arr.std(ddof=1)>0 else None,
            "p_one_sided":float(np.mean(arr>=obs-1e-15)),
            "block_sd":float(v.std(ddof=1)) if n>1 else 0.0}

def tier_pool(target,donor,faces,meta,touch_neighbors,exclude_same_bif=False):
    tm=meta[target]; dm=meta[donor]
    pool=[]
    for c in faces:
        if c in (target,donor): continue
        if c in touch_neighbors.get(target,set()): continue
        if c not in meta: continue
        cm=meta[c]
        if cm["quire"]!=tm["quire"]: continue
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

def run_on_split(trainfolds,targetfolds,label):
    _,TE=build_split(trainfolds,targetfolds)
    faces=groups_by_face(TE); meta=load_meta()
    # Attach metadata only where exact face exists.
    for f in list(faces):
        if f not in meta:
            # try canonical exact normalization
            nf=normface(f)
            if nf in meta and nf!=f: meta[f]=meta[nf]
    resolved=[]; unresolved=[]
    for q,a,b in CONTACTS:
        aa,ra=resolve(a,faces); bb,rb=resolve(b,faces)
        if aa is None or bb is None:
            unresolved.append({"quire":q,"a":a,"b":b,"a_resolution":ra,"b_resolution":rb})
            continue
        if aa not in meta or bb not in meta:
            unresolved.append({"quire":q,"a":a,"b":b,"a_face":aa,"b_face":bb,"reason":"metadata_missing"})
            continue
        resolved.append((q,aa,bb,ra,rb))
    touch=collections.defaultdict(set)
    for _,a,b,_,_ in resolved:
        touch[a].add(b); touch[b].add(a)

    dirs=[]
    for q,a,b,ra,rb in resolved:
        edge="|".join(sorted((a,b)))
        same_bif=(meta[a]["bifolio"]==meta[b]["bifolio"] and meta[a]["quire"]==meta[b]["quire"])
        for t,d in ((a,b),(b,a)):
            lt=donor_bits(faces[t],faces[d])
            tier,ctrl=tier_pool(t,d,faces,meta,touch,exclude_same_bif=True)
            if not ctrl: continue
            lc=[donor_bits(faces[t],faces[c]) for c in ctrl]
            row={"edge":edge,"quire":q,"target":t,"donor":d,"same_bif_touch":same_bif,
                 "tier":tier,"controls":ctrl,"touch_loss":lt,"control_loss":float(np.mean(lc)),
                 "advantage":float(np.mean(lc)-lt),"nscore":faces[t]["nscore"]}
            # Direct contact vs own-bifolium non-touching donor.
            sb=[c for c in faces if c!=t and c not in touch.get(t,set()) and c in meta
                and meta[c]["quire"]==meta[t]["quire"] and meta[c]["bifolio"]==meta[t]["bifolio"]]
            if sb:
                sb=sorted(sb,key=lambda c:(abs(math.log(max(faces[c]["nscore"],1)/max(faces[d]["nscore"],1))),c))
                c=sb[0]; lb=donor_bits(faces[t],faces[c])
                row.update({"own_bif_non_touch":c,"own_bif_loss":lb,
                            "touch_adv_over_own_bif":float(lb-lt)})
            dirs.append(row)

    def edge_blocks(field,pred=lambda r:True):
        z=collections.defaultdict(list)
        for r in dirs:
            if pred(r) and r.get(field) is not None: z[r["edge"]].append(r[field])
        return {k:float(np.mean(v)) for k,v in z.items()}

    allb=edge_blocks("advantage")
    crossb=edge_blocks("advantage",lambda r:not r["same_bif_touch"])
    sameb=edge_blocks("advantage",lambda r:r["same_bif_touch"])
    direct=edge_blocks("touch_adv_over_own_bif",lambda r:not r["same_bif_touch"] and "touch_adv_over_own_bif" in r)

    return {"label":label,"trainfolds":list(trainfolds),"targetfolds":list(targetfolds),
            "n_faces":len(faces),"resolved_contacts":len(resolved),"unresolved_contacts":unresolved,
            "directional_rows":dirs,
            "all_touch_vs_local_non_touch":{"blocks":allb,"test":exact_signflip(list(allb.values()))},
            "cross_bif_touch_vs_local_non_touch":{"blocks":crossb,"test":exact_signflip(list(crossb.values()))},
            "same_bif_touch_vs_local_non_touch":{"blocks":sameb,"test":exact_signflip(list(sameb.values()))},
            "cross_bif_touch_vs_own_bif_non_touch":{"blocks":direct,"test":exact_signflip(list(direct.values()))}}

def main():
    primary=run_on_split((2,3,4),(0,1),"untouched_folds01")
    out={"firewall":"NO_P70","model":"frozen MRC latent-line baseline; exact-face donor tilt L2=10 lambda=.25",
         "contact_source":"public.vms_touching_face_topology_v01 downstream_eligible edges, frozen manually into script",
         "primary":primary}
    print("TOUCHING_FACE_DISCRIMINATOR_JSON="+json.dumps(out,separators=(",",":")),flush=True)

if __name__=="__main__": main()
