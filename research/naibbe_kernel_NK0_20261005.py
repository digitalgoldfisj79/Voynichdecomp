#!/usr/bin/env python3
# NK0 — Naibbe positive-control anatomy in frozen Voynich FORM coordinates.
# Preregistered programme: voynich_naibbe_kernel_programme_20261005
# No claim that literal Naibbe generated MS408.
import csv, io, json, math, urllib.request, collections
import numpy as np

SEED=20261005
NPERM=20000
NMATCH=2000
GRESHKO_COMMIT="f2675ec5dd275268bc64dd48ea64fc0e0e9827a2"
TABLE_URL=f"https://raw.githubusercontent.com/greshko/naibbe-cipher/{GRESHKO_COMMIT}/references/naibbe_tables.csv"
NAIBBE_URL=f"https://raw.githubusercontent.com/greshko/naibbe-cipher/{GRESHKO_COMMIT}/naibbe_v2.py"
FORM_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/74ae7aa0088dc07436000f777d9c5d7d705730c2/research/latent_line_state_test_20261004.py"
TABLES=["alpha","beta1","beta2","beta3","gamma1","gamma2"]
STATES=["unigram","prefix","suffix"]
CARD_WEIGHTS={
 "52":{"alpha":20,"beta1":8,"beta2":8,"beta3":8,"gamma1":4,"gamma2":4},
 "78":{"alpha":28,"beta1":14,"beta2":11,"beta3":11,"gamma1":7,"gamma2":7},
}
GALLOW=set("fkpt")
K8_GROUPS=[(0,),(1,),(2,10),(3,7,8),(4,11),(5,),(6,),(9,)]
G_OF={c:g for g,cs in enumerate(K8_GROUPS) for c in cs}

m={"__name__":"latent"}
exec(compile(urllib.request.urlopen(FORM_URL,timeout=60).read().decode(),FORM_URL,"exec"),m)
rows=m["rows"]; segment=m["segment"]; FORM_ST=m["ST"]
real_types=sorted({r["token"] for r in rows})

txt=urllib.request.urlopen(TABLE_URL,timeout=60).read().decode("utf-8-sig")
rr=list(csv.DictReader(io.StringIO(txt)))
parsed=[]
for r in rr:
    code=r["code"].strip(); glyph=r["glyphs"].strip()
    parts=code.split("_")
    if len(parts)<3: continue
    state,table=parts[0],parts[1]
    letter="_".join(parts[2:])
    if state not in STATES or table not in TABLES or not glyph: continue
    try:
        pcs=segment(glyph)
    except Exception as e:
        parsed.append({"state":state,"table":table,"letter":letter,"glyph":glyph,"error":str(e)})
        continue
    cls=[FORM_ST[p] for p in pcs]
    route=tuple((G_OF[cls[i]],cls[i+1]) for i in range(len(cls)-1))
    parsed.append({
      "state":state,"table":table,"letter":letter,"glyph":glyph,"pieces":pcs,
      "entry":cls[0],"final":cls[-1],"piece_count":len(pcs),"raw_length":len(glyph),
      "gallows_presence":int(any(ch in GALLOW for ch in glyph)),
      "gallows_count":sum(ch in GALLOW for ch in glyph),
      "route":route,
      "attested_exact":glyph in set(real_types)
    })

bad=[x for x in parsed if "error" in x]
good=[x for x in parsed if "error" not in x]
if bad:
    raise RuntimeError("Unsegmentable Naibbe codewords: "+json.dumps(bad[:20]))

letters=sorted({x["letter"] for x in good})
LID={x:i for i,x in enumerate(letters)}

def factor(vals):
    mp={}; out=[]
    for v in vals:
        key=json.dumps(v,separators=(",",":"),sort_keys=True) if isinstance(v,(dict,list)) else repr(v)
        if key not in mp: mp[key]=len(mp)
        out.append(mp[key])
    return np.asarray(out,dtype=np.int32),len(mp)

def mi_disc(y,f,w,ny=None,nf=None):
    y=np.asarray(y,dtype=np.int32); f=np.asarray(f,dtype=np.int32); w=np.asarray(w,float)
    if ny is None: ny=int(y.max())+1
    if nf is None: nf=int(f.max())+1
    J=np.bincount(y*nf+f,weights=w,minlength=ny*nf).reshape(ny,nf)
    s=J.sum()
    if s<=0:return float("nan")
    P=J/s; py=P.sum(1); pf=P.sum(0)
    ii,jj=np.nonzero(P>0)
    return float(np.sum(P[ii,jj]*np.log2(P[ii,jj]/(py[ii]*pf[jj]))))

FEATURES=["entry","route","final","piece_count","raw_length","gallows_presence","gallows_count"]

def prep_state(state, subset=None):
    a=[x for x in good if x["state"]==state and (subset is None or subset(x))]
    y=np.array([LID[x["letter"]] for x in a],dtype=np.int32)
    table=np.array([TABLES.index(x["table"]) for x in a],dtype=np.int16)
    feats={}
    for k in FEATURES:
        vals=[list(x[k]) if k=="route" else x[k] for x in a]
        feats[k]=factor(vals)
    return a,y,table,feats

def perm_audit(state, subset=None, nperm=NPERM):
    a,y,t,feats=prep_state(state,subset)
    groups=[np.where(t==j)[0] for j in range(len(TABLES))]
    rng=np.random.default_rng(SEED+sum(ord(c) for c in state)+(0 if subset is None else 10000))
    out={"n":len(a),"letters":len(set(y.tolist())),"tables":{TABLES[j]:int(len(groups[j])) for j in range(6)},"decks":{}}
    for deck,wmap in CARD_WEIGHTS.items():
        w=np.array([wmap[x["table"]] for x in a],float)
        rec={}
        obs={k:mi_disc(y,ff[0],w,len(letters),ff[1]) for k,ff in feats.items()}
        null={k:np.empty(nperm,float) for k in FEATURES}
        yp=y.copy()
        for b in range(nperm):
            for ix in groups:
                if len(ix)>1: yp[ix]=rng.permutation(y[ix])
            for k,(fc,nf) in feats.items():
                null[k][b]=mi_disc(yp,fc,w,len(letters),nf)
        for k in FEATURES:
            mu=float(null[k].mean()); sd=float(null[k].std(ddof=1))
            rec[k]={
              "mi_bits":float(obs[k]),"null_mean":mu,"null_sd":sd,
              "z":float((obs[k]-mu)/sd) if sd>0 else None,
              "p_ge":float((1+np.sum(null[k]>=obs[k]))/(nperm+1))
            }
        out["decks"][deck]=rec
    return out

# Exact-token ablation: remove codewords that are exact real VMS token types.
primary={s:perm_audit(s) for s in STATES}
ablation={}
for s in STATES:
    kept=[x for x in good if x["state"]==s and not x["attested_exact"]]
    counts=collections.Counter(x["table"] for x in kept)
    if len(kept)>=30 and min([counts.get(t,0) for t in TABLES])>=3:
        ablation[s]=perm_audit(s,lambda x:not x["attested_exact"],nperm=5000)
    else:
        ablation[s]={"status":"insufficient_after_exact_attested_removal","n":len(kept),"per_table":dict(counts)}

# Length-matched random real-VMS type control.
pools=collections.defaultdict(list)
for tok in real_types:
    try:
        pcs=segment(tok)
    except Exception:
        continue
    cls=[FORM_ST[p] for p in pcs]
    pools[len(tok)].append({
      "entry":cls[0],"final":cls[-1],"piece_count":len(pcs),"raw_length":len(tok),
      "gallows_presence":int(any(ch in GALLOW for ch in tok)),
      "gallows_count":sum(ch in GALLOW for ch in tok),
      "route":tuple((G_OF[cls[i]],cls[i+1]) for i in range(len(cls)-1))
    })

def matched_control(state,nrep=NMATCH):
    a,y,t,_=prep_state(state)
    rng=np.random.default_rng(SEED+50000+sum(ord(c) for c in state))
    out={"reps":nrep,"decks":{}}
    for deck,wmap in CARD_WEIGHTS.items():
        w=np.array([wmap[x["table"]] for x in a],float)
        obs={}
        for k in FEATURES:
            fc,nf=factor([list(x[k]) if k=="route" else x[k] for x in a])
            obs[k]=mi_disc(y,fc,w,len(letters),nf)
        vals={k:np.empty(nrep,float) for k in FEATURES}
        for b in range(nrep):
            samp=[]
            for x in a:
                pool=pools.get(x["raw_length"],[])
                if not pool:
                    # nearest-length fallback is explicit and counted below
                    lens=sorted(pools,key=lambda z:abs(z-x["raw_length"]))
                    pool=pools[lens[0]]
                samp.append(pool[int(rng.integers(len(pool)))])
            for k in FEATURES:
                fc,nf=factor([list(z[k]) if k=="route" else z[k] for z in samp])
                vals[k][b]=mi_disc(y,fc,w,len(letters),nf)
        rec={}
        for k in FEATURES:
            mu=float(vals[k].mean());sd=float(vals[k].std(ddof=1))
            rec[k]={
              "observed_mi_bits":float(obs[k]),"matched_mean":mu,"matched_sd":sd,
              "z_vs_length_matched_real_types":float((obs[k]-mu)/sd) if sd>0 else None,
              "p_ge":float((1+np.sum(vals[k]>=obs[k]))/(nrep+1))
            }
        out["decks"][deck]=rec
    return out

matched={s:matched_control(s) for s in STATES}

summary={
 "programme":"voynich_naibbe_kernel_programme_20261005",
 "phase":"NK0",
 "status":"complete",
 "seed":SEED,"nperm":NPERM,"nmatch":NMATCH,
 "greshko_commit":GRESHKO_COMMIT,
 "greshko_table_url":TABLE_URL,
 "greshko_naibbe_url":NAIBBE_URL,
 "frozen_form_url":FORM_URL,
 "card_weights":CARD_WEIGHTS,
 "n_codewords":len(good),"n_letters":len(letters),"letters":letters,
 "exact_attested_fraction_by_state":{
   s:float(np.mean([x["attested_exact"] for x in good if x["state"]==s])) for s in STATES
 },
 "primary_permutation":primary,
 "exact_attested_ablation":ablation,
 "length_matched_real_type_control":matched,
 "interpretation_guardrail":"Positive-control anatomy only; Naibbe tables are Voynich-derived/tuned. Literal cards/prefix-suffix ontology are not promoted."
}
print("NAIBBE_NK0_JSON="+json.dumps(summary,separators=(",",":")),flush=True)
