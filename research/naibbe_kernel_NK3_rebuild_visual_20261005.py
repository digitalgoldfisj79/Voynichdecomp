#!/usr/bin/env python3
"""NK3 rebuild — DINOv3 visual-family arm, using SHA-frozen Nomic projection of private HF canonical store."""
import collections, hashlib, io, json, re, urllib.request, os
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import requests
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import Ridge

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/b9795285e87c9d7332c29abe5123237100bce143/research/naibbe_kernel_NK3_rebuild_core_20261005.py"
c={"__name__":"nk3r_core"};exec(compile(urllib.request.urlopen(CORE_URL,timeout=120).read().decode(),CORE_URL,"exec"),c)
build_rows=c["build_rows"];internal_feature=c["internal_feature"];evaluate_family=c["evaluate_family"]
fnum=c["fnum"];folds=c["folds"];BIF=c["BIF"];SEED=c["SEED"]

ORG="edwardbozzard";SLUG="voynich-dinov3-words-v3-overlay"
EXPECTED_SHA="e0ace0453be66ad5e32ed494fbcf31f027bcca247b4e1a6b5ec5454c64f532de"
KGRID=(4,8,12,16,24,32)

def load_atlas():
    meta=requests.get(f"https://api-atlas.nomic.ai/v1/project/{ORG}/{SLUG}",timeout=120).json()
    pid=meta["id"];pr=meta["atlas_indices"][0]["projections"][0]["id"]
    base=f"https://api-atlas.nomic.ai/v1/project/{pid}/index/projection/{pr}/quadtree"
    man=feather.read_table(io.BytesIO(requests.get(base+"/manifest.feather",timeout=120).content))
    rows=[];embs=[];words=[]
    for ni,key in enumerate(man["key"].to_pylist()):
        main=feather.read_table(io.BytesIO(requests.get(f"{base}/{key}.feather",timeout=180).content)).to_pandas()
        et=feather.read_table(io.BytesIO(requests.get(f"{base}/{key}.embeddings.feather",timeout=180).content))
        wt=feather.read_table(io.BytesIO(requests.get(f"{base}/{key}.d29yZA==.feather",timeout=180).content))
        assert len(main)==len(et)==len(wt)
        rows.append(main);embs.append(np.asarray(et["_embeddings"].to_pylist(),dtype=np.float32));words.extend(wt["word"].to_pylist())
    md=pd.concat(rows,ignore_index=True);X=np.concatenate(embs);md["word"]=words
    keep=~md["_row_id"].duplicated();md=md.loc[keep].copy();X=X[np.asarray(keep)]
    o=np.argsort(md["_row_id"].to_numpy());md=md.iloc[o].reset_index(drop=True);X=X[o]
    assert len(md)==37886,(len(md),)
    X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    sha=hashlib.sha256(X.astype(np.float32).tobytes()).hexdigest()
    return md,X,sha,{"project_id":pid,"projection_id":pr}

def canon_num(f):
    m=re.search(r"(\d+)",str(f));return int(m.group(1)) if m else None

rows=build_rows("TTLI")
strict_allowed=collections.defaultdict(set)
for r in rows:strict_allowed[fnum(r["folio"])].add(r["token"])

md,X,sha,prov=load_atlas()
if sha!=EXPECTED_SHA:raise RuntimeError(("DINO overlay SHA changed",sha,EXPECTED_SHA))
word=np.array([str(x).lower() for x in md["word"]],object)
folio_col="folio" if "folio" in md.columns else ("page" if "page" in md.columns else None)
if folio_col is None:raise RuntimeError(("no folio column",list(md.columns)))
nums=np.array([canon_num(x) for x in md[folio_col]],object)
keep=np.array([n in strict_allowed and w in strict_allowed[n] for n,w in zip(nums,word)],bool)
md=md.loc[keep].reset_index(drop=True);X=X[keep];word=word[keep];nums=nums[keep]

vf=np.full(len(md),-1,int)
for i,n in enumerate(nums):
    if n in BIF and BIF[n] in folds:vf[i]=int(folds[BIF[n]])
keep=vf>=0;md=md.loc[keep].reset_index(drop=True);X=X[keep];word=word[keep];nums=nums[keep];vf=vf[keep]

def type_means(mask):
    by=collections.defaultdict(list)
    for i in np.where(mask)[0]:by[word[i]].append(i)
    ts=sorted(by);M=np.stack([X[by[t]].mean(0) for t in ts]);M/=np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-12)
    wt=np.array([len(by[t]) for t in ts],float)
    return ts,M,wt

trt,trX,trw=type_means(np.isin(vf,[2,3]))
allt,allX,allw=type_means(np.ones(len(vf),bool))
all_lookup={t:i for i,t in enumerate(allt)}
# reduce DINO dimension using discovery only.
pc=PCA(n_components=min(48,trX.shape[1],max(2,len(trt)-1)),random_state=SEED).fit(trX)
Ztr=pc.transform(trX);Zall=pc.transform(allX)
sc=StandardScaler().fit(Ztr);Ztr=sc.transform(Ztr);Zall=sc.transform(Zall)

def choose_k(Z):
    scores=[]
    for k in KGRID:
        if len(Z)<=k*3:continue
        km=KMeans(n_clusters=k,random_state=SEED,n_init=20,max_iter=500).fit(Z,sample_weight=trw)
        s=float(silhouette_score(Z,km.labels_,sample_size=min(2000,len(Z)),random_state=SEED))
        sizes=np.bincount(km.labels_,minlength=k);tiny=float(np.mean(sizes<5));scores.append((s-.05*tiny,k,s,tiny))
    scores.sort(reverse=True)
    return scores[0][1],[{"K":q[1],"score":q[0],"silhouette":q[2],"tiny_share":q[3]} for q in scores]

K,grid=choose_k(Ztr)
km=KMeans(n_clusters=K,random_state=SEED,n_init=30,max_iter=500).fit(Ztr,sample_weight=trw)
lab=km.predict(Zall)
rawmap={t:int(lab[i]) for i,t in enumerate(allt)}

# Visual residual arm: remove discovery-linear FORM predictability in DINO PCA space, then cluster residuals.
Ftr=np.stack([internal_feature(t) for t in trt]);Fall=np.stack([internal_feature(t) for t in allt])
fs=StandardScaler().fit(Ftr);ridge=Ridge(alpha=10.).fit(fs.transform(Ftr),Ztr,sample_weight=trw)
Rtr=Ztr-ridge.predict(fs.transform(Ftr));Rall=Zall-ridge.predict(fs.transform(Fall))
K2,grid2=choose_k(Rtr)
km2=KMeans(n_clusters=K2,random_state=SEED+1,n_init=30,max_iter=500).fit(Rtr,sample_weight=trw)
lab2=km2.predict(Rall);resmap={t:int(lab2[i]) for i,t in enumerate(allt)}

FAST=os.getenv("NK3_FAST","0")=="1"
NPERM=5 if FAST else 300
print("NK3R_VIS_POP",len(rows),collections.Counter(r["fold"] for r in rows),"visual_overlap",len(md),"types",len(allt),flush=True)
raw_ev=evaluate_family(rows,rawmap,nshuffle=NPERM,nlabel=NPERM,seed=SEED+6100)
res_ev=evaluate_family(rows,resmap,nshuffle=NPERM,nlabel=NPERM,seed=SEED+6200)

out={
 "phase":"NK3_REBUILD_VISUAL","status":"complete","population":"TTLI strict +P0",
 "provenance":{"canonical_private_hf":"Digitalgoldfish79/vdino3-crops","atlas_projection":f"{ORG}/{SLUG}",
               "normalized_embedding_sha256":sha,"expected_sha256":EXPECTED_SHA,"sha_verified":sha==EXPECTED_SHA,**prov},
 "coverage":{"atlas_rows_total":37886,"strict_overlap_visual_rows":int(len(md)),"discovery_visual_types":len(trt),"all_visual_types":len(allt)},
 "arms":{
   "DINO_RAW":{"role":"candidate_primary","K":K,"internal_K_grid":grid,"evaluation":raw_ev},
   "DINO_FORM_RESIDUAL":{"role":"candidate_secondary","K":K2,"internal_K_grid":grid2,
                         "residualization":"ridge predicts discovery DINO-PCA coordinates from frozen current-token FORM only; cluster residual","evaluation":res_ev}
 },
 "representation_note":"Visual arm is TTLI-framed because Atlas word labels descend from Takahashi; transcription replication is therefore not expected for this representation."
}
print("NK3_REBUILD_VISUAL_JSON="+json.dumps(out,separators=(",",":")),flush=True)
