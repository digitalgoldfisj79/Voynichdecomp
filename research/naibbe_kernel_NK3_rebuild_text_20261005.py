#!/usr/bin/env python3
"""NK3 rebuild — text/internal family arms."""
import collections, hashlib, json, urllib.request, os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CORE_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/b9795285e87c9d7332c29abe5123237100bce143/research/naibbe_kernel_NK3_rebuild_core_20261005.py"
c={"__name__":"nk3r_core"};exec(compile(urllib.request.urlopen(CORE_URL,timeout=120).read().decode(),CORE_URL,"exec"),c)
build_rows=c["build_rows"];internal_feature=c["internal_feature"];evaluate_family=c["evaluate_family"]
safe_form=c["safe_form"];rlenbin=c["rlenbin"];freqbin=c["freqbin"];SEED=c["SEED"]
KGRID=(4,8,12,16,24,32)

def lev(a,b,cap=3):
    if abs(len(a)-len(b))>=cap:return cap
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return min(prev[-1],cap)

def all_types(rows):return sorted(set(r["token"] for r in rows))
def disc_counts(rows):
    return collections.Counter(r["token"] for r in rows if r["fold"] in (2,3))

def fit_form_kmeans(rows,k=12,pca=False,append_ed=None):
    cnt=disc_counts(rows);tr=sorted(cnt)
    X=np.stack([internal_feature(t) for t in tr])
    if append_ed is not None:
        m=max(append_ed.values())+1
        E=np.zeros((len(tr),m),float)
        for i,t in enumerate(tr):E[i,int(append_ed[t])]=1.
        X=np.hstack([X,E])
    sc=StandardScaler().fit(X);Z=sc.transform(X)
    pc=None
    if pca:
        nc=min(16,Z.shape[1],max(2,len(tr)-1))
        pc=PCA(n_components=nc,random_state=SEED).fit(Z);Z=pc.transform(Z)
    if k=="auto":
        scores=[]
        for kk in KGRID:
            if len(tr)<=kk*3:continue
            km=KMeans(n_clusters=kk,random_state=SEED,n_init=20,max_iter=500).fit(Z,sample_weight=np.array([cnt[t] for t in tr],float))
            try:s=float(silhouette_score(Z,km.labels_,sample_size=min(2000,len(tr)),random_state=SEED))
            except Exception:s=-9.
            sizes=np.bincount(km.labels_,minlength=kk)
            tiny=float(np.mean(sizes<5))
            scores.append((s-.05*tiny,kk,s,tiny))
        if not scores:raise RuntimeError("no K")
        scores.sort(reverse=True);k=scores[0][1];sel={"grid":[{"K":q[1],"score":q[0],"silhouette":q[2],"tiny_share":q[3]} for q in scores],"selected":k}
    else:sel={"selected":int(k)}
    km=KMeans(n_clusters=int(k),random_state=SEED,n_init=30,max_iter=500).fit(Z,sample_weight=np.array([cnt[t] for t in tr],float))
    fmap={}
    for t in all_types(rows):
        x=internal_feature(t)[None,:]
        if append_ed is not None:
            e=np.zeros((1,max(append_ed.values())+1),float);e[0,int(append_ed.get(t,max(append_ed.values())))]=1.;x=np.hstack([x,e])
        z=sc.transform(x)
        if pc is not None:z=pc.transform(z)
        fmap[t]=int(km.predict(z)[0])
    return fmap,sel

def ed_seed_map(rows,max_seeds=24):
    cnt=disc_counts(rows);seeds=[]
    for t,_ in sorted(cnt.items(),key=lambda z:(-z[1],z[0])):
        if all(lev(t,s,3)>2 for s in seeds):
            seeds.append(t)
            if len(seeds)>=max_seeds:break
    other=len(seeds);fmap={}
    for t in all_types(rows):
        ds=[lev(t,s,3) for s in seeds]
        if ds and min(ds)<=2:fmap[t]=int(np.argmin(ds))
        else:fmap[t]=other
    return fmap,{"seeds":seeds,"other":other,"k":other+1}

def exact_control(rows,n=24):
    cnt=disc_counts(rows);top=[t for t,_ in cnt.most_common(n)];mp={t:i for i,t in enumerate(top)};other=len(top)
    return {t:mp.get(t,other) for t in all_types(rows)},{"top":top,"k":other+1}

def section_control(rows):
    # hostile control only, but keep the same discovery-only firewall as candidate arms.
    by=collections.defaultdict(collections.Counter)
    for r in rows:
        if r["fold"] in (2,3): by[r["token"]][r["section"]]+=1
    sec=sorted(set(r["section"] for r in rows if r["fold"] in (2,3)));sm={s:i for i,s in enumerate(sec)}
    other=len(sec);fmap={}
    for t in all_types(rows):
        fmap[t]=sm[by[t].most_common(1)[0][0]] if by.get(t) else other
    return fmap,{"sections":sec,"other":other}

def random_shape_hash(rows,k=12):
    cnt=collections.Counter(r["token"] for r in rows);f={}
    for t in all_types(rows):
        sf=safe_form(t);ps,cs=sf
        key=f"{cs[0]}|{cs[-1]}|{len(ps)}|{len(t)}|{freqbin(cnt[t])}|{t}"
        h=hashlib.sha256(("NK3RAND|"+key).encode()).digest();f[t]=int.from_bytes(h[:8],"big")%k
    return f,{"k":k}

FAST=os.getenv("NK3_FAST","0")=="1"
NSHUFF=5 if FAST else 250
NLABEL=5 if FAST else 250
OUT={}
for tid in ("ZLZI","ZLZB","TTLI"):
    rows=build_rows(tid)
    print("NK3R_POP",tid,len(rows),collections.Counter(r["fold"] for r in rows),flush=True)
    ed,edi=ed_seed_map(rows)
    legacy,legi=fit_form_kmeans(rows,12,False)
    learned,learni=fit_form_kmeans(rows,"auto",True)
    hybrid,hybi=fit_form_kmeans(rows,"auto",True,append_ed=ed)
    exact,exi=exact_control(rows)
    sec,seci=section_control(rows)
    rnd,rndi=random_shape_hash(rows,12)
    arms={
      "FORM_LEGACY_K12":(legacy,legi,"candidate_legacy"),
      "FORM_LEARNED":(learned,learni,"candidate_primary"),
      "FORM_ED_HYBRID":(hybrid,hybi,"candidate_primary"),
      "PURE_ED2":(ed,edi,"hostile_control"),
      "EXACT_TOKEN_TOP24":(exact,exi,"positive_control"),
      "SECTION_ONLY":(sec,seci,"hostile_control"),
      "RANDOM_SHAPE_HASH12":(rnd,rndi,"hostile_control")
    }
    rr={}
    for j,(name,(fm,info,role)) in enumerate(arms.items()):
        ev=evaluate_family(rows,fm,nshuffle=NSHUFF,nlabel=NLABEL,seed=SEED+1000*j+sum(map(ord,tid)))
        rr[name]={"role":role,"construction":info,"evaluation":ev}
        print("NK3R_TEXT_ARM",tid,name,json.dumps(ev,separators=(",",":")),flush=True)
    OUT[tid]={"n_rows":len(rows),"arms":rr}

def across(name):
    z={}
    for t in OUT:
        e=OUT[t]["arms"][name]["evaluation"]
        z[t]=None if e.get("status")!="ok" else {"gate":e["gate"],"gain":e["observed_context_gain_bits"],
          "context_z":e["context_shuffle"]["z"],"label_z":e["type_label_perm"]["z"],"folds":e["fold_gain_bits"]}
    vals=[z[t] for t in z if z[t] is not None]
    gate=bool(len(vals)==3 and z["ZLZI"]["gate"] and all(v["gain"]>0 for v in vals))
    return {"by_transcription":z,"cross_transcription_gate":gate}

summary={n:across(n) for n in ("FORM_LEGACY_K12","FORM_LEARNED","FORM_ED_HYBRID","PURE_ED2")}
out={"phase":"NK3_REBUILD_TEXT","status":"complete","population":"strict +P0",
     "discovery_folds":[2,3],"validation_fold":4,"final_folds":[0,1],
     "primary_metric":"heldout family codelength gain from external context above current-shape/section/frequency nuisance",
     "nulls":["external-context shuffle preserving fold/section/line-position","type-consistent family-label permutation within coarse current shape"],
     "results":OUT,"summary":summary}
print("NK3_REBUILD_TEXT_JSON="+json.dumps(out,separators=(",",":")),flush=True)
